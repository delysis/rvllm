//! Rootless benchmark observations. CPU cycles are never used to scale device time.
#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use libproc::libproc::pid_rusage::{pidrusage, RUsageInfoV4};
use objc2_foundation::NSProcessInfo;
use serde_json::{json, Value};

const INTERVAL: Duration = Duration::from_secs(1);
const MAX_AGE_MS: f64 = 2500.0;
const HISTORY: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Controls {
    source: &'static str,
    low_power: bool,
    power_mode: Option<u32>,
    thermal: isize,
    cpu_speed_limit: Option<u32>,
    scheduler_limit: Option<u32>,
    available_cpus: Option<u32>,
}

impl Controls {
    fn json(&self) -> Value {
        json!({"power_source":self.source,"low_power_mode":self.low_power,"pmset_power_mode":self.power_mode,
            "thermal_state":self.thermal,"cpu_speed_limit_percent":self.cpu_speed_limit,
            "scheduler_limit_percent":self.scheduler_limit,"available_cpus":self.available_cpus})
    }

    fn eligible(&self) -> bool {
        self.source != "unknown"
            && self.power_mode.is_some()
            && self.thermal == 0
            && self.cpu_speed_limit.map_or(true, |limit| limit == 100)
            && self.scheduler_limit.map_or(true, |limit| limit == 100)
    }
}

#[derive(Clone)]
struct Sample {
    start_ms: f64,
    end_ms: f64,
    controls: Controls,
    battery: String,
    limits: String,
    settings: String,
}

impl Sample {
    fn json(&self) -> Value {
        json!({"start_ms":self.start_ms,"end_ms":self.end_ms,
            "controls":self.controls.json(),"pmset_battery":self.battery,"pmset_thermal":self.limits,"pmset_settings":self.settings})
    }
}

fn pmset(query: &str) -> String {
    let mut command = Command::new("/usr/bin/pmset");
    command.arg("-g");
    if !query.is_empty() {
        command.arg(query);
    }
    match command.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => format!("unavailable: {}", String::from_utf8_lossy(&output.stderr)),
        Err(error) => format!("unavailable: {error}"),
    }
}

fn power_source(text: &str) -> &'static str {
    // Only the current source line counts; a battery's charging status does not.
    match text.lines().next() {
        Some(line) if line == "Now drawing from 'AC Power'" => "ac",
        Some(line) if line == "Now drawing from 'Battery Power'" => "battery",
        _ => "unknown",
    }
}

fn limit(text: &str, name: &str) -> Option<u32> {
    text.lines().find_map(|line| {
        let (key, value) = line.trim().split_once('=')?;
        (key.trim() == name)
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

fn host_state() -> (bool, isize) {
    objc2::rc::autoreleasepool(|_| {
        let info = NSProcessInfo::processInfo();
        (info.isLowPowerModeEnabled(), info.thermalState().0)
    })
}

fn sample(origin: Instant) -> Sample {
    let start_ms = origin.elapsed().as_secs_f64() * 1000.0;
    let battery = pmset("batt");
    let limits = pmset("therm");
    let settings = pmset("");
    let power_mode = settings.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == "powermode")
            .then(|| fields.next()?.parse().ok())
            .flatten()
    });
    let (low_power, thermal) = host_state();
    Sample {
        start_ms,
        end_ms: origin.elapsed().as_secs_f64() * 1000.0,
        controls: Controls {
            source: power_source(&battery),
            low_power,
            power_mode,
            thermal,
            cpu_speed_limit: limit(&limits, "CPU_Speed_Limit"),
            scheduler_limit: limit(&limits, "Scheduler_Limit"),
            available_cpus: limit(&limits, "CPU_Available_CPUs"),
        },
        battery,
        limits,
        settings,
    }
}

#[derive(Default)]
struct Observations {
    samples: VecDeque<Sample>,
    journal_error: Option<String>,
}

/// A one-second observer, independent of token execution. Optional JSONL keeps
/// observations even if model preparation fails. History in memory is bounded.
pub struct PowerMonitor {
    origin: Instant,
    observations: Arc<Mutex<Observations>>,
    stop: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl PowerMonitor {
    pub fn start(journal: Option<&Path>) -> std::io::Result<Self> {
        let origin = Instant::now();
        let mut journal = journal.map(std::fs::File::create_new).transpose()?;
        let initial = sample(origin);
        if let Some(file) = &mut journal {
            writeln!(file, "{}", initial.json())?;
        }
        let observations = Arc::new(Mutex::new(Observations {
            samples: VecDeque::from([initial]),
            journal_error: None,
        }));
        let shared = observations.clone();
        let (stop, receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("rvllm-power-observer".into())
            .spawn(move || {
                while matches!(
                    receiver.recv_timeout(INTERVAL),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let observation = sample(origin);
                    let error = journal.as_mut().and_then(|file| {
                        writeln!(file, "{}", observation.json())
                            .err()
                            .map(|error| error.to_string())
                    });
                    let mut state = shared.lock().unwrap_or_else(|poison| poison.into_inner());
                    if state.journal_error.is_none() {
                        state.journal_error = error;
                    }
                    if state.samples.len() == HISTORY {
                        state.samples.pop_front();
                    }
                    state.samples.push_back(observation);
                }
            })?;
        Ok(Self {
            origin,
            observations,
            stop,
            thread: Some(thread),
        })
    }

    /// Measures process-wide host work, including the observer thread but not
    /// child processes or ANE daemon work. No shell command runs on the token path.
    pub fn begin(&self) -> PhaseMeasurement<'_> {
        let host = host_state();
        let cpu = cpu_snapshot();
        PhaseMeasurement {
            monitor: self,
            start: Instant::now(),
            cpu,
            host,
        }
    }

    /// Latest asynchronous observation for an experiment launch gate. The age
    /// and journal error must be checked; this is not a device clock reading.
    pub fn latest_observation(&self) -> Value {
        let state = self.observations.lock().unwrap_or_else(|p| p.into_inner());
        match state.samples.back() {
            Some(sample) => json!({
                "sample":sample.json(),
                "age_ms":self.origin.elapsed().as_secs_f64()*1000.0-sample.end_ms,
                "observer_journal_error":state.journal_error
            }),
            None => Value::Null,
        }
    }
}

impl Drop for PowerMonitor {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn cpu_snapshot() -> Result<RUsageInfoV4, String> {
    let pid = i32::try_from(std::process::id()).map_err(|error| error.to_string())?;
    pidrusage(pid)
}

fn counter_delta(before: u64, after: u64) -> Option<u64> {
    // All-zero counters on unsupported hosts are unavailable, not zero work.
    (before != 0 && after != 0)
        .then(|| after.checked_sub(before))
        .flatten()
}

fn eligible_interval(
    samples: &[Sample],
    start_ms: f64,
    end_ms: f64,
    start_host: (bool, isize),
    end_host: (bool, isize),
) -> bool {
    let Some(first) = samples.first() else {
        return false;
    };
    let last = samples.last().expect("nonempty samples");
    let controls = &first.controls;
    first.end_ms <= start_ms
        && start_ms - first.end_ms <= MAX_AGE_MS
        && end_ms - last.end_ms <= MAX_AGE_MS
        && samples
            .windows(2)
            .all(|pair| pair[1].end_ms - pair[0].end_ms <= MAX_AGE_MS)
        && controls.eligible()
        && samples.iter().all(|sample| sample.controls == *controls)
        && start_host == (controls.low_power, controls.thermal)
        && end_host == start_host
}

pub struct PhaseMeasurement<'a> {
    monitor: &'a PowerMonitor,
    start: Instant,
    cpu: Result<RUsageInfoV4, String>,
    host: (bool, isize),
}

impl PhaseMeasurement<'_> {
    /// `units` means prompt tokens for prefill, actual ANE steps for decode, or
    /// one preparation. Never divide decode work by the Metal first token.
    pub fn finish(self, units: usize) -> Value {
        let end = Instant::now();
        let cpu = cpu_snapshot();
        let end_host = host_state();
        let start_ms = self.start.duration_since(self.monitor.origin).as_secs_f64() * 1000.0;
        let end_ms = end.duration_since(self.monitor.origin).as_secs_f64() * 1000.0;
        let (cycles, instructions, cpu_error) = match (&self.cpu, &cpu) {
            (Ok(before), Ok(after)) => (
                counter_delta(before.ri_cycles, after.ri_cycles),
                counter_delta(before.ri_instructions, after.ri_instructions),
                None,
            ),
            (Err(error), _) | (_, Err(error)) => (None, None, Some(error.as_str())),
        };
        let state = self
            .monitor
            .observations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let first = state
            .samples
            .iter()
            .rposition(|sample| sample.end_ms <= start_ms);
        let samples: Vec<_> = first.map_or_else(Vec::new, |index| {
            state
                .samples
                .iter()
                .skip(index)
                .take_while(|sample| sample.start_ms <= end_ms)
                .cloned()
                .collect()
        });
        let controls = samples.first().map(|sample| &sample.controls);
        let eligible = eligible_interval(&samples, start_ms, end_ms, self.host, end_host)
            && state.journal_error.is_none();
        let per_unit = |count: Option<u64>| {
            count
                .filter(|_| units != 0)
                .map(|count| count as f64 / units as f64)
        };
        json!({"schema":"rvllm.apple_phase_measurement.v1", "start_ms":start_ms,"end_ms":end_ms,
            "wall_ms":end.duration_since(self.start).as_secs_f64()*1000.0,"units":units,
            "process_cpu_cycles":cycles,"process_cpu_instructions":instructions,
            "cpu_cycles_per_unit":per_unit(cycles),"cpu_instructions_per_unit":per_unit(instructions),
            "cpu_cycles_per_instruction":cycles.zip(instructions).filter(|(_, instructions)| *instructions != 0).map(|(cycles,instructions)|cycles as f64/instructions as f64),
            "cpu_counter_error":cpu_error,
            "cpu_counter_scope":"proc_pid_rusage V4; all process threads; excludes child processes, GPU, ANE and daemon CPU; includes observer overhead",
            "start_host":{"low_power_mode":self.host.0,"thermal_state":self.host.1},
            "end_host":{"low_power_mode":end_host.0,"thermal_state":end_host.1},
            "power_samples":samples.iter().map(Sample::json).collect::<Vec<_>>(),
            "sampled_controls_eligible":eligible,
            "comparison_stratum":if eligible {controls.map(Controls::json)}else{None},
            "observer_journal_error":state.journal_error,
            "limitations":"Power is sampled about once per second; transient changes can be missed. Nominal thermal state does not prove constant clocks. Missing pmset limits are unknown. CPU cycles do not normalize accelerator performance. Compare repeated identical work only within matching sampled strata.",
            "cpu_frequency_hz":null,"gpu_cycles":null,"ane_cycles":null})
    }
}

/// Enumerate advertised counters. This does not enable profiling or claim that
/// any advertised counter has actually been sampled by inference.
pub fn metal_counter_capabilities() -> Value {
    use objc2_metal::{MTLCounter, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice};
    objc2::rc::autoreleasepool(|_| match MTLCreateSystemDefaultDevice() {
        None => json!({"available":false}),
        Some(device) => {
            let sets: Vec<_> = device.counterSets().map_or_else(Vec::new, |sets| sets.iter().map(|set|
                    json!({"name":set.name().to_string(),"counters":set.counters().iter().map(|counter|counter.name().to_string()).collect::<Vec<_>>()})
                ).collect());
            let info = NSProcessInfo::processInfo();
            json!({"available":true,"device":device.name().to_string(),"counter_sets":sets,
                    "os":info.operatingSystemVersionString().to_string(),"architecture":std::env::consts::ARCH,
                    "physical_memory_bytes":info.physicalMemory(),"logical_processors":info.processorCount(),
                    "inference_counter_sampling_enabled":false,"ane_cycle_counter":"unavailable"})
        }
    })
}

/// A paired observation, not a statistical speedup claim. Refuses legacy or
/// unlike power records. Callers must additionally establish identical work.
pub fn compare_phase_measurements(baseline: &Value, candidate: &Value) -> Result<Value, String> {
    for measurement in [baseline, candidate] {
        if measurement["schema"] != "rvllm.apple_phase_measurement.v1"
            || measurement["sampled_controls_eligible"] != true
            || measurement["comparison_stratum"].is_null()
        {
            return Err("missing, stale, changing or thermally limited power observations".into());
        }
    }
    if baseline["comparison_stratum"] != candidate["comparison_stratum"] {
        return Err(
            "power/processor strata differ; do not normalize device time with CPU cycles".into(),
        );
    }
    if baseline["units"].as_u64().unwrap_or(0) == 0 || baseline["units"] != candidate["units"] {
        return Err("work counts differ or are zero".into());
    }
    let ratio = |field: &str| -> Option<f64> {
        let before = baseline[field].as_f64()?;
        let after = candidate[field].as_f64()?;
        (before.is_finite() && after.is_finite() && before > 0.0 && after > 0.0)
            .then_some(before / after)
    };
    let wall_ratio = ratio("wall_ms").ok_or("invalid wall times")?;
    Ok(json!({"baseline_over_candidate_wall":wall_ratio,
        "baseline_over_candidate_cpu_cycles":ratio("process_cpu_cycles"),
        "baseline_over_candidate_cpu_instructions":ratio("process_cpu_instructions"),
        "comparison_stratum":baseline["comparison_stratum"],
        "claim":"One paired observation. Repeat alternating candidates on identical work; clocks and contention are not controlled by this check."}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_source_uses_current_supply_not_battery_presence() {
        assert_eq!(
            power_source("Now drawing from 'AC Power'\n-InternalBattery 57%; charging"),
            "ac"
        );
        assert_eq!(
            power_source("Now drawing from 'Battery Power'\n-InternalBattery 57%; discharging"),
            "battery"
        );
        assert_eq!(power_source("pmset failed"), "unknown");
        assert_eq!(
            limit(
                "CPU_Speed_Limit = 73\nScheduler_Limit = 80",
                "CPU_Speed_Limit"
            ),
            Some(73)
        );
        assert_eq!(
            limit("No CPU power status has been recorded", "CPU_Speed_Limit"),
            None
        );
    }

    #[test]
    fn unsupported_or_reset_counters_are_not_zero_work() {
        assert_eq!(counter_delta(0, 0), None);
        assert_eq!(counter_delta(100, 90), None);
        assert_eq!(counter_delta(100, 100), Some(0));
        assert_eq!(counter_delta(100, 250), Some(150));
    }

    #[test]
    fn transitions_thermal_pressure_and_sampling_gaps_reject_comparisons() {
        let baseline = Sample {
            start_ms: 0.0,
            end_ms: 10.0,
            controls: Controls {
                source: "ac",
                low_power: false,
                power_mode: Some(0),
                thermal: 0,
                cpu_speed_limit: None,
                scheduler_limit: None,
                available_cpus: None,
            },
            battery: String::new(),
            limits: String::new(),
            settings: String::new(),
        };
        let mut later = baseline.clone();
        later.start_ms = 1000.0;
        later.end_ms = 1010.0;
        let eligible = |later| {
            eligible_interval(
                &[baseline.clone(), later],
                20.0,
                1200.0,
                (false, 0),
                (false, 0),
            )
        };
        assert!(eligible(later.clone()));
        later.controls.source = "battery";
        assert!(!eligible(later.clone()));
        later.controls = baseline.controls.clone();
        later.controls.thermal = 1;
        assert!(!eligible(later.clone()));
        later.controls = baseline.controls.clone();
        later.controls.power_mode = Some(2);
        assert!(!eligible(later.clone()));
        later.controls = baseline.controls.clone();
        later.controls.cpu_speed_limit = Some(80);
        assert!(!eligible(later));
        assert!(!eligible_interval(
            &[baseline.clone()],
            20.0,
            3000.0,
            (false, 0),
            (false, 0)
        ));
        assert!(!eligible_interval(
            &[baseline],
            20.0,
            200.0,
            (false, 0),
            (true, 0)
        ));
        assert!(!eligible_interval(&[], 20.0, 200.0, (false, 0), (false, 0)));
    }

    #[test]
    fn paired_comparison_rejects_missing_and_different_power_states() {
        let baseline = json!({"schema":"rvllm.apple_phase_measurement.v1", "units":4,
            "sampled_controls_eligible":true,"comparison_stratum":{"power_source":"ac"}, "wall_ms":100.0});
        let mut candidate = baseline.clone();
        candidate["wall_ms"] = 80.0.into();
        assert_eq!(
            compare_phase_measurements(&baseline, &candidate).unwrap()
                ["baseline_over_candidate_wall"],
            1.25
        );
        candidate["comparison_stratum"]["power_source"] = "battery".into();
        assert!(compare_phase_measurements(&baseline, &candidate).is_err());
        assert!(compare_phase_measurements(&Value::Null, &baseline).is_err());
        candidate = baseline.clone();
        candidate["sampled_controls_eligible"] = false.into();
        assert!(compare_phase_measurements(&baseline, &candidate).is_err());
    }
}
