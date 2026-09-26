//! Disk-backed jobs, pinned executables, sampled launch gates and serial ownership.
#![forbid(unsafe_code)]

use rvllm_runtime::apple_measurement::PowerMonitor;
use rvllm_runtime::kernel_game::{parse_strict_json, SealedSubmission};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "rvllm.experiment_job.v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Invocation {
    executable: Pin,
    cwd: PathBuf,
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Conditions {
    power_source: String,
    low_power_mode: Option<bool>,
    pmset_power_mode: Option<u64>,
    thermal_state: Option<u64>,
    minimum_free_bytes: u64,
    disk_path: PathBuf,
    #[serde(default)]
    quiet_process_names: Vec<String>,
    #[serde(default)]
    observe_process_names: Vec<String>,
    #[serde(default)]
    idle_llama_servers: Vec<IdleLlamaServer>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IdleLlamaServer {
    pid: u32,
    port: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Job {
    schema: String,
    id: String,
    purpose: Purpose,
    command: Invocation,
    #[serde(default)]
    validator: Option<Invocation>,
    #[serde(default)]
    inputs: Vec<Pin>,
    #[serde(default)]
    kernel_game_submission: Option<Pin>,
    #[serde(default)]
    after: Vec<String>,
    conditions: Conditions,
    stable_seconds: u64,
    max_wait_seconds: u64,
    max_run_seconds: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Purpose {
    Timing,
    ExploratoryTiming,
    Correctness,
    Preparation,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 96
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_pin(pin: &Pin) -> Result<()> {
    if !pin.path.is_absolute()
        || !pin.path.is_file()
        || pin.sha256.len() != 64
        || !pin.sha256.bytes().all(|c| c.is_ascii_hexdigit())
        || digest(&pin.path)? != pin.sha256.to_ascii_lowercase()
    {
        return Err(format!("missing or changed pinned file: {}", pin.path.display()).into());
    }
    Ok(())
}

fn pin_present(pin: &Pin) -> Result<()> {
    if !pin.path.is_file() {
        return Err(format!("missing pinned file: {}", pin.path.display()).into());
    }
    Ok(())
}

impl Job {
    fn validate(&self) -> Result<()> {
        let c = &self.conditions;
        if self.schema != SCHEMA
            || !valid_id(&self.id)
            || !matches!(c.power_source.as_str(), "ac" | "battery")
            || c.pmset_power_mode.is_some_and(|mode| mode > 2)
            || c.thermal_state.is_some_and(|state| state > 3)
            || !c.disk_path.is_absolute()
            || !c.disk_path.is_dir()
            // Retained in v1 manifests for compatibility. Zero is the
            // preferred policy: sample readiness immediately and record
            // changing conditions instead of waiting for a stable stratum.
            || self.stable_seconds > 600
            || self.max_wait_seconds > 86400
            || !(1..=3600).contains(&self.max_run_seconds)
            || c.quiet_process_names
                .iter()
                .any(|s| s.is_empty() || s.contains('/'))
            || c.observe_process_names
                .iter()
                .any(|s| s.is_empty() || s.contains('/'))
            || c.idle_llama_servers.len() > 4
            || c.idle_llama_servers
                .iter()
                .any(|s| s.pid == 0 || s.port == 0)
        {
            return Err("invalid job identity, conditions or bounded durations".into());
        }
        let mut seen = BTreeSet::new();
        if self
            .after
            .iter()
            .any(|id| !valid_id(id) || id == &self.id || !seen.insert(id))
        {
            return Err("invalid, repeated or self dependency".into());
        }
        for call in std::iter::once(&self.command).chain(self.validator.iter()) {
            if !call.cwd.is_absolute()
                || !call.cwd.is_dir()
                || call
                    .env
                    .keys()
                    .any(|key| key.is_empty() || key.contains(['=', '\0']))
                || call
                    .env
                    .values()
                    .chain(call.args.iter())
                    .any(|s| s.contains('\0'))
            {
                return Err("invalid command directory, argument or environment".into());
            }
            if !call.executable.path.is_absolute()
                || call.executable.sha256.len() != 64
                || !call
                    .executable
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit())
            {
                return Err("invalid executable pin".into());
            }
        }
        for pin in self.inputs.iter().chain(self.kernel_game_submission.iter()) {
            if !pin.path.is_absolute()
                || pin.sha256.len() != 64
                || !pin.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            {
                return Err("invalid input pin".into());
            }
        }
        Ok(())
    }

    fn verify_kernel_game_submission(&self) -> Result<()> {
        let Some(pin) = &self.kernel_game_submission else {
            return Ok(());
        };
        verify_pin(pin)?;
        let bytes = fs::read(&pin.path)?;
        let submission: SealedSubmission = parse_strict_json(&bytes)
            .map_err(|error| format!("invalid kernel-game submission: {error}"))?;
        submission.validate()?;
        if submission.executable.sha256.as_str()
            != self.command.executable.sha256.to_ascii_lowercase()
        {
            return Err("kernel-game executable identity differs from queued executable".into());
        }
        let has_arg = |flag: &str, path: &Path| {
            self.command
                .args
                .windows(2)
                .any(|pair| pair[0] == flag && Path::new(&pair[1]) == path)
        };
        if !has_arg("--kernel-game-submission", &pin.path)
            || self
                .command
                .env
                .get("RVLLM_METAL_RESEARCH")
                .map(String::as_str)
                != Some(submission.candidate.as_str())
        {
            return Err("queued command is not bound to the sealed kernel-game candidate".into());
        }
        let mut required = vec![
            submission.source_tree.as_str(),
            submission.generated_source.as_str(),
            submission.model.as_str(),
            submission.reference.as_str(),
            submission.workload.as_str(),
            submission.oracle.as_str(),
        ];
        if let Some(metallib) = &submission.metallib {
            required.push(metallib.sha256.as_str());
        }
        for digest in required {
            if !self
                .inputs
                .iter()
                .any(|input| input.sha256.eq_ignore_ascii_case(digest))
            {
                return Err(format!(
                    "kernel-game required artifact {digest} is not a queued input pin"
                )
                .into());
            }
        }
        let pinned_path = |digest: &str| {
            self.inputs
                .iter()
                .find(|input| input.sha256.eq_ignore_ascii_case(digest))
                .map(|input| input.path.as_path())
        };
        let source_tree_path = pinned_path(submission.source_tree.as_str())
            .ok_or("required source-tree pin disappeared")?;
        let oracle_path =
            pinned_path(submission.oracle.as_str()).ok_or("required oracle pin disappeared")?;
        if !has_arg("--kernel-game-source-tree", source_tree_path)
            || !has_arg("--kernel-game-oracle", oracle_path)
        {
            return Err("queued command omits its pinned source-tree or oracle binding".into());
        }
        Ok(())
    }

    fn verify_files(&self) -> Result<()> {
        verify_pin(&self.command.executable)?;
        if let Some(call) = &self.validator {
            verify_pin(&call.executable)?;
        }
        for pin in &self.inputs {
            verify_pin(pin)?;
        }
        self.verify_kernel_game_submission()?;
        Ok(())
    }

    fn files_present(&self) -> Result<()> {
        for pin in std::iter::once(&self.command.executable)
            .chain(self.validator.iter().map(|call| &call.executable))
            .chain(self.inputs.iter())
            .chain(self.kernel_game_submission.iter())
        {
            pin_present(pin)?;
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = File::create_new(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn lock(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    if !fs4::fs_std::FileExt::try_lock_exclusive(&file)? {
        return Err(format!("another owner holds {}", path.display()).into());
    }
    Ok(file)
}

fn controls_match(observation: &Value, c: &Conditions) -> bool {
    let Some(age) = observation["age_ms"].as_f64() else {
        return false;
    };
    let controls = &observation["sample"]["controls"];
    age.is_finite()
        && (0.0..=2500.0).contains(&age)
        && observation.get("observer_journal_error") == Some(&Value::Null)
        && controls["power_source"] == c.power_source
        && controls["low_power_mode"]
            .as_bool()
            .is_some_and(|value| c.low_power_mode.map_or(true, |required| value == required))
        && controls["pmset_power_mode"].as_u64().is_some_and(|value| {
            value <= 2
                && c.pmset_power_mode
                    .map_or(true, |required| value == required)
        })
        && controls["thermal_state"]
            .as_u64()
            .is_some_and(|state| state <= 3)
        && c.thermal_state
            .map_or(true, |state| controls["thermal_state"] == state)
        && ["cpu_speed_limit_percent", "scheduler_limit_percent"]
            .iter()
            .all(|key| {
                controls.get(*key) == Some(&Value::Null) || controls[*key].as_u64() == Some(100)
            })
        && (controls.get("available_cpus") == Some(&Value::Null)
            || controls["available_cpus"].as_u64().is_some_and(|n| n > 0))
}

fn free_bytes(path: &Path) -> Result<u64> {
    Ok(fs4::available_space(path)?)
}

#[derive(Debug)]
struct Process {
    pid: u32,
    parent: u32,
    name: String,
}

fn parse_processes(text: &str) -> Result<Vec<Process>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let line = line.trim();
            let first = line
                .find(char::is_whitespace)
                .ok_or("missing process parent")?;
            let pid = line[..first].parse()?;
            let rest = line[first..].trim_start();
            let second = rest
                .find(char::is_whitespace)
                .ok_or("missing process command")?;
            let parent = rest[..second].parse()?;
            let path = Path::new(rest[second..].trim());
            let name = path
                .file_name()
                .ok_or("missing process name")?
                .to_string_lossy()
                .into_owned();
            Ok(Process { pid, parent, name })
        })
        .collect()
}

fn blockers(processes: &[Process], names: &[String], child: Option<u32>) -> Vec<Value> {
    let mut own = BTreeSet::from([std::process::id()]);
    // Exclude only our running trial and descendants, not other children of
    // the queue (an unrelated compile must remain visible).
    if let Some(pid) = child {
        own.insert(pid);
        loop {
            let count = own.len();
            for process in processes {
                if process.parent != std::process::id() && own.contains(&process.parent) {
                    own.insert(process.pid);
                }
            }
            if own.len() == count {
                break;
            }
        }
    }
    processes
        .iter()
        .filter(|p| !own.contains(&p.pid) && names.contains(&p.name))
        .map(|p| json!({"pid":p.pid,"name":p.name}))
        .collect()
}

fn slots_idle(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .is_some_and(|slots| {
            !slots.is_empty() && slots.iter().all(|slot| slot["is_processing"] == false)
        })
}

fn idle_server(server: &IdleLlamaServer, processes: &[Process]) -> Value {
    let identity = processes
        .iter()
        .any(|p| p.pid == server.pid && p.name == "llama-server");
    if !identity {
        return json!({"pid":server.pid,"port":server.port,"idle":false,"error":"server identity absent"});
    }
    let response = Command::new("/usr/bin/curl")
        .args([
            "--disable",
            "--noproxy",
            "*",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "1",
            "--max-filesize",
            "65536",
            &format!("http://127.0.0.1:{}/slots", server.port),
        ])
        .output();
    let idle = response
        .as_ref()
        .is_ok_and(|r| r.status.success() && r.stdout.len() <= 65536 && slots_idle(&r.stdout));
    json!({"pid":server.pid,"port":server.port,"idle":idle})
}

/// One queue pass shares raw observations, never a job's policy decision.
/// Drop this at the end of the pass. Active-child probes always start fresh.
#[derive(Default)]
struct ProbeCache {
    disks: BTreeMap<PathBuf, (Instant, u64)>,
    processes: Option<(Instant, Vec<Process>)>,
    idle: BTreeMap<(u32, u16), (Instant, Value)>,
}

impl ProbeCache {
    fn disk(&mut self, path: &Path) -> Result<(Instant, u64)> {
        if let Some(&value) = self.disks.get(path) {
            return Ok(value);
        }
        let started = Instant::now();
        let value = (started, free_bytes(path)?);
        self.disks.insert(path.to_owned(), value);
        Ok(value)
    }

    fn activity(
        &mut self,
        c: &Conditions,
        child: Option<u32>,
    ) -> Result<(Instant, Vec<Value>, Vec<Value>, Vec<Value>)> {
        if self.processes.is_none() {
            let started = Instant::now();
            let result = Command::new("/bin/ps")
                .args(["-axo", "pid=,ppid=,comm="])
                .output()?;
            if !result.status.success() {
                return Err("process activity unavailable".into());
            }
            self.processes = Some((
                started,
                parse_processes(std::str::from_utf8(&result.stdout)?)?,
            ));
        }
        let (started, processes) = self
            .processes
            .as_ref()
            .ok_or("missing activity observation")?;
        let mut oldest = *started;
        let mut checks = Vec::new();
        for server in &c.idle_llama_servers {
            let (observed, value) = self
                .idle
                .entry((server.pid, server.port))
                .or_insert_with(|| (Instant::now(), idle_server(server, processes)));
            oldest = oldest.min(*observed);
            checks.push(value.clone());
        }
        let mut competing = blockers(processes, &c.quiet_process_names, child);
        let observed = blockers(processes, &c.observe_process_names, child);
        competing.retain(|p| {
            !checks
                .iter()
                .any(|s| s["idle"] == true && s["pid"] == p["pid"])
        });
        Ok((oldest, competing, observed, checks))
    }
}

fn sample_age_ms(started: Instant, now: Instant) -> Option<f64> {
    now.checked_duration_since(started)
        .filter(|age| *age <= Duration::from_millis(2500))
        .map(|age| age.as_secs_f64() * 1000.0)
}

fn probe(monitor: &PowerMonitor, c: &Conditions, child: Option<u32>) -> Result<Value> {
    probe_shared(monitor, c, child, &mut ProbeCache::default())
}

fn probe_shared(
    monitor: &PowerMonitor,
    c: &Conditions,
    child: Option<u32>,
    cache: &mut ProbeCache,
) -> Result<Value> {
    let started = Instant::now();
    let (disk_observed, available) = cache.disk(&c.disk_path)?;
    let initial_power = monitor.latest_observation();
    // A blocked power/disk stratum cannot become ready by scanning processes.
    // Avoid paying for ps and idle-server probes for every such job: those
    // redundant subprocesses can themselves break the bounded sampling gap
    // of another, eligible job. A ready result still requires every check.
    if !controls_match(&initial_power, c) || available < c.minimum_free_bytes {
        return Ok(
            json!({"ready":false,"power":initial_power,"free_bytes":available,
            "competing_processes":[],"observed_processes":[],"idle_server_checks":[],
            "activity_sampled":false,
            "probe_ms":started.elapsed().as_secs_f64()*1000.0}),
        );
    }
    let (activity_observed, competing, observed, idle_checks) = cache.activity(c, child)?;
    let observation = monitor.latest_observation();
    let probe_ms = started.elapsed().as_secs_f64() * 1000.0;
    let raw_sample_age_ms = sample_age_ms(disk_observed.min(activity_observed), Instant::now());
    Ok(json!({"ready":controls_match(&observation,c)
        && available >= c.minimum_free_bytes && competing.is_empty()
        && idle_checks.iter().all(|s|s["idle"]==true) && probe_ms<=2500.0 && raw_sample_age_ms.is_some(),
        "power":observation,"free_bytes":available,"competing_processes":competing,
        "observed_processes":observed,
        "idle_server_checks":idle_checks,"activity_sampled":true,"probe_ms":probe_ms,
        "raw_sample_age_ms":raw_sample_age_ms}))
}

/// Never let an early error detach a live accelerator child or release its lock.
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.wait();
    }
}

/// Call only after pin verification has completed and its worker was joined.
/// Prepare output handles before the last gate check, leaving no hash or
/// report I/O between that check and spawn. Validators verify their own pin.
fn launch_verified(
    call: &Invocation,
    output: &Path,
    label: &str,
    before_spawn: impl FnOnce() -> Result<()>,
) -> Result<OwnedChild> {
    let directory = output.to_str().ok_or("output path is not UTF-8")?;
    let mut command = Command::new(&call.executable.path);
    command
        .current_dir(&call.cwd)
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("LANG", "C")
        .envs(
            call.env
                .iter()
                .map(|(key, value)| (key, value.replace("{output}", directory))),
        )
        .args(
            call.args
                .iter()
                .map(|arg| arg.replace("{output}", directory)),
        )
        .stdin(Stdio::null())
        .stdout(File::create_new(output.join(format!("{label}.stdout")))?)
        .stderr(File::create_new(output.join(format!("{label}.stderr")))?);
    before_spawn()?;
    Ok(OwnedChild(command.spawn()?))
}

fn stopped(queue: &Path, stop: &AtomicBool) -> bool {
    stop.load(Ordering::Relaxed) || queue.join("STOP").exists()
}

fn phase_eligible(phase: &Value, c: &Conditions) -> bool {
    if c.thermal_state == Some(0) {
        phase["sampled_controls_eligible"] == true
    } else if c.thermal_state.is_none() && phase["sampled_controls_eligible"] == true {
        true
    } else {
        super::fair::validate(phase).is_ok()
    }
}

fn purpose_accepts_ineligible(purpose: Purpose) -> bool {
    matches!(
        purpose,
        Purpose::Preparation | Purpose::Correctness | Purpose::ExploratoryTiming
    )
}

fn execute(
    job: &Job,
    queue: &Path,
    monitor: &PowerMonitor,
    stop: &AtomicBool,
    wait_started: Instant,
) -> Result<Option<bool>> {
    let mut observe = || -> std::result::Result<bool, String> {
        if wait_started.elapsed() > Duration::from_secs(job.max_wait_seconds) {
            return Err(format!(
                "job {} expired before launch; no trial started",
                job.id
            ));
        }
        if stopped(queue, stop) {
            return Ok(false);
        }
        let current = probe(monitor, &job.conditions, None).map_err(|e| e.to_string())?;
        // Probes perform I/O. STOP and the original deadline can change while
        // they run; neither may be checked only before that blocking work.
        if stopped(queue, stop) {
            return Ok(false);
        }
        if wait_started.elapsed() > Duration::from_secs(job.max_wait_seconds) {
            return Err(format!(
                "job {} expired before launch; no trial started",
                job.id
            ));
        }
        Ok(current["ready"] == true)
    };
    // Hashing can exceed the observation freshness budget. Keep sampling
    // readiness while the scoped verifier runs, then join it. No condition
    // dwell is required; every sampled transition remains in the receipts.
    if !super::prelaunch::verify(
        || job.verify_files().map_err(|e| e.to_string()),
        &mut observe,
    )? {
        return Ok(None);
    }
    let output = queue.join("results").join(&job.id);
    fs::create_dir(&output)?; // Never replay an existing/incomplete attempt.
    atomic_json(&output.join("job.json"), &serde_json::to_value(job)?)?;
    atomic_json(
        &output.join("report.json"),
        &json!({"status":"starting","id":job.id}),
    )?;
    let phase = monitor.begin();
    let started = Instant::now();
    let mut child = launch_verified(&job.command, &output, "trial", || {
        if observe()? {
            return Ok(());
        }
        // This attempt already owns durable output. Preserve it and halt,
        // rather than deleting it or silently retrying the same manifest.
        atomic_json(
            &output.join("report.json"),
            &json!({"schema":"rvllm.experiment_result.v1","id":job.id,
                "purpose":job.purpose,"status":"failed","trial_started":false,
                "sampled_conditions_eligible":false,"stop_requested":stopped(queue,stop),
                "error":"launch gate lost after attempt setup; no trial started"}),
        )?;
        Err("launch gate lost after attempt setup; no trial started".into())
    })?;
    let pid = child.0.id();
    atomic_json(
        &output.join("report.json"),
        &json!({"status":"running","id":job.id,"pid":pid}),
    )?;
    let mut violations = Vec::new();
    let mut observations = File::create_new(output.join("conditions.jsonl"))?;
    let mut overdue = false;
    let exit = loop {
        match probe(monitor, &job.conditions, Some(pid)) {
            Ok(value) => {
                writeln!(observations, "{value}")?;
                if value["ready"] != true && violations.len() < 32 {
                    violations.push(value);
                }
            }
            Err(error) => {
                let value = json!({"observation_error":error.to_string()});
                writeln!(observations, "{value}")?;
                if violations.len() < 32 {
                    violations.push(value);
                }
            }
        }
        if !overdue && started.elapsed() > Duration::from_secs(job.max_run_seconds) {
            overdue = true;
            atomic_json(
                &output.join("report.json"),
                &json!({
                "status":"overdue-awaiting-safe-child-exit","id":job.id,"pid":pid,
                "claim":"No forced termination or automatic retry."}),
            )?;
        }
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        std::thread::sleep(Duration::from_secs(1));
    };
    drop(child);
    let measurement = phase.finish(1);
    let files_unchanged = job.verify_files().map_err(|e| e.to_string());
    let eligible =
        violations.is_empty() && !overdue && phase_eligible(&measurement, &job.conditions);
    let mut validation = Value::Null;
    let mut accepted = exit.success()
        && !overdue
        && files_unchanged.is_ok()
        && (purpose_accepts_ineligible(job.purpose) || eligible);
    if accepted {
        if let Some(call) = &job.validator {
            verify_pin(&call.executable)?;
            let mut validator = launch_verified(call, &output, "validation", || Ok(()))?;
            let status = validator.0.wait()?;
            accepted = status.success();
            validation = json!({"exit_code":status.code(),"success":status.success()});
        }
    }
    let rejected = exit.success()
        && !overdue
        && files_unchanged.is_ok()
        && job.purpose == Purpose::Timing
        && !eligible;
    let status = if accepted {
        "succeeded"
    } else if rejected {
        "rejected"
    } else {
        "failed"
    };
    let report = json!({"schema":"rvllm.experiment_result.v1","id":job.id,"purpose":job.purpose,
        "status":status,
        "exit_code":exit.code(),"signal_or_missing_exit_code":exit.code().is_none(),
        "sampled_conditions_eligible":eligible,"violations":violations,"overdue":overdue,
        "files_unchanged":files_unchanged.is_ok(),"file_error":files_unchanged.err(),
        "validation":validation,"measurement":measurement,
        "kernel_game_submission_sha256":job.kernel_game_submission.as_ref().map(|pin|pin.sha256.to_ascii_lowercase()),
        "stop_requested":stopped(queue,stop),
        "claim":"Preparation and exploratory-timing success do not qualify performance promotion. Exploratory timing may succeed when sampled_conditions_eligible is false; retain and stratify all observations. Outer process duration includes startup and is not token throughput. CPU counters belong to the queue, excluding its child. Backend reports/validators establish numerical correctness and phase timing. Sampled conditions cannot prove fixed clocks or absence of all competing work."});
    atomic_json(&output.join("report.json"), &report)?;
    eprintln!("experiment {}: {}", job.id, report["status"]);
    Ok(Some(accepted || rejected))
}

fn manifest_paths(queue: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(queue.join("jobs"))? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.path().extension().is_some_and(|e| e == "json") {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

#[derive(Debug, PartialEq)]
enum DependencyState {
    Ready,
    Waiting,
    Failed(String),
}

fn dependency_state(job: &Job, queue: &Path) -> Result<DependencyState> {
    for dependency in &job.after {
        let report = queue.join("results").join(dependency).join("report.json");
        if !report.exists() {
            return Ok(DependencyState::Waiting);
        }
        let value = read_json(&report)?;
        if !matches!(value["status"].as_str(), Some("succeeded" | "rejected")) {
            return Ok(DependencyState::Failed(dependency.clone()));
        }
    }
    Ok(DependencyState::Ready)
}

fn quarantine_manifest(queue: &Path, path: &Path, job: &Job, reason: &str) -> Result<()> {
    let directory = queue.join("quarantined-jobs");
    fs::create_dir_all(&directory)?;
    let destination = directory.join(format!("{}.json", job.id));
    let receipt = directory.join(format!("{}.receipt.json", job.id));
    if destination.exists() || receipt.exists() {
        return Err(format!("quarantine identity already exists for {}", job.id).into());
    }
    let manifest_sha256 = digest(path)?;
    fs::rename(path, &destination)?;
    let report_path = queue.join("results").join(&job.id).join("report.json");
    let report = if report_path.is_file() {
        json!({"path":report_path,"sha256":digest(&report_path)?})
    } else {
        Value::Null
    };
    atomic_json(
        &receipt,
        &json!({
            "schema":"rvllm.experiment_quarantine.v1",
            "id":job.id,
            "reason":reason,
            "manifest":{"path":destination,"sha256":manifest_sha256},
            "report":report,
            "claim":"Quarantine preserves terminal evidence and prevents replay; independent jobs may continue."
        }),
    )?;
    Ok(())
}

fn quarantine_receipts(queue: &Path) -> Result<Vec<Value>> {
    let directory = queue.join("quarantined-jobs");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        if entry.file_type()?.is_file() && name.to_string_lossy().ends_with(".receipt.json") {
            paths.push(entry.path());
        }
    }
    paths.sort();
    paths.into_iter().map(|path| read_json(&path)).collect()
}

fn run(queue: &Path, accelerator_lock: &Path, idle_seconds: Option<u64>) -> Result<()> {
    let _queue_lock = lock(&queue.join("worker.lock"))?;
    let result = run_owned(queue, accelerator_lock, idle_seconds);
    if let Err(error) = &result {
        let _ = atomic_json(
            &queue.join("state.json"),
            &json!({"status":"halted","error":error.to_string(),"pid":std::process::id()}),
        );
    }
    result
}

fn run_owned(queue: &Path, accelerator_lock: &Path, idle_seconds: Option<u64>) -> Result<()> {
    let _accelerator_lock = lock(accelerator_lock)?;
    // A stopped queue needs neither a power observer nor a signal handler.
    // This also permits a real executable smoke without sampling hardware.
    if queue.join("STOP").exists() {
        atomic_json(
            &queue.join("state.json"),
            &json!({"status":"stopped","pid":std::process::id()}),
        )?;
        return Ok(());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
    // Sampling invokes macOS power tools. Keep it entirely off while an
    // always-on daemon has no work instead of burning CPU and growing an idle
    // journal forever. A monitor remains live across condition waits and job
    // execution so its history still spans the complete admission interval.
    let mut monitor = None;
    let mut idle = Instant::now();
    let mut idle_jobs_modified = None;
    let mut waiting = BTreeMap::<String, Instant>::new();
    loop {
        if stopped(queue, &stop) {
            atomic_json(
                &queue.join("state.json"),
                &json!({"status":"stopped","pid":std::process::id()}),
            )?;
            return Ok(());
        }
        let jobs_modified = fs::metadata(queue.join("jobs"))?.modified()?;
        if idle_jobs_modified == Some(jobs_modified) {
            if idle_seconds.is_some_and(|seconds| idle.elapsed() >= Duration::from_secs(seconds)) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        let mut selected = None;
        let mut pending = Vec::new();
        let mut probes = ProbeCache::default();
        for path in manifest_paths(queue)? {
            let job: Job = serde_json::from_reader(BufReader::new(File::open(&path)?))?;
            let result = queue.join("results").join(&job.id);
            if result.exists() {
                let report = read_json(&result.join("report.json"))?;
                if !matches!(report["status"].as_str(), Some("succeeded" | "rejected")) {
                    if idle_seconds.is_none() {
                        quarantine_manifest(
                            queue,
                            &path,
                            &job,
                            "terminal failed or incomplete result",
                        )?;
                        continue;
                    }
                    return Err(format!(
                        "job {} has failed or incomplete output; no replay",
                        job.id
                    )
                    .into());
                }
                continue;
            }
            job.validate()?;
            match dependency_state(&job, queue)? {
                DependencyState::Ready => {}
                DependencyState::Waiting => {
                    pending.push(json!({"id":job.id,"waiting_for_dependencies":job.after}));
                    continue;
                }
                DependencyState::Failed(dependency) if idle_seconds.is_none() => {
                    quarantine_manifest(
                        queue,
                        &path,
                        &job,
                        &format!("dependency {dependency} did not succeed"),
                    )?;
                    continue;
                }
                DependencyState::Failed(dependency) => {
                    return Err(
                        format!("dependency {dependency} did not succeed; queue stopped").into(),
                    );
                }
            }
            let started = waiting.entry(job.id.clone()).or_insert_with(Instant::now);
            if started.elapsed() > Duration::from_secs(job.max_wait_seconds) {
                return Err(format!(
                    "job {} expired waiting for conditions; no trial started",
                    job.id
                )
                .into());
            }
            if monitor.is_none() {
                let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                monitor = Some(PowerMonitor::start(Some(
                    &queue.join(format!("power-{stamp}-{}.jsonl", std::process::id())),
                ))?);
            }
            let observation = probe_shared(
                monitor.as_ref().expect("monitor initialized above"),
                &job.conditions,
                None,
                &mut probes,
            )?;
            pending.push(
                json!({"id":job.id,"wait_seconds":started.elapsed().as_secs(),
                "conditions":observation}),
            );
            if observation["ready"] == true {
                selected = Some(job);
                break;
            }
        }
        if let Some(job) = selected {
            idle = Instant::now();
            if stopped(queue, &stop) {
                continue;
            }
            atomic_json(
                &queue.join("state.json"),
                &json!({"status":"starting","id":job.id,"pid":std::process::id()}),
            )?;
            let wait_started = waiting
                .get(&job.id)
                .ok_or("selected job is missing its waiting deadline")?;
            match execute(
                &job,
                queue,
                monitor
                    .as_ref()
                    .ok_or("selected job is missing its power monitor")?,
                &stop,
                *wait_started,
            )? {
                Some(false) => {
                    if idle_seconds.is_none() {
                        quarantine_manifest(
                            queue,
                            &queue.join("jobs").join(format!("{}.json", job.id)),
                            &job,
                            "trial or validator failed",
                        )?;
                        waiting.remove(&job.id);
                        continue;
                    }
                    return Err(
                        format!("job {} failed; queue stopped without retry", job.id).into(),
                    );
                }
                Some(true) => {
                    waiting.remove(&job.id);
                }
                None => {}
            }
        } else {
            let status = if pending.is_empty() {
                "idle"
            } else {
                "waiting"
            };
            atomic_json(
                &queue.join("state.json"),
                &json!({"status":status,"pid":std::process::id(),"pending":pending}),
            )?;
            if pending.is_empty() {
                monitor = None;
                // Submission publishes a manifest by renaming it into jobs/;
                // that directory mtime is the cheap generation signal. Avoid
                // reparsing every historical manifest/result once per second
                // while still noticing newly published work promptly.
                idle_jobs_modified = Some(jobs_modified);
            } else {
                idle_jobs_modified = None;
            }
            if !pending.is_empty() {
                idle = Instant::now();
            }
            if idle_seconds.is_some_and(|seconds| idle.elapsed() >= Duration::from_secs(seconds)) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

pub(super) fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let usage = "usage: rvllm_experiment_queue submit QUEUE JOB.json | run QUEUE GLOBAL_LOCK [IDLE_SECONDS] | daemon QUEUE GLOBAL_LOCK | status QUEUE | stop QUEUE";
    if args.len() < 2 {
        return Err(usage.into());
    }
    let queue = PathBuf::from(&args[1]);
    if !queue.is_absolute() {
        return Err("queue path must be absolute".into());
    }
    match args[0].to_str() {
        Some("submit") if args.len() == 3 => {
            let bytes = fs::read(&args[2])?;
            let job: Job = serde_json::from_slice(&bytes)?;
            job.validate()?;
            // Staging must not scan model-sized inputs while another job is
            // timing. Full hashes remain mandatory before and after execution.
            job.files_present()?;
            fs::create_dir_all(queue.join("jobs"))?;
            fs::create_dir_all(queue.join("results"))?;
            // The producer lock prevents two submitters from replacing an id.
            let _producer = lock(&queue.join("submit.lock"))?;
            let destination = queue.join("jobs").join(format!("{}.json", job.id));
            if destination.exists()
                || queue.join("results").join(&job.id).exists()
                || queue
                    .join("quarantined-jobs")
                    .join(format!("{}.json", job.id))
                    .exists()
            {
                return Err("job id already exists".into());
            }
            for dependency in &job.after {
                if !queue
                    .join("jobs")
                    .join(format!("{dependency}.json"))
                    .is_file()
                {
                    return Err(format!("submit dependency {dependency} first").into());
                }
            }
            atomic_json(&destination, &serde_json::to_value(&job)?)?;
            println!("{}", json!({"submitted":job.id,"queue":queue}));
            Ok(())
        }
        Some("run") if args.len() == 3 || args.len() == 4 => {
            let lock_path = PathBuf::from(&args[2]);
            if !lock_path.is_absolute() {
                return Err("global lock must be absolute".into());
            }
            let idle = if args.len() == 4 {
                args[3].to_str().ok_or("invalid idle time")?.parse()?
            } else {
                300
            };
            if idle > 86400 {
                return Err("idle time must be <= 86400 seconds".into());
            }
            run(&queue, &lock_path, Some(idle))
        }
        Some("daemon") if args.len() == 3 => {
            let lock_path = PathBuf::from(&args[2]);
            if !lock_path.is_absolute() {
                return Err("global lock must be absolute".into());
            }
            run(&queue, &lock_path, None)
        }
        Some("stop") if args.len() == 2 => {
            File::create_new(queue.join("STOP"))?;
            println!("Stop recorded; an active child is allowed to finish.");
            Ok(())
        }
        Some("status") if args.len() == 2 => {
            let state = queue.join("state.json");
            let mut jobs = Vec::new();
            for path in manifest_paths(&queue)? {
                let job: Job = serde_json::from_reader(BufReader::new(File::open(path)?))?;
                let report = queue.join("results").join(&job.id).join("report.json");
                let result = if report.exists() {
                    read_json(&report)?
                } else {
                    Value::Null
                };
                jobs.push(json!({"id":job.id,"status":result["status"],
                    "sampled_conditions_eligible":result["sampled_conditions_eligible"],
                    "exit_code":result["exit_code"],"report":report}));
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"state":if state.exists(){read_json(&state)?}else{Value::Null},
                        "jobs":jobs,"quarantined":quarantine_receipts(&queue)?})
                )?
            );
            Ok(())
        }
        _ => Err(usage.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn conditions() -> Conditions {
        Conditions {
            power_source: "ac".into(),
            low_power_mode: Some(true),
            pmset_power_mode: Some(1),
            thermal_state: Some(0),
            minimum_free_bytes: 1,
            disk_path: PathBuf::from("/"),
            quiet_process_names: vec!["cargo".into()],
            observe_process_names: vec![],
            idle_llama_servers: vec![],
        }
    }
    fn observation() -> Value {
        json!({"age_ms":1.0,"observer_journal_error":null,"sample":{"controls":{
            "power_source":"ac","low_power_mode":true,"pmset_power_mode":1,"thermal_state":0,
            "available_cpus":null,"cpu_speed_limit_percent":null,"scheduler_limit_percent":null}}})
    }
    #[test]
    fn gate_rejects_unknown_restricted_stale_and_changed_power() {
        let original = observation();
        assert!(controls_match(&original, &conditions()));
        for (pointer, value) in [
            ("/age_ms", json!(2501)),
            ("/observer_journal_error", json!("io")),
            ("/sample/controls/power_source", json!("battery")),
            ("/sample/controls/thermal_state", json!(1)),
            ("/sample/controls/low_power_mode", json!(false)),
            ("/sample/controls/cpu_speed_limit_percent", json!(80)),
            ("/sample/controls/available_cpus", json!(0)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(!controls_match(&changed, &conditions()), "{pointer}");
        }
    }
    #[test]
    fn unpinned_jobs_accept_every_known_thermal_state_but_never_unknown() {
        let mut c = conditions();
        c.thermal_state = None;
        let mut value = observation();
        assert!(controls_match(&value, &c));
        for state in [json!(1), json!(2), json!(3)] {
            value["sample"]["controls"]["thermal_state"] = state;
            assert!(controls_match(&value, &c));
        }
        for state in [json!(4), Value::Null] {
            value["sample"]["controls"]["thermal_state"] = state;
            assert!(!controls_match(&value, &c));
        }
    }

    #[test]
    fn unpinned_power_modes_accept_known_values_but_never_unknown() {
        let mut c = conditions();
        c.low_power_mode = None;
        c.pmset_power_mode = None;
        let mut value = observation();
        for low_power_mode in [false, true] {
            value["sample"]["controls"]["low_power_mode"] = json!(low_power_mode);
            for mode in 0..=2 {
                value["sample"]["controls"]["pmset_power_mode"] = json!(mode);
                assert!(controls_match(&value, &c));
            }
        }
        for mode in [json!(3), Value::Null] {
            value["sample"]["controls"]["pmset_power_mode"] = mode;
            assert!(!controls_match(&value, &c));
        }
        value["sample"]["controls"]["pmset_power_mode"] = json!(1);
        value["sample"]["controls"]["low_power_mode"] = Value::Null;
        assert!(!controls_match(&value, &c));
    }

    #[test]
    fn exploratory_timing_retains_ineligible_data_without_weakening_timing() {
        assert!(purpose_accepts_ineligible(Purpose::ExploratoryTiming));
        assert!(purpose_accepts_ineligible(Purpose::Correctness));
        assert!(purpose_accepts_ineligible(Purpose::Preparation));
        assert!(!purpose_accepts_ineligible(Purpose::Timing));
    }
    #[test]
    fn only_owned_trial_descendants_are_exempt_from_activity_gate() {
        let processes = parse_processes(
            "11 1 /usr/bin/cargo\n22 1 /tmp/trial\n23 22 /tmp/cargo\n24 23 /tmp/cargo\n",
        )
        .unwrap();
        let values = blockers(&processes, &["cargo".into()], Some(22));
        assert_eq!(values, vec![json!({"pid":11,"name":"cargo"})]);
        assert!(parse_processes("bad process").is_err());
    }

    #[test]
    fn shared_raw_samples_keep_job_policies_and_child_ownership_separate() {
        let now = Instant::now();
        let mut cache = ProbeCache {
            processes: Some((
                now,
                parse_processes("11 1 /tmp/cargo\n22 1 /tmp/llama-server\n").unwrap(),
            )),
            idle: BTreeMap::from([((22, 8093), (now, json!({"pid":22,"port":8093,"idle":true})))]),
            ..ProbeCache::default()
        };
        let mut c = conditions();
        c.quiet_process_names = vec!["cargo".into(), "llama-server".into()];
        c.observe_process_names = vec!["cargo".into(), "llama-server".into()];
        c.idle_llama_servers = vec![IdleLlamaServer {
            pid: 22,
            port: 8093,
        }];
        assert_eq!(
            cache.activity(&c, None).unwrap().1,
            vec![json!({"pid":11,"name":"cargo"})]
        );
        assert_eq!(cache.activity(&c, None).unwrap().2.len(), 2);
        c.idle_llama_servers.clear();
        assert_eq!(
            cache.activity(&c, Some(11)).unwrap().1,
            vec![json!({"pid":22,"name":"llama-server"})]
        );
        assert_eq!(cache.activity(&c, None).unwrap().1.len(), 2);
        c.quiet_process_names.clear();
        assert!(cache.activity(&c, None).unwrap().1.is_empty());
        assert_eq!(cache.activity(&c, None).unwrap().2.len(), 2);
        assert_eq!(cache.processes.as_ref().unwrap().0, now);
    }

    #[test]
    fn shared_samples_expire_without_extending_observation_time() {
        let start = Instant::now();
        assert_eq!(sample_age_ms(start, start), Some(0.0));
        assert_eq!(
            sample_age_ms(start, start + Duration::from_millis(2500)),
            Some(2500.0)
        );
        assert_eq!(
            sample_age_ms(start, start + Duration::from_millis(2501)),
            None
        );
        assert_eq!(sample_age_ms(start + Duration::from_millis(1), start), None);
    }
    #[test]
    fn hashes_and_exclusive_locks_prevent_changed_or_duplicate_execution() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("binary");
        fs::write(&path, b"before").unwrap();
        let pin = Pin {
            path: path.clone(),
            sha256: digest(&path).unwrap(),
        };
        verify_pin(&pin).unwrap();
        fs::write(&path, b"after").unwrap();
        pin_present(&pin).unwrap(); // Cheap staging is not hash qualification.
        assert!(verify_pin(&pin).is_err());
        fs::remove_file(&path).unwrap();
        assert!(pin_present(&pin).is_err());
        let path = directory.path().join("lock");
        let held = lock(&path).unwrap();
        assert!(lock(&path).is_err());
        drop(held);
        assert!(lock(&path).is_ok());
    }
    #[test]
    fn malformed_ids_and_unrecognized_manifest_fields_are_rejected() {
        for id in ["", "../escape", "space id", "/abs", "a.json"] {
            assert!(!valid_id(id));
        }
        assert!(valid_id("02-baseline_A"));
        assert!(serde_json::from_value::<Conditions>(json!({
            "power_source":"ac","low_power_mode":true,"pmset_power_mode":1,"thermal_state":0,
            "minimum_free_bytes":1,"disk_path":"/","unknown":true}))
        .is_err());
    }
    #[test]
    fn idle_server_exception_requires_all_explicit_slots_idle() {
        assert!(slots_idle(br#"[{"id":0,"is_processing":false}]"#));
        for bytes in [
            b"[]".as_slice(),
            b"{}",
            b"invalid",
            br#"[{"is_processing":true}]"#,
            br#"[{"id":0}]"#,
            br#"[{"is_processing":false},{"is_processing":true}]"#,
        ] {
            assert!(!slots_idle(bytes));
        }
    }

    #[test]
    fn failed_dependency_blocks_downstream_work() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("results/first")).unwrap();
        atomic_json(
            &dir.path().join("results/first/report.json"),
            &json!({"status":"failed"}),
        )
        .unwrap();
        let pin = Pin {
            path: "/unused".into(),
            sha256: "0".repeat(64),
        };
        let job = Job {
            schema: SCHEMA.into(),
            id: "second".into(),
            purpose: Purpose::Timing,
            command: Invocation {
                executable: pin,
                cwd: "/".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            validator: None,
            inputs: vec![],
            kernel_game_submission: None,
            after: vec!["first".into()],
            conditions: conditions(),
            stable_seconds: 1,
            max_wait_seconds: 10,
            max_run_seconds: 10,
        };
        assert_eq!(
            dependency_state(&job, dir.path()).unwrap(),
            DependencyState::Failed("first".into())
        );
    }

    #[test]
    fn condition_rejected_dependency_allows_downstream_work() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("results/first")).unwrap();
        atomic_json(
            &dir.path().join("results/first/report.json"),
            &json!({"status":"rejected","sampled_conditions_eligible":false}),
        )
        .unwrap();
        let job = Job {
            schema: SCHEMA.into(),
            id: "second".into(),
            purpose: Purpose::Timing,
            command: Invocation {
                executable: Pin {
                    path: "/unused".into(),
                    sha256: "0".repeat(64),
                },
                cwd: "/".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            validator: None,
            inputs: vec![],
            kernel_game_submission: None,
            after: vec!["first".into()],
            conditions: conditions(),
            stable_seconds: 1,
            max_wait_seconds: 10,
            max_run_seconds: 10,
        };
        assert_eq!(
            dependency_state(&job, dir.path()).unwrap(),
            DependencyState::Ready
        );
    }

    #[test]
    fn quarantine_preserves_manifest_and_report_identities() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("jobs")).unwrap();
        fs::create_dir_all(dir.path().join("results/second")).unwrap();
        let report = dir.path().join("results/second/report.json");
        atomic_json(&report, &json!({"status":"failed"})).unwrap();
        let job = Job {
            schema: SCHEMA.into(),
            id: "second".into(),
            purpose: Purpose::Correctness,
            command: Invocation {
                executable: Pin {
                    path: "/unused".into(),
                    sha256: "0".repeat(64),
                },
                cwd: "/".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            validator: None,
            inputs: vec![],
            kernel_game_submission: None,
            after: vec![],
            conditions: conditions(),
            stable_seconds: 0,
            max_wait_seconds: 10,
            max_run_seconds: 10,
        };
        let manifest = dir.path().join("jobs/second.json");
        atomic_json(&manifest, &serde_json::to_value(&job).unwrap()).unwrap();
        let manifest_hash = digest(&manifest).unwrap();
        let report_hash = digest(&report).unwrap();

        quarantine_manifest(dir.path(), &manifest, &job, "test failure").unwrap();

        assert!(!manifest.exists());
        let quarantined = dir.path().join("quarantined-jobs/second.json");
        assert_eq!(digest(&quarantined).unwrap(), manifest_hash);
        let receipt = read_json(&dir.path().join("quarantined-jobs/second.receipt.json")).unwrap();
        assert_eq!(receipt["reason"], "test failure");
        assert_eq!(receipt["manifest"]["sha256"], manifest_hash);
        assert_eq!(receipt["report"]["sha256"], report_hash);
        assert_eq!(quarantine_receipts(dir.path()).unwrap(), vec![receipt]);
        assert!(quarantine_manifest(dir.path(), &quarantined, &job, "retry").is_err());
    }

    #[test]
    fn last_gate_refusal_prevents_spawn_after_output_setup() {
        let dir = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        let call = Invocation {
            executable: Pin {
                sha256: digest(&executable).unwrap(),
                path: executable,
            },
            cwd: dir.path().to_owned(),
            args: vec!["--list".into()],
            env: BTreeMap::new(),
        };
        let mut checked = false;
        let result = launch_verified(&call, dir.path(), "trial", || {
            checked = true;
            assert!(dir.path().join("trial.stdout").is_file());
            assert!(dir.path().join("trial.stderr").is_file());
            Err("stale launch gate".into())
        });
        assert!(checked);
        assert!(matches!(result, Err(error) if error.to_string() == "stale launch gate"));
    }

    #[test]
    fn output_placeholder_expands_in_environment_values() {
        let dir = tempfile::tempdir().unwrap();
        let executable = PathBuf::from("/usr/bin/env");
        let call = Invocation {
            executable: Pin {
                sha256: digest(&executable).unwrap(),
                path: executable,
            },
            cwd: dir.path().to_owned(),
            args: vec![],
            env: BTreeMap::from([(
                "RVLLM_QUEUE_OUTPUT_TEST".into(),
                "{output}/receipt.json".into(),
            )]),
        };
        let mut child = launch_verified(&call, dir.path(), "trial", || Ok(())).unwrap();
        assert!(child.0.wait().unwrap().success());
        let stdout = fs::read_to_string(dir.path().join("trial.stdout")).unwrap();
        assert!(stdout.lines().any(|line| {
            line == format!(
                "RVLLM_QUEUE_OUTPUT_TEST={}/receipt.json",
                dir.path().display()
            )
        }));
    }

    #[test]
    fn stopped_worker_returns_before_observer_or_manifest_reads() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("jobs")).unwrap();
        fs::create_dir(dir.path().join("results")).unwrap();
        fs::write(dir.path().join("jobs/must-not-read.json"), b"invalid").unwrap();
        fs::write(dir.path().join("STOP"), b"preserved").unwrap();
        run_owned(dir.path(), &dir.path().join("hardware.lock"), Some(0)).unwrap();
        assert_eq!(
            read_json(&dir.path().join("state.json")).unwrap()["status"],
            "stopped"
        );
        assert_eq!(fs::read(dir.path().join("STOP")).unwrap(), b"preserved");
        assert_eq!(fs::read_dir(dir.path().join("results")).unwrap().count(), 0);
        assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("power-")
        }));
    }

    #[test]
    fn idle_worker_does_not_start_a_power_observer() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("jobs")).unwrap();
        fs::create_dir(dir.path().join("results")).unwrap();
        run_owned(dir.path(), &dir.path().join("hardware.lock"), Some(0)).unwrap();
        assert_eq!(
            read_json(&dir.path().join("state.json")).unwrap()["status"],
            "idle"
        );
        assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("power-")
        }));
    }
}
