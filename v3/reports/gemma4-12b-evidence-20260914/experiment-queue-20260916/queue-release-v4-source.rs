//! Disk-backed jobs, pinned executables, sampled launch gates and serial ownership.
#![forbid(unsafe_code)]

use rvllm_runtime::apple_measurement::PowerMonitor;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
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
    low_power_mode: bool,
    pmset_power_mode: u64,
    thermal_state: Option<u64>,
    minimum_free_bytes: u64,
    disk_path: PathBuf,
    #[serde(default)]
    quiet_process_names: Vec<String>,
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
            || c.pmset_power_mode > 2
            || c.thermal_state.is_some_and(|state| state > 1)
            || (self.purpose == Purpose::Timing && c.thermal_state.is_none())
            || !c.disk_path.is_absolute()
            || !c.disk_path.is_dir()
            || !(1..=600).contains(&self.stable_seconds)
            || self.max_wait_seconds < self.stable_seconds
            || self.max_wait_seconds > 86400
            || !(1..=3600).contains(&self.max_run_seconds)
            || c.quiet_process_names
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
        for pin in &self.inputs {
            if !pin.path.is_absolute()
                || pin.sha256.len() != 64
                || !pin.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            {
                return Err("invalid input pin".into());
            }
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
        Ok(())
    }

    fn files_present(&self) -> Result<()> {
        for pin in std::iter::once(&self.command.executable)
            .chain(self.validator.iter().map(|call| &call.executable))
            .chain(self.inputs.iter())
        {
            pin_present(pin)?;
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
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
        && controls["low_power_mode"] == c.low_power_mode
        && controls["pmset_power_mode"] == c.pmset_power_mode
        && matches!(controls["thermal_state"].as_u64(), Some(0 | 1))
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

fn probe(monitor: &PowerMonitor, c: &Conditions, child: Option<u32>) -> Result<Value> {
    let started = Instant::now();
    let available = free_bytes(&c.disk_path)?;
    let result = Command::new("/bin/ps")
        .args(["-axo", "pid=,ppid=,comm="])
        .output()?;
    if !result.status.success() {
        return Err("process activity unavailable".into());
    }
    let processes = parse_processes(std::str::from_utf8(&result.stdout)?)?;
    let idle_checks: Vec<_> = c
        .idle_llama_servers
        .iter()
        .map(|s| idle_server(s, &processes))
        .collect();
    let mut competing = blockers(&processes, &c.quiet_process_names, child);
    competing.retain(|p| {
        !idle_checks
            .iter()
            .any(|s| s["idle"] == true && s["pid"] == p["pid"])
    });
    let observation = monitor.latest_observation();
    let probe_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(json!({"ready":controls_match(&observation,c)
        && available >= c.minimum_free_bytes && competing.is_empty()
        && idle_checks.iter().all(|s|s["idle"]==true) && probe_ms<=2500.0,
        "power":observation,"free_bytes":available,"competing_processes":competing,
        "idle_server_checks":idle_checks,"probe_ms":probe_ms}))
}

struct StableGate {
    since: Option<Instant>,
    last_observed: Option<Instant>,
    controls: Option<Value>,
}

impl StableGate {
    fn new() -> Self {
        Self {
            since: None,
            last_observed: None,
            controls: None,
        }
    }
    fn observe(&mut self, probe: &Value, now: Instant, required: Duration) -> bool {
        let controls = &probe["power"]["sample"]["controls"];
        if probe["ready"] != true {
            *self = Self::new();
            return false;
        }
        let continuous = self
            .last_observed
            .and_then(|last| now.checked_duration_since(last))
            .is_some_and(|gap| gap <= Duration::from_millis(2500));
        if !continuous || self.controls.as_ref() != Some(controls) {
            self.controls = Some(controls.clone());
            self.since = Some(now);
        }
        self.last_observed = Some(now);
        self.since
            .is_some_and(|start| now.duration_since(start) >= required)
    }
}

/// Never let an early error detach a live accelerator child or release its lock.
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.wait();
    }
}

fn launch(call: &Invocation, output: &Path, label: &str) -> Result<OwnedChild> {
    verify_pin(&call.executable)?;
    let directory = output.to_str().ok_or("output path is not UTF-8")?;
    let mut command = Command::new(&call.executable.path);
    command
        .current_dir(&call.cwd)
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("LANG", "C")
        .envs(&call.env)
        .args(
            call.args
                .iter()
                .map(|arg| arg.replace("{output}", directory)),
        )
        .stdin(Stdio::null())
        .stdout(File::create_new(output.join(format!("{label}.stdout")))?)
        .stderr(File::create_new(output.join(format!("{label}.stderr")))?);
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

fn execute(
    job: &Job,
    queue: &Path,
    monitor: &PowerMonitor,
    stop: &AtomicBool,
    expected_controls: &Value,
) -> Result<Option<bool>> {
    job.verify_files()?;
    let current = probe(monitor, &job.conditions, None)?;
    if stopped(queue, stop)
        || current["ready"] != true
        || current["power"]["sample"]["controls"] != *expected_controls
    {
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
    let mut child = launch(&job.command, &output, "trial")?;
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
        && (job.purpose == Purpose::Preparation || eligible);
    if accepted {
        if let Some(call) = &job.validator {
            let mut validator = launch(call, &output, "validation")?;
            let status = validator.0.wait()?;
            accepted = status.success();
            validation = json!({"exit_code":status.code(),"success":status.success()});
        }
    }
    let report = json!({"schema":"rvllm.experiment_result.v1","id":job.id,"purpose":job.purpose,
        "status":if accepted {"succeeded"}else{"failed"},
        "exit_code":exit.code(),"signal_or_missing_exit_code":exit.code().is_none(),
        "sampled_conditions_eligible":eligible,"violations":violations,"overdue":overdue,
        "files_unchanged":files_unchanged.is_ok(),"file_error":files_unchanged.err(),
        "validation":validation,"measurement":measurement,
        "stop_requested":stopped(queue,stop),
        "claim":"Preparation success does not qualify performance. Outer process duration includes startup and is not token throughput. CPU counters belong to the queue, excluding its child. Backend reports/validators establish numerical correctness and phase timing. Sampled conditions cannot prove fixed clocks or absence of all competing work."});
    atomic_json(&output.join("report.json"), &report)?;
    eprintln!("experiment {}: {}", job.id, report["status"]);
    Ok(Some(accepted))
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

fn dependencies_ready(job: &Job, queue: &Path) -> Result<bool> {
    for dependency in &job.after {
        let report = queue.join("results").join(dependency).join("report.json");
        if !report.exists() {
            return Ok(false);
        }
        let value = read_json(&report)?;
        if value["status"] != "succeeded" {
            return Err(format!("dependency {dependency} did not succeed; queue stopped").into());
        }
    }
    Ok(true)
}

fn run(queue: &Path, accelerator_lock: &Path, idle_seconds: u64) -> Result<()> {
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

fn run_owned(queue: &Path, accelerator_lock: &Path, idle_seconds: u64) -> Result<()> {
    let _accelerator_lock = lock(accelerator_lock)?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let monitor = PowerMonitor::start(Some(
        &queue.join(format!("power-{stamp}-{}.jsonl", std::process::id())),
    ))?;
    let mut idle = Instant::now();
    let mut waiting = BTreeMap::<String, (Instant, StableGate)>::new();
    loop {
        if stopped(queue, &stop) {
            atomic_json(
                &queue.join("state.json"),
                &json!({"status":"stopped","pid":std::process::id()}),
            )?;
            return Ok(());
        }
        let mut selected = None;
        let mut pending = Vec::new();
        for path in manifest_paths(queue)? {
            let job: Job = serde_json::from_reader(File::open(&path)?)?;
            let result = queue.join("results").join(&job.id);
            if result.exists() {
                let report = read_json(&result.join("report.json"))?;
                if report["status"] != "succeeded" {
                    return Err(format!(
                        "job {} has failed or incomplete output; no replay",
                        job.id
                    )
                    .into());
                }
                continue;
            }
            job.validate()?;
            if !dependencies_ready(&job, queue)? {
                pending.push(json!({"id":job.id,"waiting_for_dependencies":job.after}));
                continue;
            }
            let (started, gate) = waiting
                .entry(job.id.clone())
                .or_insert_with(|| (Instant::now(), StableGate::new()));
            if started.elapsed() > Duration::from_secs(job.max_wait_seconds) {
                return Err(format!(
                    "job {} expired waiting for conditions; no trial started",
                    job.id
                )
                .into());
            }
            let observation = probe(&monitor, &job.conditions, None)?;
            pending.push(
                json!({"id":job.id,"wait_seconds":started.elapsed().as_secs(),
                "conditions":observation}),
            );
            if gate.observe(
                &observation,
                Instant::now(),
                Duration::from_secs(job.stable_seconds),
            ) {
                selected = Some((job, observation["power"]["sample"]["controls"].clone()));
                break;
            }
        }
        if let Some((job, controls)) = selected {
            idle = Instant::now();
            if stopped(queue, &stop) {
                continue;
            }
            atomic_json(
                &queue.join("state.json"),
                &json!({"status":"starting","id":job.id,"pid":std::process::id()}),
            )?;
            match execute(&job, queue, &monitor, &stop, &controls)? {
                Some(false) => {
                    return Err(
                        format!("job {} failed; queue stopped without retry", job.id).into(),
                    )
                }
                Some(true) => {
                    waiting.remove(&job.id);
                    // Running this trial interrupts every other job's quiet
                    // window, even when the child finishes between samples.
                    for (_, gate) in waiting.values_mut() {
                        *gate = StableGate::new();
                    }
                }
                None => {
                    // Hashing inputs takes time. If conditions changed, keep
                    // the original waiting deadline and begin a fresh window.
                    if let Some((_, gate)) = waiting.get_mut(&job.id) {
                        *gate = StableGate::new();
                    }
                }
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
            if !pending.is_empty() {
                idle = Instant::now();
            }
            if idle.elapsed() >= Duration::from_secs(idle_seconds) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

pub(super) fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let usage="usage: rvllm_experiment_queue submit QUEUE JOB.json | run QUEUE GLOBAL_LOCK [IDLE_SECONDS] | status QUEUE | stop QUEUE";
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
            if destination.exists() || queue.join("results").join(&job.id).exists() {
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
            run(&queue, &lock_path, idle)
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
                let job: Job = serde_json::from_reader(File::open(path)?)?;
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
                    &json!({"state":if state.exists(){read_json(&state)?}else{Value::Null},"jobs":jobs})
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
            low_power_mode: true,
            pmset_power_mode: 1,
            thermal_state: Some(0),
            minimum_free_bytes: 1,
            disk_path: PathBuf::from("/"),
            quiet_process_names: vec!["cargo".into()],
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
    fn stable_window_resets_on_activity_or_any_control_change() {
        let mut gate = StableGate::new();
        let now = Instant::now();
        let needed = Duration::from_secs(5);
        let mut p = json!({"ready":true,"power":observation()});
        assert!(!gate.observe(&p, now, needed));
        for seconds in 1..5 {
            assert!(!gate.observe(&p, now + Duration::from_secs(seconds), needed));
        }
        assert!(gate.observe(&p, now + needed, needed));
        p["ready"] = json!(false);
        assert!(!gate.observe(&p, now + needed, needed));
        p["ready"] = json!(true);
        assert!(!gate.observe(&p, now + needed, needed));
        p["power"]["sample"]["controls"]["available_cpus"] = json!(16);
        assert!(!gate.observe(&p, now + needed + Duration::from_secs(1), needed));
    }

    #[test]
    fn stable_window_rejects_observation_gaps_and_backwards_time() {
        let mut gate = StableGate::new();
        let now = Instant::now();
        let needed = Duration::from_secs(5);
        let p = json!({"ready":true,"power":observation()});
        for seconds in [0, 2, 4] {
            assert!(!gate.observe(&p, now + Duration::from_secs(seconds), needed));
        }
        assert!(gate.observe(&p, now + Duration::from_secs(5), needed));
        // Equal controls on either side of an unobserved interval do not
        // establish uninterrupted quiet. The whole window must start again.
        for seconds in [8, 10, 12] {
            assert!(!gate.observe(&p, now + Duration::from_secs(seconds), needed));
        }
        assert!(gate.observe(&p, now + Duration::from_secs(13), needed));
        assert!(!gate.observe(&p, now + Duration::from_secs(12), needed));
    }

    #[test]
    fn preparation_can_accept_benign_thermals_but_never_unknown_or_serious() {
        let mut c = conditions();
        c.thermal_state = None;
        let mut value = observation();
        assert!(controls_match(&value, &c));
        value["sample"]["controls"]["thermal_state"] = json!(1);
        assert!(controls_match(&value, &c));
        for state in [json!(2), json!(3), Value::Null] {
            value["sample"]["controls"]["thermal_state"] = state;
            assert!(!controls_match(&value, &c));
        }
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
            after: vec!["first".into()],
            conditions: conditions(),
            stable_seconds: 1,
            max_wait_seconds: 10,
            max_run_seconds: 10,
        };
        assert!(dependencies_ready(&job, dir.path()).is_err());
    }
}
