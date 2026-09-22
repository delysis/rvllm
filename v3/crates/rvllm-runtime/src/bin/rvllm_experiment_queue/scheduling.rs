//! Inspect the whole queue before selecting any candidate. No device calls.
#![forbid(unsafe_code)]

use super::*;

type Waiting = BTreeMap<String, (Instant, StableGate)>;

/// A lexically early ready job cannot hide a later failed/incomplete attempt.
/// Successful receipts must belong to the exact submitted manifest.
fn pending_jobs(queue: &Path) -> Result<Vec<Job>> {
    let mut jobs = BTreeMap::new();
    for path in manifest_paths(queue)? {
        let job: Job = serde_json::from_reader(BufReader::new(File::open(&path)?))?;
        if job.schema != SCHEMA
            || !valid_id(&job.id)
            || path.file_stem().and_then(|s| s.to_str()) != Some(job.id.as_str())
        {
            return Err(format!("manifest filename/identity mismatch: {}", path.display()).into());
        }
        if jobs.insert(job.id.clone(), job).is_some() {
            return Err("duplicate job identity".into());
        }
    }
    // Removing a manifest must not hide its failed or abandoned attempt.
    for entry in fs::read_dir(queue.join("results"))? {
        let entry = entry?;
        let id = entry
            .file_name()
            .into_string()
            .map_err(|_| "non-UTF-8 result identity")?;
        if !entry.file_type()?.is_dir() || !jobs.contains_key(&id) {
            return Err(format!("orphan or non-directory result {id}; review required").into());
        }
    }
    let mut completed = BTreeSet::new();
    for job in jobs.values() {
        if let Some(report) = successful_result(queue, &job.id)? {
            if report["purpose"] != serde_json::to_value(job.purpose)?
                || (job.validator.is_some() && report["validation"]["success"] != true)
            {
                return Err(format!("result policy differs from manifest {}", job.id).into());
            }
            let saved: Job = serde_json::from_reader(BufReader::new(File::open(
                queue.join("results").join(&job.id).join("job.json"),
            )?))?;
            if serde_json::to_value(&saved)? != serde_json::to_value(job)? {
                return Err(format!("attempted manifest {} changed; no replay", job.id).into());
            }
            completed.insert(job.id.clone());
        } else {
            job.validate()?;
        }
        if job.after.iter().any(|id| !jobs.contains_key(id)) {
            return Err(format!("job {} has a missing dependency", job.id).into());
        }
    }
    // Normal submit order prevents cycles; reject externally altered graphs too.
    let mut resolved = completed.clone();
    loop {
        let before = resolved.len();
        for job in jobs.values() {
            if job.after.iter().all(|id| resolved.contains(id)) {
                resolved.insert(job.id.clone());
            }
        }
        if resolved.len() == jobs.len() {
            break;
        }
        if resolved.len() == before {
            return Err("cyclic pending dependencies; no trial started".into());
        }
    }
    Ok(jobs
        .into_values()
        .filter(|job| !completed.contains(&job.id))
        .collect())
}

pub(super) fn select(
    queue: &Path,
    waiting: &mut Waiting,
    mut probe: impl FnMut(&Conditions) -> Result<Value>,
    clock: impl Fn() -> Instant,
) -> Result<(Option<Job>, Vec<Value>)> {
    // Finish this census before the first probe or selection, including after restart.
    let jobs = pending_jobs(queue)?;
    let mut pending = Vec::new();
    for job in jobs {
        if !dependencies_ready(&job, queue)? {
            pending.push(json!({"id":job.id,"waiting_for_dependencies":job.after}));
            continue;
        }
        let now = clock();
        let (started, gate) = waiting
            .entry(job.id.clone())
            .or_insert_with(|| (now, StableGate::new()));
        if now.saturating_duration_since(*started) > Duration::from_secs(job.max_wait_seconds) {
            return Err(format!(
                "job {} expired waiting for conditions; no trial started",
                job.id
            )
            .into());
        }
        let observation = probe(&job.conditions)?;
        let observed = clock();
        pending.push(json!({"id":job.id,
            "wait_seconds":observed.saturating_duration_since(*started).as_secs(),
            "conditions":observation}));
        if gate.observe(
            &observation,
            observed,
            Duration::from_secs(job.stable_seconds),
        ) {
            return Ok((Some(job), pending));
        }
    }
    Ok((None, pending))
}

/// Hash only while both worker and accelerator locks are held by the caller.
/// Cache raw digests, never a job's expected-hash decision.
pub(super) fn audit(queue: &Path) -> Result<Value> {
    let jobs = pending_jobs(queue)?;
    let mut hashes = BTreeMap::<PathBuf, std::result::Result<String, String>>::new();
    let mut reports = Vec::new();
    let mut all_match = true;
    for job in &jobs {
        let mut pins = Vec::new();
        for pin in std::iter::once(&job.command.executable)
            .chain(job.validator.iter().map(|call| &call.executable))
            .chain(job.inputs.iter())
        {
            let actual = hashes
                .entry(pin.path.clone())
                .or_insert_with(|| digest(&pin.path).map_err(|e| e.to_string()));
            let matches = actual
                .as_ref()
                .is_ok_and(|hash| *hash == pin.sha256.to_ascii_lowercase());
            all_match &= matches;
            pins.push(json!({"path":pin.path,"expected_sha256":pin.sha256,
                "actual_sha256":actual.as_ref().ok(),"error":actual.as_ref().err(),"matches":matches}));
        }
        reports.push(json!({"id":job.id,"pins":pins,"conditions":job.conditions}));
    }
    Ok(json!({
        "schema":"rvllm.experiment_preflight.v1","queue":queue,
        "worker_sha256":digest(&std::env::current_exe()?)?,
        "stop_present":queue.join("STOP").exists(),"pending_jobs":reports,
        "all_pins_match":all_match,"accelerator_trials_started":0,
        "claim":"Read-only campaign inspection under exclusive locks. No STOP removal, power qualification, live scheduling, numerical acceptance or performance claim. Pin validity is not cached for later launches."
    }))
}
