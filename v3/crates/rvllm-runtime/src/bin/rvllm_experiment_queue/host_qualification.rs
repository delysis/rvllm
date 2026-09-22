//! Explicit, bounded live host scheduling with stock true/touch children only.
#![forbid(unsafe_code)]

use super::*;

pub(super) fn run(output: &Path, global_lock: &Path, conditions_job: &Path) -> Result<()> {
    if !output.is_absolute() || !global_lock.is_absolute() || !conditions_job.is_absolute() {
        return Err("qualification paths must be absolute".into());
    }
    // Read ONLY the conditions. Never execute the supplied campaign command.
    let source_bytes = fs::read(conditions_job)?;
    let source_hash = format!("{:x}", Sha256::digest(&source_bytes));
    let source: Job = serde_json::from_slice(&source_bytes)?;
    let accelerator_guard = lock(global_lock)?;
    fs::create_dir(output)?; // A fresh evidence directory, never an old campaign.
    let mut saved_source = File::create_new(output.join("conditions-job.json"))?;
    saved_source.write_all(&source_bytes)?;
    saved_source.sync_all()?;
    let result = qualify(output, &accelerator_guard, source.conditions).and_then(|()| {
        if digest(conditions_job)? != source_hash {
            return Err("conditions source changed during qualification".into());
        }
        Ok(())
    });
    let report = json!({"schema":"rvllm.host_scheduling_qualification.v1",
        "status":if result.is_ok(){"passed"}else{"failed"},
        "error":result.as_ref().err().map(|e|e.to_string()),
        "worker_sha256":digest(&std::env::current_exe()?).ok(),
        "conditions_job":conditions_job,"conditions_job_sha256":source_hash,
        "accelerator_trials_started":0,
        "claim":"Live host gate/scheduling only; true and touch are the only trial programs. No model load, numerical, ANE, S2, KV-import or throughput qualification. Failed attempts are retained without retries."});
    atomic_json(&output.join("qualification.json"), &report)?;
    result
}

fn qualify(output: &Path, accelerator_guard: &File, conditions: Conditions) -> Result<()> {
    fs::create_dir(output.join("jobs"))?;
    fs::create_dir(output.join("results"))?;
    let _queue_guard = lock(&output.join("worker.lock"))?;
    let invocation = |path: &str, args: Vec<String>| -> Result<Invocation> {
        let path = PathBuf::from(path);
        Ok(Invocation {
            executable: Pin {
                sha256: digest(&path)?,
                path,
            },
            cwd: output.to_owned(),
            args,
            env: BTreeMap::new(),
        })
    };
    let first = Job {
        schema: SCHEMA.into(),
        id: "00-unready-control".into(),
        purpose: Purpose::Preparation,
        command: invocation("/usr/bin/true", vec![])?,
        validator: None,
        inputs: vec![],
        after: vec![],
        conditions: conditions.clone(),
        stable_seconds: 2,
        max_wait_seconds: 30,
        max_run_seconds: 5,
    };
    let mut blocked = first.clone();
    blocked.conditions.minimum_free_bytes = u64::MAX;
    let mut ready = first;
    ready.id = "01-ready".into();
    let mut dependent = ready.clone();
    dependent.id = "02-dependent-stop".into();
    dependent.after.push(ready.id.clone());
    dependent.command = invocation(
        "/usr/bin/touch",
        vec![output
            .join("STOP")
            .to_str()
            .ok_or("qualification path must be UTF-8")?
            .to_owned()],
    )?;
    for job in [&blocked, &ready, &dependent] {
        job.validate()?;
        job.verify_files()?;
        atomic_json(
            &output.join("jobs").join(format!("{}.json", job.id)),
            &serde_json::to_value(job)?,
        )?;
    }
    // The SAME global lease covers setup, all hash work, observation and children.
    run_locked(output, 0, accelerator_guard)?;
    if output.join("results").join(&blocked.id).exists() || !output.join("STOP").is_file() {
        return Err("blocked job ran or dependent stop did not finish".into());
    }
    for id in [&ready.id, &dependent.id] {
        let report = successful_result(output, id)?.ok_or("host fixture did not execute")?;
        if report["purpose"] != "preparation" || report["sampled_conditions_eligible"] != false {
            return Err("host fixture claimed timing eligibility".into());
        }
    }
    Ok(())
}
