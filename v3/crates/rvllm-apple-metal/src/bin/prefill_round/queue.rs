//! Immutable manifests for the existing rvllm.experiment_job.v1 queue.
//! No subprocess or accelerator access in this module. Every timing job owns
//! exactly one arm; numerical admission happens again inside `sample`.
#![forbid(unsafe_code)]

use super::{abs, admitted_screen, hash, library, load, median, save, text, Result, ARMS, BASE};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};

fn pin(path: &Path) -> Result<Value> {
    let path = fs::canonicalize(path)?;
    Ok(json!({"path": path, "sha256": hash(&path)?}))
}

fn identifier(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 96
        || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("invalid queue identifier".into());
    }
    Ok(())
}

fn config(path: &Path) -> Result<Value> {
    let value = load(path)?;
    if value["schema"] != "rvllm.prefill-round.queue-config.v1" {
        return Err("wrong prefill queue config schema".into());
    }
    identifier(text(&value, "campaign")?)?;
    for key in ["root", "referee", "driver", "libraries", "referee_lock"] {
        abs(text(&value, key)?)?;
    }
    let referee = fs::canonicalize(text(&value, "referee")?)?;
    if hash(&referee)? != hash(&std::env::current_exe()?)? {
        return Err("queue referee is not the running, source-bound executable".into());
    }
    if !value["conditions"].is_object() {
        return Err("explicit queue condition policy is required".into());
    }
    // Zero dwell; the existing queue records conditions and enforces its own
    // complete policy. This generator does not silently relax that policy.
    Ok(value)
}

fn input_pins(c: &Value, arms: &[&str]) -> Result<Vec<Value>> {
    let mut pins = vec![pin(&abs(text(c, "driver")?)?)?];
    let directory = abs(text(c, "libraries")?)?;
    for arm in arms {
        library(arm, &directory)?; // Source must match this executable.
        for suffix in ["metal", "source.json", "metallib"] {
            pins.push(pin(&directory.join(format!("{arm}.{suffix}")))?);
        }
    }
    Ok(pins)
}

fn emit(c: &Value, root: &Path, id: &str, purpose: &str, args: Vec<String>,
        inputs: Vec<Value>, after: Vec<String>) -> Result<PathBuf> {
    identifier(id)?;
    let job = json!({
        "schema": "rvllm.experiment_job.v1", "id": id, "purpose": purpose,
        "command": {"executable": pin(&abs(text(c, "referee")?)?)?,
                    "cwd": root, "args": args, "env": {}},
        "inputs": inputs, "after": after, "conditions": c["conditions"],
        "stable_seconds": 0, "max_wait_seconds": 7200, "max_run_seconds": 3600
    });
    let path = root.join("jobs").join(format!("{id}.json"));
    if path.exists() { return Err("refusing to replace a queue manifest".into()); }
    save(&path, &job)?;
    Ok(path)
}

pub fn screen_jobs(path: &Path) -> Result<()> {
    let c = config(path)?;
    let root = abs(text(&c, "root")?)?;
    let limit = c["limit"].as_u64().ok_or("limit required")?;
    if limit > u32::MAX as u64 || !super::plan::RUNGS.contains(&(limit as u32)) {
        return Err("unpublished correctness limit".into());
    }
    let selections = c["selections"].as_array().ok_or("selections required")?;
    if selections.is_empty() || selections.len() > 32 {
        return Err("select 1..32 arm/operator jobs, not an unbounded campaign".into());
    }
    fs::create_dir(&root)?;
    fs::create_dir(root.join("jobs"))?;
    fs::create_dir(root.join("screens"))?;
    let mut outputs = Vec::new();
    for (index, selection) in selections.iter().enumerate() {
        let arm = text(selection, "arm")?;
        let role = text(selection, "role")?;
        if !ARMS.contains(&arm) { return Err("unknown arm".into()); }
        valid_role(arm, role)?;
        let id = format!("{}-s{index:02}-m{limit}", text(&c, "campaign")?);
        let output = root.join("screens").join(&id);
        let args = vec!["screen".into(), arm.into(), role.into(), limit.to_string(),
            text(&c, "driver")?.into(), text(&c, "libraries")?.into(),
            output.to_string_lossy().into_owned(), text(&c, "referee_lock")?.into()];
        // Screen arms are independent. The queue owns accelerator serialization;
        // making one arm depend on another would hide later evidence on failure.
        let job = emit(&c, &root, &id, "correctness", args, input_pins(&c, &[arm])?, Vec::new())?;
        outputs.push(json!({"id": id, "job": job, "arm": arm, "role": role,
                           "screen": output.join("screen.json")}));
    }
    save(&root.join("index.json"), &json!({"base": BASE, "jobs": outputs,
        "claim": "one arm/operator per job; each advances internally only after all prior numerical fixtures pass",
        "production_promotion": false}))
}

fn valid_role(arm: &str, role: &str) -> Result<()> {
    let attention = ["attention-local", "attention-global"].contains(&role);
    let projection = ["qkv-local", "qkv-global", "gate-up", "gate-gelu", "o-local", "o-global",
        "down", "o-local-norm", "o-global-norm", "down-norm"].contains(&role);
    let candidate = arm.parse::<super::MetalResearchCandidate>().ok();
    let supported = if attention {
        ["normal", "simd-attention"].contains(&arm)
            || candidate.is_some_and(|c| super::plan::attention_kernel(c, 256).is_some())
    } else {
        projection && (["normal", "mma32", "metal-mma32-load4"].contains(&arm)
            || candidate.is_some_and(|c| super::plan::projection_kernels(c).is_some()))
    };
    if !supported { return Err("arm does not own requested operator".into()); }
    Ok(())
}

/// Call only after both screens have completed. Their retained cells become
/// immutable input pins. The generated ABBA-BAAB manifests are eight separate
/// native queue jobs, not one mixed-arm timing process.
pub fn timing_jobs(path: &Path) -> Result<()> {
    let c = config(path)?;
    let root = abs(text(&c, "root")?)?;
    let a = text(&c, "a")?;
    let b = text(&c, "b")?;
    let role = text(&c, "role")?;
    let m = c["m"].as_u64().ok_or("target M required")?;
    if ![256, 512, 1024, 2048].contains(&m) || a == b { return Err("two distinct arms/target M required".into()); }
    valid_role(a, role)?;
    valid_role(b, role)?;
    let driver = abs(text(&c, "driver")?)?;
    let libs = abs(text(&c, "libraries")?)?;
    let sa = abs(text(&c, "screen_a")?)?;
    let sb = abs(text(&c, "screen_b")?)?;
    admitted_screen(&sa, a, role, m as u32, &driver, &libs)?;
    admitted_screen(&sb, b, role, m as u32, &driver, &libs)?;
    let mut inputs = input_pins(&c, &[a, b])?;
    for screen in [&sa, &sb] {
        inputs.push(pin(screen)?);
        for cell in load(screen)?["completed"].as_array().ok_or("screen cells")? {
            inputs.push(pin(&abs(text(cell, "path")?)?)?);
        }
    }
    fs::create_dir(&root)?;
    fs::create_dir(root.join("jobs"))?;
    fs::create_dir(root.join("samples"))?;
    let mut after = Vec::new();
    let mut entries = Vec::new();
    for (index, arm) in [a, b, b, a, b, a, a, b].into_iter().enumerate() {
        let id = format!("{}-t{index:02}-m{m}", text(&c, "campaign")?);
        let output = root.join("samples").join(&id);
        let screen = if arm == a { &sa } else { &sb };
        let args = vec!["sample".into(), arm.into(), role.into(), m.to_string(),
            screen.to_string_lossy().into_owned(), driver.to_string_lossy().into_owned(),
            libs.to_string_lossy().into_owned(), output.to_string_lossy().into_owned(),
            text(&c, "referee_lock")?.into()];
        let job = emit(&c, &root, &id, "exploratory_timing", args, inputs.clone(), after)?;
        entries.push(json!({"id": id, "job": job, "arm": arm, "cell": output.join("cell.json")}));
        after = vec![id];
    }
    save(&root.join("adjudication.json"), &json!({"schema": "rvllm.prefill-round.adjudication.v1",
        "base": BASE, "a": a, "b": b, "role": role, "m": m, "samples": entries,
        "driver": driver, "libraries": libs, "production_promotion": false}))
}

/// Read-only adjudication; does not invoke the driver. Conditions stay explicitly
/// unqualified until the eight external queue receipts are independently reviewed.
pub fn adjudicate(path: &Path, output: &Path) -> Result<()> {
    let p = load(path)?;
    if p["schema"] != "rvllm.prefill-round.adjudication.v1" || p["base"] != BASE {
        return Err("adjudication schema/base".into());
    }
    let a = text(&p, "a")?;
    let b = text(&p, "b")?;
    let driver_sha = hash(&abs(text(&p, "driver")?)?)?;
    let referee_sha = hash(&std::env::current_exe()?)?;
    let libs = abs(text(&p, "libraries")?)?;
    let entries = p["samples"].as_array().ok_or("samples missing")?;
    if entries.len() != 8 || a == b { return Err("incomplete counterbalanced design".into()); }
    let mut runs: Vec<Value> = Vec::new();
    for (entry, arm) in entries.iter().zip([a, b, b, a, b, a, a, b]) {
        let cell_path = abs(text(entry, "cell")?)?;
        let c = load(&cell_path)?;
        let (_, body, libsha) = library(arm, &libs)?;
        if entry["arm"] != arm || c["schema"] != "rvllm.prefill-round.cell.v1" || c["status"] != "pass"
            || c["arm"] != arm || c["role"] != p["role"] || c["m"] != p["m"] || c["base"] != BASE
            || c["fixture"] != "structured" || c["timer_valid"] != true || c["source_body_sha256"] != body
            || c["driver_sha256"] != driver_sha || c["referee_sha256"] != referee_sha || c["library_sha256"] != libsha
            || c["oracle"]["complete_elements"] != true || c["oracle"]["bitwise_repeat"] != true
        { return Err("unqualified, reordered, or differently sealed sample".into()); }
        if hash(&abs(text(&c, "output_file")?)?)? != text(&c, "output_sha256")? {
            return Err("sample raw output changed".into());
        }
        if let Some(first) = runs.first() {
            if first["input_sha256"] != c["fixture_read_only_sha256"] { return Err("mismatched input bytes".into()); }
            let matched_schedule = [a,b].iter().all(|v| v.parse::<super::MetalResearchCandidate>()
                .ok().is_some_and(|v| super::plan::projection_kernels(v).is_some()));
            if matched_schedule && first["output_sha256"] != c["output_sha256"] {
                return Err("matched projection schedules differ bitwise".into());
            }
        }
        let samples = c["samples"].as_array().ok_or("timers missing")?.iter()
            .map(|v| v["gpu_ns"].as_f64().ok_or("missing GPU timer"))
            .collect::<std::result::Result<Vec<_>,_>>()?;
        runs.push(json!({"arm": arm, "median_gpu_ns": median(&samples)?, "cell": cell_path,
            "cell_sha256": hash(&cell_path)?, "input_sha256": c["fixture_read_only_sha256"],
            "output_sha256": c["output_sha256"]}));
    }
    let times = |arm: &str| runs.iter().filter(|v| v["arm"] == arm)
        .map(|v| v["median_gpu_ns"].as_f64().expect("checked timer")).collect::<Vec<_>>();
    let ta = times(a); let tb = times(b);
    let drift = |values: &[f64]| {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            / values.iter().copied().fold(f64::INFINITY, f64::min) - 1.0
    };
    let stable = drift(&ta) <= 0.05 && drift(&tb) <= 0.05;
    // This is a new tournament-local 5% timing gate, NOT a modification to any
    // production, existing kernel-game, or numerical tolerance.
    let result = json!({"schema": "rvllm.prefill-round.adjudicated.v1", "base": BASE,
        "status": if stable {"timing-screen-pass"} else {"timing-screen-rejected-drift"},
        "a": a, "b": b, "role": p["role"], "m": p["m"], "order": "ABBA-BAAB",
        "a_drift": drift(&ta), "b_drift": drift(&tb), "descriptive_a_over_b": median(&ta)?/median(&tb)?,
        "runs": runs, "plan_sha256": hash(path)?, "conditions_qualified": false,
        "production_promotion": false});
    if output.exists() { return Err("refusing to replace adjudication".into()); }
    save(output, &result)?;
    if !stable { return Err("5% within-arm drift gate failed; retained all samples".into()); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifiers_and_operator_ownership_fail_closed() {
        assert!(identifier("p26-a_0").is_ok());
        for bad in ["", "../x", "foo bar"] { assert!(identifier(bad).is_err()); }
        assert!(valid_role("normal", "down-norm").is_ok());
        assert!(valid_role("metal-prefill-pipeline32x64", "qkv-global").is_ok());
        assert!(valid_role("metal-prefill-q4k16", "down").is_err());
        assert!(valid_role("simd-attention", "attention-global").is_ok());
    }
}
