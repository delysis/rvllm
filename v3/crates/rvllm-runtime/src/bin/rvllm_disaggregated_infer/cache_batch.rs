//! Bounded cache preparation in fresh, serial child processes. No inference.
#![forbid(unsafe_code)]

use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

const PARTS: [(&str, &str, usize); 4] = [
    ("qkv", "QueryKeyValue", 48),
    ("output", "Output", 48),
    ("ffn-int8", "FeedForwardInt8", 48),
    ("head-attention", "VocabularyAndAttention", 18),
];

pub(super) struct Cancellation {
    signal: Arc<AtomicBool>,
    marker: Option<PathBuf>,
}

impl Cancellation {
    pub(super) fn install() -> Result<Self, String> {
        let signal = Arc::new(AtomicBool::new(false));
        let handler = signal.clone();
        ctrlc::try_set_handler(move || handler.store(true, Ordering::Release))
            .map_err(|e| format!("install cache shutdown handler: {e}"))?;
        Ok(Self {
            signal,
            marker: std::env::var_os("RVLLM_ANE_CACHE_CANCEL_FILE").map(PathBuf::from),
        })
    }

    pub(super) fn requested(&self) -> bool {
        self.signal.load(Ordering::Acquire)
            || self.marker.as_ref().is_some_and(|path| path.exists())
    }
}

// Even an I/O/reporting error must wait for the child to finish unloading.
// There is no forced kill or automatic retry of a private-framework operation.
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.wait();
    }
}

fn wait_child(
    child: &mut OwnedChild,
    cancellation: &Cancellation,
    marker: &Path,
) -> Result<ExitStatus, String> {
    let mut notified = false;
    loop {
        if cancellation.requested() && !notified {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(marker)
                .map_err(|e| format!("record cache cancellation: {e}"))?;
            notified = true;
        }
        if let Some(status) = child
            .0
            .try_wait()
            .map_err(|e| format!("observe cache child: {e}"))?
        {
            return Ok(status);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn unsigned(report: &Value, key: &str) -> Result<u64, String> {
    report[key]
        .as_u64()
        .ok_or_else(|| format!("missing cache receipt field {key}"))
}

fn validate_part(
    report: &Value,
    journal: &str,
    model: &Path,
    capacity: usize,
    part: &str,
    count: usize,
    inspect: bool,
) -> Result<Value, String> {
    let schema = if inspect {
        "rvllm.ane_cache_inspection.v1"
    } else {
        "rvllm.ane_cache_provision.v1"
    };
    if report["schema"] != schema
        || report["part"] != part
        || report["model_dir"] != json!(model)
        || unsigned(report, "global_context_capacity")? != capacity as u64
        || unsigned(report, "inference_evaluations")? != 0
        || report["cache_policy"]
            != if inspect {
                "require-existing"
            } else {
                "bounded-reuse-or-compile"
            }
        || unsigned(report, "maximum_explicit_compiles")? != if inspect { 0 } else { count as u64 }
    {
        return Err("cache child receipt identity or work mismatch".into());
    }
    let compiled = unsigned(report, "compiler_calls")?;
    let available = unsigned(report, "programs_prepared")?;
    if available > count as u64 || compiled > if inspect { 0 } else { count as u64 } {
        return Err("cache child exceeded its graph or compilation bound".into());
    }
    let mut missing = Vec::new();
    if inspect {
        let entries = report["entries"]
            .as_array()
            .ok_or("cache inspection entries missing")?;
        if entries.len() != count {
            return Err("cache inspection entry count mismatch".into());
        }
        let mut names = std::collections::BTreeSet::new();
        let mut present = 0;
        for entry in entries {
            let name = entry["name"].as_str().ok_or("cache entry name missing")?;
            if !names.insert(name) {
                return Err("duplicate cache inspection entry".into());
            }
            match entry["available"].as_bool() {
                Some(true) => present += 1,
                Some(false) => missing.push(name),
                None => return Err("cache availability flag missing".into()),
            }
        }
        let expected: std::collections::BTreeSet<_> =
            if part == "VocabularyAndAttention" && count >= 2 {
                [
                    "attention/sliding".to_string(),
                    "attention/global".to_string(),
                ]
                .into_iter()
                .chain((0..count - 2).map(|tile| format!("vocabulary/{}", tile * 16384)))
                .collect()
            } else {
                (0..count)
                    .map(|layer| format!("{part}/layer/{layer}"))
                    .collect()
            };
        if names
            .iter()
            .copied()
            .ne(expected.iter().map(String::as_str))
        {
            return Err("cache inspection visited unexpected graph names".into());
        }
        if present != available {
            return Err("cache availability totals disagree".into());
        }
    } else if available != count as u64 {
        return Err("cache preparation did not complete every graph".into());
    }
    let mut counts = BTreeMap::<String, u64>::new();
    let mut lifecycle = BTreeMap::<String, [u64; 3]>::new();
    for line in journal.lines() {
        let event: Value = serde_json::from_str(line).map_err(|e| format!("cache journal: {e}"))?;
        let stage = event["stage"]
            .as_str()
            .ok_or("cache journal stage missing")?;
        let id = event["model_id"]
            .as_str()
            .ok_or("cache journal model identity missing")?;
        *counts.entry(stage.into()).or_default() += 1;
        if stage.ends_with("_failed") || stage.starts_with("evaluate_") {
            return Err(format!("cache child has forbidden journal event {stage}"));
        }
        let slot = match stage {
            "load_completed" => Some(0),
            "unload_completed" => Some(1),
            "unload_returned" => Some(2),
            _ => None,
        };
        if let Some(slot) = slot {
            if id.is_empty() {
                return Err("cache lifecycle event has no model identity".into());
            }
            lifecycle.entry(id.into()).or_default()[slot] += 1;
        }
    }
    let count_stage = |stage: &str| counts.get(stage).copied().unwrap_or(0);
    if count_stage("compile_begin") != compiled
        || count_stage("compile_completed") != compiled
        || count_stage("descriptor_created") != count as u64
        || count_stage("load_begin") != available
        || count_stage("load_completed") != available
        || count_stage("unload_begin") != available
        || lifecycle
            .values()
            .any(|&n| n[0] == 0 || n[0] != n[1] || n[0] != n[2])
    {
        return Err("cache receipt and successful graph lifecycles disagree".into());
    }
    Ok(
        json!({"available":available,"missing":missing,"compiler_calls":compiled,
        "successful_unloads":count_stage("unload_completed"),"journal_events":counts}),
    )
}

fn save_report(
    root: &Path,
    model: &Path,
    capacity: usize,
    inspect: bool,
    parts: &[Value],
    status: &str,
    error: Option<&str>,
) -> Result<(), String> {
    let available: u64 = parts
        .iter()
        .filter_map(|p| p["validated"]["available"].as_u64())
        .sum();
    let compiled: u64 = parts
        .iter()
        .filter_map(|p| p["validated"]["compiler_calls"].as_u64())
        .sum();
    let complete = status == "complete";
    let report = json!({"schema":"rvllm.ane_cache_batch.v1","operation":if inspect {"inspect"} else {"prepare"},
        "model_dir":model,"global_context_capacity":capacity,"status":status,"error":error,"parts":parts,
        "expected_programs":162,"verified_available_programs":available,"verified_compiler_calls":compiled,
        "all_programs_available_at_visit":complete && available == 162,
        "claim":"Serial, bounded, fresh-process cache visits; availability can change. Does not prove simultaneous residency or numerical inference."});
    // Write a complete JSON document at each boundary; preserve a previous
    // completed-part receipt if writing the next one fails.
    let temporary = root.join("report.json.next");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(temporary, root.join("report.json")).map_err(|e| e.to_string())
}

pub(super) fn run_all(
    model: &Path,
    root: &Path,
    capacity: usize,
    inspect: bool,
    cancellation: &Cancellation,
) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let marker = root.join("cancel-requested");
    let mut completed = Vec::new();
    save_report(root, model, capacity, inspect, &completed, "running", None)?;
    for (flag, part, count) in PARTS {
        if cancellation.requested() {
            save_report(
                root,
                model,
                capacity,
                inspect,
                &completed,
                "cancelled",
                None,
            )?;
            return Err("ANE cache batch cancelled".into());
        }
        let directory = root.join(flag);
        let journal = directory.join("driver-phases.jsonl");
        eprintln!(
            "ANE cache {flag}: starting bounded {}",
            if inspect { "inspection" } else { "preparation" }
        );
        let outcome = (|| -> Result<Value, String> {
            let stdout = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(root.join(format!("{flag}.stdout.log")))
                .map_err(|e| e.to_string())?;
            let stderr = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(root.join(format!("{flag}.stderr.log")))
                .map_err(|e| e.to_string())?;
            let child = Command::new(&executable)
                .arg("--model-dir")
                .arg(model)
                .arg(if inspect {
                    "--inspect-ane-cache"
                } else {
                    "--prepare-ane-cache"
                })
                .arg(flag)
                .arg("--context-capacity")
                .arg(capacity.to_string())
                .arg("--output-dir")
                .arg(&directory)
                .env("RVLLM_ANE_DIAGNOSTIC_JOURNAL", &journal)
                .env("RVLLM_ANE_CACHE_CANCEL_FILE", &marker)
                .stdin(Stdio::null())
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr))
                .spawn()
                .map_err(|e| format!("spawn cache child: {e}"))?;
            let mut child = OwnedChild(child);
            let status = wait_child(&mut child, cancellation, &marker)?;
            if !status.success() {
                return Err(format!(
                    "cache child {flag} exited with {status}; see {}",
                    root.join(format!("{flag}.stderr.log")).display()
                ));
            }
            let report: Value = serde_json::from_slice(
                &std::fs::read(directory.join("report.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let journal = std::fs::read_to_string(&journal).map_err(|e| e.to_string())?;
            validate_part(&report, &journal, model, capacity, part, count, inspect)
        })();
        match outcome {
            Ok(validated) => {
                eprintln!(
                    "ANE cache {flag}: {} available, {} compiler calls",
                    validated["available"], validated["compiler_calls"]
                );
                completed.push(json!({"part":flag,"report":directory.join("report.json"),"validated":validated}));
                save_report(root, model, capacity, inspect, &completed, "running", None)?;
            }
            Err(error) => {
                completed.push(json!({"part":flag,"error":error}));
                let status = if cancellation.requested() {
                    "cancelled"
                } else {
                    "failed"
                };
                save_report(
                    root,
                    model,
                    capacity,
                    inspect,
                    &completed,
                    status,
                    Some(&error),
                )?;
                return Err(error);
            }
        }
    }
    let status = if cancellation.requested() {
        "cancelled"
    } else {
        "complete"
    };
    save_report(root, model, capacity, inspect, &completed, status, None)?;
    if status == "cancelled" {
        return Err("ANE cache batch cancelled".into());
    }
    println!(
        "ANE cache {} complete; receipt: {}",
        if inspect { "inspection" } else { "preparation" },
        root.join("report.json").display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn receipt(inspect: bool) -> Value {
        json!({"schema":if inspect {"rvllm.ane_cache_inspection.v1"} else {"rvllm.ane_cache_provision.v1"},
            "part":"QueryKeyValue","model_dir":"/model","global_context_capacity":1024,
            "inference_evaluations":0,"programs_prepared":1,"compiler_calls":0,
            "cache_policy":if inspect {"require-existing"} else {"bounded-reuse-or-compile"},
            "maximum_explicit_compiles":if inspect {0} else {2},
            "entries":[{"name":"QueryKeyValue/layer/0","available":true},{"name":"QueryKeyValue/layer/1","available":false}]})
    }
    fn journal() -> String {
        [
            ("descriptor_created", "a"),
            ("load_begin", "a"),
            ("load_completed", "a"),
            ("unload_begin", "a"),
            ("unload_completed", "a"),
            ("unload_returned", "a"),
            ("descriptor_created", "b"),
        ]
        .into_iter()
        .map(|(stage, model_id)| json!({"stage":stage,"model_id":model_id}).to_string() + "\n")
        .collect()
    }
    fn check(report: &Value, events: &str) -> Result<Value, String> {
        validate_part(
            report,
            events,
            Path::new("/model"),
            1024,
            "QueryKeyValue",
            2,
            true,
        )
    }
    #[test]
    fn inspection_distinguishes_missing_graphs_from_failed_visits() {
        let r = receipt(true);
        let good = check(&r, &journal()).unwrap();
        assert_eq!(good["available"], 1);
        assert_eq!(good["missing"], json!(["QueryKeyValue/layer/1"]));
        for stage in ["compile_begin", "evaluate_begin", "unload_failed"] {
            let events = journal() + &json!({"stage":stage,"model_id":"a"}).to_string() + "\n";
            assert!(check(&r, &events).is_err());
        }
        assert!(check(&r, &journal().replace("unload_completed", "unrelated")).is_err());
        assert!(check(&r, &journal().replace("unload_returned", "unrelated")).is_err());
        let wrong_identity: String = journal()
            .lines()
            .map(|line| {
                let mut event: Value = serde_json::from_str(line).unwrap();
                if event["stage"] == "unload_completed" {
                    event["model_id"] = json!("b");
                }
                event.to_string() + "\n"
            })
            .collect();
        assert!(check(&r, &wrong_identity).is_err());
    }
    #[test]
    fn mismatched_or_incomplete_child_receipts_are_rejected() {
        for (key, bad) in [
            ("part", json!("Output")),
            ("model_dir", json!("/other")),
            ("global_context_capacity", json!(64)),
            ("inference_evaluations", json!(1)),
            ("compiler_calls", json!(1)),
            ("programs_prepared", json!(2)),
        ] {
            let mut r = receipt(true);
            r[key] = bad;
            assert!(check(&r, &journal()).is_err(), "accepted {key}");
        }
        let mut r = receipt(true);
        r["entries"][1]["name"] = json!("QueryKeyValue/layer/0");
        assert!(check(&r, &journal()).is_err());
        assert!(validate_part(
            &receipt(false),
            &journal(),
            Path::new("/model"),
            1024,
            "QueryKeyValue",
            2,
            false
        )
        .is_err());
    }

    #[test]
    fn preparation_requires_completed_compiles_and_unloads() {
        let mut r = receipt(false);
        r["programs_prepared"] = json!(2);
        r["compiler_calls"] = json!(1);
        let mut events = journal();
        for stage in [
            "compile_begin",
            "compile_completed",
            "load_begin",
            "load_completed",
            "unload_begin",
            "unload_completed",
            "unload_returned",
        ] {
            events += &(json!({"stage":stage,"model_id":"b"}).to_string() + "\n");
        }
        let check = |events: &str| {
            validate_part(
                &r,
                events,
                Path::new("/model"),
                1024,
                "QueryKeyValue",
                2,
                false,
            )
        };
        let valid = check(&events).unwrap();
        assert_eq!(valid["compiler_calls"], 1);
        assert_eq!(valid["successful_unloads"], 2);
        assert!(check(&events.replace("compile_completed", "incomplete")).is_err());
        assert!(check(&events.replace("unload_returned", "incomplete")).is_err());
    }
}
