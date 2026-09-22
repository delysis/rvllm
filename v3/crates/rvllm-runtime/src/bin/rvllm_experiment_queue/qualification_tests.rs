use super::*;

fn fixture_job(root: &Path, id: &str) -> Job {
    serde_json::from_value(json!({
        "schema": SCHEMA, "id": id, "purpose": "preparation",
        "command": {"executable": {"path": root.join("fixture"), "sha256": "0".repeat(64)},
            "cwd": root, "args": []},
        "conditions": {"power_source": "ac", "low_power_mode": false, "pmset_power_mode": 0,
            "thermal_state": 0, "minimum_free_bytes": 1, "disk_path": root},
        "stable_seconds": 1, "max_wait_seconds": 10, "max_run_seconds": 10
    }))
    .unwrap()
}

#[test]
fn dependency_report_must_identify_its_job() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("results/first")).unwrap();
    atomic_json(
        &root.path().join("results/first/report.json"),
        &json!({"schema": "rvllm.experiment_result.v1", "id": "another-job", "status": "succeeded"}),
    )
    .unwrap();
    let mut job = fixture_job(root.path(), "second");
    job.after.push("first".into());
    assert!(
        dependencies_ready(&job, root.path()).is_err(),
        "a result for another job must not satisfy this dependency"
    );
}
