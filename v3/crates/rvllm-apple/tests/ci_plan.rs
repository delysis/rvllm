use std::{fs, path::PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn ci_workflow() -> String {
    let path = repo_root().join(".github/workflows/ci.yml");
    fs::read_to_string(path).expect("ci workflow should be readable")
}

fn job_section(workflow: &str, job: &str) -> String {
    let job_header = format!("  {job}:");
    let mut in_job = false;
    let mut lines = Vec::new();

    for line in workflow.lines() {
        if line == job_header {
            in_job = true;
        } else if in_job
            && line.starts_with("  ")
            && !line.starts_with("    ")
            && line.trim_end().ends_with(':')
        {
            break;
        }

        if in_job {
            lines.push(line);
        }
    }

    assert!(in_job, "missing CI job: {job}");
    lines.join("\n")
}

#[test]
fn ci_plan_has_expected_apple_and_host_jobs() {
    let workflow = ci_workflow();

    assert!(workflow.contains("linux-host-tests"));
    assert!(workflow.contains("macos-metal-tests"));
    assert!(workflow.contains("private-ane-tests"));
    assert!(workflow.contains("cargo test -p rvllm-apple --no-default-features"));
    assert!(workflow.contains("cargo test -p rvllm-runtime --features apple --no-default-features"));
}

#[test]
fn private_ane_tests_are_manual_only_and_feature_gated() {
    let workflow = ci_workflow();
    let section = job_section(&workflow, "private-ane-tests");

    assert!(workflow.contains("run_private_ane"));
    assert!(section.contains("github.event_name == 'workflow_dispatch'"));
    assert!(section.contains("inputs.run_private_ane"));
    assert!(section.contains("self-hosted"));
    assert!(section.contains("environment: private-ane"));
    assert!(section.contains("--features private-ane"));
}

#[test]
fn normal_ci_jobs_do_not_enable_private_ane() {
    let workflow = ci_workflow();

    for job in ["linux-host-tests", "macos-metal-tests"] {
        let section = job_section(&workflow, job);
        assert!(!section.contains("run_private_ane"));
        assert!(!section.contains("private-ane"));
        assert!(!section.contains("--features private-ane"));
    }
}
