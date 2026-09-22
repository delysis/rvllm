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

fn fixture_queue() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("jobs")).unwrap();
    fs::create_dir(root.path().join("results")).unwrap();
    root
}

fn submit_fixture(root: &Path, job: &Job) {
    atomic_json(
        &root.join("jobs").join(format!("{}.json", job.id)),
        &serde_json::to_value(job).unwrap(),
    )
    .unwrap();
}

fn successful_report(id: &str) -> Value {
    json!({"schema":"rvllm.experiment_result.v1","id":id,"status":"succeeded",
        "purpose":"preparation","exit_code":0,"files_unchanged":true,"overdue":false,
        "signal_or_missing_exit_code":false,"sampled_conditions_eligible":false,"validation":null})
}

fn complete_fixture(root: &Path, job: &Job, status: &str) {
    let output = root.join("results").join(&job.id);
    fs::create_dir(&output).unwrap();
    atomic_json(
        &output.join("job.json"),
        &serde_json::to_value(job).unwrap(),
    )
    .unwrap();
    let mut report = successful_report(&job.id);
    report["status"] = json!(status);
    atomic_json(&output.join("report.json"), &report).unwrap();
}

fn ready(conditions: &Conditions) -> Result<Value> {
    Ok(json!({"ready":conditions.minimum_free_bytes != u64::MAX,
        "power":{"sample":{"controls":{"thermal_state":0}}}}))
}

fn assert_census_refuses(root: &Path) {
    let result = scheduling::select(
        root,
        &mut BTreeMap::new(),
        |_| panic!("must inspect every attempt before probing any candidate"),
        Instant::now,
    );
    assert!(result.is_err());
}

#[test]
fn dependency_report_must_identify_its_job() {
    let root = fixture_queue();
    fs::create_dir_all(root.path().join("results/first")).unwrap();
    atomic_json(
        &root.path().join("results/first/report.json"),
        &successful_report("another-job"),
    )
    .unwrap();
    let mut job = fixture_job(root.path(), "second");
    job.after.push("first".into());
    assert!(dependencies_ready(&job, root.path()).is_err());
}

#[test]
fn receipt_schema_and_incomplete_attempts_cannot_satisfy_dependencies() {
    let root = fixture_queue();
    let mut job = fixture_job(root.path(), "second");
    job.after.push("first".into());
    assert!(!dependencies_ready(&job, root.path()).unwrap());
    let output = root.path().join("results/first");
    fs::create_dir(&output).unwrap();
    assert!(dependencies_ready(&job, root.path()).is_err());
    for report in [
        json!({"id":"first","status":"succeeded"}),
        json!({"schema":"wrong","id":"first","status":"succeeded"}),
        json!({"schema":"rvllm.experiment_result.v1","id":"first","status":"running"}),
        json!({"schema":"rvllm.experiment_result.v1","id":"first","status":"failed"}),
    ] {
        atomic_json(&output.join("report.json"), &report).unwrap();
        assert!(dependencies_ready(&job, root.path()).is_err());
    }
}

#[test]
fn later_failure_blocks_an_earlier_otherwise_ready_job() {
    let root = fixture_queue();
    let earlier = fixture_job(root.path(), "00-ready");
    let later = fixture_job(root.path(), "99-failed");
    for job in [&earlier, &later] {
        submit_fixture(root.path(), job);
    }
    complete_fixture(root.path(), &later, "failed");
    assert_census_refuses(root.path());
    assert!(!root.path().join("results/00-ready").exists());
}

#[test]
fn removing_a_manifest_does_not_hide_its_abandoned_attempt() {
    let root = fixture_queue();
    submit_fixture(root.path(), &fixture_job(root.path(), "00-ready"));
    fs::create_dir(root.path().join("results/99-abandoned")).unwrap();
    assert_census_refuses(root.path());
}

#[test]
fn altered_completed_manifest_is_not_reused_as_success() {
    let root = fixture_queue();
    let mut done = fixture_job(root.path(), "99-done");
    complete_fixture(root.path(), &done, "succeeded");
    done.command.args.push("changed".into());
    submit_fixture(root.path(), &done);
    submit_fixture(root.path(), &fixture_job(root.path(), "00-ready"));
    assert_census_refuses(root.path());
}

#[test]
fn filename_and_job_identity_must_agree_before_result_lookup() {
    let root = fixture_queue();
    submit_fixture(root.path(), &fixture_job(root.path(), "00-ready"));
    let path = root.path().join("jobs/99-alias.json");
    for id in ["00-ready", "../outside", "/absolute"] {
        atomic_json(
            &path,
            &serde_json::to_value(fixture_job(root.path(), id)).unwrap(),
        )
        .unwrap();
        assert_census_refuses(root.path());
    }
}

#[test]
fn missing_and_cyclic_dependencies_are_rejected_before_probes() {
    let root = fixture_queue();
    let mut a = fixture_job(root.path(), "a");
    let mut b = fixture_job(root.path(), "b");
    a.after.push("b".into());
    submit_fixture(root.path(), &a);
    assert_census_refuses(root.path());
    b.after.push("a".into());
    submit_fixture(root.path(), &b);
    assert_census_refuses(root.path());
}

#[test]
fn independent_ready_job_and_its_dependency_make_progress_without_relaxing_gates() {
    let root = fixture_queue();
    let mut blocked = fixture_job(root.path(), "00-blocked");
    blocked.conditions.minimum_free_bytes = u64::MAX;
    let runnable = fixture_job(root.path(), "01-ready");
    let mut dependent = fixture_job(root.path(), "02-dependent");
    dependent.after.push(runnable.id.clone());
    for job in [&blocked, &runnable, &dependent] {
        submit_fixture(root.path(), job);
    }
    let now = Instant::now();
    let mut waiting = BTreeMap::new();
    assert!(scheduling::select(root.path(), &mut waiting, ready, || now)
        .unwrap()
        .0
        .is_none());
    let selected = scheduling::select(root.path(), &mut waiting, ready, || {
        now + Duration::from_secs(1)
    })
    .unwrap()
    .0
    .unwrap();
    assert_eq!(selected.id, runnable.id);
    assert_eq!(waiting[&blocked.id].0, now);
    assert!(waiting[&blocked.id].1.since.is_none());
    complete_fixture(root.path(), &selected, "succeeded");
    assert!(scheduling::select(root.path(), &mut waiting, ready, || now
        + Duration::from_secs(2))
    .unwrap()
    .0
    .is_none());
    let selected = scheduling::select(root.path(), &mut waiting, ready, || {
        now + Duration::from_secs(3)
    })
    .unwrap()
    .0
    .unwrap();
    assert_eq!(selected.id, dependent.id);
    assert!(!root.path().join("results/00-blocked").exists());
}

#[test]
fn observation_gap_resets_quiet_window_without_extending_wait_deadline() {
    let root = fixture_queue();
    let job = fixture_job(root.path(), "ready");
    submit_fixture(root.path(), &job);
    let now = Instant::now();
    let mut waiting = BTreeMap::new();
    for seconds in [0, 3] {
        assert!(scheduling::select(root.path(), &mut waiting, ready, || now
            + Duration::from_secs(seconds))
        .unwrap()
        .0
        .is_none());
    }
    assert_eq!(waiting["ready"].0, now);
    assert_eq!(waiting["ready"].1.since, Some(now + Duration::from_secs(3)));
    assert!(scheduling::select(root.path(), &mut waiting, ready, || now
        + Duration::from_secs(11))
    .is_err());
}

#[test]
fn listener_identity_requires_the_same_pid_and_loopback_port() {
    assert!(listener_report_matches(b"p22\nn127.0.0.1:8093\n", 22, 8093));
    assert!(listener_report_matches(b"p22\nn*:8093\n", 22, 8093));
    for bytes in [
        b"p23\nn127.0.0.1:8093\n".as_slice(),
        b"p22\nn127.0.0.1:8094\n",
        b"p22\nn192.0.2.1:8093\n",
        b"p22\np23\nn*:8093\n",
        b"n*:8093\n",
        b"p22\nn[::1]:8093\n",
        b"garbage",
    ] {
        assert!(!listener_report_matches(bytes, 22, 8093));
    }
}

#[test]
fn live_loopback_listener_is_bound_to_its_actual_owner() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    assert!(listener_owned_by(std::process::id(), port));
    assert!(!listener_owned_by(u32::MAX, port));
}

#[test]
fn preparation_never_claims_timing_eligibility() {
    for raw in [false, true] {
        assert!(!timing_eligible(Purpose::Preparation, raw));
        assert_eq!(timing_eligible(Purpose::Timing, raw), raw);
    }
}

#[test]
fn active_sampling_rejects_gaps_even_when_both_snapshots_are_ready() {
    let start = Instant::now();
    for (gap, expected) in [(0, true), (2500, true), (2501, false)] {
        let mut value = json!({"ready":true});
        check_observation_gap(&mut value, start, start + Duration::from_millis(gap));
        assert_eq!(value["ready"], expected);
        assert_eq!(value["observation_gap_ms"], gap as f64);
    }
    let mut value = json!({"ready":true});
    check_observation_gap(&mut value, start + Duration::from_secs(1), start);
    assert_eq!(value["ready"], false);
}

#[test]
fn real_child_launch_has_a_positive_control_and_no_side_effect_after_refusal() {
    let root = fixture_queue();
    let marker = root.path().join("child-ran");
    let path = PathBuf::from("/usr/bin/touch");
    let call = Invocation {
        executable: Pin {
            sha256: digest(&path).unwrap(),
            path,
        },
        cwd: root.path().to_owned(),
        args: vec![marker.to_str().unwrap().into()],
        env: BTreeMap::new(),
    };
    verify_pin(&call.executable).unwrap();
    assert!(launch_verified(&call, root.path(), "refused", || Err("STOP".into())).is_err());
    assert!(!marker.exists());
    let mut child = launch_verified(&call, root.path(), "permitted", || Ok(())).unwrap();
    assert!(child.0.wait().unwrap().success());
    assert!(marker.is_file());
}

#[test]
fn archived_campaign_receipts_remain_compatible_without_replaying_them() {
    let archive = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../reports/gemma4-12b-evidence-20260914/experiment-queue-20260916/baseline-isolated-v7",
    );
    let root = fixture_queue();
    let mut pending = 0;
    let mut completed = 0;
    for path in manifest_paths(&archive).unwrap() {
        let mut job: Job =
            serde_json::from_reader(BufReader::new(File::open(path).unwrap())).unwrap();
        let saved = archive.join("results").join(&job.id);
        if saved.is_dir() {
            let output = root.path().join("results").join(&job.id);
            fs::create_dir(&output).unwrap();
            for name in ["report.json", "job.json"] {
                fs::copy(saved.join(name), output.join(name)).unwrap();
            }
            completed += 1;
        } else {
            // Relocate ONLY pending cwd/disk probes; historical receipt bytes stay exact.
            job.command.cwd = root.path().to_owned();
            if let Some(validator) = &mut job.validator {
                validator.cwd = root.path().to_owned();
            }
            job.conditions.disk_path = root.path().to_owned();
            pending += 1;
        }
        submit_fixture(root.path(), &job);
    }
    let (_, observations) = scheduling::select(
        root.path(),
        &mut BTreeMap::new(),
        |_| Ok(json!({"ready":false})),
        Instant::now,
    )
    .unwrap();
    assert_eq!((completed, pending), (7, 29));
    assert_eq!(observations.len(), 29);
    assert_eq!(
        fs::read_dir(root.path().join("results")).unwrap().count(),
        7
    );
}

#[test]
fn internally_inconsistent_success_is_not_accepted() {
    let root = fixture_queue();
    fs::create_dir(root.path().join("results/first")).unwrap();
    let mut job = fixture_job(root.path(), "second");
    job.after.push("first".into());
    for (key, value) in [
        ("exit_code", json!(1)),
        ("files_unchanged", json!(false)),
        ("overdue", json!(true)),
        ("signal_or_missing_exit_code", json!(true)),
        ("purpose", json!("timing")),
        ("trial_started", json!(false)),
        ("validation", json!({"success":false})),
    ] {
        let mut report = successful_report("first");
        report[key] = value;
        atomic_json(&root.path().join("results/first/report.json"), &report).unwrap();
        assert!(
            dependencies_ready(&job, root.path()).is_err(),
            "accepted {key}"
        );
    }
    atomic_json(
        &root.path().join("results/first/report.json"),
        &successful_report("first"),
    )
    .unwrap();
    assert!(dependencies_ready(&job, root.path()).unwrap());
}

#[test]
fn pin_audit_caches_observed_bytes_not_another_jobs_expected_hash() {
    let root = fixture_queue();
    let path = root.path().join("fixture");
    fs::write(&path, b"actual executable bytes").unwrap();
    let mut good = fixture_job(root.path(), "a-correct");
    good.command.executable.sha256 = digest(&path).unwrap();
    let bad = fixture_job(root.path(), "b-changed");
    for job in [&good, &bad] {
        submit_fixture(root.path(), job);
    }
    fs::write(root.path().join("STOP"), b"preserved").unwrap();
    let _queue_guard = lock(&root.path().join("worker.lock")).unwrap();
    let _accelerator_guard = lock(&root.path().join("hardware.lock")).unwrap();
    let report = scheduling::audit(root.path()).unwrap();
    assert_eq!(report["all_pins_match"], false);
    assert_eq!(report["pending_jobs"][0]["pins"][0]["matches"], true);
    assert_eq!(report["pending_jobs"][1]["pins"][0]["matches"], false);
    assert_eq!(
        report["pending_jobs"][0]["pins"][0]["actual_sha256"],
        report["pending_jobs"][1]["pins"][0]["actual_sha256"]
    );
    assert_eq!(fs::read(root.path().join("STOP")).unwrap(), b"preserved");
    assert_eq!(
        fs::read_dir(root.path().join("results")).unwrap().count(),
        0
    );
}

#[test]
fn result_purpose_and_required_validator_are_bound_to_the_manifest() {
    for has_validator in [false, true] {
        let root = fixture_queue();
        let mut job = fixture_job(root.path(), "done");
        if has_validator {
            job.validator = Some(job.command.clone());
        } else {
            job.purpose = Purpose::Timing;
        }
        submit_fixture(root.path(), &job);
        complete_fixture(root.path(), &job, "succeeded");
        assert_census_refuses(root.path());
    }
}
