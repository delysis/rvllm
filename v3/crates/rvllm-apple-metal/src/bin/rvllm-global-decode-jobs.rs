//! Offline, explicit queue preparation. No accelerator execution, implicit
//! submission, shell, environment-based dispatch, winner selection or promotion.
#![forbid(unsafe_code)]
use rvllm_apple_metal::{MetalFloatType, MetalKernelOptions, MetalResearchCandidate};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
const PREFIX: &str = "RVLLM_METAL_GLOBAL_DECODE_";

fn write(path: &Path, bytes: &[u8]) -> Result {
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn json_new(path: &Path, value: &Value) -> Result {
    write(path, &serde_json::to_vec_pretty(value)?)
}
fn json_new_or_identical(path: &Path, value: &Value) -> Result {
    let bytes = serde_json::to_vec_pretty(value)?;
    match std::fs::File::create_new(path) {
        Ok(mut file) => {
            file.write_all(&bytes)?;
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if read(path)? == *value {
                Ok(())
            } else {
                Err(format!("immutable artifact differs: {}", path.display()).into())
            }
        }
        Err(error) => Err(error.into()),
    }
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn absolute(text: &str) -> Result<PathBuf> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err("paths must be absolute".into());
    }
    Ok(path)
}
fn hash(path: &Path) -> Result<String> {
    let output = Command::new("/usr/bin/shasum")
        .args(["-a", "256", "--"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err("shasum failed".into());
    }
    let line = String::from_utf8(output.stdout)?;
    let value = line.split_whitespace().next().ok_or("no SHA256")?;
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("bad SHA256".into());
    }
    Ok(value.to_ascii_lowercase())
}
fn pin(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":hash(path)?}))
}
fn candidates() -> Vec<MetalResearchCandidate> {
    rvllm_apple_metal::research_catalog::ALL_CANDIDATES
        .iter()
        .copied()
        .filter(|c| c.global_decode_tile().is_some() || c.split_global_decode_tile().is_some())
        .collect()
}
fn validate_selected(selected: &[String]) -> Result {
    let candidates = candidates();
    for (index, name) in selected.iter().enumerate() {
        if !candidates.iter().any(|candidate| candidate.name() == name)
            || selected[..index].contains(name)
        {
            return Err(format!("unknown or duplicate selected candidate: {name}").into());
        }
    }
    Ok(())
}
fn validate_timing_request(length: u32, selected: &[String]) -> Result {
    if !matches!(length, 256 | 512 | 1024 | 2048 | 4096) {
        return Err("timing length must be 256, 512, 1024, 2048, or 4096".into());
    }
    validate_selected(selected)
}
fn tool(name: &str) -> Result<PathBuf> {
    let output = Command::new("/usr/bin/xcrun")
        .args(["--sdk", "macosx", "--find", name])
        .output()?;
    if !output.status.success() {
        return Err(format!("missing Xcode tool {name}").into());
    }
    absolute(String::from_utf8(output.stdout)?.trim())
}
fn compile(source: &Path, directory: &Path) -> Result {
    std::fs::create_dir(directory)?;
    let source_hash = hash(source)?;
    let metal = tool("metal")?;
    let linker = tool("metallib")?;
    let compiler_pins = json!([
        pin(Path::new("/usr/bin/xcrun"))?,
        pin(&metal)?,
        pin(&linker)?
    ]);
    let air = directory.join("kernel.air");
    let library = directory.join("kernel.metallib");
    let compile_args = vec![
        "--sdk".into(),
        "macosx".into(),
        "metal".into(),
        "-std=metal3.1".into(),
        "-fno-fast-math".into(),
        "-c".into(),
        source.to_string_lossy().into_owned(),
        "-o".into(),
        air.to_string_lossy().into_owned(),
    ];
    let link_args = vec![
        "--sdk".into(),
        "macosx".into(),
        "metallib".into(),
        air.to_string_lossy().into_owned(),
        "-o".into(),
        library.to_string_lossy().into_owned(),
    ];
    for (name, args) in [("metal", &compile_args), ("metallib", &link_args)] {
        let output = Command::new("/usr/bin/xcrun").args(args).output()?;
        write(&directory.join(format!("{name}.stdout")), &output.stdout)?;
        write(&directory.join(format!("{name}.stderr")), &output.stderr)?;
        if !output.status.success() {
            return Err(format!("{name} failed; preserve logs").into());
        }
    }
    if hash(source)? != source_hash
        || compiler_pins
            != json!([
                pin(Path::new("/usr/bin/xcrun"))?,
                pin(&metal)?,
                pin(&linker)?
            ])
    {
        return Err("source/compiler identity changed during compilation".into());
    }
    json_new(
        &directory.join("build.json"),
        &json!({"schema":"rvllm.global-decode.build.v1",
        "status":"compiled","source_sha256":source_hash,"metallib_sha256":hash(&library)?,
        "flags":["-std=metal3.1","-fno-fast-math"],"compile_argv":compile_args,
        "link_argv":link_args,"tool_pins":compiler_pins,"native_execution":false}),
    )
}
fn id(config: &Value, candidate: MetalResearchCandidate, suffix: &str) -> Result<String> {
    let campaign = config["campaign"].as_str().ok_or("campaign missing")?;
    if let Some(tile) = candidate.global_decode_tile() {
        Ok(format!(
            "{campaign}-r{}p{}t{}-{suffix}",
            tile.rows, tile.panel, tile.threads
        ))
    } else {
        let tile = candidate
            .split_global_decode_tile()
            .ok_or("candidate has no global decode identity")?;
        Ok(format!(
            "{campaign}-split-r{}s{}t{}-{suffix}",
            tile.rows, tile.partition, tile.threads
        ))
    }
}
fn succeeded(queue: &Path, id: &str) -> Result<PathBuf> {
    let directory = queue.join("results").join(id);
    let report = read(&directory.join("report.json"))?;
    if report["status"] != "succeeded" || report["files_unchanged"] != true {
        return Err(
            format!("required queue result is not a successful unchanged-input job: {id}").into(),
        );
    }
    Ok(directory)
}
#[allow(clippy::too_many_arguments)]
fn job(
    config: &Value,
    root: &Path,
    job_id: &str,
    purpose: &str,
    executable: &Path,
    args: Vec<String>,
    env: Value,
    inputs: Vec<Value>,
    after: Vec<String>,
) -> Result {
    let value = json!({"schema":"rvllm.experiment_job.v1","id":job_id,"purpose":purpose,
        "command":{"executable":pin(executable)?,"cwd":root,"args":args,"env":env},
        "inputs":inputs,"after":after,"conditions":config["conditions"],
        "stable_seconds":0,"max_wait_seconds":7200,"max_run_seconds":3600});
    json_new_or_identical(&root.join("jobs").join(format!("{job_id}.json")), &value)
}

fn advancement_id(config: &Value, length: u32) -> Result<String> {
    Ok(format!(
        "{}-advance-L{length}",
        config["campaign"].as_str().ok_or("campaign missing")?
    ))
}

fn advancement_job(
    config: &Value,
    root: &Path,
    length: u32,
    stage_jobs: &[String],
    expected: &[String],
) -> Result<String> {
    let job_id = advancement_id(config, length)?;
    let executable = absolute(
        config["job_generator"]["path"]
            .as_str()
            .ok_or("job generator missing")?,
    )?;
    let mut args = vec![
        "advance".into(),
        root.to_string_lossy().into_owned(),
        length.to_string(),
        "{output}".into(),
    ];
    args.extend(expected.iter().cloned());
    job(
        config,
        root,
        &job_id,
        "preparation",
        &executable,
        args,
        json!({}),
        vec![pin(&root.join("campaign.json"))?],
        stage_jobs.to_vec(),
    )?;
    Ok(job_id)
}
fn source_path(root: &Path, c: MetalResearchCandidate, flavor: &str) -> PathBuf {
    root.join(format!("{}-{flavor}.metal", c.name()))
}
fn prepare(campaign: &str, root: &Path, queue: &Path, test: &Path, conditions: &Path) -> Result {
    if campaign.is_empty()
        || campaign.len() > 32
        || !campaign
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("campaign ID must be 1..32 ASCII letters/digits/-/_".into());
    }
    let policy = read(conditions)?;
    // The power source is always explicit. Other controls may be intentionally
    // unconstrained for exploratory campaigns; the queue records every sampled
    // value so later analysis can stratify rather than waiting for an idealized
    // host state. Confirmation campaigns should pin all four controls.
    if !matches!(policy["power_source"].as_str(), Some("ac" | "battery"))
        || !(policy["low_power_mode"].is_null() || policy["low_power_mode"].is_boolean())
        || !(policy["pmset_power_mode"].is_null()
            || matches!(policy["pmset_power_mode"].as_u64(), Some(0..=2)))
        || !(policy["thermal_state"].is_null()
            || matches!(policy["thermal_state"].as_u64(), Some(0..=2)))
    {
        return Err(
            "power source must be explicit; optional controls must be null or valid".into(),
        );
    }
    std::fs::create_dir(root)?;
    std::fs::create_dir(root.join("jobs"))?;
    let executable = std::env::current_exe()?;
    let queue_runner = executable
        .parent()
        .ok_or("generator has no parent directory")?
        .join("rvllm_experiment_queue");
    let retainer = executable
        .parent()
        .ok_or("generator has no parent directory")?
        .join("rvllm-retain-abba");
    let exploratory = policy["low_power_mode"].is_null()
        || policy["pmset_power_mode"].is_null()
        || policy["thermal_state"].is_null();
    let config = json!({"schema":"rvllm.global-decode.campaign.v1","campaign":campaign,
        "queue":queue,"test_executable":pin(test)?,"job_generator":pin(&executable)?,
        "queue_runner":pin(&queue_runner)?,
        "abba_retainer":pin(&retainer)?,"exploratory":exploratory,
        "conditions":policy,"conditions_input":pin(conditions)?,
        "screen_length":256,"advancement_lengths":[512,1024,2048],
        "deferred_confirmation_lengths":[4096],
        "split_kv":false,"local_prefill":false,
        "promotion":false,"status":"proposed_unqualified"});
    json_new(&root.join("campaign.json"), &config)?;
    let mut all = Vec::new();
    let metal = tool("metal")?;
    let linker = tool("metallib")?;
    for candidate in candidates() {
        let core = rvllm_apple_metal::kernels::kernel_source_with_options(
            MetalFloatType::Bf16,
            MetalKernelOptions {
                research: candidate,
                ..MetalKernelOptions::default()
            },
        );
        for flavor in ["core", "oracle"] {
            let path = source_path(root, candidate, flavor);
            let mut bytes = core.as_bytes().to_vec();
            if flavor == "oracle" {
                bytes.push(b'\n');
                bytes.extend_from_slice(include_bytes!(
                    "../research_shaders/global_decode_oracle.metal"
                ));
            }
            write(&path, &bytes)?;
            let job_id = id(&config, candidate, &format!("compile-{flavor}"))?;
            job(
                &config,
                root,
                &job_id,
                "preparation",
                &executable,
                vec![
                    "compile".into(),
                    path.to_string_lossy().into_owned(),
                    "{output}/build".into(),
                ],
                json!({}),
                vec![
                    pin(&path)?,
                    pin(&root.join("campaign.json"))?,
                    pin(Path::new("/usr/bin/xcrun"))?,
                    pin(&metal)?,
                    pin(&linker)?,
                ],
                vec![],
            )?;
            all.push(job_id);
        }
    }
    json_new(&root.join("compile-jobs.json"), &json!(all))
}
fn generate(root: &Path, timing: bool, length: Option<u32>, selected: &[String]) -> Result {
    let config_path = root.join("campaign.json");
    let config = read(&config_path)?;
    let queue = absolute(config["queue"].as_str().ok_or("queue missing")?)?;
    let test = absolute(
        config["test_executable"]["path"]
            .as_str()
            .ok_or("test missing")?,
    )?;
    if pin(&test)? != config["test_executable"]
        || pin(&std::env::current_exe()?)? != config["job_generator"]
    {
        return Err("test/generator executable identity drift".into());
    }
    if timing
        && candidates().into_iter().any(|candidate| {
            candidate.split_global_decode_tile().is_some()
                && (selected.is_empty() || selected.iter().any(|name| name == candidate.name()))
        })
    {
        return Err(
            "split-KV timing is fail-closed until the v2 two-kernel receipt/verifier is implemented"
                .into(),
        );
    }
    let retainer = if timing {
        let path = absolute(
            config["abba_retainer"]["path"]
                .as_str()
                .ok_or("ABBA retainer missing")?,
        )?;
        if pin(&path)? != config["abba_retainer"] {
            return Err("ABBA retainer identity drift".into());
        }
        Some(path)
    } else {
        None
    };
    let timing_length = if timing {
        Some(length.unwrap_or(256))
    } else {
        None
    };
    validate_selected(selected)?;
    if let Some(length) = timing_length {
        validate_timing_request(length, selected)?;
    }
    let mut all = Vec::new();
    // Never prune the family from partial results. A failed compile/oracle stops
    // generation; revised families require a new explicit campaign identity.
    for candidate in candidates() {
        if !selected.is_empty() && !selected.iter().any(|name| name == candidate.name()) {
            continue;
        }
        // The split oracle validates the production partial+merge entry points
        // directly and therefore needs the exact core source.  Only the
        // single-pass oracle uses the appended diagnostic entry point.
        let flavor = if timing || candidate.split_global_decode_tile().is_some() {
            "core"
        } else {
            "oracle"
        };
        let compile_id = id(&config, candidate, &format!("compile-{flavor}"))?;
        let built = succeeded(&queue, &compile_id)?.join("build");
        let source = source_path(root, candidate, flavor);
        let library = built.join("kernel.metallib");
        let build_path = built.join("build.json");
        let build = read(&build_path)?;
        if build["status"] != "compiled"
            || build["source_sha256"] != hash(&source)?
            || build["metallib_sha256"] != hash(&library)?
        {
            return Err("compiled identity mismatch".into());
        }
        let mut inputs = vec![
            pin(&source)?,
            pin(&library)?,
            pin(&build_path)?,
            pin(&config_path)?,
        ];
        let mut after = vec![compile_id];
        let mut env = json!({});
        env[format!("{PREFIX}CANDIDATE")] = json!(candidate.name());
        env[format!("{PREFIX}SOURCE")] = json!(source);
        env[format!("{PREFIX}METALLIB")] = json!(library);
        env[format!("{PREFIX}BUILD_RECEIPT")] = json!(build_path);
        if timing {
            inputs.push(pin(&test)?);
            let oracle_id = id(&config, candidate, "oracle")?;
            let oracle_path = succeeded(&queue, &oracle_id)?.join(
                if candidate.split_global_decode_tile().is_some() {
                    "native/split-oracle.json"
                } else {
                    "native/oracle.json"
                },
            );
            let oracle = read(&oracle_path)?;
            if oracle["status"] != "passed"
                || oracle["identity"]["candidate"] != candidate.name()
                || oracle["identity"]["core_sha256"] != hash(&source)?
                || oracle["identity"]["test_executable_sha256"]
                    != config["test_executable"]["sha256"]
            {
                return Err(
                    "passed native oracle with exact source/executable identity required".into(),
                );
            }
            inputs.push(pin(&oracle_path)?);
            env[format!("{PREFIX}ORACLE_RECEIPT")] = json!(oracle_path);
            after.push(oracle_id);
            for case in oracle["cases"].as_array().ok_or("oracle cases missing")? {
                let path = absolute(case["bf16_file"].as_str().ok_or("oracle output missing")?)?;
                if case["bf16_sha256"] != hash(&path)? {
                    return Err("oracle output changed".into());
                }
                inputs.push(pin(&path)?);
            }
        }
        let lengths = vec![timing_length];
        for length in lengths {
            let suffix = length.map_or("oracle".into(), |n| format!("abba-L{n}"));
            let job_id = id(&config, candidate, &suffix)?;
            // The existing queue expands {output} in argv ONLY, never in env.
            // Its exclusive result directory is results/<immutable job ID>.
            env[format!("{PREFIX}REPORT_DIR")] =
                json!(queue.join("results").join(&job_id).join("native"));
            let test_name = if timing {
                "global_decode_abba"
            } else if candidate.split_global_decode_tile().is_some() {
                "global_decode_split_device_oracle"
            } else {
                "global_decode_device_oracle"
            };
            if let Some(n) = length {
                env[format!("{PREFIX}LENGTH")] = json!(n.to_string());
            }
            let mut args = vec![
                "--ignored".into(),
                "--exact".into(),
                format!("attention_global_decode_device_tests::{test_name}"),
                "--test-threads=1".into(),
                "--nocapture".into(),
            ];
            let command = if let Some(retainer) = &retainer {
                let test_sha = config["test_executable"]["sha256"]
                    .as_str()
                    .ok_or("test executable SHA-256 missing")?;
                args.insert(0, test_sha.into());
                args.insert(0, test.to_string_lossy().into_owned());
                retainer
            } else {
                &test
            };
            job(
                &config,
                root,
                &job_id,
                if timing && config["exploratory"] == true {
                    "exploratory_timing"
                } else if timing {
                    "timing"
                } else {
                    "correctness"
                },
                command,
                args,
                env.clone(),
                inputs.clone(),
                after.clone(),
            )?;
            all.push(job_id);
        }
    }
    let output = timing_length.map_or_else(
        || "oracle-jobs.json".to_owned(),
        |length| format!("timing-jobs-L{length}.json"),
    );
    if let Some(length) = timing_length {
        let expected = candidates()
            .into_iter()
            .filter(|candidate| {
                selected.is_empty() || selected.iter().any(|name| name == candidate.name())
            })
            .map(|candidate| candidate.name().to_owned())
            .collect::<Vec<_>>();
        let advance = advancement_job(&config, root, length, &all, &expected)?;
        all.push(advance);
    }
    json_new_or_identical(&root.join(output), &json!(all))
}

#[derive(Debug, Clone)]
struct StageScore {
    candidate: String,
    candidate_ms_per_dispatch: f64,
    control_drift_passed: bool,
    receipt_path: PathBuf,
    receipt_sha256: String,
    queue_report_path: PathBuf,
    queue_report_sha256: String,
}

fn score_stage_cell(
    root: &Path,
    queue: &Path,
    config: &Value,
    length: u32,
    candidate_name: &str,
) -> Result<StageScore> {
    let candidate = candidates()
        .into_iter()
        .find(|candidate| candidate.name() == candidate_name)
        .ok_or("unknown advancement candidate")?;
    let job_id = id(config, candidate, &format!("abba-L{length}"))?;
    let result = queue.join("results").join(&job_id);
    let queue_report_path = result.join("report.json");
    let queue_report = read(&queue_report_path)?;
    if queue_report["schema"] != "rvllm.experiment_result.v1"
        || queue_report["id"] != job_id
        || queue_report["status"] != "succeeded"
        || queue_report["files_unchanged"] != true
    {
        return Err(format!(
            "stage queue result is not successful and immutable: {candidate_name}"
        )
        .into());
    }
    let receipt_path = result.join("native/abba.json");
    let receipt = read(&receipt_path)?;
    let source = source_path(root, candidate, "core");
    let compile_id = id(config, candidate, "compile-core")?;
    let build_dir = succeeded(queue, &compile_id)?.join("build");
    let library = build_dir.join("kernel.metallib");
    let build_path = build_dir.join("build.json");
    let oracle_id = id(config, candidate, "oracle")?;
    let oracle_path = succeeded(queue, &oracle_id)?.join("native/oracle.json");
    let oracle = read(&oracle_path)?;
    if receipt["schema"] != "rvllm.global-decode.abba.v1"
        || receipt["status"] != "collected"
        || receipt["candidate"] != candidate_name
        || receipt["length"] != length
        || receipt["baseline"] != "attention_decode_f16 (BF16 typed)"
        || receipt["blocks"] != 5
        || receipt["dispatches_per_sample"] != 100
        || receipt["warmups_per_arm"] != 5
        || receipt["source_compiles_during_samples"] != 0
        || receipt["promotion"] != false
        || receipt["identity"]["candidate"] != candidate_name
        || receipt["identity"]["oracle_library"] != false
        || receipt["identity"]["source_sha256"] != hash(&source)?
        || receipt["identity"]["core_sha256"] != hash(&source)?
        || receipt["identity"]["metallib_sha256"] != hash(&library)?
        || receipt["identity"]["build_receipt_sha256"] != hash(&build_path)?
        || receipt["identity"]["test_executable_sha256"] != config["test_executable"]["sha256"]
        || receipt["oracle_receipt_sha256"] != hash(&oracle_path)?
        || oracle["status"] != "passed"
        || oracle["identity"]["candidate"] != candidate_name
        || oracle["identity"]["core_sha256"] != hash(&source)?
        || oracle["identity"]["test_executable_sha256"] != config["test_executable"]["sha256"]
    {
        return Err(format!("stage receipt identity or work mismatch: {candidate_name}").into());
    }
    let tile = candidate
        .global_decode_tile()
        .ok_or("candidate has no global-decode tile")?;
    if receipt["identity"]["rows"] != tile.rows
        || receipt["identity"]["panel"] != tile.panel
        || receipt["identity"]["threads"] != tile.threads
        || receipt["identity"]["grid"] != json!([16 / tile.rows, 1, 1])
    {
        return Err(format!("stage receipt launch geometry mismatch: {candidate_name}").into());
    }
    let samples = receipt["samples"]
        .as_array()
        .ok_or("ABBA samples missing")?;
    if samples.len() != 20 {
        return Err("ABBA receipt must contain exactly 20 samples".into());
    }
    let mut arms = BTreeMap::<&str, usize>::from([("A", 0), ("B", 0)]);
    let mut blocks = BTreeMap::<u64, BTreeMap<&str, usize>>::new();
    let mut candidate_seconds = 0.0;
    for sample in samples {
        let arm = sample["arm"].as_str().ok_or("sample arm missing")?;
        let block = sample["block"].as_u64().ok_or("sample block missing")?;
        let dispatches = sample["dispatches"]
            .as_u64()
            .ok_or("sample dispatches missing")?;
        let seconds = sample["gpu_seconds"]
            .as_f64()
            .ok_or("sample GPU time missing")?;
        if dispatches != 100 || !seconds.is_finite() || seconds <= 0.0 {
            return Err("ABBA sample work or GPU time is invalid".into());
        }
        *arms.get_mut(arm).ok_or("unknown ABBA arm")? += 1;
        *blocks.entry(block).or_default().entry(arm).or_default() += 1;
        if arm == "B" {
            candidate_seconds += seconds;
        }
    }
    if arms != BTreeMap::from([("A", 10), ("B", 10)])
        || blocks.len() != 5
        || blocks
            .values()
            .any(|block| block.get("A") != Some(&2) || block.get("B") != Some(&2))
    {
        return Err("ABBA ordering cells are incomplete or non-equivalent".into());
    }
    Ok(StageScore {
        candidate: candidate_name.to_owned(),
        candidate_ms_per_dispatch: candidate_seconds * 1000.0 / 10.0 / 100.0,
        control_drift_passed: receipt["control_drift_passed"] == true,
        receipt_sha256: hash(&receipt_path)?,
        receipt_path,
        queue_report_sha256: hash(&queue_report_path)?,
        queue_report_path,
    })
}

fn select_survivors(length: u32, scores: &mut [StageScore]) -> Result<Vec<String>> {
    if scores.is_empty() {
        return Err("cannot advance an empty stage".into());
    }
    scores.sort_by(|left, right| {
        left.candidate_ms_per_dispatch
            .total_cmp(&right.candidate_ms_per_dispatch)
            .then_with(|| left.candidate.cmp(&right.candidate))
    });
    if scores.len() == 1 && matches!(length, 256 | 512 | 1024 | 2048) {
        return Ok(vec![scores[0].candidate.clone()]);
    }
    let (anchor, multiplier) = match length {
        256 | 512 | 1024 if scores.len() >= 2 => (scores[1].candidate_ms_per_dispatch, 1.10),
        2048 => (scores[0].candidate_ms_per_dispatch, 1.05),
        _ => return Err("stage length or candidate count cannot satisfy policy".into()),
    };
    Ok(scores
        .iter()
        .take_while(|score| score.candidate_ms_per_dispatch <= anchor * multiplier)
        .map(|score| score.candidate.clone())
        .collect())
}

fn submit_or_verify(queue_runner: &Path, queue: &Path, manifest: &Path) -> Result {
    let expected = read(manifest)?;
    let id = expected["id"].as_str().ok_or("generated job ID missing")?;
    let status = Command::new(queue_runner)
        .arg("submit")
        .arg(queue)
        .arg(manifest)
        .status()?;
    if status.success() {
        return Ok(());
    }
    let queued = queue.join("jobs").join(format!("{id}.json"));
    let attempted = queue.join("results").join(id).join("job.json");
    let (existing, attempted_result) = if queued.is_file() {
        (queued, None)
    } else if attempted.is_file() {
        (
            attempted,
            Some(queue.join("results").join(id).join("report.json")),
        )
    } else {
        return Err(
            format!("queue submission failed without an identical durable job: {id}").into(),
        );
    };
    if read(&existing)? != expected {
        return Err(format!("queue already contains a different job identity: {id}").into());
    }
    if let Some(report_path) = attempted_result {
        let report = read(&report_path)?;
        if report["id"] != id || !matches!(report["status"].as_str(), Some("running" | "succeeded"))
        {
            return Err(format!("queue already contains a failed or invalid attempt: {id}").into());
        }
    }
    Ok(())
}

fn advance(root: &Path, length: u32, output: &Path, expected: &[String]) -> Result {
    if !matches!(length, 256 | 512 | 1024 | 2048 | 4096) || expected.is_empty() {
        return Err("invalid or empty advancement stage".into());
    }
    if expected.iter().collect::<BTreeSet<_>>().len() != expected.len() {
        return Err("duplicate advancement candidate".into());
    }
    let config_path = root.join("campaign.json");
    let config = read(&config_path)?;
    if pin(&std::env::current_exe()?)? != config["job_generator"] {
        return Err("job generator identity drift".into());
    }
    let queue = absolute(config["queue"].as_str().ok_or("queue missing")?)?;
    let queue_runner = absolute(
        config["queue_runner"]["path"]
            .as_str()
            .ok_or("queue runner missing")?,
    )?;
    if pin(&queue_runner)? != config["queue_runner"] {
        return Err("queue runner identity drift".into());
    }
    let mut scores = expected
        .iter()
        .map(|candidate| score_stage_cell(root, &queue, &config, length, candidate))
        .collect::<Result<Vec<_>>>()?;
    let selected = if length == 4096 {
        Vec::new()
    } else {
        select_survivors(length, &mut scores)?
    };
    let next_length = match length {
        256 => Some(512),
        512 => Some(1024),
        1024 => Some(2048),
        2048 => Some(4096),
        4096 => None,
        _ => unreachable!(),
    };
    let score_json = scores
        .iter()
        .map(|score| {
            json!({
                "candidate":score.candidate,
                "candidate_mean_ms_per_dispatch":score.candidate_ms_per_dispatch,
                "control_drift_passed":score.control_drift_passed,
                "native_receipt":{"path":score.receipt_path,"sha256":score.receipt_sha256},
                "queue_report":{"path":score.queue_report_path,"sha256":score.queue_report_sha256}
            })
        })
        .collect::<Vec<_>>();
    let receipt = json!({
        "schema":"rvllm.global-decode.advancement.v1",
        "campaign":config["campaign"],
        "campaign_sha256":hash(&config_path)?,
        "completed_length":length,
        "next_length":next_length,
        "expected_candidates":expected,
        "scores":score_json,
        "selected_candidates":selected,
        "rule":if length == 2048 {"fastest plus candidates within 5 percent"}
            else if length == 4096 {"terminal evidence only; no automatic promotion"}
            else {"fastest two plus candidates within 10 percent of second-fastest"},
        "promotion":false
    });
    if !output.is_absolute() || !output.is_dir() {
        return Err(
            "advancement output must be the existing absolute queue result directory".into(),
        );
    }
    json_new_or_identical(&output.join("advancement.json"), &receipt)?;
    let Some(next_length) = next_length else {
        return Ok(());
    };
    generate(root, true, Some(next_length), &selected)?;
    let list = read(&root.join(format!("timing-jobs-L{next_length}.json")))?;
    for job_id in list.as_array().ok_or("generated timing job list missing")? {
        let job_id = job_id.as_str().ok_or("generated timing job ID missing")?;
        submit_or_verify(
            &queue_runner,
            &queue,
            &root.join("jobs").join(format!("{job_id}.json")),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_directory(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "rvllm-global-decode-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        directory
    }

    fn score(candidate: &str, milliseconds: f64) -> StageScore {
        StageScore {
            candidate: candidate.into(),
            candidate_ms_per_dispatch: milliseconds,
            control_drift_passed: true,
            receipt_path: PathBuf::from("receipt"),
            receipt_sha256: "a".repeat(64),
            queue_report_path: PathBuf::from("report"),
            queue_report_sha256: "b".repeat(64),
        }
    }

    #[test]
    fn timing_request_accepts_screen_and_explicit_survivors() {
        validate_timing_request(256, &[]).unwrap();
        validate_timing_request(
            512,
            &[
                "metal-global-d512-r16p128t128".to_owned(),
                "metal-global-d512-r16p64t128".to_owned(),
            ],
        )
        .unwrap();
    }

    #[test]
    fn timing_request_rejects_unknown_length_candidate_and_duplicate() {
        assert!(validate_timing_request(128, &[]).is_err());
        assert!(validate_timing_request(512, &["not-a-candidate".to_owned()]).is_err());
        let duplicate = "metal-global-d512-r16p128t128".to_owned();
        assert!(validate_timing_request(512, &[duplicate.clone(), duplicate]).is_err());
    }

    #[test]
    fn selected_oracle_candidates_are_strictly_validated() {
        validate_selected(&["metal-global-d512-r1p128t32".to_owned()]).unwrap();
        assert!(validate_selected(&["not-a-candidate".to_owned()]).is_err());
        let duplicate = "metal-global-d512-r1p128t32".to_owned();
        assert!(validate_selected(&[duplicate.clone(), duplicate]).is_err());
    }

    #[test]
    fn successive_halving_uses_absolute_candidate_time_and_widens_ties() {
        let mut screen = vec![
            score("slow", 12.0),
            score("fast", 10.0),
            score("near_second", 11.0),
            score("second", 10.5),
        ];
        assert_eq!(
            select_survivors(256, &mut screen).unwrap(),
            ["fast", "second", "near_second"]
        );
        let mut finalists = vec![
            score("outside", 10.51),
            score("winner", 10.0),
            score("tie", 10.5),
        ];
        assert_eq!(
            select_survivors(2048, &mut finalists).unwrap(),
            ["winner", "tie"]
        );
        let mut control = vec![score("control", 3.5)];
        assert_eq!(select_survivors(256, &mut control).unwrap(), ["control"]);
    }

    #[test]
    fn immutable_json_recovery_accepts_only_identical_content() {
        let directory = temp_directory("immutable");
        let path = directory.join("receipt.json");
        let value = json!({"selected":["a","b"]});
        json_new_or_identical(&path, &value).unwrap();
        json_new_or_identical(&path, &value).unwrap();
        assert!(json_new_or_identical(&path, &json!({"selected":["b"]})).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn stage_scoring_requires_sealed_identity_and_exact_work() {
        let directory = temp_directory("stage");
        let root = directory.join("campaign");
        let queue = directory.join("queue");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir_all(queue.join("results")).unwrap();
        let candidate = candidates()[0];
        let campaign = "synthetic";
        let config = json!({
            "campaign":campaign,
            "test_executable":{"sha256":"test-pin"}
        });
        let source = source_path(&root, candidate, "core");
        std::fs::write(&source, b"sealed source").unwrap();
        let compile_id = id(&config, candidate, "compile-core").unwrap();
        let build_dir = queue.join("results").join(&compile_id).join("build");
        std::fs::create_dir_all(&build_dir).unwrap();
        let library = build_dir.join("kernel.metallib");
        let build = build_dir.join("build.json");
        std::fs::write(&library, b"sealed library").unwrap();
        std::fs::write(&build, b"sealed build receipt").unwrap();
        json_new(
            &queue.join("results").join(&compile_id).join("report.json"),
            &json!({"status":"succeeded","files_unchanged":true}),
        )
        .unwrap();
        let oracle_id = id(&config, candidate, "oracle").unwrap();
        let oracle_dir = queue.join("results").join(&oracle_id);
        std::fs::create_dir_all(oracle_dir.join("native")).unwrap();
        let oracle = oracle_dir.join("native/oracle.json");
        json_new(
            &oracle,
            &json!({"status":"passed","identity":{"candidate":candidate.name(),
                "core_sha256":hash(&source).unwrap(),"test_executable_sha256":"test-pin"}}),
        )
        .unwrap();
        json_new(
            &oracle_dir.join("report.json"),
            &json!({"status":"succeeded","files_unchanged":true}),
        )
        .unwrap();
        let timing_id = id(&config, candidate, "abba-L256").unwrap();
        let timing_dir = queue.join("results").join(timing_id);
        std::fs::create_dir_all(timing_dir.join("native")).unwrap();
        let tile = candidate.global_decode_tile().unwrap();
        let mut samples = Vec::new();
        for block in 0..5 {
            for arm in ["A", "B", "B", "A"] {
                samples.push(json!({"arm":arm,"block":block,"dispatches":100,
                    "gpu_seconds":if arm == "A" {2.0} else {1.0}}));
            }
        }
        let receipt_path = timing_dir.join("native/abba.json");
        let receipt = json!({"schema":"rvllm.global-decode.abba.v1","status":"collected",
            "baseline":"attention_decode_f16 (BF16 typed)","blocks":5,
            "candidate":candidate.name(),"length":256,"dispatches_per_sample":100,
            "warmups_per_arm":5,"source_compiles_during_samples":0,"promotion":false,
            "control_drift_passed":false,"oracle_receipt_sha256":hash(&oracle).unwrap(),
            "identity":{"candidate":candidate.name(),"oracle_library":false,
                "source_sha256":hash(&source).unwrap(),"core_sha256":hash(&source).unwrap(),
                "metallib_sha256":hash(&library).unwrap(),"build_receipt_sha256":hash(&build).unwrap(),
                "test_executable_sha256":"test-pin","rows":tile.rows,"panel":tile.panel,
                "threads":tile.threads,"grid":[16 / tile.rows,1,1]},"samples":samples});
        json_new(&receipt_path, &receipt).unwrap();
        json_new(
            &timing_dir.join("report.json"),
            &json!({"schema":"rvllm.experiment_result.v1","status":"succeeded",
                "id":timing_dir.file_name().unwrap().to_str().unwrap(),"files_unchanged":true}),
        )
        .unwrap();
        let score = score_stage_cell(&root, &queue, &config, 256, candidate.name()).unwrap();
        assert_eq!(score.candidate_ms_per_dispatch, 10.0);
        let mut changed = receipt;
        changed["samples"][0]["dispatches"] = json!(99);
        std::fs::write(&receipt_path, serde_json::to_vec_pretty(&changed).unwrap()).unwrap();
        assert!(score_stage_cell(&root, &queue, &config, 256, candidate.name()).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
fn main() -> Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [action,file] if action=="test-exe" => {
            let text = std::fs::read_to_string(absolute(file)?)?;
            let mut found = Vec::new();
            for line in text.lines() {
                let value: Value = serde_json::from_str(line)?;
                if value["reason"]=="compiler-artifact" && value["target"]["name"]=="rvllm_apple_metal"
                    && value["profile"]["test"]==true {
                    if let Some(path) = value["executable"].as_str() { found.push(path.to_owned()); }
                }
            }
            found.sort(); found.dedup();
            if found.len()!=1 { return Err("expected one exact crate test executable in cargo JSON".into()); }
            println!("{}",found[0]); Ok(())
        }
        [action,source,directory] if action=="compile" => compile(&absolute(source)?,&absolute(directory)?),
        [action,campaign,root,queue,test,conditions] if action=="prepare" =>
            prepare(campaign,&absolute(root)?,&absolute(queue)?,&absolute(test)?,&absolute(conditions)?),
        [action,root] if action=="oracle-jobs" => generate(&absolute(root)?,false,None,&[]),
        [action,root,selected @ ..] if action=="oracle-jobs" && !selected.is_empty() =>
            generate(&absolute(root)?,false,None,selected),
        [action,root] if action=="timing-jobs" => generate(&absolute(root)?,true,None,&[]),
        [action,root,length,selected @ ..] if action=="timing-jobs" && !selected.is_empty() => {
            let length = length.parse()?;
            generate(&absolute(root)?,true,Some(length),selected)
        }
        [action,root,length,output,expected @ ..] if action=="advance" && !expected.is_empty() =>
            advance(&absolute(root)?,length.parse()?,&absolute(output)?,expected),
        _=>Err("usage: rvllm-global-decode-jobs test-exe CARGO_JSON | prepare ID ROOT QUEUE TEST_EXE CONDITIONS_JSON | compile SOURCE FRESH_OUTPUT_DIR | oracle-jobs ROOT [CANDIDATE...] | timing-jobs ROOT [LENGTH CANDIDATE...] | advance ROOT LENGTH OUTPUT EXPECTED_CANDIDATE...".into()),
    }
}
