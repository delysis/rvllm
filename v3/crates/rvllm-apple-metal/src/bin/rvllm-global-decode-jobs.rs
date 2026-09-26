//! Offline, explicit queue preparation. No accelerator execution, implicit
//! submission, shell, environment-based dispatch, winner selection or promotion.
#![forbid(unsafe_code)]
use rvllm_apple_metal::decode_round_campaign as round_two;
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
/// Snapshot a build output before publishing a job that depends on its bytes.
/// Content-addressed names let independent campaigns share identical binaries
/// without ever replacing a file that an older manifest has pinned.
fn snapshot_executable(source: &Path, queue: &Path, role: &str) -> Result<PathBuf> {
    let source_hash = hash(source)?;
    let directory = queue.join("executables");
    std::fs::create_dir_all(&directory)?;
    let destination = directory.join(format!("{source_hash}-{role}"));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
    {
        Ok(mut output) => {
            let mut input = std::fs::File::open(source)?;
            std::io::copy(&mut input, &mut output)?;
            output.sync_all()?;
            std::fs::set_permissions(&destination, std::fs::metadata(source)?.permissions())?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    if hash(source)? != source_hash || hash(&destination)? != source_hash {
        return Err(format!(
            "executable snapshot changed or is incomplete: {}",
            destination.display()
        )
        .into());
    }
    Ok(destination)
}
fn candidates() -> Vec<MetalResearchCandidate> {
    rvllm_apple_metal::research_catalog::ALL_CANDIDATES
        .iter()
        .copied()
        .filter(|c| {
            c.global_decode_tile().is_some()
                || c.split_global_decode_tile().is_some()
                || c.decode_round_operator()
        })
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
    if length == 0 {
        if selected.is_empty()
            || selected.iter().any(|name| {
                name.parse::<MetalResearchCandidate>()
                    .map_or(true, |c| !c.decode_round_operator())
            })
        {
            return Err("projection cells require explicit operator selectors and LENGTH=0".into());
        }
        return validate_selected(selected);
    }
    if !matches!(length, 256 | 512 | 1024 | 2048 | 4096) {
        return Err("timing length must be 256, 512, 1024, 2048, or 4096".into());
    }
    validate_selected(selected)?;
    if selected.iter().any(|name| {
        name.parse::<MetalResearchCandidate>().is_ok_and(|c| {
            c.decode_round_operator()
                || (c.global_capacity_tokens() != 0 && length > c.global_capacity_tokens())
        })
    }) {
        return Err("requested length is outside the selected candidate contract".into());
    }
    Ok(())
}

fn operator_oracle_identity_matches(candidate: MetalResearchCandidate, identity: &Value) -> bool {
    if !candidate.decode_round_operator() {
        return false;
    }
    let kernel = candidate.kernels()[0];
    let Some(launch) = rvllm_apple_metal::research_decode::operator_launch(candidate) else {
        return false;
    };
    let rows = launch.rows_per_group;
    let output_rows = if candidate.qmv_w4() || candidate.qmv_w8() {
        3840
    } else {
        15360
    };
    if launch.kernel != kernel || rows == 0 || output_rows % rows != 0 {
        return false;
    }
    let threads = launch.threads;
    let grid = output_rows / rows;
    identity["rows"] == rows
        && identity["keys"] == 1
        && identity["panel"] == 32
        && identity["threads"] == threads
        && identity["grid"] == json!([grid, 1, 1])
        && identity["kernel"] == kernel.name()
        && identity["kernels"][0]["threads"] == threads
}

fn split_streaming_fp64_evidence(oracle: &Value) -> bool {
    oracle["streaming_fp32_max_abs_bound"] == 5.0e-5
        && oracle["streaming_fp64_max_abs_bound"] == 5.0e-4
        && oracle["streaming_fp64_relative_l2_bound"] == 1.0e-4
        && oracle["cases"].as_array().is_some_and(|cases| {
            !cases.is_empty()
                && cases.iter().all(|case| {
                    case["independent_cpu_reference"] == "scalar FP64"
                        && case["once_rounded_bf16"] == true
                        && case["repeatable"] == true
                        && case["guard_bytes_preserved"] == true
                        && case["max_fp64_abs_error"]
                            .as_f64()
                            .is_some_and(|error| error.is_finite() && error <= 5.0e-4)
                        && case["relative_l2_error"]
                            .as_f64()
                            .is_some_and(|error| error.is_finite() && error <= 1.0e-4)
                })
        })
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
        "air_sha256":hash(&air)?,"air_path":air,
        "flags":["-std=metal3.1","-fno-fast-math"],"compile_argv":compile_args,
        "link_argv":link_args,"tool_pins":compiler_pins,"native_execution":false}),
    )
}
fn id(config: &Value, candidate: MetalResearchCandidate, suffix: &str) -> Result<String> {
    let campaign = config["campaign"].as_str().ok_or("campaign missing")?;
    if candidate.round_two() {
        let value = format!(
            "{campaign}-{}-{suffix}",
            candidate.name().trim_start_matches("metal-")
        );
        if value.len() > 96 {
            return Err("queue ID exceeds 96 bytes".into());
        }
        return Ok(value);
    }
    if let Some(tile) = candidate.global_decode_tile() {
        if tile.keys != 8 || tile.per_tile_softmax {
            return Ok(format!(
                "{campaign}-r{}k{}p{}t{}{}-{suffix}",
                tile.rows,
                tile.keys,
                tile.panel,
                tile.threads,
                if tile.simd_matrix {
                    "-mma"
                } else if tile.per_tile_softmax {
                    "-tile"
                } else {
                    "-key"
                }
            ));
        }
        Ok(format!(
            "{campaign}-r{}p{}t{}-{suffix}",
            tile.rows, tile.panel, tile.threads
        ))
    } else if candidate.decode_round_operator() {
        Ok(format!(
            "{campaign}-{}-{suffix}",
            candidate.name().trim_start_matches("metal-")
        ))
    } else {
        let tile = candidate
            .split_global_decode_tile()
            .ok_or("candidate has no global decode identity")?;
        Ok(if tile.simd_matrix {
            format!(
                "{campaign}-split-r{}k{}s{}t{}-mma-{suffix}",
                tile.rows, tile.keys, tile.partition, tile.threads
            )
        } else {
            format!(
                "{campaign}-split-r{}s{}t{}-{suffix}",
                tile.rows, tile.partition, tile.threads
            )
        })
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
    let mut inputs = inputs;
    if config["screen_protocol"] == round_two::PROTOCOL {
        validate_observation_policy(&config["conditions"])?;
        if !config["confirmation_of"].is_null() {
            inputs.push(config["confirmation_of"].clone());
        }
    }
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

fn operator_advancement_id(config: &Value, stage: &str) -> Result<String> {
    if !matches!(stage, "compile" | "oracle") {
        return Err("unknown operator advancement stage".into());
    }
    Ok(format!(
        "{}-operator-advance-{stage}",
        config["campaign"].as_str().ok_or("campaign missing")?
    ))
}

fn operator_advancement_job(
    config: &Value,
    root: &Path,
    stage: &str,
    after: Vec<String>,
    names: &[String],
) -> Result<String> {
    let job_id = operator_advancement_id(config, stage)?;
    let generator = absolute(
        config["job_generator"]["path"]
            .as_str()
            .ok_or("job generator missing")?,
    )?;
    let mut args = vec![
        "operator-advance".into(),
        root.to_string_lossy().into_owned(),
        stage.into(),
        "{output}".into(),
    ];
    args.extend_from_slice(names);
    job(
        config,
        root,
        &job_id,
        "preparation",
        &generator,
        args,
        json!({}),
        vec![pin(&root.join("campaign.json"))?],
        after,
    )?;
    Ok(job_id)
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
#[allow(clippy::too_many_arguments)]
fn prepare(
    campaign: &str,
    root: &Path,
    queue: &Path,
    test: &Path,
    conditions: &Path,
    selected: &[String],
    round_two_mode: bool,
    confirmation: Option<&Path>,
) -> Result {
    validate_selected(selected)?;
    if !round_two_mode
        && selected.iter().any(|name| {
            name.parse::<MetalResearchCandidate>()
                .is_ok_and(MetalResearchCandidate::round_two)
        })
    {
        return Err("new candidates require prepare-round2".into());
    }
    if round_two_mode
        && (selected.is_empty()
            || selected.iter().any(|name| {
                name.parse::<MetalResearchCandidate>()
                    .map_or(true, |c| !round_two::supported(c))
            }))
    {
        return Err("prepare-round2 requires explicit supported candidate names".into());
    }

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
    // host state. Round-two confirmations keep the same observation-only policy.
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
    if round_two_mode {
        validate_observation_policy(&policy)?;
    }
    let confirmation_pin = if let Some(prior_root) = confirmation {
        let prior_path = prior_root.join("campaign.json");
        let prior = read(&prior_path)?;
        if !round_two_mode
            || prior["screen_protocol"] != round_two::PROTOCOL
            || prior["campaign"] == campaign
            || prior["test_executable"]["sha256"] != hash(test)?
            || prior["job_generator"]["sha256"] != hash(&std::env::current_exe()?)?
            || selected.iter().any(|name| {
                !prior["candidate_names"]
                    .as_array()
                    .is_some_and(|names| names.iter().any(|n| n == name.as_str()))
            })
        {
            return Err("confirmation requires a distinct ID, same frozen binaries and prior candidate names".into());
        }
        Some(pin(&prior_path)?)
    } else {
        None
    };
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
    let generator_snapshot = snapshot_executable(&executable, queue, "generator")?;
    let test_snapshot = snapshot_executable(test, queue, "test")?;
    let runner_snapshot = snapshot_executable(&queue_runner, queue, "submitter")?;
    let retainer_snapshot = snapshot_executable(&retainer, queue, "retainer")?;
    let exploratory = policy["low_power_mode"].is_null()
        || policy["pmset_power_mode"].is_null()
        || policy["thermal_state"].is_null();
    // Legacy implicit campaigns stay unchanged. New arms must be named.
    let family = candidates()
        .into_iter()
        .filter(|c| {
            if selected.is_empty() {
                !c.explicit_storage_abi()
            } else {
                selected.iter().any(|name| name == c.name())
            }
        })
        .collect::<Vec<_>>();
    let operator_family = !family.is_empty() && family.iter().all(|c| c.decode_round_operator());
    let names = family.iter().map(|c| c.name()).collect::<Vec<_>>();
    let config = json!({"schema":"rvllm.global-decode.campaign.v1","campaign":campaign,
        "candidate_names":names,
        "screen_protocol":if round_two_mode { Some(round_two::PROTOCOL) } else { None },
        "confirmation_of":confirmation_pin,
        "conditions_are_observations_only":round_two_mode,
        "source_base_commit":if round_two_mode { Some("593e1d6fb25088608198f2248dcac038d19fe761") } else { None },
        "queue":queue,"test_executable":pin(&test_snapshot)?,"job_generator":pin(&generator_snapshot)?,
        "queue_runner":pin(&runner_snapshot)?,
        "abba_retainer":pin(&retainer_snapshot)?,"exploratory":exploratory,
        "conditions":policy,"conditions_input":pin(conditions)?,
        "screen_length":256,"advancement_lengths":[512,1024,2048],
        "deferred_confirmation_lengths":if round_two_mode { json!([]) } else { json!([4096]) },
        "split_kv":round_two_mode && family.iter().any(|candidate| candidate.split_global_decode_tile().is_some()),
        "local_prefill":false,
        "promotion":false,"status":"proposed_unqualified"});
    json_new(&root.join("campaign.json"), &config)?;
    if round_two_mode {
        let mut arms = Vec::new();
        for &candidate in &family {
            arms.push(json!({"candidate":candidate.name(),
                "operator_keys":candidate.operator_keys(),
                "capacity_tokens":candidate.global_capacity_tokens(),
                "geometry":expected_geometry(candidate)?,
                "kernels":candidate.kernels().iter().map(|k| json!({
                    "name":k.name(),"threads":k.limits().0,
                    "source_threadgroup_bytes":k.limits().1})).collect::<Vec<_>>(),
                "status":"unqualified"}));
        }
        json_new(
            &root.join("round2-plan.json"),
            &json!({"schema":round_two::PROTOCOL,
            "campaign":campaign,"arms":arms,"initial_attention_length":256,
            "advancement_lengths":[512,1024,2048],"blocks":10,"samples":40,
            "control_drift_limit":0.05,"automatic_submission":false,
            "automatic_retry":false,"sample_pruning":false,"promotion":false}),
        )?;
    }

    let mut all = Vec::new();
    let metal = tool("metal")?;
    let linker = tool("metallib")?;
    for &candidate in &family {
        let core = rvllm_apple_metal::kernels::kernel_source_with_options(
            MetalFloatType::Bf16,
            MetalKernelOptions {
                research: candidate,
                ..MetalKernelOptions::default()
            },
        );
        let flavors: &[&str] = if candidate.decode_round_operator()
            || candidate
                .split_global_decode_tile()
                .is_some_and(|t| t.streaming())
        {
            &["core"]
        } else {
            &["core", "oracle"]
        };
        for &flavor in flavors {
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
                &generator_snapshot,
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
    if operator_family {
        let operator_names = names
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let oracle_ids = family
            .iter()
            .map(|&candidate| id(&config, candidate, "oracle"))
            .collect::<Result<Vec<_>>>()?;
        operator_advancement_job(&config, root, "oracle", oracle_ids, &operator_names)?;
        let compile_advance =
            operator_advancement_job(&config, root, "compile", all.clone(), &operator_names)?;
        all.push(compile_advance);
    }
    json_new(&root.join("compile-jobs.json"), &json!(all))
}
fn generate(root: &Path, timing: bool, length: Option<u32>, selected: &[String]) -> Result {
    let config_path = root.join("campaign.json");
    let config = read(&config_path)?;
    let second_round = config["screen_protocol"] == round_two::PROTOCOL;
    if second_round {
        validate_observation_policy(&config["conditions"])?;
    }

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
        if selected.is_empty()
            && config["candidate_names"].as_array().is_some_and(|names| {
                names.iter().any(|name| {
                    name.as_str()
                        .and_then(|s| s.parse::<MetalResearchCandidate>().ok())
                        .is_some_and(MetalResearchCandidate::explicit_storage_abi)
                })
            })
        {
            return Err("new-round timing requires explicit candidate selection".into());
        }
        validate_timing_request(length, selected)?;
    }
    if let Some(family) = config["candidate_names"].as_array() {
        if selected
            .iter()
            .any(|name| !family.iter().any(|item| item == name.as_str()))
        {
            return Err("candidate was not prepared in this immutable campaign".into());
        }
    }
    let mut all = Vec::new();
    // Never prune the family from partial results. A failed compile/oracle stops
    // generation; revised families require a new explicit campaign identity.
    for candidate in candidates() {
        if config["candidate_names"]
            .as_array()
            .is_some_and(|names| !names.iter().any(|name| name == candidate.name()))
        {
            continue;
        }
        if !selected.is_empty() && !selected.iter().any(|name| name == candidate.name()) {
            continue;
        }
        if candidate.round_two() && !second_round {
            return Err("new candidate cannot bypass the second-round campaign protocol".into());
        }
        let prerequisite = if second_round && timing {
            Some(admit_round_two(
                root,
                candidate,
                timing_length.ok_or("length missing")?,
            )?)
        } else {
            None
        };
        // The split oracle validates the production partial+merge entry points
        // directly and therefore needs the exact core source.  Only the
        // single-pass oracle uses the appended diagnostic entry point.
        let flavor = if timing
            || candidate.split_global_decode_tile().is_some()
            || candidate.decode_round_operator()
        {
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
        if second_round {
            inputs.push(pin(&root.join("round2-plan.json"))?);
            if let Some(prerequisite) = &prerequisite {
                inputs.extend(prerequisite.iter().cloned());
            }
        }
        let mut after = vec![compile_id];
        let mut env = json!({});
        env[format!("{PREFIX}CANDIDATE")] = json!(candidate.name());
        env[format!("{PREFIX}SOURCE")] = json!(source);
        env[format!("{PREFIX}METALLIB")] = json!(library);
        env[format!("{PREFIX}BUILD_RECEIPT")] = json!(build_path);
        let split_matrix = candidate
            .split_global_decode_tile()
            .is_some_and(|tile| tile.simd_matrix);
        if timing {
            inputs.push(pin(&test)?);
            let oracle_id = id(&config, candidate, "oracle")?;
            let oracle_path = succeeded(&queue, &oracle_id)?.join(if split_matrix {
                "native/split-matrix-oracle.json"
            } else if candidate.split_global_decode_tile().is_some() {
                "native/split-oracle.json"
            } else if candidate
                .global_decode_tile()
                .is_some_and(|tile| tile.simd_matrix)
            {
                "native/matrix-oracle.json"
            } else {
                "native/oracle.json"
            });
            let oracle = read(&oracle_path)?;
            let matrix = candidate
                .global_decode_tile()
                .is_some_and(|tile| tile.simd_matrix)
                || split_matrix;
            let expected_oracle_schema = if candidate.decode_round_operator() {
                "rvllm.decode-round.oracle.v1"
            } else if split_matrix {
                "rvllm.global-decode.split-matrix-oracle.v1"
            } else if candidate.split_global_decode_tile().is_some() {
                "rvllm.global-decode.split-oracle.v1"
            } else if matrix {
                "rvllm.global-decode.matrix-oracle.v1"
            } else {
                "rvllm.global-decode.oracle.v1"
            };
            if oracle["schema"] != expected_oracle_schema
                || oracle["status"] != "passed"
                || oracle["identity"]["candidate"] != candidate.name()
                || oracle["identity"]["core_sha256"] != hash(&source)?
                || oracle["identity"]["test_executable_sha256"]
                    != config["test_executable"]["sha256"]
                || (candidate.decode_round_operator()
                    && !operator_oracle_identity_matches(candidate, &oracle["identity"]))
                || ((matrix
                    || candidate
                        .global_decode_tile()
                        .is_some_and(|t| t.streaming()))
                    && (oracle["numerical_contract"]
                        != "independent-fp64-absolute-and-relative-l2-plus-exact-once-rounded-bf16"
                        || oracle["fp64_max_abs_bound"] != 5.0e-4
                        || oracle["fp64_relative_l2_bound"] != 1.0e-4))
                || (candidate
                    .split_global_decode_tile()
                    .is_some_and(|tile| tile.streaming())
                    && !split_streaming_fp64_evidence(&oracle))
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
        let cells = if timing && candidate.decode_round_operator() {
            candidate
                .operator_keys()
                .iter()
                .map(|&k| (timing_length, Some(k as u32)))
                .collect::<Vec<_>>()
        } else {
            vec![(timing_length, None)]
        };
        for (length, operator_k) in cells {
            let suffix = if let Some(k) = operator_k {
                format!("abba-L0-K{k}")
            } else {
                length.map_or("oracle".into(), |n| format!("abba-L{n}"))
            };
            if let Some(k) = operator_k {
                env[format!("{PREFIX}OPERATOR_K")] = json!(k.to_string());
            }
            let job_id = id(&config, candidate, &suffix)?;
            // Pin the exclusive result directory directly. The queue also expands
            // {output} in argv/env, but this identity does not depend on expansion.
            env[format!("{PREFIX}REPORT_DIR")] =
                json!(queue.join("results").join(&job_id).join("native"));
            let test_name = if candidate.decode_round_operator() {
                if timing {
                    "decode_round_abba"
                } else {
                    "decode_round_oracle"
                }
            } else if timing
                && candidate
                    .split_global_decode_tile()
                    .is_some_and(|t| t.streaming())
            {
                "global_decode_split_stream_abba"
            } else if timing && candidate.split_global_decode_tile().is_some() {
                "global_decode_split_abba_v2"
            } else if timing {
                "global_decode_abba"
            } else if split_matrix {
                "global_decode_split_matrix_device_oracle"
            } else if candidate.split_global_decode_tile().is_some() {
                "global_decode_split_device_oracle"
            } else if candidate
                .global_decode_tile()
                .is_some_and(|tile| tile.simd_matrix)
            {
                "global_decode_matrix_device_oracle"
            } else {
                "global_decode_device_oracle"
            };
            if let Some(n) = length {
                env[format!("{PREFIX}LENGTH")] = json!(n.to_string());
            }
            let mut args = vec![
                "--ignored".into(),
                "--exact".into(),
                format!(
                    "{}::{test_name}",
                    if candidate.decode_round_operator() {
                        "research_decode_device_tests"
                    } else {
                        "attention_global_decode_device_tests"
                    }
                ),
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
                (config["candidate_names"].as_array().map_or(true, |names| {
                    names.iter().any(|name| name == candidate.name())
                })) && (selected.is_empty() || selected.iter().any(|name| name == candidate.name()))
            })
            .map(|candidate| candidate.name().to_owned())
            .collect::<Vec<_>>();
        // The additive round has no automatic advancement or promotion path.
        // L512 is a separate explicit request after reviewing the L256 screen.
        let next_round = expected.iter().any(|name| {
            name.parse::<MetalResearchCandidate>()
                .is_ok_and(MetalResearchCandidate::explicit_storage_abi)
        });
        if !next_round {
            let advance = advancement_job(&config, root, length, &all, &expected)?;
            all.push(advance);
        }
    }
    json_new_or_identical(&root.join(output), &json!(all))
}

#[derive(Debug, Clone)]
struct StageScore {
    candidate: String,
    candidate_ms_per_dispatch: f64,
    partial_ms_per_operation: Option<f64>,
    merge_ms_per_operation: Option<f64>,
    control_drift_passed: bool,
    receipt_path: PathBuf,
    receipt_sha256: String,
    queue_report_path: PathBuf,
    queue_report_sha256: String,
}

fn sample_work_and_gpu_seconds(sample: &Value, split: bool, arm: &str) -> Result<(u64, f64)> {
    if !split {
        return Ok((
            sample["dispatches"]
                .as_u64()
                .ok_or("sample dispatches missing")?,
            sample["gpu_seconds"]
                .as_f64()
                .ok_or("sample GPU time missing")?,
        ));
    }
    let operations = sample["operations"]
        .as_u64()
        .ok_or("sample operations missing")?;
    let partial = sample["partial_gpu_seconds"]
        .as_f64()
        .ok_or("sample partial GPU time missing")?;
    let merge = sample["merge_gpu_seconds"]
        .as_f64()
        .ok_or("sample merge GPU time missing")?;
    let total = sample["total_gpu_seconds"]
        .as_f64()
        .ok_or("sample total GPU time missing")?;
    let candidate_arm = arm == "B";
    let component_sum = partial + merge;
    let sum_tolerance = 8.0 * f64::EPSILON * total.abs().max(component_sum.abs()).max(1.0);
    if sample["partial_dispatches"].as_u64() != Some(if candidate_arm { 100 } else { 0 })
        || sample["merge_dispatches"].as_u64() != Some(if candidate_arm { 100 } else { 0 })
        || sample["baseline_dispatches"].as_u64() != Some(if candidate_arm { 0 } else { 100 })
        || (candidate_arm && (total - component_sum).abs() > sum_tolerance)
        || (candidate_arm && (partial <= 0.0 || merge <= 0.0))
        || (!candidate_arm && (partial != 0.0 || merge != 0.0))
    {
        return Err("split ABBA component work or total is inconsistent".into());
    }
    Ok((operations, total))
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
    let split = candidate.split_global_decode_tile();
    let split_matrix = split.is_some_and(|tile| tile.simd_matrix);
    let matrix = candidate
        .global_decode_tile()
        .is_some_and(|tile| tile.simd_matrix)
        || split_matrix;
    let oracle_path = succeeded(queue, &oracle_id)?.join(if split_matrix {
        "native/split-matrix-oracle.json"
    } else if split.is_some() {
        "native/split-oracle.json"
    } else if matrix {
        "native/matrix-oracle.json"
    } else {
        "native/oracle.json"
    });
    let oracle = read(&oracle_path)?;
    let expected_schema = if split.is_some() {
        "rvllm.global-decode.abba.v2"
    } else {
        "rvllm.global-decode.abba.v1"
    };
    let expected_oracle_schema = if split_matrix {
        "rvllm.global-decode.split-matrix-oracle.v1"
    } else if split.is_some() {
        "rvllm.global-decode.split-oracle.v1"
    } else if matrix {
        "rvllm.global-decode.matrix-oracle.v1"
    } else {
        "rvllm.global-decode.oracle.v1"
    };
    if receipt["schema"] != expected_schema
        || receipt["status"] != "collected"
        || receipt["candidate"] != candidate_name
        || receipt["length"] != length
        || receipt["baseline"] != "attention_decode_f16 (BF16 typed)"
        || receipt["blocks"] != 5
        || (split.is_none() && receipt["dispatches_per_sample"] != 100)
        || (split.is_some()
            && (receipt["operations_per_sample"] != 100
                || receipt["candidate_dispatches_per_operation"] != 2
                || receipt["timing_metric"]
                    != "total_gpu_seconds = partial_gpu_seconds + merge_gpu_seconds"
                || receipt["conditions_are_observations_only"] != true
                || receipt["correctness_prerequisite"] != "passed"))
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
        || oracle["schema"] != expected_oracle_schema
        || oracle["status"] != "passed"
        || (matrix
            && (oracle["numerical_contract"]
                != "independent-fp64-absolute-and-relative-l2-plus-exact-once-rounded-bf16"
                || oracle["fp64_max_abs_bound"] != 5.0e-4
                || oracle["fp64_relative_l2_bound"] != 1.0e-4))
        || (split.is_some() && oracle["identity"] != receipt["identity"])
        || oracle["identity"]["candidate"] != candidate_name
        || oracle["identity"]["core_sha256"] != hash(&source)?
        || oracle["identity"]["test_executable_sha256"] != config["test_executable"]["sha256"]
    {
        return Err(format!("stage receipt identity or work mismatch: {candidate_name}").into());
    }
    if let Some(tile) = candidate.global_decode_tile() {
        if receipt["identity"]["rows"] != tile.rows
            || receipt["identity"]["panel"] != tile.panel
            || receipt["identity"]["threads"] != tile.threads
            || receipt["identity"]["grid"] != json!([16 / tile.rows, 1, 1])
        {
            return Err(format!("stage receipt launch geometry mismatch: {candidate_name}").into());
        }
    } else if let Some(tile) = split {
        if receipt["identity"]["rows"] != tile.rows
            || receipt["identity"]["keys"] != tile.keys
            || receipt["identity"]["panel"] != tile.panel
            || receipt["identity"]["threads"] != tile.threads
            || receipt["identity"]["simd_matrix"] != tile.simd_matrix
            || receipt["identity"]["grid"] != json!([2, 16, 1])
            || receipt["identity"]["scratch_bytes"] != 16 * 16 * 514 * 4
            || receipt["identity"]["kernels"].as_array().map(Vec::len) != Some(2)
        {
            return Err(
                format!("split stage receipt launch geometry mismatch: {candidate_name}").into(),
            );
        }
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
    let mut partial_seconds = 0.0;
    let mut merge_seconds = 0.0;
    for sample in samples {
        let arm = sample["arm"].as_str().ok_or("sample arm missing")?;
        let block = sample["block"].as_u64().ok_or("sample block missing")?;
        let (work, seconds) = sample_work_and_gpu_seconds(sample, split.is_some(), arm)?;
        if work != 100 || !seconds.is_finite() || seconds <= 0.0 {
            return Err("ABBA sample work or GPU time is invalid".into());
        }
        *arms.get_mut(arm).ok_or("unknown ABBA arm")? += 1;
        *blocks.entry(block).or_default().entry(arm).or_default() += 1;
        if arm == "B" {
            candidate_seconds += seconds;
            if split.is_some() {
                partial_seconds += sample["partial_gpu_seconds"].as_f64().unwrap();
                merge_seconds += sample["merge_gpu_seconds"].as_f64().unwrap();
            }
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
        partial_ms_per_operation: split.map(|_| partial_seconds * 1000.0 / 10.0 / 100.0),
        merge_ms_per_operation: split.map(|_| merge_seconds * 1000.0 / 10.0 / 100.0),
        control_drift_passed: split.is_some() || receipt["control_drift_passed"] == true,
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
    if expected.iter().any(|name| {
        name.parse::<MetalResearchCandidate>()
            .is_ok_and(MetalResearchCandidate::explicit_storage_abi)
    }) {
        return Err("the next decode round has no automatic advancement path".into());
    }
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
                "partial_mean_ms_per_operation":score.partial_ms_per_operation,
                "merge_mean_ms_per_operation":score.merge_ms_per_operation,
                "total_mean_ms_per_operation":score.candidate_ms_per_dispatch,
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

/// Queue-owned stage transition for exact-shape storage operators. Compile
/// success permits an oracle job; only oracle success permits timing jobs.
/// This does not select a winner or promote a production route.
fn operator_advance(root: &Path, stage: &str, output: &Path, expected: &[String]) -> Result {
    let config_path = root.join("campaign.json");
    let config = read(&config_path)?;
    if expected.is_empty()
        || config["candidate_names"] != json!(expected)
        || expected.iter().any(|name| {
            name.parse::<MetalResearchCandidate>()
                .map_or(true, |candidate| !candidate.decode_round_operator())
        })
    {
        return Err("operator advancement requires the entire exact candidate family".into());
    }
    if pin(&std::env::current_exe()?)? != config["job_generator"] {
        return Err("job generator identity drift".into());
    }
    let queue = absolute(config["queue"].as_str().ok_or("queue missing")?)?;
    let job_id = operator_advancement_id(&config, stage)?;
    if output != queue.join("results").join(&job_id) || !output.is_dir() {
        return Err("operator advancement output is not its queue result directory".into());
    }
    let queue_runner = absolute(
        config["queue_runner"]["path"]
            .as_str()
            .ok_or("queue runner missing")?,
    )?;
    if pin(&queue_runner)? != config["queue_runner"] {
        return Err("queue runner identity drift".into());
    }
    let list = match stage {
        "compile" => {
            generate(root, false, None, expected)?;
            root.join("oracle-jobs.json")
        }
        "oracle" => {
            generate(root, true, Some(0), expected)?;
            root.join("timing-jobs-L0.json")
        }
        _ => return Err("unknown operator advancement stage".into()),
    };
    let jobs = read(&list)?;
    let ids = jobs
        .as_array()
        .ok_or("generated operator job list missing")?;
    if ids.is_empty() {
        return Err("generated operator job list is empty".into());
    }
    for id in ids {
        let id = id.as_str().ok_or("generated operator job ID missing")?;
        submit_or_verify(
            &queue_runner,
            &queue,
            &root.join("jobs").join(format!("{id}.json")),
        )?;
    }
    if stage == "compile" {
        submit_or_verify(
            &queue_runner,
            &queue,
            &root.join("jobs").join(format!(
                "{}.json",
                operator_advancement_id(&config, "oracle")?
            )),
        )?;
    }
    json_new_or_identical(
        &output.join("operator-advancement.json"),
        &json!({"schema":"rvllm.decode-round.operator-advancement.v1",
            "campaign":config["campaign"],"campaign_sha256":hash(&config_path)?,
            "stage":stage,"candidate_names":expected,"submitted_jobs":jobs,
            "submitted_list_sha256":hash(&list)?,"promotion":false}),
    )
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
            partial_ms_per_operation: None,
            merge_ms_per_operation: None,
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
    fn split_v2_sample_accepts_roundtrip_ulp_and_fails_closed() {
        let sample = json!({"operations":100,"partial_dispatches":100,
            "merge_dispatches":100,"baseline_dispatches":0,
            "partial_gpu_seconds":0.75,"merge_gpu_seconds":0.25,
            "total_gpu_seconds":1.0});
        assert_eq!(
            sample_work_and_gpu_seconds(&sample, true, "B").unwrap(),
            (100, 1.0)
        );
        let mut roundtrip_ulp = sample.clone();
        roundtrip_ulp["total_gpu_seconds"] = json!(1.0 + f64::EPSILON);
        assert_eq!(
            sample_work_and_gpu_seconds(&roundtrip_ulp, true, "B").unwrap(),
            (100, 1.0 + f64::EPSILON)
        );
        let mut mismatched = sample.clone();
        mismatched["total_gpu_seconds"] = json!(0.75);
        assert!(sample_work_and_gpu_seconds(&mismatched, true, "B").is_err());
        let mut incomplete = sample;
        incomplete
            .as_object_mut()
            .unwrap()
            .remove("merge_gpu_seconds");
        assert!(sample_work_and_gpu_seconds(&incomplete, true, "B").is_err());
    }

    #[test]
    fn selected_oracle_candidates_are_strictly_validated() {
        validate_selected(&["metal-global-d512-r1p128t32".to_owned()]).unwrap();
        assert!(validate_selected(&["not-a-candidate".to_owned()]).is_err());
        let duplicate = "metal-global-d512-r1p128t32".to_owned();
        assert!(validate_selected(&[duplicate.clone(), duplicate]).is_err());
    }

    #[test]
    fn atlas_schedule_ids_do_not_alias_the_k8_control_or_each_other() {
        let config = json!({"campaign":"test"});
        let names = [
            "metal-global-d512-r16p64t128",
            "metal-global-d512-atlas_r16k16p64t128",
            "metal-global-d512-atlas_r16k32p64t128",
            "metal-global-d512-atlas_tile_r16k16p64t128",
            "metal-global-d512-atlas_tile_r16k32p64t128",
            "metal-global-d512-atlas_mma_r16k16p64t128",
            "metal-global-d512-atlas_mma_r16k32p64t128",
            "metal-global-d512-atlas_mma_r16k16p128t128",
            "metal-global-d512-atlas_mma_r8k32p64t128",
            "metal-global-d512-atlas_mma_r16k16p64t64",
            "metal-global-d512-atlas_mma_r16k64p64t128",
            "metal-global-d512-split-r8s256t128",
            "metal-global-d512-split-mma_r8k32s256t128",
        ];
        let ids = names
            .into_iter()
            .map(|name| {
                let candidate = name.parse::<MetalResearchCandidate>().unwrap();
                id(&config, candidate, "oracle").unwrap()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), names.len());
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
    fn executable_snapshot_is_content_addressed_and_never_repaired_in_place() {
        let directory = temp_directory("executables");
        let executable = std::env::current_exe().unwrap();
        let first = snapshot_executable(&executable, &directory, "test").unwrap();
        assert_eq!(hash(&first).unwrap(), hash(&executable).unwrap());
        assert_eq!(
            snapshot_executable(&executable, &directory, "test").unwrap(),
            first
        );
        std::fs::write(&first, b"corrupted snapshot").unwrap();
        assert!(snapshot_executable(&executable, &directory, "test").is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn operator_continuation_refuses_wrong_family_stage_and_output_owner() {
        let directory = temp_directory("operator-advance");
        let root = directory.join("campaign");
        let queue = directory.join("queue");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&queue).unwrap();
        let names = vec![MetalResearchCandidate::QmvW8G32R4Sg8K8.name().to_owned()];
        json_new(
            &root.join("campaign.json"),
            &json!({"campaign":"operator-probe","candidate_names":names,
                "queue":queue,"job_generator":pin(&std::env::current_exe().unwrap()).unwrap()}),
        )
        .unwrap();
        assert!(
            operator_advancement_id(&read(&root.join("campaign.json")).unwrap(), "other").is_err()
        );
        assert!(operator_advance(&root, "compile", &directory, &["off".into()]).is_err());
        assert!(operator_advance(&root, "compile", &directory, &names).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn operator_timing_rejects_stale_launch_geometry_in_native_oracle() {
        let candidate = MetalResearchCandidate::QmvW8G32R4Sg8K8;
        let mut identity = json!({"rows":32,"keys":1,"panel":32,"threads":256,
            "grid":[120,1,1],"kernel":candidate.kernels()[0].name(),
            "kernels":[{"threads":256}]});
        assert!(operator_oracle_identity_matches(candidate, &identity));
        identity["rows"] = json!(16);
        identity["threads"] = json!(64);
        identity["grid"] = json!([240, 1, 1]);
        assert!(!operator_oracle_identity_matches(candidate, &identity));
    }

    #[test]
    fn round_two_operator_oracles_admit_only_their_exact_launch_geometry() {
        use MetalResearchCandidate as C;
        let cases = [
            (C::FfnBf16R2Sg2, 4, 64, 3840),
            (C::FfnBf16R4Sg4, 16, 128, 960),
            (C::QmvW4G32R4Sg4, 16, 128, 240),
            (C::QmvW8G32R4Sg4K8192, 16, 128, 240),
            (C::QmvW8G32R2Sg4K4096, 8, 128, 480),
        ];
        for (candidate, rows, threads, grid) in cases {
            let kernel = candidate.kernels()[0];
            let mut identity = json!({"rows":rows,"keys":1,"panel":32,"threads":threads,
                "grid":[grid,1,1],"kernel":kernel.name(),
                "kernels":[{"threads":threads}]});
            assert!(operator_oracle_identity_matches(candidate, &identity));
            identity["grid"] = json!([grid + 1, 1, 1]);
            assert!(!operator_oracle_identity_matches(candidate, &identity));
        }
    }

    #[test]
    fn split_streaming_oracle_requires_complete_fp64_evidence() {
        let mut oracle = json!({
            "streaming_fp32_max_abs_bound":5.0e-5,
            "streaming_fp64_max_abs_bound":5.0e-4,
            "streaming_fp64_relative_l2_bound":1.0e-4,
            "cases":[{"independent_cpu_reference":"scalar FP64",
                "once_rounded_bf16":true,"repeatable":true,
                "guard_bytes_preserved":true,
                "max_fp64_abs_error":2.0e-6,"relative_l2_error":1.0e-6}]
        });
        assert!(split_streaming_fp64_evidence(&oracle));
        oracle["cases"][0]["max_fp64_abs_error"] = json!(5.1e-4);
        assert!(!split_streaming_fp64_evidence(&oracle));
        oracle["cases"][0]["max_fp64_abs_error"] = json!(2.0e-6);
        oracle["cases"][0]["independent_cpu_reference"] = json!("unknown");
        assert!(!split_streaming_fp64_evidence(&oracle));
        oracle["cases"] = json!([]);
        assert!(!split_streaming_fp64_evidence(&oracle));
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
            &json!({"schema":"rvllm.global-decode.oracle.v1","status":"passed",
                "identity":{"candidate":candidate.name(),
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

fn validate_observation_policy(policy: &Value) -> Result {
    let keys = [
        "power_source",
        "low_power_mode",
        "pmset_power_mode",
        "thermal_state",
        "quiet_process_names",
        "observe_process_names",
        "idle_llama_servers",
        "minimum_free_bytes",
        "disk_path",
    ];
    if !policy.as_object().is_some_and(|object| {
        object.len() == keys.len() && keys.iter().all(|key| object.contains_key(*key))
    }) {
        return Err("round2 conditions require exactly the nine explicit queue fields".into());
    }
    if !matches!(policy["power_source"].as_str(), Some("ac" | "battery"))
        || !policy["low_power_mode"].is_null()
        || !policy["pmset_power_mode"].is_null()
        || !policy["thermal_state"].is_null()
        || policy["quiet_process_names"] != json!([])
        || policy["idle_llama_servers"] != json!([])
        || policy["minimum_free_bytes"] != 0
        || !policy["disk_path"]
            .as_str()
            .is_some_and(|p| Path::new(p).is_absolute())
        || !policy["observe_process_names"]
            .as_array()
            .is_some_and(|names| {
                ["cargo", "rustc"]
                    .iter()
                    .all(|required| names.iter().any(|n| n == *required))
                    && names.iter().all(|n| {
                        n.as_str()
                            .is_some_and(|s| !s.is_empty() && !s.contains('/'))
                    })
            })
    {
        return Err("round2 requires null power-mode/thermal constraints, no quiet/idle-server gates, zero disk threshold, and observed (not gated) cargo/rustc".into());
    }
    Ok(())
}

fn expected_geometry(candidate: MetalResearchCandidate) -> Result<Value> {
    if !round_two::supported(candidate) {
        return Err("not a second-round candidate".into());
    }
    let (rows, keys, panel, threads, grid, scratch) =
        if let Some(tile) = candidate.global_decode_tile() {
            (
                tile.rows as usize,
                tile.keys as usize,
                tile.panel as usize,
                tile.threads as usize,
                json!([16 / tile.rows, 1, 1]),
                0,
            )
        } else if let Some(tile) = candidate.split_global_decode_tile() {
            (
                tile.rows as usize,
                tile.keys as usize,
                tile.panel as usize,
                tile.threads as usize,
                json!([16 / tile.rows, tile.partitions, 1]),
                tile.partitions as usize * 16 * 514 * 4,
            )
        } else {
            let launch = rvllm_apple_metal::research_decode::operator_launch(candidate)
                .ok_or("operator launch missing")?;
            let n = if candidate.ffn_decode() { 15360 } else { 3840 };
            (
                launch.rows_per_group,
                1,
                32,
                launch.threads,
                json!([n / launch.rows_per_group, 1, 1]),
                0,
            )
        };
    Ok(
        json!({"rows":rows,"keys":keys,"panel":panel,"threads":threads,"grid":grid,
        "scratch_bytes":scratch,"simd_matrix":false,
        "max_logical_capacity_tokens":candidate.global_capacity_tokens()}),
    )
}

fn validate_resource_identity(identity: &Value, candidate: MetalResearchCandidate) -> Result {
    if identity["candidate"] != candidate.name()
        || identity["gpu_family"] != "Apple9"
        || !identity["device_name"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        || identity["kernel"] != candidate.kernels()[0].name()
    {
        return Err("native candidate/device identity mismatch".into());
    }
    for (key, value) in expected_geometry(candidate)?
        .as_object()
        .ok_or("bad geometry")?
    {
        if identity[key] != *value {
            return Err(format!("native geometry mismatch: {key}").into());
        }
    }
    let kernels = identity["kernels"]
        .as_array()
        .ok_or("native kernel resources missing")?;
    if kernels.len() != candidate.kernels().len() {
        return Err("kernel family size mismatch".into());
    }
    for (actual, expected) in kernels.iter().zip(candidate.kernels()) {
        let (threads, shared) = expected.limits();
        if actual["name"] != expected.name()
            || actual["threads"].as_u64() != Some(threads as u64)
            || actual["source_threadgroup_bytes"].as_u64() != Some(shared as u64)
            || actual["actual_execution_width"] != 32
            || !actual["actual_max_threads"]
                .as_u64()
                .is_some_and(|n| n >= threads as u64)
            || actual["actual_static_threadgroup_bytes"].as_u64().is_none()
        {
            return Err("missing/mismatched public PSO resource evidence".into());
        }
    }
    Ok(())
}

/// Compare the submitted queue manifest with the generated one, including
/// scalar limits and all pins. The queue serializes additional default-null
/// fields; compare the contractual fields rather than JSON whitespace/defaults.
fn verified_round_two_job(root: &Path, queue: &Path, job_id: &str) -> Result<(Value, Vec<Value>)> {
    let generated_path = root.join("jobs").join(format!("{job_id}.json"));
    let submitted_path = queue.join("jobs").join(format!("{job_id}.json"));
    let generated = read(&generated_path)?;
    let submitted = read(&submitted_path)?;
    for key in [
        "schema",
        "id",
        "purpose",
        "command",
        "inputs",
        "after",
        "stable_seconds",
        "max_wait_seconds",
        "max_run_seconds",
    ] {
        if generated[key] != submitted[key] {
            return Err(
                format!("submitted job differs from generated contract: {job_id}: {key}").into(),
            );
        }
    }
    // These campaign jobs have no validator/submission override. Reject an
    // injected optional command even though the queue serializes null defaults.
    for key in ["validator", "kernel_game_submission"] {
        if !generated[key].is_null() || !submitted[key].is_null() {
            return Err("unexpected optional queue command or submission override".into());
        }
    }
    // Optional condition fields are always explicit for this protocol.
    if generated["conditions"] != submitted["conditions"] {
        return Err("submitted observation policy differs".into());
    }
    validate_observation_policy(&submitted["conditions"])?;
    if submitted["id"] != job_id || submitted["stable_seconds"] != 0 {
        return Err("submitted job ID/dwell mismatch".into());
    }
    let directory = succeeded(queue, job_id)?;
    let report_path = directory.join("report.json");
    let report = read(&report_path)?;
    if report["schema"] != "rvllm.experiment_result.v1"
        || report["id"] != job_id
        || report["overdue"] != false
        || report["exit_code"] != 0
    {
        return Err("queue receipt identity/runtime failure".into());
    }
    let mut pins = vec![
        pin(&generated_path)?,
        pin(&submitted_path)?,
        pin(&report_path)?,
    ];
    for item in std::iter::once(&submitted["command"]["executable"]).chain(
        submitted["inputs"]
            .as_array()
            .ok_or("job input pins missing")?
            .iter(),
    ) {
        let path = absolute(item["path"].as_str().ok_or("pin path missing")?)?;
        if pin(&path)? != *item {
            return Err("submitted input/executable pin drift".into());
        }
        pins.push(item.clone());
    }
    Ok((submitted, pins))
}

/// Read-only revalidation. Summaries cannot authorize advancement: all raw
/// sample files, submitted manifests, executable/source/build/library pins,
/// native numerical oracle and matching output are required again.
fn round_two_cell(
    root: &Path,
    candidate: MetalResearchCandidate,
    length: u32,
    operator_k: Option<usize>,
) -> Result<Value> {
    let config_path = root.join("campaign.json");
    let config = read(&config_path)?;
    if config["screen_protocol"] != round_two::PROTOCOL
        || config["source_base_commit"] != "593e1d6fb25088608198f2248dcac038d19fe761"
        || !config["candidate_names"]
            .as_array()
            .is_some_and(|names| names.iter().any(|name| name == candidate.name()))
    {
        return Err("candidate/campaign is not second-round sealed input".into());
    }
    round_two::predecessor(candidate, length)?;
    validate_observation_policy(&config["conditions"])?;
    for key in [
        "test_executable",
        "job_generator",
        "abba_retainer",
        "queue_runner",
    ] {
        let path = absolute(
            config[key]["path"]
                .as_str()
                .ok_or("executable pin missing")?,
        )?;
        if config[key] != pin(&path)? {
            return Err(format!("{key} identity drift").into());
        }
    }
    let queue = absolute(config["queue"].as_str().ok_or("queue missing")?)?;
    let suffix = if let Some(k) = operator_k {
        format!("abba-L0-K{k}")
    } else {
        format!("abba-L{length}")
    };
    let job_id = id(&config, candidate, &suffix)?;
    let (job, mut pins) = verified_round_two_job(root, &queue, &job_id)?;
    if job["purpose"] != "exploratory_timing" {
        return Err("round2 timing must remain exploratory".into());
    }
    let directory = queue.join("results").join(&job_id).join("native");
    let receipt_path = directory.join("abba.json");
    let receipt = read(&receipt_path)?;
    let stats = round_two::validate_collection(&receipt)?;
    if receipt["candidate"] != candidate.name()
        || receipt["length"].as_u64() != Some(length as u64)
        || receipt["operator_k"].as_u64() != operator_k.map(|k| k as u64)
    {
        return Err("cell shape identity mismatch".into());
    }
    validate_resource_identity(&receipt["identity"], candidate)?;
    pins.push(pin(&receipt_path)?);
    pins.push(pin(&config_path)?);
    for (index, sample) in receipt["samples"]
        .as_array()
        .ok_or("samples missing")?
        .iter()
        .enumerate()
    {
        let path = directory.join(format!("sample-{}-{}.json", index / 4, index % 4));
        if read(&path)? != *sample {
            return Err("raw sample differs from aggregate; never prune".into());
        }
        pins.push(pin(&path)?);
    }
    let env = &job["command"]["env"];
    let source = absolute(
        env[format!("{PREFIX}SOURCE")]
            .as_str()
            .ok_or("source missing")?,
    )?;
    let library = absolute(
        env[format!("{PREFIX}METALLIB")]
            .as_str()
            .ok_or("library missing")?,
    )?;
    let build_path = absolute(
        env[format!("{PREFIX}BUILD_RECEIPT")]
            .as_str()
            .ok_or("build missing")?,
    )?;
    let build = read(&build_path)?;
    if source != source_path(root, candidate, "core")
        || receipt["identity"]["oracle_library"] != false
        || build["schema"] != "rvllm.global-decode.build.v1"
        || receipt["identity"]["core_sha256"] != hash(&source)?
        || receipt["identity"]["source_sha256"] != hash(&source)?
        || receipt["identity"]["metallib_sha256"] != hash(&library)?
        || receipt["identity"]["build_receipt_sha256"] != hash(&build_path)?
        || receipt["identity"]["test_executable_sha256"] != config["test_executable"]["sha256"]
        || build["status"] != "compiled"
        || build["flags"] != json!(["-std=metal3.1", "-fno-fast-math"])
        || build["source_sha256"] != hash(&source)?
        || build["metallib_sha256"] != hash(&library)?
    {
        return Err("compiled native identity mismatch".into());
    }
    let air = build_path
        .parent()
        .ok_or("build has no parent")?
        .join("kernel.air");
    if build["air_sha256"] != hash(&air)? {
        return Err("AIR identity drift".into());
    }
    pins.push(pin(&air)?);
    let compile_id = id(&config, candidate, "compile-core")?;
    let (_, compile_pins) = verified_round_two_job(root, &queue, &compile_id)?;
    pins.extend(compile_pins);
    let oracle_id = id(&config, candidate, "oracle")?;
    let (oracle_job, oracle_pins) = verified_round_two_job(root, &queue, &oracle_id)?;
    pins.extend(oracle_pins);
    let oracle_path = absolute(
        env[format!("{PREFIX}ORACLE_RECEIPT")]
            .as_str()
            .ok_or("oracle missing")?,
    )?;
    let expected_oracle = queue.join("results").join(&oracle_id).join("native").join(
        if candidate.split_global_decode_tile().is_some() {
            "split-oracle.json"
        } else {
            "oracle.json"
        },
    );
    let oracle = read(&oracle_path)?;
    if oracle_path != expected_oracle
        || receipt["oracle_receipt_sha256"] != hash(&oracle_path)?
        || oracle["status"] != "passed"
    {
        return Err("native oracle identity mismatch".into());
    }
    validate_resource_identity(&oracle["identity"], candidate)?;
    let expected_oracle_library = candidate.split_global_decode_tile().is_none();
    if oracle["identity"]["oracle_library"].as_bool() != Some(expected_oracle_library) {
        return Err("oracle library role mismatch".into());
    }
    if candidate.global_decode_tile().is_some() {
        let oracle_compile_id = id(&config, candidate, "compile-oracle")?;
        let (_, oracle_compile_pins) = verified_round_two_job(root, &queue, &oracle_compile_id)?;
        pins.extend(oracle_compile_pins);
    }
    for key in [
        "candidate",
        "core_sha256",
        "test_executable_sha256",
        "device_name",
        "gpu_family",
    ] {
        if oracle["identity"][key] != receipt["identity"][key] {
            return Err("oracle/timing identity drift".into());
        }
    }
    let oracle_env = &oracle_job["command"]["env"];
    for (field, variable) in [
        ("source_sha256", "SOURCE"),
        ("metallib_sha256", "METALLIB"),
        ("build_receipt_sha256", "BUILD_RECEIPT"),
    ] {
        let path = absolute(
            oracle_env[format!("{PREFIX}{variable}")]
                .as_str()
                .ok_or("oracle pin missing")?,
        )?;
        if oracle["identity"][field] != hash(&path)? {
            return Err("oracle compiled pin drift".into());
        }
    }
    let cases = oracle["cases"].as_array().ok_or("oracle cases missing")?;
    let label = format!("L{length}");
    let case = cases
        .iter()
        .find(|case| {
            operator_k.map_or(case["label"] == label, |k| {
                case["k"].as_u64() == Some(k as u64)
            })
        })
        .ok_or("cell has no correctness case")?;
    if case["repeats"] != 3 {
        return Err("three repeated uses required".into());
    }
    if candidate.decode_round_operator() {
        if oracle["schema"] != "rvllm.decode-round.oracle.v1"
            || case["repeated_bit_exact"] != true
            || case["guards_untouched"] != true
            || case["exact_dispatch_accounting"] != true
            || case["persistent_arena_unchanged"] != true
        {
            return Err("operator numerical/guard/accounting evidence missing".into());
        }
        if candidate.ffn_decode()
            && !cases.iter().any(|case| {
                case["dense_ffn_fixture"] == true
                    && case["k"] == 3840
                    && case["repeats"] == 3
                    && case["repeated_bit_exact"] == true
                    && case["guards_untouched"] == true
                    && case["exact_dispatch_accounting"] == true
                    && case["persistent_arena_unchanged"] == true
            })
        {
            return Err("full dense FFN oracle case missing".into());
        }
    } else if candidate.split_global_decode_tile().is_some() {
        if oracle["schema"] != "rvllm.global-decode.split-oracle.v1"
            || oracle["streaming_fp32_max_abs_bound"] != 5e-5
            || oracle["streaming_fp64_max_abs_bound"] != 5e-4
            || oracle["streaming_fp64_relative_l2_bound"] != 1e-4
            || oracle["host_refusals"] != 21
            || oracle["shader_refusals"] != 4
            || case["repeatable"] != true
            || case["guard_bytes_preserved"] != true
            || case["once_rounded_bf16"] != true
        {
            return Err("split numerical/guard evidence missing".into());
        }
    } else if oracle["schema"] != "rvllm.global-decode.oracle.v1"
        || oracle["fp64_max_abs_bound"] != 5e-4
        || oracle["fp64_relative_l2_bound"] != 1e-4
        || case["exact_fp32_gpu_oracle"] != true
        || case["exact_once_rounded_bf16"] != true
        || case["read_inputs_and_guard_bytes_preserved"] != true
        || case["sampled_gpu_dots_match_cpu_fp32"] != true
        || case["sampled_gpu_dots_pass_fp64_bound"] != true
    {
        return Err("unsplit numerical/guard evidence missing".into());
    }
    // Hash ALL oracle outputs, not only the requested timing length.
    for case in cases {
        let path = absolute(case["bf16_file"].as_str().ok_or("oracle output missing")?)?;
        if case["bf16_sha256"] != hash(&path)? {
            return Err("oracle output changed".into());
        }
        pins.push(pin(&path)?);
    }
    pins.push(pin(&oracle_path)?);
    // Stable, de-duplicated pin list. Conflicting pins are a hard failure.
    let mut unique = BTreeMap::new();
    for item in pins {
        let path = item["path"].as_str().ok_or("pin path missing")?.to_owned();
        if unique
            .insert(path, item.clone())
            .is_some_and(|old| old != item)
        {
            return Err("conflicting proof input identities".into());
        }
    }
    let queue_report = read(&queue.join("results").join(&job_id).join("report.json"))?;
    Ok(
        json!({"schema":round_two::PROTOCOL,"campaign":config["campaign"],
        "candidate":candidate.name(),"length":length,"operator_k":operator_k,
        "statistics":stats.receipt(),"pins":unique.into_values().collect::<Vec<_>>(),
        "queue_sampled_conditions_eligible":queue_report["sampled_conditions_eligible"],
        "conditions_are_observations_only":true,
        "confirmation_of":config["confirmation_of"],"promotion":false}),
    )
}

fn admit_round_two(
    root: &Path,
    candidate: MetalResearchCandidate,
    length: u32,
) -> Result<Vec<Value>> {
    let Some(previous) = round_two::predecessor(candidate, length)? else {
        return Ok(Vec::new());
    };
    // Revalidate the complete chain, not just an editable advancement summary.
    // Depth is bounded by the four predeclared lengths, so no unbounded recursion.
    let mut inputs = admit_round_two(root, candidate, previous)?;
    let proof = round_two_cell(root, candidate, previous, None)?;
    if proof["statistics"]["exploratory_advance_only"] != true {
        return Err(
            "preceding screen failed paired exploratory diagnostic; preserve it, do not retry"
                .into(),
        );
    }
    let path = root.join(format!(
        "{}-L{previous}-to-L{length}.screen.json",
        candidate.name()
    ));
    json_new_or_identical(&path, &proof)?;
    inputs.push(pin(&path)?);
    inputs.extend(
        proof["pins"]
            .as_array()
            .ok_or("proof pins missing")?
            .iter()
            .cloned(),
    );
    // Avoid redundant large executable hashing in queue input validation.
    let mut unique = BTreeMap::new();
    for item in inputs {
        let path = item["path"].as_str().ok_or("pin path missing")?.to_owned();
        if unique
            .insert(path, item.clone())
            .is_some_and(|old| old != item)
        {
            return Err("conflicting advancement input identities".into());
        }
    }
    Ok(unique.into_values().collect())
}

fn review_round_two(root: &Path, length: u32, selected: &[String]) -> Result {
    validate_selected(selected)?;
    let mut cells = Vec::new();
    for name in selected {
        let candidate: MetalResearchCandidate = name.parse()?;
        if candidate.decode_round_operator() {
            for &k in candidate.operator_keys() {
                cells.push(round_two_cell(root, candidate, length, Some(k))?);
            }
        } else {
            cells.push(round_two_cell(root, candidate, length, None)?);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"schema":round_two::PROTOCOL,
        "cells":cells,"automatic_submission":false,"automatic_retry":false,"promotion":false}))?
    );
    Ok(())
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
        [action,candidate,path] if action=="emit-source" => {
            let candidate: MetalResearchCandidate = candidate.parse()?;
            validate_selected(&[candidate.name().to_owned()])?;
            let source = rvllm_apple_metal::kernels::kernel_source_with_options(
                MetalFloatType::Bf16,
                MetalKernelOptions {
                    research: candidate,
                    ..MetalKernelOptions::default()
                },
            );
            write(&absolute(path)?, source.as_bytes())
        }
        [action,campaign,root,queue,test,conditions,selected @ ..] if action=="prepare" =>
            prepare(campaign,&absolute(root)?,&absolute(queue)?,&absolute(test)?,&absolute(conditions)?,selected,false,None),
        [action,campaign,root,queue,test,conditions,selected @ ..] if action=="prepare-round2" =>
            prepare(campaign,&absolute(root)?,&absolute(queue)?,&absolute(test)?,&absolute(conditions)?,selected,true,None),
        [action,campaign,root,queue,test,conditions,prior,selected @ ..] if action=="confirm-round2" =>
            prepare(campaign,&absolute(root)?,&absolute(queue)?,&absolute(test)?,&absolute(conditions)?,selected,true,Some(&absolute(prior)?)),
        [action,root,length,selected @ ..] if action=="review-round2" && !selected.is_empty() =>
            review_round_two(&absolute(root)?,length.parse()?,selected),
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
        [action,root,stage,output,expected @ ..] if action=="operator-advance" && !expected.is_empty() =>
            operator_advance(&absolute(root)?,stage,&absolute(output)?,expected),
        _=>Err("usage: rvllm-global-decode-jobs test-exe CARGO_JSON | emit-source CANDIDATE FRESH_FILE | prepare ID ROOT QUEUE TEST_EXE CONDITIONS_JSON [CANDIDATE...] | compile SOURCE FRESH_OUTPUT_DIR | oracle-jobs ROOT [CANDIDATE...] | timing-jobs ROOT [LENGTH CANDIDATE...] | advance ROOT LENGTH OUTPUT EXPECTED_CANDIDATE... | operator-advance ROOT compile|oracle QUEUE_OUTPUT EXPECTED_CANDIDATE... | prepare-round2 ID ROOT QUEUE TEST_EXE CONDITIONS_JSON CANDIDATE... | confirm-round2 ID ROOT QUEUE TEST_EXE CONDITIONS_JSON PRIOR_ROOT CANDIDATE... | review-round2 ROOT LENGTH CANDIDATE...".into()),
    }
}

#[cfg(test)]
mod next_round_queue_tests {
    use super::*;
    #[test]
    fn projection_cells_and_short_lengths_are_explicit_and_bounded() {
        let ffn = MetalResearchCandidate::FfnBf16R4Sg2.name().to_owned();
        let short = MetalResearchCandidate::GlobalD512ShortR4T128
            .name()
            .to_owned();
        assert!(validate_timing_request(0, &[ffn.clone()]).is_ok());
        assert!(validate_timing_request(0, &[]).is_err());
        assert!(validate_timing_request(256, &[ffn]).is_err());
        assert!(validate_timing_request(256, &[short.clone()]).is_ok());
        assert!(validate_timing_request(512, &[short.clone()]).is_ok());
        assert!(validate_timing_request(1024, &[short.clone()]).is_err());
        assert!(validate_timing_request(0, &[short]).is_err());
    }
}

#[cfg(test)]
mod round_two_policy_tests {
    use super::*;

    fn policy() -> Value {
        json!({
            "power_source": "ac", "low_power_mode": null,
            "pmset_power_mode": null, "thermal_state": null,
            "minimum_free_bytes": 0, "disk_path": "/",
            "quiet_process_names": [], "idle_llama_servers": [],
            "observe_process_names": ["cargo", "rustc"]
        })
    }

    #[test]
    fn observations_cannot_be_replaced_with_waiting_conditions() {
        assert!(validate_observation_policy(&policy()).is_ok());
        for (key, value) in [
            ("thermal_state", json!(0)),
            ("low_power_mode", json!(false)),
            ("pmset_power_mode", json!(2)),
            ("quiet_process_names", json!(["cargo"])),
            ("idle_llama_servers", json!([{"pid": 1, "port": 8080}])),
            ("minimum_free_bytes", json!(1)),
            ("observe_process_names", json!(["cargo"])),
            ("disk_path", json!("relative")),
            ("unexpected", json!(true)),
        ] {
            let mut changed = policy();
            changed[key] = value;
            assert!(validate_observation_policy(&changed).is_err(), "{key}");
        }
        let mut missing = policy();
        missing.as_object_mut().unwrap().remove("thermal_state");
        assert!(validate_observation_policy(&missing).is_err());
    }

    #[test]
    fn new_candidates_have_unique_bounded_manifest_identities() {
        let config = json!({"campaign": "round2-test", "screen_protocol": round_two::PROTOCOL});
        let mut identities = BTreeSet::new();
        let mut count = 0;
        for candidate in candidates()
            .into_iter()
            .filter(|candidate| candidate.round_two())
        {
            count += 1;
            let geometry = expected_geometry(candidate).unwrap();
            assert_eq!(geometry["simd_matrix"], false);
            for suffix in ["compile-core", "oracle", "abba-L2048", "abba-L0-K15360"] {
                let identity = id(&config, candidate, suffix).unwrap();
                assert!(identity.len() <= 96);
                assert!(identities.insert(identity));
            }
            if candidate.decode_round_operator() {
                assert_eq!(geometry["max_logical_capacity_tokens"], 0);
                assert!(!candidate.operator_keys().is_empty());
            } else {
                assert_eq!(geometry["max_logical_capacity_tokens"], 2048);
                assert!(round_two::predecessor(candidate, 4096).is_err());
                assert_eq!(round_two::predecessor(candidate, 2048).unwrap(), Some(1024));
            }
        }
        assert_eq!(count, 8);
    }
}
