//! Offline, explicit queue preparation. No accelerator execution, implicit
//! submission, shell, environment-based dispatch, winner selection or promotion.
#![forbid(unsafe_code)]
use rvllm_apple_metal::{MetalFloatType, MetalKernelOptions, MetalResearchCandidate};
use serde_json::{json, Value};
use std::{
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
        .filter(|c| c.global_decode_tile().is_some())
        .collect()
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
    let tile = candidate.global_decode_tile().unwrap();
    Ok(format!(
        "{campaign}-r{}p{}t{}-{suffix}",
        tile.rows, tile.panel, tile.threads
    ))
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
        "stable_seconds":30,"max_wait_seconds":7200,"max_run_seconds":3600});
    json_new(&root.join("jobs").join(format!("{job_id}.json")), &value)
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
    let retainer = executable
        .parent()
        .ok_or("generator has no parent directory")?
        .join("rvllm-retain-abba");
    let exploratory = policy["low_power_mode"].is_null()
        || policy["pmset_power_mode"].is_null()
        || policy["thermal_state"].is_null();
    let config = json!({"schema":"rvllm.global-decode.campaign.v1","campaign":campaign,
        "queue":queue,"test_executable":pin(test)?,"job_generator":pin(&executable)?,
        "abba_retainer":pin(&retainer)?,"exploratory":exploratory,
        "conditions":policy,"conditions_input":pin(conditions)?,
        "lengths":[256,512,1024,2048,4096],"split_kv":false,"local_prefill":false,
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
fn generate(root: &Path, timing: bool) -> Result {
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
    let mut all = Vec::new();
    // Never prune the family from partial results. A failed compile/oracle stops
    // generation; revised families require a new explicit campaign identity.
    for candidate in candidates() {
        let flavor = if timing { "core" } else { "oracle" };
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
            let oracle_path = succeeded(&queue, &oracle_id)?.join("native/oracle.json");
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
        let lengths: Vec<Option<u32>> = if timing {
            [256, 512, 1024, 2048, 4096].map(Some).to_vec()
        } else {
            vec![None]
        };
        for length in lengths {
            let suffix = length.map_or("oracle".into(), |n| format!("abba-L{n}"));
            let job_id = id(&config, candidate, &suffix)?;
            // The existing queue expands {output} in argv ONLY, never in env.
            // Its exclusive result directory is results/<immutable job ID>.
            env[format!("{PREFIX}REPORT_DIR")] =
                json!(queue.join("results").join(&job_id).join("native"));
            let test_name = if timing {
                "global_decode_abba"
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
    json_new(
        &root.join(if timing {
            "timing-jobs.json"
        } else {
            "oracle-jobs.json"
        }),
        &json!(all),
    )
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
        [action,root] if action=="oracle-jobs" => generate(&absolute(root)?,false),
        [action,root] if action=="timing-jobs" => generate(&absolute(root)?,true),
        _=>Err("usage: rvllm-global-decode-jobs test-exe CARGO_JSON | prepare ID ROOT QUEUE TEST_EXE CONDITIONS_JSON | compile SOURCE FRESH_OUTPUT_DIR | oracle-jobs ROOT | timing-jobs ROOT".into()),
    }
}
