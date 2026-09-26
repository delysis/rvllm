//! Exact-source export and strict offline Metal compilation for donor trials.
//! No model loading, network activity, GPU work, or promotion.
use rvllm_apple_metal::{
    donor12b, kernels, MetalFloatType, MetalKernelOptions, MetalResearchCandidate,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
fn write_new(path: &Path, bytes: &[u8]) -> Result {
    let mut f = fs::File::create_new(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn hash(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn run(directory: &Path, name: &str, args: &[&str]) -> Result {
    let output = Command::new("xcrun")
        .args(args)
        .current_dir(directory)
        .output()?;
    write_new(&directory.join(format!("{name}.stdout")), &output.stdout)?;
    write_new(&directory.join(format!("{name}.stderr")), &output.stderr)?;
    if !output.status.success() {
        return Err(format!("xcrun {name} failed: {}", output.status).into());
    }
    Ok(())
}
fn main() -> Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 || !matches!(args[0].as_str(), "source" | "build") {
        return Err("usage: rvllm-donor12b-source (source|build) SELECTOR ABSOLUTE_PATH".into());
    }
    let candidate: MetalResearchCandidate = args[1].parse()?;
    if donor12b::simdgroups(candidate).is_none() {
        return Err("explicit donor12b selector required".into());
    }
    let target = PathBuf::from(&args[2]);
    if !target.is_absolute() {
        return Err("output must be an absolute, fresh path".into());
    }
    let source = kernels::kernel_source_with_options(
        MetalFloatType::Bf16,
        MetalKernelOptions {
            research: candidate,
            ..MetalKernelOptions::default()
        },
    );
    if args[0] == "source" {
        return write_new(&target, source.as_bytes());
    }
    fs::create_dir(&target)?;
    write_new(&target.join("core.metal"), source.as_bytes())?;
    run(
        &target,
        "sdk-version",
        &["--sdk", "macosx", "--show-sdk-version"],
    )?;
    run(
        &target,
        "metal",
        &[
            "--sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-fno-fast-math",
            "-c",
            "core.metal",
            "-o",
            "core.air",
        ],
    )?;
    run(
        &target,
        "metallib",
        &[
            "--sdk",
            "macosx",
            "metallib",
            "core.air",
            "-o",
            "core.metallib",
        ],
    )?;
    let receipt = json!({"status":"compiled","candidate":candidate.name(),
        "source_sha256":hash(&target.join("core.metal"))?,
        "metallib_sha256":hash(&target.join("core.metallib"))?,
        "flags":["-std=metal3.1","-fno-fast-math"],
        "gpu_executed":false,"promotion":false});
    write_new(
        &target.join("build.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
