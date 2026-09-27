//! Build and validate immutable, self-contained Apple inference packages.

use rvllm_apple::{
    build_apple_model_package, AppleLowBitExportRequest, AppleLowBitWeightFormat,
    AppleModelPackage, AppleModelPackageBuildConfig, AppleWeightFormat,
};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
Usage:
  rvllm_apple_package build --model-dir DIR --metallib-root DIR --output DIR [--package-id ID] [--weight-format auto|f16|bf16] [--low-bit-proj TENSOR=w4a16-group32|w8a16-group32]...
  rvllm_apple_package validate PACKAGE_DIR

The build command never modifies the Hugging Face source directory and refuses
to overwrite an existing output package.";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<String>) -> Result<(), String> {
    let mut arguments = arguments.into_iter();
    match arguments.next().as_deref() {
        Some("build") => build(arguments.collect()),
        Some("validate") => validate(arguments.collect()),
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(command) => Err(format!("unknown command {command:?}")),
        None => Err("missing command".to_owned()),
    }
}

fn build(arguments: Vec<String>) -> Result<(), String> {
    let mut model_dir = None;
    let mut metallib_root = None;
    let mut output_dir = None;
    let mut package_id = None;
    let mut weight_format = None;
    let mut low_bit_projections = Vec::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model-dir" => set_once(&mut model_dir, PathBuf::from(value), &flag)?,
            "--metallib-root" => set_once(&mut metallib_root, PathBuf::from(value), &flag)?,
            "--output" => set_once(&mut output_dir, PathBuf::from(value), &flag)?,
            "--package-id" => set_once(&mut package_id, value, &flag)?,
            "--weight-format" => {
                let parsed = match value.as_str() {
                    "auto" => None,
                    "f16" => Some(AppleWeightFormat::F16),
                    "bf16" => Some(AppleWeightFormat::Bf16),
                    "w4a16-group32" | "w8a16-group32" => {
                        return Err(
                            "W4/W8 assembly is unavailable until the dedicated quantizing exporter is enabled"
                                .to_owned(),
                        )
                    }
                    _ => return Err(format!("unsupported weight format {value:?}")),
                };
                if weight_format.is_some() {
                    return Err("duplicate --weight-format".to_owned());
                }
                weight_format = Some(parsed);
            }
            "--low-bit-proj" | "--low-bit-down-proj" => {
                low_bit_projections.push(parse_low_bit_projection(&value)?);
            }
            _ => return Err(format!("unknown build option {flag:?}")),
        }
    }
    let model_dir = model_dir.ok_or_else(|| "missing --model-dir".to_owned())?;
    let metallib_root = metallib_root.ok_or_else(|| "missing --metallib-root".to_owned())?;
    let output_dir = output_dir.ok_or_else(|| "missing --output".to_owned())?;
    let package_id = package_id.unwrap_or_else(|| {
        model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("apple-model")
            .to_owned()
    });
    let report = build_apple_model_package(&AppleModelPackageBuildConfig {
        model_dir,
        metallib_root,
        output_dir,
        package_id,
        weight_format: weight_format.unwrap_or(None),
        low_bit_projections,
    })?;
    println!("Apple model package built and validated");
    println!("path: {}", report.output_dir.display());
    println!("identity_sha256: {}", report.identity_fingerprint);
    println!("architecture: {}", report.architecture);
    println!("native_weight_format: {:?}", report.weight_format);
    println!("weight_shards: {}", report.weight_shards);
    println!("weight_bytes: {}", report.weight_bytes);
    println!("low_bit_projection_count: {}", report.low_bit_tensors);
    println!("low_bit_projection_bytes: {}", report.low_bit_bytes);
    println!("low_bit_projection_formats: {:?}", report.low_bit_formats);
    println!("metal_variants: {}", report.metal_variants);
    Ok(())
}

fn parse_low_bit_projection(value: &str) -> Result<AppleLowBitExportRequest, String> {
    let (tensor_name, format) = value.split_once('=').ok_or_else(|| {
        "--low-bit-proj requires TENSOR=w4a16-group32 or TENSOR=w8a16-group32".to_owned()
    })?;
    if tensor_name.trim().is_empty() {
        return Err("--low-bit-proj tensor name must not be empty".to_owned());
    }
    let format = match format {
        "w4a16-group32" => AppleLowBitWeightFormat::W4A16,
        "w8a16-group32" => AppleLowBitWeightFormat::W8A16,
        _ => return Err(format!("unsupported low-bit projection format {format:?}")),
    };
    Ok(AppleLowBitExportRequest {
        tensor_name: tensor_name.to_owned(),
        format,
    })
}

fn validate(arguments: Vec<String>) -> Result<(), String> {
    if arguments.len() != 1 {
        return Err("validate requires exactly one package directory".to_owned());
    }
    let package = AppleModelPackage::open(&arguments[0])
        .map_err(|error| format!("package validation failed: {error}"))?;
    println!("Apple model package is valid");
    println!("path: {}", package.root().display());
    println!("package_id: {}", package.manifest().package_id);
    println!("architecture: {}", package.manifest().architecture);
    println!(
        "identity_sha256: {}",
        lower_hex(&package.identity_fingerprint())
    );
    Ok(())
}

fn set_once<T>(target: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if target.replace(value).is_some() {
        return Err(format!("duplicate {flag}"));
    }
    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_or_incomplete_commands() {
        assert!(run(vec!["unknown".to_owned()]).is_err());
        assert!(run(vec!["build".to_owned()]).is_err());
        assert!(run(vec!["validate".to_owned()]).is_err());
    }

    #[test]
    fn low_bit_format_fails_closed() {
        let error = run(vec![
            "build".to_owned(),
            "--model-dir".to_owned(),
            "/tmp/model".to_owned(),
            "--metallib-root".to_owned(),
            "/tmp/metal".to_owned(),
            "--output".to_owned(),
            "/tmp/package".to_owned(),
            "--weight-format".to_owned(),
            "w4a16-group32".to_owned(),
        ])
        .expect_err("W4 must fail closed");
        assert!(error.contains("dedicated quantizing exporter"));
    }

    #[test]
    fn parses_explicit_tensor_level_low_bit_projection() {
        let request = parse_low_bit_projection("model.layers.0.mlp.down_proj.weight=w4a16-group32")
            .expect("parse low-bit sidecar request");
        assert_eq!(request.tensor_name, "model.layers.0.mlp.down_proj.weight");
        assert_eq!(request.format, AppleLowBitWeightFormat::W4A16);
        assert!(parse_low_bit_projection("model.layers.0.mlp.down_proj.weight=w4").is_err());
        assert!(parse_low_bit_projection("=w8a16-group32").is_err());
    }
}
