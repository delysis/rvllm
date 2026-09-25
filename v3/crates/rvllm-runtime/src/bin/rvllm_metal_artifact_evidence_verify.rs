//! Strict verifier for `rvllm.metal_artifact_evidence.v1` companion reports.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::parse_strict_json;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

const SCHEMA: &str = "rvllm.metal_artifact_evidence.v1";

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn exact_keys(object: &Map<String, Value>, expected: &[&str], label: &str) -> Result<(), String> {
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} fields differ: expected {expected:?}, got {actual:?}"
        ))
    }
}

fn string<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("{label} is not a SHA-256 digest"))
    }
}

fn verify_artifact(value: &Value, label: &str) -> Result<(), String> {
    let identity = object(value, label)?;
    exact_keys(identity, &["bytes", "path", "sha256"], label)?;
    let path = Path::new(string(identity, "path")?);
    let expected_len = identity
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.bytes must be a nonnegative integer"))?;
    let expected_digest = string(identity, "sha256")?;
    digest(expected_digest, &format!("{label}.sha256"))?;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() as u64 != expected_len {
        return Err(format!("{label} byte length does not match"));
    }
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected_digest.to_ascii_lowercase() {
        return Err(format!("{label} SHA-256 does not match"));
    }
    Ok(())
}

fn verify_capture(value: &Value, label: &str) -> Result<(), String> {
    let capture = object(value, label)?;
    exact_keys(capture, &["artifact", "exit_code", "status"], label)?;
    let status = string(capture, "status")?;
    if !matches!(status, "captured" | "unavailable") {
        return Err(format!("{label}.status is invalid"));
    }
    let exit = capture
        .get("exit_code")
        .ok_or_else(|| format!("{label}.exit_code is missing"))?;
    if !exit.is_null() && exit.as_i64().is_none() {
        return Err(format!("{label}.exit_code must be an integer or null"));
    }
    if (status == "captured") != (exit.as_i64() == Some(0)) {
        return Err(format!("{label} status disagrees with its exit code"));
    }
    verify_artifact(&capture["artifact"], &format!("{label}.artifact"))
}

fn verify_report(report: &Value) -> Result<(), String> {
    let root = object(report, "report")?;
    exact_keys(
        root,
        &[
            "air",
            "claims",
            "collector",
            "compiler_artifacts",
            "device",
            "metallib",
            "pipeline_resources",
            "schema",
            "scope",
            "source",
            "status",
            "toolchain",
        ],
        "report",
    )?;
    if string(root, "schema")? != SCHEMA || string(root, "status")? != "captured" {
        return Err("report schema/status is not a captured v1 artifact".to_owned());
    }
    if string(root, "scope")?
        != "rvLLM artifact evidence only; contains no MLX trace or performance claim"
    {
        return Err("report scope is not the fail-closed rvLLM-only scope".to_owned());
    }
    for key in ["source", "air", "metallib", "collector"] {
        verify_artifact(&root[key], key)?;
    }

    let toolchain = object(&root["toolchain"], "toolchain")?;
    exact_keys(
        toolchain,
        &[
            "compile_argv",
            "compile_capture",
            "link_argv",
            "link_capture",
            "metal",
            "metal_objdump",
            "metallib",
            "sdk_path",
            "sdk_version",
        ],
        "toolchain",
    )?;
    for key in ["metal", "metallib", "metal_objdump"] {
        let tool = object(&toolchain[key], &format!("toolchain.{key}"))?;
        exact_keys(tool, &["identity", "version"], &format!("toolchain.{key}"))?;
        verify_artifact(&tool["identity"], &format!("toolchain.{key}.identity"))?;
        if string(tool, "version")?.is_empty() {
            return Err(format!("toolchain.{key}.version is empty"));
        }
    }
    for key in ["compile_argv", "link_argv"] {
        let argv = toolchain[key]
            .as_array()
            .ok_or_else(|| format!("toolchain.{key} must be an array"))?;
        if argv.is_empty() || argv.iter().any(|item| item.as_str().is_none()) {
            return Err(format!("toolchain.{key} must contain strings"));
        }
    }
    for key in ["sdk_path", "sdk_version"] {
        if string(toolchain, key)?.is_empty() {
            return Err(format!("toolchain.{key} is empty"));
        }
    }
    verify_capture(&toolchain["compile_capture"], "toolchain.compile_capture")?;
    verify_capture(&toolchain["link_capture"], "toolchain.link_capture")?;

    let device = object(&root["device"], "device")?;
    exact_keys(device, &["identity_sha256", "public_properties"], "device")?;
    let device_digest = string(device, "identity_sha256")?;
    digest(device_digest, "device.identity_sha256")?;
    let canonical = serde_json::to_vec(&device["public_properties"])
        .map_err(|e| format!("serialize device properties: {e}"))?;
    if format!("{:x}", Sha256::digest(&canonical)) != device_digest.to_ascii_lowercase() {
        return Err("device identity does not match its public properties".to_owned());
    }

    let resources = root["pipeline_resources"]
        .as_array()
        .ok_or_else(|| "pipeline_resources must be an array".to_owned())?;
    if resources.is_empty() {
        return Err("pipeline_resources must not be empty".to_owned());
    }
    let mut kernels = BTreeSet::new();
    for (index, resource) in resources.iter().enumerate() {
        let resource = object(resource, &format!("pipeline_resources[{index}]"))?;
        exact_keys(
            resource,
            &[
                "kernel",
                "max_total_threads_per_threadgroup",
                "provenance",
                "static_threadgroup_memory_bytes",
                "thread_execution_width",
            ],
            &format!("pipeline_resources[{index}]"),
        )?;
        let kernel = string(resource, "kernel")?;
        if kernel.is_empty() || !kernels.insert(kernel) {
            return Err("pipeline kernel names must be nonempty and unique".to_owned());
        }
        for key in [
            "thread_execution_width",
            "max_total_threads_per_threadgroup",
        ] {
            if resource.get(key).and_then(Value::as_u64).unwrap_or(0) == 0 {
                return Err(format!("pipeline {kernel} has invalid {key}"));
            }
        }
        if resource
            .get("static_threadgroup_memory_bytes")
            .and_then(Value::as_u64)
            .is_none()
        {
            return Err(format!("pipeline {kernel} has invalid static memory"));
        }
        if string(resource, "provenance")?
            != "public MTLComputePipelineState getters on the sealed metallib"
        {
            return Err(format!("pipeline {kernel} has unknown provenance"));
        }
    }

    let compiler = object(&root["compiler_artifacts"], "compiler_artifacts")?;
    exact_keys(
        compiler,
        &["build_tables", "disassembly", "interpretation"],
        "compiler_artifacts",
    )?;
    verify_capture(&compiler["build_tables"], "compiler_artifacts.build_tables")?;
    verify_capture(&compiler["disassembly"], "compiler_artifacts.disassembly")?;

    let claims = object(&root["claims"], "claims")?;
    exact_keys(
        claims,
        &[
            "low_bit_unpack_machine_arithmetic",
            "occupancy",
            "register_count",
            "register_residency",
            "simd_matrix_machine_lowering",
        ],
        "claims",
    )?;
    for key in ["register_count", "register_residency", "occupancy"] {
        let claim = object(&claims[key], &format!("claims.{key}"))?;
        exact_keys(claim, &["reason", "status"], &format!("claims.{key}"))?;
        if string(claim, "status")? != "unavailable" || string(claim, "reason")?.is_empty() {
            return Err(format!("claims.{key} is not fail-closed"));
        }
    }
    for key in [
        "simd_matrix_machine_lowering",
        "low_bit_unpack_machine_arithmetic",
    ] {
        let claim = object(&claims[key], &format!("claims.{key}"))?;
        exact_keys(claim, &["reason", "status"], &format!("claims.{key}"))?;
        if string(claim, "status")? != "unverified" || string(claim, "reason")?.is_empty() {
            return Err(format!("claims.{key} is not fail-closed"));
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or_else(|| "usage: rvllm_metal_artifact_evidence_verify <evidence.json>".to_owned())?;
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }
    let bytes =
        std::fs::read(&path).map_err(|e| format!("read {}: {e}", Path::new(&path).display()))?;
    let report: Value = parse_strict_json(&bytes).map_err(|e| format!("strict JSON: {e}"))?;
    verify_report(&report)?;
    println!("verified {}", Path::new(&path).display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_keys_reject_extension_fields() {
        let object = serde_json::json!({"a":1,"b":2});
        assert!(exact_keys(object.as_object().unwrap(), &["a"], "sample").is_err());
    }

    #[test]
    fn digest_shape_is_strict() {
        assert!(digest(&"a".repeat(64), "test").is_ok());
        assert!(digest(&"g".repeat(64), "test").is_err());
        assert!(digest(&"a".repeat(63), "test").is_err());
    }

    #[test]
    fn duplicate_json_is_rejected_before_validation() {
        assert!(parse_strict_json::<Value>(br#"{"schema":"a","schema":"b"}"#).is_err());
    }
}
