//! Reproducible, fail-closed evidence for an offline-compiled Metal artifact.
//!
//! This tool deliberately records public compiler output and public PSO
//! metadata without interpreting either as proof of register allocation,
//! occupancy, or a particular machine-instruction lowering.

#![forbid(unsafe_code)]

use objc2_metal::{MTLComputePipelineState, MTLDevice};
use rvllm_apple_metal::MetalContext;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SCHEMA: &str = "rvllm.metal_artifact_evidence.v1";

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file =
        fs::File::create_new(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    file.sync_all()
        .map_err(|e| format!("sync {}: {e}", path.display()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_identity(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(json!({
        "path": path.to_string_lossy(),
        "bytes": bytes.len(),
        "sha256": sha256(&bytes),
    }))
}

fn run(mut command: Command, label: &str) -> Result<Output, String> {
    command
        .output()
        .map_err(|e| format!("execute {label}: {e}"))
}

fn successful_stdout(command: Command, label: &str) -> Result<String, String> {
    let output = run(command, label)?;
    if !output.status.success() {
        return Err(format!(
            "{label} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn xcrun_find(tool: &str) -> Result<PathBuf, String> {
    let mut command = Command::new("xcrun");
    command.args(["--toolchain", "Metal", "--find", tool]);
    let path = successful_stdout(command, &format!("locate {tool}"))?;
    let path = PathBuf::from(path);
    if !path.is_absolute() {
        return Err(format!("xcrun returned a non-absolute {tool} path"));
    }
    Ok(path)
}

fn tool_version(path: &Path) -> Result<String, String> {
    let mut command = Command::new(path);
    command.arg("--version");
    let output = run(command, &format!("{} --version", path.display()))?;
    if !output.status.success() {
        return Err(format!("{} --version failed", path.display()));
    }
    let mut bytes = output.stdout;
    bytes.extend_from_slice(&output.stderr);
    Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
}

fn write_capture(path: &Path, output: &Output) -> Result<Value, String> {
    let mut bytes = output.stdout.clone();
    if !output.stderr.is_empty() {
        bytes.extend_from_slice(b"\n--- stderr ---\n");
        bytes.extend_from_slice(&output.stderr);
    }
    write_new(path, &bytes)?;
    Ok(json!({
        "status": if output.status.success() { "captured" } else { "unavailable" },
        "exit_code": output.status.code(),
        "artifact": file_identity(path)?,
    }))
}

fn unavailable_claims() -> Value {
    let reason = "not exposed by the public Metal compiler or MTLComputePipelineState API; raw compiler artifacts are not interpreted as a stable semantic ISA contract";
    json!({
        "register_count": {"status":"unavailable", "reason":reason},
        "register_residency": {"status":"unavailable", "reason":reason},
        "occupancy": {"status":"unavailable", "reason":reason},
        "simd_matrix_machine_lowering": {"status":"unverified", "reason":reason},
        "low_bit_unpack_machine_arithmetic": {"status":"unverified", "reason":reason},
    })
}

fn usage() -> String {
    "usage: rvllm-metal-artifact-evidence <SOURCE.metal> <FRESH-OUTPUT-DIR> [--compiled-build BUILD.json] <KERNEL>...".to_owned()
}

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(args.next().ok_or_else(usage)?);
    let output_dir = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut remaining: Vec<_> = args.collect();
    let compiled_build = if remaining.first().is_some_and(|s| s == "--compiled-build") {
        if remaining.len() < 3 {
            return Err(usage());
        }
        remaining.remove(0);
        Some(PathBuf::from(remaining.remove(0)))
    } else {
        None
    };
    let kernels: Vec<String> = remaining
        .into_iter()
        .map(|arg| {
            arg.into_string()
                .map_err(|_| "kernel name is not UTF-8".to_owned())
        })
        .collect::<Result<_, _>>()?;
    if kernels.is_empty() || kernels.iter().any(|name| name.is_empty()) {
        return Err(usage());
    }
    fs::create_dir(&output_dir).map_err(|e| format!("create {}: {e}", output_dir.display()))?;

    let source_identity = file_identity(&source)?;
    let metal = xcrun_find("metal")?;
    let metallib_tool = xcrun_find("metallib")?;
    let objdump = xcrun_find("metal-objdump")?;
    let sdk_path = {
        let mut command = Command::new("xcrun");
        command.args(["--sdk", "macosx", "--show-sdk-path"]);
        successful_stdout(command, "resolve macOS SDK")?
    };
    let sdk_version = {
        let mut command = Command::new("xcrun");
        command.args(["--sdk", "macosx", "--show-sdk-version"]);
        successful_stdout(command, "resolve macOS SDK version")?
    };

    let air = output_dir.join("artifact.air");
    let metallib = output_dir.join("artifact.metallib");
    let compile_log = output_dir.join("metal-compile.log");
    let link_log = output_dir.join("metallib-link.log");

    // Reuse the actual queue artifact without recompiling. A separate
    // recompilation is never presented as evidence about the timed metallib.
    let (compile_capture, link_capture, sealed_build) = if let Some(path) = compiled_build {
        let build: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let directory = path.parent().ok_or("build receipt has no parent")?;
        let original_air = directory.join("kernel.air");
        let original_library = directory.join("kernel.metallib");
        if build["schema"] != "rvllm.global-decode.build.v1"
            || build["status"] != "compiled"
            || build["flags"] != json!(["-std=metal3.1", "-fno-fast-math"])
            || build["source_sha256"] != source_identity["sha256"]
            || build["air_sha256"] != file_identity(&original_air)?["sha256"]
            || build["metallib_sha256"] != file_identity(&original_library)?["sha256"]
        {
            return Err("sealed queue build/source/AIR/metallib identity mismatch".into());
        }
        write_new(&air, &fs::read(&original_air).map_err(|e| e.to_string())?)?;
        write_new(
            &metallib,
            &fs::read(&original_library).map_err(|e| e.to_string())?,
        )?;
        if file_identity(&air)?["sha256"] != build["air_sha256"]
            || file_identity(&metallib)?["sha256"] != build["metallib_sha256"]
        {
            return Err("artifact changed while copying; preserve failed evidence".into());
        }
        (
            json!({"status":"reused-not-recompiled"}),
            json!({"status":"reused-not-relinked"}),
            json!({"receipt":file_identity(&path)?,"build":build,
                "original_air":file_identity(&original_air)?,
                "original_metallib":file_identity(&original_library)?}),
        )
    } else {
        let mut compile = Command::new(&metal);
        compile.args(["-std=metal3.1", "-fno-fast-math", "-c"]);
        compile.arg(&source).arg("-o").arg(&air);
        let compile_output = run(compile, "metal compile")?;
        let compile_capture = write_capture(&compile_log, &compile_output)?;
        if !compile_output.status.success() {
            return Err(format!(
                "Metal compilation failed; see {}",
                compile_log.display()
            ));
        }

        let mut link = Command::new(&metallib_tool);
        link.arg(&air).arg("-o").arg(&metallib);
        let link_output = run(link, "metallib link")?;
        let link_capture = write_capture(&link_log, &link_output)?;
        if !link_output.status.success() {
            return Err(format!("metallib link failed; see {}", link_log.display()));
        }

        (compile_capture, link_capture, Value::Null)
    };

    let build_tables_path = output_dir.join("metal-objdump-build-tables.txt");
    let mut build_tables_command = Command::new(&objdump);
    build_tables_command.args(["--metallib", "--build-table=all"]);
    build_tables_command.arg(&metallib);
    let build_tables = write_capture(
        &build_tables_path,
        &run(build_tables_command, "metal-objdump build tables")?,
    )?;

    let disassembly_path = output_dir.join("metal-objdump-disassembly.txt");
    let mut disassembly_command = Command::new(&objdump);
    disassembly_command.args(["--metallib", "--disassemble"]);
    disassembly_command.arg(&metallib);
    let disassembly = write_capture(
        &disassembly_path,
        &run(disassembly_command, "metal-objdump disassembly")?,
    )?;

    let mut context = MetalContext::new().map_err(|e| e.to_string())?;
    context
        .load_metallib(&metallib)
        .map_err(|e| e.to_string())?;
    let mut pipeline_resources = Vec::with_capacity(kernels.len());
    for name in &kernels {
        let pipeline = context.make_pipeline(name).map_err(|e| {
            format!("requested kernel {name:?} is absent or cannot form a PSO: {e}")
        })?;
        pipeline_resources.push(json!({
            "kernel": name,
            "thread_execution_width": pipeline.threadExecutionWidth(),
            "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup(),
            "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength(),
            "provenance": "public MTLComputePipelineState getters on the sealed metallib",
        }));
    }

    let capabilities = context.capabilities();
    let device_basis = json!({
        "name": context.device().name().to_string(),
        "registry_id": context.device().registryID(),
        "gpu_family": format!("{:?}", capabilities.gpu_family),
        "has_unified_memory": capabilities.has_unified_memory,
        "max_threadgroup_memory_bytes": capabilities.max_threadgroup_memory_length,
        "recommended_max_working_set_bytes": capabilities.recommended_max_working_set_size,
    });
    let device_identity_sha256 = sha256(
        &serde_json::to_vec(&device_basis)
            .map_err(|e| format!("serialize device identity: {e}"))?,
    );
    let report = json!({
        "schema": SCHEMA,
        "status": "captured",
        "scope": "rvLLM artifact evidence only; contains no MLX trace or performance claim",
        "collector": file_identity(
            &std::env::current_exe().map_err(|e| format!("resolve collector executable: {e}"))?
        )?,
        "source": source_identity,
        "sealed_queue_build":sealed_build,
        "artifact_origin":if sealed_build.is_null() {"independent-strict-recompilation"} else {"exact-queue-compiled-artifact"},
        "air": file_identity(&air)?,
        "metallib": file_identity(&metallib)?,
        "toolchain": {
            "metal": {"identity":file_identity(&metal)?, "version":tool_version(&metal)?},
            "metallib": {"identity":file_identity(&metallib_tool)?, "version":tool_version(&metallib_tool)?},
            "metal_objdump": {"identity":file_identity(&objdump)?, "version":tool_version(&objdump)?},
            "sdk_path": sdk_path,
            "sdk_version": sdk_version,
            "compile_argv": if sealed_build.is_null() { json!([metal.to_string_lossy(), "-std=metal3.1", "-fno-fast-math", "-c", source.to_string_lossy(), "-o", air.to_string_lossy()]) } else { sealed_build["build"]["compile_argv"].clone() },
            "link_argv": if sealed_build.is_null() { json!([metallib_tool.to_string_lossy(), air.to_string_lossy(), "-o", metallib.to_string_lossy()]) } else { sealed_build["build"]["link_argv"].clone() },
            "compile_capture": compile_capture,
            "link_capture": link_capture,
        },
        "device": {"identity_sha256":device_identity_sha256, "public_properties":device_basis},
        "pipeline_resources": pipeline_resources,
        "compiler_artifacts": {
            "build_tables": build_tables,
            "disassembly": disassembly,
            "interpretation": "raw public-tool output only; availability does not establish a particular executed machine lowering",
        },
        "claims": unavailable_claims(),
    });
    let report_path = output_dir.join("evidence.json");
    if file_identity(&source)? != source_identity {
        return Err("source changed during evidence capture".into());
    }
    write_new(
        &report_path,
        &serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?,
    )?;
    println!("{}", report_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_semantic_claims_fail_closed() {
        let claims = unavailable_claims();
        assert_eq!(claims["register_residency"]["status"], "unavailable");
        assert_eq!(
            claims["simd_matrix_machine_lowering"]["status"],
            "unverified"
        );
        assert_eq!(
            claims["low_bit_unpack_machine_arithmetic"]["status"],
            "unverified"
        );
        assert!(claims.to_string().contains("public Metal"));
    }

    #[test]
    fn digest_is_lowercase_sha256() {
        assert_eq!(
            sha256(b"rvllm"),
            "e5781ccf12ac8e7217e56901da572584c8b49a93d0a3e52c80c7f7d46aa533c4"
        );
    }
}
