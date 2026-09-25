#![forbid(unsafe_code)]
use serde::Deserialize;
use std::{collections::BTreeSet, env, fs};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    status: String,
    candidate: String,
    requested_tokens: u32,
    default_off: bool,
    qkv_boundary: String,
    output_projection_boundary: String,
    generated: Generated,
    tensorops: TensorOps,
    cases: Vec<Case>,
    scope: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Generated {
    generator_version: String,
    entrypoint: String,
    source_sha256: String,
    executable_sha256: String,
    compiler_artifacts: String,
    pipeline_resources: PipelineResources,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineResources {
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    static_threadgroup_memory_bytes: u64,
    provenance: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorOps {
    status: String,
    reason: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    label: String,
    tokens: u32,
    head_dim: u32,
    kv_heads: u32,
    window: u32,
    contexts: Vec<u32>,
    starts: Vec<u32>,
    commands: u32,
    guards_unchanged: bool,
    repeatable_output_bits: bool,
    relative_l2_vs_scalar: f64,
    max_abs_vs_scalar: f64,
    sampled_fp64_relative_l2: f64,
    sampled_fp64_max_abs: f64,
    gpu_ms: Timings,
    wall_ms: Timings,
    gpu_median_ratio: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Timings {
    existing_simd_control: Vec<f64>,
    tiled_candidate: Vec<f64>,
}

fn hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let receipt = args.next().ok_or("receipt path required")?;
    let tokens: u32 = args.next().ok_or("token count required")?.parse()?;
    let executable = args.next().ok_or("executable sha256 required")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let r: Receipt = serde_json::from_slice(&fs::read(receipt)?)?;
    if r.schema != "rvllm.gemma4.metal_prefill_referee.v1"
        || r.status != "qualified"
        || r.candidate != "research_gemma4_prefill_tiled_online_bf16"
        || r.requested_tokens != tokens
        || !r.default_off
        || r.qkv_boundary != "external_bf16"
        || r.output_projection_boundary != "external_bf16"
        || r.generated.generator_version != "gemma4-prefill-online-v1"
        || r.generated.entrypoint != r.candidate
        || !hash(&r.generated.source_sha256)
        || r.generated.executable_sha256 != executable
        || !hash(&executable)
        || r.generated.compiler_artifacts
            != "jit-library; offline AIR/metallib/disassembly required separately"
        || r.generated.pipeline_resources.thread_execution_width != 32
        || r.generated
            .pipeline_resources
            .max_total_threads_per_threadgroup
            < 32
        || r.generated
            .pipeline_resources
            .static_threadgroup_memory_bytes
            > 32768
        || r.generated.pipeline_resources.provenance.is_empty()
        || r.tensorops.status != "unsupported"
        || r.tensorops.reason.is_empty()
        || r.scope.is_empty()
    {
        return Err("receipt identity or boundary contract failed".into());
    }
    let labels: BTreeSet<_> = r.cases.iter().map(|c| c.label.as_str()).collect();
    for required in [
        "first-hole",
        "middle-hole",
        "last-hole",
        "requested-boundary",
    ] {
        if !labels.contains(required) {
            return Err(format!("missing {required}").into());
        }
    }
    for c in &r.cases {
        if c.tokens == 0
            || !matches!(c.head_dim, 256 | 512)
            || c.kv_heads == 0
            || c.contexts.is_empty()
            || c.starts.len() != c.contexts.len()
            || c.commands < 15
            || !c.guards_unchanged
            || !c.repeatable_output_bits
            || !c.sampled_fp64_relative_l2.is_finite()
            || c.sampled_fp64_relative_l2 >= 0.004
            || !c.sampled_fp64_max_abs.is_finite()
            || c.sampled_fp64_max_abs >= 0.01
            || !c.relative_l2_vs_scalar.is_finite()
            || c.relative_l2_vs_scalar >= 0.003
            || !c.max_abs_vs_scalar.is_finite()
            || c.max_abs_vs_scalar >= 0.032
            || c.gpu_ms.existing_simd_control.len() != 6
            || c.gpu_ms.tiled_candidate.len() != 6
            || c.wall_ms.existing_simd_control.len() != 6
            || c.wall_ms.tiled_candidate.len() != 6
            || !c.gpu_median_ratio.is_finite()
            || c.gpu_median_ratio <= 0.0
            || c.window > 1024
        {
            return Err(format!("case {} failed", c.label).into());
        }
    }
    println!(
        "{{\"validated\":true,\"tokens\":{tokens},\"cases\":{}}}",
        r.cases.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hashes_are_exact() {
        assert!(hash(&"a".repeat(64)));
        assert!(!hash(&"g".repeat(64)));
        assert!(!hash(&"a".repeat(63)));
    }
}
