#![forbid(unsafe_code)]
use serde::Deserialize;
use std::{collections::BTreeSet, env, fs};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    status: String,
    candidates: Vec<String>,
    requested_tokens: u32,
    default_off: bool,
    qkv_boundary: String,
    output_projection_boundary: String,
    generated: Generated,
    tensorops: TensorOpsEvidence,
    cases: Vec<Case>,
    scope: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Generated {
    generator_version: String,
    conventional: SourceIdentity,
    tensorops: SourceIdentity,
    executable_sha256: String,
    compiler_artifacts: String,
    pipeline_resources: PipelineResources,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceIdentity {
    entrypoint: String,
    source_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineResources {
    conventional: PipelineResource,
    tensorops: PipelineResource,
    queried_max_threadgroup_memory_bytes: usize,
    provenance: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineResource {
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    static_threadgroup_memory_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorOpsEvidence {
    status: String,
    capability_evidence: String,
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
    conventional: Accuracy,
    tensorops: Accuracy,
    gpu_ms: Timings,
    wall_ms: Timings,
    gpu_median_ratio: Ratios,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Accuracy {
    relative_l2_vs_scalar: Option<f64>,
    relative_l2_vs_reference: Option<f64>,
    max_abs_vs_scalar: Option<f64>,
    max_abs_vs_reference: Option<f64>,
    sampled_fp64_relative_l2: f64,
    sampled_fp64_max_abs: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Timings {
    existing_simd_control: Vec<f64>,
    tiled_candidate: Vec<f64>,
    tensorops_candidate: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ratios {
    control_over_conventional: f64,
    control_over_tensorops: f64,
    conventional_over_tensorops: f64,
}

fn hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn positive_six(values: &[f64]) -> bool {
    values.len() == 6 && values.iter().all(|x| x.is_finite() && *x > 0.0)
}

fn accuracy(
    a: &Accuracy,
    l2: Option<f64>,
    max_abs: Option<f64>,
    l2_bound: f64,
    max_bound: f64,
) -> bool {
    let Some(l2) = l2 else { return false };
    let Some(max_abs) = max_abs else { return false };
    l2.is_finite()
        && l2 < l2_bound
        && max_abs.is_finite()
        && max_abs < max_bound
        && a.sampled_fp64_relative_l2.is_finite()
        && a.sampled_fp64_relative_l2 < 0.004
        && a.sampled_fp64_max_abs.is_finite()
        && a.sampled_fp64_max_abs < 0.01
}

fn pipeline_is_credible(resource: &PipelineResource, queried_max: usize) -> bool {
    resource.thread_execution_width == 32
        && resource.max_total_threads_per_threadgroup >= 32
        && resource.static_threadgroup_memory_bytes as usize <= queried_max
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
    let expected_candidates = [
        "research_gemma4_prefill_tiled_online_bf16",
        "research_gemma4_prefill_tensorops_bf16",
    ];
    if r.schema != "rvllm.gemma4.metal_prefill_referee.v2"
        || r.status != "qualified"
        || r.candidates != expected_candidates
        || r.requested_tokens != tokens
        || !r.default_off
        || r.qkv_boundary != "external_bf16"
        || r.output_projection_boundary != "external_bf16"
        || r.generated.generator_version != "gemma4-prefill-online-v2"
        || r.generated.conventional.entrypoint != expected_candidates[0]
        || r.generated.tensorops.entrypoint != expected_candidates[1]
        || !hash(&r.generated.conventional.source_sha256)
        || !hash(&r.generated.tensorops.source_sha256)
        || r.generated.conventional.source_sha256 == r.generated.tensorops.source_sha256
        || r.generated.executable_sha256 != executable
        || !hash(&executable)
        || r.generated.compiler_artifacts
            != "jit-library; offline AIR/metallib/disassembly/resource identity still required before promotion"
        || r.generated.pipeline_resources.queried_max_threadgroup_memory_bytes < 640
        || !pipeline_is_credible(
            &r.generated.pipeline_resources.conventional,
            r.generated.pipeline_resources.queried_max_threadgroup_memory_bytes,
        )
        || !pipeline_is_credible(
            &r.generated.pipeline_resources.tensorops,
            r.generated.pipeline_resources.queried_max_threadgroup_memory_bytes,
        )
        || r.generated.pipeline_resources.provenance.is_empty()
        || r.tensorops.status != "compiled-and-refereed"
        || !r.tensorops.capability_evidence.contains("compiled")
        || r.scope.is_empty()
    {
        return Err("receipt identity, hardware-probe or boundary contract failed".into());
    }

    let labels: BTreeSet<_> = r.cases.iter().map(|c| c.label.as_str()).collect();
    for required in [
        "first-hole",
        "middle-hole",
        "last-hole",
        "requested-boundary",
        "causal-poison-future",
        "sliding-two-chunks",
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
            || c.commands < 23
            || !c.guards_unchanged
            || !c.repeatable_output_bits
            || !accuracy(
                &c.conventional,
                c.conventional.relative_l2_vs_scalar,
                c.conventional.max_abs_vs_scalar,
                0.003,
                0.032,
            )
            || !accuracy(
                &c.tensorops,
                c.tensorops.relative_l2_vs_reference,
                c.tensorops.max_abs_vs_reference,
                0.004,
                0.032,
            )
            || !positive_six(&c.gpu_ms.existing_simd_control)
            || !positive_six(&c.gpu_ms.tiled_candidate)
            || !positive_six(&c.gpu_ms.tensorops_candidate)
            || !positive_six(&c.wall_ms.existing_simd_control)
            || !positive_six(&c.wall_ms.tiled_candidate)
            || !positive_six(&c.wall_ms.tensorops_candidate)
            || ![
                c.gpu_median_ratio.control_over_conventional,
                c.gpu_median_ratio.control_over_tensorops,
                c.gpu_median_ratio.conventional_over_tensorops,
            ]
            .into_iter()
            .all(|x| x.is_finite() && x > 0.0)
            || c.window > 1024
        {
            return Err(format!("case {} failed", c.label).into());
        }
    }

    println!(
        "{{\"validated\":true,\"schema\":\"v2\",\"tokens\":{tokens},\"cases\":{}}}",
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

    #[test]
    fn timing_vectors_are_exact_positive_sixes() {
        assert!(positive_six(&[1.0; 6]));
        assert!(!positive_six(&[1.0; 5]));
        assert!(!positive_six(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]));
    }
}
