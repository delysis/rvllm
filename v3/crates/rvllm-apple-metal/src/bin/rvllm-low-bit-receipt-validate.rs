//! Fail-closed validator for native-BF16 low-bit real-weight receipts.

use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeSet, env, error::Error, fs, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}

#[derive(Debug)]
struct Args {
    receipt: PathBuf,
    tensor: String,
    role: String,
    samples: usize,
    candidate: String,
    format: Option<String>,
    m: Option<usize>,
    order: Option<String>,
}

fn args() -> Result<Args> {
    let mut receipt = None;
    let mut tensor = None;
    let mut role = None;
    let mut samples = None;
    let mut candidate = None;
    let mut format = None;
    let mut m = None;
    let mut order = None;
    let mut a = env::args().skip(1);
    while let Some(flag) = a.next() {
        let value = a.next().ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--receipt" => receipt = Some(value.into()),
            "--tensor" => tensor = Some(value),
            "--role" => role = Some(value),
            "--samples" => samples = Some(value.parse()?),
            "--candidate" => candidate = Some(value),
            "--format" => format = Some(value),
            "--m" => m = Some(value.parse()?),
            "--order" => order = Some(value),
            _ => return fail(format!("unknown option {flag:?}")),
        }
    }
    Ok(Args {
        receipt: receipt.ok_or("--receipt is required")?,
        tensor: tensor.ok_or("--tensor is required")?,
        role: role.ok_or("--role is required")?,
        samples: samples.ok_or("--samples is required")?,
        candidate: candidate.ok_or("--candidate is required")?,
        format,
        m,
        order,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    claim: String,
    model_dir: PathBuf,
    config_sha256: String,
    tensor: String,
    tensor_role: String,
    source_dtype: String,
    shape: Vec<usize>,
    source_file: PathBuf,
    source_file_offset: usize,
    source_tensor_bytes: usize,
    source_tensor_sha256: String,
    abi: Abi,
    candidate_schedule: String,
    direct_order: Option<String>,
    conditions_policy: Option<String>,
    generated_msl_sha256: String,
    executable_sha256: String,
    compile_counts: CompileCounts,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Abi {
    activation: String,
    output: String,
    scales: String,
    accumulation: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompileCounts {
    metal_libraries: usize,
    pipeline_states: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    m: usize,
    n: usize,
    k: usize,
    accuracy: Accuracy,
    native_accuracy: Option<Accuracy>,
    guard_unchanged: bool,
    repeatable_output_bits: bool,
    cross_schedule_output_bits_equal: Option<bool>,
    identity: Identity,
    dispatch: Dispatch,
    timing: Timing,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Accuracy {
    max_abs: f64,
    relative_l2: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    packed_values_sha256: String,
    scales_f16le_sha256: String,
    activations_bf16le_sha256: String,
    cpu_low_bit_reference_bf16le_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dispatch {
    format: String,
    role: String,
    candidate_abi: Option<String>,
    exact_correctness_dispatches_verified: usize,
    exact_timing_dispatch_count_verified: bool,
    timing_dispatches: Option<usize>,
    timing_dispatches_per_schedule: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Timing {
    method: String,
    blocks: Option<usize>,
    samples_per_arm: usize,
    native_kernel: Option<String>,
    native_weight_conversion: Option<String>,
    candidate_kernel: Option<String>,
    n4_kernel: Option<String>,
    n8_kernel: Option<String>,
    activation_dtype: String,
    output_dtype: String,
    scale_dtype: String,
    accumulation_dtype: String,
    native_ms: Option<Vec<f64>>,
    candidate_ms: Option<Vec<f64>>,
    n4_ms: Option<Vec<f64>>,
    n8_ms: Option<Vec<f64>>,
    native_median_ms: Option<f64>,
    candidate_median_ms: Option<f64>,
    n4_median_ms: Option<f64>,
    n8_median_ms: Option<f64>,
    speedup: Option<f64>,
    n4_over_n8_speedup: Option<f64>,
    n8_over_n4_speedup: Option<f64>,
}

fn finite_nonnegative(a: &Accuracy) -> bool {
    a.max_abs.is_finite() && a.max_abs >= 0.0 && a.relative_l2.is_finite() && a.relative_l2 >= 0.0
}
fn hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn positive(values: &[f64], count: usize) -> bool {
    values.len() == count && values.iter().all(|v| v.is_finite() && *v > 0.0)
}
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    Some(ordered[ordered.len() / 2])
}
fn nearly_equal(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-12_f64.max(expected.abs() * 1e-9)
}
fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        fail(message)
    }
}

fn validate(r: &Receipt, a: &Args) -> Result<()> {
    require(
        r.schema == "rvllm.metal_low_bit_real_weight_bf16.v1",
        "bad schema",
    )?;
    require(r.claim == "real checkpoint projection operator evidence only; not full-route or model-quality evidence", "wrong claim boundary")?;
    require(
        !r.model_dir.as_os_str().is_empty() && !r.source_file.as_os_str().is_empty(),
        "empty source path",
    )?;
    require(
        r.shape.len() == 2 && r.shape.iter().all(|v| *v > 0) && r.source_tensor_bytes > 0,
        "invalid source shape",
    )?;
    require(r.source_file_offset > 0, "invalid source offset")?;
    require(
        r.tensor == a.tensor && r.tensor_role == a.role,
        "tensor or role mismatch",
    )?;
    require(r.source_dtype == "Bf16", "source was not BF16")?;
    require(
        r.abi.activation == "BF16"
            && r.abi.output == "BF16"
            && r.abi.scales == "F16"
            && r.abi.accumulation == "F32",
        "wrong ABI",
    )?;
    require(
        r.candidate_schedule == a.candidate,
        "wrong candidate schedule",
    )?;
    for value in [
        &r.config_sha256,
        &r.source_tensor_sha256,
        &r.generated_msl_sha256,
        &r.executable_sha256,
    ] {
        require(hash(value), "invalid receipt hash")?;
    }
    let direct = a.candidate == "n4-vs-n8";
    require(
        r.compile_counts.metal_libraries == 1,
        "wrong Metal library count",
    )?;
    require(
        r.compile_counts.pipeline_states == if direct { 4 } else { 3 },
        "wrong pipeline count",
    )?;
    require(
        !direct || r.conditions_policy.as_deref() == Some("observed externally; never a wait gate"),
        "wrong conditions policy",
    )?;
    let expected: BTreeSet<(String, usize)> = if direct {
        let f = a
            .format
            .clone()
            .ok_or("direct validation requires --format")?;
        let m = a.m.ok_or("direct validation requires --m")?;
        let o = a
            .order
            .as_deref()
            .ok_or("direct validation requires --order")?;
        require(
            matches!(f.as_str(), "w4a16" | "w8a16")
                && matches!(m, 1 | 4)
                && matches!(o, "ABBA" | "BAAB"),
            "invalid direct selector",
        )?;
        require(r.direct_order.as_deref() == Some(o), "wrong direct order")?;
        [(f, m)].into_iter().collect()
    } else {
        require(
            matches!(a.candidate.as_str(), "scalar" | "n4" | "n8" | "vector"),
            "invalid legacy candidate",
        )?;
        [
            ("w4a16".into(), 1),
            ("w4a16".into(), 4),
            ("w8a16".into(), 1),
            ("w8a16".into(), 4),
        ]
        .into_iter()
        .collect()
    };
    require(r.cases.len() == expected.len(), "wrong case count")?;
    let mut seen = BTreeSet::new();
    for c in &r.cases {
        require(
            c.n == r.shape[0] && c.k == r.shape[1],
            "case/source shape mismatch",
        )?;
        let key = (c.dispatch.format.clone(), c.m);
        require(
            expected.contains(&key) && seen.insert(key.clone()),
            "unexpected or duplicate case",
        )?;
        require(
            c.dispatch.role == a.role && c.guard_unchanged && c.repeatable_output_bits,
            "case invariant failed",
        )?;
        require(finite_nonnegative(&c.accuracy), "invalid accuracy")?;
        for h in [
            &c.identity.packed_values_sha256,
            &c.identity.scales_f16le_sha256,
            &c.identity.activations_bf16le_sha256,
            &c.identity.cpu_low_bit_reference_bf16le_sha256,
        ] {
            require(hash(h), "invalid case hash")?;
        }
        require(
            c.dispatch.exact_timing_dispatch_count_verified,
            "timing dispatch unverified",
        )?;
        let count = 2 * a.samples;
        require(c.timing.samples_per_arm == count, "wrong sample count")?;
        require(
            c.timing.activation_dtype == "BF16"
                && c.timing.output_dtype == "BF16"
                && c.timing.scale_dtype == "F16"
                && c.timing.accumulation_dtype == "F32",
            "wrong timing ABI",
        )?;
        if direct {
            require(
                c.native_accuracy.is_none() && c.cross_schedule_output_bits_equal == Some(true),
                "direct equality invariant failed",
            )?;
            require(
                c.dispatch.candidate_abi.is_none()
                    && c.dispatch.exact_correctness_dispatches_verified == 4
                    && c.dispatch.timing_dispatches.is_none()
                    && c.dispatch.timing_dispatches_per_schedule == Some(count),
                "wrong direct dispatch",
            )?;
            require(
                c.timing.method
                    == format!(
                        "{} wall-clock commit-to-completion",
                        a.order.as_deref().unwrap()
                    )
                    && c.timing.blocks == Some(a.samples),
                "wrong direct timing method",
            )?;
            let kernel_prefix = if c.dispatch.format == "w4a16" {
                "experimental_projection_w4abf16_bf16"
            } else {
                "experimental_projection_w8abf16_bf16"
            };
            require(
                c.timing.n4_kernel.as_deref() == Some(&format!("{kernel_prefix}_n4"))
                    && c.timing.n8_kernel.as_deref() == Some(&format!("{kernel_prefix}_n8")),
                "wrong direct kernels",
            )?;
            require(
                c.timing.native_kernel.is_none()
                    && c.timing.native_weight_conversion.is_none()
                    && c.timing.candidate_kernel.is_none()
                    && c.timing.native_ms.is_none()
                    && c.timing.candidate_ms.is_none()
                    && c.timing.native_median_ms.is_none()
                    && c.timing.candidate_median_ms.is_none()
                    && c.timing.speedup.is_none(),
                "foreign legacy timing fields",
            )?;
            let n4_samples = c.timing.n4_ms.as_deref().unwrap_or(&[]);
            let n8_samples = c.timing.n8_ms.as_deref().unwrap_or(&[]);
            require(
                positive(n4_samples, count) && positive(n8_samples, count),
                "invalid direct samples",
            )?;
            let n4_median = median(n4_samples).ok_or("invalid N4 median")?;
            let n8_median = median(n8_samples).ok_or("invalid N8 median")?;
            require(
                c.timing
                    .n4_median_ms
                    .is_some_and(|value| nearly_equal(value, n4_median))
                    && c.timing
                        .n8_median_ms
                        .is_some_and(|value| nearly_equal(value, n8_median))
                    && c.timing
                        .n4_over_n8_speedup
                        .is_some_and(|value| nearly_equal(value, n8_median / n4_median))
                    && c.timing
                        .n8_over_n4_speedup
                        .is_some_and(|value| nearly_equal(value, n4_median / n8_median)),
                "direct derived timing does not match samples",
            )?;
        } else {
            require(
                c.native_accuracy.as_ref().is_some_and(finite_nonnegative)
                    && c.cross_schedule_output_bits_equal.is_none(),
                "legacy accuracy invariant failed",
            )?;
            require(
                c.dispatch.exact_correctness_dispatches_verified == 2
                    && c.dispatch.timing_dispatches == Some(1 + count)
                    && c.dispatch.timing_dispatches_per_schedule.is_none(),
                "wrong legacy dispatch",
            )?;
            require(
                c.timing.method == "ABBA wall-clock commit-to-completion"
                    && positive(c.timing.native_ms.as_deref().unwrap_or(&[]), count)
                    && positive(c.timing.candidate_ms.as_deref().unwrap_or(&[]), count),
                "invalid legacy timing",
            )?;
            let suffix = match a.candidate.as_str() {
                "scalar" => "",
                "n4" => "_n4",
                "n8" => "_n8",
                "vector" => {
                    if c.dispatch.format == "w4a16" {
                        "_n4_packed2"
                    } else {
                        "_n8_k4"
                    }
                }
                _ => unreachable!(),
            };
            let expected_kernel = format!(
                "experimental_projection_{}_bf16{suffix}",
                if c.dispatch.format == "w4a16" {
                    "w4abf16"
                } else {
                    "w8abf16"
                }
            );
            require(
                c.timing.candidate_kernel.as_deref() == Some(&expected_kernel)
                    && c.timing.native_kernel.as_deref()
                        == Some("gemm_f16_vec8 compiled as typed BF16")
                    && c.timing.native_weight_conversion.as_deref() == Some("none"),
                "wrong legacy kernels",
            )?;
            require(
                c.timing.blocks.is_none()
                    && c.timing.n4_kernel.is_none()
                    && c.timing.n8_kernel.is_none()
                    && c.timing.n4_ms.is_none()
                    && c.timing.n8_ms.is_none()
                    && c.timing.n4_median_ms.is_none()
                    && c.timing.n8_median_ms.is_none()
                    && c.timing.n4_over_n8_speedup.is_none()
                    && c.timing.n8_over_n4_speedup.is_none(),
                "foreign direct timing fields",
            )?;
            require(
                [
                    c.timing.native_median_ms,
                    c.timing.candidate_median_ms,
                    c.timing.speedup,
                ]
                .into_iter()
                .all(|v| v.is_some_and(|x| x.is_finite() && x > 0.0)),
                "invalid legacy derived timing",
            )?;
        }
    }
    require(seen == expected, "case matrix incomplete")?;
    Ok(())
}

fn main() -> Result<()> {
    let a = args()?;
    let bytes = fs::read(&a.receipt)?;
    // Typed deserialization with deny_unknown_fields also rejects duplicate
    // struct keys, preventing last-key-wins receipt ambiguity.
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    validate(&receipt, &a)?;
    println!(
        "{}",
        json!({"validated":true,"tensor":a.tensor,"role":a.role,"cases":receipt.cases.len(),"samples":a.samples,"direct":a.candidate=="n4-vs-n8"})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_top_level_key_fails() {
        let s = r#"{"schema":"x","schema":"y"}"#;
        assert!(serde_json::from_str::<Receipt>(s).is_err());
    }
    #[test]
    fn unknown_top_level_key_fails() {
        let s = r#"{"unknown":1}"#;
        assert!(serde_json::from_str::<Receipt>(s).is_err());
    }
    #[test]
    fn hashes_are_lower_hex() {
        assert!(hash(&"a".repeat(64)));
        assert!(!hash(&"A".repeat(64)));
    }
}
