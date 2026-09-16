//! One-layer, same-weights comparison; no full-model route changes.
#![forbid(unsafe_code)]

use super::*;
use rvllm_apple::ane_linear::compile_budget_used;
use rvllm_runtime::apple_measurement::{compare_phase_measurements, PowerMonitor};
use serde_json::Value;

#[derive(Clone, Copy)]
pub(super) enum Variant {
    Palette,
    Stacked,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Palette => "palette",
            Self::Stacked => "stacked",
        }
    }

    fn source_bytes(self, weights: &AneInt8FfnWeights) -> Result<usize, String> {
        match self {
            Self::Palette => weights.lut8_source_blob_bytes(),
            Self::Stacked => weights.stacked_source_blob_bytes(),
        }
    }
}

pub(super) fn run(
    weights: &AneInt8FfnWeights,
    dense: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    policy: AneProgramCachePolicy,
    layer: usize,
    reconstructed_hash: &str,
    input_sources: &[Value],
    variant: Variant,
) -> Result<Value, String> {
    let mut affine = AneGatedFfn::compile_int8_with_cache_policy(
        weights,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut candidate = match variant {
        Variant::Palette => AneGatedFfn::compile_int8_palette_with_cache_policy(weights, policy)?,
        Variant::Stacked => AneGatedFfn::compile_int8_stacked_with_cache_policy(weights, policy)?,
    };
    let mut control = AneGatedFfn::compile_with_cache_policy(
        &dense[0],
        &dense[1],
        &dense[2],
        HIDDEN,
        INTERMEDIATE,
        policy,
    )?;
    let mut outputs: [Vec<f16>; 3] = std::array::from_fn(|_| vec![f16::ZERO; HIDDEN]);
    let mut errors = Vec::new();
    let mut candidate_matches_affine = true;
    let mut candidate_matches_dense = true;
    for input in inputs {
        let reference = cpu_ffn(dense, input);
        affine.project(input, &mut outputs[0])?;
        candidate.project(input, &mut outputs[1])?;
        control.project(input, &mut outputs[2])?;
        let mut statistics: [ErrorStats; 3] = std::array::from_fn(|_| ErrorStats::default());
        for (output, stats) in outputs.iter().zip(&mut statistics) {
            for (actual, expected) in output.iter().zip(&reference.output) {
                stats.observe(actual.to_f32(), *expected)?;
                if (actual.to_f32() - expected).abs() > 0.01 + 0.02 * expected.abs() {
                    return Err(format!(
                        "INT8 representation backend mismatch: {actual} versus {expected}"
                    ));
                }
            }
        }
        candidate_matches_affine &= outputs[1]
            .iter()
            .zip(&outputs[0])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        candidate_matches_dense &= outputs[1]
            .iter()
            .zip(&outputs[2])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        errors.push(json!({"affine_candidate_dense":statistics.map(|s|s.report())}));
    }
    let mut trials: Vec<Value> = Vec::new();
    let mut pairs = Vec::new();
    let mut timing_preflight = None;
    let journal_enabled = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some();
    // A durable journal proves driver lifecycle. It is excluded from timing.
    if !journal_enabled {
        let monitor = PowerMonitor::start(None).map_err(|error| error.to_string())?;
        let preflight = monitor.begin().finish(1);
        let can_time = preflight["sampled_controls_eligible"] == true;
        timing_preflight = Some(preflight);
        if can_time {
            let input = inputs.last().ok_or("missing input")?;
            for _ in 0..8 {
                affine.project(input, &mut outputs[0])?;
                candidate.project(input, &mut outputs[1])?;
            }
            let repetitions = 128;
            for block in 0..3 {
                let start = trials.len();
                let order = if block % 2 == 0 {
                    [false, true, true, false]
                } else {
                    [true, false, false, true]
                };
                for use_candidate in order {
                    let phase = monitor.begin();
                    let program = if use_candidate {
                        &mut candidate
                    } else {
                        &mut affine
                    };
                    for _ in 0..repetitions {
                        program.project(input, &mut outputs[1])?;
                    }
                    let measurement = phase.finish(repetitions);
                    let eligible = measurement["sampled_controls_eligible"] == true;
                    trials.push(json!({"candidate":use_candidate,"measurement":measurement}));
                    if !eligible {
                        break;
                    }
                }
                if trials.len() - start != 4 {
                    break;
                }
                for (a, b) in [(start, start + 1), (start + 3, start + 2)] {
                    let (baseline, candidate) = if trials[a]["candidate"] == false {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    let comparison = compare_phase_measurements(
                        &trials[baseline]["measurement"],
                        &trials[candidate]["measurement"],
                    );
                    pairs.push(match comparison {
                    Ok(comparison) => json!({"baseline":baseline,"candidate":candidate,"eligible":true,"comparison":comparison}),
                    Err(reason) => json!({"baseline":baseline,"candidate":candidate,"eligible":false,"reason":reason}),
                });
                }
                if trials
                    .last()
                    .is_some_and(|trial| trial["measurement"]["sampled_controls_eligible"] != true)
                {
                    break;
                }
            }
        }
    }
    drop(control);
    drop(candidate);
    drop(affine);
    Ok(
        json!({"schema":"rvllm.exact_int8_layout_ane.v1","layer":layer,"candidate_layout":variant.name(),
        "reconstructed_fp16_sha256":reconstructed_hash,"exact_weight_parity":true,
        "affine_source_bytes":weights.source_blob_bytes(),"candidate_source_bytes":variant.source_bytes(weights)?,
        "input_sources":input_sources,"per_input_backend_errors":errors,
        "candidate_outputs_bit_identical_to_affine":candidate_matches_affine,
        "candidate_outputs_bit_identical_to_dense":candidate_matches_dense,
        "compiler_calls":compile_budget_used(),"driver_journal_enabled":journal_enabled,
        "timing_preflight":timing_preflight,"trials":trials,"pairs":pairs,"models_dropped":3,
        "claim":"One FFN with identical reconstructed INT8 weights and FP16 activations. No full-model speedup or physical bandwidth claim. Timing pairs require eligible identical sampled controls; CPU cycles exclude ANE."}),
    )
}
