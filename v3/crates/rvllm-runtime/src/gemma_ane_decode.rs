//! Resident, single-request Gemma 4 12B decode after synchronized Metal prefill.
//! Dense projections and attention execute on ANE. Small normalization, RoPE,
//! residual and sampling operations use safe Rust with explicit FP16 boundaries.
//! This module is research-only and never silently falls back to another device.

#![forbid(unsafe_code)]

use crate::ane_prefill::{AnePrefillSnapshot, PrefillLayerShape};
use crate::gemma_head_ranking::{
    validate_head_ranking_configuration, validate_head_ranking_mode, HeadRankingPlan,
    HeadRankingStats, HeadTop5,
};
#[cfg(test)]
use half::bf16;
use half::f16;
use rvllm_apple::ane_attention::{AneAttention, AneAttentionProgram};
use rvllm_apple::ane_attention_layout::{KvImportPacking, PackedAttentionLayout};
use rvllm_apple::ane_dynamic_ffn::{AneDynamicFfn, AneDynamicFfnProgram};
use rvllm_apple::ane_dynamic_linear::{AneDynamicLinear, AneDynamicLinearProgram};
use rvllm_apple::ane_int8_ffn_weights::{AneInt8FfnWeights, AneInt8LinearWeights};
use rvllm_apple::ane_linear::{AneGatedFfn, AneLinear, AneProgramCachePolicy};
use rvllm_apple::ane_lut4_ffn_weights::AneLut4FfnWeights;
use rvllm_apple::gemma_decode_math::{
    add_residual_f16, rms_norm_f16_in_place, scale_layer_f16, GemmaRope,
};
use rvllm_apple_metal::weight_loader::{scan_safetensor_tensors, SafetensorTensorInfo};
use rvllm_core::{DType, TokenId};
use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

const HIDDEN: usize = 3840;
const INTERMEDIATE: usize = 15360;
const LAYERS: usize = 48;
const VOCAB: usize = 262144;
const HEAD_ROWS: usize = 16384;

#[path = "ane_two_token_reference.rs"]
pub mod two_token_reference;

/// Alternate placement keeps the larger FFN weights compiled into each graph.
/// Both arrangements retain every program and weight allocation during decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneWeightPlan {
    DynamicFfn,
    StaticFfnDynamicOutput,
    /// Every constant program must already exist in this client's daemon cache.
    /// No compile fallback: preparation stops cleanly on the first cache miss.
    StaticAllCached,
    /// Experimental scalar LUT4 FFNs; other weights remain FP16 constants.
    /// Requires separately provisioned entries and full-model quality checks.
    StaticLut4FfnCached,
    /// Experimental per-output-channel INT8 FFNs; other weights remain FP16.
    StaticInt8FfnCached,
    /// Default-off, single-I/O output-channel chunked INT8 FFN.
    StaticInt8Chunk4FfnCached,
    StaticInt8Down4FfnCached,
    StaticInt8InterleavedFfnCached,
    StaticInt8FfnTransposeAttentionCached,
    /// Experimental INT8 sliding QKV and ordinary INT8 FFNs. Global QKV stays FP16.
    StaticInt8FfnSlidingQkvCached,
    /// Four output-row tiles, only for the already-experimental INT8 sliding QKV.
    StaticInt8FfnSlidingQkvTiles4Cached,
    /// Experimental stacked gate/up layout with the same INT8 weights.
    StaticInt8StackedFfnCached,
    /// Qualification only: compare both FFNs on every activation, bit for bit.
    /// Requires strict cache loading and a durable driver journal.
    StaticInt8StackedFfnChecked,
}

impl AneWeightPlan {
    pub fn name(self) -> &'static str {
        match self {
            Self::DynamicFfn => "dynamic-ffn",
            Self::StaticFfnDynamicOutput => "static-ffn",
            Self::StaticAllCached => "static-all-cached",
            Self::StaticLut4FfnCached => "static-lut4-ffn-cached",
            Self::StaticInt8FfnCached => "static-int8-ffn-cached",
            Self::StaticInt8Chunk4FfnCached => "static-int8-chunk4-ffn-cached",
            Self::StaticInt8Down4FfnCached => "static-int8-down4-ffn-cached",
            Self::StaticInt8InterleavedFfnCached => "static-int8-interleaved-ffn-cached",
            Self::StaticInt8FfnTransposeAttentionCached => {
                "static-int8-ffn-transpose-attention-cached"
            }
            Self::StaticInt8FfnSlidingQkvCached => "static-int8-ffn-sliding-qkv-cached",
            Self::StaticInt8FfnSlidingQkvTiles4Cached => {
                "static-int8-ffn-sliding-qkv-tiles4-cached"
            }
            Self::StaticInt8StackedFfnCached => "static-int8-stacked-ffn-cached",
            Self::StaticInt8StackedFfnChecked => "static-int8-stacked-ffn-checked",
        }
    }

    pub fn program_count(self) -> usize {
        match self {
            Self::DynamicFfn => 115,
            Self::StaticFfnDynamicOutput => 116,
            Self::StaticAllCached
            | Self::StaticLut4FfnCached
            | Self::StaticInt8FfnCached
            | Self::StaticInt8Chunk4FfnCached
            | Self::StaticInt8Down4FfnCached
            | Self::StaticInt8InterleavedFfnCached
            | Self::StaticInt8FfnTransposeAttentionCached
            | Self::StaticInt8FfnSlidingQkvCached
            | Self::StaticInt8FfnSlidingQkvTiles4Cached
            | Self::StaticInt8StackedFfnCached => 162,
            Self::StaticInt8StackedFfnChecked => 210,
        }
    }

    pub fn cache_policy(self) -> AneProgramCachePolicy {
        match self {
            Self::StaticAllCached
            | Self::StaticLut4FfnCached
            | Self::StaticInt8FfnCached
            | Self::StaticInt8Chunk4FfnCached
            | Self::StaticInt8Down4FfnCached
            | Self::StaticInt8InterleavedFfnCached
            | Self::StaticInt8FfnTransposeAttentionCached
            | Self::StaticInt8FfnSlidingQkvCached
            | Self::StaticInt8FfnSlidingQkvTiles4Cached
            | Self::StaticInt8StackedFfnCached
            | Self::StaticInt8StackedFfnChecked => AneProgramCachePolicy::RequireExisting,
            Self::DynamicFfn | Self::StaticFfnDynamicOutput => AneProgramCachePolicy::Compile,
        }
    }

    fn static_ffn_precision(self) -> StaticFfnPrecision {
        match self {
            Self::StaticInt8Chunk4FfnCached => StaticFfnPrecision::Int8Chunk4,
            Self::StaticInt8Down4FfnCached => StaticFfnPrecision::Int8Down4,
            Self::StaticInt8InterleavedFfnCached => StaticFfnPrecision::Int8Interleaved,
            Self::StaticInt8FfnTransposeAttentionCached => StaticFfnPrecision::Int8,
            Self::StaticInt8FfnCached
            | Self::StaticInt8FfnSlidingQkvCached
            | Self::StaticInt8FfnSlidingQkvTiles4Cached => StaticFfnPrecision::Int8,
            Self::StaticInt8StackedFfnCached | Self::StaticInt8StackedFfnChecked => {
                StaticFfnPrecision::Int8Stacked
            }
            Self::StaticLut4FfnCached => StaticFfnPrecision::Lut4,
            _ => StaticFfnPrecision::Fp16,
        }
    }

    fn quantizes_qkv(self, sliding: bool) -> bool {
        matches!(
            self,
            Self::StaticInt8FfnSlidingQkvCached | Self::StaticInt8FfnSlidingQkvTiles4Cached
        ) && sliding
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticFfnPrecision {
    Fp16,
    Int8,
    Int8Stacked,
    Int8Chunk4,
    Int8Down4,
    Int8Interleaved,
    Lut4,
}

/// Each provisioning part fits in a fresh process with at most 48 explicit
/// compiler calls. Entries are reused when present; one graph is held at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneStaticCachePart {
    QueryKeyValue,
    /// Forty INT8 sliding projections and the eight original FP16 global projections.
    QueryKeyValueSlidingInt8,
    QueryKeyValueSlidingInt8Tiles4,
    Output,
    FeedForward,
    FeedForwardLut4,
    FeedForwardInt8,
    FeedForwardInt8Stacked,
    FeedForwardInt8Chunk4,
    FeedForwardInt8Down4,
    FeedForwardInt8Interleaved,
    VocabularyAndAttention,
    AttentionTransposeFlags,
}

pub fn provision_static_cache(model_dir: &Path, part: AneStaticCachePart) -> Result<usize, String> {
    provision_static_cache_with_capacity(model_dir, part, 64)
}

pub fn provision_static_cache_with_capacity(
    model_dir: &Path,
    part: AneStaticCachePart,
    capacity: usize,
) -> Result<usize, String> {
    provision_static_cache_with_capacity_until(model_dir, part, capacity, &|| false)
}

/// Stop between graph operations, after any loaded graph has been dropped.
/// Cancellation never interrupts a synchronous private-framework operation.
pub fn provision_static_cache_with_capacity_until(
    model_dir: &Path,
    part: AneStaticCachePart,
    capacity: usize,
    should_stop: &dyn Fn() -> bool,
) -> Result<usize, String> {
    Ok(visit_static_cache(model_dir, part, capacity, false, should_stop)?.len())
}

/// Availability observed by a strict load, followed by owner destruction.
/// Source staging may be recreated, but no compiler call or evaluation occurs.
#[derive(Clone, Debug)]
pub struct AneStaticCacheEntry {
    pub name: String,
    pub available: bool,
}

pub fn inspect_static_cache_with_capacity(
    model_dir: &Path,
    part: AneStaticCachePart,
    capacity: usize,
) -> Result<Vec<AneStaticCacheEntry>, String> {
    inspect_static_cache_with_capacity_until(model_dir, part, capacity, &|| false)
}

pub fn inspect_static_cache_with_capacity_until(
    model_dir: &Path,
    part: AneStaticCachePart,
    capacity: usize,
    should_stop: &dyn Fn() -> bool,
) -> Result<Vec<AneStaticCacheEntry>, String> {
    visit_static_cache(model_dir, part, capacity, true, should_stop)
}

fn record_cache_entry(
    entries: &mut Vec<AneStaticCacheEntry>,
    name: String,
    result: Result<(), String>,
    inspect: bool,
) -> Result<(), String> {
    let available = match result {
        Ok(()) => true,
        Err(error)
            if inspect
                && error == "ANE model is absent from the current client's compiled cache" =>
        {
            false
        }
        Err(error) => return Err(format!("ANE cache {name}: {error}")),
    };
    eprintln!(
        "ANE cache {name}: {}",
        if available { "loaded" } else { "absent" }
    );
    entries.push(AneStaticCacheEntry { name, available });
    Ok(())
}

fn visit_static_cache(
    model_dir: &Path,
    part: AneStaticCachePart,
    capacity: usize,
    inspect: bool,
    should_stop: &dyn Fn() -> bool,
) -> Result<Vec<AneStaticCacheEntry>, String> {
    let check_stop = || {
        if should_stop() {
            Err("ANE cache operation cancelled".to_string())
        } else {
            Ok(())
        }
    };
    check_stop()?;
    let (arch, entries) = validated_weights(model_dir, capacity)?;
    let policy = if inspect {
        AneProgramCachePolicy::RequireExisting
    } else {
        AneProgramCachePolicy::ReuseOrCompileUpTo(
            if part == AneStaticCachePart::AttentionTransposeFlags {
                2
            } else if part == AneStaticCachePart::VocabularyAndAttention {
                2 + VOCAB / HEAD_ROWS
            } else {
                LAYERS
            },
        )
    };
    let mut results = Vec::new();
    if part == AneStaticCachePart::AttentionTransposeFlags {
        for (name, layout) in [
            (
                "attention-transpose/sliding",
                PackedAttentionLayout::sliding(16, 8, 256, 1024)?,
            ),
            (
                "attention-transpose/global",
                PackedAttentionLayout::new(16, 1, 512, capacity)?,
            ),
        ] {
            check_stop()?;
            record_cache_entry(
                &mut results,
                name.into(),
                AneAttentionProgram::compile_transpose_flags_with_cache_policy(layout, policy)
                    .map(drop),
                inspect,
            )?;
        }
        check_stop()?;
        return Ok(results);
    }
    if part == AneStaticCachePart::VocabularyAndAttention {
        check_stop()?;
        record_cache_entry(
            &mut results,
            "attention/sliding".into(),
            AneAttentionProgram::compile_sliding_with_cache_policy(16, 8, 256, 1024, policy)
                .map(drop),
            inspect,
        )?;
        check_stop()?;
        record_cache_entry(
            &mut results,
            "attention/global".into(),
            AneAttentionProgram::compile_with_cache_policy(16, 1, 512, capacity, policy).map(drop),
            inspect,
        )?;
        let embedding = &entries[&format!("{}.embed_tokens.weight", arch.weight_prefix)];
        for first in (0..VOCAB).step_by(HEAD_ROWS) {
            check_stop()?;
            let weights = load_rows(embedding, first, HEAD_ROWS)?;
            record_cache_entry(
                &mut results,
                format!("vocabulary/{first}"),
                AneLinear::compile_with_cache_policy(&weights, HIDDEN, HEAD_ROWS, 1, policy)
                    .map(drop),
                inspect,
            )?;
        }
        check_stop()?;
        return Ok(results);
    }
    for index in 0..LAYERS {
        check_stop()?;
        let prefix = format!("{}.layers.{index}", arch.weight_prefix);
        let load = |suffix: &str| load_tensor(&entries[&format!("{prefix}.{suffix}")]);
        let shape = layer_shape(&arch, index);
        let q_width = shape.query_heads * shape.head_dim;
        let result = match part {
            AneStaticCachePart::QueryKeyValue
            | AneStaticCachePart::QueryKeyValueSlidingInt8
            | AneStaticCachePart::QueryKeyValueSlidingInt8Tiles4 => {
                let mut weights = load("self_attn.q_proj.weight")?;
                weights.extend(load("self_attn.k_proj.weight")?);
                let shared_value = shape.sliding_window.is_none();
                if !shared_value {
                    weights.extend(load("self_attn.v_proj.weight")?);
                }
                let rows =
                    q_width + shape.kv_heads * shape.head_dim * if shared_value { 1 } else { 2 };
                if part == AneStaticCachePart::QueryKeyValueSlidingInt8Tiles4 && !shared_value {
                    load_static_qkv_tiles4(&weights, rows, policy).map(drop)
                } else {
                    load_static_qkv(
                        &weights,
                        rows,
                        part == AneStaticCachePart::QueryKeyValueSlidingInt8 && !shared_value,
                        policy,
                    )
                    .map(drop)
                }
            }
            AneStaticCachePart::Output => {
                let weights = load("self_attn.o_proj.weight")?;
                AneLinear::compile_with_cache_policy(&weights, q_width, HIDDEN, 1, policy).map(drop)
            }
            AneStaticCachePart::FeedForward
            | AneStaticCachePart::FeedForwardLut4
            | AneStaticCachePart::FeedForwardInt8
            | AneStaticCachePart::FeedForwardInt8Stacked
            | AneStaticCachePart::FeedForwardInt8Chunk4
            | AneStaticCachePart::FeedForwardInt8Down4
            | AneStaticCachePart::FeedForwardInt8Interleaved => {
                let gate = load("mlp.gate_proj.weight")?;
                let up = load("mlp.up_proj.weight")?;
                let down = load("mlp.down_proj.weight")?;
                load_static_ffn(
                    &gate,
                    &up,
                    &down,
                    match part {
                        AneStaticCachePart::FeedForwardInt8 => StaticFfnPrecision::Int8,
                        AneStaticCachePart::FeedForwardInt8Chunk4 => StaticFfnPrecision::Int8Chunk4,
                        AneStaticCachePart::FeedForwardInt8Down4 => StaticFfnPrecision::Int8Down4,
                        AneStaticCachePart::FeedForwardInt8Interleaved => {
                            StaticFfnPrecision::Int8Interleaved
                        }
                        AneStaticCachePart::FeedForwardInt8Stacked => {
                            StaticFfnPrecision::Int8Stacked
                        }
                        AneStaticCachePart::FeedForwardLut4 => StaticFfnPrecision::Lut4,
                        _ => StaticFfnPrecision::Fp16,
                    },
                    policy,
                )
                .map(drop)
            }
            AneStaticCachePart::VocabularyAndAttention
            | AneStaticCachePart::AttentionTransposeFlags => unreachable!(),
        };
        record_cache_entry(
            &mut results,
            format!("{part:?}/layer/{index}"),
            result,
            inspect,
        )?;
    }
    check_stop()?;
    Ok(results)
}

fn load_static_qkv_tiles4(
    weights: &[f16],
    rows: usize,
    policy: AneProgramCachePolicy,
) -> Result<AneLinear, String> {
    if rows != 8192 {
        return Err("tiled INT8 QKV supports only the 8192-row sliding projection".into());
    }
    let quantized = AneInt8LinearWeights::quantize(weights, HIDDEN, rows)?;
    AneLinear::compile_int8_tiles4_with_cache_policy(&quantized, policy)
}

fn load_static_qkv(
    weights: &[f16],
    rows: usize,
    int8: bool,
    policy: AneProgramCachePolicy,
) -> Result<AneLinear, String> {
    if int8 {
        // The packed global shape regressed in qualified component trials.
        // Reject an accidental global selection before any private API call.
        if rows != 8192 {
            return Err("INT8 QKV candidate only supports the 8192-row sliding projection".into());
        }
        let quantized = AneInt8LinearWeights::quantize(weights, HIDDEN, rows)?;
        AneLinear::compile_int8_with_cache_policy(&quantized, 1, policy)
    } else {
        AneLinear::compile_with_cache_policy(weights, HIDDEN, rows, 1, policy)
    }
}

fn load_static_ffn(
    gate: &[f16],
    up: &[f16],
    down: &[f16],
    precision: StaticFfnPrecision,
    policy: AneProgramCachePolicy,
) -> Result<AneGatedFfn, String> {
    match precision {
        StaticFfnPrecision::Lut4 => {
            let weights = AneLut4FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_lut4_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Int8 => {
            let weights = AneInt8FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_int8_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Int8Chunk4 => {
            let weights = AneInt8FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_int8_chunk4_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Int8Down4 => {
            let weights = AneInt8FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_int8_down4_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Int8Interleaved => {
            let weights = AneInt8FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_int8_interleaved_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Int8Stacked => {
            let weights = AneInt8FfnWeights::quantize(gate, up, down, HIDDEN, INTERMEDIATE)?;
            AneGatedFfn::compile_int8_stacked_with_cache_policy(&weights, policy)
        }
        StaticFfnPrecision::Fp16 => {
            AneGatedFfn::compile_with_cache_policy(gate, up, down, HIDDEN, INTERMEDIATE, policy)
        }
    }
}

enum OutputProjection {
    Static(AneLinear),
    Dynamic(AneDynamicLinear),
}

impl OutputProjection {
    fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        match self {
            Self::Static(layer) => layer.project(input, output),
            Self::Dynamic(layer) => layer.project(input, output),
        }
    }
}

enum FeedForward {
    Dynamic(AneDynamicFfn),
    Static(AneGatedFfn),
    CheckedStacked {
        baseline: AneGatedFfn,
        candidate: AneGatedFfn,
        baseline_output: Vec<f16>,
        layer: usize,
        comparisons: usize,
    },
}

impl FeedForward {
    fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        match self {
            Self::Dynamic(layer) => layer.project(input, output),
            Self::Static(layer) => layer.project(input, output),
            Self::CheckedStacked {
                baseline,
                candidate,
                baseline_output,
                layer,
                comparisons,
            } => {
                baseline.project(input, baseline_output)?;
                candidate.project(input, output)?;
                check_ffn_output(*layer, baseline_output, output)?;
                *comparisons = comparisons
                    .checked_add(1)
                    .ok_or("FFN comparison count overflow")?;
                Ok(())
            }
        }
    }
}

fn check_ffn_output(layer: usize, baseline: &[f16], candidate: &[f16]) -> Result<(), String> {
    if baseline.len() != candidate.len() {
        return Err(format!(
            "stacked INT8 FFN layer {layer}: output length mismatch"
        ));
    }
    for (index, (a, b)) in baseline.iter().zip(candidate).enumerate() {
        if !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits() {
            return Err(format!("stacked INT8 FFN layer {layer} output {index}: baseline={a:?} candidate={b:?}, bits {:04x}/{:04x}", a.to_bits(), b.to_bits()));
        }
    }
    Ok(())
}

struct Layer {
    shape: PrefillLayerShape,
    qkv: AneLinear,
    attention: AneAttention,
    output: OutputProjection,
    ffn: FeedForward,
    input_norm: Vec<f16>,
    query_norm: Vec<f16>,
    key_norm: Vec<f16>,
    post_attention_norm: Vec<f16>,
    pre_ffn_norm: Vec<f16>,
    post_ffn_norm: Vec<f16>,
    scalar: f32,
    shared_value: bool,
    projected: Vec<f16>,
    value: Vec<f16>,
    attended: Vec<f16>,
}

/// Timings cover application calls, including their I/O and scheduling waits.
/// They are not hardware counters or an isolated ANE device-time measurement.
#[derive(Default, Debug)]
pub struct AneDecodeTimes {
    pub qkv_ms: f64,
    pub attention_ms: f64,
    pub output_ms: f64,
    pub ffn_ms: f64,
    pub vocabulary_ms: f64,
    pub host_ms: f64,
    pub total_ms: f64,
}

pub struct AneDecodedToken {
    pub token: TokenId,
    pub position: usize,
    pub top_five: [(u32, f32); 5],
    pub times: AneDecodeTimes,
}

/// All programs and requests stay on the constructing thread. The shared ANE
/// program handles are deliberately !Send/!Sync; no unsafe transfer is needed.
pub struct GemmaAneDecode {
    layers: Vec<Layer>,
    head: Vec<AneLinear>,
    final_norm: Vec<f16>,
    embedding_file: File,
    embedding_info: SafetensorTensorInfo,
    embedding_bytes: Vec<u8>,
    hidden: Vec<f16>,
    normalized: Vec<f16>,
    branch: Vec<f16>,
    head_output: Vec<f16>,
    sliding_rope: GemmaRope,
    global_rope: GemmaRope,
    capacity: usize,
    epsilon: f32,
    softcap: f32,
    next_position: Option<usize>,
    head_ranking: HeadRankingPlan,
    head_ranking_eligible: bool,
    head_ranking_timing: bool,
    last_head_ranking: Option<(HeadRankingStats, Option<f64>)>,
}

impl GemmaAneDecode {
    /// Default-off CPU head transformation work. This never changes projection
    /// weights, ANE requests, quantization, or the model's softcap expression.
    pub fn configure_head_ranking(
        &mut self,
        plan: HeadRankingPlan,
        timing: bool,
    ) -> Result<(), String> {
        // Enforce the same control/compile boundary for direct library callers,
        // not merely callers routed through the CLI option validator.
        validate_head_ranking_mode(
            plan,
            timing,
            false,
            false,
            false,
            0,
            self.head_ranking_eligible,
        )?;
        validate_head_ranking_configuration(plan, timing, self.softcap)?;
        self.head_ranking = plan;
        self.head_ranking_timing = timing;
        self.last_head_ranking = None;
        Ok(())
    }

    /// CPU transformation time excludes every ANE head projection and surface
    /// call. `None` means no successful observed step, never zero device time.
    pub fn head_ranking_observation(&self) -> Option<(HeadRankingStats, Option<f64>)> {
        self.last_head_ranking
    }

    /// Counts successful bit-identical comparisons in the explicit check plan.
    /// The candidate output continues through the decoder; no fallback occurs.
    pub fn stacked_ffn_checks_per_layer(&self) -> Option<Vec<usize>> {
        self.layers
            .iter()
            .map(|layer| match &layer.ffn {
                FeedForward::CheckedStacked { comparisons, .. } => Some(*comparisons),
                _ => None,
            })
            .collect()
    }

    /// Validate architecture and source tensor metadata without initializing a device.
    pub fn validate_configuration(model_dir: &Path, capacity: usize) -> Result<(), String> {
        validated_weights(model_dir, capacity).map(|_| ())
    }

    /// Prepare once. The current arrangement uses 115 compiled programs:
    /// 96 QKV/O projections, 16 vocabulary tiles, two attention graphs and one
    /// dynamic FFN graph. No compilation or weight packing occurs during decode.
    pub fn load(model_dir: &Path, capacity: usize) -> Result<Self, String> {
        Self::load_with_weight_plan(model_dir, capacity, AneWeightPlan::DynamicFfn)
    }

    pub fn load_with_weight_plan(
        model_dir: &Path,
        capacity: usize,
        weights: AneWeightPlan,
    ) -> Result<Self, String> {
        Self::load_with_compile_budget(model_dir, capacity, weights, 0)
    }

    /// Allow a bounded number of cache misses after explicit provisioning.
    /// Zero retains strict cache-only loading; no failure retries are made.
    pub fn load_with_compile_budget(
        model_dir: &Path,
        capacity: usize,
        weights: AneWeightPlan,
        compile_budget: usize,
    ) -> Result<Self, String> {
        if matches!(
            weights,
            AneWeightPlan::StaticInt8Chunk4FfnCached
                | AneWeightPlan::StaticInt8FfnSlidingQkvTiles4Cached
                | AneWeightPlan::StaticInt8Down4FfnCached
                | AneWeightPlan::StaticInt8InterleavedFfnCached
                | AneWeightPlan::StaticInt8FfnTransposeAttentionCached
        ) && compile_budget != 0
        {
            return Err(
                "candidate inference requires zero compile budget; provision separately".into(),
            );
        }
        if weights == AneWeightPlan::StaticInt8StackedFfnChecked
            && (compile_budget != 0 || std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_none())
        {
            return Err(
                "stacked FFN checking requires zero compile budget and a driver journal".into(),
            );
        }
        if compile_budget > 16
            || (compile_budget != 0
                && weights.cache_policy() != AneProgramCachePolicy::RequireExisting)
        {
            return Err(
                "ANE compile budget must be 0..=16 and applies only to cached weight plans".into(),
            );
        }
        let (arch, entries) = validated_weights(model_dir, capacity)?;
        let cache_policy = if compile_budget != 0 {
            AneProgramCachePolicy::ReuseOrCompileUpTo(compile_budget)
        } else {
            weights.cache_policy()
        };
        let (sliding, global) = if weights == AneWeightPlan::StaticInt8FfnTransposeAttentionCached {
            (
                AneAttentionProgram::compile_transpose_flags_with_cache_policy(
                    PackedAttentionLayout::sliding(16, 8, 256, 1024)?,
                    cache_policy,
                )?,
                AneAttentionProgram::compile_transpose_flags_with_cache_policy(
                    PackedAttentionLayout::new(16, 1, 512, capacity)?,
                    cache_policy,
                )?,
            )
        } else {
            (
                AneAttentionProgram::compile_sliding_with_cache_policy(
                    16,
                    8,
                    256,
                    1024,
                    cache_policy,
                )?,
                AneAttentionProgram::compile_with_cache_policy(16, 1, 512, capacity, cache_policy)?,
            )
        };
        let ffn = if weights == AneWeightPlan::DynamicFfn {
            Some(AneDynamicFfnProgram::compile(HIDDEN, INTERMEDIATE)?)
        } else {
            None
        };
        let output_programs = if weights == AneWeightPlan::StaticFfnDynamicOutput {
            Some([
                AneDynamicLinearProgram::compile(4096, HIDDEN)?,
                AneDynamicLinearProgram::compile(8192, HIDDEN)?,
            ])
        } else {
            None
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let started = Instant::now();
            let prefix = format!("{}.layers.{index}", arch.weight_prefix);
            let shape = layer_shape(&arch, index);
            let q_width = shape.query_heads * shape.head_dim;
            let kv_width = shape.kv_heads * shape.head_dim;
            let shared_value = shape.sliding_window.is_none();
            let load = |suffix: &str| load_tensor(&entries[&format!("{prefix}.{suffix}")]);
            let mut qkv_weights = load("self_attn.q_proj.weight")?;
            qkv_weights.extend(load("self_attn.k_proj.weight")?);
            if !shared_value {
                qkv_weights.extend(load("self_attn.v_proj.weight")?);
            }
            let projection_width = q_width + kv_width * if shared_value { 1 } else { 2 };
            let qkv =
                if weights == AneWeightPlan::StaticInt8FfnSlidingQkvTiles4Cached && !shared_value {
                    load_static_qkv_tiles4(&qkv_weights, projection_width, cache_policy)?
                } else {
                    load_static_qkv(
                        &qkv_weights,
                        projection_width,
                        weights.quantizes_qkv(!shared_value),
                        cache_policy,
                    )?
                };
            drop(qkv_weights);
            let output_weights = load("self_attn.o_proj.weight")?;
            let output = match &output_programs {
                Some(programs) => OutputProjection::Dynamic(
                    programs[usize::from(shared_value)].create_layer(&output_weights)?,
                ),
                None => OutputProjection::Static(AneLinear::compile_with_cache_policy(
                    &output_weights,
                    q_width,
                    HIDDEN,
                    1,
                    cache_policy,
                )?),
            };
            drop(output_weights);
            let gate = load("mlp.gate_proj.weight")?;
            let up = load("mlp.up_proj.weight")?;
            let down = load("mlp.down_proj.weight")?;
            let layer_ffn = match &ffn {
                Some(program) => FeedForward::Dynamic(program.create_layer(&gate, &up, &down)?),
                None if weights == AneWeightPlan::StaticInt8StackedFfnChecked => {
                    let packed =
                        AneInt8FfnWeights::quantize(&gate, &up, &down, HIDDEN, INTERMEDIATE)?;
                    FeedForward::CheckedStacked {
                        baseline: AneGatedFfn::compile_int8_with_cache_policy(
                            &packed,
                            cache_policy,
                        )?,
                        candidate: AneGatedFfn::compile_int8_stacked_with_cache_policy(
                            &packed,
                            cache_policy,
                        )?,
                        baseline_output: vec![f16::ZERO; HIDDEN],
                        layer: index,
                        comparisons: 0,
                    }
                }
                None => FeedForward::Static(load_static_ffn(
                    &gate,
                    &up,
                    &down,
                    weights.static_ffn_precision(),
                    cache_policy,
                )?),
            };
            drop((gate, up, down));
            layers.push(Layer {
                shape,
                qkv,
                output,
                ffn: layer_ffn,
                attention: if shared_value { &global } else { &sliding }.create_request()?,
                input_norm: load("input_layernorm.weight")?,
                query_norm: load("self_attn.q_norm.weight")?,
                key_norm: load("self_attn.k_norm.weight")?,
                post_attention_norm: load("post_attention_layernorm.weight")?,
                pre_ffn_norm: load("pre_feedforward_layernorm.weight")?,
                post_ffn_norm: load("post_feedforward_layernorm.weight")?,
                scalar: load("layer_scalar")?[0].to_f32(),
                shared_value,
                projected: vec![f16::ZERO; projection_width],
                value: vec![f16::ZERO; kv_width],
                attended: vec![f16::ZERO; q_width],
            });
            eprintln!(
                "ANE prepared layer {index}/{} in {:.3}s",
                LAYERS - 1,
                started.elapsed().as_secs_f64()
            );
        }
        let embedding_info =
            entries[&format!("{}.embed_tokens.weight", arch.weight_prefix)].clone();
        let embedding_file = File::open(&embedding_info.file).map_err(|e| e.to_string())?;
        let mut head = Vec::with_capacity(VOCAB / HEAD_ROWS);
        for first in (0..VOCAB).step_by(HEAD_ROWS) {
            let weights = load_rows(&embedding_info, first, HEAD_ROWS)?;
            head.push(AneLinear::compile_with_cache_policy(
                &weights,
                HIDDEN,
                HEAD_ROWS,
                1,
                cache_policy,
            )?);
            eprintln!(
                "ANE prepared vocabulary rows {first}..{}",
                first + HEAD_ROWS
            );
        }
        Ok(Self {
            layers,
            head,
            final_norm: load_tensor(&entries[&format!("{}.norm.weight", arch.weight_prefix)])?,
            embedding_file,
            embedding_info,
            embedding_bytes: vec![0; HIDDEN * 2],
            hidden: vec![f16::ZERO; HIDDEN],
            normalized: vec![f16::ZERO; HIDDEN],
            branch: vec![f16::ZERO; HIDDEN],
            head_output: vec![f16::ZERO; HEAD_ROWS],
            sliding_rope: GemmaRope::new(256, 256, arch.rope_theta_sliding)?,
            global_rope: GemmaRope::new(512, 128, arch.rope_theta_global)?,
            capacity,
            epsilon: arch.rms_norm_eps,
            softcap: arch.logit_softcap,
            next_position: None,
            head_ranking: HeadRankingPlan::Baseline,
            head_ranking_eligible: weights == AneWeightPlan::StaticInt8FfnCached
                && compile_budget == 0,
            head_ranking_timing: false,
            last_head_ranking: None,
        })
    }

    /// Replace every layer's KV state after validating the entire snapshot.
    /// A partial import/evaluation failure leaves the decoder unusable until a
    /// complete successful import; layers can never silently resume out of step.
    pub fn import_prefill(&mut self, snapshot: &AnePrefillSnapshot) -> Result<(), String> {
        self.import_prefill_with_packing(snapshot, KvImportPacking::Baseline)
    }

    /// Experimental import reusing one host packing allocation across layers.
    /// It preserves the original surface writes and fail-stop state contract.
    /// The ordinary import remains the default until comparison is complete.
    pub fn import_prefill_reusing_scratch(
        &mut self,
        snapshot: &AnePrefillSnapshot,
    ) -> Result<(), String> {
        self.import_prefill_with_packing(snapshot, KvImportPacking::ReuseScratch)
    }

    /// Explicit handoff-only strategy; does not select a Metal or ANE kernel.
    pub fn import_prefill_with_packing(
        &mut self,
        snapshot: &AnePrefillSnapshot,
        packing: KvImportPacking,
    ) -> Result<(), String> {
        if snapshot.tokens == 0
            || snapshot.tokens > self.capacity
            || snapshot.layers.len() != LAYERS
        {
            return Err("ANE prefill token count or layer count invalid".into());
        }
        for (layer, saved) in self.layers.iter().zip(&snapshot.layers) {
            let count = snapshot.tokens * layer.shape.kv_heads * layer.shape.head_dim;
            if saved.shape != layer.shape
                || saved.keys.len() != count
                || saved.values.len() != count
                || saved
                    .keys
                    .iter()
                    .chain(&saved.values)
                    .any(|v| !v.is_finite())
            {
                return Err("ANE prefill has an incompatible or nonfinite KV tensor".into());
            }
        }
        self.next_position = None;
        let mut scratch = Vec::new();
        for (layer, saved) in self.layers.iter_mut().zip(&snapshot.layers) {
            if packing == KvImportPacking::Blocked32 {
                layer.attention.import_cache_blocked32_with_scratch(
                    &saved.keys,
                    &saved.values,
                    snapshot.tokens,
                    &mut scratch,
                )?;
            } else if packing == KvImportPacking::ReuseScratch {
                layer.attention.import_cache_with_scratch(
                    &saved.keys,
                    &saved.values,
                    snapshot.tokens,
                    &mut scratch,
                )?;
            } else {
                layer
                    .attention
                    .import_cache(&saved.keys, &saved.values, snapshot.tokens)?;
            }
        }
        self.next_position = Some(snapshot.tokens);
        Ok(())
    }

    /// Consume the first token sampled by Metal (then each actual ANE output).
    /// The caller owns EOS stopping. There is no forced-reference-token path.
    pub fn decode(&mut self, token: TokenId) -> Result<AneDecodedToken, String> {
        self.decode_with_observer(token, &mut |_, _| Ok(()))
    }

    /// Optional first-step layer capture for independent numerical diagnosis.
    /// The observer sees the post-scalar FP16 residual and cannot mutate it.
    pub fn decode_with_observer(
        &mut self,
        token: TokenId,
        observer: &mut impl FnMut(usize, &[f16]) -> Result<(), String>,
    ) -> Result<AneDecodedToken, String> {
        self.decode_with_diagnostic_observers(token, observer, &mut |_, _| Ok(()))
    }

    /// Capture residuals and the actual normalized input to each FFN without
    /// changing inference arithmetic. Observer failures preserve fail-stop KV
    /// ownership: a complete prefill import is required before another decode.
    /// Captured execution includes diagnostic work and is not a timing trial.
    pub fn decode_with_diagnostic_observers(
        &mut self,
        token: TokenId,
        observer: &mut impl FnMut(usize, &[f16]) -> Result<(), String>,
        ffn_input_observer: &mut impl FnMut(usize, &[f16]) -> Result<(), String>,
    ) -> Result<AneDecodedToken, String> {
        let position = self
            .next_position
            .ok_or("ANE decoder requires a complete prefill import")?;
        if token.raw() as usize >= VOCAB || position >= self.capacity {
            return Err("ANE token is outside the vocabulary or context capacity".into());
        }
        self.next_position = None;
        let result = self.decode_inner(token, position, observer, ffn_input_observer)?;
        self.next_position = Some(position + 1);
        Ok(result)
    }

    fn decode_inner(
        &mut self,
        token: TokenId,
        position: usize,
        observer: &mut impl FnMut(usize, &[f16]) -> Result<(), String>,
        ffn_input_observer: &mut impl FnMut(usize, &[f16]) -> Result<(), String>,
    ) -> Result<AneDecodedToken, String> {
        self.last_head_ranking = None;
        let started = Instant::now();
        let mut times = AneDecodeTimes::default();
        let offset = self.embedding_info.file_offset + token.raw() as usize * HIDDEN * 2;
        self.embedding_file
            .read_exact_at(&mut self.embedding_bytes, offset as u64)
            .map_err(|e| e.to_string())?;
        decode_f16(
            &self.embedding_bytes,
            self.embedding_info.dtype,
            &mut self.hidden,
        )?;
        let scale = f16::from_f32((HIDDEN as f32).sqrt()).to_f32();
        scale_layer_f16(&mut self.hidden, scale)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if layer.attention.tokens_seen() != position {
                return Err("ANE layer KV positions disagree".into());
            }
            self.normalized.copy_from_slice(&self.hidden);
            rms_norm_f16_in_place(
                &mut self.normalized,
                HIDDEN,
                Some(&layer.input_norm),
                self.epsilon,
            )?;
            let timer = Instant::now();
            layer.qkv.project(&self.normalized, &mut layer.projected)?;
            times.qkv_ms += milliseconds(timer);
            let q_width = layer.shape.query_heads * layer.shape.head_dim;
            let kv_width = layer.shape.kv_heads * layer.shape.head_dim;
            let (query, kv) = layer.projected.split_at_mut(q_width);
            let (key, projected_value) = kv.split_at_mut(kv_width);
            // Global V is the unnormalized, unrotated K projection.
            layer.value.copy_from_slice(if layer.shared_value {
                key
            } else {
                projected_value
            });
            rms_norm_f16_in_place(
                query,
                layer.shape.head_dim,
                Some(&layer.query_norm),
                self.epsilon,
            )?;
            rms_norm_f16_in_place(
                key,
                layer.shape.head_dim,
                Some(&layer.key_norm),
                self.epsilon,
            )?;
            rms_norm_f16_in_place(&mut layer.value, layer.shape.head_dim, None, self.epsilon)?;
            let rope = if layer.shared_value {
                &mut self.global_rope
            } else {
                &mut self.sliding_rope
            };
            rope.apply_f16(query, position as u32)?;
            rope.apply_f16(key, position as u32)?;
            let timer = Instant::now();
            layer
                .attention
                .decode(query, key, &layer.value, &mut layer.attended)?;
            times.attention_ms += milliseconds(timer);
            let timer = Instant::now();
            layer.output.project(&layer.attended, &mut self.branch)?;
            times.output_ms += milliseconds(timer);
            rms_norm_f16_in_place(
                &mut self.branch,
                HIDDEN,
                Some(&layer.post_attention_norm),
                self.epsilon,
            )?;
            add_residual_f16(&mut self.hidden, &self.branch)?;
            self.normalized.copy_from_slice(&self.hidden);
            rms_norm_f16_in_place(
                &mut self.normalized,
                HIDDEN,
                Some(&layer.pre_ffn_norm),
                self.epsilon,
            )?;
            ffn_input_observer(index, &self.normalized)?;
            let timer = Instant::now();
            layer.ffn.project(&self.normalized, &mut self.branch)?;
            times.ffn_ms += milliseconds(timer);
            rms_norm_f16_in_place(
                &mut self.branch,
                HIDDEN,
                Some(&layer.post_ffn_norm),
                self.epsilon,
            )?;
            add_residual_f16(&mut self.hidden, &self.branch)?;
            scale_layer_f16(&mut self.hidden, layer.scalar)?;
            observer(index, &self.hidden)?;
        }
        rms_norm_f16_in_place(
            &mut self.hidden,
            HIDDEN,
            Some(&self.final_norm),
            self.epsilon,
        )?;
        let mut top_five = [(u32::MAX, f32::NEG_INFINITY); 5];
        let mut ranker = if self.head_ranking == HeadRankingPlan::SoftcapPrune {
            Some(HeadTop5::new(self.head_ranking, self.softcap)?)
        } else {
            None
        };
        let mut ranking_ms = 0.0;
        for (tile, head) in self.head.iter_mut().enumerate() {
            let timer = Instant::now();
            head.project(&self.hidden, &mut self.head_output)?;
            times.vocabulary_ms += milliseconds(timer);
            let ranking_started = self.head_ranking_timing.then(Instant::now);
            if let Some(ranker) = &mut ranker {
                for (row, &value) in self.head_output.iter().enumerate() {
                    ranker.observe((tile * HEAD_ROWS + row) as u32, value)?;
                }
            } else {
                for (row, value) in self.head_output.iter().enumerate() {
                    if !value.is_finite() {
                        return Err("ANE vocabulary projection is nonfinite".into());
                    }
                    let divided = f16::from_f32(value.to_f32() / self.softcap);
                    let squashed = f16::from_f32(divided.to_f32().tanh());
                    let logit = f16::from_f32(squashed.to_f32() * self.softcap).to_f32();
                    if let Some(rank) = top_five.iter().position(|&(_, best)| logit > best) {
                        top_five.copy_within(rank..4, rank + 1);
                        top_five[rank] = ((tile * HEAD_ROWS + row) as u32, logit);
                    }
                }
            }
            if let Some(started) = ranking_started {
                ranking_ms += milliseconds(started);
            }
        }
        let stats = if let Some(ranker) = ranker {
            top_five = ranker.finish(VOCAB as u32)?;
            ranker.stats()
        } else {
            HeadRankingStats {
                considered: VOCAB as u32,
                transformed: VOCAB as u32,
                pruned: 0,
            }
        };
        if self.head_ranking != HeadRankingPlan::Baseline || self.head_ranking_timing {
            self.last_head_ranking = Some((stats, self.head_ranking_timing.then_some(ranking_ms)));
        }
        times.total_ms = milliseconds(started);
        times.host_ms = times.total_ms
            - times.qkv_ms
            - times.attention_ms
            - times.output_ms
            - times.ffn_ms
            - times.vocabulary_ms;
        Ok(AneDecodedToken {
            token: TokenId(top_five[0].0),
            position,
            top_five,
            times,
        })
    }
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn validated_weights(
    model_dir: &Path,
    capacity: usize,
) -> Result<(Gemma4Arch, BTreeMap<String, SafetensorTensorInfo>), String> {
    let arch = Gemma4Arch::from_dir(model_dir).map_err(|e| e.to_string())?;
    validate_architecture(&arch, capacity)?;
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let text = config.get("text_config").unwrap_or(&config);
    if text["hidden_activation"] != "gelu_pytorch_tanh" || text["attention_bias"] != false {
        return Err("ANE Gemma decode requires tanh GELU and bias-free projections".into());
    }
    let entries = scan_safetensor_tensors(model_dir).map_err(|e| e.to_string())?;
    validate_tensors(&entries, &arch)?;
    Ok((arch, entries))
}

fn layer_shape(arch: &Gemma4Arch, index: usize) -> PrefillLayerShape {
    let sliding = arch.layer_types[index] == Gemma4LayerType::SlidingAttention;
    PrefillLayerShape {
        query_heads: 16,
        kv_heads: if sliding { 8 } else { 1 },
        head_dim: if sliding { 256 } else { 512 },
        sliding_window: sliding.then_some(1024),
    }
}

fn validate_architecture(arch: &Gemma4Arch, capacity: usize) -> Result<(), String> {
    if (
        arch.hidden_size,
        arch.intermediate_size,
        arch.num_hidden_layers,
        arch.vocab_size,
    ) != (HIDDEN, INTERMEDIATE, LAYERS, VOCAB)
        || (
            arch.num_attention_heads,
            arch.num_kv_heads_sliding,
            arch.num_kv_heads_global,
            arch.head_dim_sliding,
            arch.head_dim_global,
        ) != (16, 8, 1, 256, 512)
        || arch.layer_types.len() != LAYERS
        || arch.sliding_window_size != 1024
        || arch.enable_moe_block
        || arch.use_double_wide_mlp
        || arch.num_kv_shared_layers != 0
        || arch.hidden_size_per_layer_input != 0
        || !arch.tie_word_embeddings
        || !arch.attention_k_eq_v
        || arch.rope_theta_sliding != 10000.0
        || arch.rope_theta_global != 1000000.0
        || arch.partial_rotary_factor_global != 0.25
        || arch.logit_softcap != 30.0
        || arch.rms_norm_eps != 1e-6
        || !matches!(capacity, 64 | 1024)
    {
        return Err("ANE decoder currently requires the dense Gemma 4 12B architecture and a qualified 64- or 1024-token global context capacity".into());
    }
    Ok(())
}

fn validate_tensors(
    entries: &BTreeMap<String, SafetensorTensorInfo>,
    arch: &Gemma4Arch,
) -> Result<(), String> {
    let check = |name: String, shape: &[usize]| {
        let entry = entries
            .get(&name)
            .ok_or_else(|| format!("missing ANE tensor {name}"))?;
        if entry.shape != shape
            || !matches!(entry.dtype, DType::Bf16 | DType::F16)
            || entry.nbytes != shape.iter().product::<usize>() * 2
        {
            return Err(format!("unsupported ANE tensor shape/dtype: {name}"));
        }
        Ok(())
    };
    for index in 0..LAYERS {
        let prefix = format!("{}.layers.{index}", arch.weight_prefix);
        let shape = layer_shape(arch, index);
        let q = shape.query_heads * shape.head_dim;
        let kv = shape.kv_heads * shape.head_dim;
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            check(format!("{prefix}.{suffix}"), &[HIDDEN])?;
        }
        for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            check(format!("{prefix}.{suffix}"), &[shape.head_dim])?;
        }
        check(format!("{prefix}.layer_scalar"), &[1])?;
        for (suffix, dimensions) in [
            ("self_attn.q_proj.weight", [q, HIDDEN]),
            ("self_attn.k_proj.weight", [kv, HIDDEN]),
            ("self_attn.o_proj.weight", [HIDDEN, q]),
            ("mlp.gate_proj.weight", [INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", [INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", [HIDDEN, INTERMEDIATE]),
        ] {
            check(format!("{prefix}.{suffix}"), &dimensions)?;
        }
        if shape.sliding_window.is_some() {
            check(format!("{prefix}.self_attn.v_proj.weight"), &[kv, HIDDEN])?;
        }
    }
    check(format!("{}.norm.weight", arch.weight_prefix), &[HIDDEN])?;
    check(
        format!("{}.embed_tokens.weight", arch.weight_prefix),
        &[VOCAB, HIDDEN],
    )
}

fn load_tensor(entry: &SafetensorTensorInfo) -> Result<Vec<f16>, String> {
    read_values(entry, 0, entry.nbytes)
}

fn load_rows(entry: &SafetensorTensorInfo, first: usize, rows: usize) -> Result<Vec<f16>, String> {
    let [height, width] = entry.shape.as_slice() else {
        return Err("ANE row load requires a matrix".into());
    };
    if first
        .checked_add(rows)
        .filter(|&end| end <= *height)
        .is_none()
    {
        return Err("ANE row load exceeds tensor".into());
    }
    read_values(entry, first * width * 2, rows * width * 2)
}

fn read_values(
    entry: &SafetensorTensorInfo,
    offset: usize,
    bytes: usize,
) -> Result<Vec<f16>, String> {
    let file = File::open(&entry.file).map_err(|e| e.to_string())?;
    let mut data = vec![0; bytes];
    file.read_exact_at(&mut data, (entry.file_offset + offset) as u64)
        .map_err(|e| e.to_string())?;
    let mut values = vec![f16::ZERO; bytes / 2];
    decode_f16(&data, entry.dtype, &mut values)?;
    Ok(values)
}

#[cfg(test)]
fn decode_f16_scalar(bytes: &[u8], dtype: DType, output: &mut [f16]) -> Result<(), String> {
    if bytes.len() != output.len() * 2 {
        return Err("ANE FP16 conversion length mismatch".into());
    }
    for (bytes, value) in bytes.chunks_exact(2).zip(output) {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        *value = match dtype {
            DType::Bf16 => f16::from_f32(bf16::from_bits(bits).to_f32()),
            DType::F16 => f16::from_bits(bits),
            _ => return Err("ANE decoder supports BF16 or FP16 checkpoint weights".into()),
        };
        if !value.is_finite() {
            return Err("ANE checkpoint conversion produced a nonfinite FP16 value".into());
        }
    }
    Ok(())
}

fn decode_f16(bytes: &[u8], dtype: DType, output: &mut [f16]) -> Result<(), String> {
    if output.len().checked_mul(2) != Some(bytes.len()) {
        return Err("ANE FP16 conversion length mismatch".into());
    }
    if bytes.is_empty() {
        return Ok(());
    }
    match dtype {
        DType::Bf16 => {
            let mut maximum = 0_u16;
            for (pair, value) in bytes.chunks_exact(2).zip(output) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                let magnitude = bits & 0x7fff;
                maximum = maximum.max(magnitude);
                let converted = if magnitude >= 0x3880 {
                    // BF16 has fewer fraction bits: normal finite FP16 values
                    // need only an exponent-bias adjustment and three zeros.
                    magnitude.wrapping_sub(0x3800).wrapping_shl(3)
                } else if magnitude <= 0x3300 {
                    // At or below half the smallest FP16 subnormal, ties-to-even
                    // produces signed zero.
                    0
                } else {
                    let significand = ((magnitude & 0x7f) | 0x80) << 3;
                    let shift = 113 - (magnitude >> 7);
                    let whole = significand >> shift;
                    let remainder = significand & ((1 << shift) - 1);
                    let halfway = 1 << (shift - 1);
                    whole
                        + u16::from(remainder > halfway || (remainder == halfway && whole & 1 != 0))
                };
                *value = f16::from_bits((bits & 0x8000) | converted);
            }
            // 0x4780 is BF16 65536 and overflows FP16. This also rejects all
            // infinity/NaN encodings. Callers discard the output on error.
            if maximum >= 0x4780 {
                return Err("ANE checkpoint conversion produced a nonfinite FP16 value".into());
            }
        }
        DType::F16 => {
            let mut maximum = 0_u16;
            for (pair, value) in bytes.chunks_exact(2).zip(output) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                maximum = maximum.max(bits & 0x7fff);
                *value = f16::from_bits(bits);
            }
            if maximum >= 0x7c00 {
                return Err("ANE checkpoint conversion produced a nonfinite FP16 value".into());
            }
        }
        _ => return Err("ANE decoder supports BF16 or FP16 checkpoint weights".into()),
    }
    Ok(())
}

#[cfg(test)]
#[path = "ane_checkpoint_conversion_tests.rs"]
mod conversion_tests;

#[cfg(test)]
#[path = "ane_int8_projection_tests.rs"]
mod int8_projection_tests;

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "private ANE layer-0 fused attention/output compile-source probe; one compiler attempt, zero evaluations, explicit journal and receipt"]
    fn hardware_layer0_fused_attention_output_compile_source() {
        use rvllm_apple::ane_attention::AneAttentionOutputCompile;
        use rvllm_apple::ane_linear::compile_budget_used;

        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA_MODEL_DIR").expect("model directory required"),
        );
        let receipt_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_FUSED_ATTENTION_OUTPUT_RECEIPT")
                .expect("receipt path required"),
        );
        let journal_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").expect("driver journal path required"),
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite fused receipt"
        );
        assert!(
            !journal_path.exists(),
            "refusing to append to an old driver journal"
        );
        assert_eq!(compile_budget_used(), 0, "probe requires a fresh process");

        let prepared = (|| -> Result<_, String> {
            let (arch, entries) = super::validated_weights(&model, 1024)?;
            let shape = super::layer_shape(&arch, 0);
            let layout = match shape.sliding_window {
                Some(window) => rvllm_apple::ane_attention_layout::PackedAttentionLayout::sliding(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    window,
                )?,
                None => rvllm_apple::ane_attention_layout::PackedAttentionLayout::new(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    1024,
                )?,
            };
            let name = format!("{}.layers.0.self_attn.o_proj.weight", arch.weight_prefix);
            let weights = super::load_tensor(
                entries
                    .get(&name)
                    .ok_or_else(|| format!("missing ANE tensor {name}"))?,
            )?;
            let identity =
                AneAttentionOutputCompile::source_identity(layout, &weights, super::HIDDEN)?;
            let result = AneAttentionOutputCompile::compile_layer(
                layout,
                &weights,
                super::HIDDEN,
                super::AneProgramCachePolicy::ReuseOrCompileUpTo(1),
            )
            .map(drop);
            drop(weights);
            Ok((identity, result))
        })();
        let compiler_calls = compile_budget_used();
        let (status, identity, error) = match prepared {
            Ok((identity, Ok(()))) => ("compiled", Some(identity), None),
            Ok((identity, Err(error))) => ("failed", Some(identity), Some(error)),
            Err(error) => ("failed", None, Some(error)),
        };
        let receipt = serde_json::json!({
            "schema":"rvllm.ane_fused_attention_output_compile_source.v1",
            "status":status,
            "layer":0,
            "model_dir":model,
            "cache_identity":identity.as_ref().map(|value| serde_json::json!({
                "mil_sha256":value.mil_sha256,
                "weight_blob_sha256":value.weight_blob_sha256,
                "input_bytes":value.input_bytes,
                "output_bytes":value.output_bytes,
            })),
            "compiler_calls":compiler_calls,
            "compiler_call_limit":1,
            "accelerator_evaluations":0,
            "external_inputs":1,
            "external_outputs":1,
            "driver_journal":journal_path,
            "error":error,
            "claim":"Layer-0 compile-source evidence only; no inference, correctness, speed, route, or promotion claim."
        });
        std::fs::write(
            &receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize fused receipt"),
        )
        .expect("preserve fused receipt");
        assert_eq!(
            compiler_calls, 1,
            "compile-source requires exactly one compiler attempt"
        );
        assert_eq!(
            status, "compiled",
            "fused compile failed; inspect preserved receipt and journal"
        );
    }

    #[test]
    #[ignore = "private ANE layer-0 fused attention/output exact-cache reload probe; zero compiler calls and zero evaluations"]
    fn hardware_layer0_fused_attention_output_reload_existing() {
        use rvllm_apple::ane_attention::AneAttentionOutputCompile;
        use rvllm_apple::ane_linear::compile_budget_used;

        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA_MODEL_DIR").expect("model directory required"),
        );
        let receipt_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_FUSED_ATTENTION_OUTPUT_RECEIPT")
                .expect("receipt path required"),
        );
        let journal_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").expect("driver journal path required"),
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite fused reload receipt"
        );
        assert!(
            !journal_path.exists(),
            "refusing to append to an old driver journal"
        );
        assert_eq!(compile_budget_used(), 0, "probe requires a fresh process");

        let prepared = (|| -> Result<_, String> {
            let (arch, entries) = super::validated_weights(&model, 1024)?;
            let shape = super::layer_shape(&arch, 0);
            let layout = match shape.sliding_window {
                Some(window) => rvllm_apple::ane_attention_layout::PackedAttentionLayout::sliding(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    window,
                )?,
                None => rvllm_apple::ane_attention_layout::PackedAttentionLayout::new(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    1024,
                )?,
            };
            let name = format!("{}.layers.0.self_attn.o_proj.weight", arch.weight_prefix);
            let weights = super::load_tensor(
                entries
                    .get(&name)
                    .ok_or_else(|| format!("missing ANE tensor {name}"))?,
            )?;
            let identity =
                AneAttentionOutputCompile::source_identity(layout, &weights, super::HIDDEN)?;
            let result = AneAttentionOutputCompile::compile_layer(
                layout,
                &weights,
                super::HIDDEN,
                super::AneProgramCachePolicy::RequireExisting,
            )
            .map(drop);
            drop(weights);
            Ok((identity, result))
        })();
        let compiler_calls = compile_budget_used();
        let (status, identity, error) = match prepared {
            Ok((identity, Ok(()))) => ("loaded", Some(identity), None),
            Ok((identity, Err(error))) => ("failed", Some(identity), Some(error)),
            Err(error) => ("failed", None, Some(error)),
        };
        let receipt = serde_json::json!({
            "schema":"rvllm.ane_fused_attention_output_reload_existing.v1",
            "status":status,
            "layer":0,
            "model_dir":model,
            "cache_identity":identity.as_ref().map(|value| serde_json::json!({
                "mil_sha256":value.mil_sha256,
                "weight_blob_sha256":value.weight_blob_sha256,
                "input_bytes":value.input_bytes,
                "output_bytes":value.output_bytes,
            })),
            "compiler_calls":compiler_calls,
            "compiler_call_limit":0,
            "accelerator_evaluations":0,
            "external_inputs":1,
            "external_outputs":1,
            "driver_journal":journal_path,
            "error":error,
            "claim":"Layer-0 exact-cache reload evidence only; no inference, correctness, speed, route, or promotion claim."
        });
        std::fs::write(
            &receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize fused reload receipt"),
        )
        .expect("preserve fused reload receipt");
        assert_eq!(
            compiler_calls, 0,
            "exact-cache reload must not call the compiler"
        );
        assert_eq!(
            status, "loaded",
            "fused exact-cache reload failed; inspect preserved receipt and journal"
        );
    }

    #[test]
    #[ignore = "private ANE layer-0 fused attention/output compile-then-reload probe; one compiler call, two loads, zero evaluations"]
    fn hardware_layer0_fused_attention_output_compile_then_reload() {
        use rvllm_apple::ane_attention::AneAttentionOutputCompile;
        use rvllm_apple::ane_linear::compile_budget_used;

        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA_MODEL_DIR").expect("model directory required"),
        );
        let receipt_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_FUSED_ATTENTION_OUTPUT_RECEIPT")
                .expect("receipt path required"),
        );
        let journal_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").expect("driver journal path required"),
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite fused compile-reload receipt"
        );
        assert!(
            !journal_path.exists(),
            "refusing to append to an old driver journal"
        );
        assert_eq!(compile_budget_used(), 0, "probe requires a fresh process");

        let prepared = (|| -> Result<_, String> {
            let (arch, entries) = super::validated_weights(&model, 1024)?;
            let shape = super::layer_shape(&arch, 0);
            let layout = match shape.sliding_window {
                Some(window) => rvllm_apple::ane_attention_layout::PackedAttentionLayout::sliding(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    window,
                )?,
                None => rvllm_apple::ane_attention_layout::PackedAttentionLayout::new(
                    shape.query_heads,
                    shape.kv_heads,
                    shape.head_dim,
                    1024,
                )?,
            };
            let name = format!("{}.layers.0.self_attn.o_proj.weight", arch.weight_prefix);
            let weights = super::load_tensor(
                entries
                    .get(&name)
                    .ok_or_else(|| format!("missing ANE tensor {name}"))?,
            )?;
            let identity =
                AneAttentionOutputCompile::source_identity(layout, &weights, super::HIDDEN)?;
            let first = AneAttentionOutputCompile::compile_layer(
                layout,
                &weights,
                super::HIDDEN,
                super::AneProgramCachePolicy::ReuseOrCompileUpTo(1),
            )?;
            let compiler_calls_after_first_load = compile_budget_used();
            drop(first);
            let second = AneAttentionOutputCompile::compile_layer(
                layout,
                &weights,
                super::HIDDEN,
                super::AneProgramCachePolicy::RequireExisting,
            )?;
            let compiler_calls_after_second_load = compile_budget_used();
            drop(second);
            drop(weights);
            Ok((
                identity,
                compiler_calls_after_first_load,
                compiler_calls_after_second_load,
            ))
        })();
        let compiler_calls = compile_budget_used();
        let (status, identity, after_first, after_second, error) = match prepared {
            Ok((identity, after_first, after_second)) => (
                "loaded_twice",
                Some(identity),
                Some(after_first),
                Some(after_second),
                None,
            ),
            Err(error) => ("failed", None, None, None, Some(error)),
        };
        let receipt = serde_json::json!({
            "schema":"rvllm.ane_fused_attention_output_compile_then_reload.v1",
            "status":status,
            "layer":0,
            "model_dir":model,
            "cache_identity":identity.as_ref().map(|value| serde_json::json!({
                "mil_sha256":value.mil_sha256,
                "weight_blob_sha256":value.weight_blob_sha256,
                "input_bytes":value.input_bytes,
                "output_bytes":value.output_bytes,
            })),
            "compiler_calls_after_first_load":after_first,
            "compiler_calls_after_second_load":after_second,
            "compiler_calls":compiler_calls,
            "compiler_call_limit":1,
            "loads":2,
            "accelerator_evaluations":0,
            "external_inputs":1,
            "external_outputs":1,
            "driver_journal":journal_path,
            "error":error,
            "claim":"Layer-0 same-client compile/reload evidence only; no inference, correctness, speed, route, cross-process persistence, or promotion claim."
        });
        std::fs::write(
            &receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize fused compile-reload receipt"),
        )
        .expect("preserve fused compile-reload receipt");
        assert_eq!(
            compiler_calls, 1,
            "second load must reuse the compiled graph"
        );
        assert_eq!(
            after_first,
            Some(1),
            "first load must consume the single compiler call"
        );
        assert_eq!(
            after_second,
            Some(1),
            "second load must not call the compiler"
        );
        assert_eq!(
            status, "loaded_twice",
            "fused same-client reload failed; inspect preserved receipt and journal"
        );
    }

    #[test]
    #[ignore = "private ANE corrected fused attention/output real-weight oracle; one compile and twelve bounded evaluations"]
    fn hardware_layer0_fused_attention_output_real_weight_oracle() {
        use rvllm_apple::ane_attention::AneAttentionOutputCompile;
        use rvllm_apple::ane_linear::compile_budget_used;

        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA_MODEL_DIR").expect("model directory required"),
        );
        let receipt_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_FUSED_ATTENTION_OUTPUT_RECEIPT")
                .expect("receipt path required"),
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite oracle receipt"
        );
        assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
        assert_eq!(compile_budget_used(), 0, "oracle requires a fresh process");

        let (arch, entries) = super::validated_weights(&model, 1024).unwrap();
        let shape = super::layer_shape(&arch, 0);
        let layout = rvllm_apple::ane_attention_layout::PackedAttentionLayout::sliding(
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.sliding_window.unwrap(),
        )
        .unwrap();
        let name = format!("{}.layers.0.self_attn.o_proj.weight", arch.weight_prefix);
        let weights = super::load_tensor(entries.get(&name).unwrap()).unwrap();
        let identity =
            AneAttentionOutputCompile::source_identity(layout, &weights, super::HIDDEN).unwrap();

        // Every token has the same V. Attention therefore has an exact,
        // score-independent result at every tested mask/ring boundary: each
        // query head receives the V row of its KV head. This makes head/group
        // flattening and the real checkpoint projection independently visible.
        let kv_value: Vec<f16> = (0..layout.kv_width())
            .map(|i| f16::from_f32(((i * 17 + 5) % 127) as f32 / 1024.0 - 0.0625))
            .collect();
        let mut attended = vec![f16::ZERO; layout.query_width()];
        for kv_head in 0..layout.kv_heads() {
            for group in 0..layout.groups() {
                let head = kv_head * layout.groups() + group;
                let src = kv_head * layout.head_dim();
                let dst = head * layout.head_dim();
                attended[dst..dst + layout.head_dim()]
                    .copy_from_slice(&kv_value[src..src + layout.head_dim()]);
            }
        }
        let expected: Vec<f32> = weights
            .chunks_exact(layout.query_width())
            .map(|row| {
                row.iter()
                    .zip(&attended)
                    .map(|(weight, value)| weight.to_f32() * value.to_f32())
                    .sum()
            })
            .collect();
        assert_eq!(expected.len(), super::HIDDEN);
        assert!(expected.iter().all(|value| value.is_finite()));

        let program = AneAttentionOutputCompile::compile_layer(
            layout,
            &weights,
            super::HIDDEN,
            super::AneProgramCachePolicy::ReuseOrCompileUpTo(1),
        )
        .unwrap();
        let mut request = program.create_request().unwrap();
        let query = vec![f16::ZERO; layout.query_width()];
        let mut encoded_query = vec![0; query.len() * 2];
        layout.encode_query(&query, &mut encoded_query).unwrap();
        let mut mask = vec![0; layout.capacity() * 2];
        let mut cases = Vec::new();
        let mut violations = 0_usize;
        let mut evaluations = 0_usize;
        let guard = 64_usize;
        for tokens in [1_usize, 31, 32, 33, 1024, 1025] {
            let keys = vec![f16::ZERO; tokens * layout.kv_width()];
            let values: Vec<_> = (0..tokens).flat_map(|_| kv_value.iter().copied()).collect();
            let packed = layout.import_cache(&keys, &values, tokens).unwrap();
            let mut guarded = vec![0xa5_u8; guard + packed.len() + guard];
            let input = &mut guarded[guard..guard + packed.len()];
            input.copy_from_slice(&packed);
            for (row, data) in input
                .chunks_exact_mut(layout.row_bytes())
                .zip(encoded_query.chunks_exact(layout.groups() * 2))
            {
                row[..data.len()].copy_from_slice(data);
            }
            layout.encode_mask(tokens, &mut mask).unwrap();
            let mask_offset = layout.mask_offset();
            input[mask_offset..mask_offset + mask.len()].copy_from_slice(&mask);

            let mut repeat_outputs = Vec::new();
            let mut maximum_absolute_error = 0.0_f32;
            for _ in 0..2 {
                let mut actual = vec![f16::ZERO; super::HIDDEN];
                request.evaluate_packed(input, &mut actual).unwrap();
                evaluations += 1;
                for (actual, expected) in actual.iter().zip(&expected) {
                    let error = (actual.to_f32() - expected).abs();
                    let tolerance = 0.002 + 0.01 * expected.abs();
                    if !actual.is_finite() || error > tolerance {
                        violations += 1;
                    }
                    maximum_absolute_error = maximum_absolute_error.max(error);
                }
                repeat_outputs.push(actual);
            }
            let repeated_bits_identical = repeat_outputs[0]
                .iter()
                .zip(&repeat_outputs[1])
                .all(|(first, second)| first.to_bits() == second.to_bits());
            if !repeated_bits_identical {
                violations += 1;
            }
            let guards_unchanged = guarded[..guard].iter().all(|&byte| byte == 0xa5)
                && guarded[guard + packed.len()..]
                    .iter()
                    .all(|&byte| byte == 0xa5);
            if !guards_unchanged {
                violations += 1;
            }
            cases.push(serde_json::json!({
                "tokens":tokens,
                "maximum_absolute_error":maximum_absolute_error,
                "repeated_output_bits_identical":repeated_bits_identical,
                "input_guards_unchanged":guards_unchanged,
            }));
        }
        drop(request);
        drop(program);
        let compiler_calls = compile_budget_used();
        let receipt = serde_json::json!({
            "schema":"rvllm.ane_fused_attention_output_real_weight_oracle.v1",
            "status":if violations == 0 { "passed" } else { "failed" },
            "layer":0,
            "model_dir":model,
            "cache_identity":{
                "mil_sha256":identity.mil_sha256,
                "weight_blob_sha256":identity.weight_blob_sha256,
                "input_bytes":identity.input_bytes,
                "output_bytes":identity.output_bytes,
            },
            "compiler_calls":compiler_calls,
            "compiler_call_limit":1,
            "accelerator_evaluations":evaluations,
            "external_inputs":1,
            "external_outputs":1,
            "tested_tokens":[1,31,32,33,1024,1025],
            "cases":cases,
            "violations":violations,
            "tolerance":"0.002 + 0.01 * abs(independent_fp32_projection_reference)",
            "claim":"Corrected layer-0 fused attention/output component oracle only; constant-V construction makes the attention result independent of scores and exposes head/group flattening. No timing, full-route, checkpoint-quality, or promotion claim."
        });
        std::fs::write(
            receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize fused oracle receipt"),
        )
        .expect("preserve fused oracle receipt");
        assert!(
            compiler_calls <= 1,
            "oracle may reuse the exact cached graph but must never compile more than once"
        );
        assert_eq!(evaluations, 12);
        assert_eq!(violations, 0, "inspect preserved fused oracle receipt");
    }

    #[test]
    #[ignore = "cached-only ANE fused attention/output timing; 12 alternating arms, zero compiler calls"]
    fn hardware_layer0_fused_attention_output_abba_timing() {
        use rvllm_apple::ane_attention::{AneAttentionOutputCompile, AneAttentionProgram};
        use rvllm_apple::ane_linear::{compile_budget_used, AneLinear};
        use std::time::Instant;

        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA_MODEL_DIR").expect("model directory required"),
        );
        let receipt_path = std::path::PathBuf::from(
            std::env::var_os("RVLLM_ANE_FUSED_ATTENTION_OUTPUT_RECEIPT")
                .expect("receipt path required"),
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite timing receipt"
        );
        assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
        assert_eq!(compile_budget_used(), 0, "timing requires a fresh process");

        let (arch, entries) = super::validated_weights(&model, 1024).unwrap();
        let shape = super::layer_shape(&arch, 0);
        let layout = rvllm_apple::ane_attention_layout::PackedAttentionLayout::sliding(
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.sliding_window.unwrap(),
        )
        .unwrap();
        let name = format!("{}.layers.0.self_attn.o_proj.weight", arch.weight_prefix);
        let weights = super::load_tensor(entries.get(&name).unwrap()).unwrap();
        let identity =
            AneAttentionOutputCompile::source_identity(layout, &weights, super::HIDDEN).unwrap();
        let policy = super::AneProgramCachePolicy::ReuseOrCompileUpTo(3);
        let mut baseline_attention = AneAttentionProgram::compile_sliding_with_cache_policy(
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.sliding_window.unwrap(),
            policy,
        )
        .unwrap()
        .create_request()
        .unwrap();
        let mut baseline_output = AneLinear::compile_with_cache_policy(
            &weights,
            layout.query_width(),
            super::HIDDEN,
            1,
            policy,
        )
        .unwrap();
        let mut fused =
            AneAttentionOutputCompile::compile_layer(layout, &weights, super::HIDDEN, policy)
                .unwrap()
                .create_request()
                .unwrap();
        let setup_compiler_calls = compile_budget_used();
        assert!(
            setup_compiler_calls <= 3,
            "bounded timing setup exceeded its compiler-call budget"
        );

        let imported = 1024_usize;
        let keys: Vec<_> = (0..imported * layout.kv_width())
            .map(|i| f16::from_f32(((i * 13 + 7) % 97) as f32 / 1024.0 - 0.046875))
            .collect();
        let values: Vec<_> = (0..keys.len())
            .map(|i| f16::from_f32(((i * 17 + 5) % 127) as f32 / 1024.0 - 0.0625))
            .collect();
        baseline_attention
            .import_cache(&keys, &values, imported)
            .unwrap();
        fused.import_cache(&keys, &values, imported).unwrap();
        let query: Vec<_> = (0..layout.query_width())
            .map(|i| f16::from_f32(((i * 19 + 3) % 113) as f32 / 1024.0 - 0.0546875))
            .collect();
        let key: Vec<_> = (0..layout.kv_width())
            .map(|i| f16::from_f32(((i * 23 + 11) % 109) as f32 / 1024.0 - 0.052734375))
            .collect();
        let value: Vec<_> = (0..layout.kv_width())
            .map(|i| f16::from_f32(((i * 29 + 13) % 131) as f32 / 1024.0 - 0.0634765625))
            .collect();
        let mut attended = vec![f16::ZERO; layout.query_width()];
        let mut baseline_projected = vec![f16::ZERO; super::HIDDEN];
        let mut fused_projected = vec![f16::ZERO; super::HIDDEN];

        // Equal warmup work establishes both persistent request paths before
        // any measured block. Sliding positions remain comparable thereafter.
        for _ in 0..8 {
            baseline_attention
                .decode(&query, &key, &value, &mut attended)
                .unwrap();
            baseline_output
                .project(&attended, &mut baseline_projected)
                .unwrap();
            fused
                .decode(&query, &key, &value, &mut fused_projected)
                .unwrap();
        }
        let maximum_warmup_difference = baseline_projected
            .iter()
            .zip(&fused_projected)
            .map(|(baseline, fused)| (baseline.to_f32() - fused.to_f32()).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum_warmup_difference < 0.004);

        let repetitions = 64_usize;
        let reverse_order = std::env::var("RVLLM_ANE_FUSED_TIMING_ORDER")
            .map(|value| value == "BAAB_FIRST")
            .unwrap_or(false);
        let (sequence_name, sequence) = if reverse_order {
            (
                "BAAB/ABBA/BAAB",
                [
                    "fused", "baseline", "baseline", "fused", // BAAB
                    "baseline", "fused", "fused", "baseline", // ABBA
                    "fused", "baseline", "baseline", "fused", // BAAB
                ],
            )
        } else {
            (
                "ABBA/BAAB/ABBA",
                [
                    "baseline", "fused", "fused", "baseline", // ABBA
                    "fused", "baseline", "baseline", "fused", // BAAB
                    "baseline", "fused", "fused", "baseline", // ABBA
                ],
            )
        };
        let mut observations = Vec::new();
        let mut baseline_ms = Vec::new();
        let mut fused_ms = Vec::new();
        for (index, arm) in sequence.into_iter().enumerate() {
            let started = Instant::now();
            match arm {
                "baseline" => {
                    for _ in 0..repetitions {
                        baseline_attention
                            .decode(&query, &key, &value, &mut attended)
                            .unwrap();
                        baseline_output
                            .project(&attended, &mut baseline_projected)
                            .unwrap();
                    }
                }
                "fused" => {
                    for _ in 0..repetitions {
                        fused
                            .decode(&query, &key, &value, &mut fused_projected)
                            .unwrap();
                    }
                }
                _ => unreachable!(),
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let per_token_ms = elapsed_ms / repetitions as f64;
            if arm == "baseline" {
                baseline_ms.push(per_token_ms);
            } else {
                fused_ms.push(per_token_ms);
            }
            observations.push(serde_json::json!({
                "index":index,
                "block":index / 4,
                "order":if reverse_order {
                    match index / 4 { 1 => "ABBA", _ => "BAAB" }
                } else {
                    match index / 4 { 1 => "BAAB", _ => "ABBA" }
                },
                "arm":arm,
                "repetitions":repetitions,
                "elapsed_ms":elapsed_ms,
                "milliseconds_per_token":per_token_ms,
            }));
        }
        let median = |values: &mut Vec<f64>| {
            values.sort_by(f64::total_cmp);
            (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
        };
        let baseline_median_ms = median(&mut baseline_ms);
        let fused_median_ms = median(&mut fused_ms);
        let ratio = baseline_median_ms / fused_median_ms;
        let baseline_drift = baseline_ms[baseline_ms.len() - 1] / baseline_ms[0] - 1.0;
        let fused_drift = fused_ms[fused_ms.len() - 1] / fused_ms[0] - 1.0;
        let compiler_calls = compile_budget_used();
        let receipt = serde_json::json!({
            "schema":"rvllm.ane_fused_attention_output_abba_timing.v1",
            "status":"measured",
            "layer":0,
            "model_dir":model,
            "cache_identity":{
                "mil_sha256":identity.mil_sha256,
                "weight_blob_sha256":identity.weight_blob_sha256,
                "input_bytes":identity.input_bytes,
                "output_bytes":identity.output_bytes,
            },
            "setup_compiler_calls":setup_compiler_calls,
            "compiler_calls_after_timing":compiler_calls,
            "setup_compiler_call_limit":3,
            "compiler_calls_during_timing":compiler_calls - setup_compiler_calls,
            "warmup_tokens_per_arm":8,
            "measured_tokens_per_arm":repetitions * 6,
            "sequence":sequence_name,
            "baseline_evaluations_per_token":2,
            "fused_evaluations_per_token":1,
            "baseline_median_ms_per_token":baseline_median_ms,
            "fused_median_ms_per_token":fused_median_ms,
            "median_baseline_over_fused":ratio,
            "baseline_range_drift_fraction":baseline_drift,
            "fused_range_drift_fraction":fused_drift,
            "maximum_warmup_output_difference":maximum_warmup_difference,
            "observations":observations,
            "claim":"Layer-0 component timing only. Both arms use persistent KV surfaces and incremental newest-Q/K/V/mask writes. Baseline includes attention evaluation/read plus output-projection write/evaluation/read; fused includes one evaluation/read. No full-route or promotion claim."
        });
        std::fs::write(
            receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize fused timing receipt"),
        )
        .expect("preserve fused timing receipt");
        assert_eq!(
            compiler_calls, setup_compiler_calls,
            "no compiler call is permitted during warmup or timing"
        );
        assert_eq!(baseline_attention.tokens_seen(), fused.tokens_seen());
    }

    #[test]
    fn interleaved_is_cached_only_and_does_not_change_projection_precision() {
        let plan = super::AneWeightPlan::StaticInt8InterleavedFfnCached;
        assert_eq!(plan.program_count(), 162);
        assert_eq!(
            plan.cache_policy(),
            super::AneProgramCachePolicy::RequireExisting
        );
        assert_eq!(
            plan.static_ffn_precision(),
            super::StaticFfnPrecision::Int8Interleaved
        );
        assert!(!plan.quantizes_qkv(true));
        assert!(!plan.quantizes_qkv(false));
        let result = super::GemmaAneDecode::load_with_compile_budget(
            std::path::Path::new("/must-not-read-interleaved-model"),
            1024,
            plan,
            1,
        );
        assert!(matches!(result, Err(error) if error.contains("zero compile budget")));
    }

    #[test]
    fn transpose_attention_does_not_change_weights_program_count_or_compile_policy() {
        let plan = super::AneWeightPlan::StaticInt8FfnTransposeAttentionCached;
        assert_eq!(plan.program_count(), 162);
        assert_eq!(plan.static_ffn_precision(), super::StaticFfnPrecision::Int8);
        assert!(!plan.quantizes_qkv(true));
        let result = super::GemmaAneDecode::load_with_compile_budget(
            std::path::Path::new("/must-not-read-transpose-model"),
            1024,
            plan,
            1,
        );
        assert!(matches!(result,Err(error) if error.contains("zero compile budget")));
    }

    #[test]
    fn down4_is_one_cached_program_per_layer_and_never_compiles_on_inference() {
        let plan = super::AneWeightPlan::StaticInt8Down4FfnCached;
        assert_eq!(plan.program_count(), 162);
        assert_eq!(
            plan.static_ffn_precision(),
            super::StaticFfnPrecision::Int8Down4
        );
        assert!(!plan.quantizes_qkv(true));
        let result = super::GemmaAneDecode::load_with_compile_budget(
            std::path::Path::new("/must-not-read-down4-model"),
            1024,
            plan,
            1,
        );
        assert!(matches!(result,Err(error) if error.contains("zero compile budget")));
    }

    #[test]
    fn tiles4_is_single_variant_sliding_only_and_cached() {
        let plan = AneWeightPlan::StaticInt8FfnSlidingQkvTiles4Cached;
        assert_eq!(plan.program_count(), 162);
        assert_eq!(plan.cache_policy(), AneProgramCachePolicy::RequireExisting);
        assert_eq!(plan.static_ffn_precision(), StaticFfnPrecision::Int8);
        assert!(plan.quantizes_qkv(true));
        assert!(!plan.quantizes_qkv(false));
        assert!(
            matches!(load_static_qkv_tiles4(&[], 8704, plan.cache_policy()),
            Err(error) if error.contains("8192-row sliding"))
        );
        assert!(matches!(GemmaAneDecode::load_with_compile_budget(
            Path::new("/must-not-read-candidate-checkpoint"), 1024, plan, 1),
            Err(error) if error.contains("zero compile budget")));
    }
    #[test]
    fn chunk4_is_independent_cached_and_never_quantizes_qkv() {
        let plan = AneWeightPlan::StaticInt8Chunk4FfnCached;
        assert_eq!(plan.name(), "static-int8-chunk4-ffn-cached");
        assert_eq!(plan.program_count(), 162);
        assert_eq!(plan.cache_policy(), AneProgramCachePolicy::RequireExisting);
        assert_eq!(plan.static_ffn_precision(), StaticFfnPrecision::Int8Chunk4);
        assert!(!plan.quantizes_qkv(true));
        assert!(!plan.quantizes_qkv(false));
        let result = GemmaAneDecode::load_with_compile_budget(
            Path::new("/must-not-read-candidate-checkpoint"),
            1024,
            plan,
            1,
        );
        assert!(matches!(result, Err(error) if error.contains("zero compile budget")));
    }
    #[test]
    fn sliding_qkv_candidate_preserves_global_precision_and_ffn_policy() {
        let candidate = AneWeightPlan::StaticInt8FfnSlidingQkvCached;
        assert!(candidate.quantizes_qkv(true));
        assert!(!candidate.quantizes_qkv(false));
        assert_eq!(candidate.static_ffn_precision(), StaticFfnPrecision::Int8);
        assert_eq!(
            candidate.cache_policy(),
            AneProgramCachePolicy::RequireExisting
        );
        assert_eq!(candidate.program_count(), 162);
        for baseline in [
            AneWeightPlan::DynamicFfn,
            AneWeightPlan::StaticFfnDynamicOutput,
            AneWeightPlan::StaticAllCached,
            AneWeightPlan::StaticLut4FfnCached,
            AneWeightPlan::StaticInt8FfnCached,
            AneWeightPlan::StaticInt8StackedFfnCached,
            AneWeightPlan::StaticInt8StackedFfnChecked,
        ] {
            assert!(!baseline.quantizes_qkv(true));
            assert!(!baseline.quantizes_qkv(false));
        }
    }

    #[test]
    fn sliding_qkv_candidate_rejects_global_shape_before_device_access() {
        let error = load_static_qkv(&[], 8704, true, AneProgramCachePolicy::RequireExisting)
            .err()
            .expect("global INT8 QKV must not reach the private API");
        assert!(error.contains("8192-row sliding"));
        // Even the admitted geometry still validates the complete weight shape.
        let error = load_static_qkv(&[], 8192, true, AneProgramCachePolicy::RequireExisting)
            .err()
            .expect("missing weights must fail before device access");
        assert!(error.contains("nonempty, 32-aligned matrix"));
    }

    #[test]
    fn stacked_ffn_check_rejects_compilation_before_checkpoint_access() {
        let error = GemmaAneDecode::load_with_compile_budget(
            Path::new("/no-checkpoint-should-be-read"),
            1024,
            AneWeightPlan::StaticInt8StackedFfnChecked,
            1,
        )
        .err()
        .expect("checked route must reject compilation");
        assert!(error.contains("zero compile budget"));
    }

    #[test]
    fn stacked_ffn_check_requires_finite_bit_identical_outputs() {
        let values = [f16::ZERO, f16::from_f32(-1.25), f16::MAX];
        check_ffn_output(47, &values, &values).unwrap();
        assert!(check_ffn_output(0, &values, &values[..2]).is_err());
        let mut changed = values;
        changed[1] = f16::from_bits(changed[1].to_bits() + 1);
        assert!(check_ffn_output(47, &values, &changed)
            .unwrap_err()
            .contains("layer 47 output 1"));
        changed = values;
        changed[0] = f16::NEG_ZERO;
        assert!(check_ffn_output(0, &values, &changed).is_err());
        for invalid in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
            assert!(check_ffn_output(0, &[invalid], &[invalid]).is_err());
        }
    }

    #[test]
    fn cache_cancellation_precedes_checkpoint_access() {
        let absent = Path::new("/no-checkpoint-should-be-read");
        for inspect in [false, true] {
            let result = visit_static_cache(
                absent,
                AneStaticCachePart::QueryKeyValue,
                1024,
                inspect,
                &|| true,
            );
            assert_eq!(result.unwrap_err(), "ANE cache operation cancelled");
        }
    }

    #[test]
    fn cache_inspection_continues_only_for_confirmed_absence() {
        let mut entries = Vec::new();
        super::record_cache_entry(&mut entries, "first".into(), Ok(()), true).unwrap();
        super::record_cache_entry(
            &mut entries,
            "missing".into(),
            Err("ANE model is absent from the current client's compiled cache".into()),
            true,
        )
        .unwrap();
        assert!(entries[0].available);
        assert!(!entries[1].available);
        assert!(super::record_cache_entry(
            &mut entries,
            "bad".into(),
            Err("ANE load failed".into()),
            true
        )
        .is_err());
        assert!(super::record_cache_entry(
            &mut entries,
            "provision".into(),
            Err("ANE model is absent from the current client's compiled cache".into()),
            false
        )
        .is_err());
        assert_eq!(entries.len(), 2);
    }
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    #[ignore = "host-only full checkpoint FFN source audit; requires RVLLM_GEMMA4_MODEL_DIR; no accelerator calls"]
    fn vectorized_int8_source_matches_all_checkpoint_layers() {
        let model = PathBuf::from(std::env::var_os("RVLLM_GEMMA4_MODEL_DIR").expect("model path"));
        let (arch, entries) = validated_weights(&model, 1024).unwrap();
        let mut verified = Vec::new();
        for index in 0..LAYERS {
            let prefix = format!("{}.layers.{index}.mlp", arch.weight_prefix);
            let gate = load_tensor(&entries[&format!("{prefix}.gate_proj.weight")]).unwrap();
            let up = load_tensor(&entries[&format!("{prefix}.up_proj.weight")]).unwrap();
            let down = load_tensor(&entries[&format!("{prefix}.down_proj.weight")]).unwrap();
            let scalar = AneInt8FfnWeights::quantize_scalar_reference(
                &gate,
                &up,
                &down,
                HIDDEN,
                INTERMEDIATE,
            )
            .unwrap();
            let vector =
                AneInt8FfnWeights::quantize(&gate, &up, &down, HIDDEN, INTERMEDIATE).unwrap();
            // All scales are finite and strictly positive by construction,
            // so equality of their FP16 values also implies identical bits.
            assert!(scalar == vector, "INT8 source differs at layer {index}");
            verified.push(index);
            eprintln!("INT8 source identical at layer {index}/{}", LAYERS - 1);
        }
        if let Some(path) = std::env::var_os("RVLLM_INT8_PARITY_RECEIPT") {
            let receipt = serde_json::json!({"schema":"rvllm.gemma12b_int8_source_parity.v1",
                "model_dir":model,"verified_layers":verified,"coefficients_compared":LAYERS*HIDDEN*INTERMEDIATE*3,
                "all_values_and_scale_bits_identical":true,"ane_calls":0,"claim":"Host source parity only; no inference or speed claim."});
            std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        }
    }

    #[test]
    fn embedding_tiles_respect_tensor_offset_rows_and_dtype() {
        for dtype in [DType::Bf16, DType::F16] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&[0xa5; 17]).unwrap();
            let values = [
                0.0, 1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0, 9.0, -10.0, 11.0,
            ];
            for value in values {
                let bits = match dtype {
                    DType::Bf16 => bf16::from_f32(value).to_bits(),
                    _ => f16::from_f32(value).to_bits(),
                };
                file.write_all(&bits.to_le_bytes()).unwrap();
            }
            file.write_all(&[0xa5; 13]).unwrap();
            let info = SafetensorTensorInfo {
                name: "embedding".into(),
                dtype,
                shape: vec![3, 4],
                file: file.path().into(),
                file_offset: 17,
                nbytes: 24,
            };
            let row = load_rows(&info, 1, 1).unwrap();
            assert_eq!(
                row,
                values[4..8]
                    .iter()
                    .map(|&x| f16::from_f32(x))
                    .collect::<Vec<_>>()
            );
            let tail = load_rows(&info, 1, 2).unwrap();
            assert_eq!(
                tail,
                values[4..]
                    .iter()
                    .map(|&x| f16::from_f32(x))
                    .collect::<Vec<_>>()
            );
            assert!(load_rows(&info, 2, 2).is_err());
            assert!(load_rows(&info, usize::MAX, 1).is_err());
            assert_eq!(load_tensor(&info).unwrap().len(), values.len());
        }
    }

    #[test]
    fn conversion_rejects_nonfinite_or_unrepresentable_weights() {
        let mut output = [f16::ZERO; 1];
        for bits in [
            bf16::INFINITY.to_bits(),
            bf16::NAN.to_bits(),
            bf16::MAX.to_bits(),
        ] {
            assert!(decode_f16(&bits.to_le_bytes(), DType::Bf16, &mut output).is_err());
        }
        assert!(decode_f16(&f16::NAN.to_le_bytes(), DType::F16, &mut output).is_err());
        assert!(decode_f16(&[0], DType::F16, &mut output).is_err());
    }
}

#[cfg(test)]
#[path = "gemma_ane_ffn_oracle_tests.rs"]
mod component_oracles;
