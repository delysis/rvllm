//! Opt-in GPU timestamp instrumentation for normal Metal command buffers.
//!
//! This module is absent unless `metal-stage-instrumentation` is enabled, so
//! production dispatches pay no branch, allocation, lock, or encoder cost.

use objc2::runtime::ProtocolObject;
use objc2_foundation::{ns_string, NSRange};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLCounterResultTimestamp,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSet, MTLDevice,
    MTLStorageMode,
};
use serde_json::{json, Value};

const COUNTER_ERROR_VALUE: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalStage {
    Embedding,
    QkvSliding,
    QkvFull,
    AttentionSliding,
    AttentionFull,
    OProjectionSliding,
    OProjectionFull,
    FfnGateUpActivation,
    FfnDown,
    NormResidual,
    LmHead,
}

impl MetalStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::QkvSliding => "qkv_sliding",
            Self::QkvFull => "qkv_full",
            Self::AttentionSliding => "attention_sliding",
            Self::AttentionFull => "attention_full",
            Self::OProjectionSliding => "o_projection_sliding",
            Self::OProjectionFull => "o_projection_full",
            Self::FfnGateUpActivation => "ffn_gate_up_activation",
            Self::FfnDown => "ffn_down",
            Self::NormResidual => "norm_residual",
            Self::LmHead => "lm_head",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Span {
    stage: MetalStage,
    begin: usize,
    end: usize,
}

pub struct MetalStageProfiler {
    samples: objc2::rc::Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    next_sample: usize,
    open: Option<(MetalStage, usize)>,
    spans: Vec<Span>,
}

impl std::fmt::Debug for MetalStageProfiler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalStageProfiler")
            .field("next_sample", &self.next_sample)
            .field("span_count", &self.spans.len())
            .finish_non_exhaustive()
    }
}

impl MetalStageProfiler {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, max_spans: usize) -> Result<Self, String> {
        let set = device
            .counterSets()
            .and_then(|sets| {
                sets.iter()
                    .find(|set| &*set.name() == ns_string!("timestamp"))
            })
            .ok_or_else(|| "Metal timestamp counter set is unavailable".to_owned())?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        unsafe { descriptor.setSampleCount(max_spans.saturating_mul(2)) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|error| format!("create Metal timestamp sample buffer: {error}"))?;
        Ok(Self {
            samples,
            next_sample: 0,
            open: None,
            spans: Vec::with_capacity(max_spans),
        })
    }

    /// Inserts a barriered GPU timestamp. Calls are diagnostic-path-only.
    pub unsafe fn begin(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        stage: MetalStage,
    ) {
        debug_assert!(self.open.is_none());
        let index = self.next_sample;
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("timestamp encoder");
        encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(&self.samples, index, true);
        encoder.endEncoding();
        self.next_sample += 1;
        self.open = Some((stage, index));
    }

    pub unsafe fn end(&mut self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        let (stage, begin) = self.open.take().expect("unmatched Metal stage end");
        let end = self.next_sample;
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("timestamp encoder");
        encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(&self.samples, end, true);
        encoder.endEncoding();
        self.next_sample += 1;
        self.spans.push(Span { stage, begin, end });
    }

    /// Resolve only after the owning command buffer has completed.
    pub unsafe fn receipt(&self) -> Result<Value, String> {
        if self.open.is_some() {
            return Err("unclosed Metal stage span".into());
        }
        let data = self
            .samples
            .resolveCounterRange(NSRange::new(0, self.next_sample))
            .ok_or_else(|| "Metal timestamp resolve returned no data".to_owned())?;
        let expected = self.next_sample * std::mem::size_of::<MTLCounterResultTimestamp>();
        if data.length() != expected {
            return Err(format!(
                "Metal timestamp byte count mismatch: expected {expected}, got {}",
                data.length()
            ));
        }
        let mut timestamps = vec![MTLCounterResultTimestamp { timestamp: 0 }; self.next_sample];
        data.getBytes_length(
            std::ptr::NonNull::new(timestamps.as_mut_ptr().cast()).unwrap(),
            expected,
        );
        Ok(build_receipt(&self.spans, &timestamps))
    }
}

fn build_receipt(spans: &[Span], timestamps: &[MTLCounterResultTimestamp]) -> Value {
    let mut totals = std::collections::BTreeMap::<&'static str, u64>::new();
    let stages = spans
        .iter()
        .map(|span| {
            let start = timestamps[span.begin].timestamp;
            let end = timestamps[span.end].timestamp;
            let duration_ns =
                if start != COUNTER_ERROR_VALUE && end != COUNTER_ERROR_VALUE && end >= start {
                    Some(end - start)
                } else {
                    None
                };
            if let Some(duration_ns) = duration_ns {
                *totals.entry(span.stage.name()).or_default() += duration_ns;
            }
            json!({"stage": span.stage.name(), "gpu_duration_ns": duration_ns})
        })
        .collect::<Vec<_>>();
    json!({
            "schema": "rvllm.metal_stage_timing.v1",
            "clock": "MTLCommonCounterSetTimestamp",
            "barriered": true,
            "totals_gpu_duration_ns": totals,
            "stages": stages,
            "caveats": [
                "qkv includes fused Q/K normalization, RoPE, and cache writes when selected",
                "o_projection may include fused post-attention normalization",
                "o_projection also includes the attention residual and pre-FFN normalization boundary",
                "ffn_down may include fused post-FFN normalization",
                "MoE routing, expert gate/up, expert down, and normalization are inseparable and reported under ffn_down",
                "lm_head includes final normalization and sampling/logit finalization"
            ]
    })
}

#[cfg(test)]
mod tests {
    use super::{build_receipt, MetalStage, Span, COUNTER_ERROR_VALUE};
    use objc2_metal::MTLCounterResultTimestamp;
    #[test]
    fn stage_names_are_stable_receipt_keys() {
        assert_eq!(MetalStage::QkvSliding.name(), "qkv_sliding");
        assert_eq!(MetalStage::LmHead.name(), "lm_head");
    }

    #[test]
    fn receipt_is_strict_and_fails_closed_on_bad_samples() {
        let spans = [
            Span {
                stage: MetalStage::Embedding,
                begin: 0,
                end: 1,
            },
            Span {
                stage: MetalStage::LmHead,
                begin: 2,
                end: 3,
            },
        ];
        let samples = [10, 30, COUNTER_ERROR_VALUE, 90]
            .map(|timestamp| MTLCounterResultTimestamp { timestamp });
        let receipt = build_receipt(&spans, &samples);
        assert_eq!(receipt["schema"], "rvllm.metal_stage_timing.v1");
        assert_eq!(receipt["stages"][0]["gpu_duration_ns"], 20);
        assert!(receipt["stages"][1]["gpu_duration_ns"].is_null());
    }
}
