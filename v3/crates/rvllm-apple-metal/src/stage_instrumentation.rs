//! Opt-in GPU timestamp instrumentation for normal Metal command buffers.
//!
//! This module is absent unless `metal-stage-instrumentation` is enabled, so
//! production dispatches pay no branch, allocation, lock, or encoder cost.

use objc2::runtime::ProtocolObject;
use objc2_foundation::{ns_string, NSRange};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLStorageMode,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplingMode {
    StageBoundary,
    DispatchBoundary,
}

fn select_sampling_mode(
    stage_boundary: bool,
    dispatch_boundary: bool,
) -> Result<SamplingMode, String> {
    if stage_boundary {
        Ok(SamplingMode::StageBoundary)
    } else if dispatch_boundary {
        Ok(SamplingMode::DispatchBoundary)
    } else {
        Err(
            "Metal timestamp sampling is unsupported at compute stage and dispatch boundaries"
                .into(),
        )
    }
}

pub struct MetalStageProfiler {
    samples: objc2::rc::Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    next_sample: usize,
    open: Option<(MetalStage, usize)>,
    spans: Vec<Span>,
    sampling: SamplingMode,
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
        let sampling = select_sampling_mode(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
        )?;
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
            sampling,
        })
    }

    unsafe fn sample(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>, index: usize) {
        let encoder = match self.sampling {
            SamplingMode::StageBoundary => {
                let descriptor = MTLComputePassDescriptor::computePassDescriptor();
                let attachment = descriptor
                    .sampleBufferAttachments()
                    .objectAtIndexedSubscript(0);
                attachment.setSampleBuffer(Some(&self.samples));
                attachment.setStartOfEncoderSampleIndex(usize::MAX);
                attachment.setEndOfEncoderSampleIndex(index);
                command_buffer
                    .computeCommandEncoderWithDescriptor(&descriptor)
                    .expect("stage-boundary timestamp encoder")
            }
            SamplingMode::DispatchBoundary => {
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .expect("dispatch-boundary timestamp encoder");
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &self.samples,
                    index,
                    true,
                );
                encoder
            }
        };
        encoder.endEncoding();
    }

    /// Inserts a barriered GPU timestamp. Calls are diagnostic-path-only.
    pub unsafe fn begin(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        stage: MetalStage,
    ) {
        debug_assert!(self.open.is_none());
        let index = self.next_sample;
        self.sample(command_buffer, index);
        self.next_sample += 1;
        self.open = Some((stage, index));
    }

    pub unsafe fn end(&mut self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        let (stage, begin) = self.open.take().expect("unmatched Metal stage end");
        let end = self.next_sample;
        self.sample(command_buffer, end);
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
        Ok(build_receipt(&self.spans, &timestamps, self.sampling))
    }
}

fn build_receipt(
    spans: &[Span],
    timestamps: &[MTLCounterResultTimestamp],
    sampling: SamplingMode,
) -> Value {
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
            "sampling_point": match sampling { SamplingMode::StageBoundary => "compute_stage_boundary", SamplingMode::DispatchBoundary => "compute_dispatch_boundary" },
            "barriered": matches!(sampling, SamplingMode::DispatchBoundary),
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
    use super::{
        build_receipt, select_sampling_mode, MetalStage, SamplingMode, Span, COUNTER_ERROR_VALUE,
    };
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
        let receipt = build_receipt(&spans, &samples, SamplingMode::StageBoundary);
        assert_eq!(receipt["schema"], "rvllm.metal_stage_timing.v1");
        assert_eq!(receipt["stages"][0]["gpu_duration_ns"], 20);
        assert!(receipt["stages"][1]["gpu_duration_ns"].is_null());
    }

    #[test]
    fn sampling_capability_selection_never_calls_an_unsupported_path() {
        assert_eq!(
            select_sampling_mode(true, false).unwrap(),
            SamplingMode::StageBoundary
        );
        assert_eq!(
            select_sampling_mode(false, true).unwrap(),
            SamplingMode::DispatchBoundary
        );
        assert!(select_sampling_mode(false, false).is_err());
    }

    #[test]
    fn live_device_sampling_mode_is_feature_probed() {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandQueue, MTLCounterSamplingPoint,
            MTLCreateSystemDefaultDevice, MTLDevice,
        };
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let selected = select_sampling_mode(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
        );
        assert!(
            selected.is_ok(),
            "live Metal device exposes no timestamp sampling point"
        );
        if let Ok(SamplingMode::StageBoundary) = selected {
            assert!(device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary));
        } else if let Ok(SamplingMode::DispatchBoundary) = selected {
            assert!(device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary));
        }
        let queue = device.newCommandQueue().expect("command queue");
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let mut profiler = super::MetalStageProfiler::new(&device, 1).expect("stage profiler");
        unsafe {
            profiler.begin(&command_buffer, MetalStage::Embedding);
            profiler.end(&command_buffer);
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let receipt = unsafe { profiler.receipt() }.expect("timestamp receipt");
        assert_eq!(receipt["stages"][0]["stage"], "embedding");
    }
}
