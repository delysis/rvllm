//! Compute pipeline state object cache.
//!
//! Compiles Metal functions to PSOs once at init. PSOs are keyed by
//! function name. No compilation happens during inference.

use crate::context::MetalContext;
use crate::research_evidence::{
    ResearchDispatchCounters, ResearchDispatchSnapshot, ResearchKernel,
};
use crate::{MetalFloatType, MetalKernelOptions};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
use rvllm_apple::device::AppleGpuFamily;
use rvllm_apple::{AppleLowBitTensorRole, AppleLowBitWeightFormat};
use rvllm_core::Result;
use std::cell::Cell;
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LowBitDispatchSnapshot {
    pub counts: [u64; AppleLowBitTensorRole::COUNT * 2],
    pub overflowed: bool,
}

impl LowBitDispatchSnapshot {
    pub fn checked_since(self, earlier: Self) -> core::result::Result<Self, &'static str> {
        if self.overflowed || earlier.overflowed {
            return Err("low-bit dispatch counter overflow");
        }
        let mut counts = [0; AppleLowBitTensorRole::COUNT * 2];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = self.counts[index]
                .checked_sub(earlier.counts[index])
                .ok_or("low-bit dispatch counters reset within a request")?;
        }
        Ok(Self {
            counts,
            overflowed: false,
        })
    }
    pub fn count(&self, format: AppleLowBitWeightFormat, role: AppleLowBitTensorRole) -> u64 {
        let format_base = match format {
            AppleLowBitWeightFormat::W4A16 => 0,
            AppleLowBitWeightFormat::W8A16 => AppleLowBitTensorRole::COUNT,
        };
        self.counts[format_base + role.index()]
    }

    /// Require one exact low-bit route. This rejects both missing dispatches
    /// and unexpected work in another role or format, so a qualification
    /// receipt cannot mistake a partial/fallback route for the requested one.
    pub fn verify_exact(
        &self,
        format: AppleLowBitWeightFormat,
        expected_by_role: [u64; AppleLowBitTensorRole::COUNT],
    ) -> core::result::Result<(), &'static str> {
        if self.overflowed {
            return Err("low-bit dispatch counter overflow");
        }
        for role_index in 0..AppleLowBitTensorRole::COUNT {
            let expected = expected_by_role[role_index];
            let base = match format {
                AppleLowBitWeightFormat::W4A16 => 0,
                AppleLowBitWeightFormat::W8A16 => AppleLowBitTensorRole::COUNT,
            };
            if self.counts[base + role_index] != expected {
                return Err("low-bit dispatch ledger does not match the required role counts");
            }
            let other_base = if base == 0 {
                AppleLowBitTensorRole::COUNT
            } else {
                0
            };
            if self.counts[other_base + role_index] != 0 {
                return Err("low-bit dispatch ledger contains an unexpected weight format");
            }
        }
        Ok(())
    }
}

/// Cached compute pipeline state objects, keyed by function name.
pub struct PipelineCache {
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    gpu_family: AppleGpuFamily,
    float_type: Option<MetalFloatType>,
    kernel_options: MetalKernelOptions,
    max_threadgroup_memory: usize,
    research_dispatches: ResearchDispatchCounters,
    low_bit_dispatches: Cell<[u64; AppleLowBitTensorRole::COUNT * 2]>,
    low_bit_dispatch_overflowed: Cell<bool>,
    // Physical encoder correction for the current synchronously encoded layer.
    // Only the explicit donor family writes this; no atomics or allocation.
    donor_layer_encoder_correction: Cell<i64>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::with_kernel_options(MetalKernelOptions::from_development_environment())
    }

    #[must_use]
    pub fn with_kernel_options(kernel_options: MetalKernelOptions) -> Self {
        Self {
            pipelines: HashMap::new(),
            gpu_family: AppleGpuFamily::Unknown,
            float_type: None,
            kernel_options,
            max_threadgroup_memory: 0,
            research_dispatches: ResearchDispatchCounters::default(),
            low_bit_dispatches: Cell::new([0; AppleLowBitTensorRole::COUNT * 2]),
            low_bit_dispatch_overflowed: Cell::new(false),
            donor_layer_encoder_correction: Cell::new(0),
        }
    }

    #[must_use]
    pub const fn kernel_options(&self) -> MetalKernelOptions {
        self.kernel_options
    }

    /// Cumulative encoded work. Read only outside active encoding; pair with
    /// successful command-buffer collection before calling it completed work.
    #[must_use]
    pub fn research_dispatch_snapshot(&self) -> ResearchDispatchSnapshot {
        self.research_dispatches.snapshot()
    }

    pub(crate) fn reset_donor_layer_encoder_correction(&self) {
        self.donor_layer_encoder_correction.set(0);
    }

    pub(crate) fn add_donor_layer_encoder_correction(&self, delta: i64) {
        // Bounded to a handful of encoders per layer; reset before encoding.
        self.donor_layer_encoder_correction
            .set(self.donor_layer_encoder_correction.get() + delta);
    }

    /// Adjustment to the legacy logical encoder estimator, for the just
    /// encoded layer only. This is not a GPU-completion or timing receipt.
    pub fn donor_layer_encoder_correction(&self) -> i64 {
        if crate::donor12b::simdgroups(self.kernel_options.research).is_some() {
            self.donor_layer_encoder_correction.get()
        } else {
            0
        }
    }

    pub(crate) fn record_research_dispatch(&self, kernel: ResearchKernel) {
        self.research_dispatches.record(kernel);
    }

    pub fn low_bit_dispatch_snapshot(&self) -> LowBitDispatchSnapshot {
        LowBitDispatchSnapshot {
            counts: self.low_bit_dispatches.get(),
            overflowed: self.low_bit_dispatch_overflowed.get(),
        }
    }

    pub(crate) fn record_low_bit_dispatch(
        &self,
        format: AppleLowBitWeightFormat,
        role: AppleLowBitTensorRole,
    ) {
        let mut counts = self.low_bit_dispatches.get();
        let base = match format {
            AppleLowBitWeightFormat::W4A16 => 0,
            AppleLowBitWeightFormat::W8A16 => AppleLowBitTensorRole::COUNT,
        };
        let slot = &mut counts[base + role.index()];
        if let Some(next) = slot.checked_add(1) {
            *slot = next;
        } else {
            self.low_bit_dispatch_overflowed.set(true);
        }
        self.low_bit_dispatches.set(counts);
    }

    /// Compile a named function from the context's library into a PSO.
    pub fn compile(&mut self, ctx: &MetalContext, function_name: &str) -> Result<()> {
        self.float_type = None;
        let pso = ctx.make_pipeline(function_name)?;
        self.pipelines.insert(function_name.to_owned(), pso);
        tracing::debug!(function = function_name, "Compiled Metal PSO");
        Ok(())
    }

    /// Compile one typed library. Piecemeal replacement clears its dtype so
    /// dtype-specific optimizations cannot use an unclassified pipeline cache.
    pub fn compile_all_for_type(
        &mut self,
        ctx: &MetalContext,
        dtype: MetalFloatType,
    ) -> Result<()> {
        self.compile_all(ctx)?;
        self.float_type = Some(dtype);
        Ok(())
    }

    #[must_use]
    pub const fn float_type(&self) -> Option<MetalFloatType> {
        self.float_type
    }

    /// Compile all required kernel functions for inference.
    pub fn compile_all(&mut self, ctx: &MetalContext) -> Result<()> {
        self.gpu_family = ctx.capabilities().gpu_family;
        self.max_threadgroup_memory = ctx.capabilities().max_threadgroup_memory_length;
        let required = crate::kernels::KERNEL_NAMES;
        for name in required {
            self.compile(ctx, name)?;
        }
        // Optional source may be absent from an explicitly supplied metallib.
        // Clear any old PSO before attempting replacement: a caught failure must
        // not leave a previous library's dtype or executable active.
        for kernel in self.kernel_options.research.kernels() {
            let name = kernel.name();
            self.pipelines.remove(name);
            match self.compile(ctx, name) {
                Ok(()) => tracing::info!(
                    candidate = self.kernel_options.research.name(),
                    function = name,
                    "Research PSO compiled; not hardware-qualified"
                ),
                Err(error) => tracing::warn!(candidate = self.kernel_options.research.name(),
                    function = name, %error, "Research PSO unavailable; known-good fallback retained"),
            }
        }
        tracing::info!(count = required.len(), "All required Metal PSOs compiled");
        Ok(())
    }

    /// No platform assumptions are inferred from a product name. A missing,
    /// piecemeal, untyped, wrong-family, or resource-incompatible PSO is refused.
    pub(crate) fn research_pso(
        &self,
        name: &str,
        threads: usize,
        planned_bytes: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if self.gpu_family != AppleGpuFamily::Apple9
            || self.float_type.is_none()
            || self.kernel_options.quantized_bf16_accumulation
            || !self
                .kernel_options
                .research
                .kernels()
                .iter()
                .any(|kernel| kernel.name() == name && kernel.limits() == (threads, planned_bytes))
        {
            return None;
        }
        let pso = self.pipelines.get(name)?;
        crate::research::launch_fits(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            pso.staticThreadgroupMemoryLength(),
            self.max_threadgroup_memory,
            threads,
            planned_bytes,
        )
        .then_some(pso)
    }

    /// Get a cached PSO by name.
    pub fn get(
        &self,
        name: &str,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.pipelines.get(name).ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::PipelineMissing { name: "unknown" },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "get_pso",
                    device: "apple-silicon",
                },
            )
        })
    }

    pub fn len(&self) -> usize {
        self.pipelines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Live GPU family captured when the inference PSOs were compiled.
    ///
    /// Shape-specialized dispatch must fail back to portable kernels if the
    /// cache was assembled piecemeal and therefore has no device identity.
    #[must_use]
    pub const fn gpu_family(&self) -> AppleGpuFamily {
        self.gpu_family
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("count", &self.pipelines.len())
            .field("functions", &self.pipelines.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_bit_dispatch_ledger_separates_roles_and_formats() {
        let cache = PipelineCache::default();
        let before = cache.low_bit_dispatch_snapshot();
        cache.record_low_bit_dispatch(
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitTensorRole::QueryProjection,
        );
        cache.record_low_bit_dispatch(
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitTensorRole::DenseDownProjection,
        );
        let snapshot = cache.low_bit_dispatch_snapshot();
        let delta = snapshot
            .clone()
            .checked_since(before)
            .expect("checked delta");
        assert_eq!(
            snapshot.count(
                AppleLowBitWeightFormat::W4A16,
                AppleLowBitTensorRole::QueryProjection
            ),
            1
        );
        assert_eq!(
            snapshot.count(
                AppleLowBitWeightFormat::W8A16,
                AppleLowBitTensorRole::QueryProjection
            ),
            0
        );

        let mut expected = [0; AppleLowBitTensorRole::COUNT];
        expected[AppleLowBitTensorRole::QueryProjection.index()] = 1;
        expected[AppleLowBitTensorRole::DenseDownProjection.index()] = 1;
        assert!(snapshot
            .verify_exact(AppleLowBitWeightFormat::W4A16, expected)
            .is_ok());
        assert!(snapshot
            .verify_exact(AppleLowBitWeightFormat::W8A16, expected)
            .is_err());
        assert_eq!(delta, snapshot);
        let reset = LowBitDispatchSnapshot {
            counts: [0; AppleLowBitTensorRole::COUNT * 2],
            overflowed: false,
        };
        assert!(reset.checked_since(snapshot).is_err());
    }
}
