//! Metal device context: device discovery, command queue, and library management.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLibrary,
};
use rvllm_apple::device::{AppleAcceleratorTarget, AppleGpuFamily};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

use crate::memory_budget::{
    AppleMemoryBudget, AppleMemoryBudgetError, AppleMemoryBudgetInput, AppleMemoryPlatform,
};

/// Metal device context. Owns the device, command queue, and compiled
/// shader library. Created once at engine init; shared (immutably) by
/// all inference operations.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    target: AppleAcceleratorTarget,
    capabilities: MetalDeviceCapabilities,
}

/// Capabilities queried from the live `MTLDevice` rather than inferred from a
/// marketing name. These values drive portable macOS/iOS memory and launch
/// policy without pretending that a particular product string is exhaustive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalDeviceCapabilities {
    pub gpu_family: AppleGpuFamily,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_size: u64,
    pub max_threadgroup_memory_length: usize,
}

impl MetalDeviceCapabilities {
    /// Metal bytes available to inference after platform-default headroom.
    #[must_use]
    pub const fn default_inference_budget_bytes(self) -> u64 {
        let reserve_percent = if cfg!(target_os = "ios") {
            AppleMemoryPlatform::Ios.default_reserve_percent()
        } else {
            AppleMemoryPlatform::MacOs.default_reserve_percent()
        };
        self.recommended_max_working_set_size
            .saturating_mul((100 - reserve_percent) as u64)
            / 100
    }
}

fn highest_supported_apple_gpu_family(
    mut supports: impl FnMut(MTLGPUFamily) -> bool,
) -> AppleGpuFamily {
    if supports(MTLGPUFamily::Apple10) {
        AppleGpuFamily::Apple10
    } else if supports(MTLGPUFamily::Apple9) {
        AppleGpuFamily::Apple9
    } else if supports(MTLGPUFamily::Apple8) {
        AppleGpuFamily::Apple8
    } else if supports(MTLGPUFamily::Apple7) {
        AppleGpuFamily::Apple7
    } else {
        AppleGpuFamily::Unknown
    }
}

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "metal",
        op,
        device: "apple-silicon",
    }
}

impl MetalContext {
    /// Create a Metal context using the system default device.
    ///
    /// Fails with `MetalUnavailable` if no Metal device is found, or if
    /// the GPU family is below Apple7 (pre-M1 hardware).
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RvllmError::apple(AppleError::MetalUnavailable, ctx("init")))?;

        let name = device.name().to_string();
        let gpu_family = highest_supported_apple_gpu_family(|family| device.supportsFamily(family));
        let mut target = AppleAcceleratorTarget::from_device_name(&name, 1);
        target.gpu_family = gpu_family;
        target.architecture_gen = gpu_family.architecture_gen();
        target.has_nax = gpu_family.has_nax();

        if gpu_family == AppleGpuFamily::Unknown {
            return Err(RvllmError::apple(
                AppleError::UnsupportedDevice {
                    name: "metal_gpu_family_below_apple7",
                },
                ctx("init"),
            ));
        }

        let capabilities = MetalDeviceCapabilities {
            gpu_family,
            has_unified_memory: device.hasUnifiedMemory(),
            recommended_max_working_set_size: device.recommendedMaxWorkingSetSize(),
            max_threadgroup_memory_length: device.maxThreadgroupMemoryLength(),
        };

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RvllmError::apple(AppleError::MetalUnavailable, ctx("create_queue")))?;

        tracing::info!(
            device = %name,
            gpu_family = ?target.gpu_family,
            tier = ?target.tier,
            ane_cores = target.ane_cores,
            "Metal context initialized"
        );

        Ok(Self {
            device,
            queue,
            library: None,
            target,
            capabilities,
        })
    }

    /// Compile Metal Shading Language source into a library.
    pub fn compile_library(&mut self, source: &str) -> Result<()> {
        let ns_source = NSString::from_str(source);
        let lib = self
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| {
                tracing::error!(error = %e, "Metal shader compilation failed");
                RvllmError::apple(
                    AppleError::MilCompileFailed {
                        procedure: "metallib",
                    },
                    ctx("compile_library"),
                )
            })?;
        self.library = Some(lib);
        Ok(())
    }

    /// Load a pre-compiled .metallib file.
    pub fn load_metallib(&mut self, path: &std::path::Path) -> Result<()> {
        let ns_path = NSString::from_str(&path.to_string_lossy());
        let url = objc2_foundation::NSURL::fileURLWithPath(&ns_path);
        let lib = self.device.newLibraryWithURL_error(&url).map_err(|_| {
            RvllmError::apple(
                AppleError::MetallibMissing {
                    path: path.to_path_buf(),
                },
                ctx("load_metallib"),
            )
        })?;
        self.library = Some(lib);
        Ok(())
    }

    /// Create a compute pipeline state object from a named function.
    pub fn make_pipeline(
        &self,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let lib = self.library.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::PipelineMissing { name: "no_library" },
                ctx("make_pipeline"),
            )
        })?;
        let ns_name = NSString::from_str(function_name);
        let func = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::PipelineMissing { name: "unknown" },
                ctx("get_function"),
            )
        })?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|_| {
                RvllmError::apple(
                    AppleError::PipelineMissing { name: "unknown" },
                    ctx("compile_pso"),
                )
            })?;
        Ok(pso)
    }

    #[inline]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &*self.device
    }

    #[inline]
    pub fn device_retained(&self) -> &Retained<ProtocolObject<dyn MTLDevice>> {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &*self.queue
    }

    #[inline]
    pub fn queue_retained(&self) -> &Retained<ProtocolObject<dyn MTLCommandQueue>> {
        &self.queue
    }

    #[inline]
    pub fn library(&self) -> Option<&ProtocolObject<dyn MTLLibrary>> {
        self.library.as_deref()
    }

    #[inline]
    pub fn target(&self) -> &AppleAcceleratorTarget {
        &self.target
    }

    #[inline]
    pub fn capabilities(&self) -> MetalDeviceCapabilities {
        self.capabilities
    }

    #[inline]
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.capabilities.recommended_max_working_set_size
    }

    /// Derive independent weight, three-slot scratch, metadata, and paged-KV
    /// accounts from the live device working-set recommendation.
    pub fn memory_budget(
        &self,
        weights_bytes: u64,
        scratch_slot_bytes: u64,
        metadata_bytes: u64,
    ) -> std::result::Result<AppleMemoryBudget, AppleMemoryBudgetError> {
        let platform =
            AppleMemoryPlatform::current().ok_or(AppleMemoryBudgetError::MissingWorkingSetLimit)?;
        AppleMemoryBudget::derive(AppleMemoryBudgetInput::with_platform_defaults(
            platform,
            self.recommended_max_working_set_size(),
            weights_bytes,
            scratch_slot_bytes,
            metadata_bytes,
        ))
    }

    #[inline]
    pub fn default_inference_budget_bytes(&self) -> u64 {
        self.capabilities.default_inference_budget_bytes()
    }

    /// Maximum threadgroup memory in bytes (Apple9: 32KB, Apple10+: 64KB).
    pub fn max_threadgroup_memory(&self) -> usize {
        self.capabilities.max_threadgroup_memory_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_family_selection_uses_highest_supported_live_family() {
        assert_eq!(
            highest_supported_apple_gpu_family(|family| family <= MTLGPUFamily::Apple9),
            AppleGpuFamily::Apple9
        );
        assert_eq!(
            highest_supported_apple_gpu_family(|_| false),
            AppleGpuFamily::Unknown
        );
    }

    #[test]
    fn working_set_budget_preserves_platform_headroom() {
        let capabilities = MetalDeviceCapabilities {
            gpu_family: AppleGpuFamily::Apple9,
            has_unified_memory: true,
            recommended_max_working_set_size: 1_000,
            max_threadgroup_memory_length: 32_768,
        };
        assert_eq!(
            capabilities.default_inference_budget_bytes(),
            1_000
                * (100
                    - AppleMemoryPlatform::current()
                        .expect("Apple-only context test")
                        .default_reserve_percent() as u64)
                / 100
        );
    }
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.target.device_name)
            .field("gpu_family", &self.target.gpu_family)
            .field("tier", &self.target.tier)
            .finish()
    }
}
