use std::path::PathBuf;

use rvllm_core::{AppleCtx, AppleError, ConfigError, Result, RvllmError};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MetalPrefillBackend {
    MlxPrototype,
    DirectMetal,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalPrefillConfig {
    pub backend: MetalPrefillBackend,
    pub metallib_path: Option<PathBuf>,
    pub max_prompt_tokens: usize,
    pub max_batch: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrefillContract {
    pub command_buffers_per_layer_group: u32,
    pub allocates_in_hot_path: bool,
    pub owns_kv_cache_write: bool,
    pub uses_persistent_parameter_buffers: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum DirectMetalPipelineName {
    RmsNorm,
    Matmul,
    Rope,
    Attention,
}

impl DirectMetalPipelineName {
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            DirectMetalPipelineName::RmsNorm => "rvllm_rms_norm",
            DirectMetalPipelineName::Matmul => "rvllm_matmul",
            DirectMetalPipelineName::Rope => "rvllm_rope",
            DirectMetalPipelineName::Attention => "rvllm_attention",
        }
    }
}

pub const DIRECT_METAL_PREFILL_PIPELINES: [DirectMetalPipelineName; 4] = [
    DirectMetalPipelineName::RmsNorm,
    DirectMetalPipelineName::Matmul,
    DirectMetalPipelineName::Rope,
    DirectMetalPipelineName::Attention,
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DirectMetalContextConfig {
    prefill: MetalPrefillConfig,
    contract: PrefillContract,
    pipeline_names: [DirectMetalPipelineName; 4],
}

impl MetalPrefillConfig {
    #[must_use]
    pub const fn direct_contract() -> PrefillContract {
        PrefillContract {
            command_buffers_per_layer_group: 1,
            allocates_in_hot_path: false,
            owns_kv_cache_write: true,
            uses_persistent_parameter_buffers: true,
        }
    }

    #[must_use]
    pub const fn mlx_prototype_contract() -> PrefillContract {
        PrefillContract {
            command_buffers_per_layer_group: 0,
            allocates_in_hot_path: true,
            owns_kv_cache_write: false,
            uses_persistent_parameter_buffers: false,
        }
    }
}

impl DirectMetalContextConfig {
    pub fn new(prefill: MetalPrefillConfig) -> Result<Self> {
        Self::validate_prefill(&prefill)?;
        let config = Self {
            prefill,
            contract: MetalPrefillConfig::direct_contract(),
            pipeline_names: DIRECT_METAL_PREFILL_PIPELINES,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        Self::validate_prefill(&self.prefill)?;
        if self.contract != MetalPrefillConfig::direct_contract() {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "contract",
                    reason: "direct Metal context must use the direct prefill contract".to_string(),
                },
                "contract",
            ));
        }
        if self.pipeline_names != DIRECT_METAL_PREFILL_PIPELINES {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "pipeline_names",
                    reason: "direct Metal context must use the prefill pipeline set".to_string(),
                },
                "pipeline_names",
            ));
        }
        Ok(())
    }

    fn validate_prefill(prefill: &MetalPrefillConfig) -> Result<()> {
        if prefill.backend != MetalPrefillBackend::DirectMetal {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "mlx-prototype",
                    op: "direct_metal_context",
                },
                metal_ctx("config"),
            ));
        }
        if prefill.metallib_path.is_none() {
            return Err(RvllmError::config(
                ConfigError::MissingField {
                    name: "metallib_path",
                },
                "metallib_path",
            ));
        }
        if prefill.max_prompt_tokens == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "max_prompt_tokens",
                    reason: "must be greater than zero".to_string(),
                },
                "max_prompt_tokens",
            ));
        }
        if prefill.max_batch == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "max_batch",
                    reason: "must be greater than zero".to_string(),
                },
                "max_batch",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub const fn prefill(&self) -> &MetalPrefillConfig {
        &self.prefill
    }

    #[must_use]
    pub const fn contract(&self) -> &PrefillContract {
        &self.contract
    }

    #[must_use]
    pub const fn pipeline_names(&self) -> &[DirectMetalPipelineName; 4] {
        &self.pipeline_names
    }

    #[must_use]
    pub fn metallib_path(&self) -> Result<&std::path::Path> {
        match self.prefill.metallib_path.as_deref() {
            Some(path) => Ok(path),
            None => Err(RvllmError::config(
                ConfigError::MissingField {
                    name: "metallib_path",
                },
                "metallib_path",
            )),
        }
    }
}

fn metal_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "direct-metal",
        op,
        device: "apple-silicon",
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct DirectMetalContext {
    config: DirectMetalContextConfig,
    device_name: String,
    handles: direct_metal_ffi::DirectMetalHandles,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl DirectMetalContext {
    pub fn new(config: DirectMetalContextConfig) -> Result<Self> {
        config.validate()?;
        let handles = direct_metal_ffi::DirectMetalHandles::new(&config)?;
        let device_name = handles.device_name().to_string();
        Ok(Self {
            config,
            device_name,
            handles,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &DirectMetalContextConfig {
        &self.config
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[must_use]
    pub fn loaded_pipeline_count(&self) -> usize {
        self.handles.loaded_pipeline_count()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod direct_metal_ffi {
    #![allow(unsafe_code)]

    use std::collections::BTreeMap;
    use std::path::Path;

    use objc2::msg_send;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_foundation::NSString;
    use rvllm_core::{AppleError, Result, RvllmError};

    use super::{metal_ctx, DirectMetalContextConfig, DirectMetalPipelineName};

    #[link(name = "Metal", kind = "framework")]
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut AnyObject;
    }

    pub struct DirectMetalHandles {
        device_name: String,
        _device: Retained<AnyObject>,
        _command_queue: Retained<AnyObject>,
        _library: Retained<AnyObject>,
        pipelines: BTreeMap<DirectMetalPipelineName, Retained<AnyObject>>,
    }

    impl DirectMetalHandles {
        pub fn new(config: &DirectMetalContextConfig) -> Result<Self> {
            let metallib_path = config.metallib_path()?;
            ensure_metallib_exists(metallib_path)?;
            let device = system_default_device()?;
            let device_name = objc_device_name(&device);
            let command_queue = new_command_queue(&device)?;
            let library = new_library_with_file(&device, metallib_path)?;
            let pipelines = new_compute_pipelines(&device, &library, config.pipeline_names())?;

            Ok(Self {
                device_name,
                _device: device,
                _command_queue: command_queue,
                _library: library,
                pipelines,
            })
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        pub fn loaded_pipeline_count(&self) -> usize {
            self.pipelines.len()
        }
    }

    fn ensure_metallib_exists(path: &Path) -> Result<()> {
        if path.is_file() {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::MetallibMissing {
                    path: path.to_path_buf(),
                },
                metal_ctx("load_metallib"),
            ))
        }
    }

    fn system_default_device() -> Result<Retained<AnyObject>> {
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        unsafe { Retained::from_raw(raw) }.ok_or_else(|| {
            RvllmError::apple(AppleError::MetalUnavailable, metal_ctx("create_device"))
        })
    }

    fn objc_device_name(device: &AnyObject) -> String {
        let raw: *mut NSString = unsafe { msg_send![device, name] };
        if raw.is_null() {
            String::new()
        } else {
            unsafe { &*raw }.to_string()
        }
    }

    fn new_command_queue(device: &AnyObject) -> Result<Retained<AnyObject>> {
        let queue: Option<Retained<AnyObject>> = unsafe { msg_send![device, newCommandQueue] };
        queue.ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                metal_ctx("create_command_queue"),
            )
        })
    }

    fn new_library_with_file(device: &AnyObject, path: &Path) -> Result<Retained<AnyObject>> {
        let path_buf = path.to_path_buf();
        let Some(path) = path.to_str() else {
            return Err(RvllmError::apple(
                AppleError::MetallibMissing { path: path_buf },
                metal_ctx("load_metallib"),
            ));
        };
        let path = NSString::from_str(path);
        let mut error: *mut AnyObject = std::ptr::null_mut();
        let library: Option<Retained<AnyObject>> =
            unsafe { msg_send![device, newLibraryWithFile: &*path, error: &mut error] };
        library.ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetallibMissing { path: path_buf },
                metal_ctx("load_metallib"),
            )
        })
    }

    fn new_compute_pipelines(
        device: &AnyObject,
        library: &AnyObject,
        pipeline_names: &[DirectMetalPipelineName; 4],
    ) -> Result<BTreeMap<DirectMetalPipelineName, Retained<AnyObject>>> {
        let mut pipelines = BTreeMap::new();
        for name in pipeline_names {
            let symbol = NSString::from_str(name.symbol());
            let function: Option<Retained<AnyObject>> =
                unsafe { msg_send![library, newFunctionWithName: &*symbol] };
            let Some(function) = function else {
                return Err(RvllmError::apple(
                    AppleError::PipelineMissing {
                        name: name.symbol(),
                    },
                    metal_ctx("load_pipeline"),
                ));
            };

            let mut error: *mut AnyObject = std::ptr::null_mut();
            let pipeline: Option<Retained<AnyObject>> = unsafe {
                msg_send![device, newComputePipelineStateWithFunction: &*function, error: &mut error]
            };
            let Some(pipeline) = pipeline else {
                return Err(RvllmError::apple(
                    AppleError::PipelineMissing {
                        name: name.symbol(),
                    },
                    metal_ctx("compile_pipeline"),
                ));
            };
            pipelines.insert(*name, pipeline);
        }
        Ok(pipelines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_metal_contract_forbids_hot_path_allocations() {
        let c = MetalPrefillConfig::direct_contract();
        assert_eq!(c.command_buffers_per_layer_group, 1);
        assert!(!c.allocates_in_hot_path);
        assert!(c.owns_kv_cache_write);
        assert!(c.uses_persistent_parameter_buffers);
    }

    #[test]
    fn mlx_contract_is_marked_prototype() {
        let c = MetalPrefillConfig::mlx_prototype_contract();
        assert!(c.allocates_in_hot_path);
        assert!(!c.uses_persistent_parameter_buffers);
    }

    #[test]
    fn direct_pipeline_symbols_are_stable() {
        assert_eq!(DirectMetalPipelineName::RmsNorm.symbol(), "rvllm_rms_norm");
        assert_eq!(DirectMetalPipelineName::Matmul.symbol(), "rvllm_matmul");
        assert_eq!(DirectMetalPipelineName::Rope.symbol(), "rvllm_rope");
        assert_eq!(
            DirectMetalPipelineName::Attention.symbol(),
            "rvllm_attention"
        );
    }
}
