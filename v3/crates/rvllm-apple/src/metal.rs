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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum MetalBufferRole {
    Activation,
    Scratch,
    KvCache,
    Parameters,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalBufferRequest {
    pub name: String,
    pub bytes: usize,
    pub align: usize,
    pub role: MetalBufferRole,
}

impl MetalBufferRequest {
    #[must_use]
    pub fn new(name: impl Into<String>, bytes: usize, align: usize, role: MetalBufferRole) -> Self {
        Self {
            name: name.into(),
            bytes,
            align,
            role,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct MetalBufferBinding {
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalBufferAllocation {
    pub name: String,
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
    pub role: MetalBufferRole,
}

impl MetalBufferAllocation {
    #[must_use]
    pub fn binding(&self) -> MetalBufferBinding {
        MetalBufferBinding {
            offset: self.offset,
            bytes: self.bytes,
        }
    }

    #[must_use]
    pub fn end_offset(&self) -> usize {
        self.offset + self.bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalBufferArenaPlan {
    capacity_bytes: usize,
    used_bytes: usize,
    parameter_index: usize,
    allocations: Vec<MetalBufferAllocation>,
}

impl MetalBufferArenaPlan {
    pub fn new(capacity_bytes: usize, requests: &[MetalBufferRequest]) -> Result<Self> {
        if requests.is_empty() {
            return invalid_arena("buffer request list is empty");
        }

        let mut used_bytes = 0usize;
        let mut parameter_index = None;
        let mut allocations = Vec::with_capacity(requests.len());

        for request in requests {
            validate_request(request, &allocations)?;

            let offset = match checked_align_up(used_bytes, request.align) {
                Some(v) => v,
                None => return invalid_arena("buffer offset overflow"),
            };
            let end = match offset.checked_add(request.bytes) {
                Some(v) => v,
                None => return invalid_arena("buffer end offset overflow"),
            };
            if end > capacity_bytes {
                return arena_too_small(end, capacity_bytes);
            }

            if request.role == MetalBufferRole::Parameters {
                if parameter_index.is_some() {
                    return invalid_arena("multiple persistent parameter buffers");
                }
                parameter_index = Some(allocations.len());
            }

            allocations.push(MetalBufferAllocation {
                name: request.name.clone(),
                offset,
                bytes: request.bytes,
                align: request.align,
                role: request.role,
            });
            used_bytes = end;
        }

        let parameter_index = match parameter_index {
            Some(v) => v,
            None => return invalid_arena("persistent parameter buffer missing"),
        };

        Ok(Self {
            capacity_bytes,
            used_bytes,
            parameter_index,
            allocations,
        })
    }

    #[must_use]
    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[must_use]
    pub fn allocations(&self) -> &[MetalBufferAllocation] {
        &self.allocations
    }

    #[must_use]
    pub fn binding(&self, name: &str) -> Option<MetalBufferBinding> {
        self.allocations
            .iter()
            .find(|allocation| allocation.name == name)
            .map(MetalBufferAllocation::binding)
    }

    #[must_use]
    pub fn parameter_binding(&self) -> MetalBufferBinding {
        self.allocations[self.parameter_index].binding()
    }

    #[must_use]
    pub fn contract(&self) -> PrefillContract {
        MetalPrefillConfig::direct_contract()
    }

    #[must_use]
    pub fn has_overlaps(&self) -> bool {
        self.allocations
            .windows(2)
            .any(|pair| pair[0].end_offset() > pair[1].offset)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PrefillLayerGroup {
    pub id: u32,
    pub first_layer: u32,
    pub layer_count: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum MetalPrefillOp {
    InputLayerNormFp8Quant,
    QkvProjection,
    QkvRmsNorm,
    RopeAndKvCacheWrite,
    PagedPrefillAttention,
    AttentionOutputQuant,
    OutputProjectionResidualNorm,
    PreFeedforwardNormFp8Quant,
    GateUpProjection,
    GeluMulFp8Quant,
    DownProjectionResidualNorm,
}

pub const PREFILL_LAYER_OPS: [MetalPrefillOp; 11] = [
    MetalPrefillOp::InputLayerNormFp8Quant,
    MetalPrefillOp::QkvProjection,
    MetalPrefillOp::QkvRmsNorm,
    MetalPrefillOp::RopeAndKvCacheWrite,
    MetalPrefillOp::PagedPrefillAttention,
    MetalPrefillOp::AttentionOutputQuant,
    MetalPrefillOp::OutputProjectionResidualNorm,
    MetalPrefillOp::PreFeedforwardNormFp8Quant,
    MetalPrefillOp::GateUpProjection,
    MetalPrefillOp::GeluMulFp8Quant,
    MetalPrefillOp::DownProjectionResidualNorm,
];

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct MetalPrefillCommand {
    pub layer_group: u32,
    pub layer: u32,
    pub op: MetalPrefillOp,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum MetalPrefillCommandEvent {
    BeginCommandBuffer {
        layer_group: u32,
    },
    EncodeOp {
        layer_group: u32,
        layer: u32,
        op: MetalPrefillOp,
    },
    CommitCommandBuffer {
        layer_group: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalPrefillCommandBufferRecipe {
    pub layer_group: PrefillLayerGroup,
    commands: Vec<MetalPrefillCommand>,
    timeline: Vec<MetalPrefillCommandEvent>,
}

impl PrefillLayerGroup {
    pub fn new(id: u32, first_layer: u32, layer_count: u32) -> Result<Self> {
        let group = Self {
            id,
            first_layer,
            layer_count,
        };
        group.validate()?;
        Ok(group)
    }

    pub fn validate(self) -> Result<()> {
        if self.layer_count == 0 {
            return Err(metal_recipe_err(
                "prefill layer group must contain at least one layer",
            ));
        }
        if self.first_layer.checked_add(self.layer_count).is_none() {
            return Err(metal_recipe_err("prefill layer group range overflows"));
        }
        Ok(())
    }

    #[must_use]
    pub fn layers(self) -> std::ops::Range<u32> {
        self.first_layer..(self.first_layer + self.layer_count)
    }
}

impl MetalPrefillOp {
    #[must_use]
    pub const fn prefill_layer_order() -> &'static [Self] {
        &PREFILL_LAYER_OPS
    }
}

impl MetalPrefillCommandBufferRecipe {
    pub fn for_layer_group(layer_group: PrefillLayerGroup) -> Result<Self> {
        layer_group.validate()?;

        let ops_per_layer = MetalPrefillOp::prefill_layer_order().len();
        let command_count = layer_group.layer_count as usize * ops_per_layer;
        let mut commands = Vec::with_capacity(command_count);
        let mut timeline = Vec::with_capacity(command_count + 2);
        timeline.push(MetalPrefillCommandEvent::BeginCommandBuffer {
            layer_group: layer_group.id,
        });

        for layer in layer_group.layers() {
            for &op in MetalPrefillOp::prefill_layer_order() {
                commands.push(MetalPrefillCommand {
                    layer_group: layer_group.id,
                    layer,
                    op,
                });
                timeline.push(MetalPrefillCommandEvent::EncodeOp {
                    layer_group: layer_group.id,
                    layer,
                    op,
                });
            }
        }

        timeline.push(MetalPrefillCommandEvent::CommitCommandBuffer {
            layer_group: layer_group.id,
        });

        Ok(Self {
            layer_group,
            commands,
            timeline,
        })
    }

    #[must_use]
    pub fn encoded_ops(&self) -> std::slice::Iter<'_, MetalPrefillCommand> {
        self.commands.iter()
    }

    #[must_use]
    pub fn timeline(&self) -> std::slice::Iter<'_, MetalPrefillCommandEvent> {
        self.timeline.iter()
    }

    #[must_use]
    pub const fn command_buffers_per_layer_group(&self) -> u32 {
        1
    }

    #[must_use]
    pub fn per_op_commit_count(&self) -> usize {
        self.timeline
            .iter()
            .take(self.timeline.len().saturating_sub(1))
            .filter(|event| matches!(event, MetalPrefillCommandEvent::CommitCommandBuffer { .. }))
            .count()
    }
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

fn validate_request(
    request: &MetalBufferRequest,
    previous: &[MetalBufferAllocation],
) -> Result<()> {
    if request.name.is_empty() {
        return invalid_arena("buffer name is empty");
    }
    if previous
        .iter()
        .any(|allocation| allocation.name == request.name)
    {
        return invalid_arena("duplicate buffer name");
    }
    if request.bytes == 0 {
        return invalid_arena("buffer size is zero");
    }
    if request.align == 0 {
        return invalid_arena("buffer alignment is zero");
    }
    if !request.align.is_power_of_two() {
        return invalid_arena("buffer alignment is not a power of two");
    }
    Ok(())
}

fn checked_align_up(value: usize, align: usize) -> Option<usize> {
    let mask = align.checked_sub(1)?;
    value.checked_add(mask).map(|aligned| aligned & !mask)
}

fn metal_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "direct-metal",
        op,
        device: "apple-silicon",
    }
}

fn metal_recipe_err(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidMetalRecipe { reason },
        AppleCtx {
            backend: "direct-metal",
            op: "prefill_command_recipe",
            device: "apple-silicon",
        },
    )
}

fn invalid_arena<T>(reason: &'static str) -> Result<T> {
    Err(RvllmError::apple(
        AppleError::InvalidBufferArena { reason },
        metal_ctx("plan_buffer_arena"),
    ))
}

fn arena_too_small<T>(requested: usize, capacity: usize) -> Result<T> {
    Err(RvllmError::apple(
        AppleError::BufferArenaTooSmall {
            requested,
            capacity,
        },
        metal_ctx("plan_buffer_arena"),
    ))
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
    ) -> Result<std::collections::BTreeMap<DirectMetalPipelineName, Retained<AnyObject>>> {
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
    use rvllm_core::AppleError;

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

    #[test]
    fn arena_plan_has_stable_offsets_and_reuses_parameter_buffer() {
        let requests = vec![
            MetalBufferRequest::new("tokens", 24, 16, MetalBufferRole::Activation),
            MetalBufferRequest::new("qkv", 96, 64, MetalBufferRole::Scratch),
            MetalBufferRequest::new("parameters", 256, 256, MetalBufferRole::Parameters),
            MetalBufferRequest::new("kv-cache", 1024, 256, MetalBufferRole::KvCache),
        ];

        let first = match MetalBufferArenaPlan::new(4096, &requests) {
            Ok(v) => v,
            Err(e) => panic!("unexpected arena planning error: {e}"),
        };
        let second = match MetalBufferArenaPlan::new(4096, &requests) {
            Ok(v) => v,
            Err(e) => panic!("unexpected arena planning error: {e}"),
        };

        assert_eq!(first, second);
        assert_eq!(first.contract(), MetalPrefillConfig::direct_contract());
        assert_eq!(first.allocations().len(), requests.len());
        assert!(!first.has_overlaps());

        let qkv = match first.binding("qkv") {
            Some(v) => v,
            None => panic!("missing qkv binding"),
        };
        assert_eq!(qkv.offset % 64, 0);

        let params_for_first_launch = first.parameter_binding();
        let params_for_next_launch = first.parameter_binding();
        assert_eq!(params_for_first_launch, params_for_next_launch);
        assert_eq!(params_for_first_launch.offset % 256, 0);
        assert_eq!(params_for_first_launch.bytes, 256);
    }

    #[test]
    fn arena_plan_rejects_insufficient_capacity_with_typed_error() {
        let requests = vec![
            MetalBufferRequest::new("parameters", 256, 256, MetalBufferRole::Parameters),
            MetalBufferRequest::new("kv-cache", 1024, 256, MetalBufferRole::KvCache),
        ];

        let err = match MetalBufferArenaPlan::new(512, &requests) {
            Ok(v) => panic!("expected arena planning failure, got {v:?}"),
            Err(e) => e,
        };

        match err {
            rvllm_core::RvllmError::Apple {
                err:
                    AppleError::BufferArenaTooSmall {
                        requested,
                        capacity,
                    },
                ..
            } => {
                assert_eq!(requested, 1280);
                assert_eq!(capacity, 512);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn prefill_layer_group_recipe_orders_ops_inside_one_command_buffer() {
        let group = match PrefillLayerGroup::new(2, 4, 2) {
            Ok(group) => group,
            Err(e) => panic!("unexpected layer group error: {e}"),
        };
        let recipe = match MetalPrefillCommandBufferRecipe::for_layer_group(group) {
            Ok(recipe) => recipe,
            Err(e) => panic!("unexpected recipe error: {e}"),
        };

        let mut expected = Vec::new();
        for layer in [4, 5] {
            for op in PREFILL_LAYER_OPS {
                expected.push((layer, op));
            }
        }

        let encoded: Vec<_> = recipe
            .encoded_ops()
            .map(|command| (command.layer, command.op))
            .collect();
        assert_eq!(encoded, expected);
        assert_eq!(recipe.command_buffers_per_layer_group(), 1);
    }

    #[test]
    fn prefill_layer_group_recipe_commits_once_after_all_ops() {
        let group = match PrefillLayerGroup::new(0, 0, 1) {
            Ok(group) => group,
            Err(e) => panic!("unexpected layer group error: {e}"),
        };
        let recipe = match MetalPrefillCommandBufferRecipe::for_layer_group(group) {
            Ok(recipe) => recipe,
            Err(e) => panic!("unexpected recipe error: {e}"),
        };
        let timeline: Vec<_> = recipe.timeline().copied().collect();

        assert_eq!(
            timeline.first(),
            Some(&MetalPrefillCommandEvent::BeginCommandBuffer {
                layer_group: group.id,
            })
        );
        assert_eq!(
            timeline.last(),
            Some(&MetalPrefillCommandEvent::CommitCommandBuffer {
                layer_group: group.id,
            })
        );
        assert_eq!(
            timeline
                .iter()
                .filter(|event| matches!(
                    event,
                    MetalPrefillCommandEvent::CommitCommandBuffer { .. }
                ))
                .count(),
            1
        );
        assert_eq!(recipe.per_op_commit_count(), 0);
    }
}
