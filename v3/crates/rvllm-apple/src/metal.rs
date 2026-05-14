use std::path::PathBuf;

use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
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
}
