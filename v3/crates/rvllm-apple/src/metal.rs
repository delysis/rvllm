use std::path::PathBuf;

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
}
