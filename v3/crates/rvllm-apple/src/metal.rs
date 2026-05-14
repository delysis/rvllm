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

        let expected_layer_ops = [
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
        let mut expected = Vec::new();
        for layer in [4, 5] {
            for op in expected_layer_ops {
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
