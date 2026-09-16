use crate::iosurface::IoSurfaceTensorDesc;
use crate::plan::RolloutBucket;
use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const HANDOFF_SCHEMA_V2: u16 = 2;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum HandoffKind {
    MetalPrefillToMetalDecode,
    MetalPrefillToAneFfnRollout,
    MetalPrefillToAneRolloutExperimental,
    MetalDecodeToAneFfn,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SurfaceId(pub u64);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum HandoffSurfaceRole {
    ActivationInput,
    ActivationOutput,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct HandoffSurfaceBinding {
    pub role: HandoffSurfaceRole,
    pub surface_id: SurfaceId,
    pub desc: IoSurfaceTensorDesc,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct HandoffSequenceMeta {
    pub req_id: ReqId,
    pub token_start: u32,
    pub token_end: u32,
    pub position: u32,
    pub context_len: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum StateHandleKind {
    KvCache,
    Hidden,
    Logits,
    Scratch,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StateHandle {
    pub kind: StateHandleKind,
    pub id: u64,
    pub bytes: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct HandoffKvChain {
    pub id: u32,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandoffCapsule {
    pub schema_version: u16,
    pub kind: HandoffKind,
    pub req_ids: Vec<ReqId>,
    pub tokens_flat: Vec<TokenId>,
    pub cu_seqlens: Vec<u32>,
    pub positions: Vec<u32>,
    pub context_lens: Vec<u32>,
    pub query_start_positions: Vec<u32>,
    pub kv_chains: Vec<HandoffKvChain>,
    pub max_blocks_per_seq: u32,
    pub block_tables: Vec<u32>,
    pub slot_mapping: Vec<i32>,
    pub model_layout_fingerprint: [u8; 32],
    pub rollout_bucket: Option<RolloutBucket>,
    pub state_handles: Vec<StateHandle>,
    pub input_surface: Option<SurfaceId>,
    pub output_surface: Option<SurfaceId>,
    pub activation_surfaces: Vec<HandoffSurfaceBinding>,
    pub layout_hash: [u8; 32],
}

impl HandoffCapsule {
    #[must_use]
    pub fn new(
        kind: HandoffKind,
        req_ids: Vec<ReqId>,
        tokens_flat: Vec<TokenId>,
        cu_seqlens: Vec<u32>,
        positions: Vec<u32>,
        context_lens: Vec<u32>,
    ) -> Self {
        let query_start_positions = cu_seqlens
            .windows(2)
            .enumerate()
            .map(|(index, span)| {
                context_lens
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(span[1].saturating_sub(span[0]))
            })
            .collect();
        let mut capsule = Self {
            schema_version: HANDOFF_SCHEMA_V2,
            kind,
            req_ids,
            tokens_flat,
            cu_seqlens,
            positions,
            query_start_positions,
            context_lens,
            kv_chains: Vec::new(),
            max_blocks_per_seq: 0,
            block_tables: Vec::new(),
            slot_mapping: Vec::new(),
            model_layout_fingerprint: [0; 32],
            rollout_bucket: None,
            state_handles: Vec::new(),
            input_surface: None,
            output_surface: None,
            activation_surfaces: Vec::new(),
            layout_hash: [0; 32],
        };
        capsule.layout_hash = capsule.compute_layout_hash();
        capsule
    }

    #[must_use]
    pub fn num_sequences(&self) -> usize {
        self.req_ids.len()
    }

    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.validate().is_ok()
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != HANDOFF_SCHEMA_V2 {
            return Err(self.err("unsupported handoff schema version"));
        }
        if self.req_ids.is_empty() {
            return Err(self.err("req_ids must be non-empty"));
        }
        match self.kind {
            HandoffKind::MetalPrefillToAneFfnRollout
            | HandoffKind::MetalPrefillToAneRolloutExperimental => {
                let Some(bucket) = self.rollout_bucket else {
                    return Err(self.err("rollout capsules require a rollout bucket"));
                };
                if bucket.seqs == 0 || bucket.tokens == 0 {
                    return Err(self.err("rollout bucket tokens/seqs must be >= 1"));
                }
                let seqs = self.req_ids.len() as u32;
                if !bucket.fits(seqs, 1) {
                    return Err(self.err("rollout bucket seqs capacity is too small"));
                }
            }
            _ => {
                if self.rollout_bucket.is_some() {
                    return Err(self.err("rollout bucket is only valid for ANE rollout capsules"));
                }
            }
        }
        let mut seen = HashSet::with_capacity(self.req_ids.len());
        for req_id in &self.req_ids {
            let raw = req_id.raw();
            if !seen.insert(raw) {
                return Err(self.err("req_ids must be unique"));
            }
        }
        if self.cu_seqlens.len() != self.req_ids.len() + 1 {
            return Err(self.err("cu_seqlens length must equal req_ids + 1"));
        }
        if self.positions.len() != self.req_ids.len() {
            return Err(self.err("positions length must equal req_ids"));
        }
        if self.context_lens.len() != self.req_ids.len() {
            return Err(self.err("context_lens length must equal req_ids"));
        }
        if self.query_start_positions.len() != self.req_ids.len() {
            return Err(self.err("query_start_positions length must equal req_ids"));
        }
        if !self.kv_chains.is_empty() && self.kv_chains.len() != self.req_ids.len() {
            return Err(self.err("kv_chains length must be zero or equal req_ids"));
        }
        if !self.kv_chains.is_empty() && self.max_blocks_per_seq == 0 {
            return Err(self.err("paged KV chains require a nonzero block-table width"));
        }
        if self.max_blocks_per_seq == 0 {
            if !self.block_tables.is_empty() {
                return Err(self.err("block tables require max_blocks_per_seq"));
            }
        } else if self.block_tables.len() != self.req_ids.len() * self.max_blocks_per_seq as usize {
            return Err(self.err("block table shape does not match sequences"));
        }
        if !self.slot_mapping.is_empty() && self.slot_mapping.len() != self.tokens_flat.len() {
            return Err(self.err("slot_mapping length must equal tokens_flat"));
        }
        if self.cu_seqlens.first().copied() != Some(0) {
            return Err(self.err("cu_seqlens must start at zero"));
        }
        if self.cu_seqlens.last().copied() != Some(self.tokens_flat.len() as u32) {
            return Err(self.err("cu_seqlens must end at tokens_flat length"));
        }
        if !self.cu_seqlens.windows(2).all(|w| w[0] <= w[1]) {
            return Err(self.err("cu_seqlens must be monotonic"));
        }
        for index in 0..self.req_ids.len() {
            let position = self.positions[index];
            let context_len = self.context_lens[index];
            if context_len == 0 {
                return Err(self.err("context_lens must be positive"));
            }
            if position + 1 != context_len {
                return Err(self.err("each context_len must equal position + 1"));
            }
            let query_len = self.cu_seqlens[index + 1] - self.cu_seqlens[index];
            let expected_context = self.query_start_positions[index]
                .checked_add(query_len)
                .ok_or_else(|| self.err("query context length overflow"))?;
            if expected_context != context_len {
                return Err(self.err("query start + query length must equal context length"));
            }
        }
        if self.compute_layout_hash() != self.layout_hash {
            return Err(self.err("layout hash mismatch"));
        }
        Ok(())
    }

    #[must_use]
    pub fn with_state_handle(mut self, handle: StateHandle) -> Self {
        self.state_handles.push(handle);
        self.layout_hash = self.compute_layout_hash();
        self
    }

    #[must_use]
    pub fn with_paged_kv(
        mut self,
        kv_chains: Vec<HandoffKvChain>,
        max_blocks_per_seq: u32,
        block_tables: Vec<u32>,
        slot_mapping: Vec<i32>,
        model_layout_fingerprint: [u8; 32],
    ) -> Self {
        self.kv_chains = kv_chains;
        self.max_blocks_per_seq = max_blocks_per_seq;
        self.block_tables = block_tables;
        self.slot_mapping = slot_mapping;
        self.model_layout_fingerprint = model_layout_fingerprint;
        self.layout_hash = self.compute_layout_hash();
        self
    }

    #[must_use]
    pub fn with_rollout_bucket(mut self, rollout_bucket: Option<RolloutBucket>) -> Self {
        self.rollout_bucket = rollout_bucket;
        self.layout_hash = self.compute_layout_hash();
        self
    }

    #[must_use]
    pub fn with_surfaces(mut self, input: Option<SurfaceId>, output: Option<SurfaceId>) -> Self {
        self.input_surface = input;
        self.output_surface = output;
        self.layout_hash = self.compute_layout_hash();
        self
    }

    #[must_use]
    pub fn with_activation_surface(mut self, binding: HandoffSurfaceBinding) -> Self {
        self.activation_surfaces.push(binding);
        self.layout_hash = self.compute_layout_hash();
        self
    }

    #[must_use]
    pub fn sequence_metadata(&self) -> Vec<HandoffSequenceMeta> {
        let mut meta = Vec::with_capacity(self.req_ids.len());
        for idx in 0..self.req_ids.len() {
            meta.push(HandoffSequenceMeta {
                req_id: self.req_ids[idx],
                token_start: self.cu_seqlens[idx],
                token_end: self.cu_seqlens[idx + 1],
                position: self.positions[idx],
                context_len: self.context_lens[idx],
            });
        }
        meta
    }

    fn err(&self, reason: &'static str) -> RvllmError {
        RvllmError::apple(
            AppleError::HandoffMalformed { reason },
            AppleCtx {
                backend: "rvllm-apple",
                op: "handoff",
                device: "apple-silicon",
            },
        )
    }

    #[must_use]
    fn compute_layout_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"rvllm.apple-handoff.v2");
        h.update(self.schema_version.to_le_bytes());

        let kind = match self.kind {
            HandoffKind::MetalPrefillToMetalDecode => 0_u8,
            HandoffKind::MetalPrefillToAneFfnRollout => 1_u8,
            HandoffKind::MetalPrefillToAneRolloutExperimental => 2_u8,
            HandoffKind::MetalDecodeToAneFfn => 3_u8,
        };

        h.update([
            kind,
            u8::from(self.input_surface.is_some()),
            u8::from(self.output_surface.is_some()),
            u8::from(self.rollout_bucket.is_some()),
        ]);
        h.update((self.req_ids.len() as u64).to_le_bytes());
        h.update((self.tokens_flat.len() as u64).to_le_bytes());
        h.update((self.cu_seqlens.len() as u64).to_le_bytes());
        h.update((self.positions.len() as u64).to_le_bytes());
        h.update((self.context_lens.len() as u64).to_le_bytes());
        h.update((self.state_handles.len() as u64).to_le_bytes());
        h.update((self.activation_surfaces.len() as u64).to_le_bytes());
        h.update((self.query_start_positions.len() as u64).to_le_bytes());
        h.update((self.kv_chains.len() as u64).to_le_bytes());
        h.update(self.max_blocks_per_seq.to_le_bytes());
        h.update((self.block_tables.len() as u64).to_le_bytes());
        h.update((self.slot_mapping.len() as u64).to_le_bytes());
        h.update(self.model_layout_fingerprint);
        if let Some(bucket) = self.rollout_bucket {
            h.update(bucket.seqs.to_le_bytes());
            h.update(bucket.tokens.to_le_bytes());
        }

        for req in &self.req_ids {
            h.update(req.raw().to_le_bytes());
        }
        for token in &self.tokens_flat {
            h.update(token.raw().to_le_bytes());
        }
        for offset in &self.cu_seqlens {
            h.update(offset.to_le_bytes());
        }
        for position in &self.positions {
            h.update(position.to_le_bytes());
        }
        for context_len in &self.context_lens {
            h.update(context_len.to_le_bytes());
        }
        for query_start in &self.query_start_positions {
            h.update(query_start.to_le_bytes());
        }
        for chain in &self.kv_chains {
            h.update(chain.id.to_le_bytes());
            h.update(chain.generation.to_le_bytes());
        }
        for block in &self.block_tables {
            h.update(block.to_le_bytes());
        }
        for slot in &self.slot_mapping {
            h.update(slot.to_le_bytes());
        }
        for handle in &self.state_handles {
            h.update(handle.id.to_le_bytes());
            h.update([handle.kind as u8]);
            h.update((handle.bytes as u64).to_le_bytes());
        }
        for binding in &self.activation_surfaces {
            h.update([binding.role as u8]);
            h.update(binding.surface_id.0.to_le_bytes());
            h.update((binding.desc.dtype.bytes() as u64).to_le_bytes());
            h.update(binding.desc.channels.to_le_bytes());
            h.update(binding.desc.spatial.to_le_bytes());
        }
        h.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsule_validates_spans_and_layout_hash() {
        let capsule = HandoffCapsule::new(
            HandoffKind::MetalPrefillToAneFfnRollout,
            vec![ReqId(1), ReqId(2)],
            vec![TokenId(10), TokenId(11), TokenId(20)],
            vec![0, 2, 3],
            vec![1, 0],
            vec![2, 1],
        )
        .with_rollout_bucket(Some(RolloutBucket { seqs: 4, tokens: 1 }));
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.num_sequences(), 2);
    }

    #[test]
    fn capsule_detects_tampering() {
        let mut capsule = HandoffCapsule::new(
            HandoffKind::MetalPrefillToAneFfnRollout,
            vec![ReqId(1)],
            vec![TokenId(10)],
            vec![0, 1],
            vec![0],
            vec![1],
        )
        .with_rollout_bucket(Some(RolloutBucket { seqs: 4, tokens: 1 }));
        capsule.tokens_flat.push(TokenId(11));
        assert!(capsule.validate().is_err());
    }

    #[test]
    fn paged_kv_handoff_validates_absolute_positions_and_flattened_tables() {
        let capsule = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(7), ReqId(9)],
            vec![TokenId(31), TokenId(32), TokenId(70)],
            vec![0, 2, 3],
            vec![33, 64],
            vec![34, 65],
        )
        .with_paged_kv(
            vec![
                HandoffKvChain {
                    id: 3,
                    generation: 11,
                },
                HandoffKvChain {
                    id: 8,
                    generation: 4,
                },
            ],
            3,
            vec![10, 11, u32::MAX, 20, 21, 22],
            vec![32, 33, 64],
            [0xA5; 32],
        );

        assert_eq!(capsule.query_start_positions, vec![32, 64]);
        assert!(capsule.is_well_formed());

        let mut malformed = capsule.clone();
        malformed.block_tables.pop();
        malformed.layout_hash = malformed.compute_layout_hash();
        assert!(malformed.validate().is_err());

        let mut stale_layout = capsule;
        stale_layout.kv_chains[0].generation += 1;
        assert!(stale_layout.validate().is_err());
    }
}
