use crate::plan::RolloutBucket;
use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const LAYOUT_HASH_SEED: u64 = 0xcbf29ce484222325u64;
const LAYOUT_HASH_PRIME: u64 = 0x100000001b3;

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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandoffCapsule {
    pub kind: HandoffKind,
    pub req_ids: Vec<ReqId>,
    pub tokens_flat: Vec<TokenId>,
    pub cu_seqlens: Vec<u32>,
    pub positions: Vec<u32>,
    pub context_lens: Vec<u32>,
    pub rollout_bucket: Option<RolloutBucket>,
    pub state_handles: Vec<StateHandle>,
    pub input_surface: Option<SurfaceId>,
    pub output_surface: Option<SurfaceId>,
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
        let mut capsule = Self {
            kind,
            req_ids,
            tokens_flat,
            cu_seqlens,
            positions,
            context_lens,
            rollout_bucket: None,
            state_handles: Vec::new(),
            input_surface: None,
            output_surface: None,
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
        if self.cu_seqlens.first().copied() != Some(0) {
            return Err(self.err("cu_seqlens must start at zero"));
        }
        if self.cu_seqlens.last().copied() != Some(self.tokens_flat.len() as u32) {
            return Err(self.err("cu_seqlens must end at tokens_flat length"));
        }
        if !self.cu_seqlens.windows(2).all(|w| w[0] <= w[1]) {
            return Err(self.err("cu_seqlens must be monotonic"));
        }
        for (req_id, (&position, &context_len)) in self
            .req_ids
            .iter()
            .zip(self.positions.iter().zip(self.context_lens.iter()))
        {
            let _ = req_id;
            if context_len == 0 {
                return Err(self.err("context_lens must be positive"));
            }
            if position + 1 != context_len {
                return Err(self.err("each context_len must equal position + 1"));
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
        let mut h = LAYOUT_HASH_SEED;
        let mut acc = [0u8; 32];

        let kind = match self.kind {
            HandoffKind::MetalPrefillToMetalDecode => 0_u8,
            HandoffKind::MetalPrefillToAneFfnRollout => 1_u8,
            HandoffKind::MetalPrefillToAneRolloutExperimental => 2_u8,
            HandoffKind::MetalDecodeToAneFfn => 3_u8,
        };

        for byte in [
            kind,
            self.req_ids.len() as u8,
            self.tokens_flat.len() as u8,
            self.cu_seqlens.len() as u8,
            self.positions.len() as u8,
            self.context_lens.len() as u8,
            self.state_handles.len() as u8,
            u8::from(self.input_surface.is_some()),
            u8::from(self.output_surface.is_some()),
            u8::from(self.rollout_bucket.is_some()),
        ] {
            h ^= u64::from(byte);
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
        }
        if let Some(bucket) = self.rollout_bucket {
            h ^= bucket.seqs as u64;
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            h ^= bucket.tokens as u64;
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
        }

        for req in &self.req_ids {
            h ^= req.raw();
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
        }
        for token in &self.tokens_flat {
            let raw = token.raw().to_le_bytes();
            for byte in raw {
                h ^= u64::from(byte);
                h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            }
        }
        for offset in &self.cu_seqlens {
            let raw = offset.to_le_bytes();
            for byte in raw {
                h ^= u64::from(byte);
                h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            }
        }
        for position in &self.positions {
            let raw = position.to_le_bytes();
            for byte in raw {
                h ^= u64::from(byte);
                h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            }
        }
        for context_len in &self.context_lens {
            let raw = context_len.to_le_bytes();
            for byte in raw {
                h ^= u64::from(byte);
                h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            }
        }
        for handle in &self.state_handles {
            h ^= u64::from(handle.id);
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            h ^= u64::from(handle.kind as u8 as u64);
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
            h ^= handle.bytes as u64;
            h = h.wrapping_mul(LAYOUT_HASH_PRIME);
        }

        acc[..8].copy_from_slice(&h.to_le_bytes());
        acc[8..16].copy_from_slice(&h.rotate_left(13).to_le_bytes());
        acc[16..24].copy_from_slice(&h.rotate_left(29).to_le_bytes());
        acc[24..32].copy_from_slice(&h.rotate_left(47).to_le_bytes());
        acc
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
}
