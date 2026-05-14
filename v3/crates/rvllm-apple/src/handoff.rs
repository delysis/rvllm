use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use serde::{Deserialize, Serialize};

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
        // Deliberately simple deterministic hash for host-testable layout
        // checks. Replace with SHA-256 when rvllm-metadata layout hashes wire in.
        let mut h = [0u8; 32];
        h[0] = self.req_ids.len() as u8;
        h[1] = self.tokens_flat.len() as u8;
        h[2] = self.cu_seqlens.len() as u8;
        h[3] = self.positions.len() as u8;
        h[4] = self.context_lens.len() as u8;
        h[5] = self.state_handles.len() as u8;
        h[6] = u8::from(self.input_surface.is_some());
        h[7] = u8::from(self.output_surface.is_some());
        h
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
        );
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
        );
        capsule.tokens_flat.push(TokenId(11));
        assert!(capsule.validate().is_err());
    }
}
