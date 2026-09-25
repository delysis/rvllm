//! Host-only ANE GQA packing and explicit-mask experiment descriptors.
//! This module deliberately has NO ANE client, selector or evaluation call.
//! Existing multi-I/O panic quarantine and the single-I/O owner are unchanged.
#![forbid(unsafe_code)]
use super::{plan, reference::Data, Error, Plan, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct PackedLayout {
    pub heads: u32,
    pub query_rows: u32,
    pub key_rows: u32,
    pub dim: u32,
    pub packed_query_elements: usize,
    pub explicit_mask_elements: usize,
    pub native_attention_admitted: bool,
    pub execution_available: bool,
    pub blockers: Vec<String>,
}
impl PackedLayout {
    pub fn new(plan: Plan, element_budget: usize) -> Result<Self> {
        plan.shape.validate()?;
        let s = plan.shape;
        let rows = s
            .queries
            .checked_mul(s.gqa())
            .ok_or_else(|| Error::new("ANE packed rows overflow"))?;
        let q = s.kv_heads as usize * rows as usize * s.dim as usize;
        let mask = (rows as usize)
            .checked_mul(s.live_keys as usize)
            .ok_or_else(|| Error::new("ANE mask overflow"))?;
        if q.checked_add(mask)
            .ok_or_else(|| Error::new("ANE packing budget overflow"))?
            > element_budget
        {
            return Err(Error::new("explicit ANE packing/mask budget exceeded"));
        }
        let mut blockers = vec![
            "No qualified single-I/O compiled adapter for this candidate".into(),
            "BF16/FP32 Metal contract is not FP16 ANE arithmetic".into(),
            "Multi-I/O ANE execution remains quarantined; descriptor is not an override".into(),
        ];
        if rows.max(s.live_keys) > 2048 || rows.min(s.live_keys) >= 512 {
            blockers.push(
                "Inspected native-SDPA shape envelope exceeded; no output averaging across tiles"
                    .into(),
            );
        }
        Ok(Self {
            heads: s.kv_heads,
            query_rows: rows,
            key_rows: s.live_keys,
            dim: s.dim,
            packed_query_elements: q,
            explicit_mask_elements: mask,
            native_attention_admitted: false,
            execution_available: false,
            blockers,
        })
    }
    /// Bit-preserving layout only. It deliberately does NOT narrow BF16 to FP16.
    pub fn pack_query(&self, plan: Plan, data: &Data) -> Result<Vec<u16>> {
        data.validate(plan)?;
        if self.heads != plan.shape.kv_heads
            || self.query_rows != plan.shape.queries * plan.shape.gqa()
            || self.packed_query_elements != plan.shape.output_elements()
            || self.dim != plan.shape.dim
        {
            return Err(Error::new("ANE descriptor/shape mismatch"));
        }
        let mut output = vec![0; self.packed_query_elements];
        for h in 0..self.heads {
            for r in 0..self.query_rows {
                let (q, qh) = plan::packed_query_head(plan.shape, r, h).unwrap();
                let input = (q as usize * 16 + qh as usize) * self.dim as usize;
                let dest = (h as usize * self.query_rows as usize + r as usize) * self.dim as usize;
                output[dest..dest + self.dim as usize]
                    .copy_from_slice(&data.q[input..input + self.dim as usize]);
            }
        }
        Ok(output)
    }
    /// One additive mask plane shared across KV heads. Original absolute token
    /// positions, NOT packed row indices, determine causality/window bounds.
    pub fn mask(&self, plan: Plan, data: &Data) -> Result<Vec<f32>> {
        data.validate(plan)?;
        if self.query_rows != plan.shape.queries * plan.shape.gqa()
            || self.key_rows != plan.shape.live_keys
            || self.explicit_mask_elements != self.query_rows as usize * self.key_rows as usize
        {
            return Err(Error::new("ANE mask descriptor mismatch"));
        }
        let mut output = vec![f32::NEG_INFINITY; self.explicit_mask_elements];
        for r in 0..self.query_rows {
            let pos = data.positions[(r / plan.shape.gqa()) as usize] as u32;
            for k in plan.shape.key_start(pos)..=pos {
                if data.physical_token(plan.shape, k).is_some() {
                    output[r as usize * self.key_rows as usize + k as usize] = 0.0;
                }
            }
        }
        Ok(output)
    }
}
