//! Diagnostic two-token transaction using two ordinary S1 evaluations.
//! This is a correctness reference, not a batched decoder or a speed path.
#![forbid(unsafe_code)]

use super::{AneDecodedToken, GemmaAneDecode, LAYERS, VOCAB};
use rvllm_core::TokenId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Positions {
    start: usize,
    after_anchor: usize,
    after_draft: usize,
}

fn preflight(
    position: Option<usize>,
    capacity: usize,
    anchor: TokenId,
    draft: TokenId,
    layers: impl Iterator<Item = usize>,
) -> Result<Positions, String> {
    let start = position.ok_or("ANE decoder requires a complete prefill import")?;
    let after_draft = start.checked_add(2).ok_or("two-token position overflow")?;
    if capacity != 1024
        || start == 0
        || after_draft > capacity
        || anchor.raw() as usize >= VOCAB
        || draft.raw() as usize >= VOCAB
    {
        return Err("two-token reference requires valid tokens and two free unwrapped slots at capacity 1024".into());
    }
    let mut count = 0;
    for frontier in layers {
        if frontier != start {
            return Err("ANE layer KV positions disagree before two-token transaction".into());
        }
        count += 1;
    }
    if count != LAYERS {
        return Err("two-token reference requires all 48 layer frontiers".into());
    }
    Ok(Positions {
        start,
        after_anchor: start + 1,
        after_draft,
    })
}

/// Exclusive ownership of two tentative appends. Dropping this unresolved
/// leaves the decoder unusable until a complete prefill import succeeds.
#[must_use = "resolve the retained prefix before publishing any prediction"]
pub struct PendingTwoToken<'a> {
    decoder: &'a mut GemmaAneDecode,
    positions: Positions,
    draft: TokenId,
    predictions: [AneDecodedToken; 2],
}

/// Only committed predictions are returned. The final prediction is pending
/// outside KV, exactly as in ordinary serial greedy decoding.
pub struct ResolvedTwoToken {
    pub predictions: Vec<AneDecodedToken>,
    pub draft_accepted: bool,
    pub next_position: usize,
}

impl GemmaAneDecode {
    /// Evaluate `[anchor, draft]` with existing single-token graphs. The anchor
    /// is the already selected token not yet in KV. No arithmetic is batched.
    /// Requires a driver journal and zero compilation for qualification.
    pub fn begin_two_token_reference(
        &mut self,
        anchor: TokenId,
        draft: TokenId,
    ) -> Result<PendingTwoToken<'_>, String> {
        self.begin_reference_observed(anchor, draft, &mut |_, _| Ok(()))
    }

    fn begin_reference_observed(
        &mut self,
        anchor: TokenId,
        draft: TokenId,
        observer: &mut impl FnMut(usize, &[half::f16]) -> Result<(), String>,
    ) -> Result<PendingTwoToken<'_>, String> {
        let positions = self.reference_positions(anchor, draft)?;
        // Never restore usability after an error: some layer surfaces may
        // already have changed, even when their logical frontier did not.
        self.next_position = None;
        let first = self.decode_inner(anchor, positions.start, observer, &mut |_, _| Ok(()))?;
        let second =
            self.decode_inner(draft, positions.after_anchor, observer, &mut |_, _| Ok(()))?;
        Ok(PendingTwoToken {
            decoder: self,
            positions,
            draft,
            predictions: [first, second],
        })
    }

    fn reference_positions(&self, anchor: TokenId, draft: TokenId) -> Result<Positions, String> {
        let positions = preflight(
            self.next_position,
            self.capacity,
            anchor,
            draft,
            self.layers
                .iter()
                .map(|layer| layer.attention.tokens_seen()),
        )?;
        if std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_none()
            || rvllm_apple::ane_linear::compile_budget_used() != 0
        {
            return Err(
                "two-token reference requires a driver journal and zero compiler calls".into(),
            );
        }
        Ok(positions)
    }
}

fn accept_draft(target: TokenId, draft: TokenId, allow_second: bool) -> bool {
    allow_second && target == draft
}

#[cfg(test)]
#[path = "ane_two_token_reference_live_tests.rs"]
mod live_tests;

#[path = "ane_two_token_layer_major.rs"]
mod layer_major;

impl PendingTwoToken<'_> {
    /// Inspect only to decide EOS, stop-string and output-budget boundaries.
    /// Do not publish until resolve succeeds; the second may be discarded.
    pub fn predictions(&self) -> &[AneDecodedToken; 2] {
        &self.predictions
    }

    /// `allow_second` must be false when the first prediction ends the request
    /// or exhausts its output budget. A mismatched draft is always rejected.
    /// Partial rollback failure preserves the decoder's poisoned state.
    pub fn resolve(self, allow_second: bool) -> Result<ResolvedTwoToken, String> {
        self.resolve_with(allow_second, |attention, retained| {
            attention.discard_last_unwrapped(retained)
        })
    }

    fn resolve_with(
        self,
        allow_second: bool,
        mut rollback: impl FnMut(
            &mut rvllm_apple::ane_attention::AneAttention,
            usize,
        ) -> Result<(), String>,
    ) -> Result<ResolvedTwoToken, String> {
        if self
            .decoder
            .layers
            .iter()
            .any(|layer| layer.attention.tokens_seen() != self.positions.after_draft)
        {
            return Err("ANE layer KV positions disagree after two-token transaction".into());
        }
        let accepted = accept_draft(self.predictions[0].token, self.draft, allow_second);
        let frontier = if accepted {
            self.positions.after_draft
        } else {
            for layer in &mut self.decoder.layers {
                rollback(&mut layer.attention, self.positions.after_anchor)?;
            }
            self.positions.after_anchor
        };
        let [first, second] = self.predictions;
        let predictions = if accepted {
            vec![first, second]
        } else {
            vec![first]
        };
        self.decoder.next_position = Some(frontier);
        Ok(ResolvedTwoToken {
            predictions,
            draft_accepted: accepted,
            next_position: frontier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_rejects_without_device_work() {
        let check = |p, c, a, d, layers: Vec<usize>| {
            preflight(p, c, TokenId(a), TokenId(d), layers.into_iter())
        };
        assert_eq!(
            check(Some(1022), 1024, 0, 1, vec![1022; 48]).unwrap(),
            Positions {
                start: 1022,
                after_anchor: 1023,
                after_draft: 1024
            }
        );
        for p in [0, 1023, 1024, usize::MAX - 1, usize::MAX] {
            assert!(check(Some(p), 1024, 0, 1, vec![p; 48]).is_err());
        }
        assert!(check(None, 1024, 0, 1, vec![84; 48]).is_err());
        assert!(check(Some(21), 64, 0, 1, vec![21; 48]).is_err());
        assert!(check(Some(21), 1024, VOCAB as u32, 1, vec![21; 48]).is_err());
        assert!(check(Some(21), 1024, 0, VOCAB as u32, vec![21; 48]).is_err());
        for count in [0, 47, 49] {
            assert!(check(Some(21), 1024, 0, 1, vec![21; count]).is_err());
        }
        for layer in 0..48 {
            let mut positions = vec![21; 48];
            positions[layer] += 1;
            assert!(check(Some(21), 1024, 0, 1, positions).is_err());
        }
    }

    #[test]
    fn eos_stop_or_output_budget_keeps_only_anchor_even_when_draft_matches() {
        assert!(accept_draft(TokenId(17), TokenId(17), true));
        assert!(!accept_draft(TokenId(17), TokenId(18), true));
        assert!(!accept_draft(TokenId(17), TokenId(17), false));
        assert!(!accept_draft(TokenId(17), TokenId(18), false));
    }
}
