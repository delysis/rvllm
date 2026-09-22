//! Exact, opt-in host work reduction for Gemma's rounded softcap/top-five head.
//! This is not an ANE kernel, quantizer, or substitute for full tensor checks.
#![forbid(unsafe_code)]

use half::f16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HeadRankingPlan {
    #[default]
    Baseline,
    SoftcapPrune,
}

impl HeadRankingPlan {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::SoftcapPrune => "cpu-head-softcap-prune",
        }
    }
}

impl std::str::FromStr for HeadRankingPlan {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "cpu-head-softcap-prune" => Ok(Self::SoftcapPrune),
            _ => Err("unknown head-ranking candidate"),
        }
    }
}

/// No ignored hardware fixture, cache operation, or other owner can acquire
/// this host experiment accidentally via the text/worker path.
pub fn validate_head_ranking_mode(
    plan: HeadRankingPlan,
    timing: bool,
    prefill_only: bool,
    runtime_worker: bool,
    cache_operation: bool,
    compile_budget: usize,
    baseline_cached_ane: bool,
) -> Result<(), &'static str> {
    if plan == HeadRankingPlan::Baseline && !timing {
        return Ok(());
    }
    if prefill_only
        || runtime_worker
        || cache_operation
        || compile_budget != 0
        || !baseline_cached_ane
    {
        return Err("head-ranking experiments require serial cached ANE decode, not prefill/cache/worker modes or compilation");
    }
    Ok(())
}

/// Opting out preserves the incumbent checkpoint math. Only an explicitly
/// requested ranking experiment or its timed control requires softcap 30.
pub fn validate_head_ranking_configuration(
    plan: HeadRankingPlan,
    timing: bool,
    softcap: f32,
) -> Result<(), &'static str> {
    if plan == HeadRankingPlan::Baseline && !timing {
        return Ok(());
    }
    if softcap.to_bits() != 30.0_f32.to_bits() {
        return Err("head ranking experiments support the Gemma 4 12B softcap 30 only");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeadRankingStats {
    pub considered: u32,
    pub transformed: u32,
    pub pruned: u32,
}

/// A raw-logit witness accompanies each *rounded-softcap* winner. It is not
/// correct to keep just the largest five raw logits: FP16 rounding and tanh
/// saturation create ties whose first-seen token must remain the winner.
#[derive(Debug)]
pub struct HeadTop5 {
    plan: HeadRankingPlan,
    best: [(u32, f32); 5],
    raw_witness: [f32; 5],
    stats: HeadRankingStats,
}

impl HeadTop5 {
    pub fn new(plan: HeadRankingPlan, softcap: f32) -> Result<Self, &'static str> {
        if softcap.to_bits() != 30.0_f32.to_bits() {
            return Err("head ranking supports the Gemma 4 12B softcap 30 only");
        }
        Ok(Self {
            plan,
            best: [(u32::MAX, f32::NEG_INFINITY); 5],
            raw_witness: [f32::NEG_INFINITY; 5],
            stats: HeadRankingStats::default(),
        })
    }

    /// Accept exactly one complete, ascending vocabulary stream. All inputs
    /// are checked for finiteness, including rows that would otherwise prune.
    pub fn observe(&mut self, token: u32, value: f16) -> Result<(), &'static str> {
        if token != self.stats.considered || token == u32::MAX {
            return Err("head ranking requires contiguous ascending token IDs");
        }
        let raw = value.to_f32();
        if !raw.is_finite() {
            return Err("non-finite vocabulary projection");
        }
        self.stats.considered += 1;
        if self.plan == HeadRankingPlan::SoftcapPrune
            && self.best[4].0 != u32::MAX
            && raw <= self.raw_witness[4]
        {
            // The supplied exhaustive target-host test must establish that F is
            // nondecreasing on finite FP16 inputs before this route is qualified.
            // F(raw) <= F(witness) cannot beat the fifth entry under the
            // incumbent strict '>' insertion rule. No new tie-breaking rule.
            self.stats.pruned += 1;
            return Ok(());
        }
        let score = rounded_softcap(value);
        self.stats.transformed += 1;
        if let Some(rank) = self.best.iter().position(|&(_, best)| score > best) {
            self.best.copy_within(rank..4, rank + 1);
            self.raw_witness.copy_within(rank..4, rank + 1);
            self.best[rank] = (token, score);
            self.raw_witness[rank] = raw;
        }
        Ok(())
    }

    pub fn finish(&self, expected_rows: u32) -> Result<[(u32, f32); 5], &'static str> {
        if expected_rows < 5 || self.stats.considered != expected_rows {
            return Err("incomplete vocabulary stream");
        }
        Ok(self.best)
    }

    pub fn stats(&self) -> HeadRankingStats {
        self.stats
    }
}

/// Keep every incumbent FP16 boundary; do not replace this with raw argmax
/// or collapse the three roundings into one final cast.
#[inline]
fn rounded_softcap(value: f16) -> f32 {
    let divided = f16::from_f32(value.to_f32() / 30.0);
    let squashed = f16::from_f32(divided.to_f32().tanh());
    f16::from_f32(squashed.to_f32() * 30.0).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent all-transform reference, not the candidate's baseline arm.
    fn reference(values: &[f16]) -> [(u32, f32); 5] {
        let mut scored: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(id, x)| {
                let a = f16::from_f32(x.to_f32() / 30.0);
                let b = f16::from_f32(a.to_f32().tanh());
                let c = f16::from_f32(b.to_f32() * 30.0);
                (id as u32, c.to_f32())
            })
            .collect();
        // Stable order for equal floating-point scores, including signed zero.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        std::array::from_fn(|i| scored[i])
    }

    fn check(values: &[f16]) -> HeadRankingStats {
        let expected = reference(values);
        for plan in [HeadRankingPlan::Baseline, HeadRankingPlan::SoftcapPrune] {
            let mut head = HeadTop5::new(plan, 30.0).unwrap();
            for (id, value) in values.iter().copied().enumerate() {
                head.observe(id as u32, value).unwrap();
            }
            let got = head.finish(values.len() as u32).unwrap();
            assert_eq!(
                got.map(|(id, score)| (id, score.to_bits())),
                expected.map(|(id, score)| (id, score.to_bits()))
            );
            assert_eq!(
                head.stats().considered,
                head.stats().transformed + head.stats().pruned
            );
            if plan == HeadRankingPlan::SoftcapPrune {
                return head.stats();
            }
        }
        unreachable!()
    }

    #[test]
    fn default_configuration_preserves_baseline_softcap_flexibility() {
        // Baseline decode uses the checkpoint's softcap, not the candidate's
        // hardcoded 30. Opting out must not introduce a new model rejection.
        for softcap in [28.0, 30.0, 32.0] {
            assert!(
                validate_head_ranking_configuration(HeadRankingPlan::Baseline, false, softcap)
                    .is_ok()
            );
            for (plan, timing) in [
                (HeadRankingPlan::Baseline, true),
                (HeadRankingPlan::SoftcapPrune, false),
                (HeadRankingPlan::SoftcapPrune, true),
            ] {
                assert_eq!(
                    validate_head_ranking_configuration(plan, timing, softcap).is_ok(),
                    softcap == 30.0
                );
            }
        }
    }

    #[test]
    fn raw_top_five_is_not_a_valid_replacement_for_rounded_top_five() {
        let mut values = vec![f16::from_f32(256.0); 5];
        values.extend((0..64).map(|i| f16::from_f32(512.0 + i as f32)));
        check(&values);
        assert_eq!(reference(&values).map(|v| v.0), [0, 1, 2, 3, 4]);
        assert!(values[5] > values[0]);
    }

    #[test]
    fn every_finite_half_has_a_monotone_rounded_softcap_on_this_host() {
        let mut values: Vec<_> = (0..=u16::MAX)
            .map(f16::from_bits)
            .filter(|v| v.is_finite())
            .collect();
        values.sort_by(|a, b| a.to_f32().total_cmp(&b.to_f32()));
        for pair in values.windows(2) {
            assert!(
                rounded_softcap(pair[0]) <= rounded_softcap(pair[1]),
                "nonmonotone transform for {:04x}, {:04x}",
                pair[0].to_bits(),
                pair[1].to_bits()
            );
        }
        check(&values);
        values.reverse();
        assert!(check(&values).pruned > 60_000);
        // Fixed permutation of the entire finite domain, independent of ranker.
        let mut permuted: Vec<_> = (0..=u16::MAX)
            .map(|i| f16::from_bits(i.wrapping_mul(40503).wrapping_add(17)))
            .filter(|v| v.is_finite())
            .collect();
        check(&permuted);
        permuted.reverse();
        check(&permuted);
    }

    #[test]
    fn ties_signed_zero_negatives_and_stream_errors_are_preserved() {
        check(&[
            f16::NEG_ZERO,
            f16::ZERO,
            f16::NEG_ZERO,
            f16::ZERO,
            f16::ZERO,
            f16::from_f32(-1.0),
            f16::NEG_ZERO,
        ]);
        let descending: Vec<_> = (0..4096).map(|i| f16::from_f32(-(i as f32))).collect();
        assert_eq!(check(&descending).transformed, 5);
        let mut ranker = HeadTop5::new(HeadRankingPlan::SoftcapPrune, 30.0).unwrap();
        assert!(ranker.observe(1, f16::ZERO).is_err());
        assert!(ranker.observe(0, f16::NAN).is_err());
        for i in 0..5 {
            ranker.observe(i, f16::ZERO).unwrap();
        }
        assert!(ranker.observe(5, f16::NEG_INFINITY).is_err());
        assert!(ranker.finish(262144).is_err());
        assert!(HeadTop5::new(HeadRankingPlan::SoftcapPrune, f32::NAN).is_err());
        assert!(HeadTop5::new(HeadRankingPlan::SoftcapPrune, 0.0).is_err());
        assert_eq!(HeadRankingPlan::default(), HeadRankingPlan::Baseline);
        for name in ["auto", "prune", "cpu-head-softcap-prune "] {
            assert!(name.parse::<HeadRankingPlan>().is_err());
        }
    }
    #[test]
    fn head_experiment_mode_gate_fails_before_model_access() {
        for (prefill, worker, cache, budget) in [
            (true, false, false, 0),
            (false, true, false, 0),
            (false, false, true, 0),
            (false, false, false, 1),
        ] {
            assert!(validate_head_ranking_mode(
                HeadRankingPlan::SoftcapPrune,
                true,
                prefill,
                worker,
                cache,
                budget,
                true
            )
            .is_err());
            assert!(validate_head_ranking_mode(
                HeadRankingPlan::Baseline,
                true,
                prefill,
                worker,
                cache,
                budget,
                true
            )
            .is_err());
        }
        assert!(validate_head_ranking_mode(
            HeadRankingPlan::SoftcapPrune,
            true,
            false,
            false,
            false,
            0,
            true
        )
        .is_ok());
        assert!(validate_head_ranking_mode(
            HeadRankingPlan::Baseline,
            false,
            true,
            true,
            true,
            16,
            false
        )
        .is_ok());
        for plan in [HeadRankingPlan::Baseline, HeadRankingPlan::SoftcapPrune] {
            assert!(validate_head_ranking_mode(plan, true, false, false, false, 0, false).is_err());
        }
    }
}
