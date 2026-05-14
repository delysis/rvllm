#![cfg(feature = "mlx")]

use rvllm_apple::{
    HandoffCapsule, HandoffKind, MlxParityCase, MlxParityOutput, MlxReferenceHarness,
    MlxReferenceMode,
};
use rvllm_core::{AppleError, ReqId, RvllmError, TokenId};

#[test]
fn mlx_reference_harness_is_non_production_and_not_a_fallback() {
    let harness = MlxReferenceHarness::prototype();

    assert_eq!(harness.mode(), MlxReferenceMode::PrototypeOnly);
    assert!(harness.is_non_production());
    assert!(!harness.can_fallback());
}

#[test]
fn mlx_parity_case_scaffold_carries_valid_prefill_handoff() {
    let handoff = HandoffCapsule::new(
        HandoffKind::MetalPrefillToMetalDecode,
        vec![ReqId(7)],
        vec![TokenId(2), TokenId(42), TokenId(13)],
        vec![0, 3],
        vec![0],
        vec![3],
    );

    let case = match MlxParityCase::prefill_logits("gemma4-e2b-prefill-smoke", handoff, 8) {
        Ok(case) => case,
        Err(e) => panic!("unexpected parity case error: {e}"),
    };

    assert_eq!(case.name(), "gemma4-e2b-prefill-smoke");
    assert_eq!(
        case.output(),
        MlxParityOutput::PrefillLogits { tolerance_ulps: 8 }
    );
    assert!(case.handoff().is_well_formed());
}

#[test]
fn mlx_reference_run_requires_explicit_executor_instead_of_fallback() {
    let handoff = HandoffCapsule::new(
        HandoffKind::MetalPrefillToMetalDecode,
        vec![ReqId(11)],
        vec![TokenId(2), TokenId(99)],
        vec![0, 2],
        vec![0],
        vec![2],
    );
    let case = match MlxParityCase::prefill_logits("gemma4-e2b-missing-executor", handoff, 8) {
        Ok(case) => case,
        Err(e) => panic!("unexpected parity case error: {e}"),
    };
    let harness = MlxReferenceHarness::prototype();

    let err = match harness.run_parity_case(&case) {
        Ok(_) => panic!("prototype MLX harness unexpectedly ran without an executor"),
        Err(err) => err,
    };

    match err {
        RvllmError::Apple {
            err: AppleError::FeatureNotAvailable { backend, op },
            ctx,
            ..
        } => {
            assert_eq!(backend, "mlx-reference");
            assert_eq!(op, "run_parity_case");
            assert_eq!(ctx.backend, "mlx-reference");
            assert_eq!(ctx.op, "run_parity_case");
        }
        other => panic!("unexpected error type: {other}"),
    }
}
