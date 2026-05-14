use rvllm_core::{AppleCtx, AppleError, RvllmError};

#[test]
fn apple_error_display_uses_reexported_types_and_context() {
    let err = RvllmError::apple(
        AppleError::ShapeBucketMissing {
            seqs: 17,
            tokens: 9,
        },
        AppleCtx {
            backend: "private-ane",
            op: "rollout",
            device: "Apple M4 Max",
        },
    );

    assert_eq!(
        format!("{err}"),
        "apple: ShapeBucketMissing seqs=17 tokens=9 backend=private-ane op=rollout device=Apple M4 Max"
    );
}

#[test]
fn apple_private_api_display_names_symbol() {
    let err = RvllmError::apple(
        AppleError::PrivateApiUnavailable {
            symbol: "ANECompileForEvaluation",
        },
        AppleCtx {
            backend: "private-ane",
            op: "compile",
            device: "Apple M4 Max",
        },
    );

    assert_eq!(
        format!("{err}"),
        "apple: PrivateApiUnavailable symbol=ANECompileForEvaluation backend=private-ane op=compile device=Apple M4 Max"
    );
}
