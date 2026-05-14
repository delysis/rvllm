const APPLE_SPEC: &str = include_str!("../../../specs/17-apple.md");

#[test]
fn apple_spec_documents_risk_modes_failures_and_benchmarks() {
    for heading in [
        "## Private API risk",
        "## Supported modes",
        "## Failure modes",
        "## Benchmark interpretation",
    ] {
        assert!(
            APPLE_SPEC.contains(heading),
            "v3/specs/17-apple.md must document {heading}"
        );
    }

    for mode in [
        "`MetalOnly`",
        "`MlxPrototype`",
        "`MetalPrefillMetalDecode`",
        "`MetalPrefillAneFfnRollout`",
        "`MetalPrefillAneRolloutExperimental`",
    ] {
        assert!(
            APPLE_SPEC.contains(mode),
            "v3/specs/17-apple.md must describe supported mode {mode}"
        );
    }

    for typed_error in [
        "`AppleError::PrivateApiUnavailable`",
        "`AppleError::ShapeBucketMissing`",
        "`AppleError::HandoffMalformed`",
        "`AppleError::FeatureNotAvailable`",
    ] {
        assert!(
            APPLE_SPEC.contains(typed_error),
            "v3/specs/17-apple.md must name typed failure {typed_error}"
        );
    }

    for benchmark_term in [
        "TTFT",
        "decode tokens/sec",
        "energy",
        "CUDA",
        "no implicit fallback",
    ] {
        assert!(
            APPLE_SPEC.contains(benchmark_term),
            "v3/specs/17-apple.md benchmark guidance must mention {benchmark_term}"
        );
    }
}
