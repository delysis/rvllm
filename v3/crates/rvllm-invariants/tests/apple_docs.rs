use std::fs;
use std::path::{Path, PathBuf};

fn apple_spec_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("specs/17-apple.md")
}

#[test]
fn apple_spec_documents_risk_modes_failures_and_benchmark_interpretation() {
    let spec = fs::read_to_string(apple_spec_path()).expect("read Apple spec");
    let required_sections = [
        "## Private API risk",
        "## Supported modes",
        "## Failure modes",
        "## Benchmark interpretation",
    ];

    let missing = required_sections
        .iter()
        .filter(|section| !spec.contains(**section))
        .copied()
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty(),
        "Apple spec missing required docs sections: {}",
        missing.join(", ")
    );
}
