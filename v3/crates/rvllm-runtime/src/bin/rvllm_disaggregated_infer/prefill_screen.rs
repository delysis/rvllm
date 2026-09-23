//! Receipt and mode policy only. No model load, device access, or ANE call.
#![forbid(unsafe_code)]

use rvllm_apple_metal::research_evidence::{
    ResearchDispatchSnapshot, RESEARCH_DISPATCH_SCHEMA, RESEARCH_KERNEL_NAMES,
};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PrefillScreenOptions {
    pub enabled: bool,
    pub reference_count: usize,
    pub output_requested: bool,
    pub text_input: bool,
    pub interactive: bool,
    pub interleave: bool,
    pub retain_metal: bool,
    pub runtime_worker: bool,
    pub capture_ane: bool,
    pub cache_operation: bool,
    pub nonbaseline_ane: bool,
    pub nonbaseline_packing: bool,
    pub compile_budget: usize,
}

impl PrefillScreenOptions {
    pub(super) fn validate(self) -> Result<(), &'static str> {
        if !self.enabled {
            return Ok(());
        }
        if !(1..=16).contains(&self.reference_count) || !self.output_requested || self.text_input {
            return Err(
                "prefill-only requires 1..=16 --hf-reference inputs and a fresh --output-dir; no text input",
            );
        }
        if self.interactive
            || self.interleave
            || self.retain_metal
            || self.runtime_worker
            || self.capture_ane
            || self.cache_operation
            || self.nonbaseline_ane
            || self.nonbaseline_packing
            || self.compile_budget != 0
        {
            return Err(
                "prefill-only cannot prepare/inspect ANE, compile, decode, or select ANE/KV candidates",
            );
        }
        Ok(())
    }
}

/// A misspelling must not turn a requested screen into a successful baseline.
/// The normal CLI's established environment fallback remains unchanged.
pub(super) fn validate_requested_candidate(
    requested: Result<&str, &std::env::VarError>,
    resolved: rvllm_apple_metal::MetalResearchCandidate,
) -> Result<(), &'static str> {
    use rvllm_apple_metal::MetalResearchCandidate;
    let requested = match requested {
        Ok(value) => value.parse::<MetalResearchCandidate>()?,
        Err(std::env::VarError::NotPresent) => MetalResearchCandidate::Off,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("non-Unicode RVLLM_METAL_RESEARCH selector");
        }
    };
    if requested != resolved {
        return Err("requested Metal candidate differs from resolved options");
    }
    Ok(())
}

pub(super) fn dispatch_report(
    requested: &str,
    before: ResearchDispatchSnapshot,
    after: ResearchDispatchSnapshot,
) -> Result<Value, &'static str> {
    let delta = after.checked_since(before)?;
    let exercised = delta.selection_exercised(requested)?;
    let complete_family = delta.complete_family_exercised(requested)?;
    let counts: serde_json::Map<String, Value> = RESEARCH_KERNEL_NAMES
        .iter()
        .zip(delta.counts)
        .map(|(&name, count)| (name.to_owned(), json!(count)))
        .collect();
    Ok(json!({
        "schema": RESEARCH_DISPATCH_SCHEMA,
        "requested": requested,
        "selection_exercised": exercised,
        "complete_family_exercised": complete_family,
        "encoded_dispatches": counts,
        "counting_boundary": "post-encode counters bracketed around synchronous prefill collection",
        "all_eligible_layers_exercised": null,
        "tensor_oracle_passed": null,
        "performance_accepted": false,
    }))
}

/// Do not overwrite even a failed or incomplete receipt. Callers own a fresh
/// directory. Flush outside the measured prefill interval, before ANE loading.
pub(super) fn write_new_json(
    path: &Path,
    report: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create_new(path)?;
    let mut output = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_options() -> PrefillScreenOptions {
        PrefillScreenOptions {
            enabled: true,
            reference_count: 1,
            output_requested: true,
            ..PrefillScreenOptions::default()
        }
    }

    #[test]
    fn prefill_only_requires_an_unchanged_reference_and_no_decode_options() {
        assert!(valid_options().validate().is_ok());
        for bad in [
            PrefillScreenOptions {
                reference_count: 0,
                ..valid_options()
            },
            PrefillScreenOptions {
                reference_count: 17,
                ..valid_options()
            },
            PrefillScreenOptions {
                reference_count: usize::MAX,
                ..valid_options()
            },
            PrefillScreenOptions {
                output_requested: false,
                ..valid_options()
            },
            PrefillScreenOptions {
                text_input: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                interactive: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                interleave: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                retain_metal: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                runtime_worker: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                capture_ane: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                cache_operation: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                nonbaseline_ane: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                nonbaseline_packing: true,
                ..valid_options()
            },
            PrefillScreenOptions {
                compile_budget: 1,
                ..valid_options()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert!(PrefillScreenOptions::default().validate().is_ok());
    }

    #[test]
    fn receipt_preserves_fallback_and_rejects_mixed_dispatches() {
        let zero = ResearchDispatchSnapshot::default();
        let fallback = dispatch_report("metal-gqa-kv8", zero, zero).unwrap();
        assert_eq!(fallback["selection_exercised"], false);
        assert_eq!(fallback["encoded_dispatches"]["research_gqa_kv8_d256"], 0);
        let mut gqa = ResearchDispatchSnapshot::default();
        gqa.counts[3] = 40;
        gqa.counts[4] = 8;
        let report = dispatch_report("metal-gqa-kv8", zero, gqa).unwrap();
        assert_eq!(report["selection_exercised"], true);
        assert_eq!(report["encoded_dispatches"]["research_gqa_kv8_d512"], 8);
        assert!(dispatch_report("metal-rounded-gate32", zero, gqa).is_err());
        assert!(dispatch_report("metal-gqa-kv8", gqa, zero).is_err());
    }

    #[test]
    fn partial_family_is_preserved_as_partial_not_full_coverage() {
        let zero = ResearchDispatchSnapshot::default();
        let mut partial = zero;
        partial.counts[10] = 48;
        let report = dispatch_report("metal-mma32-f32", zero, partial).unwrap();
        assert_eq!(report["selection_exercised"], true);
        assert_eq!(report["complete_family_exercised"], false);
        assert_eq!(report["schema"], RESEARCH_DISPATCH_SCHEMA);
        assert!(report["all_eligible_layers_exercised"].is_null());
        assert!(report["tensor_oracle_passed"].is_null());
        partial.counts[11] = 48;
        assert_eq!(
            dispatch_report("metal-mma32-f32", zero, partial).unwrap()["complete_family_exercised"],
            true
        );
    }

    #[test]
    fn screen_rejects_unresolved_or_malformed_selectors() {
        use rvllm_apple_metal::MetalResearchCandidate::{GqaKv8, Off};
        assert!(validate_requested_candidate(Ok("off"), Off).is_ok());
        assert!(validate_requested_candidate(Ok("metal-gqa-kv8"), GqaKv8).is_ok());
        assert!(validate_requested_candidate(Err(&std::env::VarError::NotPresent), Off).is_ok());
        assert!(validate_requested_candidate(Ok("metal-gqa-kv8"), Off).is_err());
        assert!(validate_requested_candidate(Ok("metal-gqa-kv8 "), Off).is_err());
        assert!(validate_requested_candidate(Ok("auto"), Off).is_err());
        let error = std::env::VarError::NotUnicode(std::ffi::OsString::new());
        assert!(validate_requested_candidate(Err(&error), Off).is_err());
    }

    #[test]
    fn existing_receipts_are_never_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prefill-result.json");
        let failed = json!({"first_token_gate_passed": false, "qualification_complete": false});
        write_new_json(&path, &failed).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(write_new_json(&path, &json!({"first_token_gate_passed": true})).is_err());
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}
