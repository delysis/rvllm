//! Fail-closed referee for ANE candidates claiming fewer program or convolution boundaries.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

const INPUT_SCHEMA: &str = "rvllm.gemma4_ane_boundary_candidate.v1";
const RECEIPT_SCHEMA: &str = "rvllm.gemma4_ane_boundary_referee.v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    control: BoundaryPlan,
    candidate: BoundaryPlan,
    evidence: Evidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryPlan {
    name: String,
    programs_per_layer: usize,
    convolutions_per_layer: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    route_selected: bool,
    compiler_calls: usize,
    cache_entries_complete: bool,
    real_activation_oracle_passed: bool,
    real_activations_compared: usize,
    full_route_token_match: bool,
    ane_execution_verified: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Receipt {
    schema: &'static str,
    accepted: bool,
    candidate_name: String,
    program_boundaries_removed_per_layer: usize,
    convolution_boundaries_removed_per_layer: usize,
    compiler_calls: usize,
    real_activations_compared: usize,
    timing_claim: bool,
}

fn referee(input: Input) -> Result<Receipt, String> {
    if input.schema != INPUT_SCHEMA {
        return Err(format!("input schema must be {INPUT_SCHEMA}"));
    }
    let reduced_programs = input.candidate.programs_per_layer < input.control.programs_per_layer;
    let reduced_convolutions =
        input.candidate.convolutions_per_layer < input.control.convolutions_per_layer;
    if !reduced_programs && !reduced_convolutions {
        return Err("candidate reduces neither program nor convolution boundaries".into());
    }
    if !input.evidence.route_selected {
        return Err("the requested candidate route was not selected".into());
    }
    if input.evidence.compiler_calls != 0 {
        return Err("qualification requires exactly zero compiler calls".into());
    }
    if !input.evidence.cache_entries_complete {
        return Err("all candidate cache entries must be verified present".into());
    }
    if !input.evidence.real_activation_oracle_passed {
        return Err("a real-activation oracle is required".into());
    }
    if input.evidence.real_activations_compared == 0 {
        return Err("the real-activation oracle compared no activations".into());
    }
    if !input.evidence.full_route_token_match {
        return Err("full-route token equality is required".into());
    }
    if !input.evidence.ane_execution_verified {
        return Err("ANE execution must be verified".into());
    }

    Ok(Receipt {
        schema: RECEIPT_SCHEMA,
        accepted: true,
        candidate_name: input.candidate.name,
        program_boundaries_removed_per_layer: input
            .control
            .programs_per_layer
            .saturating_sub(input.candidate.programs_per_layer),
        convolution_boundaries_removed_per_layer: input
            .control
            .convolutions_per_layer
            .saturating_sub(input.candidate.convolutions_per_layer),
        compiler_calls: 0,
        real_activations_compared: input.evidence.real_activations_compared,
        timing_claim: false,
    })
}

fn run(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let input: Input = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let receipt = referee(input)?;
    serde_json::to_string(&receipt).map_err(|error| format!("serialize receipt: {error}"))
}

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(path) = arguments.next() else {
        eprintln!("usage: {} INPUT.json", Path::new(&program).display());
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {} INPUT.json", Path::new(&program).display());
        std::process::exit(2);
    }
    match run(Path::new(&path)) {
        Ok(receipt) => println!("{receipt}"),
        Err(error) => {
            eprintln!("ANE boundary referee: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Input {
        Input {
            schema: INPUT_SCHEMA.into(),
            control: BoundaryPlan {
                name: "separate-attention-output".into(),
                programs_per_layer: 2,
                convolutions_per_layer: 1,
            },
            candidate: BoundaryPlan {
                name: "fused-attention-output".into(),
                programs_per_layer: 1,
                convolutions_per_layer: 1,
            },
            evidence: Evidence {
                route_selected: true,
                compiler_calls: 0,
                cache_entries_complete: true,
                real_activation_oracle_passed: true,
                real_activations_compared: 48,
                full_route_token_match: true,
                ane_execution_verified: true,
            },
        }
    }

    #[test]
    fn accepts_real_program_boundary_reduction() {
        let receipt = referee(fixture()).unwrap();
        assert!(receipt.accepted);
        assert_eq!(receipt.program_boundaries_removed_per_layer, 1);
        assert!(!receipt.timing_claim);
    }

    #[test]
    fn rejects_tiling_only_candidate() {
        let mut input = fixture();
        input.candidate.programs_per_layer = 2;
        let error = referee(input).unwrap_err();
        assert!(error.contains("neither program nor convolution"));
    }

    #[test]
    fn rejects_compile_and_oracle_loopholes() {
        let mut compiled = fixture();
        compiled.evidence.compiler_calls = 1;
        assert!(referee(compiled).is_err());

        let mut empty_oracle = fixture();
        empty_oracle.evidence.real_activations_compared = 0;
        assert!(referee(empty_oracle).is_err());

        let mut fallback = fixture();
        fallback.evidence.route_selected = false;
        assert!(referee(fallback).is_err());
    }

    #[test]
    fn strict_schema_rejects_unknown_and_negative_fields() {
        let unknown = br#"{"schema":"rvllm.gemma4_ane_boundary_candidate.v1","control":{"name":"a","programs_per_layer":2,"convolutions_per_layer":1},"candidate":{"name":"b","programs_per_layer":1,"convolutions_per_layer":1,"extra":true},"evidence":{"route_selected":true,"compiler_calls":0,"cache_entries_complete":true,"real_activation_oracle_passed":true,"real_activations_compared":1,"full_route_token_match":true,"ane_execution_verified":true}}"#;
        assert!(serde_json::from_slice::<Input>(unknown).is_err());

        let negative = br#"{"schema":"rvllm.gemma4_ane_boundary_candidate.v1","control":{"name":"a","programs_per_layer":2,"convolutions_per_layer":1},"candidate":{"name":"b","programs_per_layer":-1,"convolutions_per_layer":1},"evidence":{"route_selected":true,"compiler_calls":0,"cache_entries_complete":true,"real_activation_oracle_passed":true,"real_activations_compared":1,"full_route_token_match":true,"ane_execution_verified":true}}"#;
        assert!(serde_json::from_slice::<Input>(negative).is_err());
    }
}
