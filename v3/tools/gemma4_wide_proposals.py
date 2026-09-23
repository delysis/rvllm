#!/usr/bin/env python3
"""Validate UNPINNED proposals or render a compile-only command plan.

This tool does not launch commands, mutate a queue, discover local identities,
write manifests, authorize trials, or turn staging metadata into acceptance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

BASE = "faa753360e30ce59540d1d113994c711a1d19675"
METAL = ("metal-mma32-prefetch", "metal-attn-q4", "metal-rms-simd32")
CONTROLS = {**{name: "metal-off-mma32-simdgroup" for name in METAL},
            "ane-int8-ffn-down4": "static-int8-ffn-cached",
            "ane-int8-ffn-packed32": "static-int8-ffn-cached",
            "ane-attention-transpose-flags": "static-int8-ffn-cached",
            "cpu-head-softcap-prune": "head-ranking-baseline"}
SELECTORS = {**{name: ["RVLLM_METAL_RESEARCH", name] for name in METAL},
             "ane-int8-ffn-down4": ["--ane-weights", "static-int8-down4-ffn-cached"],
             "ane-int8-ffn-packed32": None,
             "ane-attention-transpose-flags": ["--ane-weights", "static-int8-ffn-transpose-attention-cached"],
             "cpu-head-softcap-prune": ["--head-ranking", "cpu-head-softcap-prune"]}


def read_proposal_text(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON: {value}")
    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("proposal must be a JSON object")
    return value


def validate_catalog(cases: list[dict]) -> None:
    names = [case.get("candidate") for case in cases]
    if len(names) != len(set(names)) or set(names) != set(CONTROLS):
        raise ValueError("require exactly the seven reviewed independent candidates")
    for case in cases:
        name = case["candidate"]
        if case.get("schema") != "rvllm.kernel_proposal.v1" or case.get("base_commit") != BASE:
            raise ValueError("wrong proposal schema/base; this is not an experiment-job schema")
        if case.get("admitted_to_queue") is not False:
            raise ValueError("staged proposals do not authorize live execution")
        if case.get("control") != CONTROLS[name] or case.get("selector") != SELECTORS[name]:
            raise ValueError("wrong control or unimplemented selector")
        pins = case.get("local_pins", {})
        if set(pins) != {"executable_sha256", "metallib_sha256", "reference_sha256", "model_config_sha256", "oracle_sha256"}:
            raise ValueError("local pin schema mismatch")
        if any(value is not None for value in pins.values()):
            raise ValueError("this source proposal must not invent/freeze local pins; create new native jobs separately")
        stratum = case["stratum"]
        if any(stratum.get(k) is not None for k in ("power_source", "low_power", "pmset_mode", "thermal")):
            raise ValueError("local owner must choose and observe one stratum, not inherit a historical one")
        if stratum.get("min_free_bytes") != 16 * 1024**3 or stratum.get("quiet_seconds") != 120:
            raise ValueError("launch controls changed")
        if stratum.get("idle_process_exemptions") != []:
            raise ValueError("no historical process exemptions are authorized by a source proposal")
        if case.get("metal_environment_common") != {
            "RVLLM_METAL_PREFILL_GEMM": "mma32",
            "RVLLM_METAL_PREFILL_ATTENTION": "simdgroup",
        }:
            raise ValueError("common Metal control configuration changed")
        timing = case["timing"]
        work = {"prompt_tokens": 84, "outputs_per_request": 10,
                "ane_steps_per_request": 9, "ane_evaluations_per_step": 208,
                "requests_per_block": 9, "ane_evaluations_per_full_abba": 67392,
                "independent_confirmation_required": True}
        if any(timing.get(key) != value for key, value in work.items()):
            raise ValueError("predeclared full-route work counts/confirmation changed")
        if (timing.get("block_order"), timing.get("warmups_per_block"), timing.get("measured_per_block"),
            timing.get("baseline_drift_limit"), timing.get("normalize_accelerator_by_cpu_cycles")) != (
                ["A", "B", "B", "A"], 2, 7, 0.05, False):
            raise ValueError("predeclared matched timing contract changed")
        if case.get("acceptance", {}).get("tensor_tolerances") != "unchanged-existing-oracle-pinned-locally":
            raise ValueError("a proposal cannot waive or invent device tolerances")
        blocked = name == "ane-int8-ffn-packed32"
        expected = "blocked-missing-compiled-io-descriptor" if blocked else "source-ready-native-checks-unrun"
        if case.get("status") != expected or case.get("live_command") is not None:
            raise ValueError("source-only staging must remain non-executable")
        if not case.get("experiment") or not case.get("risks"):
            raise ValueError("missing bounded experiment or risk statement")


def compile_recipe(workspace: str, target: str, output: str) -> list[dict]:
    """Print one invocation of the integrated twenty-two-arm native gate.

    This tool still does not execute the command. There is no second, divergent
    compile pipeline or a second six-arm export after the gate. The gate itself
    retains exclusive output creation, per-build executable hashes, pre-export
    checks, final source checks, nonzero test checks and no inference.
    """
    if not all(Path(p).is_absolute() for p in (workspace, target, output)):
        raise ValueError("absolute local workspace/target/fresh-output required")
    return [{"label": "integrated-native-delivery-gate", "cwd": workspace,
             "argv": ["bash", str(Path(workspace)/"tools/check_gemma4_candidate_delivery.sh"),
                      output, target],
             "expected_compile_link_arms": 22,
             "execution_authority": "local-owner-only"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="checked-in unpinned proposal directory")
    parser.add_argument("--render-compile-plan", nargs=3, metavar=("WORKSPACE", "TARGET", "NEW_OUTPUT"))
    args = parser.parse_args()
    cases = [read_proposal_text(p.read_text()) for p in sorted(args.directory.glob("*.json"))]
    validate_catalog(cases)
    value = {"status": "staged-only-not-authorized-not-queued", "count": len(cases)}
    if args.render_compile_plan:
        value["compile_plan"] = compile_recipe(*args.render_compile_plan)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
