#!/usr/bin/env python3
"""Strict validator for the native-BF16 real-weight projection referee."""

import argparse
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--candidate", choices=("scalar", "n4"))
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    require(receipt.get("schema") == "rvllm.metal_low_bit_real_weight_bf16.v1", "bad schema")
    require(receipt.get("tensor") == args.tensor, "wrong tensor")
    require(receipt.get("tensor_role") == args.role, "wrong role")
    require(receipt.get("source_dtype") == "Bf16", "source was not BF16")
    require(
        receipt.get("abi")
        == {"activation": "BF16", "output": "BF16", "scales": "F16", "accumulation": "F32"},
        "wrong ABI",
    )
    if args.candidate is not None:
        require(receipt.get("candidate_schedule") == args.candidate, "wrong candidate schedule")
    cases = receipt.get("cases")
    require(isinstance(cases, list) and len(cases) == 4, "expected four cases")
    expected = {(fmt, m) for fmt in ("w4a16", "w8a16") for m in (1, 4)}
    seen = set()
    for case in cases:
        dispatch = case.get("dispatch", {})
        key = (dispatch.get("format"), case.get("m"))
        require(key in expected and key not in seen, f"unexpected or duplicate case {key}")
        seen.add(key)
        require(dispatch.get("role") == args.role, f"wrong dispatch role for {key}")
        abi = "W4ABF16" if key[0] == "w4a16" else "W8ABF16"
        require(dispatch.get("candidate_abi") == abi, f"wrong candidate ABI for {key}")
        require(dispatch.get("exact_correctness_dispatches_verified") == 2, f"bad correctness count for {key}")
        require(dispatch.get("exact_timing_dispatch_count_verified") is True, f"timing dispatch unverified for {key}")
        require(dispatch.get("timing_dispatches") == 1 + 2 * args.samples, f"bad timing count for {key}")
        require(case.get("guard_unchanged") is True, f"guard failure for {key}")
        require(case.get("repeatable_output_bits") is True, f"repeat failure for {key}")
        for accuracy_name in ("accuracy", "native_accuracy"):
            accuracy = case.get(accuracy_name, {})
            for metric in ("max_abs", "relative_l2"):
                value = accuracy.get(metric)
                require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, f"bad {accuracy_name}.{metric} for {key}")
        timing = case.get("timing", {})
        if args.candidate is not None:
            suffix = "_n4" if args.candidate == "n4" else ""
            expected_kernel = (
                f"experimental_projection_w4abf16_bf16{suffix}"
                if key[0] == "w4a16"
                else f"experimental_projection_w8abf16_bf16{suffix}"
            )
            require(timing.get("candidate_kernel") == expected_kernel, f"wrong candidate kernel for {key}")
        require(timing.get("method") == "ABBA wall-clock commit-to-completion", f"wrong timing method for {key}")
        require(timing.get("samples_per_arm") == 2 * args.samples, f"wrong sample count for {key}")
        for arm in ("native_ms", "candidate_ms"):
            values = timing.get(arm)
            require(isinstance(values, list) and len(values) == 2 * args.samples, f"wrong {arm} length for {key}")
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values), f"bad {arm} value for {key}")
        require(all(len(value) == 64 for value in case.get("identity", {}).values()), f"bad case identity for {key}")
    require(seen == expected, "case matrix incomplete")
    for field in ("config_sha256", "source_tensor_sha256", "generated_msl_sha256", "executable_sha256"):
        require(isinstance(receipt.get(field), str) and len(receipt[field]) == 64, f"bad {field}")
    print(json.dumps({"validated": True, "tensor": args.tensor, "role": args.role, "cases": len(cases), "samples": args.samples}, sort_keys=True))


if __name__ == "__main__":
    main()
