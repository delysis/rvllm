#!/usr/bin/env python3
"""Run repeated rvLLM Apple Metal E2B profile samples.

This wrapper drives the existing ignored Rust profile test and summarizes the
JSON artifacts it emits. It is local regression instrumentation only: it does
not collect external GPU/CPU/memory/energy counters and does not establish
production performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROFILE_TEST = "real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def measured_metric(artifact: dict[str, Any], name: str) -> float:
    metric = artifact["sample"]["metrics"][name]
    measured = metric.get("Measured")
    if not isinstance(measured, dict):
        raise ValueError(f"{name} is not measured in {artifact['sample']['sample_id']}")
    value = float(measured["value"])
    if not math.isfinite(value):
        raise ValueError(f"{name} is non-finite in {artifact['sample']['sample_id']}: {value}")
    return value


def summarize(paths: Sequence[Path], output: Path) -> None:
    artifacts = [load_json(path) for path in paths]
    decode = [measured_metric(artifact, "steady_decode_tokens_per_second") for artifact in artifacts]
    prefill = [measured_metric(artifact, "prefill_tokens_per_second") for artifact in artifacts]
    command_buffers = [
        measured_metric(artifact, "command_buffers_per_token") for artifact in artifacts
    ]
    encoders_per_token = [
        artifact["metal_probe_counters"]["encoders"] / artifact["metal_probe_counters"]["tokens"]
        for artifact in artifacts
    ]
    forced_waits_per_token = [
        artifact["metal_probe_counters"]["forced_waits"] / artifact["metal_probe_counters"]["tokens"]
        for artifact in artifacts
    ]
    payload = {
        "schema": "rvllm.e2b_metal_profile_sample_summary.v1",
        "sample_count": len(artifacts),
        "sample_ids": [artifact["sample"]["sample_id"] for artifact in artifacts],
        "artifacts": [str(path) for path in paths],
        "medians": {
            "steady_decode_tokens_per_second": statistics.median(decode),
            "prefill_tokens_per_second": statistics.median(prefill),
            "command_buffers_per_token": statistics.median(command_buffers),
            "encoders_per_token": statistics.median(encoders_per_token),
            "forced_waits_per_token": statistics.median(forced_waits_per_token),
        },
        "minimums": {
            "steady_decode_tokens_per_second": min(decode),
            "prefill_tokens_per_second": min(prefill),
        },
        "maximums": {
            "steady_decode_tokens_per_second": max(decode),
            "prefill_tokens_per_second": max(prefill),
            "command_buffers_per_token": max(command_buffers),
            "encoders_per_token": max(encoders_per_token),
            "forced_waits_per_token": max(forced_waits_per_token),
        },
        "claim_boundary": "local repeated Metal-only probe summary; not production performance, ANE, or external profiler evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote E2B profile summary to {output}")


def run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("RVLLM_GEMMA4_MODEL_DIR"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rvllm-e2b-metal-profile-samples"))
    parser.add_argument("--sample-prefix", default="real-e2b-metal-probe")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print cargo commands without running them")
    args = parser.parse_args(argv)

    if args.model_dir is None:
        parser.error("--model-dir or RVLLM_GEMMA4_MODEL_DIR is required")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.output_dir / "summary.json")
    artifact_paths = []
    for index in range(args.samples):
        sample_id = f"{args.sample_prefix}-{index + 1:02d}"
        artifact_path = args.output_dir / f"{sample_id}.json"
        artifact_paths.append(artifact_path)
        env = os.environ.copy()
        env["RVLLM_GEMMA4_MODEL_DIR"] = str(args.model_dir)
        env["RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE"] = "1"
        env["RVLLM_E2B_PROFILE_SAMPLE_ID"] = sample_id
        env["RVLLM_E2B_PROFILE_JSON"] = str(artifact_path)
        cmd = [
            "cargo",
            "test",
            "-p",
            "rvllm-runtime",
            "--features",
            "apple",
            PROFILE_TEST,
            "--",
            "--ignored",
            "--nocapture",
        ]
        print(" ".join(cmd), f"# sample_id={sample_id} output={artifact_path}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=env)

    if args.dry_run:
        payload = {
            "schema": "rvllm.e2b_metal_profile_sample_summary.v1",
            "dry_run": True,
            "sample_count": args.samples,
            "artifacts": [str(path) for path in artifact_paths],
            "claim_boundary": "planned local repeated Metal-only probe summary; not production performance, ANE, or external profiler evidence",
        }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote dry-run E2B profile summary to {summary_path}")
    else:
        summarize(artifact_paths, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
