#!/usr/bin/env python3
"""Compare rvLLM Apple Metal E2B profile artifacts.

This is a local regression gate for the probe artifact emitted by
`real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters`.
It is intentionally narrower than a production benchmark suite: it compares
two single-host Metal-only artifacts and preserves unmeasured external profiler
slots such as GPU utilization, peak memory, energy, and ANE utilization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_ARTIFACT_KIND = "rvllm-real-e2b-metal-probe-profile"


class GateFailure(Exception):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except OSError as exc:
        raise GateFailure(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateFailure(f"failed to parse JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateFailure(f"{path} must contain a JSON object")
    return value


def _get(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise GateFailure(f"missing required field: {path}")
        cur = cur[part]
    return cur


def _metric_measured(artifact: dict[str, Any], name: str) -> float:
    metric = _get(artifact, f"sample.metrics.{name}")
    if not isinstance(metric, dict) or "Measured" not in metric:
        raise GateFailure(f"sample.metrics.{name} must be measured")
    measured = metric["Measured"]
    if not isinstance(measured, dict) or "value" not in measured:
        raise GateFailure(f"sample.metrics.{name}.Measured.value is missing")
    value = measured["value"]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GateFailure(f"sample.metrics.{name}.Measured.value must be finite")
    return float(value)


def _metric_recorded(artifact: dict[str, Any], name: str) -> None:
    metric = _get(artifact, f"sample.metrics.{name}")
    if not isinstance(metric, dict) or len(metric) != 1:
        raise GateFailure(f"sample.metrics.{name} must record exactly one metric state")
    state, payload = next(iter(metric.items()))
    if state not in {"Measured", "Unmeasured", "Unsupported"}:
        raise GateFailure(f"sample.metrics.{name} has unknown metric state {state!r}")
    if state == "Measured":
        _metric_measured(artifact, name)
        return
    if not isinstance(payload, dict) or not str(payload.get("reason", "")).strip():
        raise GateFailure(f"sample.metrics.{name}.{state}.reason must be non-empty")


def _percent_regression(baseline: float, current: float, higher_is_better: bool) -> float:
    if baseline <= 0.0:
        raise GateFailure("baseline metric must be positive for regression comparison")
    if higher_is_better:
        return max(0.0, (baseline - current) / baseline * 100.0)
    return max(0.0, (current - baseline) / baseline * 100.0)


def _assert_same(baseline: dict[str, Any], current: dict[str, Any], field: str) -> None:
    left = _get(baseline, field)
    right = _get(current, field)
    if left != right:
        raise GateFailure(f"{field} mismatch: baseline={left!r} current={right!r}")


def compare_artifacts(
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_decode_regression_pct: float,
    max_prefill_regression_pct: float,
    max_encoder_regression_pct: float,
    max_forced_wait_regression_pct: float,
) -> dict[str, Any]:
    for name, artifact in (("baseline", baseline), ("current", current)):
        if artifact.get("artifact_kind") != EXPECTED_ARTIFACT_KIND:
            raise GateFailure(
                f"{name}.artifact_kind must be {EXPECTED_ARTIFACT_KIND!r}, got {artifact.get('artifact_kind')!r}"
            )
        if artifact.get("schema_version") != 1:
            raise GateFailure(f"{name}.schema_version must be 1")
        sample_id = str(_get(artifact, "sample.sample_id")).strip()
        if not sample_id:
            raise GateFailure(f"{name}.sample.sample_id must be non-empty")
        if bool(_get(artifact, "sample.toy_backend_enabled")):
            raise GateFailure(f"{name} artifact was produced by a toy backend")
        for metric in (
            "steady_decode_tokens_per_second",
            "prefill_tokens_per_second",
            "command_buffers_per_token",
        ):
            _metric_measured(artifact, metric)
        for metric in (
            "first_token_latency_ms",
            "memory_peak_bytes",
            "cpu_utilization_percent",
            "gpu_utilization_percent",
            "ane_utilization_percent",
            "energy_joules",
        ):
            _metric_recorded(artifact, metric)

        counters = _get(artifact, "metal_probe_counters")
        if not isinstance(counters, dict):
            raise GateFailure(f"{name}.metal_probe_counters must be an object")
        prefill_steps = int(_get(artifact, "metal_probe_counters.prefill_steps"))
        decode_steps = int(_get(artifact, "metal_probe_counters.decode_steps"))
        command_buffers = int(_get(artifact, "metal_probe_counters.command_buffers"))
        if command_buffers != prefill_steps + decode_steps:
            raise GateFailure(
                f"{name} command_buffers={command_buffers} must equal prefill_steps+decode_steps={prefill_steps + decode_steps}"
            )
        if int(_get(artifact, "metal_probe_counters.library_compiles")) != 1:
            raise GateFailure(f"{name} must record exactly one Metal library compile")
        if int(_get(artifact, "metal_probe_counters.pipeline_state_compiles")) <= 0:
            raise GateFailure(f"{name} must record prepared PSO compile count")

    for field in (
        "artifact_kind",
        "schema_version",
        "sample.category",
        "sample.backend_name",
        "sample.model_id",
        "sample.prompt_tokens",
        "sample.generated_tokens",
        "sample.toy_backend_enabled",
    ):
        _assert_same(baseline, current, field)

    baseline_sample_id = str(_get(baseline, "sample.sample_id")).strip()
    current_sample_id = str(_get(current, "sample.sample_id")).strip()
    if baseline_sample_id == current_sample_id:
        raise GateFailure(
            "baseline and current sample IDs must be distinct for regression evidence"
        )

    baseline_decode = _metric_measured(baseline, "steady_decode_tokens_per_second")
    current_decode = _metric_measured(current, "steady_decode_tokens_per_second")
    baseline_prefill = _metric_measured(baseline, "prefill_tokens_per_second")
    current_prefill = _metric_measured(current, "prefill_tokens_per_second")
    baseline_command_buffers = _metric_measured(baseline, "command_buffers_per_token")
    current_command_buffers = _metric_measured(current, "command_buffers_per_token")

    baseline_encoders_per_token = float(_get(baseline, "metal_probe_counters.encoders")) / float(
        _get(baseline, "metal_probe_counters.tokens")
    )
    current_encoders_per_token = float(_get(current, "metal_probe_counters.encoders")) / float(
        _get(current, "metal_probe_counters.tokens")
    )
    baseline_forced_waits_per_token = float(
        _get(baseline, "metal_probe_counters.forced_waits")
    ) / float(_get(baseline, "metal_probe_counters.tokens"))
    current_forced_waits_per_token = float(_get(current, "metal_probe_counters.forced_waits")) / float(
        _get(current, "metal_probe_counters.tokens")
    )

    regressions = {
        "steady_decode_tokens_per_second": _percent_regression(
            baseline_decode, current_decode, higher_is_better=True
        ),
        "prefill_tokens_per_second": _percent_regression(
            baseline_prefill, current_prefill, higher_is_better=True
        ),
        "encoders_per_token": _percent_regression(
            baseline_encoders_per_token, current_encoders_per_token, higher_is_better=False
        ),
        "forced_waits_per_token": _percent_regression(
            baseline_forced_waits_per_token,
            current_forced_waits_per_token,
            higher_is_better=False,
        ),
    }
    thresholds = {
        "steady_decode_tokens_per_second": max_decode_regression_pct,
        "prefill_tokens_per_second": max_prefill_regression_pct,
        "encoders_per_token": max_encoder_regression_pct,
        "forced_waits_per_token": max_forced_wait_regression_pct,
    }
    failures = [
        f"{name} regression {value:.2f}% exceeds allowed {thresholds[name]:.2f}%"
        for name, value in regressions.items()
        if value > thresholds[name]
    ]
    if current_command_buffers != baseline_command_buffers:
        failures.append(
            "command_buffers_per_token changed: "
            f"baseline={baseline_command_buffers} current={current_command_buffers}"
        )
    if failures:
        raise GateFailure("; ".join(failures))

    return {
        "gate": "rvllm-apple-metal-e2b-profile-regression",
        "status": "pass",
        "baseline_sample_id": baseline_sample_id,
        "current_sample_id": current_sample_id,
        "thresholds": thresholds,
        "regressions_percent": regressions,
        "baseline": {
            "decode_tok_s": baseline_decode,
            "prefill_tok_s": baseline_prefill,
            "command_buffers_per_token": baseline_command_buffers,
            "encoders_per_token": baseline_encoders_per_token,
            "forced_waits_per_token": baseline_forced_waits_per_token,
        },
        "current": {
            "decode_tok_s": current_decode,
            "prefill_tok_s": current_prefill,
            "command_buffers_per_token": current_command_buffers,
            "encoders_per_token": current_encoders_per_token,
            "forced_waits_per_token": current_forced_waits_per_token,
        },
        "claim_boundary": "local Metal-only probe regression gate; not production performance, ANE, or external profiler evidence",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline profile artifact JSON")
    parser.add_argument("--current", type=Path, required=True, help="current profile artifact JSON")
    parser.add_argument(
        "--max-decode-regression-pct",
        type=float,
        default=10.0,
        help="maximum allowed steady decode tokens/s regression",
    )
    parser.add_argument(
        "--max-prefill-regression-pct",
        type=float,
        default=10.0,
        help="maximum allowed prefill tokens/s regression",
    )
    parser.add_argument(
        "--max-encoder-regression-pct",
        type=float,
        default=0.0,
        help="maximum allowed encoders/token increase",
    )
    parser.add_argument(
        "--max-forced-wait-regression-pct",
        type=float,
        default=0.0,
        help="maximum allowed forced-waits/token increase",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)

    try:
        report = compare_artifacts(
            _load_json(args.baseline),
            _load_json(args.current),
            args.max_decode_regression_pct,
            args.max_prefill_regression_pct,
            args.max_encoder_regression_pct,
            args.max_forced_wait_regression_pct,
        )
    except GateFailure as exc:
        failure = {
            "gate": "rvllm-apple-metal-e2b-profile-regression",
            "status": "fail",
            "reason": str(exc),
        }
        encoded = json.dumps(failure, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded, file=sys.stderr)
        return 1

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
