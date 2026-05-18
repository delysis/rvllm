#!/usr/bin/env python3
"""Self-tests for the rvLLM Apple Metal E2B profile regression gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apple_profile_gate


def metric(value: float) -> dict:
    return {"Measured": {"value": value}}


def unmeasured(reason: str = "not collected by local probe") -> dict:
    return {"Unmeasured": {"reason": reason}}


def artifact(
    sample_id: str,
    *,
    decode_tok_s: float = 2.0,
    prefill_tok_s: float = 4.0,
    command_buffers_per_token: float = 5.0 / 6.0,
    encoders: int = 3000,
    forced_waits: int = 5,
    toy_backend_enabled: bool = False,
) -> dict:
    return {
        "artifact_kind": apple_profile_gate.EXPECTED_ARTIFACT_KIND,
        "schema_version": 1,
        "sample": {
            "sample_id": sample_id,
            "category": "real-e2b-metal-probe",
            "backend_name": "apple-metal",
            "model_id": "google/gemma-4-E2B",
            "prompt_tokens": [2, 4],
            "generated_tokens": [3, 5, 7, 11],
            "toy_backend_enabled": toy_backend_enabled,
            "metrics": {
                "steady_decode_tokens_per_second": metric(decode_tok_s),
                "prefill_tokens_per_second": metric(prefill_tok_s),
                "command_buffers_per_token": metric(command_buffers_per_token),
                "first_token_latency_ms": unmeasured(),
                "memory_peak_bytes": unmeasured(),
                "cpu_utilization_percent": unmeasured(),
                "gpu_utilization_percent": unmeasured(),
                "ane_utilization_percent": unmeasured(),
                "energy_joules": unmeasured(),
            },
        },
        "metal_probe_counters": {
            "prefill_steps": 1,
            "decode_steps": 4,
            "command_buffers": 5,
            "library_compiles": 1,
            "pipeline_state_compiles": 32,
            "tokens": 6,
            "encoders": encoders,
            "forced_waits": forced_waits,
        },
    }


class AppleProfileGateTests(unittest.TestCase):
    def assert_gate_fails(self, baseline: dict, current: dict, expected: str) -> None:
        with self.assertRaises(apple_profile_gate.GateFailure) as ctx:
            apple_profile_gate.compare_artifacts(baseline, current, 10.0, 10.0, 0.0, 0.0)
        self.assertIn(expected, str(ctx.exception))

    def test_accepts_distinct_comparable_samples_within_wall_clock_threshold(self) -> None:
        report = apple_profile_gate.compare_artifacts(
            artifact("baseline"),
            artifact("current", decode_tok_s=1.85, prefill_tok_s=3.7),
            10.0,
            10.0,
            0.0,
            0.0,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["baseline_sample_id"], "baseline")
        self.assertEqual(report["current_sample_id"], "current")

    def test_rejects_wall_clock_decode_regression_even_if_encoder_count_drops(self) -> None:
        self.assert_gate_fails(
            artifact("baseline", encoders=3000),
            artifact("current", decode_tok_s=1.2, encoders=2600),
            "steady_decode_tokens_per_second regression",
        )

    def test_rejects_command_buffer_per_token_change(self) -> None:
        self.assert_gate_fails(
            artifact("baseline"),
            artifact("current", command_buffers_per_token=1.0),
            "command_buffers_per_token changed",
        )

    def test_rejects_reused_sample_id(self) -> None:
        self.assert_gate_fails(artifact("same"), artifact("same"), "sample IDs must be distinct")

    def test_rejects_unmeasured_metric_without_reason(self) -> None:
        current = artifact("current")
        current["sample"]["metrics"]["energy_joules"] = {"Unmeasured": {"reason": ""}}
        self.assert_gate_fails(artifact("baseline"), current, "energy_joules.Unmeasured.reason")

    def test_main_overwrites_output_with_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            current_path = root / "current.json"
            output_path = root / "gate.json"
            baseline_path.write_text(json.dumps(artifact("baseline")), encoding="utf-8")
            current_path.write_text(
                json.dumps(artifact("current", decode_tok_s=1.0)),
                encoding="utf-8",
            )
            output_path.write_text('{"status":"pass"}\n', encoding="utf-8")

            rc = apple_profile_gate.main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--current",
                    str(current_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "fail")
            self.assertIn("steady_decode_tokens_per_second regression", written["reason"])


if __name__ == "__main__":
    unittest.main()
