#!/usr/bin/env python3
"""Stage default-off cooperative W4/W8 Metal projection jobs; never enqueue them.

The generated manifests are target-host specific because every executable,
Cargo.lock, model config, and research source is SHA-256 pinned at staging time.
Correctness screens are dependencies of timing confirmations, so the queue
cannot advance a failed candidate merely because it was ordered earlier.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(
    os.environ.get(
        "RVLLM_GEMMA4_MODEL_DIR",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/"
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
        ),
    )
)
OUT = ROOT / "reports/gemma4-metal-low-bit-coop-20260925"
EXE = ROOT / "target/release/rvllm-low-bit-real-weight"
VALIDATOR = ROOT / "target/release/rvllm-low-bit-receipt-validate"
RESEARCH_SOURCE = ROOT / "crates/rvllm-apple-metal/src/research_shaders/low_bit_coop_bf16.metal"
HOST_SOURCE = ROOT / "crates/rvllm-apple-metal/src/low_bit_metal.rs"
ROLES = (
    ("q", "self_attn.q_proj", "query_projection"),
    ("k", "self_attn.k_proj", "key_projection"),
    ("v", "self_attn.v_proj", "value_projection"),
    ("o", "self_attn.o_proj", "output_projection"),
    ("gate", "mlp.gate_proj", "dense_gate_projection"),
    ("up", "mlp.up_proj", "dense_up_projection"),
    ("down", "mlp.down_proj", "dense_down_projection"),
)
CANDIDATES = ("coop16", "coop32")


def pin(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing required staged input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def invocation(executable: Path, args: list[str]) -> dict:
    return {
        "executable": pin(executable),
        "args": args,
        "cwd": str(ROOT.resolve()),
        "env": {},
    }


def conditions() -> dict:
    return {
        "power_source": "ac",
        "low_power_mode": None,
        "pmset_power_mode": None,
        "thermal_state": None,
        "minimum_free_bytes": 17179869184,
        "disk_path": str(ROOT.resolve()),
        "quiet_process_names": [],
        "observe_process_names": [
            "cargo",
            "rustc",
            "metal",
            "metallib",
            "llama-server",
            "ollama",
            "rvllm_disaggregated_infer",
        ],
        "idle_llama_servers": [],
    }


def main() -> None:
    jobs_dir = OUT / "manifests"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    model_config = MODEL / "config.json"
    shared_inputs = [
        pin(ROOT / "Cargo.lock"),
        pin(model_config),
        pin(RESEARCH_SOURCE),
        pin(HOST_SOURCE),
        pin(VALIDATOR),
    ]
    executable_pin = pin(EXE)
    validator_pin = pin(VALIDATOR)

    jobs: list[dict] = []
    for candidate in CANDIDATES:
        for short, stem, role in ROLES:
            tensor = f"model.language_model.layers.0.{stem}.weight"
            screen_id = (
                f"g4metal-lowbit-{candidate}-{short}-m1m4-screen-20260925"
            )
            confirm_id = (
                f"g4metal-lowbit-{candidate}-{short}-m1m4-confirm-20260925"
            )
            for phase, job_id, samples, purpose, after in (
                ("screen", screen_id, 3, "correctness", []),
                ("confirm", confirm_id, 9, "exploratory_timing", [screen_id]),
            ):
                args = [
                    "--model-dir",
                    str(MODEL.resolve()),
                    "--tensor",
                    tensor,
                    "--m",
                    "1,4",
                    "--format",
                    "both",
                    "--samples",
                    str(samples),
                    "--candidate",
                    candidate,
                ]
                validator_args = [
                    "--receipt",
                    "{output}/trial.stdout",
                    "--tensor",
                    tensor,
                    "--role",
                    role,
                    "--samples",
                    str(samples),
                    "--candidate",
                    candidate,
                ]
                job = {
                    "schema": "rvllm.experiment_job.v1",
                    "id": job_id,
                    "purpose": purpose,
                    "command": {
                        "executable": executable_pin,
                        "args": args,
                        "cwd": str(ROOT.resolve()),
                        "env": {},
                    },
                    "validator": {
                        "executable": validator_pin,
                        "args": validator_args,
                        "cwd": str(ROOT.resolve()),
                        "env": {},
                    },
                    "inputs": shared_inputs,
                    "kernel_game_submission": None,
                    "after": after,
                    "conditions": conditions(),
                    "stable_seconds": 0,
                    "max_wait_seconds": 86400,
                    "max_run_seconds": 3600,
                }
                out = jobs_dir / f"{job_id}.json"
                out.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
                jobs.append(
                    {
                        "id": job_id,
                        "path": str(out.resolve()),
                        "phase": phase,
                        "candidate": candidate,
                        "role": role,
                    }
                )

    campaign = {
        "schema": "rvllm.metal_low_bit_coop_campaign.v1",
        "claim": (
            "default-off projection-operator qualification only; no selector, "
            "full-route, checkpoint-quality, or promotion claim"
        ),
        "candidate_schedules": {
            "coop16": {
                "threads_per_threadgroup": 128,
                "outputs_per_threadgroup": 16,
                "w4_k_tile": 64,
                "w8_k_tile": 128,
            },
            "coop32": {
                "threads_per_threadgroup": 128,
                "outputs_per_threadgroup": 32,
                "w4_k_tile": 64,
                "w8_k_tile": 128,
            },
        },
        "matrix": {
            "roles": [role for _, _, role in ROLES],
            "formats": ["W4ABF16", "W8ABF16"],
            "m": [1, 4],
        },
        "policy": {
            "correctness_first": True,
            "timing_requires_its_candidate_screen": True,
            "queue_failure_blocks_dependent_timing": True,
            "defaults_unchanged": True,
        },
        "binary": executable_pin,
        "validator": validator_pin,
        "research_source": pin(RESEARCH_SOURCE),
        "host_source": pin(HOST_SOURCE),
        "jobs": jobs,
    }
    campaign_path = OUT / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "campaign": str(campaign_path.resolve()),
                "jobs": len(jobs),
                "note": "staged only; submit manifests explicitly after review",
            }
        )
    )


if __name__ == "__main__":
    main()
