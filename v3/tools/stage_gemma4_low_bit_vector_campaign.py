#!/usr/bin/env python3
"""Seal default-off W4 packed2 / W8 K4 Metal screening jobs (do not enqueue)."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3")
MODEL = Path("/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
OUT = ROOT / "reports/gemma4-metal-low-bit-vector-20260925"
EXE = ROOT / "target/release/rvllm-low-bit-real-weight"
VALIDATOR = ROOT / "target/release/rvllm-low-bit-receipt-validate"
ROLES = (
    ("q", "self_attn.q_proj", "query_projection"),
    ("k", "self_attn.k_proj", "key_projection"),
    ("v", "self_attn.v_proj", "value_projection"),
    ("o", "self_attn.o_proj", "output_projection"),
    ("gate", "mlp.gate_proj", "dense_gate_projection"),
    ("up", "mlp.up_proj", "dense_up_projection"),
    ("down", "mlp.down_proj", "dense_down_projection"),
)


def pin(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def invocation(executable: Path, args: list[str]) -> dict:
    return {"executable": pin(executable), "args": args, "cwd": str(ROOT), "env": {}}


def main() -> None:
    jobs_dir = OUT / "manifests"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    inputs = [pin(ROOT / "Cargo.lock"), pin(MODEL / "config.json"), pin(VALIDATOR)]
    prior: str | None = None
    jobs = []
    for short, stem, role in ROLES:
        tensor = f"model.language_model.layers.0.{stem}.weight"
        for phase, samples, purpose in (("screen", 3, "correctness"), ("confirm", 9, "exploratory_timing")):
            job_id = f"g4metal-lowbit-vector-{short}-m1m4-{phase}-20260925"
            args = ["--model-dir", str(MODEL), "--tensor", tensor, "--m", "1,4", "--format", "both", "--samples", str(samples), "--candidate", "vector"]
            validator_args = ["--receipt", "{output}/trial.stdout", "--tensor", tensor, "--role", role, "--samples", str(samples), "--candidate", "vector"]
            job = {
                "schema": "rvllm.experiment_job.v1", "id": job_id, "purpose": purpose,
                "command": invocation(EXE, args), "validator": invocation(VALIDATOR, validator_args),
                "inputs": inputs, "kernel_game_submission": None, "after": [prior] if prior else [],
                "conditions": {"power_source": "ac", "low_power_mode": None, "pmset_power_mode": None,
                    "thermal_state": None, "minimum_free_bytes": 17179869184, "disk_path": str(ROOT),
                    "quiet_process_names": [], "observe_process_names": ["cargo", "rustc", "metal", "metallib", "llama-server", "ollama", "rvllm_disaggregated_infer"], "idle_llama_servers": []},
                "stable_seconds": 0, "max_wait_seconds": 86400, "max_run_seconds": 3600,
            }
            path = jobs_dir / f"{job_id}.json"
            path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
            jobs.append({"id": job_id, "path": str(path), "phase": phase, "role": role})
            prior = job_id
    campaign = {
        "schema": "rvllm.metal_low_bit_vector_campaign.v1",
        "claim": "default-off projection-operator experiment; not promotion or full-route evidence",
        "candidate": {"w4a16": "n4_packed2", "w8a16": "n8_k4"},
        "matrix": {"roles": [role for _, _, role in ROLES], "formats": ["W4ABF16", "W8ABF16"], "m": [1, 4]},
        "policy": {"correctness_first": True, "serial": True, "stable_seconds": 0, "condition_changes": "record only"},
        "binary": pin(EXE), "validator": pin(VALIDATOR), "jobs": jobs,
    }
    (OUT / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"campaign": str(OUT / "campaign.json"), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
