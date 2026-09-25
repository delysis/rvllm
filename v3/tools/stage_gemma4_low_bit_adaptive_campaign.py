#!/usr/bin/env python3
"""Seal focused reverse-order jobs for the default-off low-bit selector."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3")
MODEL = Path("/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
OUT = ROOT / "reports/gemma4-metal-low-bit-adaptive-20260925"
EXE = ROOT / "target/release/rvllm-low-bit-real-weight"
VALIDATOR = ROOT / "target/release/rvllm-low-bit-receipt-validate"
CELLS = (
    ("k-w8-m1", "self_attn.k_proj", "key_projection", "w8a16"),
    ("v-w8-m1", "self_attn.v_proj", "value_projection", "w8a16"),
    ("o-w8-m1", "self_attn.o_proj", "output_projection", "w8a16"),
    ("down-w8-m1", "mlp.down_proj", "dense_down_projection", "w8a16"),
    ("down-w4-m1", "mlp.down_proj", "dense_down_projection", "w4a16"),
)


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def invoke(executable: Path, args: list[str]) -> dict:
    return {"executable": pin(executable), "args": args, "cwd": str(ROOT), "env": {}}


def main() -> None:
    manifests = OUT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    inputs = [pin(ROOT / "Cargo.lock"), pin(MODEL / "config.json"), pin(VALIDATOR)]
    prior = None
    jobs = []
    for short, stem, role, weight_format in CELLS:
        tensor = f"model.language_model.layers.0.{stem}.weight"
        for order in ("abba", "baab"):
            job_id = f"g4metal-lowbit-adaptive-{short}-{order}-20260925"
            common = ["--tensor", tensor, "--role", role, "--samples", "9", "--candidate", "adaptive", "--format", weight_format, "--m", "1", "--order", order.upper()]
            command = ["--model-dir", str(MODEL), "--tensor", tensor, "--m", "1", "--format", weight_format, "--samples", "9", "--candidate", "adaptive", "--order", order]
            job = {
                "schema": "rvllm.experiment_job.v1", "id": job_id, "purpose": "exploratory_timing",
                "command": invoke(EXE, command),
                "validator": invoke(VALIDATOR, ["--receipt", "{output}/trial.stdout", *common]),
                "inputs": inputs, "kernel_game_submission": None, "after": [prior] if prior else [],
                "conditions": {"power_source": "ac", "low_power_mode": None, "pmset_power_mode": None,
                    "thermal_state": None, "minimum_free_bytes": 17179869184, "disk_path": str(ROOT),
                    "quiet_process_names": [], "observe_process_names": ["cargo", "rustc", "metal", "metallib", "llama-server", "ollama", "rvllm_disaggregated_infer"], "idle_llama_servers": []},
                "stable_seconds": 0, "max_wait_seconds": 86400, "max_run_seconds": 3600,
            }
            path = manifests / f"{job_id}.json"
            path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
            jobs.append({"id": job_id, "path": str(path), "cell": short, "order": order.upper()})
            prior = job_id
    campaign = {"schema": "rvllm.metal_low_bit_adaptive_campaign.v1",
        "claim": "focused operator evidence only; default-off selector; not a shipping or full-route claim",
        "selector": {"w8_m1_k_v_o_down": "n4", "w4_m1_down": "n4",
            "established_w4_o_m1_m4": "n4", "established_w8_k_m4": "n4",
            "established_w8_up_m1": "n8", "otherwise": "vector"},
        "samples_per_abba_block": 2, "blocks": 9, "orders": ["ABBA", "BAAB"],
        "binary": pin(EXE), "validator": pin(VALIDATOR), "jobs": jobs}
    (OUT / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"campaign": str(OUT / "campaign.json"), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
