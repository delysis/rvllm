#!/usr/bin/env python3
"""Generate a sealed, serial all-role native-BF16 low-bit queue campaign."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3")
MODEL = Path("/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
OUT = ROOT / "reports/gemma4-metal-low-bit-bf16-seven-role-20260925"
QUEUE = ROOT / "reports/gemma4-global-decode-local-20260924/queue"
EXE = ROOT / "target/release/rvllm-low-bit-real-weight"
VALIDATOR = ROOT / "tools/validate_low_bit_real_weight_receipt.py"
PYTHON = Path("/usr/bin/python3")
ANE_TAIL = "gemma4-ane-stacked-baseline-exact-v2-12-a6-stacked-20260924"
ROLES = [
    ("q", "self_attn.q_proj", "query_projection"),
    ("k", "self_attn.k_proj", "key_projection"),
    ("v", "self_attn.v_proj", "value_projection"),
    ("o", "self_attn.o_proj", "output_projection"),
    ("gate", "mlp.gate_proj", "dense_gate_projection"),
    ("up", "mlp.up_proj", "dense_up_projection"),
    ("down", "mlp.down_proj", "dense_down_projection"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path)}


def invocation(executable: Path, args: list[str]) -> dict:
    return {
        "executable": pin(executable),
        "args": args,
        "cwd": str(ROOT),
        "env": {},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs_dir = OUT / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    common_inputs = [pin(ROOT / "Cargo.lock"), pin(MODEL / "config.json"), pin(VALIDATOR)]
    prior = ANE_TAIL
    jobs = []
    for short, stem, role in ROLES:
        tensor = f"model.language_model.layers.0.{stem}.weight"
        for phase, samples, purpose in (("screen", 3, "correctness"), ("confirm", 9, "exploratory_timing")):
            job_id = f"g4metalbf16lowbit-{short}-m1m4-{phase}-20260925"
            receipt = "{output}/trial.stdout"
            job = {
                "schema": "rvllm.experiment_job.v1",
                "id": job_id,
                "purpose": purpose,
                "command": invocation(EXE, ["--model-dir", str(MODEL), "--tensor", tensor, "--m", "1,4", "--samples", str(samples)]),
                "validator": invocation(PYTHON, [str(VALIDATOR), "--receipt", receipt, "--tensor", tensor, "--role", role, "--samples", str(samples)]),
                "inputs": common_inputs,
                "kernel_game_submission": None,
                "after": [prior],
                "conditions": {
                    "power_source": "ac",
                    "low_power_mode": None,
                    "pmset_power_mode": None,
                    "thermal_state": None,
                    "minimum_free_bytes": 17179869184,
                    "disk_path": str(ROOT),
                    "quiet_process_names": [],
                    "observe_process_names": ["cargo", "rustc", "metal", "metallib", "llama-server", "ollama", "rvllm_disaggregated_infer"],
                    "idle_llama_servers": [],
                },
                "stable_seconds": 0,
                "max_wait_seconds": 86400,
                "max_run_seconds": 3600,
            }
            path = jobs_dir / f"{job_id}.json"
            path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
            jobs.append({"id": job_id, "path": str(path), "phase": phase, "samples": samples, "tensor": tensor, "role": role, "after": [prior]})
            prior = job_id
    campaign = {
        "schema": "rvllm.metal_low_bit_bf16_campaign.v1",
        "claim": "real-checkpoint projection-operator experiment only; not checkpoint-quality, full-route, ANE, or promotion evidence",
        "queue": str(QUEUE),
        "model": {"path": str(MODEL), "config": pin(MODEL / "config.json")},
        "binary": pin(EXE),
        "validator": pin(VALIDATOR),
        "matrix": {"roles": [r[2] for r in ROLES], "formats": ["W4ABF16", "W8ABF16"], "m": [1, 4], "screen_abba_blocks": 3, "confirmation_abba_blocks": 9},
        "policy": {"correctness_first": True, "serial": True, "stable_seconds": 0, "condition_changes": "recorded, never used to discard exploratory observations", "first_dependency": ANE_TAIL},
        "jobs": jobs,
    }
    (OUT / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"campaign": str(OUT / "campaign.json"), "jobs": len(jobs), "first_after": ANE_TAIL, "last": prior}))


if __name__ == "__main__":
    main()
