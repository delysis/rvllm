#!/usr/bin/env python3
"""Queue-owned MLX timing for the exact token IDs in one rvLLM case report.

This compares workloads, not checkpoint file identity or numerical quality.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rvllm-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--generation-tokens", type=int, default=2)
    args = parser.parse_args()
    if args.trials < 1 or args.generation_tokens < 2:
        parser.error("trials must be positive and generation-tokens at least two")

    source = json.loads(args.rvllm_report.read_text(encoding="utf-8"))
    if source.get("status") != "pass" or len(source.get("cases", [])) != 1:
        parser.error("rvLLM input must be one passing case")
    prompt = source["cases"][0]["prompt_token_ids"]
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        parser.error("rvLLM prompt token IDs are missing or invalid")
    if args.report.exists():
        parser.error("output report exists; never overwrite a prior receipt")

    import mlx.core as mx
    from mlx_lm import load, stream_generate

    mx.random.seed(0)
    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        trust_remote_code=False,
    )
    # Match upstream mlx_lm.benchmark: avoid an early EOS truncating the trial.
    tokenizer._eos_token_ids = {}
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    if any(token >= vocab for token in prompt):
        parser.error("rvLLM prompt contains an ID outside MLX vocabulary")

    def trial() -> dict:
        started = time.perf_counter_ns()
        responses = list(
            stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.generation_tokens,
            )
        )
        if len(responses) != args.generation_tokens:
            raise RuntimeError("MLX generated fewer tokens than requested")
        last = responses[-1]
        if last.prompt_tokens != len(prompt):
            raise RuntimeError("MLX did not consume the specified prompt length")
        return {
            "wall_ns": time.perf_counter_ns() - started,
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": last.prompt_tps,
            "inferred_prefill_ns": round(1e9 * len(prompt) / last.prompt_tps),
            "generation_tps": last.generation_tps,
            "generated_token_ids": [int(item.token) for item in responses],
            "peak_memory_gb": last.peak_memory,
        }

    trial()  # Load, compile and warm the route outside the measured trials.
    samples = [trial() for _ in range(args.trials)]
    report = {
        "schema": "rvllm.mlx_exact_prompt_bench.v1",
        "claim": "matched prompt token IDs and length only; checkpoints and timing boundaries differ",
        "model_dir": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "rvllm_report": str(args.rvllm_report.resolve()),
        "rvllm_report_sha256": sha256(args.rvllm_report),
        "mlx_generate_source_sha256": sha256(Path(stream_generate.__code__.co_filename)),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_token_count": len(prompt),
        "generation_tokens": args.generation_tokens,
        "warmups": 1,
        "samples": samples,
    }
    with args.report.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps({"report": str(args.report), "samples": samples}))


if __name__ == "__main__":
    main()
