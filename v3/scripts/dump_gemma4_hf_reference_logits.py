#!/usr/bin/env python3
"""Dump Hugging Face/Transformers reference logits for a real Gemma checkpoint.

This is a reference hook, not an rvLLM execution path. It requires a local model
directory plus PyTorch and Transformers in the Python environment. The output is
JSON that can be compared against rvLLM logits in a separate step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence


def parse_token_ids(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            token_id = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid token id {part!r}") from exc
        if token_id < 0:
            raise argparse.ArgumentTypeError(f"token id must be non-negative: {token_id}")
        out.append(token_id)
    if not out:
        raise argparse.ArgumentTypeError("at least one token id is required")
    return out


def parse_optional_token_ids(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return parse_token_ids(raw)


def finite_float(value) -> float:
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite logit value {out}")
    return out


def selected_logits(logits, token_ids: Iterable[int]) -> list[dict]:
    vocab = int(logits.numel())
    values = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab:
            raise ValueError(f"selected token id {token_id} outside vocab size {vocab}")
        values.append({"token_id": int(token_id), "logit": finite_float(logits[token_id].item())})
    return values


def top_logits(logits, k: int) -> list[dict]:
    if k <= 0:
        return []
    k = min(int(k), int(logits.numel()))
    values, indices = logits.topk(k)
    return [
        {"token_id": int(token_id), "logit": finite_float(value)}
        for token_id, value in zip(indices.tolist(), values.tolist())
    ]


def load_prompt_ids(args, tokenizer) -> list[int]:
    if args.prompt_token_ids is not None:
        return args.prompt_token_ids
    encoded = tokenizer(args.prompt_text, return_tensors="pt", add_special_tokens=not args.no_special_tokens)
    return [int(token_id) for token_id in encoded.input_ids[0].tolist()]


def run(args) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required for this reference hook") from exc

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_dir}")

    tokenizer = None
    if args.prompt_text is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=args.trust_remote_code,
        )

    load_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    if args.device_map is None:
        model.to(args.device)
    model.eval()

    prompt_ids = load_prompt_ids(args, tokenizer)
    selected_ids = parse_optional_token_ids(args.selected_token_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    steps = []
    generated_tokens = []
    past_key_values = None
    next_input_ids = input_ids
    with torch.no_grad():
        for step in range(args.decode_steps):
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :].detach().float().cpu()
            next_token = int(torch.argmax(logits).item())
            step_payload = {
                "step": step,
                "next_token": next_token,
                "top_logits": top_logits(logits, args.top_k),
                "selected_logits": selected_logits(logits, selected_ids),
            }
            if args.full_logits:
                step_payload["logits"] = [finite_float(value) for value in logits.tolist()]
            steps.append(step_payload)
            generated_tokens.append(next_token)
            past_key_values = outputs.past_key_values
            next_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return {
        "schema": "rvllm.gemma4_hf_reference_logits.v1",
        "model_dir": str(model_dir),
        "prompt_token_ids": prompt_ids,
        "decode_steps": args.decode_steps,
        "generated_tokens": generated_tokens,
        "device": str(device),
        "full_logits": bool(args.full_logits),
        "top_k": int(args.top_k),
        "selected_token_ids": selected_ids,
        "steps": steps,
        "claim": "HF/Transformers reference artifact only; no rvLLM, Metal, ANE, or production claim.",
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Dump HF/Transformers reference logits for a local Gemma checkpoint."
    )
    parser.add_argument("model_dir", type=Path, help="local Hugging Face model directory")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-token-ids", type=parse_token_ids, help="comma-separated token ids")
    prompt.add_argument("--prompt-text", help="prompt text to tokenize with the model tokenizer")
    parser.add_argument("--decode-steps", type=int, default=1, help="number of greedy decode steps")
    parser.add_argument("--selected-token-ids", help="comma-separated token ids to always record")
    parser.add_argument("--top-k", type=int, default=10, help="top logits to record per step")
    parser.add_argument("--full-logits", action="store_true", help="write the full vocab logit vector")
    parser.add_argument("--device", default="cpu", help="torch device when --device-map is not set")
    parser.add_argument("--device-map", help="optional Transformers device_map, e.g. auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true", help="do not add tokenizer special tokens")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path")
    args = parser.parse_args(argv)
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")

    try:
        payload = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001 - command-line tool should report any failure cleanly.
        print(f"HF reference dump failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote HF reference logits to {args.output}")
    print(f"generated_tokens: {payload['generated_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
