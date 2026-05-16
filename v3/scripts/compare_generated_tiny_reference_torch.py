#!/usr/bin/env python3
"""Compare the generated tiny Gemma-shaped export with a PyTorch reference.

This is intentionally scoped to the synthetic bundle exported by
`generated_tiny_hf_reference_bundle_can_be_exported`. It does not load a real
Gemma checkpoint and does not use Transformers; raw token IDs and the exported
safetensor weights are enough for this one-layer reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    print("comparison skipped: PyTorch is not installed", file=sys.stderr)
    raise SystemExit(2) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from verify_generated_tiny_reference import (  # noqa: E402
    Tensor,
    VerificationError,
    compare_logits,
    load_safetensors,
    pick_prefix,
    require_tensor,
    verify_supported_config,
)


def tensor_values(tensors: Dict[str, Tensor], name: str, shape: Sequence[int]) -> torch.Tensor:
    tensor = require_tensor(tensors, name, shape)
    return torch.tensor(tensor.data, dtype=torch.float32).reshape(tuple(shape))


def rms_norm(input_values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return input_values * torch.rsqrt(torch.mean(input_values * input_values) + eps) * weight


def gelu_tanh(input_values: torch.Tensor) -> torch.Tensor:
    sqrt_2_over_pi = 0.7978846
    return 0.5 * input_values * (
        1.0
        + torch.tanh(
            sqrt_2_over_pi
            * (input_values + 0.044715 * input_values * input_values * input_values)
        )
    )


def run_torch_decode(bundle_dir: Path) -> Tuple[list[int], list[list[float]], dict]:
    config_path = bundle_dir / "config.json"
    weights_path = bundle_dir / "model.safetensors"
    expected_path = bundle_dir / "expected_reference.json"
    for path in (config_path, weights_path, expected_path):
        if not path.is_file():
            raise VerificationError(f"missing required bundle file {path}")

    config = json.loads(config_path.read_text())
    expected = json.loads(expected_path.read_text())
    _text_config, hidden, intermediate, vocab, eps, softcap = verify_supported_config(config)
    tensors = load_safetensors(weights_path)
    prefix = pick_prefix(tensors)
    layer = f"{prefix}.layers.0"

    embedding = tensor_values(tensors, f"{prefix}.embed_tokens.weight", [vocab, hidden])
    final_norm = tensor_values(tensors, f"{prefix}.norm.weight", [hidden])
    lm_head = tensor_values(tensors, f"{prefix}.lm_head.weight", [vocab, hidden])

    input_norm = tensor_values(tensors, f"{layer}.input_layernorm.weight", [hidden])
    post_attn_norm = tensor_values(tensors, f"{layer}.post_attention_layernorm.weight", [hidden])
    pre_ff_norm = tensor_values(tensors, f"{layer}.pre_feedforward_layernorm.weight", [hidden])
    post_ff_norm = tensor_values(tensors, f"{layer}.post_feedforward_layernorm.weight", [hidden])
    q_norm = tensor_values(tensors, f"{layer}.self_attn.q_norm.weight", [hidden])
    k_norm = tensor_values(tensors, f"{layer}.self_attn.k_norm.weight", [hidden])
    layer_scalar = tensor_values(tensors, f"{layer}.layer_scalar", [hidden])

    q_proj = tensor_values(tensors, f"{layer}.self_attn.q_proj.weight", [hidden, hidden])
    k_proj = tensor_values(tensors, f"{layer}.self_attn.k_proj.weight", [hidden, hidden])
    v_proj = tensor_values(tensors, f"{layer}.self_attn.v_proj.weight", [hidden, hidden])
    o_proj = tensor_values(tensors, f"{layer}.self_attn.o_proj.weight", [hidden, hidden])
    gate_proj = tensor_values(tensors, f"{layer}.mlp.gate_proj.weight", [intermediate, hidden])
    up_proj = tensor_values(tensors, f"{layer}.mlp.up_proj.weight", [intermediate, hidden])
    down_proj = tensor_values(tensors, f"{layer}.mlp.down_proj.weight", [hidden, intermediate])

    prompt = [int(token) for token in expected.get("prompt_tokens", [])]
    steps = int(expected.get("decode_steps", 0))
    if not prompt or steps <= 0:
        raise VerificationError("expected_reference.json must contain prompt_tokens and decode_steps")

    embed_scale = math.sqrt(float(hidden))

    def token_residual(token: int) -> torch.Tensor:
        if token < 0 or token >= vocab:
            raise VerificationError(f"token {token} is outside vocab size {vocab}")
        return embedding[token] * embed_scale

    def project_kv(token: int) -> tuple[torch.Tensor, torch.Tensor]:
        residual = token_residual(token)
        normed = rms_norm(residual, input_norm, eps)
        key = rms_norm(k_proj @ normed, k_norm, eps)
        value = v_proj @ normed
        return key, value

    k_cache = []
    v_cache = []
    for token in prompt:
        key, value = project_kv(token)
        k_cache.append(key)
        v_cache.append(value)

    current = prompt[-1]
    generated: list[int] = []
    logits_by_step: list[list[float]] = []
    for step in range(steps):
        position = len(prompt) - 1 + step
        residual = token_residual(current)
        normed = rms_norm(residual, input_norm, eps)
        query = rms_norm(q_proj @ normed, q_norm, eps)
        key = rms_norm(k_proj @ normed, k_norm, eps)
        value = v_proj @ normed

        if position < len(k_cache):
            k_cache[position] = key
            v_cache[position] = value
        else:
            k_cache.append(key)
            v_cache.append(value)

        keys = torch.stack(k_cache)
        values = torch.stack(v_cache)
        scores = torch.mv(keys, query) / math.sqrt(float(hidden))
        weights = torch.softmax(scores, dim=0)
        attn_out = torch.sum(values * weights[:, None], dim=0)

        residual = rms_norm(residual + (o_proj @ attn_out) * layer_scalar, post_attn_norm, eps)
        mlp_normed = rms_norm(residual, pre_ff_norm, eps)
        activated = gelu_tanh(gate_proj @ mlp_normed) * (up_proj @ mlp_normed)
        residual = rms_norm(residual + (down_proj @ activated) * layer_scalar, post_ff_norm, eps)

        final_hidden = rms_norm(residual, final_norm, eps)
        logits = lm_head @ final_hidden
        if softcap > 0.0:
            logits = softcap * torch.tanh(logits / softcap)
        next_token = int(torch.argmax(logits).item())
        logits_by_step.append([float(value) for value in logits.tolist()])
        generated.append(next_token)
        current = next_token

    return generated, logits_by_step, expected


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated tiny Gemma-shaped export with a PyTorch reference."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="directory containing config.json, model.safetensors, expected_reference.json",
    )
    parser.add_argument("--atol", type=float, default=2e-3, help="absolute logit tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="relative logit tolerance")
    args = parser.parse_args(argv)

    try:
        with torch.no_grad():
            generated, logits_by_step, expected = run_torch_decode(args.bundle_dir)
        expected_tokens = [int(token) for token in expected.get("generated_tokens", [])]
        if generated != expected_tokens:
            raise VerificationError(f"generated tokens {generated}, expected {expected_tokens}")
        max_diff, max_label = compare_logits(
            logits_by_step,
            expected.get("logits_by_step", []),
            args.atol,
            args.rtol,
        )
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        print(f"torch comparison failed: {exc}", file=sys.stderr)
        return 1

    print("torch comparison passed")
    print(f"bundle: {args.bundle_dir}")
    print(f"generated_tokens: {generated}")
    print(f"logit_steps: {len(logits_by_step)}")
    print(f"max_logit_abs_diff: {max_diff:.8g} ({max_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
