#!/usr/bin/env python3
"""Verify the generated tiny HF/Gemma-shaped reference bundle.

This intentionally bounded verifier reads only the synthetic bundle exported by
`generated_tiny_hf_reference_bundle_can_be_exported`: config.json,
model.safetensors, and expected_reference.json. It has no PyTorch dependency and
parses safetensors directly.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


class VerificationError(Exception):
    pass


class Tensor:
    def __init__(self, name: str, dtype: str, shape: Sequence[int], data: List[float]):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape)
        self.data = data


def half_to_float(bits: int) -> float:
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x03FF
    if exp == 0:
        if frac == 0:
            return -0.0 if sign < 0 else 0.0
        return sign * math.ldexp(frac / 1024.0, -14)
    if exp == 0x1F:
        if frac == 0:
            return sign * math.inf
        return math.nan
    return sign * math.ldexp(1.0 + frac / 1024.0, exp - 15)


def decode_tensor_data(dtype: str, raw: bytes) -> List[float]:
    if dtype == "F16":
        if len(raw) % 2 != 0:
            raise VerificationError("F16 tensor byte length is not divisible by 2")
        return [
            half_to_float(struct.unpack_from("<H", raw, offset)[0])
            for offset in range(0, len(raw), 2)
        ]
    if dtype == "F32":
        if len(raw) % 4 != 0:
            raise VerificationError("F32 tensor byte length is not divisible by 4")
        return [
            struct.unpack_from("<f", raw, offset)[0]
            for offset in range(0, len(raw), 4)
        ]
    raise VerificationError(f"unsupported safetensors dtype {dtype!r}; expected F16 or F32")


def product(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def load_safetensors(path: Path) -> Dict[str, Tensor]:
    blob = path.read_bytes()
    if len(blob) < 8:
        raise VerificationError(f"{path} is too short to be a safetensors file")
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header_start = 8
    header_end = header_start + header_len
    if header_end > len(blob):
        raise VerificationError("safetensors header length exceeds file length")
    try:
        header = json.loads(blob[header_start:header_end].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid safetensors header JSON: {exc}") from exc

    tensors: Dict[str, Tensor] = {}
    payload = blob[header_end:]
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise VerificationError(f"tensor {name!r} has invalid metadata")
        if len(offsets) != 2:
            raise VerificationError(f"tensor {name!r} has invalid data_offsets")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start or end > len(payload):
            raise VerificationError(f"tensor {name!r} offsets are outside the payload")
        data = decode_tensor_data(dtype, payload[start:end])
        expected_len = product(int(dim) for dim in shape)
        if len(data) != expected_len:
            raise VerificationError(
                f"tensor {name!r} has {len(data)} values, expected {expected_len} from shape {shape}"
            )
        tensors[name] = Tensor(name, dtype, [int(dim) for dim in shape], data)
    return tensors


def require_tensor(
    tensors: Dict[str, Tensor], name: str, shape: Sequence[int] | None = None
) -> Tensor:
    try:
        tensor = tensors[name]
    except KeyError as exc:
        raise VerificationError(f"missing tensor {name!r}") from exc
    if shape is not None and tensor.shape != list(shape):
        raise VerificationError(f"tensor {name!r} shape {tensor.shape}, expected {list(shape)}")
    return tensor


def rms_norm(input_values: Sequence[float], weight: Sequence[float], eps: float) -> List[float]:
    if len(input_values) != len(weight):
        raise VerificationError("rms_norm input and weight lengths differ")
    mean_square = sum(value * value for value in input_values) / len(input_values)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [value * scale * w for value, w in zip(input_values, weight)]


def matvec(weight: Sequence[float], rows: int, cols: int, input_values: Sequence[float]) -> List[float]:
    if len(weight) != rows * cols:
        raise VerificationError(f"matvec weight length {len(weight)}, expected {rows * cols}")
    if len(input_values) != cols:
        raise VerificationError(f"matvec input length {len(input_values)}, expected {cols}")
    out = []
    for row in range(rows):
        base = row * cols
        total = 0.0
        for col in range(cols):
            total += weight[base + col] * input_values[col]
        out.append(total)
    return out


def gelu_tanh(x: float) -> float:
    sqrt_2_over_pi = 0.7978846
    return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))


def argmax(values: Sequence[float]) -> int:
    best_idx = 0
    best_value = -math.inf
    for idx, value in enumerate(values):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def pick_prefix(tensors: Dict[str, Tensor]) -> str:
    for prefix in ("model.language_model", "model"):
        if f"{prefix}.embed_tokens.weight" in tensors:
            return prefix
    raise VerificationError("could not find model.language_model or model tensor prefix")


def verify_supported_config(config: dict) -> Tuple[dict, int, int, int, float, float]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise VerificationError("config.json must contain text_config")
    if config.get("architectures") != ["Gemma4ForConditionalGeneration"]:
        raise VerificationError("this verifier is bounded to Gemma4ForConditionalGeneration exports")
    if int(text_config.get("num_hidden_layers", 0)) != 1:
        raise VerificationError("this verifier supports only the generated one-layer tiny fixture")

    hidden = int(text_config.get("hidden_size", 0))
    intermediate = int(text_config.get("intermediate_size", 0))
    vocab = int(text_config.get("vocab_size", 0))
    head_dim = int(text_config.get("head_dim", 0))
    num_heads = int(text_config.get("num_attention_heads", 0))
    num_kv_heads = int(text_config.get("num_key_value_heads", 0))
    eps = float(text_config.get("rms_norm_eps", 0.0))
    softcap = float(text_config.get("final_logit_softcapping", 0.0))

    if hidden <= 0 or intermediate <= 0 or vocab <= 0:
        raise VerificationError("hidden_size, intermediate_size, and vocab_size must be positive")
    if head_dim != hidden or num_heads != 1 or num_kv_heads != 1:
        raise VerificationError("this verifier supports only the exported one-head hidden=head_dim fixture")
    if text_config.get("tie_word_embeddings") is not False:
        raise VerificationError("this verifier expects untied lm_head weights")
    return text_config, hidden, intermediate, vocab, eps, softcap


def run_decode(bundle_dir: Path) -> Tuple[List[int], List[List[float]], dict]:
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

    embedding = require_tensor(tensors, f"{prefix}.embed_tokens.weight", [vocab, hidden]).data
    final_norm = require_tensor(tensors, f"{prefix}.norm.weight", [hidden]).data
    lm_head = require_tensor(tensors, f"{prefix}.lm_head.weight", [vocab, hidden]).data

    input_norm = require_tensor(tensors, f"{layer}.input_layernorm.weight", [hidden]).data
    post_attn_norm = require_tensor(tensors, f"{layer}.post_attention_layernorm.weight", [hidden]).data
    pre_ff_norm = require_tensor(tensors, f"{layer}.pre_feedforward_layernorm.weight", [hidden]).data
    post_ff_norm = require_tensor(tensors, f"{layer}.post_feedforward_layernorm.weight", [hidden]).data
    q_norm = require_tensor(tensors, f"{layer}.self_attn.q_norm.weight", [hidden]).data
    k_norm = require_tensor(tensors, f"{layer}.self_attn.k_norm.weight", [hidden]).data
    layer_scalar = require_tensor(tensors, f"{layer}.layer_scalar", [hidden]).data

    q_proj = require_tensor(tensors, f"{layer}.self_attn.q_proj.weight", [hidden, hidden]).data
    k_proj = require_tensor(tensors, f"{layer}.self_attn.k_proj.weight", [hidden, hidden]).data
    v_proj = require_tensor(tensors, f"{layer}.self_attn.v_proj.weight", [hidden, hidden]).data
    o_proj = require_tensor(tensors, f"{layer}.self_attn.o_proj.weight", [hidden, hidden]).data
    gate_proj = require_tensor(tensors, f"{layer}.mlp.gate_proj.weight", [intermediate, hidden]).data
    up_proj = require_tensor(tensors, f"{layer}.mlp.up_proj.weight", [intermediate, hidden]).data
    down_proj = require_tensor(tensors, f"{layer}.mlp.down_proj.weight", [hidden, intermediate]).data

    prompt = [int(token) for token in expected.get("prompt_tokens", [])]
    steps = int(expected.get("decode_steps", 0))
    if not prompt or steps <= 0:
        raise VerificationError("expected_reference.json must contain prompt_tokens and decode_steps")

    scale = math.sqrt(float(hidden))

    def token_residual(token: int) -> List[float]:
        if token < 0 or token >= vocab:
            raise VerificationError(f"token {token} is outside vocab size {vocab}")
        base = token * hidden
        return [embedding[base + dim] * scale for dim in range(hidden)]

    def project_kv(token: int) -> Tuple[List[float], List[float]]:
        residual = token_residual(token)
        normed = rms_norm(residual, input_norm, eps)
        key = matvec(k_proj, hidden, hidden, normed)
        value = matvec(v_proj, hidden, hidden, normed)
        return rms_norm(key, k_norm, eps), value

    k_cache = []
    v_cache = []
    for token in prompt:
        key, value = project_kv(token)
        k_cache.append(key)
        v_cache.append(value)

    current = prompt[-1]
    generated = []
    logits_by_step = []
    for step in range(steps):
        position = len(prompt) - 1 + step
        residual = token_residual(current)
        normed = rms_norm(residual, input_norm, eps)
        query = rms_norm(matvec(q_proj, hidden, hidden, normed), q_norm, eps)
        key = rms_norm(matvec(k_proj, hidden, hidden, normed), k_norm, eps)
        value = matvec(v_proj, hidden, hidden, normed)

        if position < len(k_cache):
            k_cache[position] = key
            v_cache[position] = value
        else:
            k_cache.append(key)
            v_cache.append(value)

        scores = [
            sum(q * k for q, k in zip(query, cached_key)) / math.sqrt(float(hidden))
            for cached_key in k_cache
        ]
        max_score = max(scores)
        denom = sum(math.exp(score - max_score) for score in scores)
        attn_out = [0.0] * hidden
        for cached_value, score in zip(v_cache, scores):
            weight = math.exp(score - max_score) / denom
            for dim in range(hidden):
                attn_out[dim] += cached_value[dim] * weight

        projected = matvec(o_proj, hidden, hidden, attn_out)
        for dim in range(hidden):
            residual[dim] += projected[dim] * layer_scalar[dim]
        residual = rms_norm(residual, post_attn_norm, eps)

        mlp_normed = rms_norm(residual, pre_ff_norm, eps)
        gate = matvec(gate_proj, intermediate, hidden, mlp_normed)
        up = matvec(up_proj, intermediate, hidden, mlp_normed)
        activated = [gelu_tanh(g) * u for g, u in zip(gate, up)]
        mlp_out = matvec(down_proj, hidden, intermediate, activated)
        for dim in range(hidden):
            residual[dim] += mlp_out[dim] * layer_scalar[dim]
        residual = rms_norm(residual, post_ff_norm, eps)

        final_hidden = rms_norm(residual, final_norm, eps)
        logits = matvec(lm_head, vocab, hidden, final_hidden)
        if softcap > 0.0:
            logits = [softcap * math.tanh(logit / softcap) for logit in logits]
        next_token = argmax(logits)
        logits_by_step.append(logits)
        generated.append(next_token)
        current = next_token

    return generated, logits_by_step, expected


def compare_logits(
    got: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    atol: float,
    rtol: float,
) -> Tuple[float, str]:
    if len(got) != len(expected):
        raise VerificationError(f"got {len(got)} logit steps, expected {len(expected)}")
    max_diff = 0.0
    max_label = "none"
    for step_idx, (got_step, expected_step) in enumerate(zip(got, expected)):
        if len(got_step) != len(expected_step):
            raise VerificationError(
                f"step {step_idx}: got {len(got_step)} logits, expected {len(expected_step)}"
            )
        for idx, (got_value, expected_value) in enumerate(zip(got_step, expected_step)):
            diff = abs(got_value - expected_value)
            allowed = atol + rtol * abs(expected_value)
            if diff > max_diff:
                max_diff = diff
                max_label = f"step={step_idx} token={idx} got={got_value:.8g} expected={expected_value:.8g}"
            if diff > allowed:
                raise VerificationError(
                    f"logit mismatch at step {step_idx}, token {idx}: "
                    f"got {got_value:.8g}, expected {expected_value:.8g}, "
                    f"diff {diff:.8g} exceeds tolerance {allowed:.8g}"
                )
    return max_diff, max_label


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a generated tiny HF/Gemma-shaped reference bundle."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="directory containing config.json, model.safetensors, expected_reference.json",
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="absolute logit tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="relative logit tolerance")
    args = parser.parse_args(argv)

    try:
        generated, logits_by_step, expected = run_decode(args.bundle_dir)
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
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    print("verification passed")
    print(f"bundle: {args.bundle_dir}")
    print(f"generated_tokens: {generated}")
    print(f"logit_steps: {len(logits_by_step)}")
    print(f"max_logit_abs_diff: {max_diff:.8g} ({max_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
