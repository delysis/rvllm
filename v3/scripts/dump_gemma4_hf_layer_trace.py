#!/usr/bin/env python3
"""Dump compact HF/Transformers layer activation summaries for Gemma 4.

This is a reference/debug hook only. It writes JSON summaries, not full tensors,
so the output can live in /tmp and be compared with rvLLM Metal debug summaries
without committing large activation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def parse_token_ids(raw: str) -> list[int]:
    out: list[int] = []
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


def finite_float(value: Any) -> float | None:
    out = float(value)
    if math.isfinite(out):
        return out
    return None


def first_tensor(value: Any):
    try:
        import torch
    except ImportError:  # pragma: no cover - handled by run import guard.
        return None

    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_summary(tensor, *, selected_dims: Sequence[int], first_values: int) -> dict[str, Any]:
    import torch

    t = tensor.detach().float().cpu()
    flat = t.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    first_nonfinite_index = None
    if int(finite.sum().item()) != int(flat.numel()):
        first_nonfinite_index = int((~finite).nonzero(as_tuple=False)[0].item())
    if finite_values.numel() == 0:
        max_abs = 0.0
        mean_abs = 0.0
    else:
        abs_values = finite_values.abs()
        max_abs = float(abs_values.max().item())
        mean_abs = float(abs_values.mean().item())

    selected = []
    for idx in selected_dims:
        if 0 <= idx < flat.numel():
            selected.append({"index": int(idx), "value": finite_float(flat[idx].item())})

    payload: dict[str, Any] = {
        "shape": [int(dim) for dim in t.shape],
        "dtype": str(tensor.dtype),
        "total_count": int(flat.numel()),
        "finite_count": int(finite.sum().item()),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "first_nonfinite_index": first_nonfinite_index,
        "selected": selected,
        "first_values": [finite_float(value) for value in flat[:first_values].tolist()],
    }
    if t.ndim >= 3 and t.shape[0] == 1:
        per_token = []
        for token_idx in range(int(t.shape[1])):
            token_flat = t[0, token_idx].reshape(-1)
            token_finite = torch.isfinite(token_flat)
            token_values = token_flat[token_finite]
            if token_values.numel() == 0:
                token_max_abs = 0.0
                token_mean_abs = 0.0
            else:
                token_abs = token_values.abs()
                token_max_abs = float(token_abs.max().item())
                token_mean_abs = float(token_abs.mean().item())
            per_token.append(
                {
                    "token_index": token_idx,
                    "total_count": int(token_flat.numel()),
                    "finite_count": int(token_finite.sum().item()),
                    "max_abs": token_max_abs,
                    "mean_abs": token_mean_abs,
                    "first_values": [
                        finite_float(value) for value in token_flat[:first_values].tolist()
                    ],
                }
            )
        payload["per_token"] = per_token
    return payload


def capture_output(trace_tensors: dict[str, Any], key: str):
    def hook(_module, _inputs, output):
        tensor = first_tensor(output)
        if tensor is not None:
            trace_tensors[key] = tensor.detach()

    return hook


def capture_input(trace_tensors: dict[str, Any], key: str, input_index: int = 0):
    def hook(_module, inputs):
        if len(inputs) > input_index:
            tensor = first_tensor(inputs[input_index])
            if tensor is not None:
                trace_tensors[key] = tensor.detach()

    return hook


def capture_layer_input(trace_tensors: dict[str, Any]):
    def hook(_module, inputs):
        if inputs:
            tensor = first_tensor(inputs[0])
            if tensor is not None:
                trace_tensors["input_to_layer"] = tensor.detach()
        if len(inputs) > 1:
            tensor = first_tensor(inputs[1])
            if tensor is not None:
                trace_tensors["per_layer_input"] = tensor.detach()

    return hook


def capture_attention_kwargs(trace_tensors: dict[str, Any]):
    def hook(_module, _args, kwargs):
        position_embeddings = kwargs.get("position_embeddings")
        if isinstance(position_embeddings, (list, tuple)) and len(position_embeddings) == 2:
            trace_tensors["rope_cos"] = position_embeddings[0].detach()
            trace_tensors["rope_sin"] = position_embeddings[1].detach()

    return hook


def require_module(modules: dict[str, Any], name: str):
    try:
        return modules[name]
    except KeyError as exc:
        raise RuntimeError(f"module not found in HF model: {name}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required; try /tmp/rvllm-gemma4-hf-ref-venv/bin/python"
        ) from exc

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_dir}")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    if args.device_map is None:
        model.to(args.device)
    model.eval()

    prefix = f"model.language_model.layers.{args.layer}"
    modules = dict(model.named_modules())
    trace_tensors: dict[str, Any] = {}
    handles = [
        require_module(modules, prefix).register_forward_pre_hook(capture_layer_input(trace_tensors)),
        require_module(modules, f"{prefix}.self_attn").register_forward_pre_hook(
            capture_attention_kwargs(trace_tensors), with_kwargs=True
        ),
        require_module(modules, f"{prefix}.input_layernorm").register_forward_hook(
            capture_output(trace_tensors, "after_input_layernorm")
        ),
        require_module(modules, f"{prefix}.self_attn.q_proj").register_forward_hook(
            capture_output(trace_tensors, "q_projection")
        ),
        require_module(modules, f"{prefix}.self_attn.k_proj").register_forward_hook(
            capture_output(trace_tensors, "k_projection")
        ),
        require_module(modules, f"{prefix}.self_attn.v_proj").register_forward_hook(
            capture_output(trace_tensors, "v_projection")
        ),
        require_module(modules, f"{prefix}.self_attn.q_norm").register_forward_hook(
            capture_output(trace_tensors, "after_q_norm")
        ),
        require_module(modules, f"{prefix}.self_attn.k_norm").register_forward_hook(
            capture_output(trace_tensors, "after_k_norm")
        ),
        require_module(modules, f"{prefix}.self_attn.v_norm").register_forward_hook(
            capture_output(trace_tensors, "after_v_norm")
        ),
        require_module(modules, f"{prefix}.self_attn.o_proj").register_forward_pre_hook(
            capture_input(trace_tensors, "attention_output")
        ),
        require_module(modules, f"{prefix}.self_attn.o_proj").register_forward_hook(
            capture_output(trace_tensors, "after_o_proj")
        ),
        require_module(modules, f"{prefix}.post_attention_layernorm").register_forward_hook(
            capture_output(trace_tensors, "after_post_attention_layernorm")
        ),
        require_module(modules, f"{prefix}.pre_feedforward_layernorm").register_forward_hook(
            capture_output(trace_tensors, "after_pre_feedforward_layernorm")
        ),
        require_module(modules, f"{prefix}.mlp").register_forward_hook(
            capture_output(trace_tensors, "after_ffn_branch")
        ),
        require_module(modules, f"{prefix}.post_feedforward_layernorm").register_forward_hook(
            capture_output(trace_tensors, "after_post_feedforward_layernorm")
        ),
        require_module(modules, f"{prefix}.per_layer_input_gate").register_forward_hook(
            capture_output(trace_tensors, "per_layer_input_gate")
        ),
        require_module(modules, f"{prefix}.per_layer_projection").register_forward_hook(
            capture_output(trace_tensors, "per_layer_projection")
        ),
        require_module(modules, f"{prefix}.post_per_layer_input_norm").register_forward_hook(
            capture_output(trace_tensors, "post_per_layer_input_norm")
        ),
        require_module(modules, prefix).register_forward_hook(
            capture_output(trace_tensors, "final_residual_after_layer")
        ),
    ]

    input_ids = torch.tensor([args.prompt_token_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()

    if {"after_q_norm", "after_k_norm", "rope_cos", "rope_sin"} <= trace_tensors.keys():
        cos = trace_tensors["rope_cos"].to(trace_tensors["after_q_norm"].device)
        sin = trace_tensors["rope_sin"].to(trace_tensors["after_q_norm"].device)
        trace_tensors["after_rope_q"] = apply_rotary_pos_emb(
            trace_tensors["after_q_norm"], cos, sin, unsqueeze_dim=2
        ).detach()
        trace_tensors["after_rope_k"] = apply_rotary_pos_emb(
            trace_tensors["after_k_norm"], cos, sin, unsqueeze_dim=2
        ).detach()

    summaries = {
        name: tensor_summary(
            tensor,
            selected_dims=args.selected_dims,
            first_values=args.first_values,
        )
        for name, tensor in sorted(trace_tensors.items())
        if name not in {"rope_cos", "rope_sin"}
    }
    logits = outputs.logits[0, -1].detach().float().cpu()
    return {
        "schema": "rvllm.gemma4_hf_layer_trace.v1",
        "model_dir": str(model_dir),
        "prompt_token_ids": args.prompt_token_ids,
        "layer": int(args.layer),
        "device": str(next(model.parameters()).device),
        "logits_next_token": int(torch.argmax(logits).item()),
        "summaries": summaries,
        "claim": "HF/Transformers layer trace artifact only; no rvLLM, Metal, ANE, or production claim.",
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Dump compact HF/Transformers Gemma 4 layer activation summaries."
    )
    parser.add_argument("model_dir", type=Path, help="local Hugging Face model directory")
    parser.add_argument("--prompt-token-ids", type=parse_token_ids, required=True)
    parser.add_argument("--layer", type=int, default=4, help="0-based text decoder layer index")
    parser.add_argument(
        "--selected-dims",
        type=parse_token_ids,
        default=parse_token_ids("0,1,2,3,4,5,16,32,64,128,256,512,1024,1535"),
        help="comma-separated flattened tensor indices to include",
    )
    parser.add_argument("--first-values", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.first_values < 0:
        parser.error("--first-values must be non-negative")

    try:
        payload = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001 - CLI should report all failures.
        print(f"HF layer trace dump failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote HF layer trace to {args.output}")
    print(f"layer={payload['layer']} next_token={payload['logits_next_token']}")
    for name in [
        "input_to_layer",
        "after_input_layernorm",
        "q_projection",
        "after_q_norm",
        "after_rope_q",
        "attention_output",
        "after_o_proj",
        "after_ffn_branch",
        "final_residual_after_layer",
    ]:
        summary = payload["summaries"].get(name)
        if summary:
            print(
                f"{name}: finite={summary['finite_count']}/{summary['total_count']} "
                f"max_abs={summary['max_abs']:.6e} mean_abs={summary['mean_abs']:.6e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
