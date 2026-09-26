#!/usr/bin/env python3
"""Isolated, synchronized MLX Gemma 4 stage benchmark.

This is deliberately not an end-to-end inference benchmark.  It times one
operator graph at a time after materializing its inputs, using five warmups and
100 iterations with an ``mx.eval`` synchronization in every iteration.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA = "rvllm.mlx_gemma4_stage_microbenchmark.v1"
PINNED_MLX_LM_COMMIT = "87b7b583a697537aa68f47130b40884700b5f55f"
PINNED_BENCHMARK_PATH = "mlx_lm/benchmark.py"
PINNED_GEMMA4_TEXT_PATH = "mlx_lm/models/gemma4_text.py"
PINNED_MLX_COMMIT = "c215b6f88cf0fee0b0895623e4046cda797ef397"
PINNED_MLX_TIME_UTILS_PATH = "benchmarks/python/time_utils.py"
WARMUPS = 5
ITERATIONS = 100
DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096)
EPISTEMIC_LABEL = (
    "Isolated mx.eval-synchronized operator timings are diagnostic microbenchmarks. "
    "They are not end-to-end normal-route prefill or decode timings and do not by "
    "themselves qualify a framework or kernel performance claim."
)


@dataclass(frozen=True)
class CaseSpec:
    category: str
    variant: str | None
    mode: str
    prompt_or_context_tokens: int


STAGES = (
    ("embedding", None),
    ("qkv", "sliding_attention"),
    ("qkv", "full_attention"),
    ("attention_core_sdpa", "sliding_attention"),
    ("attention_core_sdpa", "full_attention"),
    ("o_projection", "sliding_attention"),
    ("o_projection", "full_attention"),
    ("ffn_gate_up_activation", None),
    ("ffn_down", None),
    ("rmsnorm_residual", None),
    ("lm_head", None),
)


def parse_lengths(raw: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("lengths must be comma-separated positive integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must be comma-separated positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must not contain duplicates")
    return lengths


def planned_cases(lengths: Iterable[int]) -> list[CaseSpec]:
    return [
        CaseSpec(category, variant, mode, length)
        for length in lengths
        for mode in ("prefill", "decode")
        for category, variant in STAGES
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_source_identity(
    source: Path, expected_commit: str, relative_path: str, repository: str
) -> dict[str, Any]:
    referenced_file = source / relative_path
    if not referenced_file.is_file():
        raise ValueError(f"pinned source is missing {referenced_file}")
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != expected_commit:
        raise ValueError(
            f"source commit mismatch: expected {expected_commit}, got {commit}"
        )
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--", relative_path],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"pinned source file has working-tree changes: {relative_path}")
    return {
        "repository": repository,
        "commit": commit,
        "source_root": str(source.resolve()),
        "referenced_path": relative_path,
        "referenced_path_sha256": sha256_file(referenced_file),
        "working_tree_matches_commit": True,
    }


def mlx_lm_source_identity(source: Path) -> dict[str, Any]:
    identity = git_source_identity(
        source,
        PINNED_MLX_LM_COMMIT,
        PINNED_GEMMA4_TEXT_PATH,
        "https://github.com/ml-explore/mlx-lm",
    )
    benchmark = source / PINNED_BENCHMARK_PATH
    if not benchmark.is_file():
        raise ValueError(f"pinned source is missing {benchmark}")
    identity["end_to_end_benchmark_path"] = PINNED_BENCHMARK_PATH
    identity["end_to_end_benchmark_sha256"] = sha256_file(benchmark)
    return identity


def mlx_protocol_source_identity(source: Path) -> dict[str, Any]:
    return git_source_identity(
        source,
        PINNED_MLX_COMMIT,
        PINNED_MLX_TIME_UTILS_PATH,
        "https://github.com/ml-explore/mlx",
    )


def model_identity(model_path: Path, requested_bits: int) -> dict[str, Any]:
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization")
    exposed_bits = int(quantization["bits"]) if quantization else 16
    if exposed_bits != requested_bits:
        raise ValueError(
            f"requested {requested_bits}-bit path but config exposes {exposed_bits}-bit weights"
        )
    index_path = model_path / "model.safetensors.index.json"
    return {
        "path": str(model_path.resolve()),
        "requested_weight_bits": requested_bits,
        "config_exposed_weight_bits": exposed_bits,
        "quantization": quantization,
        "config_sha256": sha256_file(config_path),
        "model_index_sha256": sha256_file(index_path),
        "model_type": config.get("model_type"),
    }


def protocol_receipt() -> dict[str, Any]:
    return {
        "warmups": WARMUPS,
        "iterations": ITERATIONS,
        "synchronization": "mx.eval(operator()) once per warmup and timed iteration",
        "timer": "time.perf_counter",
        "aggregation": "one enclosing timer for 100 synchronized iterations; report total and mean",
        "input_policy": "operator inputs are materialized before warmup and excluded from timing",
    }


def time_operator(mx: Any, operator: Callable[[], Any]) -> dict[str, Any]:
    for _ in range(WARMUPS):
        mx.eval(operator())
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        result = mx.eval(operator())
    total_seconds = time.perf_counter() - start
    return {
        "total_ms": total_seconds * 1_000.0,
        "mean_ms": total_seconds * 1_000.0 / ITERATIONS,
    }


def _text_model(model: Any) -> tuple[Any, Any]:
    language_model = getattr(model, "language_model", model)
    text = getattr(language_model, "model", language_model)
    if not hasattr(text, "layers") or not hasattr(text, "embed_tokens"):
        raise TypeError("loaded model does not expose Gemma 4 text layers and embeddings")
    return language_model, text


def _layer_indices(text: Any) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, layer in enumerate(text.layers):
        found.setdefault(layer.layer_type, index)
    missing = {"sliding_attention", "full_attention"} - found.keys()
    if missing:
        raise ValueError(f"Gemma 4 model lacks layer variants: {sorted(missing)}")
    return found


def _module_receipt(module: Any) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "bits": getattr(module, "bits", None),
        "group_size": getattr(module, "group_size", None),
        "mode": getattr(module, "mode", None),
        "weight_storage_dtype": str(getattr(weight, "dtype", None)),
    }


def _eval_inputs(mx: Any, *values: Any) -> None:
    mx.eval(tuple(value for value in values if value is not None and not isinstance(value, str)))


def _hidden(mx: Any, text: Any, tokens: int) -> Any:
    ids = mx.arange(tokens, dtype=mx.uint32)[None, :] % text.vocab_size
    hidden = text.embed_tokens(ids) * text.embed_scale
    mx.eval(hidden)
    return ids, hidden


def _projected_attention_inputs(
    mx: Any,
    attention: Any,
    hidden: Any,
    query_tokens: int,
    context_tokens: int,
    mask_helper: Callable[..., Any],
) -> tuple[Any, Any, Any, Any]:
    batch = hidden.shape[0]
    source = hidden[:, :context_tokens, :]
    query_source = source[:, -query_tokens:, :]
    queries = attention.q_proj(query_source).reshape(
        batch, query_tokens, attention.n_heads, attention.head_dim
    )
    queries = attention.q_norm(queries).transpose(0, 2, 1, 3)
    queries = attention.rope(queries, offset=context_tokens - query_tokens)
    raw_keys = attention.k_proj(source).reshape(
        batch, context_tokens, attention.n_kv_heads, attention.head_dim
    )
    keys = attention.k_norm(raw_keys).transpose(0, 2, 1, 3)
    keys = attention.rope(keys, offset=0)
    if attention.use_k_eq_v:
        values = raw_keys
    else:
        values = attention.v_proj(source).reshape(
            batch, context_tokens, attention.n_kv_heads, attention.head_dim
        )
    values = attention.v_norm(values).transpose(0, 2, 1, 3)
    mask = None if query_tokens == 1 else mask_helper(
        hidden,
        cache=None,
        window_size=attention.config.sliding_window if attention.is_sliding else None,
    )
    _eval_inputs(mx, queries, keys, values, mask)
    return queries, keys, values, mask


def _sdpa(
    sdpa_helper: Callable[..., Any],
    attention: Any,
    values: tuple[Any, Any, Any, Any],
) -> Any:
    queries, keys, vals, mask = values
    return sdpa_helper(
        queries,
        keys,
        vals,
        cache=None,
        scale=attention.scale,
        mask=mask,
    )


def _linear_bits(modules: Iterable[Any]) -> list[int | None]:
    return [getattr(module, "bits", None) for module in modules]


def validate_weight_modules(
    modules: Iterable[Any], requested_bits: int, category: str
) -> None:
    observed = _linear_bits(modules)
    expected = None if requested_bits == 16 else requested_bits
    mismatches = [bits for bits in observed if bits != expected]
    if mismatches:
        raise ValueError(
            f"{category} weight modules do not expose requested {requested_bits}-bit "
            f"path: observed bits {observed}"
        )


def category_requires_requested_linear_bits(category: str) -> bool:
    return category in {
        "qkv",
        "o_projection",
        "ffn_gate_up_activation",
        "ffn_down",
    }


def build_operator(
    mx: Any,
    language_model: Any,
    text: Any,
    indices: dict[str, int],
    spec: CaseSpec,
    requested_bits: int,
    sdpa_helper: Callable[..., Any],
    mask_helper: Callable[..., Any],
    geglu_helper: Callable[..., Any],
    logit_softcap_helper: Callable[..., Any],
) -> tuple[Callable[[], Any], dict[str, Any]]:
    context = spec.prompt_or_context_tokens
    query_tokens = context if spec.mode == "prefill" else 1
    ids, hidden_context = _hidden(mx, text, context)
    hidden = hidden_context if spec.mode == "prefill" else hidden_context[:, -1:, :]

    if spec.category == "embedding":
        operator = lambda: text.embed_tokens(
            ids if spec.mode == "prefill" else ids[:, -1:]
        ) * text.embed_scale
        modules = [text.embed_tokens]
    elif spec.variant is not None:
        layer = text.layers[indices[spec.variant]]
        attention = layer.self_attn
        if spec.category == "qkv":
            projections = [attention.q_proj, attention.k_proj]
            if not attention.use_k_eq_v:
                projections.append(attention.v_proj)
            operator = lambda: tuple(projection(hidden) for projection in projections)
            modules = projections
        else:
            projected = _projected_attention_inputs(
                mx, attention, hidden_context, query_tokens, context, mask_helper
            )
            if spec.category == "attention_core_sdpa":
                operator = lambda: _sdpa(sdpa_helper, attention, projected)
                modules = []
            elif spec.category == "o_projection":
                attention_output = _sdpa(sdpa_helper, attention, projected)
                attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
                    1, query_tokens, -1
                )
                mx.eval(attention_output)
                operator = lambda: attention.o_proj(attention_output)
                modules = [attention.o_proj]
            else:
                raise AssertionError(spec)
    else:
        layer = text.layers[indices["sliding_attention"]]
        if spec.category == "ffn_gate_up_activation":
            operator = lambda: geglu_helper(
                layer.mlp.gate_proj(hidden), layer.mlp.up_proj(hidden)
            )
            modules = [layer.mlp.gate_proj, layer.mlp.up_proj]
        elif spec.category == "ffn_down":
            activated = geglu_helper(
                layer.mlp.gate_proj(hidden), layer.mlp.up_proj(hidden)
            )
            mx.eval(activated)
            operator = lambda: layer.mlp.down_proj(activated)
            modules = [layer.mlp.down_proj]
        elif spec.category == "rmsnorm_residual":
            operator = lambda: hidden + layer.post_attention_layernorm(hidden)
            modules = [layer.post_attention_layernorm]
        elif spec.category == "lm_head":
            normalized = text.norm(hidden)
            mx.eval(normalized)
            head = text.embed_tokens.as_linear
            softcap = getattr(language_model, "final_logit_softcapping", None)
            operator = lambda: head(normalized)
            if softcap is not None:
                unsoftcapped = operator
                operator = lambda: logit_softcap_helper(softcap, unsoftcapped())
            modules = [text.embed_tokens]
        else:
            raise AssertionError(spec)

    if modules and category_requires_requested_linear_bits(spec.category):
        validate_weight_modules(modules, requested_bits, spec.category)
    receipt = {
        "input_shape": list(hidden.shape),
        "operator_modules": [_module_receipt(module) for module in modules],
        "operator_weight_bits": _linear_bits(modules),
        "scope": (
            "representative post-attention RMSNorm plus residual add"
            if spec.category == "rmsnorm_residual"
            else "linear projections before Q/K/V normalization and RoPE"
            if spec.category == "qkv"
            else "token lookup including Gemma embed_scale; storage format recorded as observed"
            if spec.category == "embedding"
            else "tied embedding as_linear projection plus logit softcap; shared storage format recorded as observed"
            if spec.category == "lm_head"
            else None
        ),
        "implementation_path": (
            "mlx_lm.models.base.scaled_dot_product_attention"
            if spec.category == "attention_core_sdpa"
            else None
        ),
    }
    return operator, receipt


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.mlx_lm_source)
    mlx_source = Path(args.mlx_source)
    model_path = Path(args.model)
    source_receipt = mlx_lm_source_identity(source)
    protocol_source_receipt = mlx_protocol_source_identity(mlx_source)
    identity = model_identity(model_path, args.weight_bits)
    lengths = parse_lengths(args.lengths)
    base = {
        "schema": SCHEMA,
        "epistemic_label": EPISTEMIC_LABEL,
        "protocol": protocol_receipt(),
        "mlx_protocol_source": protocol_source_receipt,
        "mlx_lm_source": source_receipt,
        "model": identity,
        "planned_cases": [asdict(case) for case in planned_cases(lengths)],
    }
    if args.plan_only:
        return {**base, "status": "planned_not_measured", "cases": []}

    sys.path.insert(0, str(source))
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.gemma4_text import geglu, logit_softcap

    mx.random.seed(0)
    # Keep stdout a single JSON document even if an upstream loader emits progress.
    with contextlib.redirect_stdout(sys.stderr):
        model, _tokenizer, _config = load(
            str(model_path),
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
            model_config={"quantize_activations": False},
            trust_remote_code=False,
        )
    mx.eval(model.parameters())
    language_model, text = _text_model(model)
    indices = _layer_indices(text)
    cases = []
    for spec in planned_cases(lengths):
        operator, operator_receipt = build_operator(
            mx,
            language_model,
            text,
            indices,
            spec,
            args.weight_bits,
            scaled_dot_product_attention,
            create_attention_mask,
            geglu,
            logit_softcap,
        )
        cases.append(
            {
                **asdict(spec),
                **operator_receipt,
                "timing": time_operator(mx, operator),
            }
        )
    return {
        **base,
        "status": "measured_microbenchmark_not_normal_route",
        "runtime": {
            "mlx_lm_version": getattr(mlx_lm, "__version__", None),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "representative_layers": indices,
        "cases": cases,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mlx-source", required=True)
    result.add_argument("--mlx-lm-source", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--weight-bits", required=True, type=int, choices=(4, 8, 16))
    result.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> int:
    try:
        receipt = benchmark(parser().parse_args())
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "epistemic_label": EPISTEMIC_LABEL,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
