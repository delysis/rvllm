#!/usr/bin/env python3
"""Capture one pinned MLX Gemma 4 Q4 normal route into an Xcode GPU trace.

This is generated-source/dispatch trace evidence, not a timing benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "rvllm.mlx_gemma4_gputrace.v1"
BUILD_SCHEMA = "rvllm.mlx_metal_debug_build.v1"
PINNED_MLX_LM_COMMIT = "87b7b583a697537aa68f47130b40884700b5f55f"
BUILD_FLAGS = ["-DMLX_METAL_DEBUG=ON"]
PROMPT_TOKENS = 256
# The token sampled from prefill plus one decode step cover both normal-route
# phases without recording dozens of repetitive decode command buffers.
GENERATION_TOKENS = 2
WEIGHT_BITS = 4


def strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(raw: str) -> None:
        raise ValueError(f"non-finite JSON number: {raw}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("capture must be a real .gputrace directory")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("capture directory contains no files")
    digest = hashlib.sha256()
    total = 0
    for item in files:
        if item.is_symlink():
            raise ValueError(f"capture contains symlink: {item}")
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        total += size
    return digest.hexdigest(), len(files), total


def git_identity(root: Path, expected: str) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected:
        raise ValueError(f"source commit mismatch: expected {expected}, got {commit}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError(f"pinned source tree is dirty: {root}")
    return {"root": str(root.resolve()), "commit": commit, "working_tree_clean": True}


def validate_build_receipt(
    path: Path, core_path: Path, mlx_source: Path, expected_mlx_commit: str
) -> dict[str, Any]:
    value = strict_json(path)
    if value.get("schema") != BUILD_SCHEMA:
        raise ValueError("unrecognized MLX debug-build receipt schema")
    if value.get("status") != "built_mlx_metal_debug":
        raise ValueError("MLX debug-build receipt is not a successful build")
    if value.get("mlx_commit") != expected_mlx_commit:
        raise ValueError("debug-build receipt does not bind the pinned MLX commit")
    if value.get("cmake_args") != BUILD_FLAGS:
        raise ValueError("debug-build receipt does not prove MLX_METAL_DEBUG=ON")
    if value.get("mlx_source") != str(mlx_source.resolve()):
        raise ValueError("debug-build source root mismatch")
    if value.get("core_path") != str(core_path.resolve()):
        raise ValueError("debug-build runtime path mismatch")
    actual = sha256_file(core_path)
    if value.get("core_sha256") != actual:
        raise ValueError("debug-build runtime hash mismatch")
    return value


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "planned_not_captured",
        "evidence_class": "generated_source_and_dispatch_trace_not_timing",
        "required_environment": {"MTL_CAPTURE_ENABLED": "1"},
        "required_build_flags": BUILD_FLAGS,
        "mlx": git_identity(args.mlx_source, args.expected_mlx_commit),
        "mlx_lm": git_identity(args.mlx_lm_source, PINNED_MLX_LM_COMMIT),
        "model": str(args.model.resolve()),
        "workload": {
            "weight_bits": WEIGHT_BITS,
            "prompt_tokens": PROMPT_TOKENS,
            "generation_tokens": GENERATION_TOKENS,
            "batch_size": 1,
            "random_seed": 0,
            "prefill_step_size": 2048,
        },
        "capture_path": str(args.capture.resolve()),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise ValueError("MTL_CAPTURE_ENABLED must equal 1")
    if args.capture.suffix != ".gputrace":
        raise ValueError("capture path must end in .gputrace")
    if args.capture.exists() or args.capture.is_symlink():
        raise FileExistsError(f"capture path already exists: {args.capture}")

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, stream_generate

    core_path = Path(mx.__file__).resolve()
    build = validate_build_receipt(
        args.build_receipt, core_path, args.mlx_source, args.expected_mlx_commit
    )
    base = plan(args)
    config = strict_json(args.model / "config.json")
    quantization = config.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("bits") != WEIGHT_BITS:
        raise ValueError("model config does not expose the requested 4-bit path")

    # Keep stdout reserved for this tool's one strict JSON document.
    with contextlib.redirect_stdout(sys.stderr):
        mx.random.seed(0)
        model, tokenizer, loaded_config = load(
            str(args.model),
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
            model_config={"quantize_activations": False},
            trust_remote_code=False,
        )
        mx.eval(model.parameters())
        tokenizer._eos_token_ids = {}
        vocab = loaded_config.get("vocab_size") or loaded_config["text_config"]["vocab_size"]
        prompt = mx.random.randint(0, vocab, (1, PROMPT_TOKENS)).tolist()[0]

        started = False
        try:
            mx.metal.start_capture(str(args.capture.resolve()))
            started = True
            responses = list(
                stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=GENERATION_TOKENS,
                    prefill_step_size=2048,
                )
            )
        finally:
            if started:
                mx.metal.stop_capture()
    if len(responses) != GENERATION_TOKENS:
        raise ValueError(f"normal route returned {len(responses)} tokens, expected {GENERATION_TOKENS}")
    capture_sha256, file_count, byte_count = sha256_tree(args.capture)
    return {
        **base,
        "status": "captured_normal_route_not_timing",
        "build": build,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", None),
            "core_path": str(core_path),
            "core_sha256": sha256_file(core_path),
            "device_info": mx.metal.device_info(),
        },
        "model_identity": {
            "config_sha256": sha256_file(args.model / "config.json"),
            "model_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        },
        "capture": {
            "format": "Xcode .gputrace directory",
            "sha256_tree_v1": capture_sha256,
            "file_count": file_count,
            "byte_count": byte_count,
        },
        "epistemic_label": (
            "This captures one MLX normal-route prefill and decode for generated-source and "
            "dispatch inspection. It is not an isolated-stage timing, end-to-end benchmark, "
            "speed comparison, correctness oracle, or promotion result."
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mlx-source", required=True, type=Path)
    result.add_argument("--expected-mlx-commit", required=True)
    result.add_argument("--mlx-lm-source", required=True, type=Path)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--capture", required=True, type=Path)
    result.add_argument("--build-receipt", type=Path)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        if not args.plan_only and args.build_receipt is None:
            raise ValueError("capture requires --build-receipt")
        receipt = plan(args) if args.plan_only else capture(args)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
