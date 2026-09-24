#!/usr/bin/env python3
"""Build pinned core MLX with MLX_METAL_DEBUG and emit a sealed JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCHEMA = "rvllm.mlx_metal_debug_build.v1"
PINNED_COMMIT = "c215b6f88cf0fee0b0895623e4046cda797ef397"
CMAKE_ARGS = ["-DMLX_METAL_DEBUG=ON"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(source: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"expected MLX {PINNED_COMMIT}, got {commit}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("pinned MLX source tree is dirty")
    return {
        "mlx_commit": commit,
        "cmake_lists_sha256": sha256(source / "CMakeLists.txt"),
        "setup_py_sha256": sha256(source / "setup.py"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        source = args.mlx_source.resolve()
        identity = source_identity(source)
        plan = {
            "schema": SCHEMA,
            "status": "planned_not_built",
            **identity,
            "mlx_source": str(source),
            "cmake_args": CMAKE_ARGS,
            "target": str(args.target.resolve()),
            "builder_python": sys.executable,
            "builder_python_sha256": sha256(Path(sys.executable)),
        }
        if args.plan_only:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.target.exists() or args.target.is_symlink():
            raise FileExistsError(f"debug target already exists: {args.target}")
        if args.receipt.exists() or args.receipt.is_symlink():
            raise FileExistsError(f"build receipt already exists: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CMAKE_ARGS"] = " ".join(CMAKE_ARGS)
        # Keep stdout reserved for this tool's single JSON receipt.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(args.target), str(source)],
            check=True,
            env=environment,
            stdout=sys.stderr,
        )
        inspection_environment = dict(os.environ)
        inspection_environment["PYTHONPATH"] = str(args.target.resolve())
        inspection = subprocess.run(
            [sys.executable, "-c", "import json,mlx.core as mx; print(json.dumps({'core_path':mx.__file__,'metal_available':mx.metal.is_available()}))"],
            check=True,
            capture_output=True,
            text=True,
            env=inspection_environment,
        )
        runtime = json.loads(inspection.stdout)
        core_path = Path(runtime["core_path"]).resolve()
        if not runtime.get("metal_available") or not core_path.is_file():
            raise ValueError("built runtime does not expose an available Metal backend")
        receipt = {
            **plan,
            "status": "built_mlx_metal_debug",
            "python": str(Path(sys.executable).resolve()),
            "python_sha256": sha256(Path(sys.executable)),
            "core_path": str(core_path),
            "core_sha256": sha256(core_path),
        }
        encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.receipt.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
