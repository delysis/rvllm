#!/usr/bin/env python3
"""Generate HF/Transformers text reference artifacts for Gemma 4 E2B.

This is a convenience wrapper around ``dump_gemma4_hf_reference_logits.py``.
It writes JSON artifacts outside the repository by default and does not execute
rvLLM, Metal, ANE, or any production serving path.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TOP_K = 16
DEFAULT_SELECTED_TOKEN_IDS = ""
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Hello",
    "Once upon a time",
    "The capital of France is",
)


@dataclass(frozen=True)
class TextCase:
    name: str
    prompt_text: str
    decode_steps: int
    output_name: str
    full_logits: bool = False
    no_special_tokens: bool = False


def prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", prompt.strip().lower()).strip("-")
    return slug or "empty"


def custom_suite_cases(args: argparse.Namespace) -> tuple[TextCase, ...]:
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    decode_steps = args.decode_steps
    cases = []
    for prompt in prompts:
        slug = prompt_slug(prompt)
        step = "step1" if decode_steps == 1 else f"steps{decode_steps}"
        prefix = "full-logits" if args.full_logits else "reference"
        output_name = f"gemma4-e2b-hf-text-{prefix}-{slug}-{step}.json"
        cases.append(
            TextCase(
                name=f"text_{prefix}_{slug}_{step}",
                prompt_text=prompt,
                decode_steps=decode_steps,
                output_name=output_name,
                full_logits=args.full_logits,
                no_special_tokens=args.no_special_tokens,
            )
        )
    return tuple(cases)


def case_command(args: argparse.Namespace, case: TextCase, output_path: Path) -> list[str]:
    script = Path(__file__).with_name("dump_gemma4_hf_reference_logits.py")
    cmd = [
        args.python,
        str(script),
        str(args.model_dir),
        "--prompt-text",
        case.prompt_text,
        "--decode-steps",
        str(case.decode_steps),
        "--selected-token-ids",
        args.selected_token_ids,
        "--top-k",
        str(args.top_k),
        "--output",
        str(output_path),
    ]
    if case.full_logits:
        cmd.append("--full-logits")
    if case.no_special_tokens:
        cmd.append("--no-special-tokens")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def write_manifest(args: argparse.Namespace, planned: list[dict]) -> None:
    payload = {
        "schema": "rvllm.gemma4_e2b_hf_text_reference_suite.v1",
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "cases": planned,
        "claim": "HF/Transformers text reference artifact manifest only; no rvLLM, Metal, ANE, performance, or production claim.",
    }
    manifest_path = args.output_dir / "gemma4-e2b-hf-text-reference-suite-manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote text reference suite manifest to {manifest_path}")


def run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Gemma 4 E2B HF text reference artifacts."
    )
    parser.add_argument("model_dir", type=Path, help="local Hugging Face Gemma 4 E2B model directory")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rvllm-e2b-text-reference-suite"))
    parser.add_argument("--python", default=sys.executable, help="Python executable for the per-case dumper")
    parser.add_argument("--prompt", action="append", help="prompt text; repeat to build a custom suite")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--selected-token-ids", default=DEFAULT_SELECTED_TOKEN_IDS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--full-logits", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write only the manifest")
    args = parser.parse_args(argv)

    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.prompt is not None and any(prompt == "" for prompt in args.prompt):
        parser.error("--prompt must not be empty")
    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    planned = []
    for case in custom_suite_cases(args):
        output_path = args.output_dir / case.output_name
        cmd = case_command(args, case, output_path)
        planned.append(
            {
                "name": case.name,
                "prompt_text": case.prompt_text,
                "decode_steps": case.decode_steps,
                "full_logits": case.full_logits,
                "no_bos": case.no_special_tokens,
                "output": str(output_path),
                "command": cmd,
            }
        )
        if args.skip_existing and output_path.exists():
            print(f"skipping existing {case.name}: {output_path}")
            continue
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    write_manifest(args, planned)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
