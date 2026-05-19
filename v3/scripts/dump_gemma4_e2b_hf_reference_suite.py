#!/usr/bin/env python3
"""Generate the standard HF/Transformers Gemma 4 E2B reference artifacts.

This is a convenience wrapper around ``dump_gemma4_hf_reference_logits.py``.
It writes JSON artifacts outside the repository by default and does not execute
rvLLM, Metal, ANE, or any production serving path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SELECTED_TOKEN_IDS = "0,1,2,3,4,5"
DEFAULT_TOP_K = 16


@dataclass(frozen=True)
class SuiteCase:
    name: str
    prompt_token_ids: str
    decode_steps: int
    output_name: str
    full_logits: bool = True


REFERENCE_SUITE: tuple[SuiteCase, ...] = (
    SuiteCase(
        name="selected_prompt_2_4_step1",
        prompt_token_ids="2,4",
        decode_steps=1,
        output_name="gemma4-e2b-hf-reference-logits.json",
        full_logits=False,
    ),
    SuiteCase(
        name="full_prompt_2_4_step1",
        prompt_token_ids="2,4",
        decode_steps=1,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-4-step1.json",
    ),
    SuiteCase(
        name="full_prompt_2_4_steps2",
        prompt_token_ids="2,4",
        decode_steps=2,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json",
    ),
    SuiteCase(
        name="full_prompt_2_4_steps4",
        prompt_token_ids="2,4",
        decode_steps=4,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-4-steps4.json",
    ),
    SuiteCase(
        name="full_prompt_2_4_steps8",
        prompt_token_ids="2,4",
        decode_steps=8,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-4-steps8.json",
    ),
    SuiteCase(
        name="full_prompt_2_17_step1",
        prompt_token_ids="2,17",
        decode_steps=1,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-17-step1.json",
    ),
    SuiteCase(
        name="full_prompt_2_17_steps2",
        prompt_token_ids="2,17",
        decode_steps=2,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-17-steps2.json",
    ),
    SuiteCase(
        name="full_prompt_2_17_42_4_step1",
        prompt_token_ids="2,17,42,4",
        decode_steps=1,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json",
    ),
    SuiteCase(
        name="full_prompt_2_17_42_4_steps2",
        prompt_token_ids="2,17,42,4",
        decode_steps=2,
        output_name="gemma4-e2b-hf-full-logits-prompt-2-17-42-4-steps2.json",
    ),
)


def parse_token_ids(raw: str) -> str:
    parts = []
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
        parts.append(str(token_id))
    if not parts:
        raise argparse.ArgumentTypeError("at least one token id is required")
    return ",".join(parts)


def case_slug(prompt_token_ids: str, decode_steps: int) -> str:
    prompt = prompt_token_ids.replace(",", "-")
    step = "step1" if decode_steps == 1 else f"steps{decode_steps}"
    return f"prompt-{prompt}-{step}"


def custom_suite_cases(args: argparse.Namespace) -> tuple[SuiteCase, ...]:
    if not args.case:
        return REFERENCE_SUITE
    decode_steps = args.decode_steps or 1
    cases = []
    for prompt_token_ids in args.case:
        slug = case_slug(prompt_token_ids, decode_steps)
        prefix = "full-logits" if args.full_logits else "reference"
        cases.append(
            SuiteCase(
                name=f"{prefix}_{slug}",
                prompt_token_ids=prompt_token_ids,
                decode_steps=decode_steps,
                output_name=f"gemma4-e2b-hf-{prefix}-{slug}.json",
                full_logits=args.full_logits,
            )
        )
    return tuple(cases)


def case_command(args: argparse.Namespace, case: SuiteCase, output_path: Path) -> list[str]:
    script = Path(__file__).with_name("dump_gemma4_hf_reference_logits.py")
    cmd = [
        args.python,
        str(script),
        str(args.model_dir),
        "--prompt-token-ids",
        case.prompt_token_ids,
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
    if args.device:
        cmd.extend(["--device", args.device])
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def write_manifest(args: argparse.Namespace, planned: list[dict]) -> None:
    payload = {
        "schema": "rvllm.gemma4_e2b_hf_reference_suite.v1",
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "cases": planned,
        "claim": "HF/Transformers reference artifact manifest only; no rvLLM, Metal, ANE, performance, or production claim.",
    }
    manifest_path = args.output_dir / "gemma4-e2b-hf-reference-suite-manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote suite manifest to {manifest_path}")


def run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the standard Gemma 4 E2B HF reference artifact suite."
    )
    parser.add_argument("model_dir", type=Path, help="local Hugging Face Gemma 4 E2B model directory")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"), help="artifact output directory")
    parser.add_argument("--python", default=sys.executable, help="Python executable for the per-case dumper")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_token_ids,
        help="comma-separated prompt token IDs; repeat to build a custom suite",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        help="decode steps for every custom --case (defaults to 1 when --case is used)",
    )
    parser.add_argument("--selected-token-ids", default=DEFAULT_SELECTED_TOKEN_IDS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--full-logits",
        action="store_true",
        help="include full logits for custom --case outputs",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write only the manifest")
    args = parser.parse_args(argv)

    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.decode_steps is not None and args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    if args.full_logits and not args.case:
        parser.error("--full-logits is only meaningful with one or more --case values")
    if args.decode_steps is not None and not args.case:
        parser.error("--decode-steps is only meaningful with one or more --case values")
    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_cases = custom_suite_cases(args)
    planned = []
    for case in suite_cases:
        output_path = args.output_dir / case.output_name
        cmd = case_command(args, case, output_path)
        planned.append(
            {
                "name": case.name,
                "prompt_token_ids": case.prompt_token_ids,
                "decode_steps": case.decode_steps,
                "full_logits": case.full_logits,
                "output": str(output_path),
                "command": cmd,
            }
        )
        if args.skip_existing and output_path.exists():
            print(f"skipping existing {case.name}: {output_path}")
            continue
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    write_manifest(args, planned)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
