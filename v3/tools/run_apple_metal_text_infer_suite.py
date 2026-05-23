#!/usr/bin/env python3
"""Run rvllm_metal_infer against an HF text reference manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPORT = Path("/tmp/rvllm-e2b-text-infer-suite-report.json")
REPORT_SCHEMA = "rvllm.apple_metal_text_infer_suite_report.v1"
CLAIM = "bounded Apple Metal text inference suite only; no production inference claim."
TEXT_INFER_SCHEMA = "rvllm.apple_metal_text_infer.v1"


@dataclass(frozen=True)
class ManifestCase:
    name: str
    prompt_text: str
    max_new_tokens: int
    reference_path: Path
    max_total_tokens: int | None = None
    no_bos: bool = False


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rvllm_metal_infer for each case in an HF text reference manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cargo-bin", default="rvllm_metal_infer")
    parser.add_argument("--cargo-package", default="rvllm-runtime")
    parser.add_argument("--features", default="apple")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true", help="print commands without running cargo")
    return parser.parse_args(argv)


def flag_value(command: Sequence[str], flag: str) -> str | None:
    for idx, item in enumerate(command):
        if item == flag and idx + 1 < len(command):
            return command[idx + 1]
    return None


def case_prompt(raw_case: dict[str, Any]) -> str:
    for key in ("prompt_text", "prompt"):
        value = raw_case.get(key)
        if isinstance(value, str) and value:
            return value
    command = raw_case.get("command")
    if isinstance(command, list):
        value = flag_value([str(item) for item in command], "--prompt-text")
        if value is not None:
            return value
    raise ValueError(f"manifest case {raw_case.get('name', '<unnamed>')} is missing prompt_text")


def resolve_reference_path(manifest_path: Path, manifest: dict[str, Any], raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest case is missing output/reference artifact path")
    path = Path(raw)
    if path.is_absolute():
        return path
    output_dir = manifest.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        return Path(output_dir) / path
    return manifest_path.parent / path


def load_cases(manifest_path: Path) -> list[ManifestCase]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest must contain a non-empty cases array")

    cases = []
    for idx, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"manifest case {idx} must be an object")
        command = raw_case.get("command")
        command = [str(item) for item in command] if isinstance(command, list) else []
        name = str(raw_case.get("name") or f"case_{idx}")
        decode_steps = raw_case.get("decode_steps") or raw_case.get("max_new_tokens")
        if decode_steps is None:
            decode_steps = flag_value(command, "--decode-steps") or flag_value(command, "--max-new-tokens")
        if decode_steps is None:
            raise ValueError(f"manifest case {name} is missing decode_steps/max_new_tokens")
        max_total_tokens = raw_case.get("max_total_tokens")
        reference_raw = (
            raw_case.get("output")
            or raw_case.get("reference")
            or raw_case.get("reference_path")
            or raw_case.get("path")
        )
        cases.append(
            ManifestCase(
                name=name,
                prompt_text=case_prompt(raw_case),
                max_new_tokens=int(decode_steps),
                reference_path=resolve_reference_path(manifest_path, manifest, reference_raw),
                max_total_tokens=int(max_total_tokens) if max_total_tokens is not None else None,
                no_bos=bool(raw_case.get("no_bos", False)),
            )
        )
    return cases


def cli_command(args: argparse.Namespace, case: ManifestCase) -> list[str]:
    command = [
        args.cargo,
        "run",
        "-p",
        args.cargo_package,
        "--features",
        args.features,
        "--bin",
        args.cargo_bin,
        "--",
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        case.prompt_text,
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--large-model-opt-in",
        "--hf-reference",
        str(case.reference_path),
        "--json",
    ]
    if case.max_total_tokens is not None:
        command.extend(["--max-total-tokens", str(case.max_total_tokens)])
    if case.no_bos:
        command.append("--no-bos")
    return command


def parse_json_stdout(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stdout[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("CLI stdout did not contain a JSON object")
    return value


def print_table(results: Sequence[dict[str, Any]]) -> None:
    print("case\tsteps\tstatus\tgenerated\treference")
    for result in results:
        generated = ",".join(str(item) for item in result.get("generated_token_ids", []))
        print(
            "\t".join(
                [
                    result["name"],
                    str(result["max_new_tokens"]),
                    result["status"],
                    generated,
                    result["reference_path"],
                ]
            )
        )


def run_case(args: argparse.Namespace, case: ManifestCase, command: Sequence[str]) -> dict[str, Any]:
    if not case.reference_path.is_file():
        return {
            "name": case.name,
            "prompt_text": case.prompt_text,
            "max_new_tokens": case.max_new_tokens,
            "reference_path": str(case.reference_path),
            "command": list(command),
            "status": "fail",
            "error": f"reference artifact does not exist: {case.reference_path}",
        }

    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "name": case.name,
            "prompt_text": case.prompt_text,
            "max_new_tokens": case.max_new_tokens,
            "reference_path": str(case.reference_path),
            "command": list(command),
            "status": "fail",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    report = parse_json_stdout(proc.stdout)
    mismatches = []
    if report.get("schema") != TEXT_INFER_SCHEMA:
        mismatches.append(f"unexpected schema: {report.get('schema')!r}")
    claim = report.get("claim")
    if not isinstance(claim, str) or "not production-ready" not in claim:
        mismatches.append("non-production claim was not preserved")
    hf_reference = report.get("hf_reference")
    matched = isinstance(hf_reference, dict) and hf_reference.get("matched") is True
    if not matched:
        mismatches.append("hf_reference.matched is not true")
    generated = report.get("generated_token_ids", [])
    if not isinstance(generated, list) or len(generated) != case.max_new_tokens:
        mismatches.append(
            f"generated_token_ids length {len(generated) if isinstance(generated, list) else '<non-list>'} "
            f"does not match max_new_tokens {case.max_new_tokens}"
        )

    return {
        "name": case.name,
        "prompt_text": case.prompt_text,
        "max_new_tokens": case.max_new_tokens,
        "reference_path": str(case.reference_path),
        "command": list(command),
        "status": "pass" if not mismatches else "fail",
        "schema": report.get("schema"),
        "claim": report.get("claim"),
        "prompt_token_ids": report.get("prompt_token_ids", []),
        "generated_token_ids": generated if isinstance(generated, list) else [],
        "output_text": report.get("output_text"),
        "hf_reference": hf_reference,
        "mismatches": mismatches,
    }


def write_report(args: argparse.Namespace, results: Sequence[dict[str, Any]], status: str) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "claim": CLAIM,
        "status": status,
        "manifest": str(args.manifest),
        "model_dir": str(args.model_dir),
        "results": list(results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote text inference suite report to {args.report}")


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model_dir}")

    cases = load_cases(args.manifest)
    results: list[dict[str, Any]] = []
    for case in cases:
        command = cli_command(args, case)
        print(shlex.join(command))
        if args.dry_run:
            results.append(
                {
                    "name": case.name,
                    "prompt_text": case.prompt_text,
                    "max_new_tokens": case.max_new_tokens,
                    "reference_path": str(case.reference_path),
                    "command": command,
                    "status": "dry_run",
                }
            )
            continue
        results.append(run_case(args, case, command))

    print_table(results)
    status = "dry_run" if args.dry_run else "pass"
    if not args.dry_run and any(result["status"] != "pass" for result in results):
        status = "fail"
    write_report(args, results, status)
    return 0 if status in {"pass", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
