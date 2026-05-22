#!/usr/bin/env python3
"""Run the diagnostic Apple Metal E2B CLI against an HF reference manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPORT = Path("/tmp/rvllm-e2b-cli-suite-report.json")
DEFAULT_TOP_K = 16
REPORT_SCHEMA = "rvllm.apple_metal_e2b_cli_suite_report.v1"
CLAIM = "diagnostic Apple Metal CLI suite only; no production inference claim."


@dataclass(frozen=True)
class ManifestCase:
    name: str
    prompt_token_ids: str
    decode_steps: int
    top_k: int
    reference_path: Path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probe_apple_metal_decode for each case in an HF reference-suite manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cargo-bin", default="probe_apple_metal_decode")
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


def case_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return ",".join(str(item) for item in value)
    raise ValueError(f"invalid prompt_token_ids in manifest case: {value!r}")


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
        command = command if isinstance(command, list) else []
        name = str(raw_case.get("name") or f"case_{idx}")
        prompt_token_ids = case_prompt(raw_case.get("prompt_token_ids"))
        decode_steps = raw_case.get("decode_steps") or flag_value(command, "--decode-steps")
        if decode_steps is None:
            raise ValueError(f"manifest case {name} is missing decode_steps")
        top_k = raw_case.get("top_k") or flag_value(command, "--top-k") or DEFAULT_TOP_K
        reference_raw = (
            raw_case.get("output")
            or raw_case.get("reference")
            or raw_case.get("reference_path")
            or raw_case.get("path")
        )
        cases.append(
            ManifestCase(
                name=name,
                prompt_token_ids=prompt_token_ids,
                decode_steps=int(decode_steps),
                top_k=int(top_k),
                reference_path=resolve_reference_path(manifest_path, manifest, reference_raw),
            )
        )
    return cases


def cli_command(args: argparse.Namespace, case: ManifestCase) -> list[str]:
    return [
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
        "--prompt-token-ids",
        case.prompt_token_ids,
        "--decode-steps",
        str(case.decode_steps),
        "--top-k",
        str(case.top_k),
        "--large-model-opt-in",
        "--hf-reference",
        str(case.reference_path),
        "--json",
    ]


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
    print("case\tprompt\tsteps\tstatus\tsampled\ttop_k\treference")
    for result in results:
        sampled = ",".join(str(item) for item in result.get("sampled_token_ids", []))
        top_k_count = result.get("top_k_count", "")
        print(
            "\t".join(
                [
                    result["name"],
                    result["prompt_token_ids"],
                    str(result["decode_steps"]),
                    result["status"],
                    sampled,
                    str(top_k_count),
                    result["reference_path"],
                ]
            )
        )


def run_case(args: argparse.Namespace, case: ManifestCase, command: Sequence[str]) -> dict[str, Any]:
    if not case.reference_path.is_file():
        return {
            "name": case.name,
            "prompt_token_ids": case.prompt_token_ids,
            "decode_steps": case.decode_steps,
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
            "prompt_token_ids": case.prompt_token_ids,
            "decode_steps": case.decode_steps,
            "reference_path": str(case.reference_path),
            "command": list(command),
            "status": "fail",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    report = parse_json_stdout(proc.stdout)
    hf_reference = report.get("hf_reference")
    matched = isinstance(hf_reference, dict) and hf_reference.get("matched") is True
    top_k = report.get("per_step_top_k") or []
    first_top_k = top_k[0].get("top_k", []) if top_k and isinstance(top_k[0], dict) else []
    return {
        "name": case.name,
        "prompt_token_ids": case.prompt_token_ids,
        "decode_steps": case.decode_steps,
        "reference_path": str(case.reference_path),
        "command": list(command),
        "status": "pass" if matched else "fail",
        "schema": report.get("schema"),
        "claim": report.get("claim"),
        "sampled_token_ids": report.get("sampled_token_ids", []),
        "top_k_count": len(first_top_k),
        "hf_reference": hf_reference,
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
    print(f"wrote CLI suite report to {args.report}")


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
                    "prompt_token_ids": case.prompt_token_ids,
                    "decode_steps": case.decode_steps,
                    "reference_path": str(case.reference_path),
                    "command": command,
                    "status": "dry_run",
                }
            )
            continue
        result = run_case(args, case, command)
        results.append(result)

    print_table(results)
    status = "dry_run" if args.dry_run else "pass"
    if not args.dry_run and any(result["status"] != "pass" for result in results):
        status = "fail"
    write_report(args, results, status)
    return 0 if status in {"pass", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
