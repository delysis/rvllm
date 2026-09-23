#!/usr/bin/env python3
"""Run reviewed public host Rust suites and export MSL; never run an accelerator.

Dependencies must already be present (CI fetches Cargo.lock explicitly first).
The native delivery gate is separate: this program cannot compile Metal, load
models, prepare caches, launch queues, run ignored tests, or prove performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import gemma4_catalog as catalog

# Labels, packages and filters are review boundaries, not arbitrary JSON argv.
BOUNDARIES = {
    "metal-catalog": ("rvllm-apple-metal", "research_catalog::tests::"),
    "metal-projection": ("rvllm-apple-metal", "research_projection::tests::"),
    "metal-research": ("rvllm-apple-metal", "research::"),
    "metal-dispatch": ("rvllm-apple-metal", "research_evidence::tests::"),
    "metal-next": ("rvllm-apple-metal", "research_next::"),
    "ane-candidates": ("rvllm-apple", "ane_int8_candidates::tests::"),
    "kv-layout": ("rvllm-apple", "ane_attention_layout::blocked32_tests::"),
    "ane-transposes": ("rvllm-apple", "ane_attention_layout::transpose_tests::"),
    "head-ranking": ("rvllm-runtime", "gemma_head_ranking::tests::"),
}
EXPORTS = catalog.exports(catalog.load())

COMMON = ["--offline", "--locked", "--release", "-j", "2", "--no-default-features"]


def load_suites(path: Path) -> list[dict]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate test inventory key: {key}")
            result[key] = value
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    if set(data) != {"schema", "suites"} or data["schema"] != 1:
        raise ValueError("unrecognized host test inventory")
    suites = data["suites"]
    if not isinstance(suites, list) or len(suites) != len(BOUNDARIES):
        raise ValueError("all reviewed suites are required")
    labels, seen = set(), set()
    for suite in suites:
        if not isinstance(suite, dict) or set(suite) != {"label", "package", "filter", "tests"}:
            raise ValueError("invalid host suite fields")
        label = suite["label"]
        if label in labels or BOUNDARIES.get(label) != (suite["package"], suite["filter"]):
            raise ValueError("unreviewed or duplicate host suite")
        labels.add(label)
        if not isinstance(suite["tests"], list) or not suite["tests"]:
            raise ValueError("empty expected test set")
        for name in suite["tests"]:
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:]*", name)
                    or not name.startswith(suite["filter"]) or name in seen):
                raise ValueError("unreviewed, malformed or duplicate test name")
            seen.add(name)
    return suites


def test_command(suite: dict) -> list[str]:
    return ["cargo", "test", *COMMON, "-p", suite["package"], "--lib", suite["filter"],
            "--", "--nocapture", "--test-threads=1"]


def check_test_output(text: str, suite: dict) -> None:
    # A positive summary alone previously failed to detect missing test modules.
    # Require every reviewed fully-qualified name exactly once, and no extras.
    passed = re.findall(r"^test ([A-Za-z_][A-Za-z0-9_:]*) \.\.\. ok$", text, re.M)
    expected = suite["tests"]
    summaries = re.findall(r"^test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;", text, re.M)
    if (len(passed) != len(expected) or set(passed) != set(expected)
            or summaries != [(str(len(expected)), "0")]):
        raise ValueError(f"{suite['label']}: exact expected host tests did not all pass")


def check_export(source: str, candidate: str) -> None:
    catalog.check_source(catalog.load(), candidate, source)


def run_checks(workspace: Path, output: Path) -> None:
    workspace = workspace.resolve(strict=True)
    suites = load_suites(workspace / "tools/gemma4_candidate_host_tests.json")
    reviewed = catalog.load(workspace / "tools/gemma4_metal_catalog.json")
    export_map = catalog.exports(reviewed)
    output.mkdir(parents=False, exist_ok=False)
    output = output.resolve(strict=True)
    result = {"status": "incomplete", "metal_compiled": False, "device_qualified": False,
              "rust_tests": 0, "exports": 0}
    env = {**os.environ, "CARGO_NET_OFFLINE": "true", "CARGO_TERM_COLOR": "never"}
    # Explicit CLI flags disable Rust harness parallelism/color where relevant.
    # Native feature selection is never accepted through the inventory.
    def record_result():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    record_result()

    def run(label: str, argv: list[str]) -> str:
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"label": label, "cwd": str(workspace), "argv": argv,
                                  "environment": {k: env[k] for k in ("CARGO_NET_OFFLINE", "CARGO_TERM_COLOR")}}) + "\n")
        process = subprocess.run(argv, cwd=workspace, env=env, capture_output=True, check=False)
        (output / f"{label}.stdout").write_bytes(process.stdout)
        (output / f"{label}.stderr").write_bytes(process.stderr)
        with (output / "exit-codes.jsonl").open("a") as log:
            log.write(json.dumps({"label": label, "exit_code": process.returncode}) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} failed with exit {process.returncode}; outputs preserved")
        return process.stdout.decode("utf-8", errors="strict")

    try:
        run("rustc-version", ["rustc", "-Vv"])
        run("cargo-version", ["cargo", "-V"])
        for suite in suites:
            check_test_output(run(suite["label"], test_command(suite)), suite)
            result["rust_tests"] += len(suite["tests"])
        emitted = run("runtime-catalog", ["cargo", "run", *COMMON, "-p", "rvllm-apple-metal", "--bin",
                                         "rvllm-metal-research-source", "--", "--catalog"])
        catalog.verify_exported(reviewed, catalog.decode(emitted))
        hashes = {"runtime-catalog.stdout": hashlib.sha256(emitted.encode()).hexdigest()}
        for dtype in ("bf16", "f16"):
            for candidate in export_map:
                label = f"{dtype}-{candidate}-export"
                source = run(label, ["cargo", "run", *COMMON, "-p", "rvllm-apple-metal", "--bin",
                                     "rvllm-metal-research-source", "--", dtype, candidate])
                catalog.check_source(reviewed, candidate, source)
                hashes[f"{label}.stdout"] = hashlib.sha256((output / f"{label}.stdout").read_bytes()).hexdigest()
                result["exports"] += 1
        (output / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        result["status"] = "host-tests-and-source-export-only"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        record_result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh existing-parent receipt directory")
    args = parser.parse_args()
    try:
        run_checks(Path(__file__).resolve().parents[1], args.output)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
