"""Exercise the build-only shell gate with fake tools; never call real Cargo/Metal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

GATE = Path(__file__).with_name("check_gemma4_candidate_delivery.sh")

FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "uname":
    print(os.environ.get("FAKE_OS", "Darwin") if args == ["-s"] else "arm64")
elif name == "cargo":
    if args[0] == "fmt" and os.environ.get("FAKE_FAIL") == "fmt":
        print("format failure", file=sys.stderr); sys.exit(7)
    if args[0] == "test":
        count = 0 if os.environ.get("FAKE_FAIL") == "zero-tests" else 4
        print(f"test result: ok. {count} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
    if args[0] == "build" and "--bin" in args:
        binary = args[args.index("--bin") + 1]
        folder = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "aarch64-apple-darwin/release"
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / binary
        if binary == "rvllm-metal-research-source":
            file.write_text('#!/bin/sh\nprintf "// compile-only fixture: %s %s\\n" "$1" "$2"\n')
        else:
            file.write_text('#!/bin/sh\necho "INFERENCE MUST NOT RUN" >&2\nexit 99\n')
        file.chmod(0o755)
elif name == "xcrun":
    if "metal" in args and "-c" in args:
        if os.environ.get("FAKE_FAIL") == "metal":
            print("Metal compile failure", file=sys.stderr); sys.exit(9)
        pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"fake AIR")
    elif "metallib" in args:
        pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"fake metallib")
    elif "--find" in args:
        print("/fake/metal")
else:
    print("fake tool version")
'''


class DeliveryGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "repo/v3"
        tools = self.workspace / "tools"
        tools.mkdir(parents=True)
        (self.workspace / "Cargo.toml").write_text("[workspace]\n")
        self.script = tools / GATE.name
        shutil.copyfile(GATE, self.script)
        fake = self.root / "bin"
        fake.mkdir()
        for name in ("uname", "cargo", "rustc", "rustfmt", "xcrun"):
            tool = fake / name
            tool.write_text(FAKE_TOOL.replace("#!/usr/bin/env python3", f"#!{sys.executable} -S", 1))
            tool.chmod(0o755)
        self.trace = self.root / "fake-tools.jsonl"
        self.out = self.root / "fresh-output"
        self.target = self.root / "target"
        self.env = {**os.environ, "PATH": str(fake) + os.pathsep + os.environ["PATH"],
                    "FAKE_LOG": str(self.trace)}

    def run_gate(self, failure=None):
        if failure:
            self.env["FAKE_FAIL"] = failure
        return subprocess.run(["bash", str(self.script), str(self.out), str(self.target)],
                              env=self.env, capture_output=True, text=True, check=False)

    def calls(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_compile_gate_exports_all_four_selections_in_both_dtypes(self):
        run = self.run_gate()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(list(self.out.glob("*.metal"))), 8)
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 8)
        self.assertEqual((self.out / "status.txt").read_text(), "compiled-only; no accelerator acceptance\n")
        commands = (self.out / "commands.txt").read_text()
        self.assertNotIn("--ignored", commands)
        self.assertNotIn("--prepare-ane-cache", commands)
        self.assertNotIn("--inspect-ane-cache", commands)
        # The inference executable is built, but every executed program is a
        # host tool/test or the explicitly host-only source exporter.
        for line in commands.splitlines():
            self.assertFalse(line.startswith(str(self.target / "aarch64-apple-darwin/release/rvllm_disaggregated_infer")))
        self.assertTrue(any(c[:2] == ["cargo", "fmt"] for c in self.calls()))
        self.assertTrue((self.out / "SHA256SUMS").read_text())

    def test_format_failure_stops_before_any_shader_compile(self):
        run = self.run_gate("fmt")
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(any("-c" in c for c in self.calls() if c[0] == "xcrun"))
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertIn("format failure", (self.out / "format.stderr").read_text())

    def test_compile_failure_is_preserved_and_not_retried(self):
        run = self.run_gate("metal")
        self.assertNotEqual(run.returncode, 0)
        compile_calls = [c for c in self.calls() if c[0] == "xcrun" and "-c" in c]
        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(len(list(self.out.glob("*.metal"))), 1)
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 0)
        self.assertIn("Metal compile failure", (self.out / "bf16-off-compile.stderr").read_text())

    def test_existing_output_is_not_overwritten(self):
        self.out.mkdir()
        marker = self.out / "failed-evidence.txt"
        marker.write_text("keep this\n")
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(marker.read_text(), "keep this\n")
        self.assertFalse((self.out / "status.txt").exists())

    def test_successful_zero_test_filter_is_rejected(self):
        run = self.run_gate("zero-tests")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("no passing tests recorded", run.stderr)
        self.assertFalse(any(c[:2] == ["cargo", "build"] for c in self.calls()))

    def test_non_macos_environment_never_reaches_cargo(self):
        self.env["FAKE_OS"] = "Linux"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(any(c[0] == "cargo" for c in self.calls()))


if __name__ == "__main__":
    unittest.main()
