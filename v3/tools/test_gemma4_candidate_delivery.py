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
MANIFEST = GATE.with_name("gemma4_candidate_rustfmt.paths")
RUSTFMT = shutil.which("rustfmt")


FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "uname":
    print(os.environ.get("FAKE_OS", "Darwin") if args == ["-s"] else os.environ.get("FAKE_ARCH", "arm64"))
elif name == "cargo":
    if args[0] == "fmt" and (os.environ.get("FAKE_FAIL") == "fmt" or os.environ.get("FAKE_WORKSPACE_DRIFT")):
        print("format failure", file=sys.stderr); sys.exit(7)
    if args[0] == "test":
        mutate = os.environ.get("FAKE_SOURCE_EDIT_DURING_TEST")
        if mutate:
            pathlib.Path(mutate).write_text("// edit after formatting\n")
        count = 0 if (os.environ.get("FAKE_FAIL") == "zero-tests" or args[-1] == os.environ.get("FAKE_ZERO_FILTER")) else 4
        print(f"test result: ok. {count} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
        if os.environ.get("FAKE_FAIL") == "test-exit":
            sys.exit(8)
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
        if binary == "rvllm-metal-research-source" and os.environ.get("FAKE_REPLACE_CLI_AT_EXPORTER_BUILD"):
            (folder / "rvllm_disaggregated_infer").write_text("different CLI after its build\n")
elif name == "rustfmt" and args != ["--version"]:
    if args != ["--edition", "2021", "--emit", "stdout"]:
        print("unexpected formatter invocation (must be nonrecursive stdin)", file=sys.stderr)
        sys.exit(97)
    source = sys.stdin.buffer.read()
    with open(os.environ["FAKE_FORMAT_LOG"], "a") as log:
        log.write(json.dumps({"cwd": os.getcwd(), "args": args, "stdin": source.decode()}) + "\n")
    if os.environ.get("FAKE_FAIL") == "fmt":
        print("format failure", file=sys.stderr); sys.exit(7)
    if os.environ.get("FAKE_FAIL") == "empty-formatter":
        sys.exit(0)
    # Like rustfmt --emit stdout, changed output still has a zero exit status.
    sys.stdout.buffer.write(source.replace(b"pub fn packet_owned( ) { }", b"pub fn packet_owned() {}"))
    mutate = os.environ.get("FAKE_MUTATE_SOURCE")
    if mutate and "// mutate earlier source now" in source.decode():
        pathlib.Path(mutate).write_text("// simulated concurrent edit\n")
elif name == "xcrun":
    if "metal" in args and "-c" in args:
        if os.environ.get("FAKE_FAIL") == "metal":
            print("Metal compile failure", file=sys.stderr); sys.exit(9)
        source = pathlib.Path(args[args.index("-c") + 1])
        replace_at = os.environ.get("FAKE_REPLACE_EXPORTER_AT")
        if source.stem == replace_at:
            exporter = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "aarch64-apple-darwin/release/rvllm-metal-research-source"
            exporter.write_text("#!/bin/sh\nprintf '// replacement exporter\\n'\n")
        if source.stem == os.environ.get("FAKE_REPLACE_CLI_AT"):
            cli = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "aarch64-apple-darwin/release/rvllm_disaggregated_infer"
            cli.write_text("different CLI during shader compilation\n")
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
        self.workspace = self.root / "repo with spaces/v3"
        tools = self.workspace / "tools"
        tools.mkdir(parents=True)
        (self.workspace / "Cargo.toml").write_text("[workspace]\n")
        self.script = tools / GATE.name
        shutil.copyfile(GATE, self.script)
        self.manifest = tools / MANIFEST.name
        shutil.copyfile(MANIFEST, self.manifest)
        self.owned = [line for line in MANIFEST.read_text().splitlines()
                      if line and not line.startswith("#")]
        for relative in self.owned:
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"// {relative}\npub fn packet_owned() {{}}\n")
        shaders = self.workspace / "crates/rvllm-apple-metal/src/research_shaders"
        shaders.mkdir(exist_ok=True)
        for name in ["short_mma16x64", "rounded_gate32", "gqa_kv8",
                     "mma32_prefetch", "attn_q4", "rms_simd32"]:
            (shaders / (name + ".metal")).write_text("// source fixture: " + name + "\n")
        fake = self.root / "bin"
        fake.mkdir()
        for name in ("uname", "cargo", "rustc", "rustfmt", "xcrun"):
            tool = fake / name
            tool.write_text(FAKE_TOOL.replace("#!/usr/bin/env python3", f"#!{sys.executable} -S", 1))
            tool.chmod(0o755)
        self.trace = self.root / "fake-tools.jsonl"
        self.format_trace = self.root / "fake-format.jsonl"
        self.out = self.root / "fresh-output"
        self.target = self.root / "target"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_")}
        self.env.update(PATH=str(fake) + os.pathsep + os.environ["PATH"],
                        FAKE_LOG=str(self.trace), FAKE_FORMAT_LOG=str(self.format_trace))

    def run_gate(self, failure=None):
        if failure:
            self.env["FAKE_FAIL"] = failure
        return subprocess.run(["bash", str(self.script), str(self.out), str(self.target)],
                              env=self.env, capture_output=True, text=True, check=False)

    def calls(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def format_calls(self):
        if not self.format_trace.exists():
            return []
        return [json.loads(line) for line in self.format_trace.read_text().splitlines()]

    def assert_no_targeted_work(self):
        self.assertFalse(any(c[:2] in (["cargo", "test"], ["cargo", "build"])
                             for c in self.calls()))
        self.assertFalse(any(c[0] == "xcrun" and "-c" in c for c in self.calls()))
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")

    def test_compile_gate_exports_all_seven_selections_in_both_dtypes(self):
        run = self.run_gate()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        stems = {f"{dtype}-{candidate}" for dtype in ("bf16", "f16")
                 for candidate in ("off", "metal-short-mma16x64", "metal-rounded-gate32", "metal-gqa-kv8", "metal-mma32-prefetch", "metal-attn-q4", "metal-rms-simd32")}
        self.assertEqual({p.stem for p in self.out.glob("*.metal")}, stems)
        self.assertEqual({p.stem for p in self.out.glob("*.metallib")}, stems)
        compiles = [c for c in self.calls() if c[0] == "xcrun" and "-c" in c]
        links = [c for c in self.calls() if c[0] == "xcrun" and "metallib" in c]
        self.assertEqual(len(compiles), 14)
        self.assertEqual(len(links), 14)
        self.assertTrue(all("-std=metal3.1" in c for c in compiles))
        self.assertEqual((self.out / "status.txt").read_text(), "compiled-only; no accelerator acceptance\n")
        commands = (self.out / "commands.txt").read_text()
        self.assertNotIn("--ignored", commands)
        self.assertNotIn("--prepare-ane-cache", commands)
        self.assertNotIn("--inspect-ane-cache", commands)
        # The inference executable is built, but every executed program is a
        # host tool/test or the explicitly host-only source exporter.
        for line in commands.splitlines():
            self.assertFalse(line.startswith(str(self.target / "aarch64-apple-darwin/release/rvllm_disaggregated_infer")))
        self.assertFalse(any(c[:2] == ["cargo", "fmt"] for c in self.calls()))
        self.assertEqual(len(self.format_calls()), 24)
        for call in self.calls():
            if call[0] == "cargo" and call[1] in ("test", "build"):
                for flag in ("--offline", "--locked", "--release", "aarch64-apple-darwin"):
                    self.assertIn(flag, call)
        self.assertEqual((self.out / "SHA256SUMS").read_text(),
                         (self.out / "built-exporter-hashes.stdout").read_text()
                         + (self.out / "built-cli-hashes.stdout").read_text()
                         + (self.out / "metal-artifact-hashes.stdout").read_text())
        codes = (self.out / "exit-codes.tsv").read_text()
        self.assertIn("artifact-unchanged\t0\n", codes)
        self.assertIn("format-source-unchanged-final\t0\n", codes)
        commands = (self.out / "commands.txt").read_text()
        self.assertIn("ane_attention_layout::blocked32_tests", commands)

    def test_unlisted_workspace_drift_does_not_block_packet_gate(self):
        unlisted = self.workspace / "crates/unrelated/src/lib.rs"
        unlisted.parent.mkdir(parents=True)
        unlisted.write_text("pub fn unformatted( ){ }\n")
        before = unlisted.read_bytes()
        owner = self.workspace / "crates/rvllm-apple-metal/src/lib.rs"
        owner.write_text(f'#[path = "{unlisted}"]\nmod unlisted;\n\npub fn packet_owned() {{}}\n')
        owner_before = owner.read_bytes()
        self.env["FAKE_WORKSPACE_DRIFT"] = "1"
        run = self.run_gate()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(unlisted.read_bytes(), before)
        self.assertEqual(owner.read_bytes(), owner_before)
        self.assertFalse(any(c[:2] == ["cargo", "fmt"] for c in self.calls()))
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 14)
        checked = (self.out / "format-manifest.stdout").read_text().splitlines()
        self.assertEqual(checked, self.owned)
        self.assertNotIn(str(unlisted.relative_to(self.workspace)), checked)
        self.assertEqual((self.out / "format-scope.txt").read_text(),
                         "packet-owned-files-only; unlisted source NOT checked\n")
        calls = self.format_calls()
        self.assertEqual([c["cwd"] for c in calls],
                         [str((self.workspace / name).parent.resolve()) for name in self.owned])
        self.assertEqual([c["stdin"] for c in calls],
                         [(self.workspace / name).read_text() for name in self.owned])

    def test_format_failure_stops_before_any_shader_compile(self):
        run = self.run_gate("fmt")
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(any("-c" in c for c in self.calls() if c[0] == "xcrun"))
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertIn("format failure", (self.out / "format-0001.stderr").read_text())

    def test_owned_drift_is_rejected_even_when_rustfmt_exits_zero(self):
        path = self.workspace / self.owned[0]
        dirty = "pub fn packet_owned( ) { }\n"
        path.write_text(dirty)
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(path.read_text(), dirty)  # Check-only, never a repair.
        self.assertEqual((self.out / "format-0001.stdout").read_text(),
                         "pub fn packet_owned() {}\n")
        codes = (self.out / "exit-codes.tsv").read_text()
        self.assertIn("format-0001\t0\n", codes)
        self.assertIn("format-0001-compare\t1\n", codes)
        self.assert_no_targeted_work()

    def test_empty_formatter_output_cannot_pass(self):
        run = self.run_gate("empty-formatter")
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual((self.out / "format-0001.stdout").read_bytes(), b"")
        self.assert_no_targeted_work()

    def test_invalid_manifests_fail_before_any_source_is_formatted(self):
        original = self.manifest.read_text()
        cases = {
            "empty": "# no paths\n",
            "duplicate": original + self.owned[0] + "\n",
            "missing": original + "crates/missing/src/lib.rs\n",
            "absolute": original + str(self.workspace / self.owned[0]) + "\n",
            "traversal": original + "crates/../outside.rs\n",
            "glob": original + "crates/*/src/lib.rs\n",
            "dot": original + "crates/./src/lib.rs\n",
            "double-slash": original + "crates//src/lib.rs\n",
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                self.out = self.root / f"invalid-{name}"
                self.trace = self.root / f"invalid-{name}.jsonl"
                self.env["FAKE_LOG"] = str(self.trace)
                self.manifest.write_text(text)
                run = self.run_gate()
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(self.format_calls(), [])
                self.assert_no_targeted_work()
                self.assertTrue((self.out / "format-manifest.stderr").read_text())

    def test_missing_manifest_is_not_silently_replaced_with_workspace_scope(self):
        self.manifest.unlink()
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("packet Rust manifest required", run.stderr)
        self.assertEqual(self.format_calls(), [])
        self.assert_no_targeted_work()

    def test_symlinked_file_or_parent_cannot_expand_the_reviewed_scope(self):
        original = self.manifest.read_text()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "lib.rs").write_text("do not read or format this\n")
        for name, relative, target in [
            ("file", "crates/linked.rs", outside / "lib.rs"),
            ("parent", "crates/linked/lib.rs", outside),
        ]:
            with self.subTest(name=name):
                self.out = self.root / f"symlink-{name}"
                self.trace = self.root / f"symlink-{name}.jsonl"
                self.env["FAKE_LOG"] = str(self.trace)
                link = self.workspace / (relative if name == "file" else "crates/linked")
                link.symlink_to(target)
                self.manifest.write_text(original + relative + "\n")
                run = self.run_gate()
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("symlink in packet Rust path", run.stderr)
                self.assertEqual(self.format_calls(), [])
                self.assert_no_targeted_work()
        self.assertEqual((outside / "lib.rs").read_text(), "do not read or format this\n")

    def test_edit_to_an_already_checked_file_invalidates_format_stage(self):
        earlier = self.workspace / self.owned[0]
        later = self.workspace / self.owned[-1]
        later.write_text(later.read_text() + "// mutate earlier source now\n")
        self.env["FAKE_MUTATE_SOURCE"] = str(earlier)
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(len(self.format_calls()), 24)
        self.assertIn("format-source-unchanged", run.stderr)
        self.assert_no_targeted_work()

    def test_nonzero_test_exit_is_not_hidden_by_a_positive_summary(self):
        run = self.run_gate("test-exit")
        self.assertEqual(run.returncode, 8)
        self.assertIn("4 passed", (self.out / "metal-policy.stdout").read_text())
        self.assertFalse(any(c[:2] == ["cargo", "build"] for c in self.calls()))

    def test_prefill_zero_test_filter_still_blocks_compilation(self):
        self.env["FAKE_ZERO_FILTER"] = "prefill_screen::tests"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("no passing tests recorded for prefill-screen", run.stderr)
        self.assertEqual(len([c for c in self.calls() if c[:2] == ["cargo", "test"]]), 6)
        self.assertFalse(any(c[:2] == ["cargo", "build"] for c in self.calls()))

    def test_source_edit_after_formatting_cannot_receive_success(self):
        self.env["FAKE_SOURCE_EDIT_DURING_TEST"] = str(self.workspace / self.owned[0])
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0, "gate passed after a checked source changed")
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertIn("format-source-unchanged-final", run.stderr)
        self.assertIn("FAILED", (self.out / "format-source-unchanged-final.stdout").read_text())

    def test_cli_replacement_by_later_build_cannot_be_repinned_as_success(self):
        self.env["FAKE_REPLACE_CLI_AT_EXPORTER_BUILD"] = "1"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0, "gate accepted a CLI replaced after its build")
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertIn("built-cli-unchanged", run.stderr)
        self.assertTrue((self.out / "built-cli-hashes.stdout").read_text())
        self.assertFalse(list(self.out.glob("*.metal")))

    def test_exporter_replacement_between_arms_stops_before_next_export(self):
        self.env["FAKE_REPLACE_EXPORTER_AT"] = "bf16-off"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0, "gate accepted mixed source exporters")
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertEqual(len(list(self.out.glob("*.metal"))), 1)
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 1)
        self.assertIn("bf16-metal-short-mma16x64-exporter-unchanged", run.stderr)
        self.assertTrue((self.out / "built-exporter-hashes.stdout").read_text())

    def test_replacement_during_last_arm_does_not_get_a_fresh_binary_hash(self):
        self.env["FAKE_REPLACE_EXPORTER_AT"] = "f16-metal-rms-simd32"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0, "gate relabeled an exporter changed in the final arm")
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 14)
        self.assertIn("artifact-unchanged", run.stderr)
        self.assertIn((self.out / "built-exporter-hashes.stdout").read_text(),
                      (self.out / "artifact-hashes.stdout").read_text())

    def test_cli_replacement_during_metal_compile_is_rejected_at_final_check(self):
        self.env["FAKE_REPLACE_CLI_AT"] = "bf16-off"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0, "gate accepted a CLI replaced during Metal compilation")
        self.assertEqual((self.out / "status.txt").read_text(), "incomplete\n")
        self.assertEqual(len(list(self.out.glob("*.metallib"))), 14)
        self.assertIn("artifact-unchanged", run.stderr)
        self.assertIn((self.out / "built-cli-hashes.stdout").read_text(),
                      (self.out / "artifact-hashes.stdout").read_text())

    def test_final_zero_test_filter_still_blocks_compilation(self):
        self.env["FAKE_ZERO_FILTER"] = "gemma_ane_decode::tests"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("no passing tests recorded for decode-policy", run.stderr)
        self.assertEqual(len([c for c in self.calls() if c[:2] == ["cargo", "test"]]), 9)
        self.assertFalse(any(c[:2] == ["cargo", "build"] for c in self.calls()))

    def test_x86_macos_environment_never_reaches_cargo(self):
        self.env["FAKE_ARCH"] = "x86_64"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(any(c[0] == "cargo" for c in self.calls()))

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

    def test_shader_edit_after_source_pin_cannot_receive_success(self):
        shader = self.workspace / "crates/rvllm-apple-metal/src/research_shaders/attn_q4.metal"
        self.env["FAKE_SOURCE_EDIT_DURING_TEST"] = str(shader)
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertNotEqual((self.out / "status.txt").read_text(),
                            "compiled-only; no accelerator acceptance\n")
        self.assertTrue((self.out / "shader-source-hashes.stdout").is_file())

    def test_non_macos_environment_never_reaches_cargo(self):
        self.env["FAKE_OS"] = "Linux"
        run = self.run_gate()
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(any(c[0] == "cargo" for c in self.calls()))


class FormatterSemanticsTests(unittest.TestCase):
    @unittest.skipUnless(RUSTFMT, "rustfmt unavailable; real formatter semantics remain unverified")
    def test_real_stdin_is_nonrecursive_and_emits_changed_bytes_on_success(self):
        # Independent native check of the key upstream assumption, not a fake
        # formatter result. It is run automatically when rustfmt is installed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "unlisted.rs"
            child.write_text("not even valid Rust; must not be parsed\n")
            clean = b"mod unlisted;\n\npub fn packet_owned() {}\n"
            dirty = b"mod unlisted;\n\npub fn packet_owned( ) { }\n"
            for source in (clean, dirty):
                run = subprocess.run([RUSTFMT, "--edition", "2021", "--emit", "stdout"],
                                     input=source, cwd=root, capture_output=True, check=False)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(run.stdout, clean)
            self.assertEqual(child.read_text(), "not even valid Rust; must not be parsed\n")


if __name__ == "__main__":
    unittest.main()
