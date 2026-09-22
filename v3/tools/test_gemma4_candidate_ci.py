"""Host-CI contract tests. Child processes here are mocked, never accelerators."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("candidate_ci", TOOLS / "run_gemma4_candidate_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class CandidateCiTests(unittest.TestCase):
    def setUp(self):
        self.suites = ci.load_suites(TOOLS / "gemma4_candidate_host_tests.json")

    @staticmethod
    def output(suite):
        lines = [f"test {name} ... ok" for name in suite["tests"]]
        return "\n".join(lines) + f"\ntest result: ok. {len(lines)} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"

    def test_suites_cover_actual_new_rust_modules_not_just_policy_headings(self):
        names = {name for suite in self.suites for name in suite["tests"]}
        self.assertGreaterEqual(len(names), 40)
        for fragment in ["research::short_mma_tests::", "research_next::temporal_tests::",
                         "ane_int8_candidates::tests::next_tests::",
                         "ane_attention_layout::blocked32_tests::",
                         "ane_attention_layout::transpose_tests::",
                         "gemma_head_ranking::tests::"]:
            self.assertTrue(any(name.startswith(fragment) for name in names), fragment)
        for suite in self.suites:
            command = ci.test_command(suite)
            for flag in ["--offline", "--locked", "--release", "--no-default-features", "--lib"]:
                self.assertIn(flag, command)
            self.assertNotIn("--features", command)
            self.assertNotIn("--ignored", command)
            self.assertNotIn("--include-ignored", command)
            self.assertNotIn("macos-private-ane-research", command)
            ci.check_test_output(self.output(suite), suite)

    def test_positive_total_with_wrong_test_names_is_rejected(self):
        suite = self.suites[0]
        mutant = self.output(suite).replace(suite["tests"][0], "unreviewed::test")
        with self.assertRaises(ValueError):
            ci.check_test_output(mutant, suite)
        with self.assertRaises(ValueError):
            ci.check_test_output("test result: ok. 99 passed; 0 failed; 0 ignored;", suite)

    def test_duplicate_missing_ignored_or_zero_tests_never_count_as_coverage(self):
        suite = self.suites[0]
        good = self.output(suite)
        for text in [good + f"test {suite['tests'][0]} ... ok\n",
                     good.replace(f"test {suite['tests'][0]} ... ok\n", ""),
                     good.replace(" ... ok", " ... ignored", 1),
                     "test result: ok. 0 passed; 0 failed; 0 ignored;",
                     good.replace("0 ignored", "1 ignored")]:
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                ci.check_test_output(text, suite)

    def test_compilation_and_source_export_are_not_hardware_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "v3"
            (workspace / "tools").mkdir(parents=True)
            inventory = workspace / "tools/gemma4_candidate_host_tests.json"
            inventory.write_bytes((TOOLS / inventory.name).read_bytes())
            commands = []

            def fake_run(argv, **kwargs):
                commands.append(argv)
                if argv[:2] == ["cargo", "test"]:
                    suite = next(s for s in self.suites if s["filter"] in argv)
                    return subprocess.CompletedProcess(argv, 0, self.output(suite).encode(), b"")
                if argv[:2] == ["cargo", "run"]:
                    candidate = argv[-1]
                    source = "\n".join("kernel void " + n + "() {}" for n in ci.EXPORTS[candidate])
                    return subprocess.CompletedProcess(argv, 0, ("// fixture\n" + source).encode(), b"")
                return subprocess.CompletedProcess(argv, 0, b"mock tool version", b"")

            output = root / "output"
            with patch.object(ci.subprocess, "run", side_effect=fake_run):
                ci.run_checks(workspace, output)
            self.assertEqual(len([c for c in commands if c[:2] == ["cargo", "run"]]), 14)
            self.assertEqual(len([c for c in commands if c[:2] == ["cargo", "test"]]), len(self.suites))
            self.assertTrue(all("rvllm_disaggregated_infer" not in c for c in commands))
            self.assertTrue(all(c[0] not in ["xcrun", "pmset"] for c in commands))
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["status"], "host-tests-and-source-export-only")
            self.assertFalse(receipt["metal_compiled"])
            self.assertFalse(receipt["device_qualified"])
            self.assertEqual(len(list(output.glob("*-export.stdout"))), 14)
            self.assertTrue((output / "source-sha256.json").exists())
            with patch.object(ci.subprocess, "run") as no_process:
                with self.assertRaises(FileExistsError):
                    ci.run_checks(workspace, output)
                no_process.assert_not_called()

    def test_nonzero_child_exit_is_retained_and_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v3/tools").mkdir(parents=True)
            (root / "v3/tools/gemma4_candidate_host_tests.json").write_bytes(
                (TOOLS / "gemma4_candidate_host_tests.json").read_bytes())
            def failure(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 7, b"partial output", b"intentional host failure")
            with patch.object(ci.subprocess, "run", side_effect=failure) as calls:
                with self.assertRaises(RuntimeError):
                    ci.run_checks(root / "v3", root / "out")
                self.assertEqual(calls.call_count, 1)
            result = json.loads((root / "out/result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("intentional host failure", (root / "out/rustc-version.stderr").read_text())

    def test_inventory_cannot_inject_a_private_test_or_erase_a_suite(self):
        original = json.loads((TOOLS / "gemma4_candidate_host_tests.json").read_text())
        for kind in ["empty", "private", "duplicate", "prefix"]:
            value = json.loads(json.dumps(original))
            if kind == "empty":
                value["suites"] = []
            elif kind == "private":
                value["suites"][0]["package"] = "rvllm-apple-ane-sys"
            elif kind == "duplicate":
                value["suites"].append(value["suites"][0])
            else:
                value["suites"][0]["filter"] = "ane_live"
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "tests.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    ci.load_suites(path)


if __name__ == "__main__":
    unittest.main()
