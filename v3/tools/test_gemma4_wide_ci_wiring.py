"""Repository integration assertions; not compiler or accelerator acceptance."""
from pathlib import Path
import json
import re
import unittest

V3 = Path(__file__).resolve().parents[1]


class WideCiWiringTests(unittest.TestCase):
    def test_delivery_gate_covers_all_seven_selections_and_new_rust_files(self):
        gate = (V3 / 'tools/check_gemma4_candidate_delivery.sh').read_text()
        selections = re.search(r'for candidate in (.*?); do', gate, re.S).group(1).replace('\\\n', '').split()
        self.assertEqual(selections, ['off', 'metal-short-mma16x64', 'metal-rounded-gate32',
                                    'metal-gqa-kv8', 'metal-mma32-prefetch', 'metal-attn-q4', 'metal-rms-simd32'])
        manifest = (V3 / 'tools/gemma4_candidate_rustfmt.paths').read_text()
        for path in ['crates/rvllm-apple-metal/src/research_next.rs',
                     'crates/rvllm-apple/src/ane_int8_next_tests.rs',
                     'crates/rvllm-apple/src/ane_packed32_layout.rs',
                     'crates/rvllm-runtime/src/gemma_head_ranking.rs']:
            self.assertIn(path, manifest.splitlines())
        self.assertIn('--lib research::', gate)
        self.assertNotIn('--lib research::tests', gate)
        for suite in ['research_next::', 'gemma_head_ranking::tests',
                      'ane_attention_layout::transpose_tests', 'gemma_ane_decode::tests']:
            self.assertIn(suite, gate)

    def test_workflow_is_host_only_regular_pr_ci_with_failure_artifacts(self):
        source = (V3.parent / '.github/workflows/gemma4-candidate-host.yml').read_text()
        for item in ['pull_request:', 'workflow_dispatch:', 'contents: read',
                     'persist-credentials: false', 'if: always()', 'cargo fetch --locked',
                     'run_gemma4_candidate_ci.py', 'run_gemma4_python_checks.py']:
            self.assertIn(item, source)
        for banned in ['pull_request_target:', 'self-hosted', '--ignored', '--include-ignored',
                       'macos-private-ane-research', 'rvllm_disaggregated_infer', 'pmset ',
                       'continue-on-error:', 'cargo publish', 'gh pr ', 'git push']:
            self.assertNotIn(banned, source)
        uses = re.findall(r'uses:\s*([^\s]+)', source)
        self.assertTrue(uses)
        self.assertTrue(all(re.fullmatch(r'[\w-]+/[\w-]+@[0-9a-f]{40}', entry) for entry in uses))

    def test_manifest_lists_present_test_functions_not_only_expected_totals(self):
        # Tests of the Rust implementation are actual checked-in #[test] items.
        # Namespace execution is independently checked from libtest output by
        # run_gemma4_candidate_ci; this assertion only detects stale names.
        data = json.loads((V3 / 'tools/gemma4_candidate_host_tests.json').read_text())
        source = '\n'.join(path.read_text() for path in (V3 / 'crates').rglob('*.rs'))
        for suite in data['suites']:
            for name in suite['tests']:
                self.assertRegex(source, r'#\[test\]\s*fn\s+' + re.escape(name.split('::')[-1]) + r'\s*\(')


if __name__ == '__main__':
    unittest.main()
