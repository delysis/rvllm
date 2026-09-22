"""Staging-contract tests; this module never launches a process or a queue."""
from pathlib import Path
import copy, importlib.util, json, unittest
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'tools/gemma4_wide_proposals.py'
SPEC_DIR=ROOT/'reports/proposals/rvllm-gemma4-wide-226dbaad-20260922'

class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('wide_proposals',MODULE)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
        cls.cases=[json.loads(p.read_text()) for p in sorted(SPEC_DIR.glob('*.json'))]

    def test_seven_explicit_candidates_are_staged_not_runnable(self):
        self.assertEqual(len(self.cases),7)
        self.module.validate_catalog(self.cases)
        for case in self.cases:
            self.assertFalse(case['admitted_to_queue'])
            self.assertIsNone(case['local_pins']['executable_sha256'])
            self.assertIsNone(case['local_pins']['reference_sha256'])
            self.assertIsNone(case['stratum']['power_source'])
            self.assertEqual(case['timing']['baseline_drift_limit'],0.05)

    def test_duplicate_or_unknown_name_never_silently_overwrites(self):
        with self.assertRaises(ValueError):self.module.validate_catalog(self.cases+[self.cases[0]])
        x=copy.deepcopy(self.cases);x[0]['candidate']='auto'
        with self.assertRaises(ValueError):self.module.validate_catalog(x)

    def test_unsupported_or_fabricated_live_pins_are_rejected(self):
        for field in ['executable_sha256','reference_sha256','model_config_sha256']:
            x=copy.deepcopy(self.cases);x[0]['local_pins'][field]='a'*64
            with self.assertRaises(ValueError):self.module.validate_catalog(x)
        x=copy.deepcopy(self.cases);x[0]['admitted_to_queue']=True
        with self.assertRaises(ValueError):self.module.validate_catalog(x)

    def test_packed32_cannot_become_an_inference_route_by_metadata(self):
        x=copy.deepcopy(self.cases);c=next(c for c in x if c['candidate']=='ane-int8-ffn-packed32')
        c['selector']=['--ane-weights','invented-packed32-cached']
        with self.assertRaises(ValueError):self.module.validate_catalog(x)
        c['selector']=None;c['status']='ready-for-local-qualification'
        with self.assertRaises(ValueError):self.module.validate_catalog(x)

    def test_wrong_controls_and_timing_normalization_are_rejected(self):
        x=copy.deepcopy(self.cases);c=next(c for c in x if c['candidate']=='ane-int8-ffn-down4')
        c['control']='static-all-cached'
        with self.assertRaises(ValueError):self.module.validate_catalog(x)
        x=copy.deepcopy(self.cases);x[0]['timing']['normalize_accelerator_by_cpu_cycles']=True
        with self.assertRaises(ValueError):self.module.validate_catalog(x)

    def test_compiler_recipe_delegates_once_to_the_complete_native_gate(self):
        steps = self.module.compile_recipe('/checkout/v3', '/cache', '/fresh-output')
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['argv'], ['bash',
            '/checkout/v3/tools/check_gemma4_candidate_delivery.sh', '/fresh-output', '/cache'])
        self.assertEqual(steps[0]['expected_compile_link_arms'], 14)
        self.assertEqual(steps[0]['execution_authority'], 'local-owner-only')
        with self.assertRaises(ValueError):
            self.module.compile_recipe('relative', '/cache', '/output')

    def test_integrated_gate_preserves_per_export_and_final_identity_checks(self):
        gate = (ROOT/'tools/check_gemma4_candidate_delivery.sh').read_text()
        before = gate.index('run "$stem-exporter-unchanged"')
        exported = gate.index('run "$stem-export"')
        self.assertLess(before, exported)
        self.assertIn('run built-cli-hashes', gate)
        self.assertIn('run built-exporter-hashes', gate)
        self.assertIn('run format-source-unchanged-final', gate)
        self.assertIn('run artifact-unchanged', gate)
        self.assertIn('run shader-source-hashes', gate)
        self.assertIn('run "$stem-shader-source-unchanged"', gate)
        self.assertIn('run shader-source-unchanged-final', gate)
        self.assertNotIn('--ignored', gate)
        self.assertNotIn('--prepare-ane-cache', gate)
        self.assertNotIn('--prefill-only', gate)


class ProposalMutationTests(unittest.TestCase):
    setUpClass = classmethod(ProposalTests.setUpClass.__func__)
    def test_work_counts_and_process_exemptions_cannot_be_relaxed(self):
        for field, value in [('outputs_per_request',1),('ane_steps_per_request',0),
                             ('prompt_tokens',6),('ane_evaluations_per_full_abba',0),
                             ('independent_confirmation_required',False)]:
            x=copy.deepcopy(self.cases); x[0]['timing'][field]=value
            with self.assertRaises(ValueError): self.module.validate_catalog(x)
        x=copy.deepcopy(self.cases);x[0]['stratum']['idle_process_exemptions']=[1038]
        with self.assertRaises(ValueError): self.module.validate_catalog(x)

    def test_json_duplicate_keys_are_not_an_ambiguous_pin(self):
        with self.assertRaises(ValueError):
            self.module.read_proposal_text('{"candidate":"off","candidate":"metal-attn-q4"}')

if __name__ == '__main__':
    unittest.main()
