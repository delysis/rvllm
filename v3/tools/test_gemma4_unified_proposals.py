"""Offline proposal mutation checks; never admits or executes a queue job."""
import copy
import json
from pathlib import Path
import unittest
import gemma4_unified_proposals as proposal


class UnifiedProposalTests(unittest.TestCase):
    def setUp(self):
        self.values = [json.loads(p.read_text()) for p in sorted(proposal.ROOT.glob('*.json'))]

    def test_five_unique_additions_and_no_second_down4_or_packed_route(self):
        names = proposal.validate_all(self.values)
        self.assertEqual(len(names), 5)
        self.assertNotIn('ane-int8-ffn-down4', names)
        self.assertNotIn('ane-int8-ffn-packed32', names)

    def test_missing_duplicate_and_unknown_candidates_fail(self):
        for values in [[], self.values[:-1], self.values + self.values[:1]]:
            with self.assertRaises(ValueError): proposal.validate_all(values)
        value = copy.deepcopy(self.values)
        value[0]['candidate'] = 'auto'
        with self.assertRaises(ValueError): proposal.validate_all(value)

    def test_changed_control_bounds_or_work_count_never_passes_as_original(self):
        for key, value in [('control', 'best-so-far'), ('compile_budget_in_inference', 1),
                           ('drift_fraction', 0.2), ('timing', {'requests': 4}),
                           ('shape', {'hidden': 5376}), ('status', 'admitted')]:
            values = copy.deepcopy(self.values)
            values[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): proposal.validate_all(values)

    def test_local_pins_and_strata_must_be_filled_by_separate_admission(self):
        for key in ['pins', 'power_stratum']:
            values = copy.deepcopy(self.values)
            inner = next(iter(values[0][key]))
            values[0][key][inner] = 'invented'
            with self.assertRaises(ValueError): proposal.validate_all(values)

    def test_rms_remains_blocked_without_an_implemented_native_adapter(self):
        values = copy.deepcopy(self.values)
        rms = next(x for x in values if x['candidate'] == 'metal-rmsnorm-simd256')
        self.assertEqual(rms['component']['state'], 'blocked-missing-direct-native-oracle')
        rms['component']['state'] = 'passed'
        with self.assertRaises(ValueError): proposal.validate_all(values)

    def test_json_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for data in ['{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}']:
            with self.assertRaises(ValueError): proposal.read_json(data)


if __name__ == '__main__': unittest.main()
