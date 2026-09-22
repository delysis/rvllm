"""Synthetic offline audit tests, not model/device evidence."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import gemma4_tensor_audit as audit

class TensorAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def pin(self, name, data):
        p = self.root / name
        p.write_bytes(data)
        return {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}

    def fixture(self, reference, candidate, dtype='f16', limits=None):
        policy = {'x': limits or {'max_abs': 0.0, 'relative_l2': 0.0, 'bitwise': True}}
        pin = self.pin('policy.json', json.dumps(policy).encode())
        r = {**self.pin('reference.bin', reference), 'dtype': dtype}
        c = {**self.pin('candidate.bin', candidate), 'dtype': dtype}
        n = len(reference) // audit.WIDTH[dtype]
        manifest = {'schema': audit.SCHEMA, 'policy': pin,
                    'pairs': [{'name': 'x', 'shape': [n], 'reference': r, 'candidate': c}]}
        return manifest

    def test_exact_half_bytes_pass_and_are_hashed(self):
        b = struct.pack('<4e', 0.0, 1.0, -2.0, 0.125)
        result = audit.run(self.fixture(b, b), self.root)
        self.assertTrue(result['supplied_tensor_policy_passed'])
        self.assertEqual(result['pairs'][0]['elements'], 4)
        self.assertFalse(result['hardware_qualification'])

    def test_same_argmax_does_not_hide_a_tensor_failure(self):
        r = struct.pack('<3f', 9.0, 2.0, 1.0)
        c = struct.pack('<3f', 9.0, -4.0, -8.0)
        result = audit.run(self.fixture(r, c, 'f32'), self.root)
        self.assertFalse(result['supplied_tensor_policy_passed'])
        self.assertEqual(result['pairs'][0]['max_abs'], 9.0)
        self.assertEqual(result['pairs'][0]['worst_index'], 2)

    def test_nonfinite_values_fail_even_when_identical(self):
        for value in (math.nan, math.inf, -math.inf):
            b = struct.pack('<2f', value, 1.0)
            r = audit.run(self.fixture(b, b, 'f32'), self.root)
            self.assertFalse(r['supplied_tensor_policy_passed'])
            self.assertEqual(r['pairs'][0]['nonfinite_pairs'], 1)
            json.dumps(r, allow_nan=False)

    def test_bf16_does_not_pass_through_fp16(self):
        b = struct.pack('<3H', 0x4780, 0xC780, 0x3F80)  # +/-65536, 1
        r = audit.run(self.fixture(b, b, 'bf16'), self.root)
        self.assertTrue(r['supplied_tensor_policy_passed'])
        self.assertEqual(r['pairs'][0]['reference_l2_squared'], 2 * 65536.0**2 + 1)

    def test_signed_zero_is_visible_to_bitwise_policy(self):
        r = audit.run(self.fixture(struct.pack('<e', 0.0), struct.pack('<e', -0.0)), self.root)
        self.assertFalse(r['supplied_tensor_policy_passed'])
        self.assertEqual(r['pairs'][0]['bit_mismatches'], 1)
        self.assertEqual(r['pairs'][0]['max_abs'], 0.0)

    def test_zero_reference_nonzero_error_fails_relative_gate(self):
        r = audit.run(self.fixture(struct.pack('<e', 0.0), struct.pack('<e', 1.0),
                     limits={'max_abs': 2.0, 'relative_l2': 10.0, 'bitwise': False}), self.root)
        self.assertFalse(r['supplied_tensor_policy_passed'])
        self.assertIsNone(r['pairs'][0]['relative_l2'])

    def test_both_numeric_gates_must_pass(self):
        b = struct.pack('<2f', 1.0, 1.0)
        c = struct.pack('<2f', 1.125, 1.0)
        for limits in ({'max_abs': 0.1, 'relative_l2': 1.0, 'bitwise': False},
                       {'max_abs': 1.0, 'relative_l2': 0.01, 'bitwise': False}):
            self.assertFalse(audit.run(self.fixture(b, c, 'f32', limits), self.root)['supplied_tensor_policy_passed'])

    def test_hash_mismatch_preserves_failed_metrics(self):
        b = struct.pack('<e', 1.0)
        m = self.fixture(b, b)
        m['pairs'][0]['candidate']['sha256'] = '0' * 64
        r = audit.run(m, self.root)
        self.assertFalse(r['supplied_tensor_policy_passed'])
        self.assertFalse(r['pairs'][0]['candidate_hash_matches'])

    def test_policy_pin_and_nonfinite_thresholds_fail_closed(self):
        b = struct.pack('<e', 1.0)
        m = self.fixture(b, b)
        m['policy']['sha256'] = '0' * 64
        with self.assertRaises(audit.AuditError): audit.run(m, self.root)
        for threshold in (math.nan, math.inf, -1.0, True):
            m = self.fixture(b, b, limits={'max_abs': threshold, 'relative_l2': 0.0, 'bitwise': False})
            with self.assertRaises(audit.AuditError): audit.run(m, self.root)

    def test_length_shape_and_unknown_dtype_are_rejected(self):
        b = struct.pack('<2e', 1.0, 2.0)
        for change in ('short', 'extra', 'shape', 'dtype', 'empty', 'duplicate'):
            m = self.fixture(b, b)
            if change == 'short': (self.root / 'candidate.bin').write_bytes(b[:-1])
            elif change == 'extra': (self.root / 'candidate.bin').write_bytes(b + b'00')
            elif change == 'shape': m['pairs'][0]['shape'] = [True]
            elif change == 'dtype': m['pairs'][0]['candidate']['dtype'] = 'float64'
            elif change == 'empty': m['pairs'] = []
            else: m['pairs'].append(m['pairs'][0])
            with self.assertRaises(audit.AuditError): audit.run(m, self.root)

    def test_chunk_boundary_cannot_skip_the_worst_element(self):
        n = audit.CHUNK_BYTES // 4 + 3
        b = struct.pack('<f', 1.0) * n
        c = b[:-4] + struct.pack('<f', 3.0)
        r = audit.run(self.fixture(b, c, 'f32'), self.root)
        self.assertEqual(r['pairs'][0]['elements'], n)
        self.assertEqual(r['pairs'][0]['worst_index'], n - 1)
        self.assertFalse(r['supplied_tensor_policy_passed'])

    def test_output_is_create_only(self):
        p = self.root / 'failed.json'
        audit.write_new(p, {'pass': False})
        before = p.read_bytes()
        with self.assertRaises(FileExistsError): audit.write_new(p, {'pass': True})
        self.assertEqual(p.read_bytes(), before)

    def test_malformed_json_shapes_fail_with_audit_errors(self):
        b = struct.pack('<e', 1.0)
        for manifest in (None, [], 3, 'wrong'):
            with self.assertRaises(audit.AuditError): audit.run(manifest, self.root)
        for field, bad in [('pairs', [None]), ('policy', []), ('policy', 'wrong')]:
            m = self.fixture(b, b)
            m[field] = bad
            with self.assertRaises(audit.AuditError): audit.run(m, self.root)
        for field in ('reference', 'candidate'):
            m = self.fixture(b, b)
            m['pairs'][0][field] = []
            with self.assertRaises(audit.AuditError): audit.run(m, self.root)
        for value in (10**1000, [], {}, 'zero'):
            with self.assertRaises(audit.AuditError): audit.limit(value)

if __name__ == '__main__': unittest.main()
