"""Host/source regression for a guard-refused fixture, not GPU execution."""
from pathlib import Path
import ast
import math
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / 'crates/rvllm-apple-metal/src'


def guard_value(expression, variables):
    """Evaluate only scalar boolean/integer MSL admission expressions."""
    expression = re.sub(r'!([^=])', r' not \1', expression)
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = ' '.join(re.sub(r'\b(\d+(?:\.\d+)?)f\b', r'\1', expression).split())
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return variables[node.value.id + '.' + node.attr]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not walk(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [walk(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = walk(node.left), walk(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Eq): return a == b
            if isinstance(op, ast.NotEq): return a != b
            if isinstance(op, ast.Lt): return a < b
            if isinstance(op, ast.Gt): return a > b
            if isinstance(op, ast.LtE): return a <= b
            if isinstance(op, ast.GtE): return a >= b
        raise ValueError('unreviewed guard syntax: ' + ast.dump(node))
    return walk(ast.parse(expression, mode='eval').body)


def shader_accepts(kernel, shape):
    source = (METAL / 'research_shaders/mma32_prefetch.metal').read_text()
    body = source.split('kernel void ' + kernel + '(', 1)[1].split('threadgroup half at', 1)[0]
    clauses = re.findall(r'if\s*\((.*?)\)\s*return;', body, re.S)
    # The final clause rejects out-of-grid work. Evaluate the exact launch/shape
    # clauses against group zero; the fixtures use correctly bounded grids.
    if len(clauses) != 3: raise ValueError('prefetch guard structure changed')
    m,n,k = shape
    v = dict(M=m,N=n,K=k,alpha=1.0,beta=0.0,
             **{'threads.x':128,'threads.y':1,'threads.z':1,'group.z':0})
    return not any(guard_value(clause,v) for clause in clauses[:2])


class PrefetchFixtureTests(unittest.TestCase):
    def test_original_first_case_is_guard_refusal_with_nan_poison(self):
        for name in ('research_gemm_mma32_prefetch','research_qkv_mma32_prefetch'):
            self.assertFalse(shader_accepts(name,(63,67,35)))
        self.assertTrue(math.isnan(struct.unpack('<f',b'\xff'*4)[0]))
        # All-ones BF16 widens to a NaN, too; no device need run to establish it.
        self.assertTrue(math.isnan(struct.unpack('<f',b'\x00\x00\xff\xff')[0]))

    def test_real_projection_roles_are_not_interchangeable_output_abis(self):
        shapes=[(63,67,35),(84,8192,3840),(652,9216,3840),
                (652,30720,3840),(652,3840,8192),(1024,3840,15360)]
        got=[(shader_accepts('research_gemm_mma32_prefetch',s),
              shader_accepts('research_qkv_mma32_prefetch',s)) for s in shapes]
        self.assertEqual(got,[(False,False),(False,True),(False,True),
                              (True,False),(True,False),(True,False)])
        self.assertEqual(sum(sum(pair) for pair in got),5)

    def test_fixture_classifies_guard_refusal_before_numerical_assertions(self):
        source=(METAL/'prefill_mma_tile_tests.rs').read_text()
        self.assertTrue('prefetch_fixture_expectation' in source, 'fixture ignores the shader admission contract')
        self.assertNotIn('assert!(float_outputs.iter().flatten().all(|x| x.is_finite()));',source)
        self.assertIn('validate_fixture_bytes',source)


    def test_capture_precedes_failure_and_is_create_only(self):
        source=(METAL/'prefill_mma_tile_tests.rs').read_text()
        capture=source.index('write_matrix_artifact(&directory.join(&file), &bytes)?')
        validation=source.index('let result = validate_fixture_bytes')
        propagation=source.index('for (path, result) in validation.into_iter()')
        self.assertLess(capture,validation)
        self.assertLess(source.index('directory.join("capture.json")'),propagation)
        self.assertIn('std::fs::File::create_new(path)?',source)
        self.assertIn('file.sync_all()',source)

    def test_refusals_cannot_fill_numerical_coverage(self):
        source=(METAL/'prefill_mma_tile_tests.rs').read_text()
        self.assertIn('prefetch_positive != [3, 2]',source)
        self.assertIn('prefetch_refusals != 7',source)
        self.assertIn('prefetch_positive[0] += usize::from(numerical[3])',source)
        self.assertIn('prefetch_positive[1] += usize::from(numerical[1])',source)
        self.assertIn('l2.push(None)',source)
        self.assertIn('"rvllm.metal_tile_comparison.v3"',source)

    def test_original_numerical_thresholds_remain(self):
        source=(METAL/'prefill_mma_tile_tests.rs').read_text()
        self.assertIn('assert!(relative < 0.0001',source)
        self.assertIn('(expected - actual).abs() < 0.005 + expected.abs() * 0.0001',source)
        self.assertIn('a.to_bits() == b.to_bits()',source)
        self.assertIn('bf16::from_f32(value).to_f32().to_bits()',source)
        self.assertIn('native_prefetch_preserves_existing_oracle_gates',source)
        self.assertIn('#[ignore = "explicit real-weight prefetch component oracle; no ANE access"]',source)

    def test_six_rust_regressions_are_registered_not_just_present(self):
        import json
        inventory=json.loads((ROOT/'tools/gemma4_candidate_host_tests.json').read_text())
        suite=next(x for x in inventory['suites'] if x['label']=='metal-projection')
        self.assertEqual(suite['filter'],'research_projection::tests::')
        expected=['prefetch_synthetic_tail_is_a_refusal_not_a_numerical_success',
                  'prefetch_production_roles_have_five_positive_and_seven_negative_arms',
                  'refusal_validation_rejects_even_one_written_byte',
                  'numerical_validation_never_accepts_untouched_or_partial_poison',
                  'fixture_size_and_canaries_are_checked_before_decoding',
                  'prefetch_fixture_has_no_unknown_or_wrong_role_fallback']
        source=(METAL/'research_projection.rs').read_text()
        for name in expected:
            self.assertEqual(suite['tests'].count('research_projection::tests::'+name),1)
            self.assertEqual(source.count('fn '+name+'('),1)

    def test_guard_expression_reader_refuses_unreviewed_operations(self):
        for expr in ['f()', 'x[0]', 'a + b', 'a ** b']:
            with self.assertRaises(ValueError): guard_value(expr,{'a':1,'b':2})

if __name__=='__main__': unittest.main()
