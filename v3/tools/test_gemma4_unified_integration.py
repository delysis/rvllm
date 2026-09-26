"""Source integration contracts, not Rust type checking or device acceptance."""
from pathlib import Path
import json,re,unittest
ROOT=Path(__file__).resolve().parents[1]
METAL=ROOT/'crates/rvllm-apple-metal/src'
APPLE=ROOT/'crates/rvllm-apple/src'
RUNTIME=ROOT/'crates/rvllm-runtime/src'
class UnifiedIntegrationTests(unittest.TestCase):
    def test_public_modules_and_ane_test_paths_are_complete(self):
        for p in [METAL/'lib.rs', APPLE/'ane_int8_candidates.rs']:
            source=p.read_text()
            self.assertNotIn('mod research_wave2',source)
            self.assertNotIn('ane_int8_wave2',source)
        for module in ['research_catalog','research_projection','research_evidence']:
            self.assertIn(f'pub mod {module};',(METAL/'lib.rs').read_text())
            self.assertTrue((METAL/f'{module}.rs').is_file())
        tests=(APPLE/'ane_int8_candidate_tests.rs').read_text()
        self.assertIn('mod oracle;',tests)
        self.assertIn('use oracle::interpret;',tests)
        self.assertIn('mod interleaved_checks;',tests)
        self.assertIn('mod oracle_checks;',tests)
    def test_one_down4_implementation_and_independent_interleaved_route(self):
        production=[p for p in APPLE.glob('ane_int8*.rs') if 'test' not in p.name and 'oracle' not in p.name]
        text='\n'.join(p.read_text() for p in production)
        self.assertEqual(text.count('fn build_ffn_down4('),1)
        self.assertFalse((APPLE/'ane_int8_wave2.rs').exists())
        for p in [RUNTIME/'gemma_ane_decode.rs',RUNTIME/'bin/rvllm_disaggregated_infer.rs']:
            self.assertIn('static-int8-interleaved-ffn-cached',p.read_text())
        self.assertIn('compile_int8_interleaved_with_cache_policy',(APPLE/'ane_linear.rs').read_text())
    def test_typed_catalog_export_has_no_device_dependency(self):
        s=(METAL/'bin/rvllm-metal-research-source.rs').read_text()
        self.assertIn('--catalog',s);self.assertIn('catalog_json()',s)
        self.assertNotIn('MetalContext',s)
        c=json.loads((ROOT/'tools/gemma4_metal_catalog.json').read_text())
        self.assertEqual(len(c['candidates']),48)
        self.assertEqual(sum(len(x['kernels']) for x in c['candidates']),86)
        self.assertEqual(c['default'],'off');self.assertIs(c['device_qualified'],False)
    def test_shared_plan_and_device_limit_boundary_both_used(self):
        s=(METAL/'layer_forward.rs').read_text();p=(METAL/'pipeline.rs').read_text()
        self.assertIn('ProjectionRequest',s)
        for expr in ['threadExecutionWidth()', 'maxTotalThreadsPerThreadgroup()', 'staticThreadgroupMemoryLength()']:
            self.assertIn(expr,p)
        self.assertIn('kernel.limits()',p)
        s=(METAL/'research_projection.rs').read_text()
        self.assertIn('projection_buffers_fit',s);self.assertIn('FallbackReason::Alignment',s)
    def test_complete_receipt_is_checked_after_durable_write(self):
        s=(RUNTIME/'bin/rvllm_disaggregated_infer/prefill_screen.rs').read_text()
        self.assertIn('complete_family_exercised',s)
        cli=(RUNTIME/'bin/rvllm_disaggregated_infer.rs').read_text()
        self.assertIn('complete_family',cli)
    def test_new_safe_modules_do_not_contain_unsafe_blocks_or_device_calls(self):
        for p in [METAL/'research_catalog.rs',METAL/'research_projection.rs',APPLE/'ane_int8_interleaved.rs',RUNTIME/'gemma_ane_ffn_oracle_tests.rs']:
            s=p.read_text();self.assertIn('#![forbid(unsafe_code)]',s)
            self.assertIsNone(re.search(r'\bunsafe\s*(?:\{|fn\b)',s))
    def test_ignored_fixture_markers_are_preserved_and_not_in_host_commands(self):
        s=(RUNTIME/'gemma_ane_ffn_oracle_tests.rs').read_text()
        for name in ['native_chunk4_matches_plain_cached_ffn','native_down4_matches_plain_cached_ffn','native_interleaved_matches_stacked_cached_ffn','host_prepare_ffn_component_pins']:
            self.assertIn('fn '+name,s)
        gate=(ROOT/'tools/check_gemma4_candidate_delivery.sh').read_text()
        self.assertNotIn('--ignored',gate);self.assertNotIn('--include-ignored',gate)
        self.assertIn('component_oracles::tests::',gate)
if __name__=='__main__':unittest.main()
