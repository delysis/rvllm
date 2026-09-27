"""Static wiring guards complement, but do not replace, compiled Rust/device tests."""
from pathlib import Path
import re
import unittest

V3 = Path(__file__).resolve().parents[1]
M = 'crates/rvllm-apple-metal/src/'
A = 'crates/rvllm-apple/src/'
R = 'crates/rvllm-runtime/src/'


def source(path):
    return (V3 / path).read_text()


class WideSourceTests(unittest.TestCase):
    def test_new_safe_modules_have_no_unsafe_or_private_ffi(self):
        for path in [M+'research_next.rs', R+'gemma_head_ranking.rs',
                     A+'ane_packed32_layout.rs', A+'ane_int8_next_tests.rs']:
            text = source(path)
            self.assertIn('#![forbid(unsafe_code)]', text)
            self.assertNotRegex(text, r'\bunsafe\s+(?:fn|impl|\{)|extern\s+"C"')
            self.assertNotIn('_ANEClient', text)

    def test_new_ledger_slots_do_not_reinterpret_the_original_five(self):
        ledger = source(M+'research_evidence.rs')
        entries = re.findall(r'"([^"]+)"', re.search(
            r'pub const RESEARCH_KERNEL_NAMES:[^=]+=(.*?);', ledger, re.S).group(1))
        self.assertEqual(entries[:10], [
            'research_gemm_mma16x64', 'research_qkv_mma16x64',
            'research_rounded_gate32', 'research_gqa_kv8_d256', 'research_gqa_kv8_d512',
            'research_gemm_mma32_prefetch', 'research_qkv_mma32_prefetch',
            'research_attn_q4_d256', 'research_attn_q4_d512', 'research_rms_simd32'])
        self.assertIn(f'RESEARCH_KERNEL_COUNT: usize = {len(entries)};', ledger)
        layer = source(M+'layer_forward.rs')
        for marker in ['pipelines.record_research_dispatch(plan.kernel)', 'ResearchKernel::Temporal256']:
            for found in re.finditer(re.escape(marker), layer):
                position = found.start()
                self.assertIn('encoder.endEncoding();', layer[max(0, position-600):position])

    def test_vector_threadgroup_abi_and_poison_diagnostics_survive(self):
        for name in ['mma32_prefetch', 'attn_q4', 'rms_simd32']:
            text = source(M+'research_shaders/'+name+'.metal')
            for match in re.finditer(r'\[\[threads_per_threadgroup\]\]', text):
                self.assertRegex(text[max(0, match.start()-25):match.start()], r'uint3\s+threads\s*$')
            self.assertNotIn('atomic', text)
            self.assertNotIn('mem_device', text)
        text = source(M+'research_shaders/attn_q4.metal')
        self.assertIn('simd_any(nonfinite_query)', text)
        self.assertIn('uint(p) + 1', text)
        self.assertNotIn('uint(p + 1)', text)
        self.assertIn('poisoned || !isfinite(value) ? half(NAN) : f16_sat(value)', text)

    def test_old_trace_fallback_cannot_stand_in_for_prefetch_or_rms_dispatch(self):
        layer = source(M+'layer_forward.rs')
        self.assertIn('trace.is_none() && supports_gemma4_prefill_mma', layer)
        self.assertIn('full_prefill: allow_prefill_mma', layer)
        self.assertIn('postnorm_plan(', layer)
        self.assertIn('full_prefill_projection,', layer)
        plan = source(M+'research_projection.rs')
        self.assertIn('!self.full_prefill', plan)
        self.assertIn('if !full_prefill', plan)
        self.assertIn('kernel.limits()', layer)

    def test_packed32_has_source_and_codec_but_no_runtime_selector(self):
        self.assertIn('pub fn ffn_packed32_source', source(A+'ane_int8_candidates.rs'))
        for path in [A+'ane_attention.rs', A+'ane_linear.rs', R+'gemma_ane_decode.rs',
                     R+'bin/rvllm_disaggregated_infer.rs']:
            self.assertNotIn('ffn_packed32', source(path))
            self.assertNotIn('static-int8-packed32', source(path))

    def test_new_ane_graphs_require_cached_explicit_plans(self):
        runtime = source(R+'gemma_ane_decode.rs')
        cli = source(R+'bin/rvllm_disaggregated_infer.rs')
        for name in ['StaticInt8Down4FfnCached', 'StaticInt8FfnTransposeAttentionCached']:
            policy = runtime[runtime.index('pub fn cache_policy'):runtime.index('fn static_ffn_precision')]
            self.assertIn('Self::'+name, policy)
            self.assertIn('AneWeightPlan::'+name, cli)
        self.assertIn('head_ranking: HeadRankingPlan::Baseline', runtime)
        self.assertIn('head_ranking_timing: false', runtime)
        self.assertEqual(cli.count('.configure_head_ranking('), 2)

    def test_default_head_configuration_is_not_an_unconditional_candidate_probe(self):
        runtime = source(R+'gemma_ane_decode.rs')
        start = runtime.index('    pub fn configure_head_ranking(')
        stop = runtime.index('    pub fn head_ranking_observation(', start)
        setter = runtime[start:stop]
        self.assertNotIn('HeadTop5::new', setter)
        self.assertIn('validate_head_ranking_configuration', setter)
        self.assertIn('self.head_ranking_eligible', setter)

    def test_host_timing_excludes_vocabulary_projection_and_surface_read(self):
        runtime = source(R+'gemma_ane_decode.rs')
        begin = runtime.index('let mut top_five')
        rank = re.search(r'let ranking_started\s*=', runtime[begin:]).start() + begin
        self.assertIn('head.project', runtime[begin:rank])
        self.assertNotIn('head.project', runtime[rank:runtime.index('times.total_ms', rank)])

    def test_head_experiment_rejects_nonbaseline_ane_before_model_access(self):
        policy = source(R+'gemma_head_ranking.rs')
        cli = source(R+'bin/rvllm_disaggregated_infer.rs')
        self.assertIn('!baseline_cached_ane', policy)
        begin = cli.index('    validate_head_ranking_mode(')
        end = cli.index('    interleave |=', begin)
        self.assertIn('ane_weights == AneWeightPlan::StaticInt8FfnCached', cli[begin:end])
        self.assertLess(begin, cli.index('let model_dir = model_dir.ok_or'))


if __name__ == '__main__':
    unittest.main()
