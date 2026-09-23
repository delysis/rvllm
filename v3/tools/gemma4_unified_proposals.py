#!/usr/bin/env python3
"""Validate the frozen unadmitted proposal backlog. Never runs a command/job.

Local pinning/admission creates a new record through the existing owner's
process, not by editing these source-review templates or old attempted jobs.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1] / 'reports/proposals/gemma4-unified-cffb22da'
BASE = 'cffb22dabcb57b5f3ecee05acd4ab810d1ce6077'
CANDIDATES = {
    'metal-mma32-f32': ('off/native-bf16-mma32', 6, 'operand-lowering-change',
                      'prefill_mma_tile_tests::native_fp32_operands_preserve_existing_oracle_gates', 24),
    'metal-mma32-load4': ('off/native-bf16-mma32', 6, 'layout-only-bitwise-fp32-gate',
                        'prefill_mma_tile_tests::native_vector_loads_preserve_existing_oracle_gates', 24),
    'metal-long-mma32x64': ('off/native-bf16-mma32', 64, 'storage-boundaries-preserved',
                          'prefill_mma_tile_tests::native_bf16_tile64_checks_both_output_abis_and_operand_paths', 36),
    'metal-rmsnorm-simd256': ('off/post-projection-rms-tree', 6, 'reduction-order-change', None, None),
    'ane-int8-ffn-interleaved': ('static-int8-stacked-ffn-cached', 1, 'same-int8-values-scales-fp16-materializations',
                              'gemma_ane_decode::component_oracles::native_interleaved_matches_stacked_cached_ffn', 6),
}


def expected(name):
    control, minimum, numerical, fixture, dispatches = CANDIDATES[name]
    ane = name.startswith('ane-')
    return {
        'schema': 'rvllm.gemma4.unified-proposal.v1', 'base_commit': BASE,
        'candidate': name, 'status': 'source-only-unadmitted', 'control': control,
        'selector': {'environment': 'RVLLM_METAL_RESEARCH', 'value': name} if not ane else
                    {'argument': '--ane-weights', 'value': 'static-int8-interleaved-ffn-cached'},
        'shape': {'hidden': 3840, 'intermediate': 15360, 'layers': 48, 'query_heads': 16,
                  'layer_families': [[8, 256, 1024], [1, 512, 0]],
                  'moe': False, 'ple': False, 'min_tokens': minimum, 'max_tokens': 1 if ane else 1024},
        'numerical_contract': numerical, 'compile_budget_in_inference': 0,
        'component': {'fixture': fixture,
                      'state': 'adapter-supplied-not-run' if fixture else 'blocked-missing-direct-native-oracle',
                      'maximum_dispatches': dispatches, 'timing_permitted': False,
                      'captured_activations': '1-3 independently pinned real inputs' if ane else None},
        'cache': {'new_programs': 48 if ane else 0, 'part': 'ffn-int8-interleaved' if ane else None,
                  'strict_inspection_required': True, 'inference_program_visits': 162},
        'pins': {key: None for key in ['source_tree', 'executable', 'model', 'reference',
                  'metallib', 'oracle_policy', 'driver_journal', 'boot', 'process_policy']},
        'power_stratum': {key: None for key in ['source', 'low_power', 'pmset_mode', 'thermal']},
        'timing': {'order': ['A', 'B', 'B', 'A'], 'warmup_requests_per_block': 2,
                   'measured_requests_per_block': 7, 'requests': 36, 'output_tokens_per_request': 10,
                   'decode_steps_per_request': 9, 'model_decode_steps': 324,
                   'ane_evaluations_per_step': 208, 'maximum_ane_evaluations': 67392,
                   'warmups_in_work_count': True, 'early_eos_invalidates_matching': True},
        'drift_fraction': 0.05, 'minimum_free_disk_gib': 16,
        'acceptance': ['unchanged-original-numerical-gates', 'actual-matching-dispatch',
                       'complete-reference-continuation', 'driver-lifecycle-and-zero-timing-compiles',
                       'matched-stratum-workload-and-capture', 'independent-confirmation'],
        'promotion_automatic': False, 'retries_automatic': False,
        'accelerator_time_normalized_by_cpu_cycles': False,
    }


def read_json(text):
    def pairs(items):
        value = {}
        for key, entry in items:
            if key in value: raise ValueError('duplicate JSON key')
            value[key] = entry
        return value
    def constant(value): raise ValueError('nonfinite JSON token: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_all(values):
    seen = set()
    if not isinstance(values, list): raise ValueError('proposal list required')
    for value in values:
        if not isinstance(value, dict): raise ValueError('proposal object required')
        name = value.get('candidate')
        if not isinstance(name, str) or name not in CANDIDATES or name in seen:
            raise ValueError('unknown or repeated candidate')
        # Canonical JSON comparison preserves bool/int distinctions too.
        if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected(name), sort_keys=True):
            raise ValueError('changed frozen proposal; use a separately reviewed admission record: ' + name)
        seen.add(name)
    if seen != set(CANDIDATES): raise ValueError('incomplete proposal backlog')
    return sorted(seen)


def main():
    try:
        paths = sorted(ROOT.glob('*.json'))
        if any(p.is_symlink() or p.stat().st_size > 65536 for p in paths):
            raise ValueError('invalid proposal file')
        names = validate_all([read_json(p.read_text()) for p in paths])
        print(json.dumps({'status': 'source-only-unadmitted', 'candidates': names,
                          'jobs_created': 0, 'hardware_executed': False}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
