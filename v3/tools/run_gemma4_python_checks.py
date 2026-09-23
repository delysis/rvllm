#!/usr/bin/env python3
"""Run installed model/contract tests from this checkout, not an external packet.

Fake-tool tests are tests of orchestration. The numerical models exercise CPU
models; they do not execute Rust or establish Metal/ANE numerical acceptance.
"""
from pathlib import Path
import argparse
import shutil
import sys
import unittest

MODULES = (
    'test_gemma4_candidate_ci',
    'test_gemma4_candidate_delivery',
    'test_gemma4_wide_proposals',
    'test_gemma4_wide_models',
    'test_gemma4_wide_source',
    'test_gemma4_wide_ci_wiring',
    'test_gemma4_tensor_audit',
    'test_gemma4_catalog',
    'test_gemma4_unified_integration',
    'test_gemma4_shader_contracts',
    'test_gemma4_unified_proposals',
    'test_gemma4_prefetch_fixture',
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-rustfmt', action='store_true',
                        help='CI must execute the real rustfmt semantics test, not skip it')
    args = parser.parse_args()
    if args.require_rustfmt and shutil.which('rustfmt') is None:
        parser.error('rustfmt is required for the CI formatter-semantics test')
    tools = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in MODULES:
        if not (tools / (module + '.py')).is_file():
            parser.error('missing required test module: ' + module)
        tests = loader.loadTestsFromName(module)
        if tests.countTestCases() == 0:
            parser.error('zero-test module: ' + module)
        suite.addTests(tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
