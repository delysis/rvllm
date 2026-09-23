# Gemma 4 unified hardware qualification execution

Date: 2026-09-23  
Source head: `ac06b9962b0b8fadcee8d37eee6e599a77183185`  
Requested base: `5043d03be1841b973db1c2be2da14a53e38eb46b`

## Prerequisites and delivery gate

The earlier 1.1 GiB blocker was cleared by removing only rebuildable Cargo
artifacts with `cargo clean` in `v3`; no database, model, report, queue,
lock, STOP marker, or raw evidence was removed. The subsequent check reported
18 GiB free. Boot, AC/battery, pmset, thermal, process policy, historical
owner lock, model snapshot, and executable/library identities were rechecked.
The unrelated llama-server was not stopped. The native delivery gate passed:
`compiled-only; no accelerator acceptance`.

## Metal matrix fixtures

All successful fixtures used the exact ignored, one-test command shape with the
full harness prefix:

```text
cargo test --offline --locked --release -j 2 --target aarch64-apple-darwin -p rvllm-apple-metal --lib layer_forward::prefill_mma_tile_tests::<fixture> -- --ignored --exact --nocapture
```

The initial filter without `layer_forward::` ran zero tests and is rejected.
Successful reports are fresh `/tmp` receipts, not repository artifacts:

| fixture | result | evidence |
|---|---|---|
| `native_short_tile_preserves_existing_oracle_gates` | correctness passed; 6 cases/24 commands; all guards and relative-L2 gates passed | `/tmp/rvllm-gemma4-hardware-20260923/native-short-tile.json` |
| `native_prefetch_preserves_existing_oracle_gates` | correctness failed: non-finite output assertion at `prefill_mma_tile_tests.rs:483`; stopped | `/tmp/rvllm-gemma4-hardware-20260923/native-prefetch.log` |
| `native_fp32_operands_preserve_existing_oracle_gates` | correctness passed; 6 cases/24 commands; all guards and relative-L2 gates passed | `/tmp/rvllm-gemma4-hardware-20260923/native-fp32-operands.json` |
| `native_vector_loads_preserve_existing_oracle_gates` | correctness passed; 6 cases/24 commands; FP32 bit parity passed | `/tmp/rvllm-gemma4-hardware-20260923/native-vector-loads.json` |
| `native_bf16_tile64_checks_both_output_abis_and_operand_paths` | correctness passed; 6 cases/36 commands; both ABIs/operand paths passed | `/tmp/rvllm-gemma4-hardware-20260923/native-bf16-tile64.json` |

No successful matrix receipt made a performance claim; qualification-only
mode reported no GPU timing acceptance.

## Prefill-only screens

The pinned reference
`v3/crates/rvllm-runtime/tests/reference/gemma4-12b-hf-chat-capital.json`
contains 21 prompt tokens. It is insufficient for the required >=64-token
GQA/temporal/long-tile positive screens, so those remain blocked rather than
being padded or invented.

Baseline (`RVLLM_METAL_RESEARCH=off`) and short-MMA
(`RVLLM_METAL_RESEARCH=metal-short-mma16x64`) were run with fresh output
directories, `--prefill-only true`, and `--ane-compile-budget 0`.

- Baseline: first-token gate passed, complete family exercised, zero ANE
  compiler calls/decode steps; `/tmp/rvllm-gemma4-hardware-20260923/prefill-baseline-fresh-20260923-2/report.json`.
- Short-MMA: same gates passed, with research dispatch counts 144 GEMM and 48
  QKV; zero ANE compiler calls/decode steps;
  `/tmp/rvllm-gemma4-hardware-20260923/prefill-short-fresh-20260923-1/report.json`.

Both are `correctness passed / performance unmeasured`: first-token-only,
not full continuation or tensor acceptance; power was battery and sampled
controls were ineligible.

## Deferred work

Baseline cache inspection/recovery, full-route continuation, host FFN pinning,
cached-only FFN comparisons, >=64-token positive screens, timing, and
promotion were not started from these bounded results. They require the
existing cache/lifecycle gates and fresh pinned references. RMS lacks its
direct native oracle; multi-I/O remains quarantined; packed32 remains blocked.
No model, cache, or bulky raw artifact belongs in git.
