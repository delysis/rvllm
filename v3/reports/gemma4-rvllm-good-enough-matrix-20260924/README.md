# Gemma 4 long-context Metal comparison — 2026-09-24

## Claim boundary

This directory records an exploratory, same-host comparison of the current
rvLLM BF16 Metal route with MLX-LM BF16 at 256, 512, 1024, 2048, and 4096 prompt
tokens. It is intended to locate large bottlenecks quickly. It is not a
promotion result: rvLLM generated two tokens per case, whereas the available
MLX-LM benchmark reports sustained generation over 64 tokens. Prefill work is
closer but still not an exact framework-semantic identity. Raw samples and
environment observations remain authoritative.

The rvLLM workload uses the exact Google Gemma 4 12B IT BF16 checkpoint and
tokenizer, one sequence, exact token counts, the `metal-mma32-load4` metallib,
and two decode steps after prefill. Each row below is checkpointed immediately
after the case finishes. MLX values are the arithmetic averages printed by
`mlx_lm.benchmark` over seven retained trials; no observations were dropped.

## Exploratory results so far

| Context | rvLLM prefill ms | rvLLM prompt tok/s | MLX prompt tok/s | MLX / rvLLM prefill | rvLLM decode tok/s | MLX decode tok/s | MLX / rvLLM decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 19,036.17 | 13.45 | 128.774 | 9.57× | 1.903 | 6.331 | 3.33× |
| 512 | 36,440.09 | 14.05 | 188.701 | 13.43× | 1.273 | 6.139 | 4.82× |
| 1024 | 73,513.99 | 13.93 | 209.018 | 15.01× | 0.807 | 6.276 | 7.78× |
| 2048 | 158,331.93 | 12.93 | 221.587 | 17.14× | 0.381 | 6.224 | 16.35× |
| 4096 | 331,904.02 | 12.34 | 230.308 | 18.66× | 0.205 | 6.369 | 31.12× |

Ratios are descriptive only. In particular, the decode denominator mismatch
(two rvLLM steps versus 64 MLX steps) can include different warmup effects.

## Immediate finding

rvLLM's last decode-step GPU execution and command-buffer wait times are nearly
identical: 469.9/470.2 ms at 256, 730.4/733.6 ms at 512, 1071.6/1081.6 ms at
1024, 2568.8/2570.1 ms at 2048, and 4295.7/4298.6 ms at 4096. At 2048 the
entire case used only 13.8 ms of host encoding while waiting 163.6 seconds on
Metal command buffers; at 4096 those figures were 14.4 ms and 341.7 seconds.
This is device work and dispatch structure, not Rust-side scheduling noise. The
rapidly widening decode gap identifies long-context attention—especially the
eight global D512 layers—as an urgent kernel target.

## ANE layer-0 FFN probes

The same campaign also exercised three resident-weight ANE FFN paths for Gemma
4 layer 0. These are direct operation probes, not end-to-end model throughput,
and are not comparable to the whole-model Metal/MLX rows above. All three
receipts report verified ANE execution with no CPU or GPU fallback.

| Weight path | Median ms | Minimum ms | p95 ms | Stored weight bytes | Error reference | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---|---:|---:|
| 4-bit LUT4 | 1.530 | 1.424 | 4.041 | 88,474,240 | original BF16 | 0.156019 | 0.708801 |
| 4-bit LUT4 | 1.530 | 1.424 | 4.041 | 88,474,240 | selected quantized CPU | 0.003156 | 0.003611 |
| 8-bit dense int8 | 3.085 | 2.980 | 3.475 | 177,016,768 | original BF16 | 0.012147 | 0.025701 |
| 8-bit dense int8 | 3.085 | 2.980 | 3.475 | 177,016,768 | selected quantized CPU | 0.002705 | 0.003446 |
| 16-bit BF16 | 54.349 | 24.499 | 60.112 | 353,894,400 | BF16 CPU | 0.002756 | 0.003857 |

The 4-bit path is about 2.02× faster than the 8-bit path by median, while the
8-bit path is about 17.62× faster than this BF16 probe. Those ratios describe
these exact implementations and measurement boundaries; they do not establish
that quantization preserves model quality. In particular, 4-bit error against
the original BF16 computation is large, and both quantized receipts explicitly
set `full_model_quality_qualified=false`. The smaller error against the selected
quantized CPU reference establishes implementation fidelity after quantization,
not semantic equivalence to BF16.

Preparation is also material but outside the steady-state medians. The 4-bit
probe spent 5,155.06 ms loading weights, 4,002.75 ms quantizing/reconstructing,
and 7,208.89 ms compiling/loading. The 8-bit probe spent 527.97 ms, 1,020.76 ms,
and 5,401.36 ms respectively. BF16 compile/load was 822.13 ms and its reported
effective weight bandwidth was 6.51 GB/s. The 4-bit queue conditions were
ineligible because an authorized Cargo/rustc process overlapped observation;
the 8-bit and BF16 queue conditions were eligible. No observations were
discarded.

Authoritative extracted receipts and their submitted manifests are the
`ane-*-report.json` and `ane-*.json` files beside this document. The queue job,
stdout, stderr, condition, and report receipts remain under `run-v3/`.

The queue marked this exploratory run condition-ineligible because the locally
authorized Cargo/rustc work used to repair the scaffold overlapped some samples.
Those processes use CPU and can contend for unified-memory bandwidth; the raw
observations are retained and no samples were excluded. The user explicitly
authorized treating this as useful exploratory evidence and requiring pristine
confirmation only later in the optimization process.

## Harness repairs made during this campaign

The session runner now writes a complete-schema checkpoint after every case and
marks whether all declared cases are complete. The power observer also bounds,
kills, and reaps every `pmset` subprocess after two seconds. Before these fixes,
a late timeout could erase valid earlier cases and a wedged `pmset` process
could prevent the persistent queue from launching any candidate.

The authoritative live artifacts are:

- `prompts.jsonl`
- `rvllm-metal-bf16-long-context-v1.json` (current submitted manifest despite
  the historical filename)
- `rvllm-metal-bf16-long-context-v3-checkpoint.json`
- `rvllm-metal-bf16-long-context-v3-profile.json` after normal completion
- `run-v3/results/rvllm-metal-bf16-long-context-20260924-v4/`

The v4 executable was built immediately before the final one-line completion
marker correction, so its checkpoint says `checkpoint_complete=false` even
with `checkpoint_completed_cases=5`. The separately written profile exists only
after `run_session_once` returned normally, and the queue receipt records exit
code zero. This historical marker defect is preserved rather than silently
rewriting the output; commit `2925564f` plus its follow-up fixes future runs.
