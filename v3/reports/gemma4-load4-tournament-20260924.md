# Gemma 4 load4 tournament — 2026-09-24

Tested tree: `b0984856f33b43037b4e2b0a1dd00fb9d6d4611a` on
`astra/gemma4-load4-tiles-20260923` (PR #4). The queue-policy and standard
amendments documented below were uncommitted while these receipts were
produced. No production selector or default changed.

## Result

All seven submitted Metal load4 configurations passed the actual-weight native
component oracle and exercised their requested GEMM and QKV families on the
full 48-layer first-token route. None is promotable. The three candidates
advanced to the sealed timing screen were slower than the unchanged
`metal-mma32-load4` incumbent. The remaining four have only one-shot
exploratory latency and are not timing-qualified.

| candidate | component oracle | exact-route first token | exploratory Metal GPU ms | exploratory prefill ms | controlled timing |
| --- | --- | --- | ---: | ---: | --- |
| `metal-load4-m16n32k64` | pass | pass | 882.760 | 1458.510 | not advanced |
| `metal-load4-m16n64k64` | pass | pass | 818.933 | 1236.281 | slower |
| `metal-load4-m32n32k64` | pass | pass | 651.347 | 1123.420 | slower |
| `metal-load4-m32n64k32` | pass | pass | 1217.032 | 1591.362 | not advanced |
| `metal-load4-m32n64k64` | pass | pass | 1482.656 | 2070.704 | not advanced |
| `metal-load4-m32n64k128` | pass | pass | 812.559 | 1195.772 | slower |
| `metal-load4-m64n64k64` | pass | pass | 2560.151 | 3127.814 | not advanced |

The exploratory columns are retained observations, not comparable promotion
statistics. They were used only to allocate the controlled screen.

## Component and route gates

Each `component-qualified` receipt covers the six supported projection roles:
sliding/global QKV, gate-up, sliding/global output and down projection. QKV
requires exact FP32 output bits. Stored projections require exact once-rounded
BF16 bits. The oracle additionally retains independently sampled FP64 dot
products, guard refusal, surrounding canaries and repeated-use stability.

Every route screen produced the pinned first token. Requested selections were
actually dispatched; a candidate arm encoded 144 candidate GEMM dispatches and
48 candidate QKV dispatches over the complete-family prefill boundary. These
screens do not establish multi-token continuation, promotion or a speedup.

## Controlled timing screen

The sealed queue executed 72 jobs: 12 warmup arms and 60 measured arms. Each of
three comparisons contains five complete ABBA blocks. No measured observation
was deleted, retried or selected after latency was visible. All first-token
gates passed. Sampled conditions were AC power, low-power mode, pmset power
mode 1 and nominal thermal state for this completed campaign.

The frozen estimator gives each complete block equal weight. Within a block,
the contrast is the mean candidate log latency minus the mean control log
latency; the reported ratio is the exponential of the mean block contrast.
Values above one mean the candidate is slower. Intervals are ordinary
two-sided 95% t intervals over five block contrasts. They are small-sample
screening intervals, not independent confirmation or familywise promotion
intervals.

| candidate | GPU candidate/control | GPU 95% interval | prefill candidate/control | prefill 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `m32n32k64` | 1.1195 | 1.0163–1.2332 | 1.0151 | 0.9616–1.0716 |
| `m32n64k128` | 1.3488 | 0.9354–1.9449 | 1.1594 | 0.9514–1.4130 |
| `m16n64k64` | 1.1870 | 0.9941–1.4173 | 1.1143 | 0.9807–1.2661 |

`m32n32k64` is the least bad complete-prefill result, but its measured GPU
execution is clearly worse. It is not a nominee. No comparison against `off`
or independent confirmation is warranted for this batch.

## Queue-policy amendment

The earlier timing manifest required one thermal value and could wait forever
for that value. The queue now accepts every *known* macOS thermal state (0–3)
when the manifest leaves thermal state unpinned, while still rejecting missing
or out-of-range state. The standard requires comparisons to remain local to
complete ABBA blocks, records each block's thermal sequence and forbids pooling
blocks as though they shared one thermal stratum. A manifest may still pin one
thermal state. This admits work under real laptop conditions; it does not claim
clock normalization or erase serious/critical thermal labels.

Focused verification:

```text
cargo test --offline --locked -p rvllm-runtime \
  --features macos-private-ane-research \
  --bin rvllm_experiment_queue
24 passed; 0 failed
```

Invoking the binary tests without `macos-private-ane-research` correctly
refuses because the target declares that required feature. Workspace warnings
printed by dependencies were pre-existing and are not counted as a clean
Clippy result.

## Evidence locations and next action

Raw component receipts:
`v3/reports/gemma4-load4-tournament-20260924-components-1/`.

Raw route screens:
`v3/reports/gemma4-load4-tournament-20260924-screen-queue/`.

Raw sealed timing campaign:
`v3/reports/gemma4-load4-tournament-20260924-timing-queue-variance-robust/`.

These directories remain local evidence because they contain large raw output
and model-derived captures. A future publication commit should carry a compact
hash manifest, not the tensors themselves.

The next round must change the kernel strategy rather than retile the same
losing load4 design. Separate, exact-format lanes are required for Metal and
ANE at 4, 8 and 16 bits. Each lane must pass an actual-weight component oracle,
full-route multi-token/KV correctness, sealed screening and independently
counterbalanced confirmation before promotion. Cross-framework MLX/llama.cpp
results belong in precision- and workload-matched strata.
