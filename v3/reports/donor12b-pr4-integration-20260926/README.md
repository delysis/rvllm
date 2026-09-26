# Donor12B Metal candidates on PR #4 — integration screen

Source integration commit `cab070ffeb9b83b2b4e182b3fe5769a2d58b65d8` is based on
PR #4 checkpoint `645a963f0fcdd76b3edc2e689bd1b796a60fd11f`. The two
pre-existing donor QMV dispatch slots 60–61 are preserved; the new SG8/SG4
families append slots 62–85 and remain default-off. This branch is **not** a
production promotion. The supplied patch and exact-base validation are
separately retained in `../donor12b-exact-base-20260926/`.

Host checks on this integration tree: offline locked Metal all-target check,
Apple-feature runtime library check, and Metal library tests passed (174
passed, 29 ignored). Patch-owned Rust was formatted; pre-existing unrelated
whole-workspace warnings/format deltas are not counted as acceptance.

The strict Metal 3.1 `-fno-fast-math` SG8 and SG4 libraries compiled on an
Apple M4 Max (Apple9). Their generated source and metallib hashes match the
exact-base source: SG8 `49f5fb616a5c8db9689896b2e97c8636c67eee5227703245498a0f6ed291b13e`
and `21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`;
SG4 `c9ee195a84710489e662f7fedd9996154359493b92290ac51faca471b46ec219`
and `94372f6c3af50e537d30758fc0d6742363174718dc3cd541a7f2aa61ee7298b0`.
The **integration** test executable is a different identity:
`459c52cd0929a09926dbeaf90db75fbb42299dc74e0ee33b5fa2acfe23a56704`.
The existing serial experiment queue executed it, with observed conditions and
no thermal-stability wait. Both source-bound native oracles passed 21/21
synthetic operator cases, across all twelve entry points per selector.

| Same-weight hot-cache projection | SG8 control → candidate µs; eligible ratio | SG4 control → candidate µs; eligible ratio |
| --- | ---: | ---: |
| W4 down, M1 K15360 | 237.23 → 112.31; 2.112× | 233.24 → 112.92; 2.066× |
| W8 global O, M1 K8192 | 154.60 → 61.15; 2.528× | 156.26 → 61.81; 2.528× |
| W8 local O, M1 K4096 | 67.13 → 30.37; 2.210× | 73.70 → 30.87; 2.388× |
| W8 local O, M9 K4096 | 316.92 → 144.20; 2.198× | 319.08 → 110.06; 2.899× |

All eight integration cells passed the in-receipt 5% drift/order gate, and the
queue marked their observed AC/low-power/thermal stratum eligible. The SG4
local-M1 cell **failed** the 5% gate in the preceding exact-base trial even
though it passed here; that run remains in the record. Neither run is an
independent checkpoint/full-route confirmation. The control is rvLLM's
`experimental_projection_*_bf16_n4`, not MLX. The candidate's faster operator
does not establish a model-level speedup, quality, or even that the candidate
dispatches on an authenticated model package. `production_promotion` and
`automatic_promotion` remain false in the native reports.

The four queue manifests, compiled library/build receipts, raw numerical
outputs, ABBA samples, and queue conditions/stdout/stderr are preserved here.
SG8/SG4 ABBA receipt SHA-256:
`29fb06ccef0be9a9aa597d7fb679a9b15b45c91148e38990ef459d401fcdab6c` /
`1e678a81d320289c80cc145d08c8d6772970ec68832c51a49d9dc8b8df05a6bf`.

## First real-weight BF16 route probe

The queue ran the same six-token prompt plus two generated tokens through the
actual 48-layer Gemma 4 12B BF16 checkpoint using the Apple-feature runtime
executable `25a1a8f69626430877274d4ce59a4e65863aa0156b07f3c01dc739ab3feb7761`.
SG8, SG4 and selector-off all returned token IDs `[9079, 236761]`; each
reported zero library and pipeline compilations during inference. Actual
candidate dispatch counts for **each** selected variant were global attention
16, local attention 80, native fused gate 96, native projection 480. These
are two decode steps across 8 global and 40 sliding layers, plus prefill
participation. Selector-off reported no research dispatches. The control
used the SG8 metallib with research selection off, so this is a route control,
not a separately built pristine shipping binary.

| Route | Prefill ms | Decode ms, two tokens | Queue conditions eligible |
| --- | ---: | ---: | --- |
| SG8 | 1295.50 | 132.27 | yes |
| SG4 | 1103.01 | 132.02 | **no** |
| Selector off | 1564.87 | 252.23 | **no** |

These are **single sequential runs**, not paired ABBA timing. The queue marked
SG4 and control condition strata ineligible despite nominal sampled AC and
thermal readings; those measurements are retained as exploratory observations,
not a speedup claim. Token agreement is a narrow continuation check, not a
teacher-forced logit, per-layer, KV, or checkpoint-quality gate. The route used
native BF16 weights, so it did **not** exercise the W4/W8 sidecar routes.

Three earlier route jobs are preserved as quarantined pre-execution failures:
the first combined `--report` with a single prompt; the second combined
`--top-logits` with session input; the third used a binary built without the
`apple` feature. None loaded weights or supplied candidate performance data.

The first long-context manifest accidentally reused a prompt file whose fifth
case is 4096 tokens, beyond the agreed screen. It was deliberately interrupted
after its 256/512/1024 cases, and the **queue job is failed/quarantined**.
`sg8-bf16-contexts.json` happens to say `status: pass` for the three completed
cases; that does **not** make the five-case job successful or supply a 2048
result. The bounded wrapper and new jobs 11–13 select only the first four
cases; they are queued separately. The old dependent jobs 09–10 are obsolete
and must not be counted or replayed as successful evidence.

Remaining gates: authenticated real-weight W4/W8 sidecar loading, per-layer
and KV comparison, long-context
attention correctness/timing, checkpoint logits/continuation quality, and
paired end-to-end timing against both the default rvLLM route and MLX.
