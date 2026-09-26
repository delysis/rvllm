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

The initial four queue manifests, compiled library/build receipts, raw numerical
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
result. The bounded wrapper and new jobs 11–13 selected only the first four
cases; their completed receipts are below. The old dependent jobs 09–10 are
obsolete and must not be counted or replayed as successful evidence.

## Bounded 256–2048-token BF16 route screen

The existing serial queue completed SG8, SG4, then selector-off on the same
real BF16 checkpoint and the same four prompts, with two generated tokens per
case. All twelve cases reported `pass`, returned `[236770, 236770]`, and
reported zero library and pipeline compilations during inference. All three
queue jobs exited successfully and their sampled condition strata were marked
eligible; conditions were observed, not used as a stability gate. Candidate
dispatch counts **per case** were global attention 16, local attention 80,
native gate 96, and native projection 288. Selector-off had no research
dispatches. The selected kernels therefore demonstrably ran during decode.
At these longer prompts, the 288 projection count corresponds to the two
decode steps, not a demonstrated candidate prefill route; prefill timings are
included to expose route/host behavior, not to credit these kernels with
prefill improvement.

| Prompt tokens | Prefill off / SG8 / SG4 ms | Two-token decode off / SG8 / SG4 ms | Exploratory decode off ÷ SG8 / SG4 |
| ---: | ---: | ---: | ---: |
| 256 | 17015.95 / 15893.98 / 16412.32 | 736.90 / 143.25 / 154.22 | 5.14× / 4.78× |
| 512 | 30086.03 / 30124.86 / 28924.09 | 1106.70 / 155.90 / 174.03 | 7.10× / 6.36× |
| 1024 | 59034.94 / 59168.43 / 58419.07 | 1806.95 / 172.11 / 211.58 | 10.50× / 8.54× |
| 2048 | 131086.00 / 131872.80 / 126566.53 | 2964.45 / 200.10 / 258.49 | 14.82× / 11.47× |

These are one run per arm in serial **SG8 → SG4 → off** order, not paired
ABBA/BAAB repetitions. The widening decode gap with context is an important
lead, but the ratios are not promoted speedups: sampled power/thermal strata
cannot establish fixed clocks or absence of contention, token agreement is
not a per-layer oracle, and the selector-off control is a selected-off route
through the SG8-equipped executable rather than a separately built shipping
binary. There is no MLX comparison in this receipt. The very long prefill
times are also a separate optimization problem; this screen does not isolate
their operation-level causes.

Raw backend outputs: `sg8-bf16-contexts-bounded.json`,
`sg4-bf16-contexts-bounded.json`, and `off-bf16-contexts-bounded.json`;
SHA-256 respectively `ae5b3c8941648ab0517c052c98396797e880dc6a8e8cc4067a34bde85f5b529d`,
`898c6ccf6733dd1bfbdd2ca17c53e96e55590c1e1f66d689fa4d3e48d3a50310`, and
`2f91c50a2936973b3186e82cc15aa7d4fcd9e8b15825274b9a92e062caaf5a5b`.
The `queue-results/g4-donor12b-pr4-{11,12,13}-*-bounded-contexts/`
directories retain job manifests, queue reports, conditions, stdout and
stderr. No 4096-token case was run by these jobs.

## First authenticated BF16 + W8 package route — continuation mismatch

The package exporter now accepts a uniform BF16 checkpoint and an explicit
group-32 W4/W8 sidecar for any supported dense projection role. It decodes
source BF16 values before quantization and records a BF16 activation ABI in
the authenticated descriptor. A fixture test compares every emitted packed
byte and FP16 scale against the independent CPU reference, then reopens the
package and checks that the source was unchanged. The package CLI accepts the
general `--low-bit-proj` flag, with `--low-bit-down-proj` retained as an alias.
The direct inference CLI now recognizes packages; its engine-session mode
explicitly refuses a package because that mode does not install sidecars.

The first real package deliberately replaces **only layer 0 dense down** with
W8. The other 47 layers and all other roles retain native BF16 weights. Its
manifest SHA-256 is
`1cb72e93d101f23f969b422586bb63058bca0bb23aa80c27e87accff1d162a69`;
the 58,982,400-byte packed W8 payload and 3,686,400-byte scale payload have
SHA-256 values `b885c80ac1f7152998d3e29917f1fd060348d3fd6ab5b1bbf448867a8d5d0842`
and `1e4bd6dace99ed4ab6cca745ba42f3a64ac662e1a143eea8b71c9c3f14d41175`.
The package's BF16 metallib is the same SG8 library used for the source
control. The package is a local ignored build artifact, not in Git; the
checked-in manifest/job/receipt identify it but do not make it portable to
another machine.

The serial queue ran a direct 48-layer short continuation with the sidecar,
using executable SHA-256
`83708e99380170fcacce10528527a6e8c9fa5c198a0054d55e3d45b79b1e1e95`.
Actual dispatch counts were `research_donor12b_sg8_batch_w8: 1` and
`research_donor12b_sg8_w8: 2`, with zero library/pipeline compilations during
inference. That confirms the sidecar reached prefill and both decode steps.
The queue process exited successfully, but **the narrow continuation check failed**:
the package returned `[496, 45518]` (`" athought"`) while the same exact
executable, prompt, SG8 metallib, and source BF16 weights returned
`[9079, 236761]` (`" Paris."`). The one-W8 run's observed host conditions
were ineligible, but that does not explain the observed token mismatch;
no latency claim is taken from it. The same-binary source control's observed
conditions were eligible. Synthetic SG8 W8-down oracles at M1 and M6, exact
3840×15360, passed along with the other 21 native cases (23/23 overall), so
they do not substitute for a real-weight quality gate. Exact BF16 token
agreement is not itself a generally valid requirement for a quantized
checkpoint; the changed continuation is a diagnostic warning, not a measured
perplexity or checkpoint-quality result.

An ignored real-checkpoint layer-zero diagnostic then ran both routes under
the same test executable (`364c15f249d4b73af09c91f21cb6a4825e7f88ef1daa920fa13354b2b4e8b63d`),
stopping before later layers or logits. The trace summaries match exactly
through the FFN activation at all reported fields and selected elements. The
first observed difference is after the down/FFN branch: the largest absolute
delta among the **selected** trace elements is 0.01171875; after the final
layer-zero residual it is 0.01318359375. Both routes remain finite. This
localizes the first observed perturbation to the intentionally replaced
projection and does not show a gross layer-zero corruption. However, the
trace contains summaries and selected elements, **not a complete elementwise
diff or an independent real-activation FP64 dot oracle**. It therefore does
not establish that the W8 shader is correct for this checkpoint, nor whether
small layer-zero error magnifies downstream enough to change the token.
The trace queue job exited successfully; its observed timing conditions were
ineligible and no timing claim is made.

An attempted selector-off control on the *same* BF16+W8 package was rejected
at preparation, before inference: the incumbent generic low-bit route only
admits F16 activations, whereas this package authentically declares BF16.
Queue job 18 is failed/quarantined with its error retained. It supplies no
numeric comparison, and the fail-closed guard was not weakened. A separately
sealed SG4 donor-library package was then built and reopened, with manifest
SHA-256 `ee05c311c551e4bb0d83e80f166afe9d5ebcdc84f862729f4656ae15eace298d`.
Its native BF16 shard and W8 packed/scales hashes match the SG8 package
exactly; only the package identity and macOS BF16 metallib differ. SG4's
independent metallib SHA-256 is
`94372f6c3af50e537d30758fc0d6742363174718dc3cd541a7f2aa61ee7298b0`.
The SG4 package returned the **same** `[496, 45518]` continuation as SG8,
with actual `research_donor12b_sg4_batch_w8: 1` and
`research_donor12b_sg4_w8: 2` dispatches, and zero inference-time compiles.
Its queue job succeeded with observed conditions eligible. This reduces the
likelihood of an SG8-only arithmetic defect, but cannot distinguish a shared
low-bit defect from a quantized-checkpoint effect.

A separate read-only host scan mapped the entire 3840×15360 layer-zero BF16
down matrix and its W8 bytes/scales, converted BF16 source bits to FP32,
dequantized every group-32 signed W8 code using the stored FP16 scale, and
computed FP64 sums of squared error in 32-row chunks. Across all 58,982,400
weights, the dequantized-weight RMSE was `6.71804e-5`, RMS-relative error
`0.00673189`, and maximum absolute weight error `0.000805855`. This is a
host diagnostic of the stored representation, not a Metal arithmetic or
model-quality oracle. The numerical summary was not emitted by the serial
referee and should be independently reproduced before promotion.

The queue also ran matched single-prompt first-token top-10 diagnostics on
the exact same executable and SG8 library. Native BF16 ranked token 9079
(`Paris`) first at logit `20.75`, ahead of token 496 at `18.875`. With the
one-W8 sidecar, both tokens were reported at `20.375`; the sampler chose
token 496 on this tie. Thus the two-token textual divergence is **not**
evidence by itself of a grossly broken output distribution. It is a
material logit shift for this prompt (token 496 +1.5, token 9079 -0.375),
and a checkpoint-wide quality gate remains missing. The output logits are
diagnostic values at the route's storage/rounding boundary, not FP64 teacher
logits. Both top-logit queue jobs passed with observed conditions eligible.

Raw records: `sg8-bf16-package-one-w8.json`,
`sg8-bf16-same-binary-control.json`, `sg4-bf16-package-one-w8.json`,
`sg8-w8down-oracle/donor12b-oracle.json`,
`sg8-bf16-one-w8-layer0-trace/{source,one-w8-package}-layer0.json`, and
corresponding `queue-results/g4-donor12b-pr4-{14,15,16,17,18,19,20,21}-*/`
directories. The top-logit JSON is in each job's `trial.stdout`.
**Do not promote W8 from this result.** A package-sidecar load defect,
real-activation numerical discrepancy, or checkpoint sensitivity remains to
be distinguished; the present evidence does not choose among them.

The separate one-projection W4 package, manifest SHA-256
`1b9c43598644619432589b66836ed22e52d1e48c968e4b93134f21bd5e324949`,
also completed its first real-weight route (queue job 26, observed conditions
eligible). It replaced only layer-0 down, exercised one batch and two decode
SG8 W4 dispatches, and returned the same first two token IDs as native BF16,
`[9079, 236761]`. This is encouraging but extremely narrow checkpoint-quality
evidence: W4 can change logits without changing these argmax tokens. The CLI
reported total package-route preparation counters of one library compile and
70 pipeline-state compiles; these are not a zero-compile package receipt.
Neither W4 nor W8 is promoted from a two-token probe.

## Hosted CI repair on the integration branch

PR #4's starting checkpoint had host-side fixtures pinned to its old
11-candidate/17-entry catalog despite containing 18 candidates and 31
exported entries. The initial stacked repair updated those counts and gated
the macOS-only artifact-inspection entry point so Linux workspace builds
fail closed. A later hosted run exposed two further stale host contracts in
sequence: the exact Metal-projection test-name list omitted four existing
load4 tests; after that correction, the host catalog still described the
archived 18-candidate/v3 prefix while the runtime actually exported 48
candidates, 86 entry points, and dispatch schema v5. Neither failure is a
device or kernel numerical failure. The current repair seals the complete
48-candidate runtime catalog in the reviewed JSON, retains exact order,
source, resource, numerical-contract, and duplicate checks, and extends
source export to all 96 candidate/dtype arms. The Rust catalog test
checks the entire golden and independently retains the 22-candidate global
family check. Local focused Python catalog/CI tests passed (13/13); the
complete local host runner then passed 70 Rust tests and all 96
candidate/dtype source exports. This is a host/source-export gate, not Metal
compilation or device qualification. A new hosted CI run is still required
before calling the repair closed.

## SG8 promotion decision in progress

The BF16 route speed result is distinct from the one-projection W8 package
quality diagnostic. It is large enough that holding SG8 default-off now
requires a specific, bounded check rather than general caution. Jobs 22–25
are an interleaved **SG8 → selector-off → selector-off → SG8** full-route
comparison at 256 prompt tokens plus two decode tokens, on the same binary,
checkpoint, prompt, metallib and direct backend. Each arm has an independent
queue receipt, records conditions without waiting for thermal stability, and
must confirm the same generated token IDs, actual candidate dispatch, and
zero inference compiles. Job 26 then probes one real W4 layer-0 down sidecar;
that does not gate the separate native-BF16 SG8 decision. The promotion
disposition uses all four timing samples, including any reversal or
condition change, rather than selecting favorable arms.

All four 256-token jobs subsequently passed, with eligible observed
conditions, matching generated IDs `[236770, 236770]`, and zero library or
pipeline compilations. SG8 dispatches were observed for all 48 layers;
selector-off had no research dispatches. Decode durations in queue order
were **333.15, 1237.18, 1237.64, 337.98 ms**, yielding 3.71× and 3.66×
paired control/SG8 ratios (3.69× from the two-arm means). The two controls
agree within 0.04%; the two SG8 arms differ by 1.45%. The result survives
reversing run order and a much slower ambient host state than the first
screen. This advances SG8 to **BF16 full-route promotion candidate**, no
longer just an operator lead. It does not yet change the production default:
two-token agreement on one artificial prompt cannot detect later-step KV or
logit drift. Jobs 27–28 ran the same prompt for 64 decoded tokens on SG8 and
selector-off. Both completed with no inference compiles. SG8 dispatched its
native family for all 64 steps and reported **9.605 s / 6.663 tok/s**;
selector-off reported **51.111 s / 1.252 tok/s**. The latter queue receipt was
condition-ineligible after a power-observer timeout, so its 5.32× timing ratio
is only diagnostic, not promotion evidence. The generated IDs agree at the
first **63** positions and differ at the final position: SG8 `236761`,
selector-off `236770`. This is a precise, newly observed numerical boundary,
not a generic reason to shelve a faster implementation. Jobs 29–30 repeat
the 64-token comparison in reverse order to determine whether that position
is reproducible. Those jobs subsequently returned the same 63-token prefix
and the same final split: selector-off `236770`, SG8 `236761`. Selector-off
reported 46.969 s / 1.363 tok/s with eligible observed conditions; SG8
reported 10.322 s / 6.200 tok/s, but its condition stratum was ineligible
because power-mode observation was intermittently unavailable. Both jobs
exited successfully. Thus the numerical difference is repeatable in two
independent processes per arm; the second timing pair remains diagnostic,
not a qualified promotion ratio. Its raw manifests, outputs, condition
samples, and queue reports are in `queue-results/g4-donor12b-pr4-{29,30}-*/`.
Until the discrepancy is understood or bounded by a
checkpoint-quality criterion, SG8 remains the BF16 performance leader but
not the production default. W4/W8 package quality is evaluated separately
and cannot veto a native-BF16 speed result.

For orientation only, the first SG8 BF16 two-token decode observations imply
13.96, 12.83, 11.62, and 10.00 tokens/s at 256–2048 contexts. The retained
MLX-LM BF16 baseline reports 6.331, 6.139, 6.276, and 6.224 sustained
tokens/s at those contexts. Thus SG8 is *provisionally* 2.21×, 2.09×, 1.85×,
and 1.61× the older MLX decode throughput, but **not** a matched MLX win:
rvLLM measured two generated tokens in single runs; MLX measured 64 tokens
over seven trials, and the runs were not interleaved. Conversely, SG8 prefill
remains roughly 8.00×, 11.10×, 12.08×, and 14.27× slower than the retained
MLX prompt-throughput observations; the donor SG8 decode kernels are not a
prefill solution. The first SG8 64-token route observation is 6.663 tok/s at
256 context, just 5.2% above the retained 6.331 tok/s MLX BF16 value, but
that comparison is also non-interleaved and is **not** an established MLX
lead. It demonstrates how much the earlier two-token extrapolation can
overstate a cross-framework result in a changing host state. The exact
original MLX/rvLLM comparison and limitations are
in `../gemma4-rvllm-good-enough-matrix-20260924/README.md`.

Remaining gates: real-weight W4/W8 sidecar correctness, per-layer
and KV comparison, dedicated long-context attention correctness/timing,
checkpoint logits/continuation quality, and paired end-to-end timing against
both the default rvLLM route and MLX. Long-context BF16 continuation alone
does not close any of these numerical or comparison gates.
