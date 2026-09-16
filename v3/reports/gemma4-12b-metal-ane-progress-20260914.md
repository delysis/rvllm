# Gemma 4 12B: Metal prefill / ANE decode progress

## Pause checkpoint, 2026-09-16

The user requested a pause and a checkpoint on the fork's main branch. The
experiment worker is stopped, its STOP marker is retained, and no new device
trial was started for this checkpoint. Read [the current handoff](../HANDOFF.md)
for completed qualification, pending jobs, frozen identities and restart rules.
Older statements below about a running worker describe historical activity.
There is still no accepted llama.cpp slowdown ratio or S2 speedup.

## Latest checked state, 2026-09-16

**A conditional Rust experiment queue now runs independently of coding work.**
It accepts pinned jobs, waits for stable power/thermal/activity conditions,
serializes accelerator work, and preserves raw observations and failed
attempts. Ten host tests and live non-accelerator scheduling/failure smoke
checks pass. The pinned native-kit llama.cpp baseline now reproduces all 90
reference output IDs across nine 84-token prompts with nine continuation
evaluations each; logs confirm 49/49 layers offloaded to Metal. A fresh strict
ANE process also reproduced all 90 IDs with 162 cached programs and zero
compiles. Unrelated builds made both attempts ineligible for a clean speed
comparison. A new ABBA campaign requires two quiet minutes and separates
nominal AC and battery strata. No baseline slowdown factor has been established.
See the
[queue and comparison report](gemma4-experiment-queue-20260916.md).

Recent strict ANE loads failed after separately successful cache repairs.
Read-only macOS logs show active `aned` CacheDelete purge handling, without
public graph identities. Conservative removal of obsolete generated Cargo
archives initially increased disk headroom by approximately 20 GiB. One bounded
repair then restored 17 misses, and the fresh strict load succeeded. Later,
verified inactive August incremental caches were also removed under Cargo
locks; measured available space rose from 16.97 to 31.36 GB. The queue rejects
contaminated timings and stops on failure rather than retrying indefinitely.

Two explicit candidates were coded while experiments ran/waited:
[sliding-only INT8 QKV](gemma4-sliding-qkv-int8-candidate-20260916.md) preserves
FP16 global QKV, and [logical token batching in the INT8 FFN](gemma4-int8-ffn-logical-batch-20260916.md)
preserves the original single-token graph byte for byte. Host checks pass;
neither has new full-model or batched-device performance acceptance. The
[multi-token feasibility report](gemma4-ane-multitoken-verification-feasibility-20260916.md)
identifies Google's exact 12B assistant and its target hidden-state/shared-KV
requirements. No assistant weights have been downloaded or executed.

**Stacked INT8 performance exploration has resumed in a separate Fair stratum.**
Apple describes Fair as minimally elevated thermals with no required corrective
action, so the earlier blanket block was too restrictive for exploration.
The nominal-only eligibility gate remains unchanged. A separate offline reader
checks raw power samples and phase endpoints for stable, matching Fair controls;
the experiment uses repeated baselines and a predeclared drift gate. It does
not estimate device clocks or normalize ANE time using CPU cycles. No speedup
is yet claimed and the default kernel is unchanged. The initial timing attempt
stopped before decode on a missing cache entry; bounded cache maintenance is
separate from timing. See the [Fair exploration receipt](gemma4-stacked-int8-fair-exploration-20260916.md)
and the [earlier nominal comparison audit](gemma4-stacked-int8-comparison-20260916.md).

The CLI now accepts `--inspect-ane-cache all-int8` and
`--prepare-ane-cache all-int8`. Four serial children use the same signed
executable, with independent bounds of 48/48/48/18 graphs. The parent checks
each receipt against its driver journal and stops on failure. SIGINT/TERM/HUP
cancel between graph operations and join the child after unload. Host tests
pass; the first live inspection found 66/162 available with no compilation
or evaluation. A parent-only SIGTERM stopped during the FFN part after all
43 loaded graphs had been unloaded. No staging residue or reboot occurred.
Recovery and subsequent generation evidence are recorded in
[the cache batch report](gemma4-cache-batch-recovery-20260916.md).

The complete batch repaired 96 missing programs, then the same signed binary
loaded all 162 from cache with zero compiler calls. Four runtime-owner requests
(21, 84, 652, 21 prompt tokens) matched all 17 reference output IDs across
13 ANE decode steps. All 2,704 evaluations and 162 unload returns completed;
no staging residue or reboot occurred. The machine remained under thermal
pressure, so this establishes recovery and correctness, not a new speedup.
The baseline kernel/precision choices are unchanged.

The stacked INT8 FFN candidate now has a full-model numerical check. Its
explicit checked plan retains both FFNs per layer and feeds the candidate
output onward after requiring exact FP16 bit parity. All 624 comparisons
(13 decode steps across 48 layers) passed on 21/84/652/21-token prompts, with
all 17 output IDs matching references. All 210 programs were cache hits;
3,328 evaluations and 210 unload returns completed without failures or
staging residue. This is journaled duplicate work, not a speed measurement.
The default runtime/HTTP path is unchanged. See the
[stacked full-model report](gemma4-stacked-int8-full-model-20260916.md).

A separate candidate-only run then matched all ten copy-reference output IDs
with 162 cached programs, nine ANE steps, 1,872 evaluations, zero compiles and
162 clean unloads. This verifies the route without duplicate FFN checking.
Fair thermal state still prevents promotion based on measured performance.

**Current optimization path: INT8.** The user's clarified four-bit exception
requires both Apple documentation and corroborating results on this laptop,
using Google's official native QAT Gemma 4 weights. Locally post-quantizing the
ordinary BF16 checkpoint does not meet that requirement. Google publishes
[official Gemma 4 QAT artifacts](https://ai.google.dev/gemma/docs/core), including
12B; their availability does not establish an ANE benefit. The old grouped-
palette prototype remains removed from live source, with its historical CPU
evidence in `grouped-lut4-host-quality/`. Those errors do not evaluate Google's
QAT weights. No grouped palette was compiled or run on ANE. Existing INT8
inference remains the working path while the source-format audit proceeds.

The [native QAT audit](gemma4-native-qat-four-bit-viability-20260915.md) now pins
Google's official packed GGUF and compressed-tensors artifacts. Their group-32
layout has no demonstrated ANE advantage here. Both devices would need one
coherent QAT model identity. The old posthoc LUT4 results cannot satisfy that
qualification.

An experimental per-output INT8 linear constructor now shares the verified FFN
quantizer and existing single-I/O convolution graph. A captured FP16 projection
retains byte-identical MIL. Six quantizer tests and 105 Apple host tests pass;
the isolated device smoke passes all three synthetic inputs, with two compiles,
six evaluations, two successful unloads, no staging residue and unchanged boot.
Full-model use of these new INT8 projections remains unqualified, and the
full model still uses FP16 QKV/output/head. See the
[INT8 projection receipt](gemma4-int8-linear-projections-20260915.md).

Actual QKV layer 1 (sliding) and layer 5 (global) now pass backend qualification
on saved states from three successful requests. Per-component quantization
relative L2 error ranges 0.499–1.469%; ANE INT8 execution error against the
reconstructed-weight CPU reference is 0.0302–0.0344%, with zero backend gate
violations. Both original cache entries needed isolated restoration. The two
restores plus qualification used six compiler calls, 18 evaluations and eight
successful unloads, with no staging residue or reboot. Strict cached timing
attempts then rejected Fair thermal state before warmup; no speedup is claimed.
Evidence: `int8-qkv-real-probe/`. Full-model QKV promotion remains pending.

The vocabulary CPU audit now covers every one of the 262,144 rows across four
authenticated final-layer captures (positions 21, 84, 230 and 652). INT8
preserves all four winning tokens, with 0.269–0.447% raw-projection relative L2
error. One fifth-ranked alternative changes. The available captures have large
winner margins; close-ranking tokens, ANE head execution and full generation
remain unqualified. This audit performs no accelerator calls. Evidence:
`int8-head-host-quality/`. The full model still uses its original FP16 head.

Vocabulary tile zero now also passes actual ANE INT8 execution on all four
states, alongside original and reconstructed FP16 controls: zero backend
tolerance failures and agreement on eleven recorded logits within the tile.
Qualification used two compiles, twelve evaluations and three successful
unloads with no staging residue or reboot. Its first strict-cache timing run
rejected Fair thermal state before warmup; no speedup is established. The
652-token winner lies outside this tile, and full-vocabulary ANE ranking
remains pending. Evidence: `int8-head-device/` and the projection report.

An INT8 stacked gate/up FFN candidate now preserves every reconstructed bit
of all 176,947,200 layer-zero coefficients. Its two internal convolutions keep
the original GELU/down tail and existing single-I/O request ABI. A small ANE
check and the full-size four-input qualification produce bit-identical output
to the current INT8 FFN. The known baseline needed one isolated compile with
zero evaluations; qualification then needed only one new candidate compile,
twelve evaluations and three clean unloads. The original graph identity
matches the earlier HTTP cache hit. All 107 Apple host tests pass. Strict
cached timing rejects Fair thermal state before warmup, so no speedup or
full-model promotion is claimed. See the
[stacked FFN receipt](gemma4-stacked-int8-ffn-experiment-20260915.md) and
`int8-stacked-ffn/`.

When thermal state returned to nominal, three strict-cache QKV runs per layer
completed with identical sampled AC/Low-Power/nominal controls and zero compiles:
72 eligible pairs total. Sliding QKV's per-run median wall ratios versus original
FP16 were 1.130, 1.112 and 1.186, but individual comparisons remain noisy and its
first reconstructed-weight comparison regressed. Packed global INT8 was slower
in every pair of both later runs against both controls (median ratios versus
original FP16 0.639 and 0.548). Global QKV stays FP16; blanket INT8 promotion is
rejected. The next isolated candidate is split global INT8 Q plus FP16 shared
K/V, comparing the complete two-submission path. See the projection report and
`int8-qkv-real-probe/timing-repetitions-summary.json`; no full-model speedup is
claimed.

A split global QKV probe now qualifies INT8 Q8192 plus original FP16 shared
K/V512 through two existing single-I/O requests. Three captured inputs have
zero backend tolerance failures and bit-identical K/V outputs versus packed
FP16. Two compiles, nine evaluations and three unloads complete cleanly with
unchanged boot. Six eligible timing pairs give a median baseline/candidate
wall ratio of 1.066, but range 0.559–1.435. A second process rejects Fair
thermal state before warmup. This is inconclusive; packed FP16 remains the
default. Evidence: `int8-global-split/` and the projection report.

A CPU-only eight-bit palette representation audit now reconstructs all
176,947,200 coefficients of the actual layer-zero FFN exactly like the existing
per-row INT8 representation. Its reconstructed SHA and four CPU-input error
reports match the affine-INT8 baseline exactly. Five host tests pass. This is
a host encoding audit with zero ANE calls, not a speedup or a new quantizer;
source bytes rise from 177,016,768 to 194,642,368 because each row has a lookup
table. One-layer ANE numerical qualification subsequently passed: palette outputs
match the dense ANE control, with all programs unloaded. A strict-cache timing
run had zero compiles but all six pairs were rejected for Fair thermal state.
Its unfavorable raw observations do not establish a qualified performance ratio.
The candidate remains experimental, with no full-model expansion. Evidence:
`exact-int8-palette-host-quality/`; [experiment contract](gemma4-exact-int8-palette-experiment-20260915.md).

The real HTTP adapter now passes JSON completions, incremental SSE, queued
requests, disconnect cancellation and fresh-request recovery on Gemma 4 12B.
Five completed requests match all 26 reference output IDs. The interrupted
request stopped after four ANE decode steps, before its nine-step completion.
SIGTERM waited for all 162 successful unloads: 162 cache hits, zero compiler
calls, 5,200 completed evaluations, zero remaining owned staging directories,
and unchanged boot. The signed server SHA is
`a2e9f7a47d28d9a8fb2dc9587a70f0e5110c6371a4501da0297569f60d5a124f`.
Evidence: `http-worker-live/`. This durable-journal run is not a speed benchmark.
All 17 HTTP host tests and the Apple benchmark compatibility check pass.

An unjournaled run of the same signed server then passed two warmups and seven
identical measured copy requests (84 prompt tokens, 10 output tokens, nine ANE
steps). All nine outputs matched the reference and all seven measured requests
had the same eligible sampled controls: battery, low-power off, power mode zero,
nominal thermal state. Median ANE decode was 6.0859 steps/s (range 5.8012–6.1281),
client completion 2300.63 ms, prefill including KV import 802.65 ms, and completed
Metal GPU interval 449.55 ms. Median per-step components: FFN 74.82 ms, QKV
31.19 ms, output projection 21.34 ms, vocabulary 18.91 ms, attention 15.79 ms,
host 3.41 ms. Component medians are separate observations, not an additive total.
This is a characterized baseline, not a causal speedup over earlier unstratified
runs. SIGTERM again succeeded with zero owned staging folders and unchanged
boot. Evidence: `http-worker-measurements/summary.json` and its raw trials.

Checkpoint BF16 conversion now uses safe integer conversion after exhaustive
16-bit parity and six eligible release-profile ABBA pairs. Median paired wall
reduction is 35.69%, CPU-cycle reduction 35.26%, and instruction reduction
70.41% for actual layer-zero gate weights. This measures host conversion only.
Development-profile experiments had the opposite ranking and remain labeled
separately. See [conversion measurement](gemma4-bf16-checkpoint-conversion-20260915.md).

The serial runtime owner passed real A/B/A and cancellation recovery with the
21-token capital and 84-token copy prompts. Five completed requests produced
all 26 expected output IDs. Cancelling the copy request after an actual ANE
token allowed the queued capital request and a fresh copy request to match.
The durable journal records 162 cache hits/loads, zero compiler calls, 4,784
completed evaluations (23 ANE steps including interrupted work), and all 162
successful unloads at critical-pressure release. The boot stayed unchanged.
Evidence: `worker-shutdown-live-cancellation/`.

Live validation exposed a process-exit defect: the request API discarded its
thread handle, allowing the CLI to terminate during owner cleanup. The next
launch stopped at the existing-directory guard. Final handle destruction now
signals cancellation and joins the owner; full response queues cannot block it.
Two deterministic shutdown regressions pass. The 142 leftover staging
directories were matched to this exact run's creation interval and known MIL
digests, then preserved by rename under `worker-shutdown-staging-recovery/`.
No daemon cache was purged. The consolidated gate passed 101 Apple, 5 ANE-system
and 132 runtime host tests; server and FFI checks also passed.

The final release, SHA
`59633fd85ec8075ebdd624b1119ce97831c5483b3928e3b96f47e027bde2f016`,
then accepted ordinary text A/B/A through `--runtime-worker true`. All prompt
IDs and 14 output IDs matched the independent references. Its journal records
162 hits/loads, zero compiler calls, 2,288 completed evaluations (11 ANE steps)
and 162 successful unloads. After process exit, all 162 known caller staging
paths were absent. The boot stayed unchanged. Evidence:
`joined-worker-text-three/`. This verifies the normal CLI exit path as well as
the preceding explicit memory-pressure release test. The runtime worker remains
an explicit CLI option; the HTTP adapter now uses the same owner.

Cache inspection first found 62/162 available graphs. Reclaiming 35.44 GB of
inactive generated dependency archives under Cargo locks recovered roughly
33 GiB of space. The bounded four-part recovery repaired exactly the 100
identified misses, with zero evaluations. Subsequent strict runs loaded all
162 graphs with zero repairs. This restores observed availability; it does not
prove an eviction policy or guarantee persistence under future disk pressure.
See `strict-full-cache-inspection-20260915/`,
`inactive-archive-headroom-recovery-20260915/` and
`headroom-recovery-provision-20260915/`.

Every measured phase now records actual process CPU cycles/instructions and
sampled power mode, AC/battery and thermal controls. Metal GPU command-buffer
time is recorded separately. CPU cycles do not normalize GPU or ANE time, and
ANE cycles are unavailable on the qualified path. Current thermally ineligible
runs and the durable journal do not establish speedups. The vectorized INT8
host quantizer is the default after exact parity across all 48 checkpoint
FFNs (8,493,465,600 coefficients and all scales); its instruction reduction is
host preparation evidence only. Broader quality and valid before/after
matched-power end-to-end performance comparisons remain unfinished.

## Earlier model qualification and timing observations

The timing observations in this section predate power-state recording. They
must not be used as causal speed comparisons against the newer measurements.

**Current status:** Metal prefill and all 48 ANE decode layers pass seven reference prompts and all 31 token IDs (24 actual ANE steps). Ordinary text input also matches independent CPU references at 21, 84, 230 and 652 prompt tokens. Native BF16 MMA32 is qualified for the targeted prefill projections at 6–1024 tokens. Global ANE attention capacity is 1024; the maximum whole-model prompt directly tested so far is 652 tokens.

Per-output INT8 FFNs, with FP16 projections/head, reached **5.347 decode tokens/sec** in the latest seven-prompt run, using SIMD prefill attention and retaining both backends. The preceding retained run observed **4.909 tokens/sec**, zero explicit compilations, and all reference tokens matched. Another recent phased run observed 3.426 tokens/sec. These are sequential shared-host observations, not a promised sustained rate or an isolated causal comparison of residency policies. The original dynamic-FFN baseline was 0.317 under contention.

The INT8 model's captured final layer differs from the same-seed FP16 ANE path by 2.727% relative L2 (maximum 3.097% at layer 9) on the short capital prompt. At the 652-token prompt, all 48 layers were checked against a fresh FP16 CPU model using the actual Metal seed: maximum relative L2 3.411% at layer 9, final 1.676% with SIMD prefill attention, and the next token matched. Four-bit scalar LUT FFNs failed broader continuation testing and remain rejected for the working path. Broader quantized quality remains to be established.

Initialization remains costly: about 5–9 seconds for Metal and 38–73 seconds for ANE in recent cached runs. The text CLI works and streams ordinary single-user turns. The persistent stdin/interleaved path passed four late-arriving requests without reinitializing either backend; a subsequent unjournaled run also passed all four requests with zero compiler attempts. The final 21-token request completed in 0.765 seconds, with 556 ms prefill and a 189 ms ANE step; this is one shared-host observation. Runtime worker/server integration, cancellation/recovery, broader quality and further prefill/decode optimization remain unfinished.

Multiple ANE inputs/outputs remain rejected before private API calls after the earlier experiment caused a system reboot. The replacement uses qualified single-input attention graphs. Source blob sizes do not establish compressed runtime residency. Detailed sections below are chronological; later results supersede historical pending-status statements.

## Initial verified results (historical)

- Official `google/gemma-4-12B-it` checkpoint revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- 48 layers: 40 sliding (16 Q heads, 8 KV heads, dimension 256) and 8 global
  (16 Q heads, 1 KV head, dimension 512). Hidden 3840, FFN 15360.
- Pinned Transformers reference revision
  `3384908511545a9c146de65f01f3d29b90b2add6`, CPU BF16.
- Prompt IDs `[2,818,5279,529,7001,563]` (BOS plus “The capital of France is”).
- Release Metal BF16 matched all 16 greedy reference tokens. This establishes
  token agreement on this case, not full-logit or general-quality equivalence.
- Current release timing: prepare 12,953.49 ms, prefill 3,874.59 ms, 16 decode
  steps 14,016.21 ms, **1.1415 tokens/s**. This is not the requested fast result.
  These are single local observations, not an isolated performance suite.
- A separate real Metal prefill exported nonzero, finite K and V for all 48
  layers. The export was rejected before ticket collection, then succeeded
  after collection. BF16 KV converted once to FP16 and packed into the proposed
  single-input attention surfaces. Capturing took 36.89 ms; packing, hashing,
  and artifact writing together took 220.08 ms. These are handoff costs, not
  ANE decode timings or zero-copy measurements.

Evidence is in [`gemma4-12b-evidence-20260914`](gemma4-12b-evidence-20260914),
including the full Metal report, all-layer KV report, and both generated MIL
programs. The reference tokens/logits are in
`crates/rvllm-runtime/tests/reference/gemma4-12b-hf-capital-steps16.json`.
Actual packed first-sliding/first-global inputs remain under
`target/gemma4-12b-prefill-handoff/`; their SHA-256 values are in the report.

## Changes relevant to this path

- Recognize the official Gemma 4 unified checkpoint identity and root dtype.
- Read EOS IDs from the checkpoint's generation metadata. Token 107 is a
  newline, not an EOS token for this model; the previous hardcoded list stopped
  the CLI and worker incorrectly. Embedded packages read generation metadata
  only when it appears in the authenticated manifest.
- A reference mismatch now makes the diagnostic CLI return failure.
- Owned Rust wrappers for resident private-ANE projections and fused GELU FFN.
  The earlier single-projection/FFN measurements were recorded before the
  reboot; their `/tmp` artifacts were lost. They are not used as current proof
  of full-model execution.
- Replace the crashing four-input attention design with one packed
  `[1,C,1,S]` surface and one output. Host appends touch only Q/new K/new V/mask;
  existing KV remains resident. Both the replacement attention constructor and
  multi-tensor sys entry point fail before private framework calls.
- Add `ModelMetalBackend::export_ane_prefill` and a GPU-only
  `rvllm_prefill_handoff` executable to exercise actual allocator page ownership,
  completion ordering, dtype conversion, and the packed input layout.

The [incident report](ane-panic-20260914.md) identifies the test process and ANE
driver fault, the documented single-input contract that was overlooked, and
the remaining uncertainty. ANEForge's e5rt multi-input support does not establish
support in the separate `_ANEInMemoryModel` interface.

## Checks completed after containment

- Runtime library with `apple`: 160 passed, 92 ignored.
- Metal CLI: 11 passed, 6 ignored.
- Gemma model identity/config tests: 6 passed.
- Generation metadata tests: 2 passed.
- Packed attention host tests: 3 passed.
- Quarantine guard: 1 passed; no ANE call.
- KV page conversion test covers both F16/BF16, reordered physical pages, a
  partial tail poisoned with NaNs, invalid live values, duplicate pages and
  a busy backend. It is included in the runtime count above.
- `git diff --check` passed.

Existing compiler warnings remain. No ignored ANE hardware test has run after
the reboot. The large pre-existing dirty worktree was preserved; these results
do not certify all changes in it, and no commit or remote release was made.

## Reproduction

From `v3`, generate the current BF16 shader source and compile it for macOS:

```sh
cargo run -p rvllm-apple-metal --bin emit_metal_kernels -- bf16 target/rvllm-current-bf16.metal target/rvllm-current-bf16-pipelines.json
xcrun --toolchain Metal -sdk macosx metal -std=metal3.1 -c target/rvllm-current-bf16.metal -o target/rvllm-current-bf16.air
xcrun --toolchain Metal -sdk macosx metallib target/rvllm-current-bf16.air -o target/rvllm-current-bf16.metallib
cargo build --release -p rvllm-runtime --features apple --bin rvllm_metal_infer --bin rvllm_prefill_handoff
```

Set `RVLLM_METAL_METALLIB_BF16` to that absolute metallib path. Run the inference
CLI with the model directory, `--prompt 'The capital of France is'`,
`--max-new-tokens 16 --max-total-tokens 64 --large-model-opt-in`, the reference
JSON via `--hf-reference`, and `--json`. Run the handoff executable with
`--model-dir`, `--prompt-token-ids 2,818,5279,529,7001,563`,
`--max-total-tokens 64`, and a new `--output-dir`. It does not invoke ANE.

Binary/artifact identities used for the recorded runs:

| Artifact | SHA-256 |
| --- | --- |
| Metal inference executable | `86c9697faaa728f41f994b89821c7eb96421855b6f3018e088ec1379ee3052c4` |
| Prefill handoff executable | `74796989614c6184048d9ad8c790211ee06abafe708c912ac45450cc933b5c05` |
| BF16 metallib | `24086ed59b6e079a5e5ac219036c90eb81db1d489a72fdd356d4c79433a4a762` |
| HF reference JSON | `195d79a60539d125c879db41f00d9b9ded9deb0c643f5ba0e217e7a8a5e98ce7` |

## Work still required

Validate the corrected single-input attention program on hardware with durable
stage logs, then wire real Gemma QKV normalization/RoPE, attention, output
projection, FFN, residuals, layer scalars, final norm and vocabulary projection
into all 48 decoder layers. Connect the verified Metal cache export to that
decoder, establish full-generation agreement across prompts/contexts, measure
actual split-device throughput and optimize the dominant costs. Host layout
tests and a successful prefill export cannot substitute for these requirements.

## Additional work while ANE validation awaits approval

No ANE hardware execution was resumed. CPU reference work established that
precision must be checked against properly formatted model inputs:

- On the original raw completion prompt, pinned Hugging Face BF16 selects token
  9079 first, while the same checkpoint loaded as FP16 selects token 496.
  Independent `from_pretrained` calls confirm this difference while retaining
  FP32 rotary-frequency buffers in both models; see `fresh-dtype-reference.json`.
  The embedding scale follows the requested dtype: 62.0 in BF16 and 61.96875
  in FP16. This comparison includes the reference model's dtype policy, not
  only the projection weights' precision.
  The highest measured module activation was 4,136 in FP16, with no nonfinite
  values in the earlier captured module outputs. That range run used sequential
  `model.to(dtype)` conversions and is retained as a diagnostic observation.
  The independent-load token comparison establishes precision sensitivity;
  it does not establish an rvLLM-only bug or FP16 range overflow on this case.
- With the checkpoint's actual chat template and thinking disabled, BF16 and
  FP16 both produce `Paris` for the capital question and `323` for 17 × 19.
  See `chat-reference.json` and `range-reference.json` in the evidence folder.
  Those complete short generations used sequential dtype conversion. Fresh
  independent loads confirm the same first token for each chat prompt in both
  dtypes; they have not yet repeated the complete generations. These are
  upstream CPU results, not ANE acceptance.
- `gemma_decode_math.rs` implements FP32 RMS statistics with FP16 tensor
  boundaries, actual Gemma gamma, unit V normalization, proportional RoPE,
  residual addition and layer scaling. Eight upstream fixtures cover both 12B
  head dimensions and positions 0, 1, 511 and 1025. Two host tests pass, including
  malformed inputs and overflow errors. This helper is not yet wired into a
  full decoder.

The program count also needs deliberate handling. A proposed arrangement uses
48 static QKV programs, 48 static output projections, 16 vocabulary tiles, two
shared attention programs, and one shared dynamic-weight FFN: **115 compiled
programs**. That is below the roughly 119-compilation limit documented by
upstream, but neither the limit nor this arrangement has been validated for a
full model on this host. Request sharing still needs implementation and proof.
This count is provisional: additional global-attention capacity variants also
consume compilation slots. The current layout's largest aligned capacity is
32,736 tokens because `32 + 2 * capacity` must fit a 65,536-element axis.
It computes over that full capacity even when most positions are masked.
The 40 sliding layers need a 1,024-token ring, with absolute positions retained
for RoPE, while the eight global layers need separately budgeted capacities.
The host layout and quarantined decoder wrapper now implement the ring;
hardware validation and global-capacity switching remain outstanding. At the current
maximum capacity, the packed input byte formula gives 11,275,072,512 bytes for
all 48 layers; limiting the sliding layers to 1,024 slots would reduce this to
878,610,432 bytes. These are layout estimates, not device memory measurements.

`ane_ffn_layout.rs` generates the single-input dynamic FFN program and packs
each layer's weights into its own surface. Gate/up transposes use 32×32 tiles;
the activation rows can be updated without rewriting weights. Two host tests
verify all three matrix orientations and the 12B dimensions. The actual layer-0
weights packed in **770.64 ms**, producing a 354,140,160-byte input with SHA-256
`5e1154ba22fc91fa62cbe991ad2351776eb971df39117320c0e4bc9466aeb2a0`.
The converted weights' SHA-256 is
`2def7d665d86b519f8c50369914aa30c07f7d73282fa0633b1a47f15a17ffdd3`.
See `dynamic-ffn-prepare.json` and `dynamic-ffn.mil`; the large input is under
`target/gemma4-12b-dynamic-ffn/`. This was CPU-only preparation; no program was
compiled or dispatched to ANE.

The existing probe now supports `--prepare-ffn-output <new-directory>` together
with `--model-dir` and `--ffn-prefix`. This mode returns before any private API
construction. Weight provenance hashing now feeds SHA-256 in 4 KiB chunks,
preserving the same canonical little-endian FP16 byte sequence.

Private API validation can opt into durable phase records by setting
`RVLLM_ANE_DIAGNOSTIC_JOURNAL` to a **new** file outside `/tmp`. Each phase is
synced before continuing. Existing logs are never truncated. The logger does
not capture prompts or tensor values, and must be disabled for performance
measurement. A host-only test verifies readable records, exclusive creation
and rejection of invalid model identifiers. It does not establish containment
of a driver failure.

## Weight traffic constrains the optimization plan

The official checkpoint's tensor shapes imply 16,986,931,200 bytes of FP16 FFN
weights, 4,812,963,840 bytes of attention projection weights, and 2,013,265,920
bytes for the tied vocabulary projection. Reading each once costs
**23,813,160,960 bytes per decoded token**, before KV, intermediate tensors,
compiled-layout expansion, or repeated reads. See `weight-budget.json` in the
evidence folder. This is a shape-derived estimate, not measured device traffic.

The historical layer-0 constant-weight fused FFN observation was 3.076792 ms.
If representative of all 48 layers, FFNs alone would take 147.686 ms/token,
limiting generation to about 6.77 tokens/s before other work. Its original
artifacts were lost in the reboot, and it does not establish dynamic-weight
FFN speed. A hypothetical uniform 115 GB/s over all projection weights would
take 207.071 ms/token. Neither extrapolation is a full-decoder benchmark.

These budgets make weight movement and amortization central to further work.
Keeping programs, requests, weights and KV resident removes avoidable setup
and copies; it does not make all model weights fit in on-chip memory. Quantized
weight execution or verified multi-token work may be needed for substantially
higher throughput. Both require independent correctness and hardware evidence.

Apple's [compression guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
distinguishes weight formats from actual runtime traffic: a compiler can expand
weights before execution, while other paths decompress during computation.
Its [palettization performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
identifies the latter behavior as the condition for bandwidth savings on ANE.
Neither document establishes that our private MIL path receives those kernels.

Before fixing the 115-program arrangement, compare the dynamic FP16 FFN with
a static compressed FFN. FFNs account for about 71.3% of the projection and
head bytes. An alternative budget is 48 static FFNs, two shape-shared dynamic
QKV programs, two shape-shared dynamic output projections, 16 vocabulary
tiles and two attention programs: **70 programs**, before additional context
variants. This would reserve compression opportunities for the largest weight
group while sharing programs for the smaller projections. Program sharing,
compression acceptance, decompression behavior, accuracy and performance all
remain unverified; this is a candidate experiment, not the chosen runtime.

## Approach refined by the megakernel research

The [source-pinned research report](megakernel-gigakernel-research-20260914.md)
reviews Hazy, Mirage MPK, Cohere, Luce, MLX and several counterexamples. Its
useful transfers are reusable program/schedule ownership, resident data,
precise dependencies and phase-specific kernels. The normal Metal path already
uses one command buffer per step, so further command reuse needs measured CPU
encoding overhead to justify it. CUDA cooperative scheduling and barriers are
not an established implementation route for either Metal or the private ANE
graph interface.

The next optimization decisions are now explicit:

1. Attribute actual QKV, attention, output projection, FFN, vocabulary and host
   costs before choosing additional fusion.
2. Establish request/program reuse and the corrected bounded attention path,
   followed by one real layer and all 48 layers. Hardware validation still
   awaits approval after the panic.
3. Compare static compressed FFNs with dynamic FP16 FFNs before selecting the
   program budget; include compiled weight expansion and actual execution cost.
4. Validate the implemented sliding ring on ANE and budget global-capacity variants. Validate the
   mixed BF16-prefill/FP16-decode path against an oracle initialized with the
   same converted KV, alongside full-generation quality checks.
5. Optimize Metal prefill with tiled GEMMs and attention specialized for both
   head dimensions. Consider cached dispatch only after identifying remaining
   encoding overhead; reserve whole-model persistence and speculation for
   later measured bottlenecks.

This review changed the plan and evidence record. It did not demonstrate a new
hardware speedup or complete the ANE decoder.

## Sliding-ring implementation and host validation

`PackedAttentionLayout::sliding` fixes the attention window at construction,
rounds storage up to 32 slots, and maps absolute positions into physical ring
slots. Prefill import keeps only the last window of K/V. Decode updates one
slot and masks both stale entries and padding; it never shifts the retained
cache. The ANE wrapper tracks `tokens_seen` independently of storage capacity
so callers retain the absolute RoPE position after wraparound. Both global and
sliding constructors still return the quarantine error before private API use.

The handoff executable selects the sliding layout from checkpoint metadata and
records capacity and retained-token count per layer. It validates all packed
shapes before Metal initialization, then checks them against the exported cache.
The revised executable has passed compilation checks but has not yet been run
for a new hardware handoff measurement; the earlier evidence remains tied to
the earlier executable and layout.

Host tests compare attention from the packed physical slots with a separate
calculation over the original chronological cache. They cover real 12B sliding
head geometry at cached-token counts 1023–1027 and after a 2053-token prefill, plus a
three-token window crossing padded 32-slot storage. All compared outputs agree
within 1e-5 in FP32. Five relevant tests passed; the ANE hardware test remained
ignored. See `ane-ring-host-tests.log` in the evidence folder. This establishes
host layout and mask correctness on those cases, not ANE attention accuracy or
speed. No private hardware execution was resumed.

## Metal System Trace attribution

A bounded Metal System Trace run of the existing release executable produced
the expected first token (9079, ` Paris`). The executable and BF16 metallib
hashes match the earlier recorded identities. The run measured 2,314.05 ms for
the six-token prefill; other builds were active, and tracing was enabled, so
this is diagnostic evidence rather than a peak-throughput comparison. CPU
encoding accounted for 0.529% of the measured prefill-plus-decode wall time.

The trace contains 529 actual prefill encoders and 532 decode encoders. The
runtime's estimated counts are higher by one encoder per layer. Encoders have
generic names, so stage attribution was inferred from the checked eleven-op
layer sequence, not obtained from shader labels. Overlapping Compute intervals
were unioned within each encoder; other GPU channels were excluded. Summed
per-encoder duration is not the same metric as whole-device wall time.

| Prefill stage | Summed encoder GPU time | Share |
| --- | ---: | ---: |
| Gate/up projection | 860.06 ms | 48.57% |
| Down projection | 377.02 ms | 21.29% |
| QKV, normalization, RoPE and cache write | 204.73 ms | 11.56% |
| Output projection | 142.90 ms | 8.07% |
| Attention | 88.32 ms | 4.99% |

See `metal-profile-run.json`, `metal-stage-profile.json` and
`metal-encoder-timings.json` in the evidence folder. The full trace and XML
exports remain under `target/gemma4-12b-metal-prefill-profile*` and
`target/gemma4-12b-metal-*.xml`. This evidence directs the next Metal experiment
toward sharing FFN weight reads across prompt rows. The existing eight-row
kernel already masks partial rows; its wider dispatch policy remains a
candidate until numerical and performance checks pass.

That dispatch experiment was subsequently **rejected and reverted**. Extending
the existing eight-row kernel to 2–19 rows changed the first generated token
from 9079 to 496 and failed the complete 16-token reference comparison, despite
all reported compute/weight/cache dtypes remaining BF16. The kernels accumulate
and reduce K differently; the numerical divergence needs finer-grained
investigation before either can substitute for the other on this checkpoint.

The candidate measured 2,230.98 ms prefill, while the restored build measured
2,396.08 ms. The unchanged decode path varied substantially too, so these runs
do not establish a speedup. The restored build passed all 16 reference tokens.
Its projection source is byte-identical to the pre-experiment copy; no candidate
dispatch change remains. Evidence: `metal-row-reuse-rejected.json` and
`metal-restored-after-row-reuse.json` in the evidence folder.

| Executable | SHA-256 |
| --- | --- |
| Rejected row-reuse candidate | `4db6d65f4304bc81b344331310344b19b336c56d238a0ab59283313ef0d7f531` |
| Current restored release | `ca0d094ff2a4c10de5f18b7f07f90a855543d0a0ee4f8ab82cc64b60cd2fba8b` |

The next projection experiment should preserve the established FP32 reduction
order while reducing synchronization overhead, with bitwise kernel comparison
before a full-model run. The trace also exposes an independent estimated
encoder-count discrepancy; its counts should not be used as measured dispatch
counts. No ANE hardware execution occurred during this work.

## Exact-order GEMV reduction experiment

The bounded experiment replaced eight threadgroup reduction barriers with
explicit SIMD shuffle stages. It preserved the original 128/64/32 and
16/8/4/2/1 addition tree. Disabling compiler reassociation was necessary:
the first FP16 trial without that restriction differed in 130 of 184,320
gate/up outputs. The final candidate used one SIMD group per output column,
with the original threadgroup reduction as a fallback for other SIMD widths.

The final ignored public-Metal comparison test passed all twelve cases with
zero output-bit differences and intact output guards. Cases cover FP16 and
BF16, ragged dimensions, one-token decode, six-token gate/up and down
projections, nineteen-row output projection, and nontrivial alpha/beta.
Timings use completed command buffers' GPU start/end timestamps and alternate
baseline/candidate order. This is a synthetic kernel comparison, not a broad
model-accuracy qualification. See `gemv-reduction-final-harness.json` and its
log in the evidence folder.

The same release executable was tested with three compiled shader libraries.
All eight raw-completion runs matched all sixteen reference tokens. Measured
times varied too much to establish a repeatable prefill gain:

| Reduction | Six-token prefill, ms | Sixteen-token decode, ms |
| --- | --- | --- |
| Original, run 1 | 1,948.90 | 6,922.92 |
| Original, run 2 | 2,715.63 | 9,954.10 |
| Original, run 3 | 2,727.15 | 10,058.94 |
| One SIMD group, run 1 | 2,065.41 | 6,450.22 |
| One SIMD group, run 2 | 2,079.03 | 6,830.19 |
| Eight SIMD groups, run 1 | 2,978.69 | 5,591.10 |
| Eight SIMD groups, run 2 | 949.76 | 5,304.99 |
| Eight SIMD groups, final session | 2,826.95 | 9,773.29 |

Other workloads were active; these are observations, not an isolated causal
speedup estimate. **No production reduction change was retained.** The source
was restored byte for byte to the pre-experiment copy, including the new test
module declaration. Only the ignored comparison harness contains the candidate.
The default BF16 metallib and release executable retain their recorded hashes.
The complete observations and candidate library identities are in
`metal-gemv-reduction-experiment.json`.

The final candidate session also ran two prompts formatted with the checkpoint's
default non-thinking chat template. Their input token IDs and first output
token match the independently loaded BF16 Transformers oracle. Observed full
outputs were `Paris` (tokens `[50429,106]`) and `323` (tokens
`[236800,236778,236800,106]`), both stopping at model EOS. These are observed
complete answers plus first-token oracle agreement, not complete-generation
logit comparisons. See `metal-gemv-reduction-chat-session.json`.

Those 21- and 28-token chat prefills took 7,543.79 and 8,718.25 ms respectively.
The longer-prompt projection path is the next profiling/optimization target;
the short GEMV reduction alone does not establish efficient GPU prefill.
ANE hardware remained paused throughout; none of these results establish
ANE decode or completed split-device inference.

## Gemma 4 12B prompt projection dispatch

Inspection found that the generic cooperative-projection cutoff of nineteen
rows came from E2B measurements. Larger prompts use the serial projection
fallback, including the fused projection/norm path. Simply raising the cutoff
to thirty-two rows did not establish a gain and was not retained as a general
policy.

The retained dispatch change shares weights across eight prompt rows using the
existing `gemm_f16_batch8` shader. Its new eligibility is limited to 20–32 rows
and the exact 12B projection shapes `(N,K)`: gate/up `(30720,3840)`, output
`(3840,4096)` or `(3840,8192)`, and down `(3840,15360)`. The Apple9/10 family
gate remains. The smaller E2B policy and 2–19-token reduction policy remain
unchanged. The corresponding projection/norm path uses two encoders so it can
select the row-sharing projection rather than the serial fused kernel.

The numerical cache ABI was incremented from five to six with an explicit
dispatch-policy identity. Even when generated tokens agree, a different
reduction order can change cached values; old persisted prompt caches must
not silently cross that boundary. Four existing host policy tests and the
numerical-fingerprint separation test passed. `git diff --check` passed.

Six new full-generation references were generated from a fresh CPU BF16 load
of the pinned Transformers implementation, preserving FP32 rotary buffers.
They cover complete responses `Paris`, `323`, `42`, `Cold`, `Au`, and `CPU`
for 21-, 28-, 27-, 23-, 28-, and 22-token chat prompts respectively. Each uses
the checkpoint's default non-thinking template, greedy generation with an
eight-token maximum, and its model-provided EOS configuration. The saved
reference-computation source is `fresh-chat-oracle-source.txt` in the evidence
folder. No dtype conversion was applied to an already-loaded model.

The source/CLI qualification set is
`crates/rvllm-runtime/tests/reference/gemma4-12b-prefill-qualification.jsonl`.
It includes those six complete chat references and the existing sixteen-token
raw-completion reference. The scoped release executable passed all seven;
the original release also passed all seven. This is a bounded BF16 text
qualification, not long-context or ANE acceptance.

The scoped executable passed the complete set again after the baseline run:
all 31 generated token IDs agree with the references in each scoped session.
Across the six chat prompts, measured total prefill was 71.43 seconds for the
baseline, 20.45 seconds for the first scoped session, and 32.41 seconds for
the scoped repeat. The repeat improved the observed prefill time on every
chat case. However, unchanged chat decode time also varied, from 17.92 seconds
in the baseline to 13.13 seconds in the scoped repeat. Other workloads were
active. These observations support retaining the bounded dispatch change but
do not establish an isolated 2.2x speedup or peak device performance.

The final scope and full observations are recorded in
`metal-prefill32-qualification.json`, `metal-prefill32-baseline-seven.json`,
`metal-prefill32-scoped-seven.json`, and
`metal-prefill32-scoped-seven-repeat.json` in the evidence folder.

| Artifact | SHA-256 |
| --- | --- |
| Original comparison executable | `ca0d094ff2a4c10de5f18b7f07f90a855543d0a0ee4f8ab82cc64b60cd2fba8b` |
| Current scoped release executable | `ba7c3b06dfea3bc84f367cd07c67f1be5b3d72a069fd16960dc38656b8572a10` |
| Unchanged default BF16 metallib | `24086ed59b6e079a5e5ac219036c90eb81db1d489a72fdd356d4c79433a4a762` |

Reproduce the qualification with the current release executable, the recorded
BF16 metallib, the official checkpoint path above, and
`--prompts-jsonl crates/rvllm-runtime/tests/reference/gemma4-12b-prefill-qualification.jsonl
--large-model-opt-in --json`. Each case carries its own reference path,
token limit, and BOS policy. ANE hardware was not executed. The full
Metal-prefill/ANE-decode objective remains unfinished.

## Sampling the first token during prefill and transferring decoder state

`ModelMetalBackend::prefill_for_ane` now runs a complete single-request prompt,
samples its first output from the final residual row in the same Metal command
buffer, collects that submission, and exports the prompt KV cache. The returned
`AneDecodeStart` owns the first token and converted cache. A decoder consumes
that first generated token at absolute position `prompt.len()`; it must not
replay the final prompt token, and it must stop before consuming an EOS token.

The operation rejects incomplete/chunked prompt plans, duplicate or missing
physical pages, inconsistent logical-page/slot mappings, stale model-layout
fingerprints, outstanding GPU ownership, shared-layer KV, and debug execution
that could skip part of the model. It uses the existing Metal sampler and
execution-slot lifecycle. The page-plan host test includes discontiguous
physical pages `[2,0]` for a 33-token prompt. The public Metal ownership test
checks rejection while an uncollected submission owns the cache.

The `rvllm_prefill_handoff` probe accepts `--hf-reference`, validates the exact
prompt IDs and model EOS policy, and fails if the first token differs. Its
optional Metal continuation verifier feeds each actual generated token into
the next absolute position and compares the full sequence. That verifier uses
the original Metal cache; it is explicitly not evidence of ANE consumption.
The probe additionally exports every layer's converted K/V tensor, in logical
`[token,kv_head,head_dim]` order, with little-endian FP16 bytes and SHA-256 hashes.

For the six-token raw completion, one prefill produced token `9079`, returned
decode position six, and exported all 48 layers. Counters were exactly one
prefill, zero decode steps and one command buffer before the continuation
verifier. The subsequent fifteen Metal decode steps matched all sixteen
reference output IDs. The final export run observed 4,093.80 ms for prefill,
first-token sampling and capture; that is one observation under other system
load, not an isolated benchmark. Writing and checking diagnostic packed inputs
is separately timed and is not part of the runtime's handoff method.

An independent fresh-FP16 CPU Transformers decoder then imported the actual
96 exported K/V files, verified every hash and shape, and continued from token
`9079` at position six. It preserved FP32 rotary buffers and used the pinned
upstream implementation with eager attention. Its complete sixteen-token
sequence also matched the BF16 reference. This establishes a bounded numerical
check of **Metal BF16 prefill -> converted FP16 KV -> FP16 CPU decode**. It does
not establish ANE arithmetic, compiler support, driver stability, or speed.

The raw-seed report, independent CPU reference, source, and logs are in
`reports/gemma4-12b-evidence-20260914/prefill-first-token-*`. The source-only
backend gate passed 46 tests, with 92 hardware tests left ignored; the separate
ownership test executed through public Metal and passed. ANE hardware remains
paused after the panic while approval of the corrected single-input attention
validation is pending.

The same handoff and independent converted-cache verification also passed all
six complete chat cases below. Across the raw completion and chats, all 31
generated token IDs match the pinned BF16 references. The CPU decoder consumes
each actual sampled token; subsequent tokens are not teacher-forced. Every
case uses one prefill command buffer, zero replayed decode steps before export,
and all 48 cache layers. This is token agreement, not bitwise logit agreement.

| Chat case | Prompt tokens | Complete answer | Metal prefill/sample/capture observation |
| --- | ---: | --- | ---: |
| Capital | 21 | Paris | 3,996.63 ms |
| Arithmetic | 28 | 323 | 3,545.08 ms |
| Addition | 27 | 42 | 4,554.14 ms |
| Opposite | 23 | Cold | 3,625.61 ms |
| Gold | 28 | Au | 3,509.45 ms |
| CPU | 22 | CPU | 2,589.37 ms |

These timings include synchronous cache capture and are not directly comparable
to the older CLI's prefill-only measurements. They are single observations
under other system load. The full machine-readable evidence is
`prefill-first-token-hybrid-qualification.json` and
`prefill-first-token-chat-metal-qualification.json`. The six chat runs used
`rvllm_prefill_handoff` SHA-256
`670a3d886a8206ae06ebaef970231878675823a008b7a570f36c6d3463067640`
and the unchanged BF16 metallib recorded above. The raw run preceded a
formatting-only final rebuild; its executable hash was not recorded. The
manifest identifies that provenance boundary explicitly.

To reproduce a seed, build `rvllm_prefill_handoff` in release mode with the
`apple` feature, set `RVLLM_METAL_METALLIB_BF16` to the recorded library, and
supply `--model-dir`, the reference's exact comma-separated
`--prompt-token-ids`, `--max-total-tokens 64`, `--hf-reference`, and a new
`--output-dir`. The saved `prefill-first-token-cpu-oracle-source.txt` accepts
`--seed-dir`, `--reference`, and a new `--output` JSON path. Run it through
`uv run --no-project --python 3.12 --with torch --with accelerate` plus
`--with 'transformers @ git+https://github.com/huggingface/transformers.git@3384908511545a9c146de65f01f3d29b90b2add6'`.
The binary tensor files remain reproducible diagnostic artifacts under
`target/gemma4-12b-hybrid-seed*`; the durable reports retain their hashes.

## Resume boundary for full ANE decode

No ANE operation was executed after the panic. The corrected single-input
attention test is the next hardware qualification, and explicit approval is
still pending. This is a whole-host reboot risk; a subprocess is not isolation.
If approved, compile only the exact ignored attention test with its reviewed
guard change, then restore the source quarantine **before** executing that
test binary. Record the binary hash and use a fresh durable diagnostic journal
outside `/tmp`. Run that bounded test once, retain any failure, and do not
automatically retry or expand it into full-model experiments.

A passing test would establish only the tested attention shapes. Full ANE
decode still needs reusable compiled FFN programs with separately owned
resident per-layer inputs, the 48-layer decoder and vocabulary head wired to
the verified handoff, complete generation/EOS qualification, and sustained
latency/memory measurements. Long-context wraparound and concurrent Metal/ANE
work also remain unqualified. The megakernel research informs these priorities:
measure weight traffic and reusable operator execution before introducing a
persistent scheduler. The overall goal is not complete.

## Approved attention hardware result

The user approved the corrected single-input test. It passed once on
2026-09-14 at 16:16 UTC: both head geometries, two compiled/loaded programs,
six completed evaluations and two unload returns. All 36,864 compared values
satisfied the 0.003 absolute-error limit. The boot time was unchanged. The
source guard was restored and its host test passed before the isolated approved
executable was launched. No retry occurred.

The durable source snapshot, executable hash, run intent, phase journal and
result are in `ane-approved-attention-20260914T161346Z/` within the evidence
directory. This resolves the pending approval for that specific test. It does
not qualify the 1,024-token sliding capacity, full decoder, sustained stability
or performance; the small sliding test used a three-token mask and 32 physical
slots. The source quarantine remains enabled while decoder integration proceeds.

## Shared FFN programs and activation accuracy

The private API wrapper now separates one compiled/loaded program from its
independently owned requests and IOSurfaces. `AneInMemoryProgram` uses `Rc` to
keep ownership on one thread; each request holds the program alive. The last
program owner unloads it. If that owner is a kernel, unload happens before its
request and surfaces are released. Creating additional requests does not
compile or load again. Compilation/load now precedes request allocation,
matching the inspected upstream bridge order.

`AneDynamicFfnProgram` builds the weight-independent graph and creates one
resident input surface per layer. `AneDynamicFfn::project` writes only the
hidden-size activation lanes, evaluates, and reads the output. It does not
repack weights or allocate in the token loop. This is the mechanism intended
to share one graph across all 48 FFNs; it is not yet a full decoder.

The first 64-by-128 FFN test compiled and evaluated but failed the unchanged
2% relative-error gate at its first output. An instrumented reproduction
measured 2.8049% relative error, with maximum absolute error about 0.00122.
Neither run produced a driver failure or reboot. Intermediate diagnostic
graphs isolated a large increase at the native GELU operation:

| Intermediate | Relative error | Maximum absolute error |
| --- | ---: | ---: |
| Gate projection | 0.03480% | 0.00024414 |
| Up projection | 0.003284% | 0.00003052 |
| Native GELU output | 1.5858% | 0.00451660 |
| Gated product | 1.6727% | 0.00222778 |

Those tap graphs are diagnostic observations, not a passing FFN qualification;
exposing a tensor can change compiler fusion. Simple CPU product/accumulator
rounding simulations did not explain the observed discrepancy. The newer
[ANE numerical discussion](https://github.com/sbryngelson/ane-guide/blob/main/part-1-machine/01-what-the-ane-is.md)
was consulted as a hypothesis source, but its general rounding explanation
does not establish the cause of this local activation error.

The dynamic FFN now expands Gemma's tanh GELU into explicit arithmetic and
`tanh` operations instead of the native `gelu` operation. The same FFN test
then passed without loosening either its relative or absolute limits. The
phase journal confirms one compile/load, two independently resident weight
inputs, seven completed evaluations and one unload. The test drops the
original program owner before evaluating either request, alternates different
weight sets, drops one request, and verifies that the remaining request still
works. The 12B dimensions have not yet been qualified by this small fixture.

Evidence and source snapshots are in `ane-shared-ffn-20260914/`,
`ane-shared-ffn-numerics-20260914/`, `ane-ffn-stages-20260914/`, and
`ane-shared-ffn-explicit-gelu-20260914/`. Journaling was enabled, so these
durations are not performance measurements. The real-weight probe now accepts
`--ffn-mode dynamic` for the subsequent 12B layer qualification.

## Real 12B dynamic FFN result and timing boundary

The actual layer-0 3840/15360 dynamic FFN passed three distinct input checks
against FP32 CPU dot products with explicit FP16 intermediate boundaries.
Maximum absolute error was 0.00385658; aggregate relative L2 error was
0.002755886. The diagnostic run completed all nine evaluations and returned
normally with unchanged boot identity. This establishes this one FFN layer,
not full-model generation.

The same binary and weights ran a separate journal-disabled 30-iteration
measurement: median 58.250208 ms, minimum 23.409375 ms, p95 93.270375 ms.
These are contended application-call timings, not an isolated device benchmark.
Subsequent inspection found load averages of 154.91/106.05/62.67 and many
compiler processes from other workspaces. The timing cannot establish the
intrinsic dynamic-weight cost or a speedup/slowdown versus the historical
constant-weight result. Host I/O versus evaluation attribution remains needed.

The evidence directories are `ane-dynamic-12b-layer0-20260914/` and
`ane-dynamic-12b-layer0-benchmark-20260914/`. Both record executable SHA-256
`4f496e2aaf6569a3598eef373ad0e62bca58be3f1aede7525574d2c8c865111d`,
weight SHA-256 `2def7d665d86b519f8c50369914aa30c07f7d73282fa0633b1a47f15a17ffdd3`,
commands, exit status and boot identity. Timings include host I/O; loading and
one-time weight packing are excluded from the timed loop.

## Full-window attention qualification: numerical failure retained

The shared-request attention test used two independent caches with the real
16-query-head / 8-KV-head / 256-dimension / 1024-window layout. It completed
two evaluations and then failed the unchanged 0.003 absolute-error check:
second request, first append, element 87 was 0.1685791 versus FP32 reference
0.17185758. The program unloaded and boot identity was unchanged. The run
stopped before any physical wrap and before the global shape. It is not a
passing attention qualification. Evidence: `ane-shared-attention-wrap-20260914/`.

The earlier short-context attention pass does not establish full-window
accuracy. A diagnostic probability-output graph is being prepared to separate
softmax normalization from the probability/value product reduction. No error
threshold has been relaxed and the source compile guard remains disabled.

`gemma_ane_decode.rs` now contains the complete resident 48-layer decode
orchestration, including shared FFN/attention programs, QKV and O projections,
16 tied-vocabulary tiles, embedding row reads, FP16 norms/RoPE/residuals,
position ownership and stage timings. `cargo check` passed. This is unexecuted
integration code, not full-model ANE acceptance; the attention guard prevents
loading it. Any failed step invalidates decoder state until a complete cache
import succeeds.

## Corrected full-window attention and qualified geometry guard

The probability-output diagnostic returned sums between 0.999159 and 1.000129.
At the previously failing element, a CPU-wide product using the actual ANE
probabilities gave 0.17177466, close to the independent 0.17185758 reference,
versus 0.1685791 from the original complete attention graph. This points to the
probability/value reduction, not a large softmax-normalization error. Exposing
an intermediate can change compiler fusion, so this is diagnostic evidence,
not a proof of the hardware's internal reduction order.

The [primary reverse-engineering numerics chapter](https://ane-guide.readthedocs.io/en/latest/part-1-machine/03-numerics.html)
reports subnormal flushing inside M1 matrix reductions and explicitly leaves
M2-through-M4 unmeasured. That supplied a scaling hypothesis, not a confirmed
M4 mechanism. The simple CPU flush simulation did not exactly reproduce the
observed output.

The retained graph multiplies probabilities by 32 before the value matmul and
multiplies its result by 1/32. Both factors are exactly representable powers of
two. With Gemma's unit-RMS values, this keeps the maximum scaled weighted value
well within FP16 range. The unchanged numerical test then passed all 86,016
output comparisons at the original 0.003 absolute bound. Its journal records
two compiles/loads, four independently owned requests, fourteen evaluations
and two unload returns, with unchanged boot identity. It includes sliding
physical wrap from an imported 1023 positions and global fill from 60 to 64.
The source creator and one request are dropped while the other remains usable.
Evidence: `ane-attention-probabilities-20260914/` and
`ane-attention-scaled-probabilities-20260914/`.

The blanket attention quarantine is now replaced by an exact geometry guard:
16 Q / 8 KV / dimension 256 / sliding-1024, or 16 Q / 1 KV / dimension 512 /
global-64. All other geometries fail before compilation. The multi-input/output
quarantine in the sys crate is unchanged. These are bounded fixture results;
full-model generation and longer global contexts still require verification.

## Attributed FFN timings

The journal-disabled, 12-sample stage profile on actual layer-0 weights reported
median input staging 0.039855 ms, ANE evaluate-call 18.373688 ms, and output
collection 0.011333 ms. Whole-call upper-median was 19.464792 ms, minimum
17.324042 ms and p95 42.791750 ms. The three numerical checks still passed
with identical reported error. Host copying is not the major observed cost;
optimization should examine the compiled dynamic matmuls and weight movement.
These are application-call timings under recorded concurrent machine load,
including scheduler waits, not hardware counters or a speedup over the prior
58.25 ms run. Evidence: `ane-dynamic-12b-stage-profile-20260914/`.

The new decoder's two focused host tests passed for embedding tile offsets,
row selection, FP16/BF16 decoding and nonfinite/overflow rejection. Its current
capacity is explicitly limited to the qualified 64-position global attention
shape. The full 48-layer hardware path remains unexecuted at this point.

## First complete 48-layer Metal-to-ANE decode passed

The fresh capital chat prompt produced `[50429, 106]` (`Paris`, EOS): Metal
prefill sampled token 50429 and exported all 48 converted KV tensors; ANE
consumed that actual token at position 21 and selected EOS 106. No Metal decode
step or CPU/GPU dense/attention fallback was used. CPU Rust handled the small
normalizations, RoPE, residual/scalar operations and greedy selection.

All 115 programs compiled and loaded, 208 requests were created, the one full
ANE step completed 208 evaluations, and all 115 models unloaded. The process
returned success and boot identity was unchanged. The exact binary SHA-256 is
`4a14f14c2276781c4f075960c696c17acc588b34550d95f39a990dcec91906e1`.
Evidence: `disaggregated-capital-20260914/`.

A fresh pinned FP16 CPU Transformer consumed the exact exported KV and compared
all 48 captured post-scalar layer residuals. It independently selected EOS 106.
The largest relative L2 difference was 0.004415517 at layer
46; the final layer's was 0.002795559. These are
observed errors, not a newly invented acceptance threshold. Its top five token
IDs also matched ANE. Source, logs and comparisons are in the same evidence
folder (`cpu-layer-oracle-source.txt`, `cpu-layer-comparison.json`).

This is one complete two-token response, not broad model qualification or a
speed result. Journaling and first-step layer captures were enabled. ANE
initialization took 183.98 s and the diagnostic decode call 7.43 s; those
measurements include logging and concurrent scheduling. A seven-prompt run of
the same binary without journaling/captures is now expanding numerical coverage
and measuring contended application timings. The full objective is still open.

## Seven-prompt end-to-end qualification passed; performance remains inadequate

The same binary passed all seven fresh Metal-prefill/ANE-decode cases, matching
all 31 pinned reference token IDs. Its one 115-program initialization was reused
across the seven requests, with a complete 48-layer cache import per request.
The run performed seven prefill command buffers, zero Metal decode steps and
24 full ANE decode steps. EOS stopping and the raw 16-token completion both
matched. No dense/attention CPU or GPU fallback occurred. The process exited
successfully and boot identity was unchanged. Evidence:
`disaggregated-seven-prompts-20260914/summary.json` and `inference/report.json`.

Journaling and layer captures were disabled. The measured 24 decode steps took
75.723 s in total (0.3169 tokens/s); median
2952.157 ms, minimum 2006.387 ms and p95
4154.369 ms. FFNs accounted for
88.79% of the measured decode time. These are contended application
latencies with load recorded before/after, not an isolated hardware benchmark.
They establish that this arrangement is too slow, not that optimization is done.

The research follow-up is `gemma4-static-ffn-options-addendum-20260914.md`.
The next candidate retains static FFNs and QKV, with two shape-shared dynamic
O programs: 116 compiled programs and only 1.762 GB of dynamic O weights, versus
16.987 GB of dynamic FFN weights. New dynamic-linear packing/code and explicit
GELU for the static FFN are being prepared; neither new path is yet qualified.
The working dynamic-FFN decoder remains the default. No goal completion is
claimed: throughput, longer global contexts, serving integration and sustained
qualification are still open.
# Static FFN candidate: component qualification

After the seven-prompt 115-program decoder qualification, a separate 116-program candidate was added behind `--ane-weights static-ffn`. The default remains `dynamic-ffn`. It uses 48 static explicit-tanh GELU FFNs and two shared dynamic output-projection programs; QKV, attention, norms, RoPE, residual boundaries, cache import and vocabulary projection remain unchanged. The verified baseline executable is preserved at `target/verified-dynamic-ffn/rvllm_disaggregated_infer` with SHA-256 `4a14f14c2276781c4f075960c696c17acc588b34550d95f39a990dcec91906e1`.

The new host packing check passed. A small shared dynamic linear graph passed four evaluations across two independent resident weight sets, after dropping its original program handle. The static FFN with the accepted explicit GELU expression passed three CPU-reference comparisons. Journals show two successful compiles/loads, seven successful evaluations and two returned unloads; boot identity was unchanged. See `gemma4-12b-evidence-20260914/ane-static-ffn-dynamic-o-components/`. These are component results, not qualification of the full 116-program decoder.

The candidate release binaries built successfully in 5m04s under substantial unrelated compiler load. Real 12B layer FFN and both output-projection shapes are being compared before any full static-weight load. The research recommendation is documented in `gemma4-static-ffn-options-addendum-20260914.md`.

Disk headroom is a constraint. Only rvllm's two obsolete iOS compiler cache directories were removed after checking that every file was older than 28 days and that neither open files nor processes referenced them. This removed 6,096 reproducible files with 5,795,622,912 allocated bytes; measured free space increased from 25,946,501,120 to 31,156,744,192 bytes while other work continued. Native binaries, model assets, source and evidence were preserved. Receipt: `gemma4-12b-evidence-20260914/obsolete-ios-build-cache-cleanup.json`.

## Real 12B component comparison and source-weight storage

All real-weight probes passed three independent CPU-reference inputs, five warmups and twelve measured evaluations per process. With journaling disabled, the same layer-0 FFN measured 54.504375 ms median with dynamic weights and 3.333 ms with compiled constant weights (16.35x observed ratio). Both had exactly the same aggregate maximum absolute error (0.0038565719) and relative L2 error (0.0027558859). These are sequential application timings under heavy unrelated load, not an isolated device benchmark or full-model speedup.

| Output projection | Constant weights median | Dynamic weights median | Relative L2 error, both |
|---|---:|---:|---:|
| Sliding, K4096 / N3840 | 0.435916 ms | 1.260959 ms | 0.0006798487 |
| Global, K8192 / N3840 | 0.738542 ms | 2.269667 ms | 0.0004169605 |

Evidence and exact binary/source hashes: `gemma4-12b-evidence-20260914/ane-real-static-ffn-dynamic-o/`. All processes returned success without a boot change. The separately journaled storage probe is excluded from the timing comparison.

The storage probe observed both the original `weights/weight.bin` and `data` at 353,894,656 logical bytes each. Later byte/header checks established that `data` is a source staging copy, not evidence of lowered ANE weights. A cleanup step removes only the exclusively owned source-weight file after successful load, retaining all framework-created files through final unload. Small static FFN and linear projection tests passed after this cleanup, with normal unloads; see `ane-source-weight-cleanup-components/`. Real-size cleanup and the complete 116-program decoder still require qualification at this point in the record.

To create headroom, an additional 11,526 regular `.rlib`, `.rmeta` and `.o` files older than seven days were removed from the inactive full-horizon debug compiler cache after checking active files/processes and rechecking each file identity immediately before deletion. Executables, bundles, source, evidence and recent compiler files were preserved. The 11,538,575,360 allocated bytes removed increased observed free disk from 26,831,814,656 to 38,340,214,784 bytes while unrelated work continued. Receipt: `gemma4-12b-evidence-20260914/obsolete-native-compiler-files-cleanup.json`.

## Complete static-FFN captured continuation: passed

Release SHA-256 `9ca813c9d58e65a4f7f9c64a8377fcfc77d80151c7a7ce3b667e8f4943d39877` completed a real Metal capital-chat prefill and full 48-layer ANE continuation with the 116-program static-FFN/dynamic-O arrangement. All 96 exported input-KV files and all 48 post-scalar layer-state files are byte-identical to the prior independently CPU-checked 115-program baseline. The final top-five IDs and logits are also identical, including EOS 106. The existing baseline CPU-error measurements therefore apply by identity; no new CPU model was run or claimed for this comparison.

Journals record 116 successful compiles/loads, 208 requests, 208 completed evaluations, 112 source-weight cleanups and 116 returned unloads. The process exited zero and boot identity was unchanged. Evidence: `gemma4-12b-evidence-20260914/disaggregated-static-ffn-capital-20260914/`, especially `baseline-state-comparison.json`, `summary.json` and the archived comparison source.

Observed preparation was 11.976 s Metal and 328.617 s ANE; prefill/sample/capture was 3.141 s. The 4.276 s decode included per-phase durable journal writes and 48 layer captures and is **not a performance result**. The unjournaled seven-prompt qualification is next.

Full preparation consumed substantially more disk than the owned per-program files alone. At 95 loaded programs, owned staging data summed to 18,846,860,288 logical bytes; the snapshot is `storage-during-preparation.json`. This is consistent with additional daemon storage, but that storage was not individually inventoried. Additional inactive compiler caches were conservatively cleaned to maintain headroom: 40,780 regular `.rlib`, `.rmeta`, `.o` files older than seven days, 13,332,164,608 allocated bytes, preserving executables/bundles/source/assets/evidence. Receipt: `additional-obsolete-compiler-files-cleanup.json`. The completed run returned to 31,936,593,920 bytes free after owned staging files were removed.

The repeat seven-prompt runner uses the same verified binary and graph identities. It budgets at most 22.1 GB of owned staging constants plus 8 GiB headroom; the preceding run already incurred persistent storage for those same identities. The earlier 32 GiB cold-start assumption underestimated total initial storage; future cold static provisioning must account for owned staging, separately measured daemon storage, transient allocation and unrelated-work headroom.

## Seven-prompt static-FFN qualification: passed, 6.72x observed improvement

The same `9ca813c9d58e65a4f7f9c64a8377fcfc77d80151c7a7ce3b667e8f4943d39877` release binary completed seven fresh Metal prefills and 24 actual ANE decode steps, matching all 31 reference IDs with zero Metal decode steps. It reused one 116-program initialization. Journaling and layer captures were disabled; the process returned success and boot identity was unchanged. The qualified executable is preserved at `target/verified-static-ffn/rvllm_disaggregated_infer` (use `--ane-weights static-ffn`). The earlier dynamic baseline remains separately preserved.

Measured ANE decode was 11.271 s total, **2.129 tokens/s**, versus 0.317 tokens/s for the prior dynamic-FFN run: **6.72x observed ratio**. Median was 380.966 ms, minimum 300.307 ms and p95 567.620 ms. All 24 steps, including the initial 2.147 s cold decode, are counted. These are sequential application runs under varying unrelated load, not an isolated causal hardware-speedup measurement.

FFNs account for 43.26% of measured decode time, output projections 29.27%, QKV 12.61%, attention 6.64%, vocabulary 6.09% and host work 2.13%. The first cold step's output projections alone took 1.227 s; steady-state rows must not be substituted for the complete-run throughput. The microprobe's 16.35x FFN-only ratio is likewise not the full-model ratio.

Preparation remains expensive: Metal 24.826 s, ANE 334.073 s. Seven prefills/sample/captures total 42.464 s. Initialization and first-token work are reported separately from decode throughput. Evidence: `gemma4-12b-evidence-20260914/disaggregated-static-ffn-seven-prompts-20260914/summary.json`, per-case results, run/source identities and the archived summary source.

Next pressure stays on real 12B inference: weight-only INT8 static FFN as an isolated encoding/quality/representation experiment, then reducing output-projection cost and compile/load overhead. The new source report is `gemma4-ffn-fusion-compression-research-20260914.md`. Compression, longer global contexts, serving integration and sustained qualification are still unimplemented or unfinished; this is not goal completion.

## INT8 FFN: real layer passes, full model remains unqualified

The separate `rvllm_ane_int8_probe` compares original FP16, per-output-row INT8
with stored FP16 scales, and the exact FP16 reconstruction of those integers
and scales. All three real layer-0 cases passed the existing backend tolerance
against their selected independent CPU reference for three inputs. Six serial
processes each performed three comparisons, five warmups and 64 timed calls.
All exited successfully with unchanged boot identity. Host packing tests and
small compressed/dense control graphs also passed.

Unjournaled conventional median latency was 4.244 ms original, 4.321 ms dense
reconstruction and 3.668 ms INT8: 1.157x original/INT8 and 1.178x dense/INT8
ratios under varying unrelated load. The compressed backend's relative L2
error against the reconstructed CPU computation was 0.0766%; against the
original CPU computation it was 1.183%, maximum absolute error 0.02561. The
separate CPU reconstruction comparison attributes about 1.184% relative L2
error to quantization itself. This is one FFN's observed error, not an accepted
full-model quality bound. The full decoder and its defaults are unchanged.

The INT8 source blob is 177,016,768 bytes versus 353,894,656 bytes FP16. In all
three modes, the owned `net.plist` exactly matches `model.mil`, and `data`
retains the source blob's header and length. **These are staging copies, not
proof of the lowered ANE weight format or resident compression.** Earlier
references to those files as compiled artifacts have been corrected above.
First-run persistent disk deltas were approximately 355 MB original, 356 MB
dense and 185 MB INT8, while later same-binary runs added no material disk.
Those global measurements and process RSS are contended and do not inventory
daemon/kernel residency. Evidence: `gemma4-12b-evidence-20260914/ane-real-int8-ffn/`
and `ane-int8-encoding-components/`, including immutable source/binary identities.

The research correction and cache-interface candidates are in
`ane-daemon-cache-provenance-20260914.md`. Read-only inspection of the installed
24G84 framework confirms Boolean `compiledModelExists` and void/no-argument
`purgeCompiledModel` on `_ANEInMemoryModel`. `_ANEStrings` reports
`/Library/Caches/com.apple.aned/24G84` and the corresponding `aneuserd` path;
ordinary filesystem access is denied. No permissions were changed. Cache
query/load-only behavior is being tested with a disposable owned small model
before using it for real 12B startup or assuming it avoids compiler limits.

## Model-specific cache reuse is verified; all-static decode is the next candidate

The small disposable 32-channel model passed cold compilation, unload and
load-only restoration from newly created source staging, then load-only reuse
in a fresh process. Twelve evaluations matched exact expected FP16 values.
Changing the weight content produced a cache miss before compile/load. After
all requests and the final program owner were dropped, this fixture alone was
purged through the locally verified void/no-argument method; the subsequent
existence query returned false. There is no production/global purge operation.
Evidence: `ane-cache-small-lifecycle/` under the evidence directory.

Release probe `746daca89329040ec3691bdc03e80660261b1b9e786df3593ae51d1003cabbb3`
then loaded the original, dense-reconstructed and compressed real layer-0 FFNs
from the earlier probe's cache. Each process required a cache hit, performed
zero explicit compiler calls, passed three unchanged CPU-reference comparisons
and completed 20 total evaluations with normal unload. The executable bytes
changed while its linker-generated code-signing identifier remained the same.
This demonstrates reuse across this rebuild; it does not establish every
component of the daemon's cache key. Evidence: `ane-cache-real-ffn-reuse/`.

A guarded cleanup now compares the complete owned `data` file to the caller's
source blob before removing that duplicate after load. A mismatch retains the
file and returns an error. This does not inspect or remove daemon files. Host
deletion-guard tests and the same small cold/load-only lifecycle passed, with
four verified source-copy removals and 12 correct evaluations. Evidence:
`ane-cache-small-source-release/`. Real-size and attention checks remain next
before a full-model run with this cleanup.

The unqualified `static-all-cached` decoder candidate retains all 48 static
QKV, O and FFN graphs, 16 head tiles and two shared attention graphs: 162 loaded
programs. Every graph must already be cached; there is no compile fallback.
The prior two full-model arrangements may have provisioned this union under
the same client identifier, but that is not assumed: missing entries will
stop preparation before a compiler call. Full residency, numerics, source
cleanup and performance still require qualification. Existing defaults and
the two preserved qualified executables remain unchanged.

The real original, dense-reconstructed and INT8 FFNs subsequently passed 60
evaluations with verified source-data removal enabled, normal unloads and no
boot change (`ane-cache-real-source-release/`). Both qualified attention
geometries also passed four loads and 12 CPU-reference evaluations, including
two required cache hits (`ane-cached-attention-source-release/`). Cold
unweighted compilation created no `data`; cached unweighted staging created an
empty file which was verified and removed. This distinction is recorded in
the phase counts rather than treated as a missing cleanup.

The first full all-static attempt stopped normally at layer 5's output
projection: its model identifier exactly matches the earlier 115-program
run's compiled entry, but the current cache query returned absent. It made
zero compiler calls, loaded and cleanly unloaded 18 programs, and performed
no ANE evaluations. Peak observed disk reduction was only about 352 MB;
boot identity was unchanged. This is a cache-availability failure, not full
residency qualification. Evidence: `disaggregated-all-static-capital/`.

Explicit preparation is now split into `qkv`, `output`, `ffn` and
`head-attention` parts, each run in a fresh process with at most 48 possible
compiler calls and one graph held at a time. The CLI's
`--prepare-ane-cache <part>` mode returns before Metal initialization or any
inference. It reuses existing entries and provisions missing ones; it never
claims numerical or complete residency qualification. The inference mode
continues to require every cache entry and fails without compile fallback.

## Complete cached all-static qualification passed; co-residency limits the gain

Provisioning reused 147 entries and rebuilt 15 missing output-projection entries
across four fresh processes. Every part returned success; each held one graph
at a time and performed no evaluations. Evidence: `ane-static-cache-provision/`.

Release `3708f35f373641681b4caa975f56e94d4cc99ccba9a1f1d48cf0374ee4412329`
then loaded all 162 cached graphs with **zero explicit compiler calls**. The
capital continuation completed 208 evaluations, verified/removed 162 owned
source-data copies and 160 source-weight files, and unloaded all 162 programs.
All 96 input-KV and 48 layer-state files, plus the final top-five IDs/logits,
are byte-identical to the independently CPU-checked baseline. No new CPU model
was run for that identity comparison. Evidence:
`disaggregated-all-static-provisioned-capital/`. The preserved executable is
`target/verified-all-static-fp16/rvllm_disaggregated_infer`.

The same binary passed seven fresh prefills and 24 actual ANE steps, matching
all 31 IDs with no Metal decode or fallback. Without journaling/captures,
decode totaled 11.144 s: **2.154 tokens/s**, median 420.335 ms, minimum 286.291 ms,
p95 738.456 ms. This is 1.011x the prior static-FFN/dynamic-O result—effectively
unchanged under varying load—and 6.795x the original dynamic-FFN result. Output
projection time fell to 1.434 s total from 3.299 s, but increased time in other
stages offset most of it. ANE preparation was 88.776 s versus 334.073 s for the
earlier explicit-compile run; Metal preparation was 24.088 s. All steps are
included. Evidence: `disaggregated-all-static-provisioned-seven-prompts/`.

The captured run observed a late 12 GB disk decline despite no remaining owned
source files, and the subsequent system query showed about 13 GB swap in use.
The seven-prompt repeat recorded swap used rising from 12,910.56 MiB before to
17,318.69 MiB after, with a sampled peak of 18,895.06 MiB during the run.
Client RSS peaked at 35.38 GiB in ten-second samples, then fell while swap grew.
These are contended system observations, not attribution of every allocation
to rvllm or proof of a descriptor leak. Free-memory percentage after process
exit cannot establish peak residency. Both runs exited normally with unchanged
boot identity.

The source review `gemma4-residency-ownership-research-20260914.md` identifies an
avoidable model-sized overlap: this diagnostic held its Metal arena throughout
ANE loading/decode. The next candidate captures every actual prefill into its
owned CPU KV snapshot, drops Metal, then loads ANE once and decodes those starts.
It records the released owner's arena capacity. This changes qualification
scheduling; overlapping new serving prefills still require separate budgeting.
No private descriptor is cleared or mutated. The new phased candidate has not
yet passed its full-model check at this point in the record.

## Phased Metal/ANE residency qualified

Release `36f2a5545015030470cc1d27f8ad2051304c7a1f2fa64160410ed29ae7b05581`
captures all requested Metal prefills, releases the Metal owner, and only then
loads the 162 cached ANE graphs. The released arena capacity is 23,932,195,544
bytes. No private descriptor is cleared. All 144 capital KV/layer-state files
and the final top-five IDs/logits are byte-identical to the previously
CPU-checked baseline. That identity comparison reuses the existing CPU error
measurements; it does not claim a new CPU model execution. Evidence:
`disaggregated-all-static-phased-capital/`. The executable is preserved in
`target/verified-all-static-phased/rvllm_disaggregated_infer`.

The matching uninstrumented binary passes seven cases, 31 IDs and 24 actual ANE
steps: 8.697 s decode, **2.760 tokens/s**, median 349.048 ms, minimum 273.152 ms,
p95 466.647 ms. Warm ANE preparation is 72.513 s; Metal preparation 25.426 s;
all seven prefill/sample/export operations total 21.338 s. All decode steps
are included, with zero Metal decode. Stage totals are 4.520 s FFN (51.97%),
1.373 s QKV, 0.939 s O, 0.790 s attention, 0.758 s head, 0.318 s host.

This is an observed 1.281x gain over co-resident all-static, 1.296x over
static-FFN/dynamic-O, and 8.707x over the original dynamic-FFN path. These are
sequential runs under changing unrelated load, not isolated causal speedups.
Ten-second sampled client RSS peaks at 27.96 GiB versus 35.38 GiB previously;
system swap used changes from 14,246.38 to 14,190.38 MiB. Those samples exclude
driver/daemon allocations and can miss peaks. Both phased runs exit normally
with the same boot identity. Evidence:
`disaggregated-all-static-phased-seven-prompts/`.

The current CLI still defaults to the original dynamic-FFN weight plan;
the optimized path is explicitly selected with `--ane-weights static-all-cached`
after cache provisioning. Phased scheduling is a diagnostic lifecycle change;
it is not yet an overlapping serving implementation. Global context remains
limited to the qualified 64-token graph. Those product and capacity limits
remain open alongside throughput optimization.

## LUT4 component routing and interleaved comparison passed

`AneLut4FfnWeights` fits three independent scalar codebooks using weighted
histograms of the exact finite FP16 coefficients, deterministic quantile
initialization and at most 50 Lloyd iterations. It reassigns every index against
the stored FP16 centroids, packs the earlier index in the low nibble, and uses
64-byte-aligned BLOB descriptors/payloads. It emits the iOS18 rank-four uint4
`constexpr_lut_to_dense` constant directly into the unchanged three-convolution
FFN and tanh-GELU expression. There is no automatic format fallback. The source
blob is 88,474,240 bytes including headers/padding, versus 353,894,656 bytes for
FP16. This is a source-storage comparison, not a resident allocation claim.
The primary-source compatibility audit is
`gemma4-lut4-mil-compatibility-audit-20260914.md`.

Two host tests checked the palette edge cases, nibble reconstruction and blob
alignment. A separate small 64/128-channel hardware fixture compiled two graphs
and completed six correct evaluations against the exact dense reconstruction,
with normal source cleanup/unload and unchanged boot identity. Evidence:
`ane-lut4-small/`.

A fresh pinned FP16 CPU model consumed the actual phased Metal capital KV and
captured all 48 pre-FFN inputs, while independently repeating the all-layer
comparison. It reproduced the previous maximum 0.442% layer relative L2 error
and final token/top-five reference. These are **CPU** FFN inputs from the actual
Metal seed, not ANE-captured FFN inputs. Layer 0's largest input magnitude is
171.5. Evidence: `cpu-ffn-inputs-capital/`.

The first real-weight probe release,
`d615e3e7ab7f505fab1e5309aa4d79db16d7402ff9141ef80ad0f1bf78689a49`,
passed all six processes: original FP16, dense LUT4 reconstruction, explicit
LUT4, then the reverse order through required cache reuse. Each checked three
synthetic inputs plus that real CPU input, five warmups and 32 timing samples
(246 evaluations in all). Three initial cache misses compiled normally; the
second pass required existing entries. System load rose above 150 during
unrelated work, and single-process medians ranged from 3.3 to 41 ms; those
sequential timings do not establish a useful comparative speedup. Evidence:
`ane-real-lut4-ffn/`.

The interleaved comparison release,
`3eb380c90736cebc1adb0df7892e4aa417caa7401fefbdeb961945005ba81e18`,
holds all three programs and requests resident, checks each against its CPU
reference, then rotates execution order across 64 cycles. It requires all
three cache entries and makes no explicit compiler call. Two fresh-process
repeats (219 evaluations each, no journal/captures) report:

| Repeat | Original FP16 | Dense LUT4 reconstruction | LUT4 | Dense/LUT4 ratio |
|---|---:|---:|---:|---:|
| 0 | 3.289 ms | 3.299 ms | 2.647 ms | 1.246x |
| 1 | 3.373 ms | 3.284 ms | 2.647 ms | 1.240x |

The paired per-cycle ratio medians are 1.246x and 1.243x. Compressed and dense
control outputs are bit-identical on all four inputs in both repeats. On the
real input, backend relative L2 error is 0.0349% against reconstruction;
quantization contributes approximately 18.595% relative L2 against the original
CPU FFN, with maximum output difference 11.192. This quality cost has not been
accepted for the model. Both runs exit normally with unchanged boot identity.
Evidence: `ane-lut4-resident-comparison/`.

The measured runtime advantage is compatible with reduced weight traffic but
does not locate decompression. Apple's [overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
explicitly describes backend-dependent load-time versus runtime decompression;
its [performance guide](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
ties latency gains to just-in-time decompression. Those general statements do
not establish this private graph's physical storage.

An explicit `static-lut4-ffn-cached` model candidate now changes only the 48 FFN
weight representations. It requires separately prepared `ffn-lut4` cache entries;
all attention, QKV, O, head, host math and current Metal prefill remain the
qualified implementations. At this point its full-model check is still pending.
The original reference-token gate is retained, and defaults have not changed.


## Cache persistence failed before LUT4 inference; recovery is bounded

All 48 LUT4 FFN programs were separately compiled and unloaded normally
(`ane-lut4-cache-provision/`, 211.55 seconds, zero evaluations). The first
model attempt then missed the global attention cache entry before inference.
A separate shared-cache pass loaded 114 programs and compiled five missing
entries. A second model attempt then missed LUT4 FFN layer 1, whose exact model
ID had just been provisioned. Neither attempt evaluated an ANE graph; both
returned ordinary cache errors, cleaned up loaded programs and preserved the
boot identity. Evidence: `disaggregated-lut4-ffn-capital/`,
`ane-lut4-shared-cache-provision/`, and
`disaggregated-lut4-ffn-provisioned-capital/`.

This demonstrates that provisioning is not a durable availability guarantee;
it does not establish eviction or any particular driver/cache cause. The new
explicit `--ane-compile-budget 0..16` permits a small number of actual misses
to compile during cached model initialization. The default remains strict
cache-only. A process-wide atomic counter reserves a permit immediately before
each compiler attempt, counts failed attempts, cannot reset or wrap, and rejects
exhaustion. All multi-input/output quarantine checks still precede private API
access. The counter's concurrent/exhaustion test and all five non-hardware ANE
sys tests pass. Full-model use of this new policy remains under qualification.

## Opt-in Metal QKV component passed

The QKV projection now has a bounded candidate for Gemma 4 12B prompts of
20 through 32 tokens on Apple9/Apple10: cooperative batch-eight dot products
write FP32 scratch; a second encoder runs the existing head normalization,
BF16 boundary, RoPE and KV-cache stores. FP32 sums are retained until head RMS,
as in the original fused path. `RVLLM_METAL_QKV_PREFILL=batch8` explicitly selects
it; other shapes and unset environments retain the original implementation.
The scratch allocation and numeric/cache ABI were updated for this candidate.

The component uses actual layer 0 and 5 checkpoint projection/norm weights and
controlled BF16 inputs, including magnitudes spanning 0.1 through 100. It checks
all planar/cache outputs, guard regions and negative cache slots. It does not
use captured model activations. Alternating six measurements per path gave:

| Geometry | Original median GPU time | Candidate median GPU time | Ratio |
|---|---:|---:|---:|
| Sliding, 21 tokens | 5.655 ms | 2.034 ms | 2.780x |
| Global, 28 tokens | 9.910 ms | 3.423 ms | 2.895x |

Maximum output relative L2 difference is 0.00442%; the largest absolute
difference is 0.0078125 in Q. Both complete projection/normalization/RoPE/cache
paths are timed. This is a component result, not a model latency or generation
quality claim. The public Metal test exits normally with unchanged boot identity.
Evidence: `metal-qkv-prefill-component/`. Full-model qualification is next.


## LUT4 full-model candidate rejected

The bounded initializer completed all 162 program loads, recovering nine actual
cache misses through nine compiler calls. It performed 208 evaluations for a
captured capital continuation and cleaned up all programs normally. The 96
exported Metal KV files are byte-identical to the phased FP16 baseline, so the
new unused Metal kernels and scratch layout did not change this seed. The
answer remained `Paris` followed by EOS, but relative L2 differences against
the qualified FP16 ANE states reached 97.0% at layer 4 and 44.6% at the final
layer. These compare two ANE runs from identical seeds, not an independent CPU
oracle. Evidence: `disaggregated-lut4-budget-capital/`.

The seven-reference run successfully prefilled all seven prompts, then failed
the first, 16-token raw continuation. It first diverged at generated index 6
(position 11), emitting 100 instead of 101, and subsequently repeated control
tokens. The existing gate reports failure after the case completes; later
references were not decoded. The observed approximately 197–200 ms warm token
times do not qualify this candidate as an optimization: it changes the model's
behavior unacceptably. The scalar per-tensor LUT4 scheme is rejected for the
working path. It remains explicit experimental code and has not become a
default. Evidence: `disaggregated-lut4-budget-seven-prompts/`.

The release used for these checks is
`18daf3cf4c8e6a48c57bcf2f890d322e3766100c63a50d2e07d780d5abc88537`;
its separate BF16 metallib is
`f3bc2a3ea26ed5f7e599e8759b21373ae075eeb77ce8b8df8d7e9526c4a963cf`.
Both runs exited through normal user-space control flow; the boot second is
unchanged and no new panic was observed.

The QKV review's diagnostic-rounding exclusion and extra-encoder accounting
have been fixed. The next full-model check explicitly combines QKV batch8 with
the original FP16 ANE weights. Metal intermediate tracing remains disabled;
ordinary synchronized KV export and ANE layer capture leave the candidate
prefill path active.


## QKV batch8 full-model and CPU gates passed

With the explicit QKV opt-in and original FP16 ANE weights, the capital capture
passed with 14 recovered compiler misses, 162 loaded programs and 208 ANE
evaluations, followed by normal unloads. The expected first token and EOS are
preserved; all 144 KV/layer state files differ from the original reduction,
confirming this was not a fallback-path capture. A fresh pinned FP16 CPU model
then consumed these actual Metal KV values and reproduced the EOS and top-five
IDs. Maximum layer relative L2 error is 0.4649% at layer 46; final-layer error
is 0.2497%. Evidence: `disaggregated-qkv-batch8-fp16-capital/`.

The same binary then passed all seven references, 31 IDs and 24 ANE steps,
with zero compiler-budget consumption and zero Metal decode. ANE execution
totaled 6,312.651 ms (3.802 tokens/s), median 252.635 ms. Initialization took
38,014.309 ms for ANE and 8,539.358 ms for Metal. Seven prefill/sample/export
measurements ranged from 1,051.803 to 1,874.452 ms, including the unchanged
six-token fallback case. These are whole-process observations under lighter
system load, not an isolated causal decode speedup from the QKV prefill edit.
Evidence: `disaggregated-qkv-batch8-fp16-seven-prompts/`.

## Native BF16 matrix component passed

A new isolated 32x32x32 tile uses four SIMD groups, native BF16 matrix operands
and four FP32 accumulator fragments per group. Cooperative row-major weight
tiles are read transposed by `simdgroup_load`; model weights need no transpose.
Every group reaches both tile barriers, and M/N/K tails are zero-loaded and
masked at stores. The test includes actual layer-0 QKV, gate/up, O and down
weights, controlled BF16 inputs outside FP16's finite range, guard regions and
sampled independent FP64 dot products. The all-tail 21x37x35 fixture is exact.
On the real matrices, maximum relative L2 difference from batch8 FP32 outputs
is 0.000138%. Measured GPU ratios are 2.521x for QKV, 3.158x gate/up, 1.952x O
and 2.412x down. These initially compare FP32 output variants; the production
BF16 output boundary is a separate check. Evidence:
`metal-native-bf16-mma-component/`; source review and primary-source mapping:
`gemma4-metal-bf16-matrix-tile-design-20260914.md`.


## Native matrix integration passed the existing full-model gate

Separate production entry points retain FP32 QKV projections and round ordinary
projections to BF16 once before their existing epilogues. Both entry points were
checked against the earlier isolated prototype, including BF16 values outside
FP16 range, tails and guard regions. Production FP32 outputs are bit-identical
to the prototype; BF16 outputs exactly equal a single rounding of those FP32
values. The typed component repeats time the correct output dtype for each
projection. Evidence: `metal-native-bf16-mma-typed-component/`.

`RVLLM_METAL_PREFILL_GEMM=mma32` now selects only BF16 pipelines with known
Apple9/10 identity and the intended 20–32-token projection shapes; piecemeal
pipeline changes clear the dtype classification. Quantized-accumulator debug
mode remains excluded. The setting also selects the FP32 QKV/normalization
split, with correct extra-encoder accounting. Numeric ABI is eight. Focused
routing/storage tests and fingerprint separation pass.

The integration release
`af1902eb97c01fe13e3535aa6eb31d0467ec5f6840ec0a949cb7d5d97eaad671`
passes the capital capture with zero compiler calls and all 208 ANE evaluations.
A fresh pinned FP16 CPU model consumes its actual Metal seed and agrees on EOS
and top-five token IDs; maximum layer relative L2 error is 0.4500%, final-layer
error 0.3083%. The same executable then passes all seven references, 31 IDs and
24 ANE steps with zero compiler calls and no Metal decode. Evidence:
`disaggregated-mma32-fp16-capital/` and
`disaggregated-mma32-fp16-seven-prompts/`.

Warm chat prefill/sample/export observations are 383.722–463.822 ms, while the
six-token raw prompt remains on the older projection path at 1,245.297 ms.
ANE decode is unchanged architecturally and measures 3.830 tokens/s, median
249.347 ms, with FFN accounting for 3,759.291 of 6,265.983 ms across 24 steps.
ANE preparation is 39.589 s and Metal preparation 8.812 s. These complete-run
observations complement the alternating component measurements; they are not
yet a paired whole-model prefill benchmark. Longer context and ordinary user
prompt/serving integration remain unfinished.

The next full-model quantization candidate uses the already component-tested
per-output-channel INT8 FFN representation, with all other weights unchanged.
All 48 entries compiled and unloaded normally in 120.98 seconds, with zero
evaluations. Evidence: `ane-int8-cache-provision/`. It has not yet passed a
full-model numerical or performance gate.


## Global-1024 component qualification passed

The exact single-input/output Q16/KV1/D512/global-1024 graph compiled once and
passed sixteen evaluations over two independent requests at imported lengths
5, 63, 511 and 1022, with two appends per request and alternating request order.
Maximum absolute error against independent FP32 attention is 0.002010, inside
the unchanged 0.003 bound. Both requests reject an append beyond 1024 before
execution and preserve their position. One program unloaded normally, and the
boot identity stayed unchanged. Evidence: `ane-global1024-qualification/`.

This supports opening only that exact additional shape. The source now accepts
an explicit `--context-capacity 1024`; the default remains 64. Full-model and
longer-prompt qualification of the new capacity are still pending. All
multi-input/output private calls remain quarantined.

## INT8 initialization hit the compile budget before inference

The first model load found 17 missing entries, compiled 16,
then stopped at the budget before any ANE evaluation. All 97 loaded programs
unloaded normally. Matching model IDs against the INT8 provisioning journal
using `descriptor_created` events identifies ten INT8 FFNs, four output
projections and three QKV projections. An initial analysis incorrectly indexed
`compile_requested`, whose model ID is empty; that classification has been
corrected in the evidence and user update. This remains a cache-availability
failure before numerical testing. A shared-entry refresh loaded 113 hits and
compiled one missing head/attention entry. One bounded model retry follows;
there is no automatic unbounded compile/retry loop.
Evidence: `disaggregated-mma32-int8-capital/`.

A cache cleanup reclaimed approximately 8.53 GB of measured free space by
removing obsolete Rust debug incremental directories only. Cargo's debug
build lock was held throughout, and every removed directory had no descendant
modified for at least thirty minutes. Source, model assets, executable files,
recent incremental state and qualification evidence were retained. Receipt:
`obsolete-incremental-cleanup-1649/`.

### INT8 FFNs passed the complete seven-prompt qualification (17:00 EDT)

The bounded retry after cache refresh succeeded with one compile, 161 cache hits, 162 program loads, 208 evaluations, and clean unload. All 96 Metal KV files were byte-identical to the MMA32 FP16 capture. Against the FP16 ANE state trajectory, maximum layer relative L2 was 0.0309681 at layer 9 and final relative L2 was 0.0272700; Paris/EOS matched. INT8 changes model weights and is not numerically identical to FP16.

The following seven-prompt run passed all 31 expected generated IDs and 24 real ANE steps with zero compiler calls and zero Metal decode steps. Decode total was 4,612.378 ms: **5.20339 tokens/sec**, median 187.867 ms. FFN total fell to 2,172.249 ms from the previous FP16 run's 3,759.291 ms; the other phase totals were similar (QKV 829.390, attention 427.502, output 557.451, vocabulary 499.299, host 126.486 ms). These are sequential whole-model runs on a shared host, not an alternating paired benchmark. Metal initialization was 8,495.692 ms; ANE initialization was 45,478.124 ms. Six chat prefills were 371.665–456.237 ms; the unoptimized six-token raw prefill was 1,257.726 ms. Initialization remains a major limitation.

Evidence: `gemma4-12b-evidence-20260914/disaggregated-mma32-int8-refreshed-capital/` and `disaggregated-mma32-int8-refreshed-seven-prompts/`, including numerical and performance summaries. Binary SHA256 `216874661769a745368a1cf2974289cbafbaacee5c723c197187322f7d3f4a5e`; preserved at `target/verified-int8-model-candidate/rvllm_disaggregated_infer`. Global context remains 64 in these runs. Boot second remained 1789388363. This qualifies these prompts; broader quantized quality and longer context remain to be measured.

### Ordinary text input passed beyond the old 64-token limit

`text-context1024-three/` ran the actual Rust CLI with three `--prompt-file` inputs and no reference tokens supplied to inference. The default INT8 FFN / global-1024 policy produced `Paris`, `The blue fox jumps over the quiet river.`, and `violet`; all 14 output IDs and 11 ANE steps matched independent CPU references. Prompt token counts were 21, 84, and 230, with exact template/tokenizer parity. All host KV snapshots are released after import. Four bounded compiler attempts repaired missing cache entries. CLI binary `c1e1f9daebbc2fa1d89f8106618d826f740b7a7c7edbac3101d3325d5ab4dbdb` is preserved at `target/verified-text-cli-candidate/`. Its Metal numeric ABI remains 8 and MMA still only covers M20–32.

The CPU-only tokenizer qualification matched six pinned chat fixtures, Python/Jinja whitespace trimming, and incremental Unicode/long-text decoding (1.68 seconds; no model/accelerator initialization). Text mode uses the checkpoint template SHA256 and explicitly supports one non-thinking user turn per prompt, with optional reports. Multiple prompt files share initialization. This is ordinary one-shot/batched text generation, not efficient interactive serving.

The longer captured prefills exposed the next bottleneck: 84 tokens took 14,831.2 ms and 230 took 32,037.0 ms. The run overlapped part of independent CPU reference generation and a host-only build; these are diagnostic observations, not a paired speed comparison. Source/evidence: `gemma4-12b-evidence-20260914/text-context1024-three/`. Fresh BF16 CPU references for 84, 230 and 652 tokens are in `long-context-cpu-oracle/`; the last is an exact `42` recall.

### Wider native BF16 matrix component passed

The bounded seven-shape experiment in `metal-mma-extended-component/` passed 175 GPU command buffers, including multi-tile M tails, N/K tails, guarded FP32 and BF16 outputs, identical production/prototype outputs, and sampled independent FP64 dots. Maximum relative L2 against the FP32 batch-eight control was 1.376e-6. Actual layer-0 larger projection cases measured 5.65–6.79x faster than direct batch-eight controls; the M6 QKV case measured 1.14x. Tiny synthetic tail matrices were slower, and are outside production shape routing. These controls are direct component kernels, not the old long-prefill dispatch policy. CPU reference generation and text inference finished before the component, and no own builds overlapped its execution.

The range-extension source resolves actual Prefill phase, native BF16, Apple9/10, dense 12B dimensions and M6–1024 before selecting MMA. Batched decode and auxiliary GEMMs keep their prior routing. O/down now deliberately materialize BF16 before RMSNorm, and encoder accounting uses the same split predicate. Numeric ABI9 distinguishes this Rust-only routing change. Scoped host shape/count checks and source review passed; full-model range-extension validation is next.

### Wider MMA32 passed actual text generation through 652 prompt tokens

`text-mma-extended-context1024-four/` passed all four ordinary prompt-file inputs, all 17 independently referenced output IDs and 13 real ANE steps. No expected tokens were passed to the Rust process. The 652-token input recovered `42`; the 84-token exact-copy and 230-token recall outputs also matched. All four prompt encodings matched the upstream template/tokenizer. There were zero compiler attempts and no Metal decode. Binary SHA256 `2ccd3cc8c300f3a1d843953217b00998556b853d7cb5fb91d85565dbcd3df251`, preserved at `target/verified-mma-extended-candidate/`, uses Metal numeric ABI9.

Captured prefills were 1,722.153, 2,168.626, 5,134.298 and 14,270.858 ms for 21, 84, 230 and 652 tokens. The earlier 84/230-token observations were 14,831/32,037 ms, but overlapped part of a CPU oracle/build; do not label this a paired end-to-end ratio. This run's Metal initialization was 8,818.928 ms, ANE initialization 57,978.037 ms. Startup remains separate from warm throughput.

## Retained seven-prompt qualification

`disaggregated-mma-extended-int8-retained-seven-prompts` passed 7/7 prompts, 31/31 generated IDs and 24 actual ANE steps, with zero explicit compilations. Binary SHA-256 `33c3ec3ebd037bc3cefaf53f860d338ee3dcc11966be69f129fbf1b24f3cec08`. Measured decode throughput 4.909422 tokens/sec; median step 192.0113 ms; ANE preparation 47.5839 seconds. Prefill/capture times were 677.465, 500.585, 488.937, 409.955, 414.440, 479.280 and 399.337 ms. All prefills preceded ANE loading in this run; retaining Metal does not itself prove a later prefill while ANE remains loaded. The corresponding capture run had 144 byte-identical KV/layer files versus phased execution and approximately 36.7 GiB observed client RSS, which excludes some driver/daemon residency. No own CPU oracle or build overlapped this throughput run.

## Late-arriving resident requests

`interactive-late-arrival-four` used binary SHA-256 `63ab8bf7cfc9edbe53175439cd97bed6c7558cc0ea7067896836501e418d1ffc`. Both backends loaded before stdin. The runner supplied prompts of 21, 84, 652 and 21 tokens, sending each only after the previous result was written. All 17 output IDs and 13 ANE steps matched the independent CPU references; stdout was Paris, the requested copied sentence, 42, Paris. The two short cases had 144 byte-identical KV/layer capture files. ANE journal: 162 programs loaded, 159 cache hits, 3 compiler repairs within budget 16, 208 requests, 2704 completed evaluations, 162 clean unload returns. Boot time unchanged. No build or CPU reference model ran in this task during this test.

This is a lifecycle capture with a durable per-evaluation `sync_all` journal; its 0.28–0.39 tokens/sec observations must not be compared with ordinary decode throughput. Cold preparation was 12.9395 seconds Metal and 96.5164 seconds ANE in this instrumented run. The new timing scopes put most prefill wall time in command-buffer completion (3813, 5620, 11426 and 2721 ms), with host KV capture 30, 42, 271 and 11 ms. These waits include GPU scheduling and any page movement; they are not GPU-only execution timings. The initial field `metal_cpu_encode_ms` means non-wait host wall time and has since been renamed `metal_host_non_wait_ms`. A subsequent source fix bounds interactive in-memory receipt history; the current executable will be requalified after building that change.

## Unjournaled persistent CLI qualification

`interactive-warm-four` used binary SHA-256 `82a06c7624bff2ee8b1bc7013ac70e30216926d61b39e67d441e3046d7117620`. All four late-arriving stdin requests again matched all 17 expected IDs, with 13 ANE steps, no layer capture, no diagnostic journal and zero compiler attempts. The report retained only the final case while all four separate case receipts remained available. Both backends initialized once; the final 21-token request completed in 0.764962 seconds from runner input to completion, including 555.841 ms prefill, 15.982 ms KV import and 189.5 ms ANE decode. The other observed prefills were 2850.564, 2449.660 and 24948.632 ms for 21, 84 and 652 tokens. These substantial variations reinforce that timings on this shared host are not guaranteed performance. Boot unchanged; no build/CPU oracle from this task overlapped execution.

The remaining long-prefill work is now targeting excessive one-thread pointwise dispatches and scalar per-head attention, guided by source review. The persistent CLI is a verified serial owner on its main thread; it is not yet the reusable `InferenceWorker`/HTTP route, and per-request cancellation/health recovery remains unfinished.

## Pointwise dispatch component

The production GELU, residual-add, residual-add-then-scale and standalone scale encoders now group independent coordinates into 256-thread groups. The compiled shaders, buffers, indexing, and rounding are unchanged; reduction kernels retain their existing geometry. `metal-pointwise-group-component` compared the original one-thread launch directly with the production encoders in 24 F16/BF16 cases, including non-multiple tails, scalar/vector scales and 21/84/652/1024-token GELU shapes. All complete guarded outputs were bit-identical; 336 GPU commands completed. Alternating six-sample GPU measurements showed BF16 GELU median ratios 1.75x at 21 tokens, 13.48x at 84, 18.55x at 652 and 3.66x at 1024. Residual/scale timings were variable, with some observed regressions; there is no blanket measured-speedup claim for those launches. Full-model capture is next.

## Full-model pointwise dispatch acceptance

`interactive-pointwise-capture-four`, binary SHA-256 `f0ed50991cdaa4cbff3812a268806c7a31cc484ce53da00ab8facd10d8b31e34`, passed all four late-arriving requests and all 17 IDs with zero compiler attempts. All 576 captured KV/layer files across four cases were byte-identical to `interactive-late-arrival-four`. The shaders and numerical ABI were unchanged; only independent pointwise grouping changed. Prefill/capture measured 2907.787, 987.319, 16813.543 and 1633.984 ms for 21, 84, 652 and 21 tokens. Source receipt and state comparison are preserved. This establishes numerical acceptance, not an isolated whole-model timing improvement: the host remains shared and the runs are sequential. The scalar per-head attention path remains unchanged while a standalone SIMD-group candidate is being evaluated.

## SIMD-group prefill attention component

`metal-prefill-simd-attention-component` compared the scalar kernel with a one-SIMD-group-per-query/head BF16 candidate at D256/D512. Eight cases covered short/652-token prompts, permuted physical pages, multiple sequences and chunked absolute positions, a restricted sliding window and poisoned causal-future values. All guards were intact and outputs finite. Maximum full-output relative L2 against the scalar kernel was 0.000049767; maximum sampled relative L2 against independent FP64 QK/softmax/PV was 0.001582523. At M652, scalar/candidate GPU medians were 129.061/25.580 ms sliding and 268.874/70.379 ms global. These are controlled component timings, not full-model throughput.

The prototype was copied into `attention_prefill_simdgroup_f16` by function/dtype/helper substitutions only. A second eight-case component (`metal-prefill-simd-attention-production-component`, 120 GPU commands) independently re-poisoned production output with BF16 NaNs before asserting exact equality with the prototype. It passed. The opt-in `RVLLM_METAL_PREFILL_ATTENTION=simdgroup` requires BF16, Apple9/10, 6–1024 prompt tokens, the dense 12B geometry, the qualified D256/window1024 or D512/global geometry, and no Metal trace. Decode is unchanged. Numeric ABI 10 fingerprints the route. Full-model qualification is in progress, so this remains opt-in.

Candidate executable SHA-256 `25f2412d09668fcce94dc8a8dcc4b13e46b39b3b87584c18c353b03d82a3acc5`; new BF16 metallib SHA-256 `6efa8fbed39f6d36fb83fe823da1a5d0733c22970b96b431439be27a07617705`. The prior verified libraries and executables are retained.

## Full-model opt-in SIMD attention qualification

`interactive-simd-attention-capture-four` passed all four late-arriving requests (21/84/652/21 prompt tokens), all 17 generated IDs and 13 ANE steps, with zero compiler attempts. The new attention arithmetic changed 141 of 144 captured files in each case versus the preceding pointwise-only run; the requested route therefore was not a silent scalar fallback. Token agreement remains exact on these cases. The 652-token same-seed fresh FP16 CPU continuation also matched next token 236778; final-layer relative L2 was 0.01676394. The maximum across all 48 layers is recorded in the CPU receipt.

Full-prefill observations were 2227.749, 2003.069, 18986.386 and 1010.772 ms. No end-to-end speedup is established by this run despite the controlled component improvement. A scalar/SIMD/SIMD/scalar GPU-only comparison, retaining all timing observations and generating only the first token so no ANE programs are loaded, is in progress to separate prefill performance from co-residency and host scheduling. Default attention remains scalar pending the broader seven-prompt and performance checks.

## Seven-reference SIMD attention gate and default selection

`disaggregated-simd-attention-interleaved-seven-prompts` passed 7/7 prompts, all 31 generated IDs and 24 actual ANE steps with the same candidate executable/library and zero compiler attempts. Both owners remained loaded; each prefill preceded that request's decode. ANE decode time was 4488.609 ms, **5.346868 tokens/sec**, median step 181.221 ms. Component totals were QKV 872.185, attention 413.115, output 590.415, FFN 1997.655, vocabulary 516.350 and other host work 98.891 ms. Prefills measured 2195.480, 848.939, 585.469, 459.715, 476.950, 591.623 and 454.455 ms. Preparation was Metal 5.190 seconds and ANE 38.557 seconds; total process wall time includes reference seed serialization and scheduling, and must not be inferred from decode timing alone.

The matched GPU-only comparison (`metal-only-prefill-attention-policy-comparison`) used four process runs in scalar/SIMD/SIMD/scalar order, each with three identical 652-token prompts and output limit one. Every first token matched, and zero ANE programs loaded. Median warm prefill across all retained warm observations was 22.716678 seconds scalar versus 17.386669 seconds SIMD (23.47% lower). Ranges overlap substantially; this is evidence of an observed aggregate improvement, not a guaranteed causal ratio on the shared host. Together with component and numerical gates, this supports enabling SIMD attention as the CLI default. That default is now in source and awaits the next build/default-path smoke check; explicit `RVLLM_METAL_PREFILL_ATTENTION=off` remains honored. Further long-prefill optimization is still required.


## Larger matrix tile rejected after component measurement

`metal-mma-tile64-component` completed 252 GPU commands across six cases with
real checkpoint weights. Both a native BF16 32×64 tile and a lossless FP32
operand-staging variant matched the current 32×32 kernel bit-for-bit for FP32
outputs and correctly rounded BF16 outputs, including M/N/K tails and inputs
beyond FP16 range. All real model shapes were slower: native throughput was
0.60–0.73× the current tile, and FP32 staging was 0.41–0.66×. Shared storage grew
from 8 KiB to 14 KiB or 20 KiB respectively. These results reject promotion;
production keeps MMA32. The FP32 control changes staging and resource use as
well as operand type, so it does not isolate the cost of BF16 matrix instructions.

The consolidated host gate after pointwise/SIMD/default changes passed 244
unit tests (5 ANE sys, 76 Metal, 163 runtime), with all accelerator tests ignored.
Log: `target/disaggregated-consolidated-unit-gates-latest.log`. Subsequent immutable
configuration changes require their own gate and real-model validation.


## Immutable Metal owner accepted on the complete disaggregated path

Binary `d98f9d0ba15d4964b094c074256549c68c7184f1e3f64182d016ad2203449725`
uses explicit native BF16 limits/kernel options, one consumed allocation/load
plan, and no process-environment mutation. Shader generation, routing, encoder
accounting and numeric identity share the captured kernel policy. Explicit
quantized accumulation is rejected before hardware initialization. The Metal
and runtime host gate passed 78 + 164 tests; the targeted invalid-policy
regression also passed after review. Evidence: `immutable-metal-host-gates/`.

`immutable-defaults-metal-only-three/` ran 21/84/652-token ordinary prompts
with output limit one while injecting incompatible legacy dtype, capacity,
trace and sync environment values. All first tokens matched, all 288 KV files
were byte-identical to the earlier SIMD run, and no ANE program was created.
Prefill observations were 879/1914/14214 ms; this is a correctness gate under
shared-host load, not an isolated speed comparison.

Two full initialization attempts exhausted their 16-compile limits, first at
layer 13 and then at layer 44. The intermediate four-part refresh repaired
37 entries (7 QKV, 5 output, 18 FFN, 7 head/attention). A subsequent strict check
using the exact provisioning executable path found 135 hits followed by a
missing output layer-44 identity, matching the previously compiled identity.
All 135 loaded programs unloaded normally; there were no compiler calls or
evaluations in this diagnostic. These facts show cache availability was not
reliably established by provisioning. They do not identify eviction, executable
partitioning, or asynchronous persistence as the cause.

After reclaiming 2.03 GB of obsolete Rust debug archives, the fully journaled
`interactive-immutable-defaults-journaled-four/` completed all four late-arriving
requests, 17 output IDs and 13 ANE steps. All 576 captured KV/layer files and the
numeric ABI were identical to `interactive-simd-attention-capture-four/`.
Journal totals: 160 hits, 2 bounded compiler repairs, 162 loads, 2704 completed
evaluations and 162 normal unloads. The summary retained one case and counted
all four requests. Boot time remained unchanged for this run. It took 481
seconds with durable per-stage journaling and heavy host delays; these timings
are not ordinary throughput measurements. Cache persistence remains a separate
startup reliability issue, and the runtime worker/server is still unfinished.


## Power-aware measurements and runtime worker, 2026-09-15

The measured CLI now records process CPU cycles and retired instructions with
`proc_pid_rusage` V4; AC/battery supply, active power mode, Low Power Mode and
Foundation thermal state; and any reported CPU/scheduler limits. A one-second
observer records raw observations without putting subprocesses on the token
path. Every preparation/prefill/import/decode phase carries its own receipt.
Metal prefill additionally records completed GPU command-buffer timestamps.
CPU cycles are not used to rescale GPU or ANE time. The M4 Max advertises only
the Metal timestamp counter set through this API; ANE cycles remain unavailable.
The comparison tool rejects missing/stale observations, sampled transitions,
thermal pressure, unlike power strata and different workloads. See
`specs/apple-performance-measurement.md`. Historical timing reports without
power records are observations, not causal speedup evidence.

Evidence is in `power-measurement-20260915/`. A live rootless probe measured
207,972,820 CPU cycles for about 100 ms of host work, versus 3,411,160 over
a 1.105 s sleep (including observer overhead). This validates CPU accounting,
not model performance. All intervals were on battery with fair thermal state
and were correctly ineligible for speed comparisons.

Release executable SHA `28db861526445debc6e21d6af3b86466043f45e6ed17dc7742acd8367cea5e9e`
and unchanged qualified metallib passed the 21/84/652-token Metal check:
all first tokens/prompts and all 288 KV files match the previous qualified
captures. GPU intervals were recorded separately from host wall time; these
thermally ineligible capture runs do not establish speedups. See
`power-aware-metal-only-three/`. The boot did not change.

`gemma_disaggregated_worker` now owns both devices inside the request API's
accelerator thread. The opt-in `--runtime-worker true` CLI adapter uses this
owner and the same checkpoint text formatter. It has bounded queues, greedy
admission, EOS/length handling, fresh per-request import, cancellation at
synchronous boundaries, failure poisoning and critical-pressure release.
CPU tests cover these paths and owner-thread destruction. The consolidated
host gate passed 129 tests (one tokenizer test ignored, Metal test module
excluded), and server/FFI adapter checks passed. A subsequent test-only change
strengthens the backpressure test with deterministic decode-progress signaling;
it still needs rerunning. The 4 measurement tests and initial 8 worker tests
passed before that strengthening.

Live worker acceptance remains pending. The initial strict-cache attempt
stopped before inference. Bounded head/attention provisioning found 18 misses,
compiled/loaded/unloaded 18 graphs and evaluated none. A subsequent strict
worker attempt hit both attention graphs, then missed layer-0 QKV. No compile
or inference occurred in that strict retry.

Read-only code-sign inspection found the new dependency graph changed the
linker's signing identifier from `rvllm_disaggregated_infer-9d1f7275eb5937c9` to
`rvllm_disaggregated_infer-4ea237a076515df2`. A separately copied and ad-hoc-signed
executable using the prior identifier hit the old sliding graph but missed
the global graph. Its bounded recovery attempt reached layer 15, then hit
the existing 16-compile limit. It never completed initialization or generated
a token. This supports investigating client namespaces and cache availability;
it does not prove that identifier stability alone solves persistence. Original
binaries were preserved. See `power-aware-worker-three/` and
`power-aware-worker-recovered-three/`. No daemon cache was purged.

Hardware/build work was held when battery reached 7%, then resumed only after
AC returned. Old, inactive local release dependency archives and debug
incremental files were removed under both Cargo profile locks, preserving
executables, source, models, evidence and ANE storage. The cleanup recovered
about 3.42 GiB; the 16 bounded compiler repairs subsequently consumed substantial
disk space. Further full provisioning must account for disk headroom.

A host-only vectorized INT8 preparation candidate has been added without
changing the default scalar quantizer. It converts each FP16 row once via the
half crate's safe slice API, performs an integer absolute-maximum reduction,
and retains exact FP16 scales, division and ties-to-even. Parity tests were
written for all half bit patterns, Gemma row widths and complete source blobs;
they have not run yet. No speedup or ANE qualification is claimed for it.

### Completed host preparation and power-accounting checks

The preceding candidate status is superseded by the completed checks. The
vectorized INT8 source quantizer is now the default. All 48 real checkpoint
FFNs, totaling 8,493,465,600 coefficients, matched the scalar reference exactly,
including every stored FP16 scale. Exhaustive half-value/error parity and real
row-width tests also passed. This changes host preparation only; source bytes,
MIL, ANE weight precision and numeric ABI are unchanged. The currently frozen
measured CLI executable still contains the scalar implementation until rebuilt.

The host ABBA experiment completed 12 trials and six pairs with identical
outputs. Median retired instructions were 6,080,468,913.5 for scalar preparation
and 867,940,234.5 for the vectorized path. Median observed CPU cycles were
1,107,104,470 and 270,180,038.5. All pairs were thermally ineligible, so the
record contains no qualified wall-time speedup. These process counters include
observer work; they do not measure accelerator work or establish full-model
startup/decode improvements. Evidence: `int8-vectorized-host-abba/`, including
the full-checkpoint source-parity receipt and frozen benchmark provenance.

The strengthened deterministic backpressure/cancellation worker tests pass.
The final host gate passed 101 Apple and 129 runtime tests, with 24 ignored
hardware/tokenizer fixtures and the Metal test module plus the legacy explicit
ANE integration test excluded. That legacy integration test was initially
included and refused private-ANE execution without its opt-in; it was not
enabled to turn a host gate green. Runtime binary checks also passed. Logs are
preserved under `int8-vectorized-host-abba/host-gates/`.

A further 9.87 GB of reproducible dependency archives older than 48 hours was
removed from an inactive worktree target only while both Cargo profile locks
were exclusively held. The receipt is in
`inactive-worktree-archive-cleanup-20260915/`. Source, model assets, executables,
bundles and ANE daemon storage were preserved. Other tasks continue consuming
disk/build resources, so this is not stable free-space capacity.

The source review is preserved in
`gemma4-ane-cache-lifetime-research-20260915.md`. Signing-identifier stability
addresses one namespace issue, but same-executable cache losses require further
explanation. A bounded, two-compile source-retention fixture is being checked.
Earlier descriptions of "normal unloads" mean `unload_returned` records only:
the old implementation discarded the driver's Boolean/NSError. The updated
implementation records `unload_completed` or `unload_failed` and reports the
error, without retrying a rejected unload. No earlier unload-success claim is
inferred from those old records.

### Source lifetime and full cache inventory

The two 30 MiB source-lifetime fixtures both remained cached after unload and
in two fresh processes, including a delayed check 63.388 seconds later. Exactly
two compiler calls, six successful loads/unloads and zero evaluations occurred.
The retained arm kept its source paths until unload; the immediate arm removed
them after load. This did not reproduce the missing-entry defect, so production
source retention remains unchanged. See `cache-lifetime-pair-20260915/` and the
cache-lifetime research report for the narrower experimental boundary.

`--inspect-ane-cache PART` now checks each real graph using strict existing-cache
loads, records confirmed misses and stops on other constructor/load failures.
It performs no compilation or inference; source staging is still recreated.
Provisioning now also enforces its existing documented per-part compiler maximum
with the shared process-wide atomic budget. The host absence/error classifier
test passed; the ANE system host gate passed five tests with six hardware tests
ignored. Source cleanup and qualified single-input/output policy are unchanged.

The release binary with vectorized preparation and cache inspection was copied,
signed with the preserved research identifier and verified. Its SHA is
`1b28218e7014de5c4d8659d4caf98f1d73bcc4e7ef890502e46febfc87ac1952`.
The final CLI help line was added after this binary compiled; parser/inspection
behavior was present in the tested executable. See
`strict-full-cache-inspection-20260915/` for the binary identity, full missing
entry lists, per-phase CPU/power records and durable journal.

The inventory loaded 17/48 QKV, 20/48 output, 20/48 INT8 FFN and 5/18 attention/head
graphs: **62 available, 100 absent**. It completed zero compiler calls, zero
evaluations and 62 checked-successful unloads. The boot stayed unchanged. This
confirms cache availability remains a material startup blocker under the same
signing identifier, rather than a small bounded-repair case. It does not identify
the system's eviction reason. Free-space observations changed substantially
during other host activity; a single high reading is not reserved capacity for
rebuilding missing entries. No full recovery was attempted from this inventory.

The runtime worker still lacks live A/B/A and cancellation acceptance, and the
HTTP adapter remains pending. The earlier qualified disaggregated CLI evidence
is preserved; current source tests or cache inspection do not replace those
remaining live gates. Historical timings remain unsuitable for comparison to
the new power-recorded series.
