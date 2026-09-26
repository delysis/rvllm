# Gemma 4 12B Metal: bounded second decode round

## Delivery and evidence status

Base: **593e1d6fb25088608198f2248dcac038d19fe761**, using the supplied exact source subset. The patch is relative to the bytes in that archive, not to a moving branch. It adds **eight default-off candidates, nine append-only dispatch slots, eleven Metal files**, a stricter offline campaign protocol, extended correctness tests, and sealed-artifact capture. It does not change a production default, checkpoint format, loader, queue implementation, or pre-existing Metal shader.

**Implemented source, not device-qualified code. Rust compilation, rustfmt, Clippy, Metal compilation, GPU execution and real-checkpoint inference were unavailable in the delivery environment.** Source checks and CPU numerical models are explicitly weaker evidence. Apply/reverse verification and exact source hashes are provided separately with the download. Review the patch and run the local gates below before queue timing or merging.

The archive has no workspace `v3/Cargo.toml`, Cargo.lock, dependent workspace crates, or runtime implementation. It also omits two pre-existing `include_str!` fixtures, `v3/tools/gemma4_metal_catalog.json` and `v3/tools/global-decode/family.json`. These were not fabricated. Use the full checkout of the exact commit locally. Additional runtime queue/CLI context was read from that commit through GitHub; no omitted runtime source was modified.

### What the intake establishes

The supplied `reports/gemma4-decode-round-20260926/INTAKE.md` and JSON remain unchanged. Their M4 Max observations compare complete **rvLLM incumbent operators**:

| Prior arm / cell | Point estimate | Interpretation for this round |
|---|---:|---|
| Fused BF16 gate/up/GELU | 1.74× | Strong reason to preserve its arithmetic and explore two small scheduling changes. |
| Bounded global attention, L256 and L512 | 30.55× at each length | Strong motivation for bounded longer-context screens; not evidence at L1024/L2048. |
| W4 K15360 | 1.31× | Failed 5% drift; follow up, not a winner. |
| W8 K8192 | 1.10× | Narrowly failed drift; follow up, not a winner. |
| W8 K4096 | 0.83× | Regression with severe drift/order sensitivity; redesign/reject cell. |

None of these are MLX comparisons or full-model speedups. No new performance measurements are supplied here. Comparing separate new-vs-incumbent cells is not a paired new-vs-prior-winner experiment; preserve that distinction when ranking schedules.

## Candidate matrix

All candidates require the existing Apple9/typed-BF16/public-PSO checks. Selectors are exact strings; default remains `off`. Source threadgroup memory below is a declaration, **not measured registers, spills, occupancy or ISA**.

| Selector | Ledger slots | Geometry / declared resources | Reason for the arm |
|---|---:|---|---|
| `metal-ffn-bf16-r2-sg2` | 60 | 64 threads; 4 rows/TG; grid 3840; 0 shared bytes | Reduce per-lane gate+up source accumulators from 8 to 4; double TG count versus prior FFN. |
| `metal-ffn-bf16-r4-sg4` | 61 | 128 threads; 16 rows/TG; grid 960; 0 shared bytes | Keep four gate/up row pairs per SIMDgroup; double SIMDgroups/TG, halve TG count. |
| `metal-qmv-w4-g32-r4-sg4` | 62 | K15360, N3840; 128 threads; 16 rows/TG; grid 240 | Four rather than eight accumulators per lane, with four SIMDgroups; retain native W4 group-32 layout. |
| `metal-qmv-w8-g32-r4-sg4-k8192` | 63 | K8192, N3840; same 128-thread geometry | Shape-specific drift follow-up; no K4096 admission. |
| `metal-qmv-w8-g32-r2-sg4-k4096` | 64 | K4096, N3840; 128 threads; 8 rows/TG; grid 480 | Separate two-row/SIMDgroup redesign; more TG parallelism, less reuse, explicitly unqualified. |
| `metal-global-d512-stream-r4t128-c2048` | 65 | Grid `[4,1,1]`; 128 threads; 2048 shared bytes; no global scratch | Cooperating four-head TG streams staged K/V; keeps the prior fixed-64 QK reduction. |
| `metal-global-d512-stream-r1t32-c2048` | 66 | Grid `[16,1,1]`; 32 threads; no shared memory or global scratch | One head/TG, direct loads, no cross-SIMDgroup staging barriers. |
| `metal-global-d512-split-stream-r4s256t128-c2048` | 67–68 | Partial `[4,8,1]`, 128 threads, 2048 shared bytes; merge `[16,1,1]`, 32 threads; 263168-byte scratch | Eight fixed 256-token partitions plus existing fixed-order FP32 merge; more parallel work, with its complete extra cost timed. |

New kernel names are the selector without `metal-`, with hyphens changed to underscores and prefix `research_`. The split family appends `_partial` and `_merge`. `round2-plan.json`, generated locally, lists exact names and resources; the unbound source plan is `tools/decode-round/campaign-plan.round2.json`.

Keep the four historical selector names for fresh controls: `metal-ffn-bf16-r4-sg2`, `metal-qmv-w4-g32-r8-sg2`, `metal-qmv-w8-g32-r8-sg2`, and `metal-global-d512-short-r4t128`. Old ledger slots 0–59 and the original 44-candidate catalog prefix retain their identities. New totals are 69 slots and 52 candidates. Dispatch schema remains `rvllm.metal.research-dispatch.v4`; array size and executable identity still must match. Do not combine receipts from different binaries merely because their schema string matches.

## Admission, arithmetic and integration

### Bounded attention, not silently broader capacity

The three new global arms require one decode token, one sequence, 16 Q heads, one KV head, D512, global window, scale exactly 1, typed BF16 input, and the existing Gemma model/arena contract. Both host plans and shader guards require **allocated logical capacity `max_blocks * block_size <= 2048`**, with checked host arithmetic. This is not just a live-prefix limit.

Thus an L256 prefix inside a 4096-capacity cache refuses before dispatch and uses the incumbent route. L2048 with 32-token pages fits; L2048 with seven-token pages would allocate capacity 2051 and refuses. The historical short selector retains its 512-capacity bound. No selector accepts L4096, or a padded capacity larger than its declaration, by accident.

Unsplit variants preserve the prior lane-zero-broadcast fixed-64 FP32 QK association and online FP32 softmax/PV, then round once to BF16. Cooperating and split variants share a read-only K/V staging helper. The solo variant removes that staging and its barriers. Split uses eight disjoint 514-float/head records: 512 numerator elements, maximum and denominator. Every valid partition writes its complete record, including wholly future or all-hole partitions, before the ordered merge. Persistent scratch is selected and allocated during model setup, not per token.

K/V are never written by these consumers. The existing owning-layer cache-write encoder remains the newest-token producer, ordered before attention. Shared-KV consumers keep the owner's offsets. Negative page IDs are holes; visible invalid positive IDs veto writes; future metadata and poisoned future K/V are not read. All-hole output is zero. Rollback is position-bounded rather than implemented by clearing the cache.

Host admission refusal creates no encoder and preserves the incumbent fallback. A deliberately malformed GPU metadata/launch test is different: its encoded dispatch is counted but its shader performs no writes. It is not represented as a successful fallback or useful operator result.

### BF16 FFN

Both new FFN arms use native BF16 Gate||Up `[30720,3840]`, input `[1,3840]`, activated output `[1,15360]`. They retain FP32 projections, the **BF16 gate and up materialization boundary before GELU**, the incumbent clamped tanh-GELU arithmetic, and BF16 final output. They fuse two existing encoders into one; neither includes down projection, residual, or normalization.

`GateUpRequest::plan` refuses prefill, wrong shape/type, quantized accumulation, low-bit gate/up descriptors, gate/up trace capture, incompatible PSO, unaligned/short/overflowing spans, or output aliasing. Refusal reaches the pre-existing path. The variants change scheduling, not weight bytes or arithmetic policy. Reduced source accumulator arrays are only a compiler hypothesis; the capture tool must establish whatever resource evidence the device exposes.

The old timing fixture is intentionally unchanged. A **second, full-shape dense BF16 correctness fixture** now exercises every K contribution and every output row, using an independent scalar FP64 construction with incumbent BF16 boundaries. Both the candidate and incumbent must pass the existing predeclared bound; three repeated bit-exact uses and all guards remain mandatory. This dense fixture is created outside timing. It is synthetic, not a checkpoint or distributional-accuracy claim.

### Group-32 QMV and prefill safety

Native signed W4 nibbles / W8 bytes, row-major group-32 **FP16** scales, FP32 scale application/FMA and BF16 activation/output remain unchanged. The source assembly explicitly bypasses generic half-to-bfloat rewriting for these shaders. Authentication remains the existing loader/descriptor contract, not something synthetic fixtures can establish.

New selectors admit only their exact K, N3840, M1 and role: W4 down projection or W8 output projection. Both layer-forward entry points retain explicit `MetalPhase::Decode` and BF16 checks. Prefill never enters these research QMV routes. A targeted BF16 decode that refuses the new schedule retains the incumbent BF16 n4 fallback, not the legacy F16 interpretation. The existing method name `try_encode_strided_bf16_r8_sg2` remains as a compatibility entry point; its immutable selected launch now determines the actual kernel/geometry.

### Real layer/full decode route

The existing layer-forward FFN adapter and unsplit/split attention adapters accept the appended selectors through their existing immutable pipeline cache. No new hot-path environment reads, logs, heap containers, buffer allocation, cache construction, or source compilation were added. Existing ledger updates and PSO admission remain in place. Persistent arena checks are not a claim that Apple's encoder internals allocate nothing.

One pipeline owner selects **one research candidate**. Independent FFN-only, attention-only and QMV-only full-route arms are supported. A combined FFN+attention policy is not implemented or claimed. The omitted external runtime aggregate encoder-count consumer must be audited locally: FFN is two incumbent encoders versus one fused encoder; split attention is one incumbent versus partial+merge. Its totals must reconcile with the operator ledger before qualification. Do not weaken or disable that gate if it exposes a mismatch.

## Correctness and measurement gates

Native oracle coverage retains noncontiguous/reverse pages, holes at multiple positions, all holes, speculative future poisoning, rollback, modified newest K/V, irregular page tails, equal-logit cases, malformed visible metadata, output/scratch guards, immutable inputs, three repeats, and persistent arena identity. The split extension includes 21 host refusals and four malformed shader-metadata refusals. Core-only timing libraries exclude the GPU oracle kernel.

Unsplit correctness retains exact comparison with its fixed-order GPU FP32 oracle and exact once-rounded BF16 output, plus independent FP64 absolute `5e-4` and relative-L2 `1e-4` bounds. Split retains its FP32 `5e-5` bound and adds FP64 `5e-4` / relative-L2 `1e-4`, repeated-use identity and once-rounded BF16. FFN/QMV retain the original `0.015625*abs(reference)+1e-5` bound. No tolerance was widened.

The new split timing test is `attention_global_decode_device_tests::global_decode_split_stream_abba`. It measures a **single complete command buffer with 100 interleaved partial-then-merge operations**, not a partial-only kernel and not separately batched component timings. Candidate samples therefore contain 200 compute encoders; control samples contain 100. New split schema is `rvllm.global-decode.abba.v3`. Legacy split v2 collection remains available for historical work but cannot authorize this protocol's advancement.

Every screen requires five warmups per arm, ten alternating ABBA/BAAB blocks, all 40 samples in exact block/position order, 100 complete operations per sample, positive finite GPU and synchronized wall time, actual encode accounting, zero source compilations during sampling, exact source/build/metallib/test identities, native oracle/output pins and raw sample files equal to the aggregate. FFN controls account for 200 encoders versus 100 candidate encoders. GPU timestamps, not parent queue wall time, are the metric.

The offline `rvllm.decode-round.screen.v2` rule permits advancement only if control max/min drift is at most 5%, **both** order-stratum median speedups exceed one, and the deterministic whole-block bootstrap 95% lower endpoint exceeds one. The bootstrap is a screen diagnostic, not proof of independence or full-route benefit. Drift failure remains a valid retained collection with `promising=false`; incomplete/mutated work is an error. No sample pruning, retry-on-loss, or automatic promotion exists.

Later attention manifests require recursive revalidation of the exact earlier chain: L256 → L512 → L1024 → L2048. Immutable proof receipts and all underlying pins become next-stage inputs. Changing a summary cannot bypass the gate. Operators use LENGTH=0 with explicit K; they never pretend to have a KV length.

### Observations, not quiet-process or thermal dwell

Use `conditions.round2-observed-ac.json`: `stable_seconds=0`; thermal, low-power and pmset constraints null; quiet processes and idle-server gates empty; disk threshold zero. Cargo/rustc activity is **observed**, never a quiet-process blocker. No waits for a thermal settling interval or unchanged readings are introduced. Queue wait bounds are expiration limits, not a request to wait for cargo or cooling.

The unchanged queue still requires fresh healthy observation data, valid known controls, the declared AC/battery stratum, and its existing CPU-restriction/ownership integrity checks. This patch does not bypass those. Exploratory timing can retain ineligible condition samples and must not promote from them. Review thermal/power/process strata explicitly; do not pool incomparable confirmation conditions. The serial queue lock is still mandatory. Failed or incomplete attempts are preserved, not repaired or replayed.

## Exact local workflow

Run in a full clean checkout of the exact base, with Xcode Metal tools, the repository toolchain/dependencies already available offline, and `jq` for the explicit shell helpers below. Replace only the absolute user-specific paths and campaign ID. Freeze sources/binaries before generating jobs.

### 1. Apply, host checks and freeze

```sh
cd /absolute/rvllm
test "$(git rev-parse HEAD)" = 593e1d6fb25088608198f2248dcac038d19fe761
git apply --check /absolute/rvllm-593e1d6f-round2.patch
git apply /absolute/rvllm-593e1d6f-round2.patch
git diff --check
cd v3
cargo fmt --all --check
cargo test --offline --locked -p rvllm-apple-metal --lib
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-global-decode-jobs
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-global-decode-report
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-retain-abba
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-metal-artifact-evidence
cargo clippy --offline --locked -p rvllm-apple-metal --all-targets -- -D warnings
cargo build --offline --locked --release -p rvllm-apple-metal \
  --bin rvllm-global-decode-jobs --bin rvllm-global-decode-report \
  --bin rvllm-retain-abba --bin rvllm-metal-artifact-evidence
cargo build --offline --locked --release -p rvllm-runtime \
  --features macos-private-ane-research --bin rvllm_experiment_queue \
  --bin rvllm_metal_infer
cargo test --offline --locked --release -p rvllm-apple-metal --lib \
  --no-run --message-format=json > /absolute/round2-test-build.jsonl
```

Rustfmt may require formatting edits: it was unavailable here. Make any required source repair/formatting change **before** freezing hashes. Run the full suite again after repairs; never make a queued source/binary silently drift. Existing ignored device tests are not executed by the non-ignored test commands above.

### 2. Prepare an explicit campaign

```sh
export V3=/absolute/rvllm/v3
export BIN="$V3/target/release"  # use the actual Cargo target directory if overridden
export ROOT=/absolute/experiments/g4-r2-a
export QUEUE=/absolute/queues/g4-r2-a
export LOCK=/absolute/queues/apple-accelerator.lock
export ID=g4-r2-a
export JOBS="$BIN/rvllm-global-decode-jobs"
export Q="$BIN/rvllm_experiment_queue"
export TEST_EXE="$($JOBS test-exe /absolute/round2-test-build.jsonl)"
export CONDITIONS="$V3/tools/decode-round/conditions.round2-observed-ac.json"

# ROOT must not exist. The generator pins real binaries, sources and tools.
"$JOBS" prepare-round2 "$ID" "$ROOT" "$QUEUE" "$TEST_EXE" "$CONDITIONS" \
  metal-ffn-bf16-r4-sg2 metal-ffn-bf16-r2-sg2 metal-ffn-bf16-r4-sg4 \
  metal-qmv-w4-g32-r8-sg2 metal-qmv-w4-g32-r4-sg4 \
  metal-qmv-w8-g32-r8-sg2 metal-qmv-w8-g32-r4-sg4-k8192 \
  metal-qmv-w8-g32-r2-sg4-k4096 \
  metal-global-d512-short-r4t128 \
  metal-global-d512-stream-r4t128-c2048 \
  metal-global-d512-stream-r1t32-c2048 \
  metal-global-d512-split-stream-r4s256t128-c2048

# Explicit submission of an already generated stage; no future-stage generator.
submit_stage() {
  jq -r '.[]' "$1" | while IFS= read -r job; do
    "$Q" submit "$QUEUE" "$ROOT/jobs/$job.json" || return 1
  done
}
submit_stage "$ROOT/compile-jobs.json"
"$Q" run "$QUEUE" "$LOCK" 600
"$Q" status "$QUEUE"
```

A queue `run` is the existing serial worker, not a thermal/quiet-process dwell command; its last argument is idle exit timeout. Inspect terminal reports before continuing. With all twelve selectors above there are fifteen compile jobs: core for every selector, plus oracle libraries for three unsplit attention selectors. No timing has yet been generated.

### 3. Correctness first; then shortest screens

```sh
"$JOBS" oracle-jobs "$ROOT"
submit_stage "$ROOT/oracle-jobs.json"
"$Q" run "$QUEUE" "$LOCK" 600

"$JOBS" timing-jobs "$ROOT" 0 \
  metal-ffn-bf16-r4-sg2 metal-ffn-bf16-r2-sg2 metal-ffn-bf16-r4-sg4 \
  metal-qmv-w4-g32-r8-sg2 metal-qmv-w4-g32-r4-sg4 \
  metal-qmv-w8-g32-r8-sg2 metal-qmv-w8-g32-r4-sg4-k8192 \
  metal-qmv-w8-g32-r2-sg4-k4096
submit_stage "$ROOT/timing-jobs-L0.json"
"$Q" run "$QUEUE" "$LOCK" 600

"$JOBS" timing-jobs "$ROOT" 256 \
  metal-global-d512-short-r4t128 \
  metal-global-d512-stream-r4t128-c2048 \
  metal-global-d512-stream-r1t32-c2048 \
  metal-global-d512-split-stream-r4s256t128-c2048
submit_stage "$ROOT/timing-jobs-L256.json"
"$Q" run "$QUEUE" "$LOCK" 600
"$JOBS" review-round2 "$ROOT" 0 metal-ffn-bf16-r2-sg2 \
  metal-ffn-bf16-r4-sg4 metal-qmv-w4-g32-r4-sg4 \
  metal-qmv-w8-g32-r4-sg4-k8192 metal-qmv-w8-g32-r2-sg4-k4096
"$JOBS" review-round2 "$ROOT" 256 \
  metal-global-d512-stream-r4t128-c2048 \
  metal-global-d512-stream-r1t32-c2048 \
  metal-global-d512-split-stream-r4s256t128-c2048
```

Choose the explicit promising names after review. For example, if the cooperating arm passes, the following creates only its next stage; each subsequent command refuses unless the complete earlier chain still passes:

```sh
export C=metal-global-d512-stream-r4t128-c2048
"$JOBS" timing-jobs "$ROOT" 512 "$C"
submit_stage "$ROOT/timing-jobs-L512.json"
"$Q" run "$QUEUE" "$LOCK" 600
"$JOBS" review-round2 "$ROOT" 512 "$C"

# Execute these only after reviewing the preceding stage.
"$JOBS" timing-jobs "$ROOT" 1024 "$C"
submit_stage "$ROOT/timing-jobs-L1024.json"
"$Q" run "$QUEUE" "$LOCK" 600
"$JOBS" review-round2 "$ROOT" 1024 "$C"
"$JOBS" timing-jobs "$ROOT" 2048 "$C"
submit_stage "$ROOT/timing-jobs-L2048.json"
"$Q" run "$QUEUE" "$LOCK" 600
"$JOBS" review-round2 "$ROOT" 2048 "$C"
```

Name all candidates selected for a stage in **one** generation command, because its manifest list is immutable. Do not successively rewrite a stage list for different candidate subsets. A refused advancement or loss is terminal evidence, not permission to prune or resample.

For confirmation, use a new empty root/queue and a distinct ID with the same frozen binaries:

```sh
"$JOBS" confirm-round2 g4-r2-b /absolute/experiments/g4-r2-b \
  /absolute/queues/g4-r2-b "$TEST_EXE" "$CONDITIONS" "$ROOT" "$C"
```

Then explicitly repeat compilation, correctness and the shortest-first chain in the new campaign. A linked confirmation config is provenance, not a passing confirmation result. The command checks distinct campaign IDs and frozen generator/test identities; it does not declare either campaign qualified or conceal an earlier failure.

### 4. Capture the actual timed Metal artifact

For a new candidate, core build IDs are exactly `$ID-${C#metal-}-compile-core`. Run collection outside timing and into a fresh directory:

```sh
export BUILD="$QUEUE/results/$ID-${C#metal-}-compile-core/build/build.json"
export CAPTURE=/absolute/evidence/g4-r2-a-cooperating
"$BIN/rvllm-metal-artifact-evidence" "$ROOT/$C-core.metal" "$CAPTURE" \
  --compiled-build "$BUILD" research_global_d512_stream_r4t128_c2048
```

For split, request both `_partial` and `_merge` kernel names. This mode hashes and copies the existing queue AIR/metallib and does **not** recompile or relink them. A default independent recompilation is distinctly labeled and cannot stand in for the timed binary. Missing objdump/PSO evidence is missing evidence, not inferred register residency. Retain original build tool pins as well as inspection-tool versions.

### 5. Real checkpoint and full route: mandatory, still unrun

The existing real-weight CLI was not given a fabricated `--candidate` mode. Synthetic operator receipts remain `checkpoint_qualified:false`. Locally load the authenticated checkpoint descriptors and exercise the actual selected adapter through the real Gemma layer before claiming real-weight QMV performance. Verify every touched layer/role, packing and FP16 scales against its sidecar identity, not merely K/N compatibility.

The existing single-prompt CLI supports this preliminary full-route invocation; it is **not** a complete qualification harness:

```sh
export MODEL=/absolute/authenticated-gemma4-12b
RVLLM_METAL_RESEARCH=metal-global-d512-stream-r4t128-c2048 \
  "$BIN/rvllm_metal_infer" --model-dir "$MODEL" \
  --prompt 'Explain why the sky is blue.' --max-new-tokens 32 \
  --max-total-tokens 2048 --large-model-opt-in --top-logits 16 --json \
  > /absolute/evidence/g4-r2-a-route.json
```

Use the existing checkpoint's supported BF16 preparation/sidecar configuration; do not infer a dtype or introduce a new environment variable. Repeat with selector `off` and each separately qualified research selector, retaining identical prompt/token IDs and immutable checkpoint evidence. A short prompt does not establish L1024/L2048 behavior. The local full-route gate must additionally establish:

- Correct per-layer source/PSO selection, exact compile/dispatch/total-encoder deltas, and no compilation during decode; captured gate/up traces intentionally force fallback.
- Unchanged prefill outputs and route/ledger behavior, including all three QMV K cells; unsupported M, capacity, dtype, role and shape preserve incumbents.
- True owner cache-write → consumer ordering, shared-KV reuse, newest-token changes, rollback and long-context behavior under a real model, not only prepared synthetic pages.
- Reference-backed intermediate tensors, logits/top-k/token sequences, longer generation and the repository's predeclared numerical/perplexity/quality gates; one plausible answer is not qualification.
- Cold prepare separated from steady decode, equivalent complete-route work, all paired samples/order strata/drift, and an independent campaign confirmation. Include FFN/attention extra or saved encoders and split scratch costs.

The external total-count consumer and any missing real-checkpoint referee wiring are local integration work, not claimed complete by this patch. No default changes follow automatically from operator success.

## Same-operation MLX comparison contract

`tools/decode-round/mlx-operation-contract.round2.json` is an explicit **plan-only** instrument contract. There is no new MLX runner, fabricated trace, tensor exporter, or measurement in this delivery. The rvLLM side gains exact timed-artifact capture; the MLX side remains a local gate.

For FFN, compare the entire Gate||Up projection plus GELU product with matching BF16 boundaries—not just the activation. For attention, use Q `[1,16,1,512]`, K/V `[1,1,L,512]`, scale 1 and explicit position/hole visibility. MLX's documented fused attention supports GQA without tiling K/V and uses FP32 softmax. A contiguous-gathered math-only comparison must disclose the gather outside timing; it is not the same paged operation. A complete paged comparison must include required gather/layout/mask/output work. Specify all-hole handling rather than silently substituting a different mask semantics.

MLX's documented affine `quantized_matmul` uses packed uint32 weights and scale/bias parameters. That is not automatically rvLLM's signed nibble/byte, group-32 FP16-scale ABI. Require lossless reconstructed-weight and scale checks over the entire matrix plus matching precision/layout before a same-storage claim. Otherwise label a dense-dequantized comparison as a different storage/workload baseline or reject the comparison.

MLX evaluation is lazy. Build fresh complete operations, force output evaluation and synchronize; repeatedly evaluating a cached array is a no-op, not inference. Record the installed MLX version/commit, actual traces and input/output hashes, not the current documentation version as a surrogate for a local installation. Use the same complete 40-sample, order-stratified, drift-gated protocol and retain all data. Neither launch counts alone nor a public API name proves identical device work.

Primary documentation inspected for the plan:

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html

## Receipt identities and review map

| Artifact | Expected identity / scope |
|---|---|
| `campaign.json` | `rvllm.global-decode.campaign.v1`, exact base declaration, frozen executable pins, explicit names, `screen_protocol=rvllm.decode-round.screen.v2`, no promotion. |
| `round2-plan.json` | Exact selected geometry, K/capacity, kernel family/resource declarations, 10 blocks/40 samples, no retry/autosubmit. |
| Queue job/result | Existing `rvllm.experiment_job.v1` / `rvllm.experiment_result.v1`; generated/submitted manifests match, inputs unchanged, exit 0, not overdue. |
| `build/build.json` | `rvllm.global-decode.build.v1`; flags exactly `-std=metal3.1`, `-fno-fast-math`; source, AIR, metallib and tool identities. |
| Operator oracle | `rvllm.decode-round.oracle.v1`; exact K; FFN sparse plus dense cases; repeated-use, bound, buffer and dispatch gates. |
| Unsplit oracle | `rvllm.global-decode.oracle.v1`; exact GPU FP32/once-rounded BF16 plus independent FP64 bounds and per-case output hashes. |
| Split oracle | `rvllm.global-decode.split-oracle.v1`; new streaming bounds and refusal evidence, three repeats, exact per-case BF16 output hashes. |
| Operator/unsplit timing | `rvllm.global-decode.abba.v1`; exact candidate/source/library/test/oracle identity and all 40 samples. |
| New split timing | `rvllm.global-decode.abba.v3`; metric `complete-interleaved-partial-merge-command-buffer`; 100 operations and 200 candidate encoders/sample. |
| Advancement proof | `rvllm.decode-round.screen.v2`; current statistics plus exact raw dependencies; summary alone never authorizes progression. |
| Artifact capture | `rvllm.metal_artifact_evidence.v1`; `artifact_origin=exact-queue-compiled-artifact` when using the new sealed-build option. |

Primary review paths under `crates/rvllm-apple-metal/src/`: `research.rs`, `research_catalog.rs`, `research_evidence.rs` for append-only registration; `research_decode*.rs`, `low_bit_metal.rs`, `layer_forward.rs` for operator admission/routing; `attention_global_decode*.rs` and new shaders for capacity/visibility/split timing; `decode_round_campaign.rs` and `bin/rvllm-global-decode-jobs.rs` for fail-closed progression; retainer/report binaries for drift preservation; artifact-evidence binary for source-grounded resource capture.

## Host evidence supplied with the download

The host validation archive includes the actual test sources/commands and JSON outputs, not invented native receipts:

| Executed check | Result and exact evidential limit |
|---|---|
| Eight archive/source invariant checks | Passed: original 60 ledger slots and 44-candidate prefix; no added unsafe Rust; 58 Rust files lexically balanced; 56 original shader/Cargo/intake files byte-identical; decode/prefill routing, bounded read-only attention and observation policy checks. Lexical checks do not parse or type-check Rust. |
| Eight candidate-source C++17 shim syntax checks | Passed with Clang. Metal attributes/address spaces and SIMD/barrier behavior are stubbed; this is **not Metal compilation or execution**, nor exact full generated-library compilation. |
| CPU numerical model | 26 attention cases × 16 heads; maximum FP64 absolute error `2.4048337875226533e-6`; maximum split-vs-unsplit FP32 error `2.2053718566894531e-6`; 36 sampled QMV rows across the three full-K shapes and 12 sampled dense-FFN rows at full K3840 passed their model checks. Host C++ math is not Metal arithmetic, packing/device execution, all-output native correctness, or performance evidence. |
| Patch consistency | See packaged apply-check JSON for clean apply, changed-file byte equality and reverse-check results against a fresh extraction of the supplied base. |
| Rustfmt/Cargo/Clippy/Apple tools | Unavailable; blocked attempts and missing workspace/include files are recorded. No Rust tests, MSL compile, real GPU test, MLX trial, real checkpoint or full-model gate is claimed passed. |

The limits are intentional: neither an attractive source schedule nor a CPU model determines M4 Max register residency, occupancy, bandwidth, attention speed at longer contexts, checkpoint accuracy or production readiness. The artifact and campaign paths are designed to gather that evidence without changing the incumbent or erasing failed observations.
