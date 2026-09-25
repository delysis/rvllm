# Gemma 4 Apple kernel campaign — current handoff, 2026-09-25

This section supersedes every older status or pause statement below. Work is
active on draft PR #4, branch `astra/gemma4-load4-tiles-20260923`, at or after
commit `df1ff2c3`. Shipping defaults remain unchanged. Local Codex owns Apple-device
execution, correctness qualification, timing, evidence retention, and any
promotion. Chat Pro may propose reviewable source changes but may not claim
local qualification.

## Current adjudicated frontier

- Global D512 BF16 decode attention: split-matrix is the stable qualified
  control. The opt-in split-32 route passes the independent native
  oracle, real Gemma route, newest-K/V, tails, holes, guards, BF16 rounding,
  exact dispatch, repeated output, and zero-compile checks, but **is not
  promotable**. Complete normal-route speedups versus split-matrix were
  1.149x/1.861x at L256, 1.015x/1.019x at L512, 0.737x/1.155x at L1024, and
  0.794x/0.683x at L2048. Separate three-sample profile medians instead report
  0.967x, 1.017x, 1.113x, and 1.140x respectively. The disagreement in
  direction at L256 and L2048 proved material cross-process/order variance;
  neither favorable subset was selected. A corrective A/B/B/A referee then
  predeclared case 0 of every fresh process as a retained warmup and measured
  cases 1 and 2. It found an L256 split-32 speedup of **1.0856x** (363.736 ms
  versus 335.063 ms), independently repeated at **1.0868x** (362.150 ms versus
  333.223 ms). Both runs had identical tokens, exact routes, zero inference
  compilation, eligible recorded conditions, and no violations. Split-32 is
  therefore a **prospective L256 winner**, not yet a production selection;
  longer-context advancement remains required. Raw reports and condition
  journals are checked in at
  `reports/gemma4-split32-normal-route-20260925/`.
- Native-BF16 Metal W4/W8 projection baseline: all seven roles (Q/K/V/O,
  gate/up/down), both formats, and M=1/M=4 pass real-weight correctness,
  exact dispatch, guards, and repeated-use checks. Only 3/28 cases repeated as
  stable >=1.05x wins under the 20% drift rule: V/W4/M1 (1.291x, 1.150x),
  Up/W4/M4 (2.220x, 2.221x), and Down/W4/M1 (1.880x, 1.945x). The campaign is
  **not promotable**; genuinely tiled native-BF16 W4 and W8 kernels are the
  next Metal priority. See
  `reports/gemma4-metal-low-bit-bf16-campaign-20260925/`.
- Native-BF16 N4 schedule: a four-output-per-SIMD candidate now passes every
  real-weight correctness case across all seven roles, W4/W8, and M1/M4. Eight
  of 28 timing cells repeat as stable wins under the same strict policy: V/W4/M4,
  Gate/W8/M4, Up/W8/M1+M4, and Down/W4+W8/M1+M4. The remaining 20 cells are
  unstable or not faster, so this is a role-specific research candidate rather
  than a generic selector. See `reports/gemma4-metal-low-bit-n4-20260925/`.
- ANE baseline versus stacked FFN: both routes are exact for the ten-token
  continuation, provisioned 210/210, and compile-free during inference.
  Corrected counterbalancing found mean stacked-minus-baseline FFN +0.016 ms
  and total +18.210 ms. Stacked is correctness-qualified but not a speed
  winner. See `reports/gemma4-ane-stacked-baseline-exact-v2-20260925.md`.
- Generated-code evidence seals source, AIR, metallib, compiler, device, and
  public PSO resource data. Apple public tooling does not expose supported
  register-count, residency, or occupancy claims; do not manufacture them.
- Checkpoint-specific W4/W8 quality-referee contracts exist, but full-model
  calibrated logit/perplexity observations for every checkpoint-format pair
  are still missing.

## Required next implementation round

1. Metal W4/W8: replace the one-output-per-SIMD correctness baseline with at
   least two materially different tiled native-BF16 schedules per format.
   Reuse unpacked values and scales across outputs; keep FP32 accumulation and
   one BF16 storage boundary. Screen every role at M=1, then advance plausible
   arms to bounded prefill M and full-route tests.
2. Attention: retain split-matrix as control. Do not promote split-32 or build a
   selector from its operator-only numbers. Investigate why its full-route
   route measurements disagree despite isolated-kernel wins, with command
   submission, scratch, synchronization, partial, and merge time separated.
   Full-route comparisons execute a predeclared counterbalanced route order
   inside one queue job; separate control-then-candidate jobs are diagnostic
   only. The first corrective L256 ABBA run completed at 0.674x using every
   observation, but exposed
   a large process-first transient that would reverse the answer if second
   cases were cherry-picked. The predeclared warmup-controlled referee and its
   independent confirmation now agree at 1.0856x and 1.0868x respectively.
   At L512 the A/B/B/A and B/A/A/B orders disagreed at 0.708x and 1.048x.
   Their combined eight-observation median is only 1.025x, with >2.4x ranges
   in both arms, so L512 is inconclusive and below margin. A bounded L1024
   L1024 likewise disagreed at 0.979x and 1.068x; its combined result is only
   1.038x with >1.8x ranges. Split-32 is therefore rejected as a current
   selector candidate beyond its repeatable L256 win. Do not run L2048 without
   a new overhead or synchronization hypothesis.
3. Prefill: implement a separate tiled online-softmax experiment and an
   explicitly hardware-gated TensorOps arm. Do not extrapolate the decode
   policy or call TensorOps ANE evidence.
4. Device-resident decode: implement the smallest bounded command-buffer token
   loop that keeps dependency state, append-visible K/V, and sampling state on
   device. Disabled instrumentation must add no per-layer allocation,
   synchronization, or logging.
5. ANE: prioritize graph fusion and launch reduction across INT8, BF16, and
   storage-only LUT4 experiments. Native 4-bit arithmetic must not be claimed
   without device evidence. Preserve exact cache identity, evaluation counts,
   zero-compile inference, exact route, and no-fallback receipts.
6. Quality: run the checkpoint-bound W4/W8 logit/NLL/perplexity referee after
   calibrating thresholds against BF16. Operator agreement alone is not model
   acceptance.

The persistent experiment queue is advancing the warmup-controlled split-32
campaign. It uses `stable_seconds=0`; changing conditions are recorded rather
than used as a thermal-stability dwell gate. Short screens must precede longer
contexts, and failed or unfavorable receipts must remain retained.

## Chat Pro status

The immutable job `rvllm-gemma4-coreai-sprint-20260925-v2` contains the full
implementation brief and current evidence. It remains unsent. Bridge fixes for
the Chat continuation interstitial and safe same-ID exhausted recovery pass
their regression suites and are installed, but Chrome has not reconnected the
extension/native host after stale-host cleanup. Do not invent a new job ID or
manually click Send. Once bridge health is responsive, resume this exact ID and
require explicit acknowledgement plus delivered package evidence before using
its output.

---

# Historical checkpoint — 2026-09-22

## Superseding Chat Pro Astra brief

The user explicitly resumed controlled candidate work on 2026-09-22 and asked
that all further extensive work be directed by this file. This section
supersedes the older pause wording below. **Chat Pro Astra must only return
reviewable patches, analysis, and tests in a packet. It must not claim to run,
qualify, time, cache, or promote kernels. The local Codex owner alone applies
packets, runs all builds and host/device tests, performs timing, evaluates
evidence, and promotes accepted work.** It is committed and pushed on
`origin/codex/gemma4-kernel-candidates` at
`ca814876b01d986a28a46bd6c94edeec1a5b6ddb`; work from that exact branch, not
from an assumed local checkout or an older eight-patch packet.

### What is actually established

- The branch contains the original six default-off candidates, formatting
  repairs, and a local Metal 3.1 ABI repair for `metal-gqa-kv8`. The repair
  changes `threads_per_threadgroup` from scalar `uint` to `uint3` at both GQA
  entry points and passes `threads.x` into the scalar helper. It was necessary
  because this machine's Metal compiler rejected mixed scalar/vector grid
  attributes.
- The targeted host checks passed: five ANE-candidate source/oracle tests, two
  byte-exact blocked32 KV-layout tests, and twelve Metal research-policy/source
  tests. Existing unrelated workspace warnings remain warnings; do not describe
  them as candidate failures or a clean full-workspace Clippy result.
- All three BF16 Metal candidate sources compiled and linked with the local
  Metal 3.1 toolchain after that repair. This is compiler evidence only.
- A real local Gemma 4 12B, six-token reference screen reached Metal prefill
  for baseline, short-MMA, rounded-gate, and GQA. Every run then failed before
  decode because the current client's compiled ANE cache is absent. The legacy
  CLI writes its useful report too late, so these are not durable successful
  prefill receipts and are not full-route correctness evidence.
- The six-token input cannot exercise `metal-gqa-kv8`: its admission range is
  64--1024 prompt tokens. Its apparent successful prefill is therefore only a
  fallback observation. Do not broaden that selector or claim GQA dispatch.

No candidate has a speed result, full-token continuation result, tensor-oracle
result, accepted ANE graph, or promotion to `main`.

### Current machine and authority

The old queue STOP marker was removed under the user's explicit resumption.
There is no active rvllm worker. The M4 Max is on AC, but its current power
mode is distinct from the historical battery/low-power-off/mode-0 stratum.
The local Gemma 4 12B snapshot is present. An unrelated llama-server is also
present and must never be stopped. Recheck free disk (minimum 16 GiB), boot,
power/thermal controls, lock ownership, cache availability and process policy
immediately before every live operation. Never change OS power settings, clear
evidence, kill an accelerator child, or run concurrent hardware owners.

### Required next implementation: evidence first

Do **not** add another kernel. First implement and test the following narrow
follow-up on top of `ca814876`; a user-supplied external proposal named
`rvllm-gemma4-followup-ca814876-20260922-03` describes the same design but is
not itself authoritative or present in Git.

1. Add a five-slot, safe-atomic Metal dispatch ledger owned by the existing
   pipeline owner: short GEMM, short QKV, rounded gate, sliding GQA-D256 and
   global GQA-D512. Increment only after a real encoder dispatch has been
   encoded and ended. A requested selector, eligible shape, or available PSO
   is not dispatch evidence.
2. Add a strict `--prefill-only true` mode to `rvllm_disaggregated_infer`.
   It requires one unchanged pinned HF reference and a fresh output directory,
   performs one synchronous Metal prefill, records a flushed/synced
   `case-N/prefill-result.json` before any ANE initialization, verifies the
   first token, captures the current seed/KV form, and exits before creating an
   ANE owner. Reject interactive/text/worker/interleaved/retained-Metal modes,
   ANE capture/cache operations, nonbaseline ANE/KV selections and nonzero
   compile budgets. A fallback-only candidate must write its receipt then fail.
3. Add a real delivery gate: rustfmt, the targeted host tests, private research
   CLI build, and baseline plus all three candidates in BF16 and FP16 through
   Metal 3.1 compile/link. It must reject zero-test filters, preserve failed
   output, use offline locked Cargo, and never invoke inference. Run the real
   gate locally; mock shell tests are only tests of the gate.

Keep the normal CLI continuation path unchanged. Prefill-only is functional
triage, not performance or tensor acceptance. Instrumented binaries must be
used on both sides of any future timing comparison.

### Astra packet feedback — resolve before sending another packet

The follow-up packet was applied locally and its four dispatch-ledger tests,
four prefill-screen tests, and six Python gate-contract tests passed under the
native arm64 release target. The new Rust files required rustfmt; the local
owner formatted only packet-owned paths.

The proposed *real* delivery gate then failed before compiling candidates.
Its `cargo fmt --all -- --check` validates the whole historical workspace,
which already has unrelated formatting drift in `rvllm-apple-ane-sys`,
`rvllm-apple-ffi`, `apple_continuous_worker`, and
`apple_metal_backend_tests`. It therefore cannot establish delivery health for
this packet. Do not paper over that failure by formatting unrelated source.
Return a replacement patch that makes the gate check an explicit, reviewed set
of packet-owned Rust paths (for example, `rustfmt --check --edition 2021` on
the files listed in a checked-in manifest), while retaining the gate's
zero-test, preserved-output, native-host and no-inference properties. Include
a contract test proving unrelated workspace formatting drift neither passes as
packet formatting nor prevents the packet gate from reaching its targeted
tests/compiler matrix.

The local owner will apply that packet, run its real gate, perform the
prefill-only and cache/full-route qualification, and return any further
evidence. Astra must not perform or claim those operations.

### Required execution order after the evidence change

1. Run the real delivery gate and commit only if it passes. Preserve the GQA
   ABI fix and do not reapply the original eight mboxes.
2. Run fresh prefill-only screens: baseline, short-MMA and rounded-gate with
   the six-token reference; GQA only with a separately pinned >=64-token
   reference. Require first-token equality, one-prefill/no-decode contract,
   and positive matching dispatch counts (zero for baseline). A prefill receipt
   alone does not qualify arithmetic or timing.
3. Recover the existing baseline cache separately via the established bounded
   fresh-serial-process `--prepare-ane-cache all-int8` mechanism, then perform
   a distinct strict `--inspect-ane-cache all-int8`. Require all 162 visits,
   zero compiler calls during inspection, lifecycle evidence and available
   status; exit status alone is insufficient. Never compile during a timing run.
4. Establish a fresh full-route baseline using the complete reference
   continuation and original driver/tensor oracles. Stop candidate timing if
   this fails.
5. Qualify one candidate at a time against its proper control: chunk4 after
   separately provisioning its 48 FFNs; tiles4 against existing INT8
   sliding-QKV (not FP16 QKV); blocked32 against reuse-scratch. Maintain the
   original numerical oracles. Only then run a predeclared ABBA screen with two
   warmups and seven measured requests per arm, a 5% baseline-drift rejection
   gate, isolated power strata and an independent confirmation.

Promote to `main` only after the relevant compiler, dispatch, tensor,
full-continuation and matched-timing evidence all pass. Commit/push every
reviewable source and handoff change; do not commit models, caches, frozen
executables or bulky raw artifacts.

**The user requested a pause and a checkpoint on `delysis/rvllm:main`.**
The end-to-end optimization goal is unfinished. Keep this campaign stopped
until the user explicitly resumes it. This document supersedes older reports
that describe the experiment worker as running.

Initial checkpoint: `74735053d96c9e0c6a0f16c8307db9e4ebab47f4`. Its first CI
run exposed Linux-only CLI/dependency defects; a narrow portability follow-up
is documented in [validation](reports/checkpoint-20260916/validation.md).
Always inspect CI for the latest commit, not just the original checkpoint.
CI on `12ef5efe82966fe292b1f2b0ba5973f6fe801429` passed Linux compile checks,
GB10 and the full Apple shipping/packaging checks. A subsequent test-only
follow-up synchronizes the Swift header and fixes platform-specific FFI test
fixtures; its local FFI suite passes 17/17. See the same validation report.
An automatic goal continuation is not an explicit request to resume experiments.

On 2026-09-21, the host-only queue continuation from commit
`65436c7c05ffe41285cb567cd87c8fb3a04c689b` was integrated. It keeps the
existing stability gate sampled while potentially large input pins are hashed,
rechecks STOP and the original deadline after probe I/O, and makes one final
gate observation immediately before spawning a trial. No campaign worker or
accelerator fixture was started. CI runs
[35643037368](https://github.com/delysis/rvllm/actions/runs/35643037368) and
[35643037377](https://github.com/delysis/rvllm/actions/runs/35643037377) passed
the macOS and Linux queue-host jobs, workspace check/test, GB10, and Apple
shipping safety. The same macOS queue suite passed locally (24 tests), and a
fresh STOP-only smoke exited without a power journal or result. See the
[continuation report](reports/experiment-queue-continuation-20260921.md).

## Current result

Gemma 4 12B has an exercised Metal prefill / ANE decode path, a persistent
runtime owner, and an HTTP adapter with JSON/SSE generation, cancellation and
recovery evidence. The current full-model ANE plan uses INT8 FFNs and FP16
QKV, output projections and vocabulary head. It is a private-API macOS
research route behind `macos-private-ane-research`, not a production shipping
acceptance claim. No new optimized default is promoted by this checkpoint.

| Work | Established result | Remaining boundary |
| --- | --- | --- |
| HTTP baseline | Five completed journaled requests matched 26 reference IDs; disconnect recovery and clean unloads passed | Broader quality/production acceptance is incomplete |
| Characterized ANE decode | Battery, low-power off, mode 0, nominal: median **6.0859 decode steps/s**, seven measured requests after two warmups | This is one historical power stratum, not a comparison against llama.cpp |
| Native-kit llama.cpp | Exact native-platform binding and llama.cpp revisions pinned; all 90 reference IDs matched; 49/49 layers on Metal | **No accepted slowdown ratio**; matched ABBA campaigns remain pending |
| Stacked INT8 FFN | All 624 full-model comparisons exact across 13 steps and 48 layers | No accepted end-to-end speedup; default unchanged |
| Two-token layer-major reference | 368,640 FP16 values / 96 layer outputs exact; transaction accept/reject/recovery passed; 3,728 evaluations, 162 clean unloads, zero compilation | Diagnostic S1 calls only; no production S2 route or drafter |
| Logical S2 FFN | Broad S1/S2 device output parity; original four-input CPU-qualified subset available | Short Fair pilot's three blocks all failed the 5% drift gate; result is **inconclusive**, not a 1.8x speedup |
| Interleaved S2 timing | 17 host tests, frozen signed executable, parser rejection checks, immutable AC/battery jobs | Jobs 07/08 have never run |
| Conditional Rust queue | Immutable pinned jobs, serialized ownership, power/thermal/activity gates, preserved failures, buffered report reads, and continuous launch-gate sampling through pin hashing | Latest source has **24 passing macOS host tests** and **8 portable Linux helper tests**; the normal binary passed a STOP-only smoke but is not campaign-frozen or live-qualified |
| Shared KV-import scratch | Two new host tests pass; explicit alternate path; existing import remains default; validates before writes and clears the full used surface | Captured-byte, device-continuation and timing qualification remain pending |

Baseline timing details: median prefill including KV import 802.65 ms,
completed Metal GPU interval 449.55 ms, and nine ANE continuation steps
1478.82 ms. Per-step component medians were FFN 74.82, QKV 31.19, output
21.34, vocabulary 18.91, attention 15.79 and host 3.41 ms. These separately
computed medians are not an additive total. Do not combine them with another
power stratum to predict a speedup.

## Constraints that must survive the pause

- Focus optimization on **Gemma 4 12B or larger**, GPU prompt processing and
  ANE decode. INT8 is the active ANE weight path. Historical four-bit artifacts
  are archived evidence, not authorization to pursue that route. The user's
  exception requires Apple documentation **and** corroboration on this machine,
  using Google's official native QAT weights. The native-kit baseline uses
  Google's QAT GGUF on Metal; it does not implement four-bit ANE inference.
- **Multi-I/O ANE is quarantined after a kernel panic.** Do not add selectors,
  retry the triggering route, remove refusal checks, or invent an override.
  Existing single-I/O paths are the qualified experimental boundary. Read
  [the panic report](reports/ane-panic-20260914.md).
- Keep AC/battery, low-power setting, pmset mode, and nominal/Fair thermals
  separate. Fair results are explicitly exploratory. Never normalize GPU/ANE
  time with CPU cycles. CPU cycles/instructions are meaningful only for
  separately measured host phases. Do not alter OS power settings.
- Preserve failed attempts, input pins, work counts, predeclared drift gates
  and raw measurements. No automatic retries or selection of favorable runs.
  Strict timing must not silently compile missing ANE programs.
- Never run two experiment workers against different locks. Do not kill an
  accelerator child to enforce a timeout; record overrun and join it safely.
  Do not stop unrelated inference servers, compilers or applications.

## Where the implementation lives

- `crates/rvllm-apple-metal`: Metal model, prefill kernels, encoder boundaries,
  BF16 matrices, QKV and tiling candidates. Metal candidates remain explicit.
- `crates/rvllm-apple`: safe single-I/O wrappers, attention surface packing,
  static INT8 FFNs, projection candidates, MIL/weight layouts.
- `crates/rvllm-apple-ane-sys`: isolated platform FFI, cache lifetime,
  source staging and durable driver diagnostics.
- `crates/rvllm-runtime/src/gemma_ane_decode.rs`: model preparation, imports,
  serial decode and explicit experimental weight plans.
- `ane_two_token_reference.rs`, `ane_two_token_layer_major.rs`,
  `ane_two_token_reference_live_tests.rs`: transactional serial and
  layer-major references. No assistant model has been downloaded or integrated.
- `ane_kv_import_scratch_tests.rs`: two ignored qualification fixtures.
  The CPU fixture makes zero device calls; the device fixture requires a
  journal and existing cached programs. Do not run either implicitly.
- `gemma_disaggregated_worker.rs`, `crates/rvllm-serve/src/gemma_disaggregated.rs`:
  persistent runtime ownership and HTTP integration.
- `crates/rvllm-runtime/src/bin/rvllm_experiment_queue/queue.rs`: standalone
  Rust queue. [Queue specification](specs/apple-experiment-queue.md).
- `crates/rvllm-runtime/src/bin/rvllm_ane_int8_probe/interleaved_timing.rs`:
  predeclared short-slot S1/S2 timing and independent drift checks.
- `reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/`:
  standalone safe Rust baseline runner and offline analyzer, each with its
  own locked Cargo workspace. These are intentionally outside the product build.

This checkpoint also preserves the pre-existing uncommitted Apple shipping
boundary, model-package, prompt-cache, scheduler, Swift host and CI work in
the same checkout. It does not assert fresh end-to-end acceptance of all those
surfaces. Public shipping builds must keep the private ANE feature disabled;
the Apple release scripts scan that boundary independently.

## Stopped queue and local artifacts

All paths below are relative to `v3/reports/gemma4-12b-evidence-20260914/`,
under the original checkout `/Users/george/Downloads/rvllm`.

- Active campaign directory: `experiment-queue-20260916/baseline-isolated-v7`.
  Its **STOP marker is present**. Last worker PID 30722 was joined with exit 0;
  `state.json` records `stopped`. There was no active accelerator child.
- Last runnable worker is **worker v6**, distinct from **queue directory v7**:
  `experiment-queue-20260916/rvllm_experiment_queue-release-v6`, SHA-256
  `bc02bf13c917874a066cd0de66b17fd81461ae612c12f07b355817eed39aa210`.
- A normal release executable built locally from the 2026-09-21 continuation
  had SHA-256
  `9c11fed2f41b6eb950eeaabf51909174fc0a64b961f918aef262c296bae7a490`.
  It passed only the fresh STOP-queue smoke; it was not copied into the
  campaign, frozen as a worker, or exercised against live launch conditions.
- Shared lock: `experiment-queue-20260916/hardware.lock`. The worker owns it
  for its whole lifetime, including while waiting.
- 36 manifests remain: seven completed and 29 pending. Completed IDs are
  04, 72, 73, 77, 78, 79 and 80. Job 04 succeeded operationally but failed
  the backend's repeat-drift gate. Queue success alone is not speed acceptance.
- Pending native-kit ABBA chains: 10–14 nominal AC, 20–24 nominal battery,
  30–34 Fair battery, 40–44 Fair AC. Each requires 120 seconds of quiet.
- Pending S2: 05/06 original longer pilots, 07/08 interleaved pilots,
  70/71 nominal post-comparison, 74/75 Fair post-comparison.
- Job 81 is still never-started in the queue. A direct host-only checkpoint
  check uses its same test filter; that is not a fabricated queue result.
  No jobs 82/83 or KV timing jobs were submitted.
- The 13 shared-probe queue tests are in
  `kv-import-scratch-20260916/queue-v7-tests.{stdout,stderr}`. Raw observations
  are shared only within one pass; each job independently applies policy.
  Age includes probe time and expires at 2.5 seconds. No gate was relaxed.

The archive contains many historical stopped/failed queues. Do not resume all
of them. Old PIDs and absolute paths are evidence, not current capabilities.
The queue's idle-server exemption (PID 1038 / port 8093 when recorded) must be
revalidated after a pause or reboot. Pending manifests may also have stale
source pins or deadlines. Never edit a completed or attempted manifest.

Git stores source snapshots, manifests, receipts, driver journals and protocols.
Frozen executables, captured tensors, build caches and downloaded upstream
copies remain **local**. [Archive selection](reports/checkpoint-20260916/archive-selection.json)
and [SHA-256 artifact inventory](reports/checkpoint-20260916/local-artifacts.jsonl)
identify exactly what was retained. A fresh clone cannot run those local jobs
without restoring the matching assets or creating a newly pinned campaign.
The ignored files have not been deleted. Models remain in the local Hugging
Face cache and are not uploaded to GitHub.

The last cache operation before this pause was a read-only inspection of
old compiler outputs: 1,908 eligible files / 394,507,904 logical bytes.
**That inspection was not applied.** Disk availability was near the queue's
16 GiB launch floor; remeasure before resuming rather than lowering the floor.

## Model and executable identity

Machine: M4 Max, macOS 15.6 (24G84), ANE 8.600.2. Historical receipts bind
their own boot and power observations; re-establish them on the next run.

| Artifact | Identity |
| --- | --- |
| Google standard model | `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Standard config SHA-256 | `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9` |
| Native-platform llama binding | `a3cf95eb1d4fa748480eb780e6fcbfc1a5c1c391` |
| Embedded llama.cpp | `5f55650a78f92aff4d48d671423e888fac0469ff` |
| Official Google QAT GGUF | revision `29d097773436b69ff9feafd636ab4cf873786537`, file `gemma-4-12b-it-qat-q4_0.gguf` |
| GGUF SHA-256 | `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` |
| Frozen disaggregated baseline | `int8-stacked-full-model-20260916/rvllm_disaggregated_infer`, SHA `aa015ca92c735e36c1da17d258031a4c743823a4b29feacb5ba46cbbc9f3c605` |
| Frozen native runner | `llama-native-kit-baseline-20260916/rvllm-native-kit-baseline`, SHA `91fcfde541135113652c62dd696fd8a560071d505e50784470901b10a9cff8fe` |
| Frozen interleaved S2 probe | `int8-s2-interleaved-20260916/rvllm_ane_int8_probe-interleaved`, SHA `cf1bb39ec914a3749badd9276479ad96f1dd86be9bef45d8784446fc6a11d63e` |
| KV reference report | `int8-stacked-full-model-20260916/checked/case-1/report.json`, SHA `0b096c565cc841ee625081d23b1752af0590c25c8f48554451e97e1fb7140ee8` |

The native comparison is a practical comparison of different precision
representations, not an equal-weight quantization experiment. The requested
`delysis/native-kit` path was absent; the actual native-platform binding is
pinned above. Work counts are 84 prompt tokens, 10 outputs, nine continuation
evaluations, two warmups and seven measured requests, with 1024 retained KV.

## Resume sequence — only after an explicit request

1. Read this document and inspect `git status`, the current commit, STOP,
   processes, disk, boot and model availability. Preserve existing evidence.
2. Build and freeze a campaign worker from the integrated queue source with
   the reused Cargo target. Its 24 macOS host tests, eight portable helper
   tests, and STOP-only executable smoke have passed; the ordinary build above
   is not a frozen campaign artifact. Run a bounded non-device scheduling
   check and verify shared observations let ready jobs make progress without
   weakening the 2.5-second freshness rule.
3. Audit pending pins, idle-server identity and previous results. Preserve
   never-started superseded manifests before replacing them with new IDs;
   retain failed/incomplete attempts unchanged. Use one shared hardware lock.
4. Finish a matched native-kit ABBA stratum and run its offline analyzer.
   Report the 5% drift verdict, raw controls and all repetitions. This is the
   outstanding answer to the user's baseline question.
5. Let interleaved S2 jobs run when their predeclared conditions hold. Keep
   original pilot failures. The broader source-7 CPU oracle discrepancy remains
   unresolved even though S1/S2 match each other; do not waive its tolerance.
6. For KV scratch, freeze the newly built test executable and qualify all
   358,026,240 packed bytes from the captured snapshot with zero device calls.
   Then explicitly stage the cached device fixture: 2,496 evaluations,
   208 requests and 162 unloads, with exact continuation at 84 and 21 tokens.
   Only afterward measure CPU packing and complete handoff. No default change
   is justified by saved allocations alone.
7. Integrate a qualified S2 projection into the verified layer-major boundary
   only if measured cost warrants it. Before assistant work, use
   `T2 + draft_cost < (1 + acceptance_probability) * T1`. Read the exact
   Google hidden-state/shared-KV contract; a high acceptance rate alone is
   insufficient. No W8A8 activation experiment has been implemented or staged.

Typical host-only commands from `v3` (explicitly **do not add `--ignored`**):

```sh
cargo test --offline --locked --release -j 2 -p rvllm-runtime \
  --features macos-private-ane-research --bin rvllm_experiment_queue
cargo test --offline --locked --release -j 2 -p rvllm-apple -p rvllm-runtime \
  --features macos-private-ane-research --lib kv_import_scratch -- --nocapture
python3 -m unittest tools/test_check_apple_release_symbols.py
```

Final checkpoint validation and exact command outcomes are recorded in
[`reports/checkpoint-20260916/validation.md`](reports/checkpoint-20260916/validation.md).
The consolidated Apple/runtime host suite passed 303 tests, with all 131
device/captured-data fixtures ignored. The separate queue suite passed 13 tests
and the release-symbol checker unit suite passed eight. Linux-only assertions,
the complete workspace and shipping builds still require CI verification.
The checkpoint also repairs old host-test boundaries: private ANE compilation
and actual toy-Metal execution are explicitly ignored hardware fixtures, while
the toy-route opt-in policy remains tested without device preparation.
Historical live hardware receipts are retained; there was no new accelerator
trial during this pause checkpoint. CI and local host tests must not be
presented as new model quality, speed or shipping acceptance.

## Reading order

1. [Current full progress report](reports/gemma4-12b-metal-ane-progress-20260914.md)
   and [queue/comparison history](reports/gemma4-experiment-queue-20260916.md).
2. [Layer-major live result](reports/gemma4-two-token-layer-major-20260916.md),
   [interleaved S2 protocol](reports/gemma4-int8-s2-interleaved-protocol-20260916.md),
   [S2 staging](reports/gemma4-int8-s2-interleaved-20260916.md),
   [KV scratch candidate](reports/gemma4-kv-import-scratch-plan-20260916.md).
3. [Megakernel/gigakernel research](reports/megakernel-gigakernel-research-20260914.md),
   [ANE bandwidth experiments](reports/gemma4-ane-bandwidth-next-experiment-20260914.md),
   [compression controls](reports/gemma4-ane-runtime-compression-controls-20260914.md),
   [assistant state contract](reports/gemma4-assistant-state-contract-20260916.md).

The research agent returned its reports earlier and is not running a hardware
trial. Further delegation is not required to resume the Rust queue. Continue
source review and coding independent variants while conditions gate trials;
keep inputs pinned until their queued checks finish.

## 2026-09-25 native-BF16 low-bit schedule frontier

The default-off Metal N4 and N8 W4/W8 schedules have completed real Gemma
seven-role screen and independent-confirmation campaigns at M=1 and M=4. Both
retain BF16 activations/output, FP16 group-32 scales, FP32 accumulation, exact
dispatch accounting, guards and bitwise repeat checks. Shipping defaults are
unchanged.

- N4 has eight stable cells out of 28 under the repeated >=1.05x speedup and
  <=20% candidate/native/speedup drift policy.
- N8 has four stable cells out of 28. Its wins are K/W8/M4, O/W4/M1,
  O/W4/M4 and Down/W8/M4.
- The union is 11/28 because Down/W8/M4 overlaps. N8 therefore adds three
  stable cells.
- These are partial operator wins, not a campaign-wide or full-model winner.
  N4 and N8 were each compared with native BF16, not directly against each
  other. A production selector requires a counterbalanced N4-versus-N8 referee
  on plausible cells plus checkpoint-bound logit/perplexity acceptance.

Evidence is in `reports/gemma4-metal-low-bit-n4-20260925/` and
`reports/gemma4-metal-low-bit-n8-20260925/`. The latter's `summary.json` is the
compact current adjudication; its `queue-receipts/` directory preserves all
fourteen unaltered job receipts.

The generated-code evidence gap is now narrowed by
`reports/gemma4-metal-artifact-evidence-attention-20260925/`. Its strict
receipt seals the exact MSL, AIR, metallib, toolchain, compiler commands,
public `metal-objdump` output and live M4 Max pipeline properties. Split-32
uses 2,912 B plus 384 B static threadgroup memory for partial plus merge;
split-matrix uses 12,512 B plus 0 B. All four report execution width 32 and a
1,024-thread pipeline maximum. Apple public APIs still do not expose supported
register-count, occupancy, residency or machine-lowering evidence, so those
claims remain explicitly unavailable or unverified.
