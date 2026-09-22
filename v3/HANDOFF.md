# Gemma 4 Metal / ANE checkpoint — paused 2026-09-16

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

On 2026-09-22, a broad kernel-candidate delegation was staged for ordinary
ChatGPT Pro through the local Chat Pro Bridge. The first job
`rvllm-gemma4-kernels-d4f5c324-20260922-01` stopped before Send because an old
tab could not identify the model selector. After the user reloaded the current
extension, `rvllm-gemma4-kernels-d4f5c324-20260922-02` positively verified
ordinary Chat and `6 Pro` and reached the Send checkpoint, but its marked user
turn was not visible during submission or one status inspection. The user found
the complete draft in the composer and sent it once manually. A subsequent
bridge inspection bound the marked user turn and reported `running` at
`https://chatgpt.com/c/6ab28a98-8104-83ea-b0d5-f301d3bfc73c`. Do not resubmit
it or use Work mode; collect the result through the same durable ID. See the
[delegation record](reports/chat-pro-kernel-delegation-20260922.md).

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
