# Gemma 4 delivery handoff

Date: 2026-09-22
Repository: `/Users/george/Downloads/rvllm`
Base reviewed: `b394d8ea5af412224d9cc28b20a16d72d9e9d2fe`

## Scope

The supplied packet `rvllm-gemma4-delivery-fix-b394d8ea-20260922` is a narrow
repair to the candidate delivery gate. It changes only the packet-scoped
formatting manifest, the delivery shell gate, and its contract tests. It does
not authorize new kernels, inference, ANE cache work, accelerator trials,
timing, promotion, or queue changes.

The current local change also normalizes the temporary-directory path in one
contract assertion so the test is valid on macOS `/private/var` aliases. This
does not change gate behavior.

## Work for ChatGPT 6 Astra in the web app

This handoff is the routing point for the remaining major work. ChatGPT 6
Astra should take the next substantive implementation/review pass from the
committed tree at `faa75336` (or a descendant containing this handoff), after
reading both this file and `v3/HANDOFF.md`. Astra may return reviewable source
patches, tests, and analysis, but must not claim native execution, device
qualification, timing, cache provisioning, or promotion. The local owner
retains those operations.

Do substantive implementation and review work only. Inspect the current tree
and this handoff before proposing changes. Any serious implementation change
must be documented here with affected files, invariants, and a reviewable
patch. Preserve the existing Gemma 4 handoff constraints in `v3/HANDOFF.md`:

- no multi-I/O ANE route or panic-quarantine bypass;
- no worker restart, STOP-marker clearing, ignored accelerator fixture, cache
  recovery, inference, timing, or promotion unless the user explicitly
  resumes and requests that phase;
- keep prefill-only functional triage separate from numerical/performance
  acceptance;
- preserve failed outputs and provenance; never silently retry or select
  favorable evidence;
- keep platform FFI isolated and safe Rust outside existing FFI boundaries.

For this packet, review the three gate files and identify any correctness,
portability, or scope issues. Do not run the real build gate or alter Git
history; the local owner performs those operations.

## Next Astra work after this piece

The gate-consistency piece is complete in `faa75336` and is intentionally
limited to the packet-owned formatting manifest, delivery shell gate,
contract tests, and this routing record. The next major work belongs to
ChatGPT 6 Astra:

1. Review the committed gate patch and its 22-test contract evidence for
   portability, race windows, and scope fidelity.
2. Propose or implement only reviewable follow-up patches, documenting
   affected files, invariants, and tests here before handoff.
3. Leave the native delivery gate, prefill-only screens, cache recovery,
   full-route qualification, tensor oracles, matched timing, and promotion to
   the local owner in the order specified by `v3/HANDOFF.md`.

Do not add kernels or broaden acceptance criteria in this continuation.

## External wide-research packet audit

The attached `rvllm-gemma4-wide-research-226dbaad-20260922` packet is not
empty and must not be dismissed as reference-only. Its self-contained host
qualification passes 35 Python tests locally, covering CPU model properties,
proposal mutation/staging guards, stage isolation, and source contracts. Its
catalog validator reports seven candidates as
`staged-only-not-authorized-not-queued`, and its compile-plan renderer emits
host commands plus candidate-specific Rust tests and Metal compile arms.

That evidence is CI-relevant but not promotion evidence. The packet's own
static receipt says 23 Rust test functions were added but not run, native Rust
tools were unavailable in its construction environment, no Git history was
created, and the native compile plan was not executed. The packet contains no
workflow definition. Its candidate source files are absent from this
checkout, so they must not be elevated to `main` by copying them wholesale.

The next Astra/local-owner decision is therefore to extract a reviewable,
host-only CI job for the packet's Python qualification and catalog checks, and
to review the seven proposals one at a time against the existing handoff. Any
Rust/Metal proposal requires the local delivery gate and the qualification
sequence in `v3/HANDOFF.md`; no packet receipt alone authorizes merge,
accelerator execution, timing, or promotion.

## Local owner responsibilities

The local Codex owner handles all building, testing, native Rust/Metal
validation, evidence review, output-directory management, commits, and pushes.
The real gate must use a fresh absolute output directory and the existing Cargo
target cache. It must remain offline/locked and must not invoke inference.

Current contract result after the macOS path-normalization fix: 17 tests pass,
including the real-rustfmt semantics test. The supplied container report is not
native build evidence; it explicitly recorded no Cargo, Metal, inference,
cache, or hardware execution.

## Acceptance boundary

A successful delivery gate means only packet-owned formatting checks, targeted
host tests, private research CLI/source-exporter builds, and eight Metal 3.1
compile/link arms completed. Its status remains
`compiled-only; no accelerator acceptance`. Prefill, cache, full-route,
tensor-oracle, timing, and promotion remain separate phases governed by
`v3/HANDOFF.md`.

## Gate consistency review proposal (base 226dbaad)

Reviewed at `226dbaad8dec9bdd3b9684a224747384b6e47319`. The local owner
reports that this revision passed the real native delivery gate. The review
found two reproducible false-pass windows under concurrent edits/rebuilds;
there is no evidence that either occurred in that completed native run.

Affected files: `v3/tools/check_gemma4_candidate_delivery.sh`,
`v3/tools/test_gemma4_candidate_delivery.py`, and this handoff. The reviewed
20-path formatting manifest is unchanged. No Rust or shader changes.

The gate now hashes each executable immediately after its build, checks the
exporter against that hash before every export, and reuses the early binary
hashes in the final artifact manifest. A later replacement fails instead of
receiving a fresh identity. A final packet-source/manifest check extends the
existing formatting-stage consistency check through tests/builds/exports.

Invariants: retain native-host refusal, offline/locked Cargo, five targeted
test filters including `ane_attention_layout::blocked32_tests`, eight Metal
3.1 compile/link arms, fresh output directories, preserved failures and no
inference. Five new deterministic fake-tool regression tests are integrated
into the existing contract suite. The native gate is not run by the reviewer.

This is bounded change detection, not atomic attestation: no shared-target
lock is acquired, transient edits restored between checks are not detected,
and unlisted sources/configuration/toolchain binaries are not pinned. Keep a
quiescent checkout and target while running the local gate. The binaries in
Cargo's target directory remain mutable; verify hashes before later use.

Apply the supplied single patch only after local review; do not reapply older
packets or rewrite completed receipts. Run the contract suite and real gate
locally in a fresh output directory. Then continue the existing prefill-only
qualification order in `v3/HANDOFF.md`; no new kernel or acceptance criterion
is introduced by this review.

## 2026-09-22 gate-review continuation

The consistency-review patch was applied on top of `226dbaad`. Its 22-test
contract suite passed in bounded batches (22/22), and the native delivery gate
passed with `compiled-only; no accelerator acceptance`; the fresh receipt is
under `/tmp/rvllm-gemma4-review-gate.WkiF9i/output`.

A host-only queue job was submitted as
`gate-review-226dbaad-delivery-20260922-1` in
`experiment-queue-20260922-gate-review-2`. It is not an accelerator
experiment. The queue run was deferred/stopped because free disk was 12.8 GiB,
below the declared 16 GiB floor. The direct gate passed; rerun the queued job
only after headroom is restored, without weakening its condition.

### Promotion blockers / to-dos

- [ ] Restore and recheck at least 16 GiB free disk before queued or live work.
- [ ] Rerun the queued host-only delivery job and retain its receipt.
- [ ] Re-provision the short-MMA candidate's ANE cache in a fresh serial
  process; its 6-token full-route attempt reached Metal prefill but failed
  before decode because the current client cache lacked the compiled model.
- [ ] Run short-MMA full continuation after cache recovery and require exact
  continuation equality plus candidate dispatch evidence.
- [ ] Run positive GQA prefill with a separately pinned >=64-token reference;
  the 6-token screen remains negative evidence only.
- [ ] Run candidate tensor/original-driver oracle checks. Token equality alone
  is not tensor acceptance.
- [ ] Run the predeclared matched ABBA timing/control campaign with two
  warmups and seven measured requests per arm, zero compiler calls, stable
  strata, and the 5% drift gate. Do not pool AC/Fair/battery results.
- [ ] Promote only after compiler, dispatch, tensor, full-continuation, and
  matched-timing evidence all pass; otherwise defer with the failed receipt.

## Wide-series source integration and CI (Astra proposal, 2026-09-22)

**Integration base: `faa753360e30ce59540d1d113994c711a1d19675`.** This supersedes
only the earlier packet-delivery plan, not the experiment pause, hardware-owner
rules or the original numerical gates. The reviewer could not fetch the locally
reported `15f1d378` / `fdc6b4ca` documentation updates. Preserve those updates;
merge this appendix with them rather than resetting or replacing this file.

### This is code to integrate and test, not another research-only handoff

The eleven-patch integration series installs the prior six new runtime
candidates, the blocked packed-I/O source prototype, their tests and seven inert
proposals, a baseline-preservation fix, CI, and the expanded native delivery
gate. Do not reapply the older eight-patch wide packet as well. Every modified
existing file has a complete preimage checked against the visible native base;
the earlier seven hunk-only Rust contexts were replaced with exact blob matches.
A full workspace/native application check remains the local owner's task.

Affected areas: Metal `research{,_next,_evidence}.rs`, `layer_forward.rs` and three
new shaders; ANE INT8 source/oracle modules and attention layout/wrappers;
`gemma_ane_decode.rs`, `gemma_head_ranking.rs`, disaggregated CLI options; the
scoped delivery gate; installed Python/JSON qualification tools; the seven staged
proposal files; `.github/workflows/gemma4-candidate-host.yml`; and this handoff.
Detailed inventory and evidence boundaries are in
`v3/reports/gemma4-wide-integration-faa75336.md` and the packet manifest.

Runtime candidates: `metal-mma32-prefetch`, `metal-attn-q4`, `metal-rms-simd32`,
`ane-int8-ffn-down4`, `ane-attention-transpose-flags`, `cpu-head-softcap-prune`.
`ane-int8-ffn-packed32` remains source/codec-only: no selector, cache part, driver
consumer or runnable job. The original six candidates retain their names and
fallbacks. Existing GQA Metal 3.1 fixes, private/shipping gates, single-I/O
quarantine, failed evidence, numerical tolerances and ignored fixtures survive.

One integration correction matters: the original proposed head-ranking setter
unconditionally constructed a softcap-30 candidate even with both experiment
and timing off. It now preserves ordinary baseline checkpoint softcaps and
checks candidate/control eligibility for direct library callers as well as the
CLI. Failed configuration is checked before mutating the owner.

### Immediate local-owner task: integrate source, execute host/compiler gates

1. Inspect the real branch/status and preserve local/unseen handoff edits. Check
   and apply the complete integration series; do not copy partial source trees
   over the checkout. Resolve only actual conflicts; no reset of local work.
2. Format/check only the 24 paths in `v3/tools/gemma4_candidate_rustfmt.paths`.
   The reviewer did not execute rustfmt. Use the existing stdin/nonrecursive
   method, not `cargo fmt --all` or recursive formatting of unrelated modules.
3. Run `python3 tools/run_gemma4_python_checks.py --require-rustfmt` from `v3`
   with the pinned NumPy test dependency. The new workflow runs these same tests
   on Ubuntu and then runs 45 named public Rust tests and fourteen source
   exports. A positive total without the expected test names is rejected.
4. Run the single expanded native `check_gemma4_candidate_delivery.sh` with a
   fresh absolute output directory and the existing Cargo target cache. Its
   original eight plus six new Metal 3.1 compile/link arms total fourteen. It
   also checks all 24 Rust files, six shader-source inputs and the new native
   host filters. Preserve failed logs; fix packet-owned compile/format issues
   rather than weakening a gate. No inference runs in this gate.
5. Commit/push the actual reviewed source, tests/workflow and updated handoff on
   the research branch under the user's local-owner authority. Return the exact
   head, CI URLs/results, native compiler/test counts and any remaining error.
   Do not complete this phase by only committing another routing/audit document.

The pipeline distinguishes source integration, public-host CI, native compiler
acceptance and candidate promotion. No result in one category proves another.
The source proposals are potential runtime implementations, not accepted speed
improvements. All default selections remain unchanged.

### Later work remains gated and locally owned

No inference, cache recovery, timing, accelerator fixture, STOP clearing or
active/attempted queue mutation was performed or authorized by this source
packet. Follow the existing phase permissions and `v3/HANDOFF.md` before any
such operation. Do not create jobs from the null-pin worksheet.

Clarification to the historical checklist above: short-MMA is a Metal candidate;
its missing ANE cache is the unchanged baseline decode cache, not a new
short-MMA-specific ANE program set. Down4 needs 48 replacement FFNs; transpose
attention needs two replacement shared programs. Host pruning needs no new ANE
program. Packed32 must stay blocked pending compiled I/O descriptors and a new
boundary review. No compressed/resident allocation saving is established.

Keep actual dispatch separate from requested selection. Old layer tracing can
force the prefetch/RMS fallback; that fallback is not their tensor oracle. GQA
still requires at least 64 prompt tokens. The original complete continuation,
component/tensor checks, workload pins, 16 GiB disk floor, one-owner policy,
power/thermal strata and predeclared 5% drift gate remain required. Major new
kernel/oracle/adapter design returns to Astra with the exact source and failure
receipt; routine integration and native validation belong to the local owner.

Reviewer evidence for this packet: 63 Python tests collected, 62 passed, one
real-rustfmt semantics test skipped because rustfmt was unavailable. These are
CPU-model, source and fake-tool orchestration checks, not Rust/Metal execution.
Rust compilation/tests, real formatting, private MIL acceptance, native delivery,
CI execution and all device/performance/promotion gates remain unrun here.

### Local-owner integration receipt (2026-09-22)

The eleven-patch replacement packet was checksum-verified (`157` hashed files)
and replay-verified in isolation (`11` patches, `0` commits, `0` source
executed, `46` final changed files, `18` restored preimages). It was applied
sequentially on `faa753360e30ce59540d1d113994c711a1d19675`; the local docs
commits `15f1d378` and `fdc6b4ca` were preserved. A host-inventory typo exposed
by the first runner was corrected: the three ANE next tests are actually under
the `next::` module, not `next_tests::`.

Concrete receipts:

- `python3 tools/run_gemma4_python_checks.py --require-rustfmt`: **63/63**
  passed with real rustfmt enforced.
- Host runner receipt:
  `/tmp/rvllm-gemma4-host-ci-rerun.9yUkF5/receipts/result.json`; **45 named
  Rust tests and 14 source exports** passed. This is host/source evidence only.
- Native delivery gate receipt:
  `/tmp/rvllm-gemma4-native-gate-rerun2.AKD8El/output`; **passed** with
  `compiled-only; no accelerator acceptance`, including 24 Rust paths, six
  shader inputs, and fourteen Metal 3.1 compile/link arms.

The host runner and native gate do not qualify inference, cache recovery,
accelerator behavior, timing, or promotion. Those remain the next Astra-owned
engineering phase after restoring the 16 GiB free-disk floor. The first host
receipt failed only because of the inventory spelling mismatch; its executed
28 tests all passed and the corrected rerun is the authoritative receipt.

### Handoff for ChatGPT 6 Pro / Astra

Use the resulting pushed commit below as the exact source head. Review the
source, receipts, and blockers before proposing kernel changes. The single next
major task is to design and execute the evidence-gated Gemma 4 promotion phase:
recover the unchanged ANE decode cache, run the short-MMA and positive GQA
full-continuation/oracle checks, then run the predeclared ABBA timing campaign
with dispatch evidence. Do not promote from host/compiler receipts alone.

Integration commit: **e52cbb549967e17e3d7f266b7d1f9d2f1efdf800**.

## Unified cffb22da continuation integration (2026-09-22)

Packet verification passed: 459 checksums, 33 exact preimages, and isolated
five-patch replay with no source execution or commits. All five mboxes applied
in order to the cffb22da base. Exactly 32 listed Rust paths were formatted.

Qualification receipts: Python checks **108/108** passed; the unified proposal
validator reported five source-only, unadmitted candidates with zero jobs and
zero hardware execution; the public host runner passed **60 named Rust tests
and 22 source exports** at `/tmp/rvllm-gemma4-unified-host.IRliEk/output`.

The native gate reached all 32 formatting checks and the catalog/evidence host
filters, then failed at `prefill-screen` compilation because rustc exhausted
the filesystem while writing artifacts (`No space left on device`). Receipt:
`/private/tmp/rvllm-gemma4-unified-native.lDh1vJ/output`. Restore disk space
and rerun the native gate in a fresh output directory. No accelerator,
inference, cache, ignored fixture, queue, power, or timing work ran.

The disk-headroom rerun exposed and fixed one genuine Metal 3.1 issue in
`mma32_load4.metal`: BF16 source rewriting rejected `vec<bfloat,4>(0.0f)`.
Both guarded vector initializers now construct the scalar element explicitly.
The fresh rerun passed with `compiled-only; no accelerator acceptance` at
`/tmp/rvllm-gemma4-unified-native-rerun2.Y71g3q/output`.

## Wave 2 packet intake and deferral (2026-09-22)

Wave 2 archive `rvllm-gemma4-wave2-faa75336-20260922.tar` was checksum
verified and its complete eight-patch series replayed cleanly against the
declared `faa75336` base. The packet is explicitly a proposal backlog: it
contains no commits, no live queue mutation, no accelerator/native execution,
and all six selectors are default-off with null pins.

The non-conflicting proposal/source artifacts were staged locally: four Metal
shader candidates, the ANE wave2 source/codec module, proposal JSON/templates,
and the fail-closed whole-tensor audit tool. The packet's tensor-audit tests
pass **13/13**, and the existing integrated Python qualification passes
**63/63** after correcting the already-known `next::` inventory spelling.
The packet's isolated Linux/source checks were not treated as native evidence.

### Deferred blockers / Astra to-dos

- [ ] Manually merge patch 8's runtime wiring into the current Wave 1 head.
  Direct application conflicts in `handoff.md`, Metal routing/evidence and
  runtime decode/prefill files because Wave 1 changed the same regions. Do not
  replace those files wholesale with the packet's `faa75336` preimages.
- [ ] Add and review the missing `research_wave2` and ANE wave2 test-module
  wiring only after the merge is reconciled; current staged artifacts are not
  an admitted runnable route.
- [ ] Extend the reviewed host inventory and delivery gate for the four new
  Metal selectors and wave2 test filters, then run real rustfmt, host Rust
  tests, and the additional Metal 3.1 compile/link matrix.
- [ ] Keep all six experiment manifests unadmitted until actual model,
  executable, library, oracle, and cache pins are populated from local assets.
- [ ] After compiler/source qualification, perform the separately gated
  tensor-oracle, full-continuation, cache, and timing work. No inference,
  cache recovery, accelerator fixture, queue admission, or timing was run for
  Wave 2.


## Unified cffb22da integration — source packet, not promotion

This appendix supersedes the Wave2 runtime-wiring deferral, not the original
numerical, ownership, power or safety constraints. Exact base:
`cffb22dabcb57b5f3ecee05acd4ab810d1ce6077`. Preserve all history above.

The completed five-patch packet repairs the orphan module declarations and
reconciles both waves through one typed catalog/checked projection plan. It
preserves ten dispatch slots and appends seven, wires four new Metal selectors
plus cached-only interleaved INT8 FFN, and deletes only the duplicated staged
Down4 implementation. Existing Down4 and packed32's blocked boundary survive.

Affected areas: Metal catalog/policy/projection/evidence, pipeline/layer-forward,
three shader ABI/bounds guards and actual-source matrix fixtures; ANE interleaved
MIL, shared source oracles, cached decoder/CLI and safe ignored real-input FFN
fixtures; offline tensor audit; host CI/catalog/native compiler inventories;
five inert proposals and local qualification recipes. Detailed report:
`v3/reports/gemma4-unified-cffb22da.md`. Local sequence:
`v3/specs/gemma4-unified-local-qualification.md`.

The historical cff Actions failure is not ambiguous: 63 Python checks passed,
then Rust compilation failed E0583 (missing research_wave2). It did not execute
its Rust inventory or source matrix. The earlier next/next_tests spelling repair
was already in cff and is preserved. Current researcher execution: 108 Python
checks collected, 107 passed, one skipped for absent rustfmt; nine independent
packet-verifier mutation tests passed. These are CPU/tool/source checks, not
Rust or accelerator execution. The prior unfinished 115-test claim is superseded
by the actual delivered inventory and its disjoint batch receipts.

Apply the COMPLETE new series against cff after read-only preimage checks;
do not reapply earlier packets or copy whole source snapshots over local work.
Format only the 32 reviewed paths. Expected local gate scope: 60 named public
Rust tests, 22 MSL exports, twelve native host-test filters and 22 Metal 3.1
compile/link arms. No local compilation or formatting is preclaimed by Astra.
Commit/push reviewed source, tests and this handoff under local-owner authority,
then inspect CI against that exact resulting head. Do not close integration
with only another routing document.

Prefill-only is bounded to 1–16 references and requires complete entry-point
families in v3 receipts; this is not full layer/tensor acceptance. Matrix adapters
use actual runtime shaders. Three ignored real-input ANE fixtures use existing
cached programs only, proper controls and at most six evaluations. A separate
host-only pin preparer uses the existing quantizer and never initializes ANE.
All remain unrun here. RMS still lacks a direct isolated native oracle and its
experiment stays blocked. GQA/temporal/full-continuation/cache/driver/timing
qualification remains required locally before performance promotion.

No production default, feature gate, numerical tolerance, multi-I/O quarantine,
ignored-device marker, STOP/attempted manifest, or power setting was weakened.
The researcher made no commits/pushes and ran no device/cache/timing operation.

## Controlled local hardware qualification transition (2026-09-23)

The user explicitly resumed execution ownership from
`5043d03be1841b973db1c2be2da14a53e38eb46b`. The local owner is authorized to
apply/build/run the committed fixtures and measure evidence; Astra remains
responsible only for missing kernel/oracle implementation. All numerical,
provenance, safety, ownership, single-I/O, packed32, STOP-marker and tolerance
rules remain in force.

The phase is currently blocked before device work: read-only prerequisite checks
found **1.1 GiB free**, below the required 16 GiB floor. AC power and thermal
state were acceptable, the unrelated llama-server was left running, and the
historical hardware lock and STOP markers were preserved. No Metal fixture,
cache inspection/preparation, inference, FFN oracle, queue job, or timing arm
was started. Full details are recorded in
`v3/reports/gemma4-unified-hardware-qualification-20260923.md`.

### Hardware-phase blocker

- [ ] Restore at least 16 GiB free without deleting evidence or unrelated work.
- [ ] Recheck boot, owner/lock, process policy, AC/pmset/thermal state, model,
  executable/library identities and cache state in a fresh receipt.
- [ ] Then run the exact named Metal fixtures, baseline prefill/full-route
  checks, independent FFN oracle fixtures, and only qualified ABBA timing jobs.

## Controlled qualification execution update (2026-09-23)

The disk prerequisite was repaired without deleting databases, reports, models,
queue state, locks, STOP markers, or raw evidence: `cargo clean` removed only
the rebuildable `v3/target` artifacts (20.7 GiB). A fresh prerequisite check
now reports 18 GiB free, AC/battery state unchanged, no thermal or pmset
warnings, the unrelated llama-server still running, and the historical owner
lock/STOP state preserved. The rebuilt delivery gate passed with the required
`compiled-only; no accelerator acceptance` boundary. Current source is
`ac06b9962b0b8fadcee8d37eee6e599a77183185`, a descendant of the requested
`5043d03be1841b973db1c2be2da14a53e38eb46b`.

The first filter invocation omitted the harness prefix and ran zero tests; it
is rejected as evidence. The corrected exact paths used
`layer_forward::prefill_mma_tile_tests::<fixture>` and ran one ignored test at
a time. Receipts and logs are under
`/tmp/rvllm-gemma4-hardware-20260923/`:

- `native-short-tile.json`: passed, 6 cases/24 commands, all guards and
  relative-L2 gates passed.
- `native-prefetch.log`: failed on the existing finite-output assertion at
  `prefill_mma_tile_tests.rs:483`; no successful report was emitted. Candidate
  stopped; no blind retry.
- `native-fp32-operands.json`: passed, 6 cases/24 commands, all guards and
  relative-L2 gates passed.
- `native-vector-loads.json`: passed, 6 cases/24 commands, FP32 bit-parity
  required and passed, all guards and relative-L2 gates passed.
- `native-bf16-tile64.json`: passed, 6 cases/36 commands, both output ABIs and
  operand paths covered, all guards and relative-L2 gates passed.

The pinned baseline reference has 21 prompt tokens, so it is valid for the
bounded functional screen but not for the required >=64-token GQA/temporal/
long-tile positive screen. Baseline prefill and short-MMA prefill were each
run into fresh output directories with `--ane-compile-budget 0`; both matched
the expected first token, exercised complete prefill entry-point families,
and recorded zero ANE compiler calls and zero ANE decode steps. They remain
`correctness passed / performance unmeasured`: first-token-only receipts are
not full-route, tensor, cache, driver, or timing acceptance, and their power
stratum was battery with sampled-control eligibility false.

Current blockers and next actions:

- [ ] Obtain or identify an independently pinned >=64-token reference before
  GQA/temporal/long-tile positive screens; do not pad the 21-token reference.
- [ ] Keep prefetch deferred after its non-finite-output failure; preserve its
  exact log and source assertion.
- [ ] Inspect/provision the unchanged baseline cache only through the bounded
  fresh-process mechanism, then separately require 162 visits, zero inspection
  compiler calls, and lifecycle evidence before full continuation.
- [ ] Run the host FFN pin preparer and separately admitted cached-only FFN
  fixtures only after cache prerequisites pass; RMS remains blocked for lack of
  its direct native oracle, multi-I/O remains quarantined, and packed32 remains
  blocked.
- [ ] Do not create timing jobs until baseline and each candidate have complete
  correctness/full-route evidence; then use only the predeclared ABBA design.

Detailed execution summary: `v3/reports/gemma4-hardware-qualification-execution-20260923.md`.

## Prefetch fixture repair and corrected qualification (2026-09-23)

Applied the two-patch packet targeting `6a79af7c81900572ee6a6f39a46280fc1f35e865`
after checkout and isolated replay verification. The first replay attempt used
an already-created directory and was refused; a fresh nonexistent replay path
then passed with two patches, six changed paths, restored preimages, and zero
source execution. Both mboxes applied cleanly. Rustfmt was run only on the two
changed Rust files. A supplied Python regression had a whitespace-sensitive
source lookup; it was corrected to match the actual multiline Rust statement,
without changing the production fixture contract.

Host/native results:

- all 116 Python tests passed with `--require-rustfmt`;
- the public runner completed 66 named Rust tests and 22 source exports;
- the compiled-only native delivery gate passed with
  `compiled-only; no accelerator acceptance`;
- the original prefetch receipt remains failed and preserved.

The corrected exact device command was run once with the full
`layer_forward::prefill_mma_tile_tests::` path, qualification-only mode, the
pinned Gemma 4 snapshot, and a fresh report path. Receipt:
`/tmp/rvllm-gemma4-prefetch-qualified-20260923-1.json`. It passed one exact
ignored test with six cases and 24 dispatches: 3 numerical GEMM, 2 numerical
QKV, and 7 guard-refusal controls. Guards and canaries were intact, FP32 bit
parity passed, and the numerical relative-L2 gates passed. No timing or
promotion claim is made; the fixture is qualification-only.

This repairs the fixture defect; it does not change shader guards, runtime
selection, tolerances, defaults, or ignored markers. Baseline cache inspection/
recovery and complete short-reference continuation remain the next local tasks.
They are independent of the >=64-token long-screen requirement, but still
require their own cache, lifecycle, tensor, driver, and provenance receipts.
