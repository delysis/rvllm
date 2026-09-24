# Gemma 4 delivery handoff

## 2026-09-23 live candidate campaign

The current native campaign is summarized in
`v3/reports/gemma4-kernel-campaign-20260923.md`.  On the merged kernel-game
vertical slice (`360a00fc`), all ten Metal candidates passed their pinned
first-token and dispatch-family screens.  The exploratory ordering makes
`metal-mma32-load4` the first controlled-timing candidate, followed by
`metal-mma32-prefetch` and `metal-rounded-gate32`; no ABBA result or speed claim
exists yet.

For ANE, the baseline, chunk4, and down4 plans completed a zero-compile
two-token full route and matched the pinned reference.  Sliding-QKV (both
variants), interleaved FFN, transpose-attention, and stacked FFN were blocked by
absent compiled-cache programs and were not repaired or retried in place.  The
delivery gate and its ANE host contracts passed, but that is not a substitute
for the missing device runs.  Preserve the raw queue directories and follow the
next-experiment order in the campaign report.  Nothing was promoted.

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

## 2026-09-23 process review: stop manual qualification and design the game

This section is the next handoff for ChatGPT Pro 6 Astra. It supersedes the
immediate instruction to continue running candidates by hand. The user has
identified the central problem correctly: a person or interactive agent
selecting commands, waiting on the machine, transcribing results and deciding
what to try next is not an acceptable long-term optimization process. It is
slow, inconsistent, difficult to audit, and gives candidate authors too many
ways to benefit from accidental differences in controls. The next substantive
work is to specify the standard that defines the game, then design a durable
asynchronous local daemon around that standard. Do not respond with another
one-off shell sweep or a larger hand-written checklist.

The user asked only for a detailed, epistemically careful problem statement at
this point. No daemon or queue changes were made in this continuation. Astra
should first return a reviewable design/specification and an implementation
plan grounded in the current source. Remote Astra must not claim local device,
power, cache, timing, launchd, GitHub-write or promotion evidence.

### Exact source and local mutation boundary

- Reviewed source head: `df305b1c84f75cc8c7e3358b482461fe0253a843`
  on `codex/gemma4-kernel-candidates`, matching the remote branch at the start
  of this pass. The worktree was clean before this handoff edit.
- The existing queue implementation is
  `v3/crates/rvllm-runtime/src/bin/rvllm_experiment_queue/queue.rs`; its current
  contract is in `v3/specs/apple-experiment-queue.md`. Reuse its typed Rust,
  immutable-job, ownership and STOP semantics where they are sound. Do not
  build an unrelated second queue beside it without first explaining why the
  current queue cannot be evolved.
- Current candidate inventories are split across the Metal typed catalog,
  runtime ANE enums, proposal JSON under `v3/reports/proposals/`, host-test
  inventory, compiler gate and historical queue manifests. These disagree in
  freshness and status. There is no single authoritative cross-backend
  candidate registry today.
- The supervised llama-server on port 8093 was stopped at the user's explicit
  request by booting out launchd service
  `com.openai.codex.fiction.v15-critic`; it did not respawn afterward.
- At the user's explicit request, Hugging Face's cache manager removed exactly
  `model/ggml-org/gpt-oss-120b-GGUF`,
  `model/ggml-org/gemma-4-26B-A4B-it-GGUF`, and
  `model/unsloth/Qwen3.5-27B-GGUF`. It reported 121.0 GiB reclaimed. The pinned
  raw Gemma 4 12B snapshot and 12B QAT GGUF were preserved.

### What this pass actually established

These results are useful evidence about the candidates, but they are also an
example of why the process needs to be replaced. All receipts below are local
`/tmp` artifacts unless already tracked in the repository. They are not durable
GitHub evidence, are not signed, and do not by themselves authorize promotion.

Host and compiler gates at `df305b1c`:

- `python3 tools/run_gemma4_python_checks.py --require-rustfmt`: **116/116**
  passed in 40.082 seconds. NumPy emitted its expected warnings while iterating
  non-finite half patterns; the tests passed.
- Fresh public host runner:
  `/tmp/rvllm-gemma4-review-host-parent.XYyd78/output/result.json`: **66 named
  Rust tests and 22 source exports** passed. Its declared status is
  `host-tests-and-source-export-only`.
- Fresh native delivery gate:
  `/tmp/rvllm-gemma4-review-native-parent.T5Ov9b/output`: passed 32 scoped
  formatting paths, native host filters and **22 BF16/FP16 Metal 3.1
  compile/link arms**. Its status is exactly
  `compiled-only; no accelerator acceptance`.

Five current-head direct Metal matrix fixtures were each selected by one exact
ignored-test name and run in qualification-only mode against the pinned raw
Gemma 4 12B model. No timing samples were requested:

| candidate / fixture family | cases | commands | direct result | receipt |
| --- | ---: | ---: | --- | --- |
| `metal-short-mma16x64` | 6 | 24 | guards passed; maximum reported relative L2 0 | `/tmp/rvllm-review-df305b1c-short.json` |
| `metal-mma32-prefetch` | 6 | 24 | 3 GEMM and 2 QKV numerical arms passed; 7 refusal controls stayed byte-poisoned; maximum relative L2 0 | `/tmp/rvllm-review-df305b1c-prefetch.json` |
| `metal-mma32-f32` | 6 | 24 | guards and operand-path oracle passed; maximum relative L2 0 | `/tmp/rvllm-review-df305b1c-f32.json` |
| `metal-mma32-load4` | 6 | 24 | guards, FP32 bit parity and vector-load oracle passed; maximum relative L2 0 | `/tmp/rvllm-review-df305b1c-load4.json` |
| `metal-long-mma32x64` fixture | 6 | 36 | both output ABIs and operand paths passed; maximum relative L2 0 | `/tmp/rvllm-review-df305b1c-long.json` |

The repository already contained an independently pinned 84-token CPU
reference at
`v3/reports/gemma4-12b-evidence-20260914/long-context-cpu-oracle/copy96.json`.
It has model revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, 84 prompt tokens
and ten expected generated tokens. Therefore the preceding handoff's claim that
no >=64-token reference had been identified was stale. This is an important
process failure: the necessary asset existed in the tree, but the manual plan
did not derive its prerequisites from an indexed authoritative registry.

Using that unchanged 84-token reference, and the 21-token reference only for
short-MMA's declared <=64-token range, a fresh prefill-only screen passed for
the baseline and every one of the ten non-default Metal selectors. Every
candidate matched its expected first token and its receipt reported
`complete_family_exercised=true` with no foreign candidate slots:

| selector | positive encoded dispatches | single-screen Metal GPU ms |
| --- | --- | ---: |
| `off` | none | 1678.059 |
| `metal-short-mma16x64` | GEMM 144, QKV 48 | 1272.260 |
| `metal-rounded-gate32` | rounded gate 48 | 1494.256 |
| `metal-gqa-kv8` | D256 40, D512 8 | 2959.946 |
| `metal-mma32-prefetch` | GEMM 144, QKV 48 | 732.397 |
| `metal-attn-q4` | D256 40, D512 8 | 1689.232 |
| `metal-rms-simd32` | RMS 96 | 2385.902 |
| `metal-mma32-f32` | GEMM 144, QKV 48 | 2275.794 |
| `metal-long-mma32x64` | GEMM 144, QKV 48 | 3532.037 |
| `metal-mma32-load4` | GEMM 144, QKV 48 | 1038.481 |
| `metal-rmsnorm-simd256` | RMS 96 | 1722.446 |

These are single cold-screen observations, not a benchmark. Although the
screens reported an eligible sampled AC/low-power/pmset-mode-1/thermal-0
stratum, there was no matched ordering, warmup policy, drift gate or independent
confirmation. It would be statistically and operationally wrong to rank the
candidates from this column.

The unchanged baseline ANE cache initially inspected as 0/162 available with
zero compiler calls. The established bounded preparation path then created
48 QKV, 48 output, 48 INT8 FFN and 18 vocabulary/attention entries, exactly 162
compiler calls and zero evaluations. A separate fresh inspection observed
**162/162 available, zero compiler calls**. Receipts:

- `/tmp/rvllm-review-df305b1c-cache-prepare-all-int8/report.json`
- `/tmp/rvllm-review-df305b1c-cache-inspect-all-int8-after/report.json`

After that inspection, the baseline and all ten Metal candidates completed a
full pinned continuation with cached ANE decode and zero compile budget. The
84-token reference produced all ten expected tokens with nine ANE decode steps;
short-MMA used the unchanged six-token/16-output reference because 84 is outside
its selector and completed all 16 expected outputs with 15 ANE steps. Every
report says `matches_reference=true`. The separate prefill-only receipts prove
candidate-family dispatch; the normal full-route reports do **not** include the
dispatch ledger, so the two facts cannot honestly be collapsed into a claim
that a single receipt proves both candidate dispatch and full continuation.

The normal full-route reports exposed a second evidence-design gap: they name
`metal_research_candidate`, model path, ANE plan and config hash, but do not
record the executable SHA-256 or metallib SHA-256. The delivery gate hashed the
build products separately, but a manual operator supplied the paths. A future
daemon must create an immutable run bundle whose source, executable, libraries,
model, reference, policy and result identities are joined in one receipt.

For completeness, the following speed-shaped numbers were observed in those
single full-route runs. They are deliberately labeled **diagnostic and
unrankable**. Decode is the sum of the per-step wall times; tokens/s is derived
from those same steps. There were no warmups, no ABBA pairing and no drift
test. Owner preparation was cold on every arm. Three late arms were explicitly
ineligible because sampled thermal state was 1 rather than 0.

| selector | prefill capture ms | decode ms | decode steps/s | sampled eligibility |
| --- | ---: | ---: | ---: | --- |
| `off` | 2723.975 | 2154.280 | 4.178 | single AC/LPM/mode-1/thermal-0 run only |
| `metal-short-mma16x64` | 1900.167 | 4614.535 | 3.251 | different 6-token/15-step workload; not comparable to 84-token arms |
| `metal-rounded-gate32` | 2397.435 | 1999.014 | 4.502 | single eligible sample only |
| `metal-gqa-kv8` | 4280.625 | 2839.778 | 3.169 | single eligible sample only |
| `metal-mma32-prefetch` | 3738.890 | 2261.956 | 3.979 | single eligible sample only |
| `metal-attn-q4` | 2134.388 | 2441.348 | 3.686 | single eligible sample only |
| `metal-rms-simd32` | 3438.396 | 2595.493 | 3.468 | single eligible sample only |
| `metal-mma32-f32` | 4065.705 | 1997.316 | 4.506 | single eligible sample only |
| `metal-long-mma32x64` | 3603.202 | 2281.130 | 3.945 | **ineligible: thermal state 1** |
| `metal-mma32-load4` | 1555.179 | 2611.105 | 3.447 | **ineligible: thermal state 1** |
| `metal-rmsnorm-simd256` | 3733.866 | 4334.185 | 2.077 | **ineligible: thermal state 1** |

The contradiction between some attractive single prefill numbers and the noisy
full-route numbers is not a puzzle to explain after the fact; it is evidence
that this collection method cannot answer the speed question. No speedup,
slowdown, winner or promotion is established.

ANE candidate work was stopped when the user redirected the task toward the
process design:

- `ffn-int8-chunk4` prepared 48/48 programs with 48 compiler calls and zero
  evaluations, then separately inspected 48/48 available with zero compiles.
- `ffn-int8-down4` prepared 48/48 programs with 48 compiler calls and zero
  evaluations. Its separate inspection had just started when the manual sweep
  was interrupted at the user's request; no inspection report exists, so cache
  availability and lifecycle completion are **unknown**, not failed.
- Interleaved FFN, transpose-flags attention, stacked control, tiled sliding
  QKV and ordinary sliding INT8 QKV were not prepared or inspected in this
  continuation.
- No ANE candidate was evaluated, compared, timed or promoted here.
- Packed32 remains deliberately blocked because it lacks a compiled I/O
  descriptor and runnable single-I/O route. It must receive a rejection result,
  not be coerced into the tournament.

### Why the current process is structurally inadequate

The problem is larger than shell ergonomics.

1. **No authoritative game state.** Candidate identity, selector, control,
   source, cache parts, oracle, workload, status and promotion boundary live in
   several Rust enums and multiple generations of JSON proposals. The same
   candidate can appear as `untrusted`, `PROPOSAL_NOT_ADMITTED`, integrated, or
   tested depending on which file an agent reads.
2. **Admission is reconstructed by the operator.** The operator currently
   chooses a model path, reference, binary, metallib, cache part, feature set,
   exact test filter and output location by reading prose. One wrong path can
   create a valid-looking but irrelevant receipt. Earlier work already ran a
   zero-test filter once and initially missed the existing 84-token reference.
3. **Evidence is split across incomparable receipts.** Compile evidence,
   component arithmetic, dispatch, first-token behavior, full continuation,
   cache lifecycle and timing are separate for good reasons, but there is no
   machine-owned evidence graph joining their immutable identities. The normal
   full route currently omits dispatch counts and executable/metallib hashes.
4. **The environment changes while a human waits.** This pass began on battery,
   moved to AC, and later crossed thermal state 0 to 1. Manual sequencing makes
   it easy to compare unlike strata or notice the mismatch only afterward.
5. **The output is volatile.** Important results live under `/tmp`; prose then
   points at them. A reboot, cleanup or another operator can remove the only raw
   evidence. GitHub does not receive a canonical machine-readable result that
   candidate-generating agents can consume.
6. **Retry incentives are underspecified.** A human can rerun an unlucky arm,
   change ordering, choose a more favorable reference, or silently replace an
   attempted manifest. Even with good intentions, this creates selection bias.
7. **Candidates do not share one valid workload.** Short-MMA admits at most 64
   prompt tokens; GQA, temporal attention and long tiles require at least 64.
   ANE and CPU candidates affect decode rather than Metal prefill. A single
   scalar leaderboard would reward workload choice rather than engineering.
8. **Missing prerequisites are confused with candidate failures.** An absent
   cache, missing direct oracle, blocked I/O descriptor, insufficient disk or
   ineligible power state are infrastructure/admission outcomes. They must not
   count as numerical or speed losses.
9. **The queue is not yet a learning loop.** Existing Rust queue work handles
   many important launch invariants, but submissions, typed qualification,
   GitHub-visible results and agent feedback are not one closed system.

### Required standard: define the game before implementing the daemon

Astra should propose a versioned, checked-in standard with the following
properties. Names and exact schemas may change after source review; weakening
the properties may not.

#### 1. One typed candidate submission

Every submission must be immutable and content-addressed. At minimum it binds:

- submission schema/version, candidate ID and revision;
- exact parent Git SHA and patch/tree digest;
- backend and candidate class (Metal projection, Metal attention, Metal norm,
  ANE single-I/O graph/layout, or CPU transform);
- selector and the exact independent control;
- owned source paths and expected entry points;
- model/checkpoint, prompt/reference, tensor policy and oracle digests;
- required positive dispatch slots and exact work counts;
- shape/admission domain and explicit fallback/refusal expectations;
- compiler feature set, toolchain identity, compile budget and cache parts;
- memory/disk/resource ceilings and prohibited process interactions;
- benchmark design, warmups, repetitions, order, drift gate, practical effect
  threshold and independent-confirmation rule;
- promotion scope and incompatibilities with other candidates.

Null executable/model/oracle/cache pins mean proposal-only. Metadata must never
turn such a proposal into a runnable job. Duplicate JSON keys, unknown fields,
ambiguous defaults, symlink scope expansion and mutable file identities must be
rejected before build or device access.

#### 2. A monotonic evidence state machine

Suggested conceptual states are `submitted`, `host-rejected`, `source-ready`,
`compiled`, `component-qualified`, `dispatch-qualified`, `cache-ready`,
`full-route-qualified`, `timing-screened`, `independently-confirmed`, and a
terminal `promoted`, `deferred`, `failed` or `blocked` outcome. The exact graph
must be typed per candidate class; ANE cache readiness is nonsensical for a CPU
candidate, and a Metal norm cannot borrow a matrix oracle.

Transitions consume immutable receipts and produce a new immutable receipt.
No stage may infer evidence from a later stage or turn a missing prerequisite
into a pass. Failure attempts remain first-class history. A source, workload,
policy or threshold change creates a new candidate revision; it does not edit
or retry the old attempt in place.

#### 3. Lexicographic scoring and anti-gaming incentives

Correctness and safety are hard prerequisites, not terms that can be traded for
speed. A candidate with a compiler, guard, tensor, dispatch, continuation,
driver-lifecycle or cache-policy failure has no speed score. Static source
counts and compilation alone score nothing.

Performance should be reported within candidate class and exact workload, not
as one misleading global leaderboard. Use paired effects from a predeclared
order, reject control drift beyond the fixed gate, retain every repetition,
and require independent confirmation before promotion. Never pool power modes,
AC/battery, thermal states, prompt lengths, cache states or controls. Include
resource regressions (compile time, program count, compiled bytes, resident
memory, scratch, startup and failure rate) beside steady-state latency rather
than allowing a narrow kernel win to hide a system loss.

Candidate authors must not control the acceptance oracle, work counts, timing
order or favorable retry policy. Consider a locally held-out deterministic
challenge set whose digest and generation contract are committed before a
campaign but whose individual cases are not selected by the submitting agent.
This is not secrecy for its own sake; it prevents optimizing only the visible
fixture while retaining reproducibility after the attempt closes.

#### 4. A persistent local scheduler, not an interactive babysitter

The daemon should be safe idiomatic Rust and should evolve the existing queue
where practical. It should run under a reviewable macOS service definition,
survive restart, and keep durable state in an append-only journal or transactional
database. It must never depend on a Codex/ChatGPT turn remaining open.

It should distinguish host-only work from hardware-owner work. Host/source
validation may run when safe; Metal/ANE execution must acquire the one hardware
owner lock and recheck STOP, source head, pins, free disk, process policy, power,
low-power setting, pmset mode, thermal state, cache state and deadline immediately
before spawning. Ineligible conditions cause a quiet wait/defer result, not an
environment mutation. The daemon must not change OS power settings, kill
unrelated services, clear STOP, erase evidence or interrupt a synchronous
accelerator call. Cancellation takes effect only at declared safe boundaries.

Build products used in device work must be copied into an immutable attempt
bundle and hashed. Do not execute a mutable Cargo target path after merely
hashing it. Each output directory is create-only. Cache preparation and strict
zero-compile inspection are distinct attempts. Timing jobs require already
inspected caches and must reject compiler calls.

#### 5. Durable GitHub feedback for candidate-generating agents

Define how the Mac publishes results without making Git history the lock or
putting bulky tensors/models in Git. A likely design is one GitHub Check or
small result commit per immutable submission, backed by durable external/local
artifacts whose hashes and retention state are in the check. The summary should
be machine-readable and human-readable, and should link source SHA, attempt ID,
state transition, exact controls, receipts and terminal blockers.

Agents proposing the next candidate must consume the authoritative result
schema, not scrape prose. Network or GitHub failure cannot erase a completed
local attempt; publication is an idempotent later transition. Conversely, a
GitHub comment must never be treated as device evidence unless its receipt
chain verifies against the registered local machine/run identity.

Promotion should remain a separate explicit reviewed action. The daemon may
recommend a candidate that satisfied the standard; it must not silently change
production defaults or merge its own code.

#### 6. Test the referee more aggressively than the contestants

The daemon and schema need deterministic tests for malformed/duplicate
submissions, altered controls, zero-test filters, source or binary replacement,
wrong model/reference, missing and partial caches, compiler calls during timing,
foreign/partial dispatch, output collisions, power transitions, thermal drift,
disk-floor changes, lock contention, STOP arrival, deadline expiry, crash and
restart, orphan process recovery, interrupted safe boundaries, duplicate GitHub
delivery, network outage and attempted replay of a completed/failed job.

Provide fake sensors, fake compiler/device commands and a model-free integration
harness so CI can exercise the entire state machine without pretending to be
native acceptance. Then provide a narrowly staged local rollout proving that
the real daemon reproduces already-known receipts before it is allowed to
consume new candidate submissions.

### Astra's requested deliverable

Return a reviewable design packet, not claims of execution. It should contain:

1. A source-grounded audit of the current queue, proposal/catalog formats,
   receipts and GitHub workflows, identifying which pieces can be retained.
2. A normative `MUST`/`MUST NOT` game specification with versioned schemas for
   candidate submission, attempt, evidence transition and published result.
3. A threat/incentive model covering accidental operator error, benchmark
   gaming, favorable retries, stale/mutable artifacts, candidate-controlled
   oracles, unsafe private-API work and GitHub publication failure.
4. A typed Rust architecture and migration plan for the existing queue,
   including durable state, launchd/service lifecycle, sensors, locks, cache
   operations, runner adapters and GitHub publisher.
5. A test matrix for the referee plus a conservative staged deployment plan.
6. A precise reconciliation of the current candidate inventory. Preserve the
   evidence above, but do not promote from it and do not invent results for the
   unfinished ANE candidates.

The standard should make the correct action the easiest action for both the
daemon and candidate-generating agents. Its success criterion is not that it
can run these ten kernels once. It is that a new candidate can be submitted,
rejected or qualified asynchronously with no interactive operator choosing
favorable commands, and that another agent can derive its next proposal from a
complete, immutable, correctly scoped result.
