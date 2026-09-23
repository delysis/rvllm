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
