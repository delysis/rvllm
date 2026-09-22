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

## Work for ChatGPT 6 Pro in the web app

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
