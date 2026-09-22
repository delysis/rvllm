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
