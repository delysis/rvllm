# Experiment queue continuation — 2026-09-21

Base: `a039716d47c1b3aec28d05812845659d9d4b696b` on `delysis/rvllm:main`.
This is a source/host-qualification continuation of [the checkpoint](../HANDOFF.md),
not a hardware campaign result or a new optimized default.

## Established starting point

All four checks on the exact base commit succeeded in
[CI run 35137950536](https://github.com/delysis/rvllm/actions/runs/35137950536):
Check, Test, GB10 compile-check, and Apple shipping release safety. This closes
the checkpoint's outstanding CI-status question, not its inference acceptance.
The ordinary Linux workspace tests do not exercise the private, macOS-only
queue module. The shipping job deliberately does not enable private ANE.

## Fixed launch-boundary gap

The old scheduler established a `StableGate`, then synchronously hashed all
pins. `execute` checked only fresh readiness and equal power controls after
hashing. It did not carry the gate's last-observation timestamp into that
check. A hash lasting more than 2.5 seconds could therefore bridge an
unobserved interval. `launch` also rehashed the executable after that check.

The continuation carries the same gate and original waiting deadline into
`execute`. A scoped, host-only verifier checks the unchanged pin set while the
owner samples launch conditions at one-second waits. Completion is checked
without imposing a mandatory one-second delay on short hashes. The owner
explicitly joins verification on success, loss of readiness, probe errors and
verification errors before any trial can start. The existing 2.5-second gap
rule remains unchanged. A successful hash is followed by another observation;
a fresh equal-valued snapshot does not waive an intervening gap.

Trial launch no longer performs an unsampled duplicate hash. Output handles
and durable attempt metadata are prepared before one last stable-gate check
immediately before `Command::spawn`. Validator executable verification remains
explicit and independent. This is still cooperative pinning, not protection
against hostile filesystem mutation or arbitrary OS preemption.

Loss of readiness during pin verification defers launch and preserves the
original waiting deadline. A changed pin or probe error halts the queue. If
readiness is lost after durable attempt setup, the attempt is retained as a
failed, `trial_started: false` result and the queue halts without retry. An I/O
or probe error after setup can leave an incomplete attempt, which likewise
cannot be replayed. No attempt is deleted to manufacture a clean run.

## New qualification path

The added host workflow builds the normal release queue executable on macOS,
records its hash/source/toolchain, and runs only that binary's non-ignored
tests. Linux compiles the binary source's std-only test path with `rustc --test`,
without enabling the deliberately macOS-only private research feature. It
exercises only the portable verification helper. macOS also exercises the
existing queue/power-policy tests and the new native regressions. These are
different coverage levels, not a Linux queue build or platform parity.

A stopped queue now returns before starting its power observer or registering
a signal handler. A native executable smoke uses a fresh temporary queue with
STOP already present and proves that no result or power journal is created,
even with an unreadable-as-JSON manifest. It neither opens models nor invokes
an accelerator. It does not qualify live scheduling progress under real power,
activity or disk conditions.

Eight portable helper tests cover initial refusal/error, final re-observation,
sampling during a pending hash, joined exit on lost readiness/probe error,
changed-pin failure and caught verifier panic. Three additional macOS tests
cover a real `StableGate` gap through verification, refusal at the last spawn
boundary, and the stopped worker's early return.

## Evidence boundaries and next gate

The initial host CI attempt stopped on pre-existing formatting differences in
untouched Apple files. The workflow now explicitly checks all three changed
queue Rust files. Its first Linux Cargo build also correctly hit the existing
macOS-only private-API guard; the workflow was corrected to test the portable
path without that feature. Neither failure justifies weakening a product guard
or silently reformatting unrelated source. The native Cargo qualification is
retained. The verifier-panic fixture runs under the test harness and does not
claim recovery from release `panic=abort`, forced termination or kernel panic.

The authoring environment has no Rust compiler or reachable Git clone endpoint.
The modified source was reconstructed from GitHub reads, with both existing
Rust files checked against their exact base Git blob hashes. This is source
identity evidence, not a compile/test result. New Rust tests, formatting and
executable-smoke results must be taken from the new workflow's actual receipts.
No new test pass is claimed by this document.

No STOP marker, queued manifest, model, frozen local executable, power setting,
shipping/private-API boundary or multi-I/O quarantine was changed. No ignored
fixture, S2 job, KV-import trial, ABBA comparison or accelerator test was run.
The local M4 worker still needs to be built/frozen and checked against newly
verified local assets before the handoff's matched ABBA stratum can proceed.
Do not reuse old PID exemptions or promote an ANE-versus-llama.cpp speed ratio
from this work.
