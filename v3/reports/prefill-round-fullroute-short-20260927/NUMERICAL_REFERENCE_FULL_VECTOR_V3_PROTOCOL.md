# Distinct-input full-vocabulary prefill diagnostic, v3 (pre-run)

This is a new correctness-only research trial. It is **not** a replay or
reinterpretation of the completed v1 or v2 jobs. No v3 job has been submitted.
The two passages and continuations in
`numerical-reference-full-vector-v3-source.json` were authored by Codex before
model scoring of these exact inputs. Source SHA-256:
`f6d06d159a16472c4dba9496f2cbdc0b6f3b3b2498b0e05dfae64cc347f603bd`.
They are synthetic probes, not natural held-out text or a quality benchmark.

The safe-Rust tokenizer fixture verified exact prompt-prefix tokenization with
the original 12B-it tokenizer SHA-256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`:

| Case | Prompt tokens including BOS | Continuation tokens | First target ID |
| --- | ---: | ---: | ---: |
| bridge-cable-v3 | 229 | 19 | 56896 |
| seed-bank-v3 | 246 | 19 | 52102 |

Only the first continuation target is scored. Its identity is source- and
tokenizer-derived, not model output. Both prompt lengths should exercise the
combined raw projection/norm and Q4 attention route; a fallback invalidates
that arm. Use the original BF16 12B-it checkpoint and exact normal/combined
metallibs, not a quantized or changed model.

The new default-off safe-Rust `--teacher-prefill-full-logits-output` hook is
feature-gated behind `metal-quality-research` and requires
`--teacher-prefill-last-logits` in single-prompt JSON teacher mode. It writes
only the final prefill row, preserving each `f32::to_bits` value as an unsigned
JSON integer, including negative infinity. NaN and positive infinity fail
closed. It requires a new absolute path, uses create-new writes, and records
the file SHA-256, byte count, vocabulary size, and negative-infinity count in
the teacher report. The source SHA-256 is
`5158928cc2f97381b8943e75657b0485f4cbb76abd55a56867e2cfb9c05277ec`;
the separately built release executable SHA-256 is
`1f9cfedd5c1a91f4a1046e2930e13071847fc0b90c7bafa2f391f3e99b3082c5`
at `/tmp/rvllm-prefill-full-vector-artifacts-20260927/rvllm_metal_infer`.
The source hook's 18 feature-gated and 11 default host tests passed; the Apple
feature check and isolated release build passed. These are source/build gates,
not device or numerical acceptance.

Before queue submission, implement and test a separate fail-closed safe-Rust
generator and referee. Freeze six **new** manifests and unique output paths in
serial dependent order, HF/control/combined for bridge then HF/control/combined
for seed, with IDs
`prefill26-fullvec-v3-{bridge,seed}-{hf,off,combined}-20260927`. Pin source,
tokenizer, exact prompt JSONLs and target IDs, checkpoint config and weights,
HF script/environment, Metal source/executable/metallibs, and generator. The
referee must verify terminal success/exit zero/unchanged pins/no overrun,
complete condition journals, exact token IDs, full 262144-logit HF output,
full 262144-logit Metal final rows, teacher receipt hashes, and actual combined
GEMM48/QKV48/raw projection96/raw norm96/Q4 D25640/D5128 dispatch per case.
Preserve all five outer queue receipt files per job and each separate HF and
Metal full-vector output, including failures. Never overwrite or resubmit a
completed ID.

HF full-prompt final row and Metal prefill-final row refer to the same logical
next-token position. Metal teacher decode step zero follows a replay of the
last prompt token and is a **different** boundary. The referee should derive
target logit/rank/NLL and greedy ID from each retained vector, using explicit
tie ordering, and check consistency with each program's scalar report.
Predeclare whole-vector exact-bit match count, finite maximum/mean/RMS absolute
difference, nonfinite classifications, and top-256 ID overlap for HF versus
each Metal route and control versus combined. Retain the raw vectors; report
both cases without favorable selection. No post-hoc numerical threshold or
automatic parity claim follows from these descriptive metrics.

For correctness-only numerical observation, the referee may label a terminal
successful HF arm timing-ineligible if **only** power-observer freshness is
violated, matching the prospective v2 policy; any other condition violation
fails closed. Do not alter the queue threshold or retroactively pass v1.
All probe timings are invalid because an M-row LM head and readback add work
and synchronization. CPU Transformers version and BF16 arithmetic may differ
from Metal. Two synthetic first-target positions cannot establish quality,
checkpoint-wide agreement, first internal arithmetic error, speed, or kernel
promotion.

## Generator checkpoint (still no submission)

The separate safe-Rust `rvllm_gemma4_full_vector_job_gen` source SHA-256 is
`36acac1144a53470d8e173a4f03225c912a2a70d7b928631482aab0d44e8f912`.
Two focused host tests and a build passed. It seals the v3 source, tokenizer,
three exact v2 manifest templates, new Metal source and executable, and
rejects changed token identities or an existing output directory. An actual
generation pass produced the six expected dependent manifests and two prompt
JSONLs under the **unsent** `numerical-reference-full-vector-v3-dry-run/`
directory. Those files remain local and unsubmitted; their paths must not be
used as final queue artifacts. Before generating an authoritative fresh
directory, finish the separate full-vector referee and review every resulting
manifest and output path. No v3 queue job has been submitted.

## Referee and authoritative manifest checkpoint (still no submission)

The separate safe-Rust `rvllm_gemma4_full_vector_summary` source SHA-256 is
`ac53c2e77926b786f3653bd3e01c726764675d25bdc3565a90c951bed45f3720`.
Its six focused host tests passed, including all six frozen manifest hashes,
stale-power-only condition classification, exact combined dispatch, HF tie
ordering, full-vector bit/negative-infinity metrics, and rejection of wrong
Metal scalar rank or full-vector receipt SHA. A host Clippy correctness gate
passed. This is not a device-result or parity verdict.

After that referee checkpoint, the generator wrote a **new** authoritative
`numerical-reference-full-vector-v3-queue/` directory, distinct from the
preserved dry run. The six sealed manifest SHA-256 values in queue order are:

| Arm | Manifest SHA-256 |
| --- | --- |
| bridge HF | `194fda0dc5807ff6d02adac0b0b66b7a8f2fc6012bb31e7d6f58ebce22604a89` |
| bridge control | `ec54f48d0b303643a3c8f5c35a84467faf1d9d4c38b77c08d2431738ead37a85` |
| bridge combined | `767d4bf67d0c0fa80c18806ccbf39ca7a12d44c431ed567f4ac2fab6aadc0744` |
| seed HF | `02fcb7a07c521e3f1fb064226aa39ba481d76c5a6de22d53a5c3dd203df877f8` |
| seed control | `2ac12e6e661796a16929a264d81c2e541ddbb38ec06948967f7199d09129e1d4` |
| seed combined | `83b1fd6fb074edc2b090d7014e555170ea2555142cfb74a4069398018eab76b1` |

Bridge and seed prompt JSONL hashes are respectively
`d59925dd4d9297fd802cd3fec73fc82162b8991b650ee8f9056b6999c120a6db`
and `b7f8c4f4761a995b7bc4696388e046dd841651a47d7b7bc60fc2271ccba923ed`.
Read-only review found the expected serial dependencies, fresh output paths,
original model/HF environment and normal/combined metallibs, exact target
IDs, new frozen Metal executable/full-vector flag and no old v2 source or
executable pin. Neither the dry run nor these authoritative jobs have been
submitted yet. Submission must use only the authoritative six manifests in
the order above, after verifying queue state and unchanged pins.
