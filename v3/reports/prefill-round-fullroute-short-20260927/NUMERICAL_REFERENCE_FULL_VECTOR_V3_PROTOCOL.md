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
full 262144-bit Metal final rows, teacher receipt hashes, and actual combined
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
