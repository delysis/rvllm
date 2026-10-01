# Distinct-input prefill-boundary numerical diagnostic, v2

The prospectively frozen safe-Rust referee accepted all six **new** immutable
correctness jobs. Each exited 0 with unchanged pinned files and no overrun.
The two HF jobs were **timing-ineligible** solely because sampled power-observer
data were older than 2,500 ms; their numerical outputs are retained under the
predeclared v2 numerical-only policy. All four Metal jobs were queue-eligible.
This is not a retry or retroactive acceptance of the separate v1 MMLU jobs:
the v1 strict referee's rejection and raw receipts remain unchanged.

The frozen [protocol](NUMERICAL_REFERENCE_DISTINCT_V2_PROTOCOL.md) pins the
original BF16 12B-it checkpoint, tokenizer, source, scripts, executables,
metallibs, prompt IDs and first target IDs. Both passages and continuations
are **Codex-authored synthetic diagnostics**, not natural held-out text. The
HF full-prompt final row and Metal prefill-final row address the same logical
next-token position. The Metal teacher decode step zero is a different
last-token-replay boundary and is not compared here. The Metal probe exposes
only target logit, rank, NLL and greedy ID, not its full vocabulary vector.

| Case / arm | Target logit | Target rank | Target NLL | Greedy ID |
| --- | ---: | ---: | ---: | ---: |
| Observatory, 213 prompt tokens, target 80444: HF | 6.78125 | 568 | 13.370230 | 107 |
| Observatory: control | 6.50000 | 715 | 13.680660 | 107 |
| Observatory: combined | 6.62500 | 621 | 13.547068 | 107 |
| Library, 203 prompt tokens, target 506: HF | 16.00000 | 14 | 6.638835 | 870 |
| Library: control | 16.37500 | 14 | 6.398744 | 870 |
| Library: combined | 16.25000 | 13 | 6.456453 | 870 |

Relative to HF, observatory control/combined target NLL is higher by
`0.310431`/`0.176838`; library control/combined NLL is lower by
`0.240091`/`0.182382`. All three arms choose the same greedy ID in each
case, but target ranks differ as shown. Neither Metal route is uniformly
closer to HF across these two positions. There is no post-hoc acceptance
threshold or favorable-case selection.

Both combined jobs actually dispatched per case: tiled GEMM 48, QKV 48, raw
projection 96, raw norm 96, Q4 attention D256 40 and D512 8. The control
research dispatch was empty. The referee checked the exact prompt/target
tokens, expected route dispatch, full 262,144-entry HF outputs and tie order,
finite Metal scores, all six frozen manifests, complete condition journals,
and SHA-256 of every outer receipt and separately written HF output. The
condition journal had 121/28/21 observations for observatory HF/control/
combined, and 116/26/21 for library. HF had 5 and 8 stale-power observations
respectively; no other violation was admitted. Observed controls were AC,
power mode 2 and thermal state 0, with no sampled competitor. This condition
record is not timing evidence: the Metal probe adds an M-row LM head and
readback/synchronization, and all probe timing fields are invalid.

Immutable evidence:

| Artifact | SHA-256 |
| --- | --- |
| `numerical-reference-distinct-v2-source.json` | `2f212a4b1c316e49b52b78fc80e33a62c35772f0028e4da7d52ee7268c3d008a` |
| `numerical-reference-distinct-v2-summary-01.json` | `21dab18bb08d9e41cf9e41f0ac66d86a695e1b68577d87a7ca4dc1f47eaa14b2` |
| `numerical-reference-distinct-v2-queue-results-01.tar.gz` (all 30 outer queue files) | `c46498b013cac66e3159e183e1a90e80874c1794f6aba20c8b945114f378ac12` |
| `numerical-reference-distinct-v2-hf-outputs-01.tar.gz` (both full HF JSONs) | `e5d4b90758fab49662e563b2649c3f4047e12d7d4ea3e367afa46618f021dc86` |
| Observatory HF full output | `76a1086bc22f727de746f9fbf9287de1e1b7c3f53e57b5153ab66a1c2c1ace74` |
| Library HF full output | `88fedd6f64664c798bc5169efef0824909286c0f38540ff33c25d033d9984874` |

The frozen referee source SHA-256 is
`477ec2399afd35f91258ef397a3ba9f039106df941f85e1a791480880897c023`;
its four focused host tests and build passed before submission. The summary
JSON contains all four scores, condition counts and violations for every arm,
plus hashes of the individual receipts. The queue archive preserves every
job, report, condition journal, stdout and stderr; the HF archive separately
preserves the complete vectors written outside stdout. No job was replayed.

This two-position, synthetic-input observation does **not** establish full
Metal-vs-HF logit parity, continuation or checkpoint-wide quality, a first
arithmetic defect, speed, or promotion. CPU Transformers version and BF16
arithmetic may differ from Metal. Independent natural-text quality and a
route-preserving same-input SG8 attention reference remain open gates.
