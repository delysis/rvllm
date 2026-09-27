# Synthetic continuation diagnostic: complete four-arm result

This report follows the frozen analysis in `HELDOUT_QUALITY_PROTOCOL.md`.
The two passages and continuations were **written by Codex**, not by a human
or sampled from an independent corpus. The submitted source JSON incorrectly
says `human-authored`; its bytes remain unchanged because all four jobs pinned
its SHA-256. The protocol's provenance correction supersedes that field.
These observations are not a checkpoint-wide quality gate, independent
numerical oracle, performance measurement, or promotion decision.

## Identity and receipt status

- PR #8 tree before this result: `fe14c959`; safe-Rust offline referee source
  introduced at `ffe6c0b0` and passed four focused tests. The referee's
  `rvllm.gemma4_teacher_summary.v1` output is
  `heldout-teacher-summary-v1.json`, SHA-256
  `da4713d541dc1f146317f9631287be85ba7d7a40905255bddbb5b2b1c5ff8fc7`.
- The complete compact queue archive, including `job.json`, `report.json`,
  `conditions.jsonl`, `trial.stdout`, and `trial.stderr` for **every** arm, is
  `heldout-teacher-queue-results-v1.tar.gz`, SHA-256
  `20f03e0dfc84b3fcf7d8c4fbe2b4b02e50ba6a1b2641277bd1ea4665ccc90d95`.
  Its listing contains all 20 expected files. The JSON summary also records
  the individual receipt-file hashes.
- The source text SHA-256 is
  `7473805e3bf9d3762dad31ba2928b2967c1353411e3cf074008628d82ea977f2`;
  tokenizer SHA-256 is
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
  All four jobs pinned original BF16 `google/gemma-4-12B-it`
  `model.safetensors` SHA-256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`,
  the same frozen executable SHA-256
  `23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463`,
  exact prompt/target IDs and their respective normal or combined metallib.
- All four serial queue jobs terminated `succeeded`, exit 0, with unchanged
  input files and no reported violations. All four sampled conditions were
  eligible in the observed AC / power-mode-2 / thermal-state-0 stratum. No
  stable-condition wait was imposed. Per-step GPU readback invalidates **all
  timing** in these jobs.
- Both combined arms actually dispatched the candidate: tiled GEMM 48, QKV
  48, raw projection 96, raw norm 96, D256 Q4 attention 40 and D512 Q4
  attention 8. Both controls had empty research dispatch counts. This is a
  normal-route candidate observation, not fallback.

## Every predeclared aggregate

| Synthetic case | Prompt / targets | Control NLL sum / mean | Combined NLL sum / mean | Combined minus control sum | Different target ranks | Different greedy IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| map-catalog | 203 / 28 | 80.899545 / 2.889269 | 80.489513 / 2.874625 | -0.410032 | 4 | 0 |
| reservoir-log | 199 / 36 | 131.855451 / 3.662651 | 133.168436 / 3.699123 | +1.312984 | 10 | 0 |

The rank-change positions are zero-based: map-catalog 1, 12, 15, 23;
reservoir-log 1, 2, 7, 10, 12, 14, 23, 24, 31, 33. Neither case has a
first differing *sampled greedy ID*. The target was also the greedy ID at
17/28 map positions and 18/36 reservoir positions in **both** routes; the
forced target sequence in `generated_token_ids` is transport and must not be
mistaken for greedy agreement. The complete JSON contains every target ID,
control/candidate NLL, NLL difference, rank, and sampled greedy ID at all
64 positions, including unfavorable positions. Largest absolute single-step
candidate-minus-control NLL was 0.364647 for map-catalog and 1.686050 for
reservoir-log. Direct sums from each raw `trial.stdout` independently match
the referee's sums and the CLI's reported totals.

## Interpretation and next gate

The candidate did not change any sampled greedy choice on these fixed
teacher-forced trajectories, but it changed target ranks and log likelihoods.
The map case's tiny aggregate improvement and reservoir case's larger
aggregate regression point in opposite directions. Two investigator-authored
cases have no calibrated acceptance threshold and cannot establish a general
quality effect. Equal greedy choices do not prove numerical equivalence.
The next quality gate needs natural independently selected text and a
separately sealed same-checkpoint, same-token-boundary CPU/HF or MLX reference;
neither is supplied by these receipts. No kernel promotion follows from this
screen. The earlier standalone M304 speed screen also failed its predeclared
5% drift gate and is not repaired by this numerical observation.
