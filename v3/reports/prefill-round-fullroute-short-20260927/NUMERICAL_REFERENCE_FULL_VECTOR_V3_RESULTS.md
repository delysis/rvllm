# Distinct-input full-vocabulary prefill diagnostic, v3: result

This report closes the predeclared correctness-only protocol in
`NUMERICAL_REFERENCE_FULL_VECTOR_V3_PROTOCOL.md`. All six immutable serial
jobs `prefill26-fullvec-v3-{bridge,seed}-{hf,off,combined}-20260927` finished
with queue status `succeeded`, exit code zero, unchanged pinned files, and
no overrun. The frozen safe-Rust referee accepted all six manifests, full
262,144-logit outputs, scalar reconstructions, condition journals, research
dispatch counts, and receipt hashes. No job was replayed.

Both Metal routes were sampled eligible, with no violations, on AC power,
power mode 2, and thermal state 0. The two CPU/HF jobs were **timing-ineligible**
solely because the power observer was stale: four observations for bridge
(age 2,607–3,487 ms) and six for seed (2,506–3,159 ms). Their observed
samples had AC/power mode 2/thermal state 0 and no sampled competitor. The
prospectively declared correctness-only policy admits those successful HF
outputs as **numerical-only observations**, retaining the violations; it
does not turn either job into timing evidence. The Metal controls had empty
research dispatch. Both combined jobs actually dispatched tiled GEMM 48,
QKV 48, raw projection 96, raw norm 96, Q4 D256 attention 40, and Q4 D512
attention 8. The separate full-vector files were checked against their
recorded SHA-256 receipts. The supplied first-target IDs in
`generated_token_ids` are transport, **not** greedy agreement; greedy IDs
below were reconstructed from the vectors.

| Synthetic case; first target | Route | Target logit | Target rank | Target NLL | Greedy ID |
| --- | --- | ---: | ---: | ---: | ---: |
| bridge-cable-v3; 229 prompt tokens; 56896 | CPU/HF | 4.40625 | 819 | 14.820960 | 107 |
|  | Metal control | 1.125 | 2247 | 17.475912 | 107 |
|  | Metal combined | 2.453125 | 1418 | 16.254852 | 107 |
| seed-bank-v3; 246 prompt tokens; 52102 | CPU/HF | 17.5 | 7 | 4.770247 | 107 |
|  | Metal control | 17.75 | 6 | 4.955670 | 107 |
|  | Metal combined | 16.75 | 8 | 5.426008 | 107 |

All 262,144 logits were finite in each vector; neither route produced a
negative-infinity logit. “Exact bits” compares each Metal `f32::to_bits`
value with the CPU/HF JSON value converted to `f32`, or the two Metal bit
vectors directly. Differences use finite logits only. Top-256 overlap uses
descending logit with token-ID tie ordering.

| Case | Comparison | Exact bits / 262,144 | Max abs. | Mean abs. | RMS | Top-256 overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bridge | HF vs control | 60 | 4.898438 | 1.670531 | 1.760072 | 216 |
| bridge | HF vs combined | 17 | 2.875000 | 1.251825 | 1.301468 | 233 |
| bridge | combined vs control | 7,445 | 2.164062 | 0.434490 | 0.507737 | 235 |
| seed | HF vs control | 296 | 11.281250 | 2.366091 | 2.716276 | 180 |
| seed | HF vs combined | 126 | 12.968750 | 2.999185 | 3.322975 | 179 |
| seed | combined vs control | 1,240 | 4.343750 | 0.659698 | 0.731576 | 234 |

The combined route is descriptively closer to HF on several bridge metrics,
including target NLL (combined minus HF +1.433891 versus control minus HF
+2.654952), but farther on seed target NLL (+0.655761 versus +0.185423)
and seed whole-vector mean/RMS differences. The same greedy ID across all
arms does **not** establish distribution parity. There is no uniform
“closer route” across these two inputs, no predeclared numerical acceptance
threshold, and no identified first internal arithmetic difference.

## Identity and retained evidence

- Original BF16 `google/gemma-4-12B-it` checkpoint, original tokenizer,
  exact prompt IDs, normal/combined metallibs, HF environment, frozen Metal
  executable, and all six manifest SHA-256 pins are in the protocol and
  tracked `numerical-reference-full-vector-v3-queue/` manifests. Synthetic
  source SHA-256:
  `f6d06d159a16472c4dba9496f2cbdc0b6f3b3b2498b0e05dfae64cc347f603bd`.
- Frozen referee source SHA-256:
  `ac53c2e77926b786f3653bd3e01c726764675d25bdc3565a90c951bed45f3720`.
  Its six focused tests, build, and host Clippy correctness check passed.
  The exact merged-tree debug binary rebuilt successfully before use.
- Accepted all-case JSON `numerical-reference-full-vector-v3-summary-01.json`
  SHA-256:
  `26a0f26d8906f122c0fb2c37c79a056d6d6accd85ef9e2ffd843fc6e2502a283`.
  It retains exact scores, all six condition journals/eligibility labels,
  every predeclared vector metric, and receipt hashes.
- All **30** outer queue receipt files (job, report, conditions, stdout,
  stderr for each arm) are in
  `numerical-reference-full-vector-v3-queue-results.tar.gz`, SHA-256
  `c7108e90da5bb4c504732484d48c82938c77f3dd4b15f6504e4b3ae4d30d4d4d`.
  The six separately written full HF/Metal vector JSONs are in
  `numerical-reference-full-vector-v3-outputs.tar.gz`, SHA-256
  `65e7351ca2e36689549a805a1d07ad40b0da608f0ca5df6d80f1f2e490a380e3`.
  The originals remain in the authoritative queue directory; the unsubmitted
  dry-run directory is preserved separately and was not used by the queue.

HF full-prompt final-row logits and Metal prefill-final logits address the
same logical next-token position; the Metal teacher decode step zero follows
a replay of the last prompt token and is **not** this boundary. CPU
Transformers version and BF16 operation order may differ from Metal. The
feature-gated Metal probe adds a full-prompt LM head/readback and
synchronization, so **all probe timing is invalid**. These are two
Codex-authored synthetic first-target positions, not natural held-out text
or a checkpoint-wide quality test. This result does not qualify numerical
parity, continuation quality, speed, or kernel promotion.
