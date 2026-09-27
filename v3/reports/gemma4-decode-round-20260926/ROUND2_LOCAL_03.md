# Gemma 4 round-two Metal screen: `g4-round2-local-03`

This is a default-off operator screen on the PR #7 source, not a kernel promotion, checkpoint-quality result, full autoregressive route, or MLX comparison. The [downloadable evidence archive](g4-round2-local-03-evidence.tar.gz) contains the campaign and all 42 copied queue-result directories (SHA-256 `a32a6c69be74b4e55591598c0ba356e8ffee542b8dd173ff702a28bdf736d511`). Its receipts retain original host paths as provenance; extracting elsewhere does not make them executable there. The original immutable queue receipts remain in the shared experiment queue. The archive retains `report.json`, native receipts, sample records, build outputs, sampled conditions, stdout, stderr, and job manifests, including nonqualifying timings. The complete campaign was run by the existing serial queue daemon; no thermal-stability wait was imposed.

## Gate accounting

| Stage | Outcome |
| --- | --- |
| Strict Metal 3.1, `-fno-fast-math` compile/link | 15/15 queue jobs succeeded |
| Native correctness oracle | 12/12 queue jobs succeeded |
| L0 FFN/W4/W8 and L256 attention ABBA/BAAB | 13/13 queue jobs succeeded; raw samples retained |
| L512 attention ABBA/BAAB | 2/2 queue jobs succeeded; raw samples retained |
| Queue sampled-condition eligibility | 42/42 jobs eligible; conditions were observations, not a dwell gate |
| Source/pipeline compiles during timed samples | zero in each timing receipt |

The first generated root, `g4-round2-local-01`, stopped before job generation because the release queue executable was absent from the generator's directory. The second root, `g4-round2-local-02`, preserved successful compile/oracle receipts but was not advanced: its original referee assumed the old FFN launch geometry. Commit `f5fd2d37` corrected the referee and independently checked the new geometry and split FP64 coverage before this fresh campaign. Neither earlier root was replayed or relabelled.

## L256 attention: same-weight rvLLM operator baseline

Each arm has ten alternating ABBA/BAAB blocks of four retained samples, with 100 operations per sample. The baseline is `attention_decode_f16 (BF16 typed)`, not MLX. Times below are mean GPU milliseconds per operation computed from the retained samples; ratios are the referee's paired-order medians. All four passed its 5% control-drift screen at this length.

| Candidate | Baseline ms/op | Candidate ms/op | ABBA / BAAB speedup | Control drift | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing short `r4t128` | 15.566 | 0.573 | 27.20 / 27.15× | 1.00% | promising only |
| Streaming `r4t128-c2048` | 15.568 | 0.566 | 26.99 / 27.94× | 1.46% | promising only |
| Streaming `r1t32-c2048` | 15.572 | 0.962 | 16.10 / 16.25× | 0.91% | promising only |
| Split-streaming `r4s256t128-c2048` (partial + merge) | 15.575 | 0.580 | 27.20 / 26.50× | 1.42% | promising only |

The new streaming and split-streaming arms **do not beat the existing short arm at L256** on these data. Their reason to continue is capacity and possible long-context crossover, not a short-context win. The split timing includes both partial and merge GPU work and accounts for two candidate compute encoders per operation.

## L512 attention: useful raw signal, failed drift gate

| Candidate | Baseline ms/op | Candidate ms/op | ABBA / BAAB speedup | Control drift | Referee result |
| --- | ---: | ---: | ---: | ---: | --- |
| Streaming `r4t128-c2048` | 73.808 | 2.124 | 37.74 / 39.88× | 79.9% | nonqualifying |
| Split-streaming `r4s256t128-c2048` | 65.556 | 1.027 | 73.27 / 72.89× | 178.6% | nonqualifying |

The split arm's 1.027 ms/op versus streaming's 2.124 ms/op is an intriguing **unpaired, separate-job** comparison, not an established 2× head-to-head win. The native rvLLM control varied far beyond the referee's 5% threshold, even though the paired ratios were positive. No L1024/L2048 jobs were generated: the sealed v2 predecessor chain correctly refuses advancement from these nonqualifying L512 screens. The raw results must not be silently reclassified as qualified or retried under the same campaign ID.

## L0 FFN and low-bit projection signals

All nine operator jobs completed. The three BF16 FFN variants yielded 1.68–1.75× paired-order medians against their native two-operation baseline, but control drift was 6.3–18.9%. W4 QMV variants showed 1.27–1.58× paired-order medians with 19.0–21.5% drift. W8 QMV results were mixed (0.83–1.87× order medians) and drifted 37.9–57.6%. **None passed the strict timing screen.** These are same-weight synthetic operator comparisons, not real-checkpoint W4/W8 quality or MLX parity.

## Required next trial design

The current v2 screen correctly retains unstable evidence, but its global max/min control-drift cutoff stalls long-context exploration under ordinary changing host conditions. Add a separately labelled **variance-tolerant exploratory continuation** to the standard process: revalidate every pinned oracle/build/sample and exact work count; use within-block A/B normalization and both ABBA and BAAB directions plus whole-block uncertainty; preserve drift as a reported covariate and never convert this exploratory path into promotion evidence. The continuation must generate a new sealed identity and queue jobs, and must not modify these v2 receipts or bypass checkpoint quality and independent confirmation. A direct paired comparison of the two L512 candidates and L1024/L2048 attention screens should follow. The final promotion gate can remain stricter and separately require controlled confirmation.

PR #7 remains default-off. Its local Rust/operator tests and host source-export gate pass; hosted checks are tracked on the PR. The separate donor12b PR #6 owns the current real-weight one-layer W4 route evidence and is not folded into these synthetic operator speed claims.
