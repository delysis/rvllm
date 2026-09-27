# Variance-tolerant Gemma 4 D512 global-decode campaign

This is an **operator** campaign on Apple M4 Max, not a Gemma 4 full-route or
MLX comparison. The existing serial experiment queue compiled and independently
oracled the single-pass streaming and split-streaming BF16 D512 attention
candidates, then advanced each through L256, L512, L1024, and L2048 without
rerunning completed jobs. The complete queue manifests, conditions, native
samples, generated Metal source, build receipts, and failed/eligible outcomes
are retained in `queue-results.tar.gz` (SHA-256
`d012010ebda2a32ba2dff48d531dd108cdb49fb5951d9955122a6e2b53007839`).
The campaign's own `campaign.json`, `round2-plan.json`, source and screen files
are adjacent to this report. Both L2048 jobs succeeded in the same sampled
power/thermal stratum, with no queue violations. Conditions were observed, not
held stable before execution.

At L2048, ten balanced ABBA/BAAB blocks timed the original BF16
`attention_decode_f16` control against each candidate. Each sample includes
100 operator iterations. The split candidate's timing includes **both** its
partial and merge dispatches in one complete interleaved command-buffer
measurement; dividing by two dispatches would misstate operation latency.

| Candidate | Mean complete GPU ms / operation | Mean control GPU ms / operation in its own run | ABBA / BAAB median control-to-candidate ratio | Control drift | Strict drift gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Single-pass streaming | 5.202 | 140.398 | 26.918× / 26.906× | 0.357% | pass |
| Split streaming, partial + merge | 0.578 | 124.856 | 211.211× / 216.850× | 5.354% | **fail** (5% limit) |

The split route is a strong *exploratory* latency prospect, not a qualified
winner. Its 5.354% control drift misses the strict 5% gate; neither result is
marked promotable by the referee. The two candidates were not directly paired
against each other; their control arms differed substantially, so dividing
5.202 by 0.578 is not an established candidate-to-candidate speed ratio.
Neither operator result proves model quality, production dispatch selection,
full-route speed, or performance relative to MLX. The native oracle records
independent FP64 and once-rounded BF16 checks for the selected kernels, not a
checkpoint-wide perplexity/logit gate.

## Immutable-screen repair

The original generator SHA-256
`24a56f7df7c41d05919e890a0b2456fd3476e3a64d071bf085ed6c6152163dcf`
could not generate the split L2048 job because it re-parsed an already-written
screen and compared `serde_json::Value`s. On this build, the stored decimal
`63.930206781609876` re-parsed as `63.93020678160987` in memory (and one tail
ratio changed at its last digits), even though the newly generated **JSON
bytes were identical** to the immutable screen. This was a false immutability
failure, not changed device evidence. The corrected generator compares the
original serialized bytes exactly, which is stricter than semantic JSON
comparison; focused regression tests and all 17 generator tests pass.

The separate `metal-global-d512-split-stream-r4s256t128-c2048-L2048.generator-repair.json`
pins the original campaign, all predecessor proofs, the original generator,
and the repaired generator SHA-256
`0dad48016c0acea3350e33e5a182bc261bad9d4410818090871b73977f1c8953`.
The new job manifest pins that repair receipt and binary in addition to the
ordinary source/build/oracle/screen inputs. The repair command generated the
same manifest twice and submitted only the **new** split L2048 job. No prior
screen, job, or queue receipt was rewritten or re-executed. The repaired
generator source is in this PR; the old campaign identity remains unchanged.

Next: obtain an independent split confirmation in a fresh sealed campaign and
directly pair the two candidates if comparing them to one another. Keep the
full-route and checkpoint-quality gates separate from these operator timings.
