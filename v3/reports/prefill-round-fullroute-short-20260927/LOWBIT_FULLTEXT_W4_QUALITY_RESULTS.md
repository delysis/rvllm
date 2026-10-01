# Full-text-projection W4 teacher-forced diagnostic

The four immutable serial jobs declared in
`LOWBIT_FULLTEXT_W4_QUALITY_PROTOCOL.md` terminated `succeeded`, exit 0,
with unchanged pinned files, eligible sampled conditions, no overrun and no
violations. Their sampled observations were AC power, `pmset` mode 2,
low-power mode off, nominal thermal state 0 and no sampled competitor. No
thermal-stability dwell was imposed. Neither a job nor a receipt was replayed
or overwritten.

The separate safe-Rust `rvllm_gemma4_fulltext_w4_teacher_summary` referee,
after the narrow input-pin lookup repair described below, accepted all four
exact tracked manifests, the original model/config/tokenizer and source
hashes, authenticated package manifest and 328-selector plan, package
sidecars, full condition journals, every teacher step and aggregate, and
actual named W4 **decode** dispatch for projection, gate/up and QKV with
donor SG8 attention. Its six focused tests passed on the repaired source.
Independent sums of `trial.stdout` per-step NLL matched the four reported
totals. The complete 43-position JSON is
`lowbit-fulltext-w4-teacher-summary-v1.json`, SHA-256
`acb578792975c546c9cbdc646a58c738f041d8ab49f820f3751a982769aa17f6`.
It includes every target, rank, NLL, candidate-minus-control delta,
sampled-greedy ID, dispatch ledger and five receipt hashes per arm. The
complete 20-file outer queue archive (job, report, condition journal,
stdout and stderr for each arm) is
`lowbit-fulltext-w4-queue-results-v1.tar.gz`, SHA-256
`4333fae2189b79729cdc025e926e2dcb7643cb53d68c2b440728f794470f9f28`.

| Synthetic case | Prompt / targets | BF16 donor NLL | Full-text W4 NLL | W4 minus donor | Rank changes | Sampled-greedy changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| estuary-sensor-v1 | 218 / 21 | 56.766026 | 48.687358 | -8.078668 | 7 | 4 |
| textile-catalog-v1 | 220 / 22 | 35.139971 | 50.946632 | +15.806661 | 6 | 4 |

The estuary target-NLL changes have 12 decreases and 9 increases; their
extremes are -2.927219 at zero-based position 2 and +1.123387 at position
5. Sampled-greedy IDs differ at positions 1, 5, 8 and 12. The textile
changes have 10 decreases and 12 increases; their extremes are -3.162714
at position 7 and +12.376257 at position 19. Sampled-greedy IDs differ at
positions 7, 10, 15 and 19. All positions, including unfavorable ones, are
retained in the JSON. The total selected-target NLL signs oppose one another
across the two cases; no uniform quality improvement is observed.

The W4 decode ledger recorded projection/gate/QKV counts of
2016/1008/1008 for estuary and 2112/1056/1056 for textile, alongside SG8
local/global attention. The native BF16 donor used the same SG8 library.
The W4 runs also retained 21/22 `native_projection` counts; this report
does not label those counts a low-bit fallback without a separate source
attribution. The frozen CLI does not separately expose generic low-bit
**prefill** dispatch. `generated_token_ids` are supplied teacher-forcing
targets, **not** sampled model agreement; actual sampled IDs are in
`teacher_forced.steps[].sampled_token_id`.

The first offline referee invocation rejected `config.json: missing or changed
input pin` before writing a summary: basename suffix matching also selected
the pinned `generation_config.json`. A focused safe-Rust repair matched exact
basenames, then the second offline invocation rejected `tokenizer.json` because
the package and original checkpoint both pin byte-identical files of that
name. The final repair accepts one or more exact-basename pins only when
**every** matching SHA equals the sealed value; it still rejects an absent
or conflicting pin. A negative regression test covers suffix collisions
and conflicting duplicates. Neither repair changed queue manifests,
conditions, package, model, targets or numerical gates. Both rejected
invocations wrote no result file; the successful third offline invocation
used a fresh create-new output. The repaired referee source SHA-256 is
`361e7934fc012a9457740557bd04b2aa3bb8b08a560155c12e33e93d9455a361`.

The source JSON SHA-256 is
`ad075af3147a6c242aed94f0c2f5ac40845c09b38c2929983be5dbd234d5b80a`;
the exact-token fixture SHA-256 is
`f4899669393b9726953a698eaa3049819de2d83822e67416d891b511e1cff65c`;
the original tokenizer SHA-256 is
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The authenticated W4 package manifest SHA-256 is
`abcd5b043322efe1756b6ac81e805f0aee9da5b04c9e10a77ae391532f026ce3`.
The sealed protocol/manifests contain all further executable, metallib,
package-asset and model pins.

This package quantizes 328 dense **text projection** matrices to group-32
W4; embeddings and multimodal tensors remain BF16. Both prompts and their
continuations were Codex-authored synthetic diagnostics, not independent or
natural held-out text. The BF16 donor is not an independent numerical oracle.
Two such cases cannot qualify checkpoint-wide W4 quality or a benchmark score;
there is no calibrated acceptance threshold or promotion. Per-step logits
readback changes scheduling, so none of these trial timings are speed
evidence.
