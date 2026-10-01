# Balanced16 W4 wave 1: terminal condition disqualification

This is a receipt-preservation record, not a numerical result or a replacement
for the prospectively sealed 32-arm referee. The original immutable job remains
in the serial queue result directory; no job was replayed, modified, or scored
to produce this record.

- Source and selection protocol: `MMLU_BALANCED16_FULLTEXT_W4_PROTOCOL.md`.
- Exact submitted manifest: `mmlu-balanced16-w4-queue/q01-w4-job.json`,
  SHA-256 `04a0ee3047896613df16336f5363951e9dcbb980861624ae5920c5787111b795`.
- Queue ID: `prefill26-mmlu-balanced16-w4-q01-w4-v1-20260928`.
- Terminal queue report: `succeeded`, exit code 0, input files unchanged,
  `sampled_conditions_eligible=false`, `overdue=false`.
- Predeclared minimum free disk: 68,719,476,736 bytes (64 GiB). All 32 reported
  condition violations sampled free disk below that guard; their range was
  66,991,669,248 to 68,685,328,384 bytes. This is a condition failure even
  though the trial process exited successfully.

The queue's five original outer receipt files are copied byte-for-byte into
`mmlu-balanced16-w4-q01-w4-disqualified-receipt.tar.gz`, SHA-256
`dc9c9db8fb4464e91ed0e2cd7fd1f512ccc4b7910c73474779970a4531ed63b8`:

| File | SHA-256 |
| --- | --- |
| `job.json` | `216b7ef27835613efed94394ba6e0af3f8aceb7ce8dc7c9c4fad19b54e952302` |
| `report.json` | `b8b223b895f79089f76e336bec6da93d27ac36f82ca1aa01cc5f36abd5ea8fc4` |
| `conditions.jsonl` | `bd3da3cac9d5b04e8dcad602763197b8e59fdb02da14d6b34b3a112fb78ec98b` |
| `trial.stdout` | `d42ed783542d944dfee467f9be728893a64e24c2aa429917137fdae71a9eb19f` |
| `trial.stderr` | `5d75cfbaeaa4829c57ed146d5c8dcd309ae87d722c5642ba875ce605b1e5856d` |

The frozen `rvllm_gemma4_mmlu_balanced16_w4_summary` requires every arm's
sampled conditions to be eligible. This wave, and therefore the planned
32-arm aggregate, cannot pass that referee with the submitted manifests and
receipts. Do not weaken the guard, score partial positions as acceptance,
rerun this ID, or select replacement cases. Pending q02/q03 jobs and the
running daemon were not changed while archiving this failure.
