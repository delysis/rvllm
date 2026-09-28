# Isolated prefill27 research snapshot

This branch records the review and component experiments for Astra's prefill27
packet, applied to rvLLM base `345504171fdadc2da2aa92d2524859a26da912a5`.
It is **not** a production integration, a kernel promotion, or an accepted
speed result. The active prefill PR is separate and unchanged by this branch.

The seven correctness rungs (M=6, 17, 64, 256, 512, 1024, 2048) used 28
immutable serial queue jobs. The wide64 down-norm output and FP32 raw
intermediate matched the incumbent bitwise on their shared fixtures through
M=2048. The MMA8K32 local-attention candidate passed the independent FP64
oracle; it matched the incumbent bitwise through M=512 but retained one-BF16-ULP
differences on specified M=1024 and M=2048 fixtures. These are synthetic
component checks, not a full-model correctness claim.

Two separate M=256 operator timing trials completed with clean queue
conditions, but their frozen adjudicators rejected the prospectively declared
5% all-sample drift gate. The down-norm trial's descriptive incumbent/new
ratio was 0.760077; the attention-local trial's was 2.490943. Neither is an
accepted speed result. The exact protocols, all-position reports, failed
adjudications, and compact queue/JSON archives are alongside this file.
Completed queue jobs were not replayed or cherry-picked.

The original binary screen and timing readbacks remain in local `screens/`
and `samples/` directories, outside this Git snapshot. The compact archives
preserve queue receipts and JSON evidence without committing approximately
15 GiB of binary readbacks. The supplied Swift test harness is included as
packet source; no new Swift driver was authored. No production routing or
full-route timing was changed or qualified here.
