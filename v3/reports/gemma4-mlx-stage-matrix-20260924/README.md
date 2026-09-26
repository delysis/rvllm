# MLX Gemma 4 stage matrix

This packet contains the remaining fourteen bounded jobs in the MLX stage
matrix. Together with the Q4/256 smoke job, they cover BF16, affine Q8 g64,
and affine Q4 g64 at 256, 512, 1024, 2048, and 4096 tokens for both prefill
and one-token decode stage shapes.

Each job executes 22 isolated operator cases with the pinned core-MLX timing
loop: five warmups followed by 100 `mx.eval`-synchronized iterations. Jobs are
single-chained to prevent concurrent model residency. The first remaining job
depends on successful completion of `mlx-gemma4-q4-stage-pp256-smoke-20260924-v1`;
therefore the full matrix is not admitted until the smoke proves that actual
Q4 modules execute and the strict receipt is parseable.

All jobs are exploratory. Their receipts preserve sampled host conditions and
process overlap, but neither isolated timings nor this packet qualify kernel
promotion. Normal-route traces and correctness evidence remain separate gates.
