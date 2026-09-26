# MLX Gemma 4 stage smoke

This is the first real-weight execution of the pinned stage microbenchmark.
It measures the Q4 group-64 model at the 256-token prefill and one-token decode
shapes (22 isolated operator cases, each using the upstream MLX five-warmup and
100 synchronized-iteration protocol).

The job is `exploratory_timing`: host state and process overlap are retained,
but the result cannot qualify promotion. A successful smoke establishes only
that actual model modules can execute and emit a strict receipt before the
complete 4/8/16-bit, 256--4096-token matrix is admitted.
