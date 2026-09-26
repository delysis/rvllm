# Gemma 4 MLX long-context queue v6

This queue supersedes, but does not overwrite, v1 through v5.

The matrix covers Gemma 4 12B BF16, affine Q8 group-64, and affine Q4
group-64 at 256, 512, 1024, 2048, and 4096 prompt tokens with 64 decode
tokens and seven trials per cell. Model, runner, and source pins are unchanged
from v3.

V6 uses the explicit `exploratory_timing` purpose. Successful pinned commands
remain successful exploratory evidence even when the generic phase monitor
cannot establish one stable comparison stratum for the entire cell. The result
still records `sampled_conditions_eligible`, every violation, all power samples,
and observed Cargo/Rust/llama/rvLLM processes. Strict `timing` jobs retain their
fail-closed condition rejection semantics for later confirmation and promotion.

This distinction is intentional: daytime exploration should keep generating
data under real host variance, while final claims still require controlled,
like-stratum, independently confirmed runs. Exploratory success is never a
promotion claim and must not be silently relabeled as controlled timing.

The analysis must retain every trial, report process-overlap counts and all
observed strata, use robust summaries, and never select favorable trials.

The queue referee used for this run is the release executable with SHA-256
`300f2b02e2bb8078ab53b95b8b9b7d79560bb11706f5d027a8b2e016cc627ed7`.
