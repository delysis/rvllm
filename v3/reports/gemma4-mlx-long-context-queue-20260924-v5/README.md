# Gemma 4 MLX long-context queue v5

This queue supersedes, but does not overwrite, v1 through v4.

The matrix covers Gemma 4 12B BF16, affine Q8 group-64, and affine Q4
group-64 at 256, 512, 1024, 2048, and 4096 prompt tokens with 64 decode
tokens and seven trials per cell. Model, runner, and source pins are unchanged
from v3.

V5 is the continuously operating exploratory matrix requested for a real
MacBook host. It admits every known macOS thermal state, Low Power Mode state,
and power mode while retaining those controls in the recorded comparison
stratum. It separates blocking activity from observed activity: Cargo, Rust,
llama, and rvLLM processes are recorded in each condition sample but do not
prevent or invalidate an exploratory cell. AC power, observer freshness,
non-restricted CPU controls, and disk headroom remain enforced.

V3 preserved two completed cells as condition-rejected evidence after unrelated
builds started mid-run. V4 introduced non-blocking process observation but did
not launch a trial because its manifests still pinned Low Power Mode and the
macOS power mode. Both histories remain intact.

These runs are exploratory measurements, not promotion evidence by themselves.
Analysis must retain every per-trial value, report process overlap and the full
power/thermal stratum, use robust summaries instead of selecting favorable
runs, and require independently qualified correctness plus controlled
confirmation before promoting a kernel.

The resident referee executable admitted for this queue has SHA-256
`046d4c8b49add5b3a509ce12e52d09efba21b3c144ac38c6327b225d46ff1d97`.
