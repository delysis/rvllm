# Gemma 4 MLX long-context queue v4

This queue supersedes, but does not overwrite, v1 through v3.

The matrix covers Gemma 4 12B BF16, affine Q8 group-64, and affine Q4
group-64 at 256, 512, 1024, 2048, and 4096 prompt tokens with 64 decode
tokens and seven trials per cell. Model, runner, and source pins are unchanged
from v3.

V3 correctly preserved two completed cells as rejected because unrelated
Cargo/Rust work began after their clean launch gates. That policy was too
strict for the requested continuously operating exploratory campaign: a long
cell could finish successfully yet become terminally unusable, with no retry,
whenever another local build started.

V4 separates an activity **gate** from activity **observation**. Its jobs do
not require named processes to remain absent. They record Cargo, Rust, llama,
and rvLLM activity in every condition sample as `observed_processes`, while
power source, Low Power Mode, power mode, thermal state, disk headroom, and
observer freshness remain enforced and recorded. Thermal state is unpinned,
so every known state is admissible and becomes part of the comparison stratum.

These runs are exploratory measurements, not promotion evidence by themselves.
Analysis must retain per-trial values, report observed-process overlap, compare
only like strata or model the recorded covariates, and require independently
qualified correctness before making a kernel promotion claim.
