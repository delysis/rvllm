# Gemma 4 MLX long-context queue v2

This is the immutable replacement for the stopped v1 queue. The v1 first job
is intentionally preserved as failed: Homebrew `mlx-lm` 0.30.0 does not
recognize the checkpoint's `gemma4_unified` model type. No v1 job was replayed.

The v2 jobs run the previously audited MLX-LM source commit
`87b7b583a697537aa68f47130b40884700b5f55f` through a compatible local MLX
runtime. A one-token diagnostic completed before submission. Each job pins the
Python executable, the MLX-LM benchmark/model/loader sources, and the selected
model config and tensor index by SHA-256.

The matrix covers BF16, affine Q8 group-size 64, and affine Q4 group-size 64 at
256, 512, 1024, 2048, and 4096 prompt tokens. Every process performs seven
trials after the benchmark's own warmup and generates 64 tokens so decode is a
measurable interval rather than a one-token timer artifact. Jobs form one
predeclared dependency chain. The resident queue admits any observed thermal
state but records it for local stratification; it still requires AC power, low
power mode, power mode 1, at least 16 GiB free, and a five-second quiet window.

These are framework reference measurements, not rvLLM candidate promotion
evidence. Cross-framework conclusions require matching representation and
workload semantics and must report those remaining differences.
