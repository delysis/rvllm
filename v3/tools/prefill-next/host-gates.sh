#!/bin/bash
# Host qualification, not Metal qualification. Run from the frozen worktree.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$here/../.."
python3 -m unittest discover -s tools -p test_gemma4_catalog.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/prefill-next/host_checks.py
cargo check --offline --locked -p rvllm-apple-metal --all-targets
cargo test --offline --locked -p rvllm-apple-metal --lib prefill_round::
cargo test --offline --locked -p rvllm-apple-metal --lib research_catalog::
cargo test --offline --locked -p rvllm-apple-metal --lib research_evidence::
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-prefill-next
cargo clippy --offline --locked -p rvllm-apple-metal --bin rvllm-prefill-next -- -D warnings
# Apple production routing type check remains a separate required gate.
