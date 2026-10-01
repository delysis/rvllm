#!/bin/bash
# This script is an acceptance procedure, NOT a record that the gates passed.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
v3=$(cd -- "$here/../.." && pwd)
cd -- "$v3"
python3 tools/prefill-round/host_checks.py
(cd tools; python3 -m unittest -v test_gemma4_catalog test_gemma4_candidate_ci)
cargo test --offline --locked -p rvllm-apple-metal --lib prefill_round
cargo test --offline --locked -p rvllm-apple-metal --lib research_catalog
cargo test --offline --locked -p rvllm-apple-metal --lib research_evidence
cargo test --offline --locked -p rvllm-apple-metal --bin rvllm-prefill-round
cargo check --offline --locked -p rvllm-apple-metal --all-targets
cargo build --offline --locked --release -p rvllm-apple-metal --bin rvllm-prefill-round
if [[ $(uname -s) == Darwin ]]; then
  cargo check --offline --locked -p rvllm-runtime --features apple --lib
fi
