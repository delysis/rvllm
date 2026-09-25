#!/usr/bin/env bash
# Format only the new module subtree and new binary. Does not walk the old lib.rs
# or format any other crate. Run once, review the diff, then run local-gate.sh.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
command -v rustfmt >/dev/null
rustfmt --edition 2021 \
  "$root/v3/crates/rvllm-apple-metal/src/attention_atlas/mod.rs" \
  "$root/v3/crates/rvllm-apple-metal/src/bin/rvllm-attention-atlas.rs"
