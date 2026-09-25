#!/usr/bin/env bash
# LOCAL OWNER ONLY. Rust host/build checks, never a device trial or queue action.
# Fresh logs; no automatic retries; no unrelated whole-workspace formatting.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
out=${1:?usage: local-gate.sh ABS_NEW_LOG_DIR}
case "$out" in /*) ;; *) echo 'output must be absolute' >&2; exit 2;; esac
mkdir -- "$out"
trap 'code=$?; printf "%s\n" "$code" > "$out/exit-code.txt"; exit "$code"' EXIT
command -v rustfmt > "$out/rustfmt-path.txt"
command -v cargo > "$out/cargo-path.txt"
rustfmt --version > "$out/rustfmt-version.txt"
cargo --version > "$out/cargo-version.txt"
rustc -vV > "$out/rustc-version.txt"
git -C "$root" rev-parse HEAD > "$out/source-head.txt"
git -C "$root" diff --binary HEAD > "$out/source-diff.patch"
git -C "$root" status --porcelain=v1 > "$out/source-status.txt"
# The diff alone omits untracked new sources. Record their bytes explicitly.
(
  cd -- "$root"
  for src in v3/crates/rvllm-apple-metal/src/attention_atlas/*.rs \
    v3/crates/rvllm-apple-metal/src/attention_atlas/shaders/*.metal \
    v3/crates/rvllm-apple-metal/src/bin/rvllm-attention-atlas.rs \
    v3/crates/rvllm-apple-metal/src/lib.rs v3/crates/rvllm-apple-metal/Cargo.toml v3/Cargo.lock; do
    shasum -a 256 "$src"
  done
) > "$out/owned-sources.sha256"
rustfmt --check --edition 2021 \
  "$root/v3/crates/rvllm-apple-metal/src/attention_atlas/mod.rs" \
  "$root/v3/crates/rvllm-apple-metal/src/bin/rvllm-attention-atlas.rs" \
  > "$out/fmt.stdout" 2> "$out/fmt.stderr"
cd -- "$root/v3"
cargo test --offline --locked -p rvllm-apple-metal --features attention-atlas-research \
  --lib attention_atlas::tests -- --list > "$out/list.stdout" 2> "$out/list.stderr"
count=$(grep -c '^attention_atlas::tests::.*: test$' "$out/list.stdout" || true)
if [[ "$count" != 21 ]]; then echo "Expected 21 named atlas tests, found $count" >&2; exit 3; fi
cargo test --offline --locked -p rvllm-apple-metal --features attention-atlas-research \
  --lib attention_atlas::tests -- --test-threads=1 > "$out/tests.stdout" 2> "$out/tests.stderr"
grep -q '21 passed; 0 failed; 0 ignored;' "$out/tests.stdout"
cargo check --offline --locked -p rvllm-apple-metal --no-default-features \
  > "$out/default-off.stdout" 2> "$out/default-off.stderr"
cargo build --offline --locked --release -p rvllm-apple-metal \
  --features attention-atlas-research --bin rvllm-attention-atlas \
  > "$out/build.stdout" 2> "$out/build.stderr"
printf 'Rust component gate passed; NO native Metal, ANE, model or timing qualification.\n' > "$out/result.txt"
