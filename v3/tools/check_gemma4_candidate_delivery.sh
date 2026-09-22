#!/usr/bin/env bash
# Build/test/export only. Never launch inference, a device fixture, or a queue.
# The output libraries are NOT a frozen experiment or numerical qualification.
set -euo pipefail
umask 077
if [[ $# -ne 2 || "$1" != /* || "$2" != /* ]]; then
    echo 'usage: bash check_gemma4_candidate_delivery.sh /absolute/NEW-output /absolute/cargo-target' >&2
    exit 2
fi
workspace="$(cd "$(dirname "$0")/.." && pwd -P)"
[[ -f "$workspace/Cargo.toml" ]] || { echo 'full v3 workspace required' >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
    echo 'requires the native arm64 macOS build host; no portable pass substitution' >&2
    exit 2
}
# mkdir is exclusive: failed/partial output directories must never be reused.
mkdir "$1"
out="$(cd "$1" && pwd -P)"
export CARGO_TARGET_DIR="$2"
export CARGO_NET_OFFLINE=true
printf 'incomplete\n' > "$out/status.txt"
cd "$workspace"

run() {
    local label="$1"
    shift
    printf '%q ' "$@" >> "$out/commands.txt"
    printf '\n' >> "$out/commands.txt"
    local code=0
    "$@" > "$out/$label.stdout" 2> "$out/$label.stderr" || code=$?
    printf '%s\t%d\n' "$label" "$code" >> "$out/exit-codes.tsv"
    if [[ $code -ne 0 ]]; then
        cat "$out/$label.stderr" >&2
        echo "failed: $label; all outputs retained in $out" >&2
        return "$code"
    fi
}

host_tests() {
    local label="$1"
    shift
    run "$label" "$@"
    # A successful zero-test filter is not coverage. Cargo's exit status also
    # has to succeed; this check is additional, never a substitute for it.
    if ! grep -Eq '^test result: ok\. [1-9][0-9]* passed;' "$out/$label.stdout"; then
        echo "no passing tests recorded for $label" >&2
        return 1
    fi
}

run cargo-version cargo --version
run rustc-version rustc --version
run rustfmt-version rustfmt --version
run metal-path xcrun --toolchain Metal --find metal
run format cargo fmt --all -- --check
common=(--offline --locked --release -j 2 --target aarch64-apple-darwin)
host_tests metal-policy cargo test "${common[@]}" -p rvllm-apple-metal --lib research::tests
host_tests dispatch-evidence cargo test "${common[@]}" -p rvllm-apple-metal --lib research_evidence::tests
host_tests ane-candidates cargo test "${common[@]}" -p rvllm-apple \
    --features macos-private-ane-research --lib ane_int8_candidates::tests
host_tests kv-layout cargo test "${common[@]}" -p rvllm-apple \
    --features macos-private-ane-research --lib ane_attention_layout::tests::blocked32_
host_tests prefill-screen cargo test "${common[@]}" -p rvllm-runtime \
    --features macos-private-ane-research --bin rvllm_disaggregated_infer prefill_screen::tests
run cli-build cargo build "${common[@]}" -p rvllm-runtime \
    --features macos-private-ane-research --bin rvllm_disaggregated_infer
run exporter-build cargo build "${common[@]}" -p rvllm-apple-metal --bin rvllm-metal-research-source
exporter="$CARGO_TARGET_DIR/aarch64-apple-darwin/release/rvllm-metal-research-source"
[[ -x "$exporter" ]] || { echo "missing built source exporter: $exporter" >&2; exit 1; }

for dtype in bf16 f16; do
    for candidate in off metal-short-mma16x64 metal-rounded-gate32 metal-gqa-kv8; do
        stem="$dtype-$candidate"
        run "$stem-export" "$exporter" "$dtype" "$candidate"
        cp "$out/$stem-export.stdout" "$out/$stem.metal"
        [[ -s "$out/$stem.metal" ]] || { echo "empty source: $stem" >&2; exit 1; }
        run "$stem-compile" xcrun --toolchain Metal -sdk macosx metal \
            -std=metal3.1 -c "$out/$stem.metal" -o "$out/$stem.air"
        run "$stem-link" xcrun --toolchain Metal -sdk macosx metallib \
            "$out/$stem.air" -o "$out/$stem.metallib"
        [[ -s "$out/$stem.metallib" ]] || { echo "empty library: $stem" >&2; exit 1; }
    done
done
run artifact-hashes shasum -a 256 "$exporter" \
    "$CARGO_TARGET_DIR/aarch64-apple-darwin/release/rvllm_disaggregated_infer" \
    "$out"/*.metal "$out"/*.metallib
cp "$out/artifact-hashes.stdout" "$out/SHA256SUMS"
printf 'compiled-only; no accelerator acceptance\n' > "$out/status.txt"
