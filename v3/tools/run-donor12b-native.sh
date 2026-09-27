#!/bin/sh
# Run from v3 in the full, patched rvLLM checkout. No model or network access.
set -eu
[ "$(uname -s)" = Darwin ] || { echo 'Apple host required' >&2; exit 2; }
[ "$#" -eq 1 ] || { echo 'usage: tools/run-donor12b-native.sh ABSOLUTE_FRESH_REPORT_ROOT' >&2; exit 2; }
case "$1" in /*) ;; *) echo 'absolute report path required' >&2; exit 2;; esac
mkdir "$1"
root=$1
cargo test -p rvllm-apple-metal --lib donor12b -- --nocapture >"$root/portable.log" 2>&1
cargo build -p rvllm-apple-metal --bin rvllm-donor12b-source >"$root/build.log" 2>&1
for selector in metal-donor12b-sg8 metal-donor12b-sg4; do
    cargo run -p rvllm-apple-metal --bin rvllm-donor12b-source -- \
        build "$selector" "$root/$selector-build" >"$root/$selector-build.log" 2>&1
    RVLLM_METAL_GLOBAL_DECODE_CANDIDATE="$selector" \
    RVLLM_METAL_GLOBAL_DECODE_SOURCE="$root/$selector-build/core.metal" \
    RVLLM_METAL_GLOBAL_DECODE_METALLIB="$root/$selector-build/core.metallib" \
    RVLLM_METAL_GLOBAL_DECODE_BUILD_RECEIPT="$root/$selector-build/build.json" \
    RVLLM_METAL_GLOBAL_DECODE_REPORT_DIR="$root/$selector-oracle" \
    cargo test -p rvllm-apple-metal --lib \
        donor12b_device_tests::native_donor12b_operator_oracle -- \
        --ignored --exact --nocapture >"$root/$selector-oracle.log" 2>&1
    RVLLM_METAL_GLOBAL_DECODE_CANDIDATE="$selector" \
    RVLLM_METAL_GLOBAL_DECODE_SOURCE="$root/$selector-build/core.metal" \
    RVLLM_METAL_GLOBAL_DECODE_METALLIB="$root/$selector-build/core.metallib" \
    RVLLM_METAL_GLOBAL_DECODE_BUILD_RECEIPT="$root/$selector-build/build.json" \
    RVLLM_METAL_GLOBAL_DECODE_REPORT_DIR="$root/$selector-abba" \
    cargo test -p rvllm-apple-metal --lib \
        donor12b_device_tests::native_donor12b_projection_abba -- \
        --ignored --exact --nocapture >"$root/$selector-abba.log" 2>&1
done
