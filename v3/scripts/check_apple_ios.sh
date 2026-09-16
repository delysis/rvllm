#!/bin/sh
set -eu

workspace_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_dir"

targets=${RVLLM_IOS_TARGETS:-"aarch64-apple-ios aarch64-apple-ios-sim"}
installed=$(rustup target list --installed)
rustc_bin=$(rustup which rustc)
cargo_bin=$(rustup which cargo)
checked=0
missing=0

for target in $targets; do
    if printf '%s\n' "$installed" | grep -Fxq "$target"; then
        echo "checking public Core ML runtime adapter for $target"
        RUSTC="$rustc_bin" "$cargo_bin" check -p rvllm-apple-coreml-runtime --lib --target "$target"
        echo "checking rvllm-apple-metal for $target"
        RUSTC="$rustc_bin" "$cargo_bin" check -p rvllm-apple-metal --lib --target "$target"
        echo "checking rvllm-runtime Apple library for $target"
        RUSTC="$rustc_bin" "$cargo_bin" check -p rvllm-runtime --features apple --lib --target "$target"
        echo "checking rvllm-apple-ffi static library for $target"
        RUSTC="$rustc_bin" "$cargo_bin" check -p rvllm-apple-ffi --lib --target "$target"
        if [ "${RVLLM_BUILD_IOS_STATICLIBS:-0}" = "1" ]; then
            echo "building shipping rvllm-apple-ffi static library for $target"
            RUSTC="$rustc_bin" "$cargo_bin" build --release -p rvllm-apple-ffi --lib --target "$target"
        fi
        checked=$((checked + 1))
    else
        echo "iOS Rust target is not installed: $target"
        missing=$((missing + 1))
    fi
done

if [ "$checked" -eq 0 ] || { [ "${RVLLM_REQUIRE_IOS_TARGETS:-0}" = "1" ] && [ "$missing" -ne 0 ]; }; then
    echo "install cross-check targets with: rustup target add aarch64-apple-ios aarch64-apple-ios-sim"
    if [ "${RVLLM_REQUIRE_IOS_TARGETS:-0}" = "1" ]; then
        exit 2
    fi
fi
