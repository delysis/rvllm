#!/bin/sh
set -eu

workspace_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_dir"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Apple shipping release checks require a macOS builder" >&2
    exit 2
fi

# The feature list is intentionally explicit. In particular, neither
# `private-ane` nor `macos-private-ane-research` may enter this build.
cargo build --release \
    -p rvllm-apple \
    -p rvllm-apple-metal \
    -p rvllm-apple-coreml-sys \
    -p rvllm-apple-coreml-runtime
cargo build --release -p rvllm-runtime --features apple --lib
cargo build --release -p rvllm-runtime --features apple --bin rvllm_metal_infer
cargo build --release -p rvllm-serve --features apple --bin rvllm-server
cargo build --release -p rvllm-apple-ffi

# `cargo check` cannot prove that the final iOS static libraries link or that
# their dependency closure is free of private symbols. Build and scan the
# actual device and simulator archives that are eligible for an XCFramework.
RVLLM_REQUIRE_IOS_TARGETS=1 RVLLM_BUILD_IOS_STATICLIBS=1 scripts/check_apple_ios.sh

# Build the public Swift host boundary for every shipping platform so its
# descriptor-protection shim is covered by the same private-symbol policy.
(
    cd apple/AppleInference
    swift build -c release
    swift build -c release \
        --triple arm64-apple-ios18.0 \
        --sdk "$(xcrun --sdk iphoneos --show-sdk-path)"
    swift build -c release \
        --triple arm64-apple-ios18.0-simulator \
        --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)"
)

python3 -m unittest tools/test_check_apple_release_symbols.py
python3 tools/check_apple_release_symbols.py \
    target/release/librvllm_apple.rlib \
    target/release/librvllm_apple_metal.rlib \
    target/release/librvllm_apple_coreml_sys.rlib \
    target/release/librvllm_apple_coreml_runtime.rlib \
    target/release/librvllm_runtime.rlib \
    target/release/librvllm_apple_ffi.a \
    target/aarch64-apple-ios/release/librvllm_apple_ffi.a \
    target/aarch64-apple-ios-sim/release/librvllm_apple_ffi.a \
    target/release/rvllm_metal_infer \
    target/release/rvllm-server \
    apple/AppleInference/.build/arm64-apple-macosx/release/CPersistentCacheHost.build/persistent_cache_host.c.o \
    apple/AppleInference/.build/arm64-apple-ios/release/CPersistentCacheHost.build/persistent_cache_host.c.o \
    apple/AppleInference/.build/arm64-apple-ios-simulator/release/CPersistentCacheHost.build/persistent_cache_host.c.o
