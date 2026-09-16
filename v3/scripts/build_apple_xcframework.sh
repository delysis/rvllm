#!/bin/sh
set -eu

workspace_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_dir"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Apple XCFramework packaging requires a macOS builder" >&2
    exit 2
fi
if [ "$#" -ne 1 ]; then
    echo "usage: $0 OUTPUT.xcframework" >&2
    exit 2
fi

output=$1
if [ -e "$output" ]; then
    echo "refusing to replace existing XCFramework: $output" >&2
    exit 2
fi

installed=$(rustup target list --installed)
rustc_bin=$(rustup which rustc)
cargo_bin=$(rustup which cargo)
for target in aarch64-apple-ios aarch64-apple-ios-sim; do
    if ! printf '%s\n' "$installed" | grep -Fxq "$target"; then
        echo "required Rust target is not installed: $target" >&2
        exit 2
    fi
done

if [ "${RVLLM_APPLE_SKIP_STATICLIB_BUILD:-0}" != "1" ]; then
    RUSTC="$rustc_bin" "$cargo_bin" build --release -p rvllm-apple-ffi --lib
    RUSTC="$rustc_bin" "$cargo_bin" build --release -p rvllm-apple-ffi --lib --target aarch64-apple-ios
    RUSTC="$rustc_bin" "$cargo_bin" build --release -p rvllm-apple-ffi --lib --target aarch64-apple-ios-sim
fi

header_dir=apple/AppleInference/Sources/CRvllmApple/include
for archive in \
    target/release/librvllm_apple_ffi.a \
    target/aarch64-apple-ios/release/librvllm_apple_ffi.a \
    target/aarch64-apple-ios-sim/release/librvllm_apple_ffi.a
do
    if [ ! -f "$archive" ]; then
        echo "missing static library: $archive" >&2
        exit 2
    fi
done

parent=$(dirname -- "$output")
mkdir -p "$parent"
staging=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-apple-xcframework.XXXXXX")
trap 'rm -rf "$staging"' EXIT HUP INT TERM
staged="$staging/RvllmApple.xcframework"

xcodebuild -create-xcframework \
    -library target/release/librvllm_apple_ffi.a -headers "$header_dir" \
    -library target/aarch64-apple-ios/release/librvllm_apple_ffi.a -headers "$header_dir" \
    -library target/aarch64-apple-ios-sim/release/librvllm_apple_ffi.a -headers "$header_dir" \
    -output "$staged"

# Scan the package as a directory so every slice must remain public-API-only.
python3 tools/check_apple_release_symbols.py "$staged"
mv "$staged" "$output"
echo "created Apple XCFramework: $output"
