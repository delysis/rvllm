#!/bin/sh
set -eu

workspace_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_root=${1:-"$workspace_dir/apple/metallib"}
scratch=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-metallib.XXXXXX")
trap 'rm -rf "$scratch"' EXIT HUP INT TERM

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Metal library compilation requires macOS and Xcode" >&2
    exit 2
fi

# Xcode 26 distributes the compiler as an on-demand Metal.xctoolchain.
# SDK-qualified `xcrun metal` can otherwise resolve the placeholder in the
# default toolchain and fail even after the component has been installed.
if xcrun --toolchain Metal --find metal >/dev/null 2>&1; then
    use_metal_toolchain=1
else
    use_metal_toolchain=0
fi

run_metal_tool() {
    if [ "$use_metal_toolchain" -eq 1 ]; then
        xcrun --toolchain Metal "$@"
    else
        xcrun "$@"
    fi
}

mkdir -p "$output_root"
for dtype in f16 bf16; do
    source_file="$scratch/rvllm-$dtype.metal"
    manifest_file="$scratch/rvllm-$dtype.json"
    cargo run --quiet --manifest-path "$workspace_dir/Cargo.toml" \
        -p rvllm-apple-metal --bin emit_metal_kernels -- \
        "$dtype" "$source_file" "$manifest_file"

    for sdk in macosx iphoneos iphonesimulator; do
        case "$sdk" in
            macosx) platform=macos ;;
            iphoneos) platform=ios ;;
            iphonesimulator) platform=ios-simulator ;;
        esac
        destination="$output_root/$platform/$dtype"
        mkdir -p "$destination"
        run_metal_tool -sdk "$sdk" metal -std=metal3.1 -c "$source_file" \
            -o "$scratch/$platform-$dtype.air"
        run_metal_tool -sdk "$sdk" metallib "$scratch/$platform-$dtype.air" \
            -o "$destination/rvllm.metallib"
        cp "$manifest_file" "$destination/pipelines.json"
        shasum -a 256 "$destination/rvllm.metallib" \
            > "$destination/rvllm.metallib.sha256"
    done
done

echo "Apple metallibs written to $output_root"
