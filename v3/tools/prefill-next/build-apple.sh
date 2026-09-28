#!/bin/bash
# Run only in a frozen exact-base-plus-patch worktree. Preparation is not timing.
set -euo pipefail
if [[ $# != 2 || "$1" != /* || "$2" != /* ]]; then
  echo 'usage: build-apple.sh ABS_REFEREE_BINARY ABS_NEW_ARTIFACT_DIRECTORY' >&2
  exit 2
fi
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || { echo 'Apple Silicon macOS required' >&2; exit 2; }
referee=$1
out=$2
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ -x "$referee" && ! -e "$out" ]] || { echo 'Need an existing referee and a new output directory' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$out")"
"$referee" source-all "$out"
{
  /usr/bin/sw_vers
  /usr/bin/xcodebuild -version
  /usr/bin/xcrun --sdk macosx --show-sdk-path
  /usr/bin/xcrun --sdk macosx metal --version
  /usr/bin/xcrun swiftc --version
  /usr/bin/shasum -a 256 "$referee" "$here/MetalArm.swift"
} > "$out/toolchain.txt" 2>&1
/usr/bin/xcrun swiftc -O -target arm64-apple-macos14.0 -framework Metal \
  "$here/MetalArm.swift" -o "$out/MetalArm" > "$out/swift-build.log" 2>&1
for arm in normal mma32 metal-mma32-load4 simd-attention \
  metal-prefill-load4-control metal-prefill-pipeline32x64 \
  metal-prefill-q4k16 metal-prefill-pipeline32x64-q4k16 \
  metal-prefill-wide64 metal-prefill-wide128 metal-prefill-mma8k32 metal-prefill-wide64-mma8k32; do
  /usr/bin/xcrun --sdk macosx metal -std=metal3.1 -fno-fast-math \
    -c "$out/$arm.metal" -o "$out/$arm.air" > "$out/$arm.compile.log" 2>&1
  /usr/bin/xcrun --sdk macosx metallib "$out/$arm.air" \
    -o "$out/$arm.metallib" > "$out/$arm.link.log" 2>&1
done
(cd "$out"; /usr/bin/shasum -a 256 -- *.metal *.air *.metallib *.source.json MetalArm toolchain.txt > BUILD-SHA256SUMS)
echo "Strict libraries and operator driver built at $out; device correctness is still unrun."
