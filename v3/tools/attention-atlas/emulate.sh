#!/usr/bin/env bash
# Optional local HOST algebra/concurrency check; not native compilation/testing.
# Caller supplies a FRESH output directory. No model, queue or device access.
set -euo pipefail
if [[ $(uname -s) != Linux ]]; then echo "This optional ucontext harness is Linux-only; use native owner gates on macOS." >&2; exit 2; fi
here=$(cd -- "$(dirname -- "$0")" && pwd)
out=${1:?usage: emulate.sh ABS_NEW_OUTPUT}
case "$out" in /*) ;; *) echo 'output must be absolute' >&2; exit 2;; esac
mkdir -- "$out"
# The transformations are deliberately narrow and recorded. The matrix shim uses
# whole matrices per host fiber, NOT the Apple SIMD fragment representation.
# Perl is only fixture preparation; no Python is part of the delivered suite.
{
  perl -0777 -pe 's/\[\[[^\]]*\]\]//g; s/threadgroup float maxima\[32\], den\[32\], weights\[32\]/static float maxima[32], den[32], weights[32]/' \
    "$here/../../crates/rvllm-apple-metal/src/attention_atlas/shaders/common.metal"
  cat "$here/../../crates/rvllm-apple-metal/src/attention_atlas/shaders/matrix.metal"
} > "$out/atlas_under_test.hpp"
cxx=${CXX:-c++}
"$cxx" --version > "$out/compiler.txt"
"$cxx" -std=c++20 -O2 -ffp-contract=off -fno-fast-math -I "$out" \
  "$here/host_msl_emulation.cpp" -o "$out/emulate" > "$out/build.stdout" 2> "$out/build.stderr"
"$out/emulate" > "$out/run.stdout" 2> "$out/run.stderr"
