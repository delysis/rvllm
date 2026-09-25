#!/usr/bin/env bash
# Use the generated header from a SUCCESSFUL emulate.sh run of the same source.
# This checks whether five mathematically wrong shader mutants are detected.
# Compile failure is NOT a successful negative control. No native API is used.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
base=${1:?usage: mutation-check.sh POSITIVE_EMULATION_DIR ABS_NEW_OUTPUT}
out=${2:?usage: mutation-check.sh POSITIVE_EMULATION_DIR ABS_NEW_OUTPUT}
case "$out" in /*) ;; *) echo 'output must be absolute' >&2; exit 2;; esac
grep -q '^positive=32 negative=128 ' "$base/run.stdout"
perl "$here/mutations.pl" "$base/atlas_under_test.hpp" "$here/host_msl_emulation.cpp" "$out"
cxx=${CXX:-c++}
"$cxx" --version > "$out/compiler.txt"
for d in "$out"/*/; do
  "$cxx" -std=c++20 -O2 -ffp-contract=off -fno-fast-math -I "$d" \
    "$d/test.cpp" -o "$d/emulate" > "$d/build.stdout" 2> "$d/build.stderr"
  code=0
  "$d/emulate" > "$d/run.stdout" 2> "$d/run.stderr" || code=$?
  printf '%s\n' "$code" > "$d/exit-code.txt"
  test "$code" -eq 1
  grep -q '^FAIL: FP64 output error$' "$d/run.stderr"
  printf '%s: compiled; wrong result rejected\n' "$(basename -- "$d")"
done
