#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${RVLLM_GENERATED_TINY_HF_REFERENCE_DIR:-}"
KEEP_BUNDLE="${RVLLM_GENERATED_TINY_KEEP_BUNDLE:-0}"
REQUIRE_TORCH="${RVLLM_REQUIRE_TORCH:-0}"

if [[ -z "$BUNDLE_DIR" ]]; then
  BUNDLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rvllm-generated-tiny-reference.XXXXXX")"
  KEEP_BUNDLE="${RVLLM_GENERATED_TINY_KEEP_BUNDLE:-0}"
fi

cleanup() {
  if [[ "${KEEP_BUNDLE}" != "1" && "$BUNDLE_DIR" == "${TMPDIR:-/tmp}"/rvllm-generated-tiny-reference.* ]]; then
    rm -rf "$BUNDLE_DIR"
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"

echo "exporting generated tiny HF/Gemma-shaped reference bundle to $BUNDLE_DIR"
RVLLM_GENERATED_TINY_HF_REFERENCE_DIR="$BUNDLE_DIR" \
  cargo test -p rvllm-runtime --features apple generated_tiny_hf_reference_bundle_can_be_exported -- --nocapture

python3 scripts/verify_generated_tiny_reference.py "$BUNDLE_DIR"

if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
  python3 scripts/compare_generated_tiny_reference_torch.py "$BUNDLE_DIR"
elif [[ "$REQUIRE_TORCH" == "1" ]]; then
  echo "PyTorch is not installed and RVLLM_REQUIRE_TORCH=1" >&2
  exit 2
else
  echo "torch comparison skipped: PyTorch is not installed"
fi

if [[ "$KEEP_BUNDLE" == "1" ]]; then
  echo "kept generated tiny reference bundle at $BUNDLE_DIR"
fi
