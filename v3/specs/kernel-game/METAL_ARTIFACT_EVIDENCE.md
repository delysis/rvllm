# Metal artifact evidence collector

`rvllm-metal-artifact-evidence` creates a strict companion artifact for a Metal
kernel-game receipt. It compiles one already-generated MSL file, captures public
compiler output, loads the resulting metallib on the current device, and queries
the requested PSOs through public Metal API getters.

Example for the current native-BF16 low-bit entry points:

```sh
tmp=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-metal-evidence.XXXXXX")
cargo run --locked -p rvllm-apple-metal --bin emit_metal_kernels -- \
  bf16 "$tmp/kernels.metal" "$tmp/pipelines.json"
cargo run --locked -p rvllm-apple-metal \
  --bin rvllm-metal-artifact-evidence -- \
  "$tmp/kernels.metal" "$tmp/evidence" \
  experimental_projection_w4abf16_bf16 \
  experimental_projection_w8abf16_bf16
```

The command fails if compilation, linking, metallib loading, or any named PSO
fails. Its `evidence.json` is usable with a timing receipt only after the source
and metallib hashes match that receipt and the JSON is retained by hash.

Verify the exact schema, reject duplicate/unknown fields, and re-hash every
referenced artifact while it remains available:

```sh
cargo run --locked -p rvllm-runtime \
  --bin rvllm_metal_artifact_evidence_verify -- \
  "$tmp/evidence/evidence.json"
```

The following fields are direct public evidence:

- source, AIR, metallib, collector, and Metal-tool hashes;
- compiler/tool versions, SDK, and command arguments;
- raw `metal-objdump` build tables and disassembly, each retained by hash;
- live device name, registry ID, GPU family/capabilities, and an identity hash;
- PSO execution width, maximum threads per threadgroup, and static threadgroup
  memory.

The device hash identifies this observation, not a permanent physical machine:
registry IDs can change across platform lifecycle events. Likewise, raw
disassembly is deliberately not interpreted as proof of executed machine
instructions. Apple's supported public interfaces do not provide a stable
register-count, register-residency, occupancy, SIMD-matrix-lowering, or
low-bit-unpack-lowering contract. Those claims therefore remain explicitly
`unavailable` or `unverified` in the report.

This evidence describes only the sealed rvLLM artifact. It neither imports MLX
traces nor makes correctness, dispatch, timing, or promotion claims.
