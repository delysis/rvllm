# Apple model package

`rvllm_apple_package` assembles an immutable, self-contained model directory for
the macOS and iOS runtime. It copies assets from a Hugging Face snapshot; it
never edits, moves, or replaces the source model.

## Build

First compile the six shipping Metal variants (macOS, iOS device, and iOS
simulator, each in F16 and BF16):

```sh
./scripts/build_apple_metallibs.sh /path/to/precompiled-metallibs
```

Then assemble and validate a new package:

```sh
cargo run --release -p rvllm-apple --features package-builder \
  --bin rvllm_apple_package -- build \
  --model-dir /path/to/hugging-face-snapshot \
  --metallib-root /path/to/precompiled-metallibs \
  --output /path/to/MyModel.rvllm \
  --package-id publisher.model.version \
  --weight-format auto
```

To produce an authenticated hybrid package while preserving every native F16
weight, explicitly name one or more dense MLP down projections:

```sh
cargo run --release -p rvllm-apple --features package-builder \
  --bin rvllm_apple_package -- build \
  --model-dir /path/to/f16-hugging-face-snapshot \
  --metallib-root /path/to/precompiled-metallibs \
  --output /path/to/MyHybridModel.rvllm \
  --package-id publisher.model.version \
  --weight-format f16 \
  --low-bit-down-proj \
    model.layers.0.mlp.down_proj.weight=w4a16-group32 \
  --low-bit-down-proj \
    model.layers.1.mlp.down_proj.weight=w8a16-group32
```

This is a tensor-level sidecar export, not a model-wide W4/W8 package. Schema
v3 retains and authenticates the native F16 shard as well as the packed values
and FP16 scales. The runtime defaults to hybrid residency. Its internal
native-replacement policy can omit all selected native projections from the
Metal arena after deterministic whole-set validation, but is not an iOS
default and remains gated on model-level numerical, quality, memory, thermal,
and performance evidence.

The output path must not exist. Assembly occurs in a sibling staging directory,
the completed package is fully re-opened and verified, and only then is it
installed with an atomic rename. A failed build removes its staging directory
and leaves the source model and destination untouched.

Validate an installed or bundled package independently:

```sh
cargo run --release -p rvllm-apple --features package-builder \
  --bin rvllm_apple_package -- \
  validate /path/to/MyModel.rvllm
```

## Accepted input

- `config.json` must be valid JSON with a non-empty `architectures[0]` and
  `model_type` (at the root or under `text_config`).
- `tokenizer.json` is required. Recognized adjacent tokenizer and processor
  metadata is copied when present.
- Weights must be exactly one `model.safetensors`, or an indexed shard set
  described by `model.safetensors.index.json`. A monolith plus an index,
  unindexed extra shards, missing shards, duplicate tensors, unsafe shard paths,
  or an index that does not exactly match tensor placement is rejected.
- Native HF package assembly accepts a uniform F16 or BF16 tensor set. It does
  not relabel ordinary tensors as W4A16/W8A16; those formats require the
  dedicated quantizing exporter and their independent quality gates. The
  initial exporter accepts explicitly named, two-dimensional, dense
  `.mlp.down_proj.weight` tensors from uniform F16 checkpoints only.
- The metallib root must contain `rvllm.metallib` and `pipelines.json` for all
  six platform/dtype variants. Pipeline manifests must declare the supported
  schema, exact dtype, 32-token KV page ABI, and a consistent kernel set.

## Integrity contract

Schema v2 records and verifies byte length plus SHA-256 for:

- `config.json` and optional model metadata;
- the safetensors index and every weight shard;
- tokenizer configuration and tokenizer data;
- every precompiled Metal library and pipeline manifest.

Schema v3 adds exact tensor-level low-bit descriptors and authenticates their
packed-value and little-endian FP16-scale files. It fixes the role to a dense
down projection, activation type to F16, ABI version to 1, and group size to
32. Package opening checks the exact shape-derived file sizes and validates the
integer domain, tail padding, scales, and zero-scale groups with bounded
row-sized reads. Schema v2 is forbidden from declaring these sidecars.

The model, tokenizer, chat-template, and numerical-ABI fingerprints are derived
from those authenticated assets and recomputed when the package opens. They are
not caller-provided trust claims. Asset paths are package-relative, duplicate or
traversing paths are rejected, and package assets or parent components may not
be symlinks. Weight checksums are streamed in bounded memory so validation does
not duplicate a multi-gigabyte model in RAM.

The manifest is `rvllm-apple-model.json`. Schema v1 is intentionally rejected:
it did not authenticate `config.json` and therefore cannot safely establish a
model or prompt-cache identity.
