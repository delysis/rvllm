# Official Gemma 4 12B QAT artifacts and ANE viability

2026-09-15. Source and remote metadata only. No model payloads were downloaded, no hardware/builds were run, and no inference code changed. This report does not reopen the closed grouped-LUT4 prototype.

**Decision:** Google publishes official, already-packed four-bit Gemma 4 12B QAT artifacts. Their provenance satisfies the requested starting-point constraint. A faithful ANE representation is plausible, but the required combination of Apple-documented benefit and corroborating laptop performance is **not established for Google's group-32 QAT representation**. Keep four-bit out of the production path pending that evidence. The earlier posthoc scalar-LUT4 component result is not qualification of these weights or this layout.

## 1. Official artifacts verified

Google's [QAT announcement](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/) distinguishes training with simulated quantization from PTQ, and publishes Q4_0 GGUF, compressed-tensors, and unquantized QAT checkpoints. The [Google model card](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) identifies the latter as half-precision output of the QAT pipeline. The specialized mobile wNa8o8 variants are for E2B/E4B; their static activations, two-bit layers and KV scheme must not be attributed to 12B.

| Official repository and frozen revision | Exact weight files and metadata sizes | Interpretation |
| --- | --- | --- |
| [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/tree/29d097773436b69ff9feafd636ab4cf873786537), `29d097773436b69ff9feafd636ab4cf873786537` | `gemma-4-12b-it-qat-q4_0.gguf`: 6,975,879,296 bytes; `mmproj-gemma-4-12b-it-qat-q4_0.gguf`: 175,115,616 bytes | Google-packaged QAT Q4_0 target; separate multimodal projector. Do not assume every tensor inside the target is Q4_0 without inventorying its metadata. |
| [google/gemma-4-12B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/tree/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee), `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee` | `model.safetensors`: 10,264,229,896 bytes; `config.json`; `recipe.yaml`; tokenizer, processor and template files | Already packed four-bit weights with floating tensors retained where excluded. Strong candidate for a direct safetensors importer without fitting another quantizer. |
| [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized/tree/b6ed86275a6a5735884e208bfed95b445a684ca2), `b6ed86275a6a5735884e208bfed95b445a684ca2` | `model.safetensors`: 23,919,549,408 bytes; config, tokenizer, processor and template files | QAT-derived weights stored as **BF16**, not a native packed INT4 checkpoint. The config has no quantization configuration. Do not run the old BF16-to-LUT4 quantizer on it. |

The Hugging Face metadata API returned `private=false`, `gated=false`, and `license=apache-2.0` for all three repositories without authentication. Google's license link resolves to [Apache License 2.0](https://ai.google.dev/gemma/apache_2); do not carry over Gemma 3's license/gating assumptions. Files and sizes above came from publisher metadata, not local downloads or compiled-artifact measurements. API sources: [GGUF](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-q4_0-gguf?blobs=true), [CT](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-w4a16-ct?blobs=true), [unquantized](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-q4_0-unquantized?blobs=true).

Publisher-listed SHA-256 identifiers for a future integrity check:

```text
GGUF target: 93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b
GGUF mmproj: cb018338a7538a9814d994bfe54644c71eb7ed54e31eae2f721e45fd3c260da7
CT weights:  60b6e3989502969d8ae04185d72ecbbc7db63978d5af747a493d53895aa6bfa3
QAT BF16:    26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0
```

These are not locally verified payload hashes. Third-party MLX/GGUF repositories, altered models, and converted “QAT” labels are not substitutes for these Google pins. The GGUF pin's commit records a corrected vocabulary; freeze tokenizer/template provenance as well as weights.

## 2. Preserve the published quantization, not merely four-bit width

The CT [pinned config](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/resolve/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee/config.json) specifies `pack-quantized`, integer weights, **4 bits, symmetric, group_size=32, strategy=group**, `actorder=null`, `dynamic=false`; input/output activation schemes and KV quantization are null. Its [recipe](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/resolve/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee/recipe.yaml) excludes the language head, embeddings and modality modules. Do not invoke its observer to refit scales: import stored codes and scales.

In compressed-tensors pin `698967f567bb22713a7c24df08c3df7a3be18284`, the [compressor](https://github.com/vllm-project/compressed-tensors/blob/698967f567bb22713a7c24df08c3df7a3be18284/src/compressed_tensors/compressors/pack_quantized/base.py) stores `weight_packed`, `weight_scale`, `weight_shape`; symmetric zero points are omitted. The [packing helper](https://github.com/vllm-project/compressed-tensors/blob/698967f567bb22713a7c24df08c3df7a3be18284/src/compressed_tensors/compressors/pack_quantized/helpers.py) offsets signed four-bit values by 8 and packs consecutive values into successive low-to-high nibbles of INT32 words. For matrix `[O,K]`, groups run along K: `W[o,k] = scale[o,k/32] * q[o,k]`, with signed q in `[-8,7]`. The [decompressor](https://github.com/vllm-project/compressed-tensors/blob/698967f567bb22713a7c24df08c3df7a3be18284/src/compressed_tensors/quantization/lifecycle/forward_helpers.py) performs arithmetic in the scale dtype before output casting. This inspected library pin is newer than the producer's declared `0.17.1.a20260602`; exact producer compatibility remains a validation item.

GGUF Q4_0 has a different physical layout: each 32-weight block carries one FP16 delta and 16 bytes. Byte j's low nibble addresses weight j; its high nibble addresses weight j+16. Reconstruction is `delta * (nibble - 8)`. Preserve delta including its sign; the implicit offset 8 is a representation bias, not a fitted affine zero point. Do not reinterpret CT's consecutive packing as GGUF's split-half packing. [llama.cpp pin 930e2fa5995789efbf249a8bf61325bb626e417b: block definition](https://github.com/ggml-org/llama.cpp/blob/930e2fa5995789efbf249a8bf61325bb626e417b/ggml/src/ggml-common.h#L194), [dequantization](https://github.com/ggml-org/llama.cpp/blob/930e2fa5995789efbf249a8bf61325bb626e417b/ggml/src/ggml-quants.c#L459).

Neither file format establishes bit-identical reconstructed weights between CT, GGUF and the QAT BF16 master. Choose one canonical published artifact and its reconstruction contract. Actual per-tensor types, scale dtypes, exceptions, permutations and shape metadata have not been inspected because weight assets were not accessed.

## 3. A mathematical MIL mapping exists; execution benefit is unknown

Apple's iOS18/macOS15 `constexpr_blockwise_shift_scale` supports INT4/UINT4 and computes `scale * (data - offset)`. For a convolution weight `[O,K,1,1]`, Google's groups map to scale shape `[O,K/32,1,1]`; use signed codes with zero offset, or unsigned codes with offset 8. Preserve every group. Replacing these scales with one per output channel changes the published weights. [Apple coremltools pin 2c134ec703f1274f73fcc4e30a1bf53a4cfef4be, operation contract](https://github.com/apple/coremltools/blob/2c134ec703f1274f73fcc4e30a1bf53a4cfef4be/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L20).

For Gemma FFNs the scale shapes would be `[15360,120,1,1]` for gate/up and `[3840,480,1,1]` for down. With two-byte scales, 32 weights occupy 16 code bytes plus 2 scale bytes: 4.5 effective bits/weight before metadata. This is source arithmetic, not measured ANE traffic.

MIL accepts FP16/FP32 scales and outputs here, not BF16. Importing a BF16 scale into FP32 preserves its value, but reproducing the source runtime's product rounding and the current FP16 ANE weight boundary requires an explicit dense reconstruction control. Private-framework acceptance, selected lowered representation and runtime compression remain unverified. This report contains no new MIL graph or LUT implementation.

## 4. What Apple actually documents

- [Compression overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html): a backend may expand weights before execution or decompress during execution; source size alone does not predict runtime memory/latency. Apple recommends testing the exact model/device combination.
- [Palettization performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html): ANE latency gains require just-in-time decompression. More LUTs can reduce performance. Published four-bit examples include MobileNetV2 0.48→0.45 ms and ResNet50 1.52→1.41 ms, but these are principally **A16/iPhone 14 Pro, iOS17, Xcode15, computeUnits=all**, not M4 Gemma or an ANE-only attribution. Palettization is not Google's group-32 linear quantization.
- [Linear-quantization performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html): Apple highlights four-bit **per-block** benefits on GPU and recommends **per-channel** scales on ANE. Its explicit M4 faster integer-compute claim concerns **INT8 weights and INT8 activations**. Google's W4A16 artifact does not supply that activation scheme.
- [Availability](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html): four-bit blockwise linear quantization arrives with iOS18/macOS15. Format availability is not a speed guarantee. The MIL operation itself permits decompression at load time or runtime.

Apple therefore documents real four-bit compression opportunities, but not the exact ANE execution benefit required here. No compiler flag forcing compressed runtime treatment for this representation was established.

## 5. Concrete viability and outstanding gates

**Proven:** official Google QAT provenance; public ungated artifacts; exact file/revision metadata; group-32 integer semantics in the CT configuration; an Apple MIL operation capable of expressing those groups mathematically.

**Not proven:** exact packed tensor inventory/producer-version compatibility; equality between Google's packaging variants; faithful BF16/FP16 reconstruction and Gemma layer numerics; ANE acceptance or on-the-fly decompression; local M4 latency/memory/energy improvement with these QAT weights.

Any future four-bit experiment under the user's existing authorization should begin with a **host-only native-artifact importer/reconstruction audit**. Preserve codes/scales and uncompressed exceptions; compare a future compressed graph against the same official artifact densely reconstructed at the chosen ANE precision. A subsequent same-device component comparison with stable power and thermal controls would determine whether Apple's possible benefit is corroborated. These are technical evidence requirements, not additional permission steps. The earlier approximately 1.24× scalar-LUT4 component result used different, posthoc-quantized weights and its full-model quality failed; it cannot satisfy this gate.

A QAT model changes model identity. Metal prefill, ANE decode, norms, embeddings/head, tokenizer/template and reference outputs must form a coherent QAT target; mixing original BF16-prefill KV with a different QAT decoder is a separate hybrid model requiring its own qualification. Frozen QAT dense controls and full-model quality are necessary even if a component becomes faster. No model download, implementation, hardware test or revival of grouped LUT4 occurred during this research.
