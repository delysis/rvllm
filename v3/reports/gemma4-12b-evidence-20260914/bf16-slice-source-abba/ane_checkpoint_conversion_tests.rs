#![forbid(unsafe_code)]
use super::*;

#[test]
fn all_half_encodings_match_scalar_conversion_or_error() {
    for dtype in [DType::Bf16, DType::F16] {
        let mut finite_bytes = Vec::new();
        let mut finite_expected = Vec::new();
        for bits in 0..=u16::MAX {
            let input = bits.to_le_bytes();
            let mut scalar = [f16::ZERO];
            let mut candidate = [f16::ZERO];
            let expected = decode_f16(&input, dtype, &mut scalar);
            let actual = decode_f16_vectorized(&input, dtype, &mut candidate);
            assert_eq!(actual, expected, "{dtype:?} {bits:04x}");
            if actual.is_ok() {
                assert_eq!(
                    candidate[0].to_bits(),
                    scalar[0].to_bits(),
                    "{dtype:?} {bits:04x}"
                );
                finite_bytes.extend(input);
                finite_expected.push(scalar[0].to_bits());
            }
        }
        // Exercise the packed SIMD lanes too; one-element calls alone would
        // qualify only the slice converter's scalar tail.
        let mut batch = vec![f16::ZERO; finite_expected.len()];
        decode_f16_vectorized(&finite_bytes, dtype, &mut batch).unwrap();
        assert!(batch
            .iter()
            .zip(finite_expected)
            .all(|(value, bits)| value.to_bits() == bits));
    }
}

#[test]
fn vectorized_conversion_handles_chunk_tails_and_length_errors() {
    for length in [0, 1, 7, 1023, 1024, 1025, 3840, 15360] {
        let mut state = 17_u32;
        let input: Vec<_> = (0..length)
            .flat_map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let magnitude = ((state >> 16) % 0x4780) as u16;
                (magnitude | (state as u16 & 0x8000)).to_le_bytes()
            })
            .collect();
        let mut scalar = vec![f16::ZERO; length];
        let mut candidate = scalar.clone();
        decode_f16(&input, DType::Bf16, &mut scalar).unwrap();
        decode_f16_vectorized(&input, DType::Bf16, &mut candidate).unwrap();
        assert!(scalar
            .iter()
            .zip(&candidate)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    for (input, length, dtype) in [
        (vec![0], 1, DType::Bf16),
        (vec![0; 4], 1, DType::F16),
        (vec![0; 2], 1, DType::F32),
    ] {
        assert_eq!(
            decode_f16(&input, dtype, &mut vec![f16::ZERO; length]),
            decode_f16_vectorized(&input, dtype, &mut vec![f16::ZERO; length])
        );
    }
}

#[test]
#[ignore = "host-only actual Gemma FFN conversion ABBA; explicit model/output paths; zero accelerator calls"]
fn checkpoint_conversion_abba() {
    use crate::apple_measurement::{compare_phase_measurements, PowerMonitor};
    use serde_json::{json, Value};
    use std::path::PathBuf;
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA4_MODEL_DIR").unwrap());
    let output = PathBuf::from(std::env::var("RVLLM_CONVERSION_BENCH_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let (arch, entries) = validated_weights(&model, 1024).unwrap();
    let entry = &entries[&format!("{}.layers.0.mlp.gate_proj.weight", arch.weight_prefix)];
    assert_eq!(entry.dtype, DType::Bf16);
    let mut bytes = vec![0; entry.nbytes];
    File::open(&entry.file)
        .unwrap()
        .read_exact_at(&mut bytes, entry.file_offset as u64)
        .unwrap();
    let mut reference = vec![f16::ZERO; bytes.len() / 2];
    let mut converted = reference.clone();
    decode_f16(&bytes, entry.dtype, &mut reference).unwrap();
    decode_f16_vectorized(&bytes, entry.dtype, &mut converted).unwrap();
    assert_eq!(reference, converted);
    let monitor = PowerMonitor::start(Some(&output.join("power-observations.jsonl"))).unwrap();
    let mut trials: Vec<Value> = Vec::new();
    let mut pairs = Vec::new();
    for block in 0..3 {
        let start = trials.len();
        for candidate in if block % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        } {
            let phase = monitor.begin();
            for _ in 0..4 {
                if candidate {
                    decode_f16_vectorized(
                        std::hint::black_box(&bytes),
                        entry.dtype,
                        std::hint::black_box(&mut converted),
                    )
                    .unwrap();
                } else {
                    decode_f16(
                        std::hint::black_box(&bytes),
                        entry.dtype,
                        std::hint::black_box(&mut converted),
                    )
                    .unwrap();
                }
            }
            let measurement = phase.finish(reference.len() * 4);
            assert!(reference
                .iter()
                .zip(&converted)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            trials.push(json!({"candidate":candidate,"measurement":measurement,"bits_match":true}));
            std::fs::write(
                output.join("trials.json"),
                serde_json::to_vec_pretty(&trials).unwrap(),
            )
            .unwrap();
        }
        for (left, right) in [(start, start + 1), (start + 3, start + 2)] {
            let (baseline, candidate) = if trials[left]["candidate"] == false {
                (left, right)
            } else {
                (right, left)
            };
            pairs.push(match compare_phase_measurements(&trials[baseline]["measurement"],&trials[candidate]["measurement"]) {
                Ok(comparison) => json!({"baseline":baseline,"candidate":candidate,"eligible":true,"comparison":comparison}),
                Err(reason) => json!({"baseline":baseline,"candidate":candidate,"eligible":false,"reason":reason}),
            });
        }
    }
    let report = json!({"schema":"rvllm.gemma12b_checkpoint_conversion_abba.v1","tensor":"layer.0.mlp.gate_proj.weight","shape":entry.shape,"dtype":"BF16","model_dir":model,"values_per_trial":reference.len()*4,"all_bits_match":true,"accelerator_calls":0,"trials":trials,"pairs":pairs,"claim":"Host checkpoint conversion only; sampled-power eligibility applies. Not full startup or decode speedup."});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
