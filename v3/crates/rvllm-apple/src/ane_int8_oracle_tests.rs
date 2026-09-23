//! Mutations attack emitted graph attributes, not a duplicate candidate formula.
#![forbid(unsafe_code)]
use super::*;

#[test]
fn unsupported_ops_and_duplicate_definitions_cannot_be_silently_skipped() -> TestResult {
    let weights = fixture_weights()?;
    let mut source = build_ffn_chunk4(&weights)?;
    let input = vec![f16::from_f32(0.125); 32];
    assert!(interpret(&source, &input).is_ok());
    let original = source.mil.clone();
    source.mil = original.replacen(" = mul(", " = unreviewed(", 1);
    assert!(interpret(&source, &input).is_err());
    let line = original.lines().find(|line| line.contains("fp16 gelu_half =")).ok_or("scalar fixture")?;
    source.mil = original.replace(line, &format!("{line}\n{line}"));
    assert!(interpret(&source, &input).is_err());
    source.mil = original.replace("} -> (y);", "} -> (gate0);");
    assert!(interpret(&source, &input).is_err());
    Ok(())
}

#[test]
fn actual_concat_and_convolution_arguments_are_resolved_not_text_presence() -> TestResult {
    let mut source = build_ffn_chunk4(&fixture_weights()?)?;
    let original = source.mil.clone();
    let input = vec![f16::from_f32(0.125); 32];
    for (old, new) in [
        ("axis = concat_axis", "axis = groups"),
        ("groups = groups", "groups = strides"),
        ("pad = pad,", "pad = strides,"),
        ("interleave = concat_interleave", "interleave = missing_bool"),
        ("val = bool(false)", "val = bool(true)"),
        ("val = int32(1)", "val = int32(2)"),
        ("val = string(\"valid\")", "val = string(\"same\")"),
    ] {
        // Binding a different scalar with the same value (groups == axis == 1)
        // is valid; test the bound value instead of demanding variable names.
        if old == "axis = concat_axis" { continue; }
        assert!(original.contains(old));
        source.mil = original.replace(old, new);
        assert!(interpret(&source, &input).is_err(), "accepted mutant {old}");
    }
    source.mil = original.replace("axis = concat_axis", "axis = groups");
    assert!(interpret(&source, &input).is_ok());
    Ok(())
}

#[test]
fn reshape_uses_its_bound_shape_constant_and_descriptors_fail_without_panics() -> TestResult {
    let weights = fixture_weights()?;
    let mut source = crate::ane_int8_candidates::interleaved::build_for_host_test(&weights)?;
    let input = vec![f16::from_f32(0.125); 32];
    source.mil = source.mil.replace("shape = gu_shape", "shape = branch_size");
    assert!(interpret(&source, &input).is_err());
    let source = build_ffn_chunk4(&weights)?;
    for bytes in [vec![], vec![0; 63], source.blob[..source.blob.len()-1].to_vec()] {
        assert!(decode_constants(&source.mil, &bytes).is_err());
    }
    for offset in [usize::MAX, usize::MAX-63, 1, source.blob.len()] {
        assert!(payload(&source.blob, offset, 4).is_err());
    }
    let mut invalid = source.blob.clone();
    invalid[0..4].copy_from_slice(&2_u32.to_le_bytes());
    assert!(decode_constants(&source.mil, &invalid).is_err());
    Ok(())
}

#[test]
fn declared_dtypes_and_positive_finite_scales_are_part_of_the_graph_contract() -> TestResult {
    let weights = fixture_weights()?;
    let mut source = build_ffn_chunk4(&weights)?;
    let original = source.mil.clone();
    let input = vec![f16::from_f32(0.125); 32];
    for (old, new) in [
        ("quantized_data = tensor<int8,", "quantized_data = tensor<uint8,"),
        ("zero_point = int8(0)", "zero_point = int8(1)"),
        ("tensor<fp16, [1, 32, 1, 1]> gate0", "tensor<fp32, [1, 32, 1, 1]> gate0"),
    ] {
        assert!(original.contains(old));
        source.mil = original.replace(old, new);
        assert!(interpret(&source, &input).is_err(), "accepted dtype mutant {old}");
    }
    source.mil = original;
    let descriptor = source.mil.lines().find(|line| line.contains("Wg0 = constexpr_")).ok_or("weight")?;
    let offsets: Vec<_> = descriptor.split("offset = uint64(").skip(1)
        .map(|s| number(s.split(')').next().unwrap_or(""))).collect::<TestResult<_>>()?;
    let scale_pos = offsets[1] + 64;
    for bits in [0_u16, 0x8000, 0x7c00, 0x7e00, 0xbc00] {
        let mut invalid = source.blob.clone();
        invalid[scale_pos..scale_pos+2].copy_from_slice(&bits.to_le_bytes());
        assert!(decode_constants(&source.mil, &invalid).is_err());
    }
    Ok(())
}
