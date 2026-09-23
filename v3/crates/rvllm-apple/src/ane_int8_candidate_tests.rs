//! Host interpretation of the *generated* MIL subset, not an ANE simulator.
//! It checks graph wiring and materialization, not the compiler's reduction.
use super::*;
use half::f16;
use std::collections::BTreeMap;

type TestResult<T = ()> = Result<T, String>;

fn number(text: &str) -> TestResult<usize> {
    text.trim()
        .parse()
        .map_err(|error| format!("number: {error}"))
}

fn payload(blob: &[u8], offset: usize, dtype: u32) -> TestResult<&[u8]> {
    let header_end = offset.checked_add(64).ok_or("descriptor overflow")?;
    let header = blob.get(offset..header_end).ok_or("truncated descriptor")?;
    if offset < 64
        || offset % 64 != 0
        || header[..4] != 0xDEAD_BEEF_u32.to_le_bytes()
        || header[4..8] != dtype.to_le_bytes()
        || header[24..].iter().any(|&b| b != 0)
    {
        return Err("descriptor ABI mismatch".into());
    }
    let len = u64::from_le_bytes(header[8..16].try_into().map_err(|_| "length")?);
    let pos = u64::from_le_bytes(header[16..24].try_into().map_err(|_| "position")?);
    let pos = usize::try_from(pos).map_err(|_| "offset overflow")?;
    let len = usize::try_from(len).map_err(|_| "length overflow")?;
    if pos != header_end || len == 0 || len % 64 != 0 {
        return Err("bad payload".into());
    }
    let end = pos.checked_add(len).ok_or("payload overflow")?;
    blob.get(pos..end).ok_or_else(|| "truncated payload".into())
}

fn dimensions(line: &str) -> TestResult<Vec<usize>> {
    let after = line.split_once('[').ok_or("missing shape")?.1;
    after
        .split_once(']')
        .ok_or("missing shape end")?
        .0
        .split(',')
        .map(number)
        .collect()
}

fn argument<'a>(expression: &'a str, name: &str) -> TestResult<&'a str> {
    let marker = format!("{name} = ");
    let tail = expression
        .split_once(&marker)
        .ok_or_else(|| format!("missing {name}"))?
        .1;
    Ok(tail
        .split([',', ')'])
        .next()
        .ok_or("missing argument")?
        .trim())
}

fn decode_constants(
    mil: &str,
    blob: &[u8],
) -> TestResult<BTreeMap<String, (usize, usize, Vec<f16>)>> {
    let header = blob.get(..64).ok_or("truncated blob header")?;
    let descriptors =
        u32::from_le_bytes(header[..4].try_into().map_err(|_| "descriptor count")?) as usize;
    if descriptors == 0
        || descriptors % 2 != 0
        || header[4..8] != 2_u32.to_le_bytes()
        || header[8..].iter().any(|&b| b != 0)
    {
        return Err("blob header ABI mismatch".into());
    }
    let mut result = BTreeMap::new();
    let mut regions = BTreeMap::new();
    for line in mil
        .lines()
        .filter(|line| line.contains(" = constexpr_affine_dequantize()"))
    {
        let (left, _) = line.trim().split_once(" = ").ok_or("assignment")?;
        let name = left.split_whitespace().last().ok_or("constant name")?;
        let shape = dimensions(line)?;
        if !left.starts_with("tensor<fp16,")
            || shape.len() != 4
            || shape[2..] != [1, 1]
            || shape[..2].contains(&0)
            || !line.contains("axis = int32(0)")
            || !line.contains("zero_point = int8(0)")
        {
            return Err("weight dtype, rank or dequantization contract".into());
        }
        let q_text = line
            .split_once("quantized_data = tensor<int8,")
            .ok_or("signed INT8 coefficients")?
            .1;
        let scale_text = line
            .split_once("scale = tensor<fp16,")
            .ok_or("FP16 scales")?
            .1;
        if dimensions(q_text)? != shape
            || dimensions(scale_text)? != [shape[0]]
            || line.matches("@model_path/weights/weight.bin").count() != 2
        {
            return Err("serialized constant shape or path".into());
        }
        let offsets = line
            .split("offset = uint64(")
            .skip(1)
            .map(|s| number(s.split(')').next().unwrap_or("")))
            .collect::<TestResult<Vec<_>>>()?;
        if offsets.len() != 2 {
            return Err("affine needs two payloads".into());
        }
        let q = payload(blob, offsets[0], 4)?;
        let scales = payload(blob, offsets[1], 1)?;
        if shape[0].checked_mul(shape[1]) != Some(q.len())
            || shape[0].checked_mul(2) != Some(scales.len())
        {
            return Err("constant shape mismatch".into());
        }
        for (offset, len) in [(offsets[0], q.len()), (offsets[1], scales.len())] {
            let end = offset
                .checked_add(64)
                .and_then(|x| x.checked_add(len))
                .ok_or("region overflow")?;
            if regions.insert(offset, end).is_some() {
                return Err("duplicate descriptor reference".into());
            }
        }
        let mut values = Vec::with_capacity(q.len());
        for (row, scale) in q.chunks_exact(shape[1]).zip(scales.chunks_exact(2)) {
            let scale = f16::from_le_bytes([scale[0], scale[1]]).to_f32();
            if !scale.is_finite() || scale <= 0.0 {
                return Err("scale must be positive and finite".into());
            }
            for byte in row {
                let value = f16::from_f32(f32::from(i8::from_le_bytes([*byte])) * scale);
                if !value.is_finite() {
                    return Err("nonfinite reconstruction".into());
                }
                values.push(value);
            }
        }
        if result
            .insert(name.into(), (shape[0], shape[1], values))
            .is_some()
        {
            return Err("duplicate weight name".into());
        }
    }
    if regions.len() != descriptors {
        return Err("descriptor count mismatch".into());
    }
    let mut end = 64;
    for (offset, next) in regions {
        if offset != end {
            return Err("overlap, gap or unused payload".into());
        }
        end = next;
    }
    if end != blob.len() {
        return Err("unreferenced blob bytes".into());
    }
    Ok(result)
}

fn projection(weights: &[f16], input: &[f16]) -> Vec<f16> {
    weights
        .chunks_exact(input.len())
        .map(|row| {
            let sum = row
                .iter()
                .zip(input)
                .fold(0.0_f32, |sum, (a, b)| sum + a.to_f32() * b.to_f32());
            f16::from_f32(sum)
        })
        .collect()
}

#[path = "ane_int8_mil_oracle.rs"]
mod oracle;
use oracle::interpret;

// Independent straight-line FP16 oracle; unlike the MIL interpreter it knows
// neither graph variable names, concat order nor BLOBFILE offsets.
fn activation(gate: f16, up: f16) -> f16 {
    let r = |v| f16::from_f32(v).to_f32();
    let g = gate.to_f32();
    let cube = r(r(g * g) * g);
    let sum = r(g + r(cube * r(0.044715)));
    let argument = r(sum * r(0.7978845608));
    let factor = r(r(argument.tanh()) + 1.0);
    f16::from_f32(r(r(g * 0.5) * factor) * up.to_f32())
}

fn fixture_weights() -> TestResult<AneInt8FfnWeights> {
    let make = |phase: usize| {
        (0..32 * 128)
            .map(|n| f16::from_f32(((n * 13 + phase) % 97) as f32 / 256.0 - 0.1875))
            .collect::<Vec<_>>()
    };
    AneInt8FfnWeights::quantize(&make(0), &make(7), &make(19), 32, 128)
}

#[test]
fn chunk4_generated_graph_and_blob_agree_with_independent_ffn() -> TestResult {
    let weights = fixture_weights()?;
    let source = build_ffn_chunk4(&weights)?;
    let decoded = decode_constants(&source.mil, &source.blob)?;
    let [gate, up, down] = weights.dequantized();
    for part in 0..4 {
        for (prefix, expected) in [("Wg", &gate), ("Wu", &up)] {
            let actual = &decoded.get(&format!("{prefix}{part}")).ok_or("chunk")?.2;
            assert_eq!(actual, &expected[part * 1024..(part + 1) * 1024]);
        }
    }
    assert_eq!(decoded.get("Wd").ok_or("down")?.2, down);
    for sample in 0..7 {
        let input = (0..32)
            .map(|n| f16::from_f32(((n * 7 + sample) % 23) as f32 / 32.0 - 0.3))
            .collect::<Vec<_>>();
        let g = projection(&gate, &input);
        let u = projection(&up, &input);
        let gated = g
            .into_iter()
            .zip(u)
            .map(|(g, u)| activation(g, u))
            .collect::<Vec<_>>();
        let expected = projection(&down, &gated);
        let actual = interpret(&source, &input)?;
        assert!(actual.iter().all(|x| x.is_finite()));
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(source.blob.len(), weights.source_blob_bytes() + 768);
    assert_eq!(&source.blob[..4], &18_u32.to_le_bytes());
    assert_eq!(source.mil.matches(" = conv(").count(), 9);
    assert_eq!(source.mil.matches(" = concat(").count(), 1);
    assert_eq!(source.mil.matches("func main").count(), 1);
    assert!(source.mil.ends_with("    } -> (y);\n}\n"));
    // The public route rejects fixture geometry rather than executing it.
    assert!(ffn_chunk4(&weights).is_err());
    Ok(())
}

#[test]
fn chunk4_preserves_baseline_gelu_order_and_rejects_broken_descriptors() -> TestResult {
    let source = build_ffn_chunk4(&fixture_weights()?)?;
    // The research packet omits the older .mil fixtures. Read the independent
    // baseline producer as text: no private-feature compilation or hardware.
    let baseline = include_str!("ane_linear.rs")
        .split_once("fn ffn_mil_with_projections(")
        .ok_or("baseline producer")?
        .1
        .split_once("#[cfg(test)]")
        .ok_or("baseline end")?
        .0;
    let fixture = baseline.split_once("r#\"").ok_or("baseline raw format")?.1;
    for part in 0..4 {
        let mut previous = 0;
        for line in fixture
            .lines()
            .filter(|l| l.contains(" = mul(") || l.contains(" = add(") || l.contains(" = tanh("))
        {
            let name = line
                .split_once(" = ")
                .ok_or("fixture assignment")?
                .0
                .split_whitespace()
                .last()
                .ok_or("fixture name")?;
            let marker = format!("> {name}{part} = ");
            let position = source.mil.find(&marker).ok_or("missing materialization")?;
            assert!(position > previous);
            previous = position;
        }
    }
    let header = fixture
        .split_once("    func main")
        .ok_or("baseline header")?
        .0
        .replace("{{", "{")
        .replace("}}", "}");
    assert!(source.mil.starts_with(&header));
    let mut corrupt = source.blob.clone();
    corrupt[64] = 0;
    assert!(decode_constants(&source.mil, &corrupt).is_err());
    assert!(decode_constants(&source.mil, &source.blob[..source.blob.len() - 1]).is_err());
    assert!(begin_blob(18, usize::MAX).is_err());
    Ok(())
}

#[test]
fn tiles4_preserves_serialized_rows_and_projection_numerics() -> TestResult {
    let dense = (0..32 * 128)
        .map(|n| f16::from_f32(((n * 29) % 113) as f32 / 256.0 - 0.2))
        .collect::<Vec<_>>();
    let weights = AneInt8LinearWeights::quantize(&dense, 32, 128)?;
    let source = build_linear_tiles4(&weights)?;
    let decoded = decode_constants(&source.mil, &source.blob)?;
    let expected = weights.dequantized();
    for part in 0..4 {
        assert_eq!(
            decoded.get(&format!("W{part}")).ok_or("tile")?.2,
            expected[part * 1024..(part + 1) * 1024]
        );
    }
    for seed in 0..7 {
        let input = (0..32)
            .map(|n| f16::from_f32(((n + seed) % 17) as f32 / 32.0))
            .collect::<Vec<_>>();
        assert_eq!(interpret(&source, &input)?, projection(&expected, &input));
    }
    assert_eq!(source.blob.len(), weights.source_blob_bytes() + 384);
    assert_eq!(&source.blob[..4], &8_u32.to_le_bytes());
    assert_eq!(source.mil.matches(" = conv(").count(), 4);
    assert_eq!(source.mil.matches(" = concat(").count(), 1);
    assert_eq!(source.budget.input_surface_bytes, 32 * 64);
    assert_eq!(source.budget.output_surface_bytes, 128 * 64);
    assert_eq!(source.budget.programs, 1);
    assert!(!source.mil.contains("gelu"));
    assert!(linear_tiles4(&weights).is_err());
    // Mutating real generated concat order must change our independent oracle.
    let mut corrupt = source;
    corrupt.mil = corrupt
        .mil
        .replace("values = (y0, y1, y2, y3)", "values = (y3, y1, y2, y0)");
    let input = vec![f16::from_f32(0.125); 32];
    assert_ne!(interpret(&corrupt, &input)?, projection(&expected, &input));
    Ok(())
}

#[test]
fn graph_shapes_are_narrow_and_do_not_admit_global_qkv() {
    assert!(ffn_chunk4_shape((3840, 15360)));
    for shape in [(3840, 8192), (4096, 3840), (8192, 3840), (3840, 16384)] {
        assert!(linear_tiles4_shape(shape));
        assert!(!ffn_chunk4_shape(shape));
    }
    for shape in [
        (3840, 8704),
        (3840, 9216),
        (3840, 262144),
        (32, 128),
        (0, 0),
        (usize::MAX, 15360),
        (3840, usize::MAX),
        (3841, 15360),
        (3840, 15361),
    ] {
        assert!(!linear_tiles4_shape(shape));
        assert!(!ffn_chunk4_shape(shape));
    }
}

// Raw serialized bytes must survive slicing even when two representations could
// happen to produce the same dequantized f16 value. Read actual MIL offsets.
fn raw_constant<'a>(
    source: &'a Int8CandidateSource,
    name: &str,
) -> TestResult<(&'a [u8], &'a [u8])> {
    let marker = format!("> {name} = constexpr_affine_dequantize()");
    let line = source
        .mil
        .lines()
        .find(|line| line.contains(&marker))
        .ok_or("raw constant")?;
    let offsets = line
        .split("offset = uint64(")
        .skip(1)
        .map(|text| number(text.split(')').next().unwrap_or("")))
        .collect::<Result<Vec<_>, _>>()?;
    if offsets.len() != 2 {
        return Err("raw payload count".into());
    }
    Ok((
        payload(&source.blob, offsets[0], 4)?,
        payload(&source.blob, offsets[1], 1)?,
    ))
}

#[test]
fn chunking_is_bit_exact_for_quantized_payloads_and_scale_payloads() -> TestResult {
    let weights = fixture_weights()?;
    let source = build_ffn_chunk4(&weights)?;
    let [gate, up, down] = weights.matrices();
    for (prefix, matrix) in [("Wg", gate), ("Wu", up)] {
        let mut q = Vec::new();
        let mut scales = Vec::new();
        for part in 0..4 {
            let (q_part, scale_part) = raw_constant(&source, &format!("{prefix}{part}"))?;
            q.extend_from_slice(q_part);
            scales.extend_from_slice(scale_part);
        }
        assert_eq!(
            q,
            matrix
                .values
                .iter()
                .map(|v| v.to_le_bytes()[0])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            scales,
            matrix
                .scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }
    let (q, scales) = raw_constant(&source, "Wd")?;
    assert_eq!(
        q,
        down.values
            .iter()
            .map(|v| v.to_le_bytes()[0])
            .collect::<Vec<_>>()
    );
    assert_eq!(
        scales,
        down.scales
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[path = "ane_int8_next_tests.rs"]
mod next;

#[path = "ane_int8_interleaved_tests.rs"]
mod interleaved_checks;

#[path = "ane_int8_oracle_tests.rs"]
mod oracle_checks;
