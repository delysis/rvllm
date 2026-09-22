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
    let read = |first: usize, bytes: usize| -> TestResult<&[u8]> {
        let last = first.checked_add(bytes).ok_or("descriptor overflow")?;
        blob.get(first..last)
            .ok_or_else(|| "truncated descriptor".into())
    };
    if offset % 64 != 0
        || read(offset, 4)? != 0xDEAD_BEEF_u32.to_le_bytes()
        || read(offset + 4, 4)? != dtype.to_le_bytes()
    {
        return Err("descriptor ABI mismatch".into());
    }
    let len = u64::from_le_bytes(read(offset + 8, 8)?.try_into().map_err(|_| "length")?);
    let pos = u64::from_le_bytes(read(offset + 16, 8)?.try_into().map_err(|_| "position")?);
    let pos = usize::try_from(pos).map_err(|_| "offset overflow")?;
    let len = usize::try_from(len).map_err(|_| "length overflow")?;
    if pos != offset + 64 || len == 0 || len % 64 != 0 {
        return Err("bad payload".into());
    }
    read(pos, len)
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
    let mut result = BTreeMap::new();
    for line in mil
        .lines()
        .filter(|line| line.contains(" = constexpr_affine_dequantize()"))
    {
        let (left, _) = line.split_once(" = ").ok_or("assignment")?;
        let name = left.split_whitespace().last().ok_or("constant name")?;
        let shape = dimensions(line)?;
        if shape.len() != 4 || shape[2..] != [1, 1] {
            return Err("weight rank".into());
        }
        let offsets = line
            .split("offset = uint64(")
            .skip(1)
            .map(|s| number(s.split(')').next().unwrap_or("")))
            .collect::<Result<Vec<_>, _>>()?;
        if offsets.len() != 2 {
            return Err("affine needs two payloads".into());
        }
        let q = payload(blob, offsets[0], 4)?;
        let scales = payload(blob, offsets[1], 1)?;
        if q.len() != shape[0] * shape[1] || scales.len() != 2 * shape[0] {
            return Err("constant shape mismatch".into());
        }
        let values = q
            .chunks_exact(shape[1])
            .zip(scales.chunks_exact(2))
            .flat_map(|(row, scale)| {
                let scale = f16::from_le_bytes([scale[0], scale[1]]).to_f32();
                row.iter()
                    .map(move |byte| f16::from_f32(f32::from(i8::from_le_bytes([*byte])) * scale))
            })
            .collect();
        result.insert(name.into(), (shape[0], shape[1], values));
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

fn interpret(source: &Int8CandidateSource, input: &[f16]) -> TestResult<Vec<f16>> {
    let weights = decode_constants(&source.mil, &source.blob)?;
    let mut values = BTreeMap::<String, Vec<f16>>::new();
    values.insert("x".into(), input.to_vec());
    for line in source.mil.lines() {
        let Some((left, expression)) = line.split_once(" = ") else {
            continue;
        };
        let name = left.split_whitespace().last().ok_or("name")?;
        if let Some((_, after)) = expression.split_once("val = fp16(") {
            let scalar = after
                .split(')')
                .next()
                .ok_or("scalar")?
                .parse::<f32>()
                .map_err(|error| format!("scalar: {error}"))?;
            values.insert(name.into(), vec![f16::from_f32(scalar)]);
            continue;
        }
        if expression.starts_with("conv(") {
            let w = weights
                .get(argument(expression, "weight")?)
                .ok_or("weight lookup")?;
            let x = values
                .get(argument(expression, "x")?)
                .ok_or("input lookup")?;
            if x.len() != w.1 {
                return Err("projection input width".into());
            }
            values.insert(name.into(), projection(&w.2, x));
        } else if expression.starts_with("concat(") {
            if !source.mil.contains("val = int32(1)") || !source.mil.contains("val = bool(false)") {
                return Err("concat contract".into());
            }
            let names = expression
                .split_once("values = (")
                .ok_or("concat inputs")?
                .1
                .split_once(')')
                .ok_or("concat inputs end")?
                .0;
            let mut output = Vec::new();
            for input in names.split(',') {
                output.extend_from_slice(values.get(input.trim()).ok_or("concat lookup")?);
            }
            values.insert(name.into(), output);
        } else if expression.starts_with("mul(")
            || expression.starts_with("add(")
            || expression.starts_with("tanh(")
        {
            let x = values.get(argument(expression, "x")?).ok_or("x lookup")?;
            let y = if expression.starts_with("tanh(") {
                None
            } else {
                Some(values.get(argument(expression, "y")?).ok_or("y lookup")?)
            };
            if y.is_some_and(|y| y.len() != 1 && y.len() != x.len()) {
                return Err("broadcast width".into());
            }
            let output = x
                .iter()
                .enumerate()
                .map(|(index, x)| {
                    let x = x.to_f32();
                    let y = y
                        .map(|y| y[if y.len() == 1 { 0 } else { index }].to_f32())
                        .unwrap_or(0.0);
                    f16::from_f32(if expression.starts_with("mul(") {
                        x * y
                    } else if expression.starts_with("add(") {
                        x + y
                    } else {
                        x.tanh()
                    })
                })
                .collect::<Vec<_>>();
            values.insert(name.into(), output);
        } else {
            continue;
        }
        let dims = dimensions(line)?;
        let count = dims
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))
            .ok_or("shape product")?;
        if values.get(name).ok_or("output lookup")?.len() != count {
            return Err("declared output shape".into());
        }
    }
    values.remove("y").ok_or_else(|| "missing output".into())
}

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
