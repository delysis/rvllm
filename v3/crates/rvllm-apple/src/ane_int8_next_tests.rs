//! Additional generated-source host oracles; no accelerator execution.
#![forbid(unsafe_code)]
use super::*;

fn interpret_next(source: &Int8CandidateSource, input: &[f16]) -> TestResult<Vec<f16>> {
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
        if expression.starts_with("reshape(") {
            let x = values
                .get(argument(expression, "x")?)
                .ok_or("reshape input")?
                .clone();
            let dims = dimensions(line)?;
            let count = dims
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .ok_or("reshape element count overflow")?;
            if count != x.len() {
                return Err("reshape changes element count".into());
            }
            values.insert(name.into(), x);
        } else if expression.starts_with("conv(") {
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

#[test]
fn down4_preserves_full_k_reductions_rows_and_gelu_materializations() -> TestResult {
    let make = |phase: usize| {
        (0..128 * 128)
            .map(|n| f16::from_f32(((n * 17 + phase) % 127) as f32 / 1024.0 - 0.05))
            .collect::<Vec<_>>()
    };
    let weights = AneInt8FfnWeights::quantize(&make(0), &make(7), &make(19), 128, 128)?;
    let source = build_ffn_down4(&weights)?;
    let decoded = decode_constants(&source.mil, &source.blob)?;
    let [gate, up, down] = weights.dequantized();
    assert_eq!(decoded.get("Wg").ok_or("gate")?.2, gate);
    assert_eq!(decoded.get("Wu").ok_or("up")?.2, up);
    for part in 0..4 {
        assert_eq!(
            decoded.get(&format!("Wd{part}")).ok_or("down tile")?.2,
            down[part * 4096..(part + 1) * 4096]
        );
    }
    for seed in 0..7 {
        let input = (0..128)
            .map(|n| f16::from_f32(((n * 7 + seed) % 37) as f32 / 64.0 - 0.2))
            .collect::<Vec<_>>();
        let g = projection(&gate, &input);
        let u = projection(&up, &input);
        let gated = g
            .into_iter()
            .zip(u)
            .map(|(g, u)| activation(g, u))
            .collect::<Vec<_>>();
        assert_eq!(interpret_next(&source, &input)?, projection(&down, &gated));
    }
    assert_eq!(source.budget.convolutions, 6);
    assert_eq!(source.budget.programs, 1);
    assert_eq!(source.blob.len(), weights.source_blob_bytes() + 384);
    assert_eq!(source.mil.matches(" = conv(").count(), 6);
    assert_eq!(source.mil.matches(" = concat(").count(), 1);
    // No new reduction or addition of separately rounded down partials.
    assert_eq!(source.mil.matches(" = add(").count(), 2);
    assert!(ffn_down4(&weights).is_err());
    assert!(ffn_down4_shape((3840, 15360)));
    for shape in [(3840, 15359), (3841, 15360), (5376, 21504), (0, 15360)] {
        assert!(!ffn_down4_shape(shape));
    }
    // Wrong concat order is observably wrong; the oracle does not follow it.
    let mut wrong = source;
    wrong.mil = wrong.mil.replace(
        "values = (down0, down1, down2, down3)",
        "values = (down3, down2, down1, down0)",
    );
    let input = vec![f16::from_f32(0.25); 128];
    let g = projection(&gate, &input);
    let u = projection(&up, &input);
    let gated = g
        .into_iter()
        .zip(u)
        .map(|(g, u)| activation(g, u))
        .collect::<Vec<_>>();
    assert_ne!(interpret(&wrong, &input)?, projection(&down, &gated));
    Ok(())
}

#[test]
fn packed32_graph_keeps_weights_and_every_ffn_rounding() -> TestResult {
    let make = |phase: usize| {
        (0..128 * 128)
            .map(|n| f16::from_f32(((n * 17 + phase) % 127) as f32 / 1024.0 - 0.05))
            .collect::<Vec<_>>()
    };
    let weights = AneInt8FfnWeights::quantize(&make(0), &make(7), &make(19), 128, 128)?;
    let source = build_ffn_packed32_source(&weights)?;
    let (baseline_blob, _) = weights.blob_and_constants();
    assert_eq!(source.blob, baseline_blob);
    assert_eq!(source.mil.matches(" = reshape(").count(), 2);
    assert_eq!(source.mil.matches(" = conv(").count(), 3);
    assert!(source
        .mil
        .contains("func main<ios18>(tensor<fp16, [1, 4, 1, 32]> x)"));
    assert!(source
        .mil
        .contains("tensor<fp16, [1, 4, 1, 32]> y = reshape"));
    let [gate, up, down] = weights.dequantized();
    for seed in 0..7 {
        let input = (0..128)
            .map(|n| f16::from_f32(((n * 13 + seed) % 31) as f32 / 128.0 - 0.1))
            .collect::<Vec<_>>();
        let g = projection(&gate, &input);
        let u = projection(&up, &input);
        let gated = g
            .into_iter()
            .zip(u)
            .map(|(g, u)| activation(g, u))
            .collect::<Vec<_>>();
        assert_eq!(interpret_next(&source, &input)?, projection(&down, &gated));
    }
    assert_eq!(source.budget.input_surface_bytes, 256);
    assert!(ffn_packed32_source(&weights).is_err());
    Ok(())
}

#[test]
fn packed32_host_codec_is_byte_exact_and_validates_before_writes() -> TestResult {
    use crate::ane_packed32_layout::Packed32Declaration;
    let layout = Packed32Declaration::gemma12b();
    assert_eq!(layout.shape(), [1, 120, 1, 32]);
    assert_eq!(layout.logical_bytes(), 7680);
    // Includes all half bit-patterns over repeated blocks, not just finite floats.
    for block in 0..18 {
        let values = (0..3840)
            .map(|i| f16::from_bits((block * 3840 + i) as u16))
            .collect::<Vec<_>>();
        let mut bytes = vec![0xa5; 7680];
        layout.pack(&values, &mut bytes)?;
        let reference = values
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(bytes, reference);
        let mut decoded = vec![f16::ZERO; 3840];
        layout.unpack(&bytes, &mut decoded)?;
        assert_eq!(
            decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let before = bytes.clone();
        assert!(layout.pack(&values[..3839], &mut bytes).is_err());
        assert_eq!(bytes, before);
        let before = decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert!(layout.unpack(&bytes[..7679], &mut decoded).is_err());
        assert_eq!(
            decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            before
        );
    }
    Ok(())
}
