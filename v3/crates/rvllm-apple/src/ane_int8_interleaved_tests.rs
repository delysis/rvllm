//! Independent coefficient and whole-output checks for paired G/U layout.
#![forbid(unsafe_code)]
use super::*;
use crate::ane_int8_candidates::interleaved;

#[test]
fn paired_rows_and_scales_preserve_exact_int8_identity_and_all_output_values() -> TestResult {
    let weights = fixture_weights()?;
    let source = interleaved::build_for_host_test(&weights)?;
    let [g, u, d] = weights.matrices();
    let (q, scales) = raw_constant(&source, "Wgu")?;
    let mut expected_q = Vec::new();
    let mut expected_scales = Vec::new();
    for row in 0..g.scales.len() {
        for matrix in [&g, &u] {
            expected_q.extend(
                matrix.values[row * g.columns..(row + 1) * g.columns]
                    .iter()
                    .map(|value| value.to_le_bytes()[0]),
            );
            expected_scales.extend_from_slice(&matrix.scales[row].to_le_bytes());
        }
    }
    assert_eq!(q, expected_q);
    assert_eq!(scales, expected_scales);
    let (dq, ds) = raw_constant(&source, "Wd")?;
    assert_eq!(
        dq,
        d.values
            .iter()
            .map(|x| x.to_le_bytes()[0])
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ds,
        d.scales
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let [gate, up, down] = weights.dequantized();
    for sample in 0..7 {
        let input = (0..32)
            .map(|n| f16::from_f32(((n * 7 + sample) % 29) as f32 / 32.0 - 0.4))
            .collect::<Vec<_>>();
        let gated = projection(&gate, &input)
            .into_iter()
            .zip(projection(&up, &input))
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
    assert_eq!(source.blob.len(), weights.source_blob_bytes() - 128);
    assert_eq!(&source.blob[..4], &4_u32.to_le_bytes());
    assert_eq!(source.budget.convolutions, 2);
    assert_eq!(source.budget.programs, 1);
    assert_eq!(source.budget.input_surface_bytes, 32 * 64);
    assert_eq!(source.budget.output_surface_bytes, 32 * 64);
    Ok(())
}

#[test]
fn wrong_pair_slicing_is_observably_wrong_and_out_of_range_slicing_is_rejected() -> TestResult {
    let weights = fixture_weights()?;
    let mut source = interleaved::build_for_host_test(&weights)?;
    let input = (0..32)
        .map(|n| f16::from_f32((n % 11) as f32 / 8.0 - 0.5))
        .collect::<Vec<_>>();
    let correct = interpret(&source, &input)?;
    source.mil = source.mil.replace("begin = up_begin", "begin = gate_begin");
    assert_ne!(interpret(&source, &input)?, correct);
    source.mil = source.mil.replace("[0, 0, 0, 0])", "[0, 0, 2, 0])");
    assert!(interpret(&source, &input).is_err());
    Ok(())
}

#[test]
fn paired_layout_admits_only_archived_geometry_and_retains_single_io() -> TestResult {
    assert!(interleaved::supports((3840, 15360)));
    for shape in [
        (0, 0),
        (32, 128),
        (3840, 15361),
        (3841, 15360),
        (usize::MAX, 15360),
    ] {
        assert!(!interleaved::supports(shape));
    }
    let weights = fixture_weights()?;
    assert!(interleaved::build(&weights).is_err());
    let source = interleaved::build_for_host_test(&weights)?;
    assert_eq!(source.mil.matches("func main").count(), 1);
    assert!(source
        .mil
        .contains("func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x)"));
    assert!(source.mil.ends_with("    } -> (y);\n}\n"));
    assert_eq!(source.mil.matches(" = reshape(").count(), 1);
    assert_eq!(source.mil.matches(" = slice_by_size(").count(), 2);
    assert_eq!(source.mil.matches(" = tanh(").count(), 1);
    assert_eq!(source.mil.matches(" = add(").count(), 2);
    Ok(())
}
