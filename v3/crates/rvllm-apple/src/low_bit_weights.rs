//! Portable packed low-bit weight formats for Apple inference.
//!
//! Version 1 stores a row-major `[rows, K]` matrix. Quantization groups never
//! cross a row boundary: every consecutive group of up to 32 K-elements has
//! one FP16 scale. W4 values use two's-complement nibbles with the earlier
//! K-element in the low nibble; W8 values use two's-complement bytes. Both
//! encodings reserve the asymmetric minimum (`-8` or `-128`) so their integer
//! domains remain signed and symmetric.

use std::{error::Error, fmt};

use half::f16;
use serde::{Deserialize, Serialize};

/// Current packed-weight cache and kernel ABI version.
pub const APPLE_LOW_BIT_WEIGHT_ABI_VERSION: u16 = 1;

/// Number of consecutive K-elements sharing one FP16 scale.
pub const APPLE_LOW_BIT_GROUP_SIZE: usize = 32;

/// Low-bit weight encoding used with FP16 activations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum AppleLowBitWeightFormat {
    /// Signed symmetric int4 weights, with two values per byte.
    W4A16 = 4,
    /// Signed symmetric int8 weights, with one value per byte.
    W8A16 = 8,
}

impl AppleLowBitWeightFormat {
    #[must_use]
    pub const fn bits(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn quantized_max(self) -> i16 {
        match self {
            Self::W4A16 => 7,
            Self::W8A16 => 127,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::W4A16 => "w4a16",
            Self::W8A16 => "w8a16",
        }
    }
}

/// Failure while constructing or validating a packed low-bit weight tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LowBitWeightError {
    UnsupportedVersion { found: u16, expected: u16 },
    ZeroDimension { dimension: &'static str },
    SizeOverflow,
    ElementCount { expected: usize, actual: usize },
    ActivationElementCount { expected: usize, actual: usize },
    PackedByteCount { expected: usize, actual: usize },
    ScaleCount { expected: usize, actual: usize },
    NonFiniteInput { index: usize },
    InvalidScale { index: usize },
    ScaleNotRepresentable { row: usize, group: usize },
    ReservedQuantizedValue { row: usize, column: usize },
    NonZeroTailPadding { row: usize },
    ZeroScaleWithNonZeroValues { row: usize, group: usize },
}

impl fmt::Display for LowBitWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found, expected } => {
                write!(
                    f,
                    "unsupported low-bit weight ABI version {found}; expected {expected}"
                )
            }
            Self::ZeroDimension { dimension } => {
                write!(f, "low-bit weight {dimension} dimension must be nonzero")
            }
            Self::SizeOverflow => write!(f, "low-bit weight shape or storage size overflowed"),
            Self::ElementCount { expected, actual } => write!(
                f,
                "low-bit source element count mismatch: expected {expected}, got {actual}"
            ),
            Self::ActivationElementCount { expected, actual } => write!(
                f,
                "low-bit projection activation count mismatch: expected {expected}, got {actual}"
            ),
            Self::PackedByteCount { expected, actual } => write!(
                f,
                "packed low-bit byte count mismatch: expected {expected}, got {actual}"
            ),
            Self::ScaleCount { expected, actual } => write!(
                f,
                "low-bit FP16 scale count mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteInput { index } => {
                write!(f, "low-bit source value at index {index} is not finite")
            }
            Self::InvalidScale { index } => write!(
                f,
                "low-bit FP16 scale at index {index} must be finite and non-negative"
            ),
            Self::ScaleNotRepresentable { row, group } => write!(
                f,
                "low-bit scale for row {row}, group {group} is not representable as FP16"
            ),
            Self::ReservedQuantizedValue { row, column } => write!(
                f,
                "low-bit value at row {row}, column {column} uses the reserved asymmetric minimum"
            ),
            Self::NonZeroTailPadding { row } => {
                write!(f, "W4A16 row {row} has nonzero tail padding")
            }
            Self::ZeroScaleWithNonZeroValues { row, group } => write!(
                f,
                "low-bit row {row}, group {group} has nonzero values with a zero scale"
            ),
        }
    }
}

impl Error for LowBitWeightError {}

/// Owned, validated version-1 W4A16 or W8A16 weight matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedAppleLowBitWeights {
    abi_version: u16,
    format: AppleLowBitWeightFormat,
    rows: usize,
    k: usize,
    values: Vec<u8>,
    scales: Vec<f16>,
}

impl PackedAppleLowBitWeights {
    /// Validate externally supplied packed values and take ownership of them.
    pub fn from_parts(
        abi_version: u16,
        format: AppleLowBitWeightFormat,
        rows: usize,
        k: usize,
        values: Vec<u8>,
        scales: Vec<f16>,
    ) -> Result<Self, LowBitWeightError> {
        let weights = Self {
            abi_version,
            format,
            rows,
            k,
            values,
            scales,
        };
        weights.validate()?;
        Ok(weights)
    }

    #[must_use]
    pub const fn abi_version(&self) -> u16 {
        self.abi_version
    }

    #[must_use]
    pub const fn format(&self) -> AppleLowBitWeightFormat {
        self.format
    }

    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.k]
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    #[must_use]
    pub fn packed_values(&self) -> &[u8] {
        &self.values
    }

    #[must_use]
    pub fn scales(&self) -> &[f16] {
        &self.scales
    }

    /// Packed bytes occupied by one row, including deterministic W4 tail padding.
    pub fn packed_bytes_per_row(&self) -> Result<usize, LowBitWeightError> {
        packed_bytes_per_row(self.format, self.k)
    }

    /// Number of FP16 scales occupied by one row.
    pub fn groups_per_row(&self) -> Result<usize, LowBitWeightError> {
        groups_per_row(self.k)
    }

    /// Revalidate the complete shape, byte layout, scale table, and integer domain.
    pub fn validate(&self) -> Result<(), LowBitWeightError> {
        validate_shape(self.abi_version, self.rows, self.k)?;

        let bytes_per_row = packed_bytes_per_row(self.format, self.k)?;
        let expected_values = self
            .rows
            .checked_mul(bytes_per_row)
            .ok_or(LowBitWeightError::SizeOverflow)?;
        if self.values.len() != expected_values {
            return Err(LowBitWeightError::PackedByteCount {
                expected: expected_values,
                actual: self.values.len(),
            });
        }

        let groups_per_row = groups_per_row(self.k)?;
        let expected_scales = self
            .rows
            .checked_mul(groups_per_row)
            .ok_or(LowBitWeightError::SizeOverflow)?;
        if self.scales.len() != expected_scales {
            return Err(LowBitWeightError::ScaleCount {
                expected: expected_scales,
                actual: self.scales.len(),
            });
        }

        for (index, scale) in self.scales.iter().enumerate() {
            let scale = scale.to_f32();
            if !scale.is_finite() || scale < 0.0 {
                return Err(LowBitWeightError::InvalidScale { index });
            }
        }

        for row in 0..self.rows {
            if self.format == AppleLowBitWeightFormat::W4A16 && self.k % 2 != 0 {
                let tail = self.values[row * bytes_per_row + bytes_per_row - 1];
                if tail & 0xf0 != 0 {
                    return Err(LowBitWeightError::NonZeroTailPadding { row });
                }
            }

            for column in 0..self.k {
                if self.quantized_value_unchecked(row, column) == -self.format.quantized_max() - 1 {
                    return Err(LowBitWeightError::ReservedQuantizedValue { row, column });
                }
            }

            for group in 0..groups_per_row {
                let scale = self.scales[row * groups_per_row + group].to_f32();
                if scale == 0.0 {
                    let start = group * APPLE_LOW_BIT_GROUP_SIZE;
                    let end = self.k.min(start + APPLE_LOW_BIT_GROUP_SIZE);
                    if (start..end).any(|column| self.quantized_value_unchecked(row, column) != 0) {
                        return Err(LowBitWeightError::ZeroScaleWithNonZeroValues { row, group });
                    }
                }
            }
        }

        Ok(())
    }

    /// Return one signed quantized value after bounds checking.
    #[must_use]
    pub fn quantized_value(&self, row: usize, column: usize) -> Option<i16> {
        if row >= self.rows || column >= self.k {
            return None;
        }
        Some(self.quantized_value_unchecked(row, column))
    }

    fn quantized_value_unchecked(&self, row: usize, column: usize) -> i16 {
        let bytes_per_row = match self.format {
            AppleLowBitWeightFormat::W4A16 => (self.k + 1) / 2,
            AppleLowBitWeightFormat::W8A16 => self.k,
        };
        let row_base = row * bytes_per_row;
        match self.format {
            AppleLowBitWeightFormat::W4A16 => {
                let byte = self.values[row_base + column / 2];
                let nibble = if column % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                sign_extend_nibble(nibble) as i16
            }
            AppleLowBitWeightFormat::W8A16 => self.values[row_base + column] as i8 as i16,
        }
    }
}

/// CPU reference quantization into the current versioned packed format.
pub fn quantize_apple_low_bit_reference(
    format: AppleLowBitWeightFormat,
    rows: usize,
    k: usize,
    source: &[f32],
) -> Result<PackedAppleLowBitWeights, LowBitWeightError> {
    validate_shape(APPLE_LOW_BIT_WEIGHT_ABI_VERSION, rows, k)?;
    let expected_elements = rows.checked_mul(k).ok_or(LowBitWeightError::SizeOverflow)?;
    if source.len() != expected_elements {
        return Err(LowBitWeightError::ElementCount {
            expected: expected_elements,
            actual: source.len(),
        });
    }
    if let Some(index) = source.iter().position(|value| !value.is_finite()) {
        return Err(LowBitWeightError::NonFiniteInput { index });
    }

    let bytes_per_row = packed_bytes_per_row(format, k)?;
    let groups_per_row = groups_per_row(k)?;
    let values_len = rows
        .checked_mul(bytes_per_row)
        .ok_or(LowBitWeightError::SizeOverflow)?;
    let scales_len = rows
        .checked_mul(groups_per_row)
        .ok_or(LowBitWeightError::SizeOverflow)?;
    let mut values = vec![0u8; values_len];
    let mut scales = vec![f16::ZERO; scales_len];
    let qmax = format.quantized_max() as f32;

    for row in 0..rows {
        for group in 0..groups_per_row {
            let start = group * APPLE_LOW_BIT_GROUP_SIZE;
            let end = k.min(start + APPLE_LOW_BIT_GROUP_SIZE);
            let row_start = row * k;
            let absmax = source[row_start + start..row_start + end]
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if absmax == 0.0 {
                f16::ZERO
            } else {
                let rounded = f16::from_f32(absmax / qmax);
                if rounded.is_infinite() {
                    return Err(LowBitWeightError::ScaleNotRepresentable { row, group });
                }
                // Preserve nonzero groups whose ideal scale is below the smallest
                // FP16 subnormal. Quantization will saturate, but the ABI remains
                // self-consistent and never turns a nonzero group into zero.
                if rounded == f16::ZERO {
                    f16::from_bits(1)
                } else {
                    rounded
                }
            };
            scales[row * groups_per_row + group] = scale;
            let scale_f32 = scale.to_f32();

            for column in start..end {
                let quantized = if scale_f32 == 0.0 {
                    0
                } else {
                    (source[row_start + column] / scale_f32)
                        .round()
                        .clamp(-qmax, qmax) as i16
                };
                write_quantized(format, &mut values, bytes_per_row, row, column, quantized);
            }
        }
    }

    PackedAppleLowBitWeights::from_parts(
        APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
        format,
        rows,
        k,
        values,
        scales,
    )
}

/// CPU reference dequantization using the stored FP16 scales exactly.
pub fn dequantize_apple_low_bit_reference(
    weights: &PackedAppleLowBitWeights,
) -> Result<Vec<f16>, LowBitWeightError> {
    weights.validate()?;
    let element_count = weights
        .rows
        .checked_mul(weights.k)
        .ok_or(LowBitWeightError::SizeOverflow)?;
    let groups_per_row = weights.groups_per_row()?;
    let mut output = Vec::with_capacity(element_count);
    for row in 0..weights.rows {
        for column in 0..weights.k {
            let group = column / APPLE_LOW_BIT_GROUP_SIZE;
            let scale = weights.scales[row * groups_per_row + group].to_f32();
            output.push(f16::from_f32(
                weights.quantized_value_unchecked(row, column) as f32 * scale,
            ));
        }
    }
    Ok(output)
}

/// Checked-in CPU reference for `C[M,N] = A[M,K] * W[N,K]^T`.
///
/// Accumulation is FP32 and the final result is rounded to FP16, matching the
/// Metal W4A16/W8A16 projection contract. Stored FP16 scales are consumed
/// exactly; this function never requantizes or reconstructs scales from the
/// original checkpoint.
pub fn project_apple_low_bit_reference(
    weights: &PackedAppleLowBitWeights,
    activations: &[f16],
    m: usize,
) -> Result<Vec<f16>, LowBitWeightError> {
    weights.validate()?;
    let expected = m
        .checked_mul(weights.k)
        .ok_or(LowBitWeightError::SizeOverflow)?;
    if activations.len() != expected {
        return Err(LowBitWeightError::ActivationElementCount {
            expected,
            actual: activations.len(),
        });
    }
    let output_len = m
        .checked_mul(weights.rows)
        .ok_or(LowBitWeightError::SizeOverflow)?;
    let groups_per_row = weights.groups_per_row()?;
    let mut output = vec![f16::ZERO; output_len];
    for activation_row in 0..m {
        for weight_row in 0..weights.rows {
            let mut accumulator = 0.0f32;
            for column in 0..weights.k {
                let scale = weights.scales
                    [weight_row * groups_per_row + column / APPLE_LOW_BIT_GROUP_SIZE]
                    .to_f32();
                let weight = weights.quantized_value_unchecked(weight_row, column) as f32 * scale;
                accumulator += activations[activation_row * weights.k + column].to_f32() * weight;
            }
            output[activation_row * weights.rows + weight_row] = f16::from_f32(accumulator);
        }
    }
    Ok(output)
}

fn validate_shape(abi_version: u16, rows: usize, k: usize) -> Result<(), LowBitWeightError> {
    if abi_version != APPLE_LOW_BIT_WEIGHT_ABI_VERSION {
        return Err(LowBitWeightError::UnsupportedVersion {
            found: abi_version,
            expected: APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
        });
    }
    if rows == 0 {
        return Err(LowBitWeightError::ZeroDimension { dimension: "row" });
    }
    if k == 0 {
        return Err(LowBitWeightError::ZeroDimension { dimension: "K" });
    }
    rows.checked_mul(k).ok_or(LowBitWeightError::SizeOverflow)?;
    Ok(())
}

fn groups_per_row(k: usize) -> Result<usize, LowBitWeightError> {
    k.checked_add(APPLE_LOW_BIT_GROUP_SIZE - 1)
        .ok_or(LowBitWeightError::SizeOverflow)
        .map(|rounded| rounded / APPLE_LOW_BIT_GROUP_SIZE)
}

fn packed_bytes_per_row(
    format: AppleLowBitWeightFormat,
    k: usize,
) -> Result<usize, LowBitWeightError> {
    match format {
        AppleLowBitWeightFormat::W4A16 => k
            .checked_add(1)
            .ok_or(LowBitWeightError::SizeOverflow)
            .map(|rounded| rounded / 2),
        AppleLowBitWeightFormat::W8A16 => Ok(k),
    }
}

fn write_quantized(
    format: AppleLowBitWeightFormat,
    values: &mut [u8],
    bytes_per_row: usize,
    row: usize,
    column: usize,
    value: i16,
) {
    let row_base = row * bytes_per_row;
    match format {
        AppleLowBitWeightFormat::W4A16 => {
            let nibble = (value as i8 as u8) & 0x0f;
            let byte = &mut values[row_base + column / 2];
            if column % 2 == 0 {
                *byte = (*byte & 0xf0) | nibble;
            } else {
                *byte = (*byte & 0x0f) | (nibble << 4);
            }
        }
        AppleLowBitWeightFormat::W8A16 => {
            values[row_base + column] = value as i8 as u8;
        }
    }
}

fn sign_extend_nibble(value: u8) -> i8 {
    ((value << 4) as i8) >> 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_values(values: &[f16]) -> Vec<f32> {
        values.iter().map(|value| value.to_f32()).collect()
    }

    #[test]
    fn w4_v1_uses_low_nibble_first_and_symmetric_domain() {
        let source = [-7.0, -1.0, 0.0, 1.0, 7.0];
        let packed = quantize_apple_low_bit_reference(
            AppleLowBitWeightFormat::W4A16,
            1,
            source.len(),
            &source,
        )
        .unwrap();

        assert_eq!(packed.abi_version(), 1);
        assert_eq!(packed.shape(), [1, 5]);
        assert_eq!(packed.packed_values(), &[0xf9, 0x10, 0x07]);
        assert_eq!(packed.quantized_value(0, 0), Some(-7));
        assert_eq!(packed.quantized_value(0, 4), Some(7));
        assert_eq!(packed.quantized_value(1, 0), None);
        assert_eq!(
            f32_values(&dequantize_apple_low_bit_reference(&packed).unwrap()),
            source
        );
    }

    #[test]
    fn w4_tail_groups_and_rows_have_independent_scales_and_padding() {
        let k = 33;
        let mut source = vec![0.0; 2 * k];
        for (index, value) in source[..k].iter_mut().enumerate() {
            *value = index as f32 / 10.0 - 1.5;
        }
        for (index, value) in source[k..].iter_mut().enumerate() {
            *value = index as f32 * 3.0 - 40.0;
        }
        let packed =
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, 2, k, &source)
                .unwrap();

        assert_eq!(packed.packed_bytes_per_row().unwrap(), 17);
        assert_eq!(packed.packed_values().len(), 34);
        assert_eq!(packed.groups_per_row().unwrap(), 2);
        assert_eq!(packed.scales().len(), 4);
        assert_eq!(packed.packed_values()[16] & 0xf0, 0);
        assert_eq!(packed.packed_values()[33] & 0xf0, 0);

        let restored = dequantize_apple_low_bit_reference(&packed).unwrap();
        for row in 0..2 {
            for group in 0..2 {
                let start = group * APPLE_LOW_BIT_GROUP_SIZE;
                let end = k.min(start + APPLE_LOW_BIT_GROUP_SIZE);
                let tolerance = packed.scales()[row * 2 + group].to_f32() * 0.51 + 0.001;
                for column in start..end {
                    let index = row * k + column;
                    assert!((restored[index].to_f32() - source[index]).abs() <= tolerance);
                }
            }
        }
    }

    #[test]
    fn w8_tail_roundtrip_matches_fp16_scale_reference() {
        let k = 35;
        let source: Vec<f32> = (0..k)
            .map(|index| ((index as f32 - 17.0) * 0.37).sin() * 12.0)
            .collect();
        let packed =
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W8A16, 1, k, &source)
                .unwrap();
        assert_eq!(packed.packed_values().len(), k);
        assert_eq!(packed.scales().len(), 2);
        assert!(packed
            .packed_values()
            .iter()
            .all(|byte| *byte != i8::MIN as u8));

        let restored = dequantize_apple_low_bit_reference(&packed).unwrap();
        for index in 0..k {
            let scale = packed.scales()[index / APPLE_LOW_BIT_GROUP_SIZE].to_f32();
            assert!((restored[index].to_f32() - source[index]).abs() <= scale * 0.51 + 0.001);
        }
    }

    #[test]
    fn zero_groups_have_zero_scales_and_values() {
        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            let packed = quantize_apple_low_bit_reference(format, 2, 37, &[0.0; 74]).unwrap();
            assert!(packed.scales().iter().all(|scale| *scale == f16::ZERO));
            assert!(packed.packed_values().iter().all(|value| *value == 0));
            assert!(dequantize_apple_low_bit_reference(&packed)
                .unwrap()
                .iter()
                .all(|value| *value == f16::ZERO));
        }
    }

    #[test]
    fn projection_reference_handles_group_and_row_tails() {
        let rows = 3;
        let k = 33;
        let m = 2;
        let source = (0..rows * k)
            .map(|index| ((index * 17 % 41) as f32 - 20.0) / 7.0)
            .collect::<Vec<_>>();
        let activations = (0..m * k)
            .map(|index| f16::from_f32(((index * 11 % 29) as f32 - 14.0) / 9.0))
            .collect::<Vec<_>>();

        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            let packed = quantize_apple_low_bit_reference(format, rows, k, &source).unwrap();
            let groups_per_row = packed.groups_per_row().unwrap();
            let expected = (0..m)
                .flat_map(|activation_row| {
                    let activations = &activations;
                    let packed = &packed;
                    (0..rows).map(move |weight_row| {
                        let mut accumulator = 0.0f32;
                        for column in 0..k {
                            let scale = packed.scales()
                                [weight_row * groups_per_row + column / APPLE_LOW_BIT_GROUP_SIZE]
                                .to_f32();
                            accumulator += activations[activation_row * k + column].to_f32()
                                * (packed.quantized_value(weight_row, column).unwrap() as f32
                                    * scale);
                        }
                        f16::from_f32(accumulator)
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(
                project_apple_low_bit_reference(&packed, &activations, m).unwrap(),
                expected
            );
        }

        let packed =
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, rows, k, &source)
                .unwrap();
        assert_eq!(
            project_apple_low_bit_reference(&packed, &activations[..activations.len() - 1], m)
                .unwrap_err(),
            LowBitWeightError::ActivationElementCount {
                expected: m * k,
                actual: m * k - 1,
            }
        );
    }

    #[test]
    fn quantize_rejects_invalid_shape_count_and_input() {
        assert_eq!(
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, 0, 32, &[])
                .unwrap_err(),
            LowBitWeightError::ZeroDimension { dimension: "row" }
        );
        assert_eq!(
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, 1, 0, &[])
                .unwrap_err(),
            LowBitWeightError::ZeroDimension { dimension: "K" }
        );
        assert_eq!(
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W8A16, 2, 3, &[0.0; 5])
                .unwrap_err(),
            LowBitWeightError::ElementCount {
                expected: 6,
                actual: 5
            }
        );
        assert_eq!(
            quantize_apple_low_bit_reference(
                AppleLowBitWeightFormat::W8A16,
                1,
                2,
                &[0.0, f32::NAN]
            )
            .unwrap_err(),
            LowBitWeightError::NonFiniteInput { index: 1 }
        );
        assert_eq!(
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W8A16, usize::MAX, 2, &[])
                .unwrap_err(),
            LowBitWeightError::SizeOverflow
        );
        assert_eq!(
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, 1, 1, &[f32::MAX])
                .unwrap_err(),
            LowBitWeightError::ScaleNotRepresentable { row: 0, group: 0 }
        );
    }

    #[test]
    fn from_parts_rejects_version_and_storage_sizes() {
        assert!(matches!(
            PackedAppleLowBitWeights::from_parts(
                2,
                AppleLowBitWeightFormat::W4A16,
                1,
                2,
                vec![0],
                vec![f16::ONE]
            ),
            Err(LowBitWeightError::UnsupportedVersion { .. })
        ));
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W4A16,
                2,
                3,
                vec![0; 3],
                vec![f16::ZERO; 2]
            )
            .unwrap_err(),
            LowBitWeightError::PackedByteCount {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W8A16,
                1,
                33,
                vec![0; 33],
                vec![f16::ZERO]
            )
            .unwrap_err(),
            LowBitWeightError::ScaleCount {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn from_parts_rejects_invalid_scales_reserved_values_and_padding() {
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W8A16,
                1,
                1,
                vec![0],
                vec![f16::NAN]
            )
            .unwrap_err(),
            LowBitWeightError::InvalidScale { index: 0 }
        );
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W8A16,
                1,
                1,
                vec![0x80],
                vec![f16::ONE]
            )
            .unwrap_err(),
            LowBitWeightError::ReservedQuantizedValue { row: 0, column: 0 }
        );
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W4A16,
                1,
                2,
                vec![0x08],
                vec![f16::ONE]
            )
            .unwrap_err(),
            LowBitWeightError::ReservedQuantizedValue { row: 0, column: 0 }
        );
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W4A16,
                1,
                1,
                vec![0xf0],
                vec![f16::ZERO]
            )
            .unwrap_err(),
            LowBitWeightError::NonZeroTailPadding { row: 0 }
        );
        assert_eq!(
            PackedAppleLowBitWeights::from_parts(
                1,
                AppleLowBitWeightFormat::W4A16,
                1,
                2,
                vec![0x01],
                vec![f16::ZERO]
            )
            .unwrap_err(),
            LowBitWeightError::ZeroScaleWithNonZeroValues { row: 0, group: 0 }
        );
    }
}
