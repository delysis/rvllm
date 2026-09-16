//! Persistent Metal execution for the versioned group-32 W4A16/W8A16 ABI.
//!
//! This module is intentionally independent from checkpoint loading. Creating
//! a projection validates and uploads one packed `[N,K]` matrix; encoding a
//! step binds caller-owned activation/output buffers without repacking or
//! allocating host-side launch data. The model loader currently selects one
//! authenticated dense down-projection sidecar as a bounded hybrid validation
//! route; whole-model selection still requires separate quality and
//! performance promotion gates.

use std::{error::Error, fmt, mem, ptr::NonNull};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice,
    MTLResourceOptions, MTLSize,
};

use crate::{context::MetalContext, pipeline::PipelineCache};
use rvllm_apple::{AppleLowBitWeightFormat, LowBitWeightError, PackedAppleLowBitWeights};

/// Metal-side validation or setup failure for a low-bit projection.
#[derive(Debug)]
pub enum LowBitMetalError {
    InvalidWeights(LowBitWeightError),
    ZeroBatch,
    DimensionTooLarge {
        dimension: &'static str,
        value: usize,
    },
    SizeOverflow,
    BufferAllocation {
        buffer: &'static str,
        bytes: usize,
    },
    BufferRange {
        buffer: &'static str,
        required_end: usize,
        actual: usize,
    },
    InvalidOffsetAlignment {
        buffer: &'static str,
        offset: usize,
    },
    InvalidStorageBytes {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    CommandEncoderUnavailable,
    Metal(rvllm_core::RvllmError),
}

impl fmt::Display for LowBitMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWeights(error) => write!(f, "invalid low-bit weights: {error}"),
            Self::ZeroBatch => write!(f, "low-bit projection batch M must be nonzero"),
            Self::DimensionTooLarge { dimension, value } => write!(
                f,
                "low-bit projection {dimension} dimension {value} exceeds the Metal u32 ABI"
            ),
            Self::SizeOverflow => write!(f, "low-bit projection byte size overflowed"),
            Self::BufferAllocation { buffer, bytes } => {
                write!(
                    f,
                    "failed to allocate {bytes} bytes for low-bit {buffer} buffer"
                )
            }
            Self::BufferRange {
                buffer,
                required_end,
                actual,
            } => write!(
                f,
                "low-bit {buffer} buffer requires byte end {required_end}, but length is {actual}"
            ),
            Self::InvalidOffsetAlignment { buffer, offset } => write!(
                f,
                "low-bit {buffer} buffer offset {offset} is not 4-byte aligned"
            ),
            Self::InvalidStorageBytes {
                buffer,
                expected,
                actual,
            } => write!(
                f,
                "low-bit {buffer} storage has {actual} bytes, expected {expected}"
            ),
            Self::CommandEncoderUnavailable => {
                write!(f, "Metal did not provide a low-bit compute command encoder")
            }
            Self::Metal(error) => write!(f, "Metal low-bit projection failed: {error}"),
        }
    }
}

impl Error for LowBitMetalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidWeights(error) => Some(error),
            Self::Metal(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LowBitWeightError> for LowBitMetalError {
    fn from(value: LowBitWeightError) -> Self {
        Self::InvalidWeights(value)
    }
}

impl From<rvllm_core::RvllmError> for LowBitMetalError {
    fn from(value: rvllm_core::RvllmError) -> Self {
        Self::Metal(value)
    }
}

/// Result returned by the low-bit Metal execution surface.
pub type LowBitMetalResult<T> = std::result::Result<T, LowBitMetalError>;

/// Copyable descriptor for a low-bit projection stored inside the model arena.
///
/// The descriptor contains no owning Metal object and is therefore safe to
/// copy into every preallocated execution-slot view. Both weight regions are
/// immutable after preparation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MetalLowBitProjectionOffsets {
    format: AppleLowBitWeightFormat,
    n: u32,
    k: u32,
    packed_values_offset: usize,
    packed_values_bytes: usize,
    scales_offset: usize,
    scales_bytes: usize,
}

impl MetalLowBitProjectionOffsets {
    /// Validate the exact group-32 storage layout selected for an arena.
    pub fn new(
        format: AppleLowBitWeightFormat,
        n: usize,
        k: usize,
        packed_values_offset: usize,
        packed_values_bytes: usize,
        scales_offset: usize,
        scales_bytes: usize,
    ) -> LowBitMetalResult<Self> {
        let n_u32 = u32_dimension("N", n)?;
        let k_u32 = u32_dimension("K", k)?;
        if n == 0 {
            return Err(LowBitMetalError::InvalidWeights(
                LowBitWeightError::ZeroDimension { dimension: "rows" },
            ));
        }
        if k == 0 {
            return Err(LowBitMetalError::InvalidWeights(
                LowBitWeightError::ZeroDimension { dimension: "K" },
            ));
        }
        validate_offset("packed weights", packed_values_offset)?;
        validate_offset("FP16 scales", scales_offset)?;
        let packed_row_bytes = match format {
            AppleLowBitWeightFormat::W4A16 => k
                .checked_add(1)
                .map(|value| value / 2)
                .ok_or(LowBitMetalError::SizeOverflow)?,
            AppleLowBitWeightFormat::W8A16 => k,
        };
        let expected_values = n
            .checked_mul(packed_row_bytes)
            .ok_or(LowBitMetalError::SizeOverflow)?;
        if packed_values_bytes != expected_values {
            return Err(LowBitMetalError::InvalidStorageBytes {
                buffer: "packed weights",
                expected: expected_values,
                actual: packed_values_bytes,
            });
        }
        let groups = k
            .checked_add(rvllm_apple::APPLE_LOW_BIT_GROUP_SIZE - 1)
            .map(|value| value / rvllm_apple::APPLE_LOW_BIT_GROUP_SIZE)
            .ok_or(LowBitMetalError::SizeOverflow)?;
        let expected_scales = n
            .checked_mul(groups)
            .and_then(|count| count.checked_mul(mem::size_of::<half::f16>()))
            .ok_or(LowBitMetalError::SizeOverflow)?;
        if scales_bytes != expected_scales {
            return Err(LowBitMetalError::InvalidStorageBytes {
                buffer: "FP16 scales",
                expected: expected_scales,
                actual: scales_bytes,
            });
        }
        Ok(Self {
            format,
            n: n_u32,
            k: k_u32,
            packed_values_offset,
            packed_values_bytes,
            scales_offset,
            scales_bytes,
        })
    }

    #[must_use]
    pub const fn format(self) -> AppleLowBitWeightFormat {
        self.format
    }

    #[must_use]
    pub const fn shape(self) -> [u32; 2] {
        [self.n, self.k]
    }

    #[must_use]
    pub const fn packed_values_offset(self) -> usize {
        self.packed_values_offset
    }

    #[must_use]
    pub const fn scales_offset(self) -> usize {
        self.scales_offset
    }

    #[must_use]
    pub const fn resident_bytes(self) -> usize {
        self.packed_values_bytes.saturating_add(self.scales_bytes)
    }

    #[must_use]
    pub const fn kernel_name(self) -> &'static str {
        match self.format {
            AppleLowBitWeightFormat::W4A16 => "projection_w4a16_f16",
            AppleLowBitWeightFormat::W8A16 => "projection_w8a16_f16",
        }
    }

    /// Encode `C[M,N] = A[M,K] * W[N,K]^T` with every tensor in one arena.
    pub fn encode(
        self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        pipelines: &PipelineCache,
        arena: &ProtocolObject<dyn MTLBuffer>,
        activation_offset: usize,
        output_offset: usize,
        m: usize,
    ) -> LowBitMetalResult<()> {
        if m == 0 {
            return Err(LowBitMetalError::ZeroBatch);
        }
        let m = u32_dimension("M", m)?;
        validate_offset("activation", activation_offset)?;
        validate_offset("output", output_offset)?;
        validate_buffer_range(
            "packed weights",
            arena.length(),
            self.packed_values_offset,
            self.packed_values_bytes,
        )?;
        validate_buffer_range(
            "FP16 scales",
            arena.length(),
            self.scales_offset,
            self.scales_bytes,
        )?;
        let activation_bytes = (m as usize)
            .checked_mul(self.k as usize)
            .and_then(|elements| elements.checked_mul(mem::size_of::<half::f16>()))
            .ok_or(LowBitMetalError::SizeOverflow)?;
        let output_bytes = (m as usize)
            .checked_mul(self.n as usize)
            .and_then(|elements| elements.checked_mul(mem::size_of::<half::f16>()))
            .ok_or(LowBitMetalError::SizeOverflow)?;
        validate_buffer_range(
            "activation",
            arena.length(),
            activation_offset,
            activation_bytes,
        )?;
        validate_buffer_range("output", arena.length(), output_offset, output_bytes)?;

        let pipeline = pipelines.get(self.kernel_name())?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(LowBitMetalError::CommandEncoderUnavailable)?;
        // SAFETY: all arena ranges, offsets, and scalar widths are validated
        // above; the descriptor constructor validated the packed layout.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(arena), activation_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(arena), self.packed_values_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(arena), self.scales_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(arena), output_offset, 3);
            set_u32(&encoder, &m, 4);
            set_u32(&encoder, &self.n, 5);
            set_u32(&encoder, &self.k, 6);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.n as usize,
                    height: m as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        Ok(())
    }
}

/// Validated and uploaded low-bit projection matrix.
///
/// Values and scales live in separate persistent shared buffers so the launch
/// path binds them directly and never reconstructs a dense matrix.
pub struct MetalLowBitProjection {
    format: AppleLowBitWeightFormat,
    n: u32,
    k: u32,
    packed_values: Retained<ProtocolObject<dyn MTLBuffer>>,
    scales: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalLowBitProjection {
    /// Validate and upload one `[N,K]` low-bit matrix.
    pub fn new(
        context: &MetalContext,
        weights: &PackedAppleLowBitWeights,
    ) -> LowBitMetalResult<Self> {
        weights.validate()?;
        let n = u32_dimension("N", weights.rows())?;
        let k = u32_dimension("K", weights.k())?;
        let values_bytes = weights.packed_values().len();
        let scales_bytes = weights
            .scales()
            .len()
            .checked_mul(mem::size_of::<half::f16>())
            .ok_or(LowBitMetalError::SizeOverflow)?;

        let packed_values = context
            .device()
            .newBufferWithLength_options(values_bytes, MTLResourceOptions::empty())
            .ok_or(LowBitMetalError::BufferAllocation {
                buffer: "packed weights",
                bytes: values_bytes,
            })?;
        let scales = context
            .device()
            .newBufferWithLength_options(scales_bytes, MTLResourceOptions::empty())
            .ok_or(LowBitMetalError::BufferAllocation {
                buffer: "FP16 scales",
                bytes: scales_bytes,
            })?;

        // SAFETY: both newly allocated shared buffers are exclusively owned by
        // this object and no command can reference them before construction
        // returns. Source and destination ranges have the exact validated size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                weights.packed_values().as_ptr(),
                packed_values.contents().as_ptr().cast::<u8>(),
                values_bytes,
            );
            let destination = scales.contents().as_ptr().cast::<u16>();
            for (index, scale) in weights.scales().iter().enumerate() {
                destination.add(index).write(scale.to_bits());
            }
        }

        Ok(Self {
            format: weights.format(),
            n,
            k,
            packed_values,
            scales,
        })
    }

    #[must_use]
    pub const fn format(&self) -> AppleLowBitWeightFormat {
        self.format
    }

    #[must_use]
    pub const fn shape(&self) -> [u32; 2] {
        [self.n, self.k]
    }

    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.packed_values
            .length()
            .saturating_add(self.scales.length())
    }

    #[must_use]
    pub const fn kernel_name(&self) -> &'static str {
        match self.format {
            AppleLowBitWeightFormat::W4A16 => "projection_w4a16_f16",
            AppleLowBitWeightFormat::W8A16 => "projection_w8a16_f16",
        }
    }

    /// Encode `C[M,N] = A[M,K] * W[N,K]^T` into an existing command buffer.
    ///
    /// This performs strict dimensional, range, and Metal-offset validation
    /// before creating an encoder. It does not commit or wait, allowing callers
    /// to compose the projection into an asynchronous inference step.
    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        pipelines: &PipelineCache,
        activations: &ProtocolObject<dyn MTLBuffer>,
        activation_offset: usize,
        output: &ProtocolObject<dyn MTLBuffer>,
        output_offset: usize,
        m: usize,
    ) -> LowBitMetalResult<()> {
        if m == 0 {
            return Err(LowBitMetalError::ZeroBatch);
        }
        let m = u32_dimension("M", m)?;
        validate_offset("activation", activation_offset)?;
        validate_offset("output", output_offset)?;

        let activation_bytes = (m as usize)
            .checked_mul(self.k as usize)
            .and_then(|elements| elements.checked_mul(mem::size_of::<half::f16>()))
            .ok_or(LowBitMetalError::SizeOverflow)?;
        let output_bytes = (m as usize)
            .checked_mul(self.n as usize)
            .and_then(|elements| elements.checked_mul(mem::size_of::<half::f16>()))
            .ok_or(LowBitMetalError::SizeOverflow)?;
        validate_buffer_range(
            "activation",
            activations.length(),
            activation_offset,
            activation_bytes,
        )?;
        validate_buffer_range("output", output.length(), output_offset, output_bytes)?;

        let pipeline = pipelines.get(self.kernel_name())?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(LowBitMetalError::CommandEncoderUnavailable)?;
        // SAFETY: every buffer range and scalar width is validated above. The
        // two persistent weight buffers exactly match the validated ABI.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(activations), activation_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.packed_values), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.scales), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), output_offset, 3);
            set_u32(&encoder, &m, 4);
            set_u32(&encoder, &self.n, 5);
            set_u32(&encoder, &self.k, 6);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.n as usize,
                    height: m as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        Ok(())
    }
}

impl fmt::Debug for MetalLowBitProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalLowBitProjection")
            .field("format", &self.format)
            .field("shape", &self.shape())
            .field("packed_bytes", &self.packed_values.length())
            .field("scale_bytes", &self.scales.length())
            .finish()
    }
}

fn u32_dimension(dimension: &'static str, value: usize) -> LowBitMetalResult<u32> {
    u32::try_from(value).map_err(|_| LowBitMetalError::DimensionTooLarge { dimension, value })
}

fn validate_offset(buffer: &'static str, offset: usize) -> LowBitMetalResult<()> {
    if offset % 4 != 0 {
        return Err(LowBitMetalError::InvalidOffsetAlignment { buffer, offset });
    }
    Ok(())
}

fn validate_buffer_range(
    buffer: &'static str,
    actual: usize,
    offset: usize,
    bytes: usize,
) -> LowBitMetalResult<()> {
    let required_end = offset
        .checked_add(bytes)
        .ok_or(LowBitMetalError::SizeOverflow)?;
    if required_end > actual {
        return Err(LowBitMetalError::BufferRange {
            buffer,
            required_end,
            actual,
        });
    }
    Ok(())
}

unsafe fn set_u32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &u32,
    index: usize,
) {
    encoder.setBytes_length_atIndex(
        NonNull::new_unchecked(value as *const u32 as *mut _),
        mem::size_of::<u32>(),
        index,
    );
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use half::f16;
    use objc2_metal::{MTLCommandQueue, MTLDevice};

    use super::*;
    use crate::arena::MetalBufferArena;
    use crate::kernels::KERNEL_SOURCE;
    use rvllm_apple::{project_apple_low_bit_reference, quantize_apple_low_bit_reference};

    fn test_context() -> LowBitMetalResult<(MetalContext, PipelineCache)> {
        let mut context = MetalContext::new()?;
        context.compile_library(KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile(&context, "projection_w4a16_f16")?;
        pipelines.compile(&context, "projection_w8a16_f16")?;
        Ok((context, pipelines))
    }

    fn shared_buffer(
        context: &MetalContext,
        bytes: usize,
        name: &'static str,
    ) -> LowBitMetalResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        context
            .device()
            .newBufferWithLength_options(bytes, MTLResourceOptions::empty())
            .ok_or(LowBitMetalError::BufferAllocation {
                buffer: name,
                bytes,
            })
    }

    unsafe fn write_f16(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[f16]) {
        let destination = buffer.contents().as_ptr().cast::<u16>();
        for (index, value) in values.iter().enumerate() {
            destination.add(index).write(value.to_bits());
        }
    }

    unsafe fn read_f16(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f16> {
        let source = buffer.contents().as_ptr().cast::<u16>();
        (0..count)
            .map(|index| f16::from_bits(source.add(index).read()))
            .collect()
    }

    #[test]
    fn metal_w4_w8_match_cpu_projection_across_tails_and_multirow() -> LowBitMetalResult<()> {
        let (context, pipelines) = test_context()?;
        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            for (m, n, k) in [
                (1usize, 1usize, 1usize),
                (1, 5, 31),
                (2, 3, 32),
                (3, 7, 33),
                (4, 5, 65),
            ] {
                let source = (0..n * k)
                    .map(|index| {
                        let wave = ((index * 29 + n * 7) % 61) as f32 - 30.0;
                        wave / 13.0
                    })
                    .collect::<Vec<_>>();
                let activations = (0..m * k)
                    .map(|index| {
                        let wave = ((index * 19 + m * 3) % 47) as f32 - 23.0;
                        f16::from_f32(wave / 17.0)
                    })
                    .collect::<Vec<_>>();
                let weights = quantize_apple_low_bit_reference(format, n, k, &source)?;
                let projection = MetalLowBitProjection::new(&context, &weights)?;
                let input = shared_buffer(
                    &context,
                    activations.len() * mem::size_of::<f16>(),
                    "test activation",
                )?;
                let output_count = m * n;
                let output = shared_buffer(
                    &context,
                    output_count * mem::size_of::<f16>(),
                    "test output",
                )?;
                unsafe {
                    write_f16(&input, &activations);
                    write_f16(&output, &vec![f16::NAN; output_count]);
                }

                let command_buffer = context
                    .queue()
                    .commandBuffer()
                    .ok_or(LowBitMetalError::CommandEncoderUnavailable)?;
                projection.encode(&command_buffer, &pipelines, &input, 0, &output, 0, m)?;
                command_buffer.commit();
                command_buffer.waitUntilCompleted();

                let expected = project_apple_low_bit_reference(&weights, &activations, m)?;
                let actual = unsafe { read_f16(&output, output_count) };
                for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                    let actual = actual.to_f32();
                    let expected = expected.to_f32();
                    let tolerance = 0.015 + expected.abs() * 0.002;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "{format:?} M={m} N={n} K={k} output[{index}] actual={actual} expected={expected} tolerance={tolerance}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn arena_offsets_match_cpu_projection_without_weight_buffers() -> LowBitMetalResult<()> {
        let (context, pipelines) = test_context()?;
        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            let (m, n, k) = (3_usize, 7_usize, 33_usize);
            let source = (0..n * k)
                .map(|index| ((index * 31 % 67) as f32 - 33.0) / 9.0)
                .collect::<Vec<_>>();
            let activations = (0..m * k)
                .map(|index| f16::from_f32(((index * 17 % 43) as f32 - 21.0) / 11.0))
                .collect::<Vec<_>>();
            let weights = quantize_apple_low_bit_reference(format, n, k, &source)?;
            let scale_bytes = weights
                .scales()
                .len()
                .checked_mul(mem::size_of::<f16>())
                .ok_or(LowBitMetalError::SizeOverflow)?;
            let mut arena = MetalBufferArena::new(context.device(), 64 * 1024)?;
            let values = arena
                .region("test_low_bit_values", weights.packed_values().len(), 16)
                .map_err(LowBitMetalError::Metal)?;
            let scales = arena
                .region("test_low_bit_scales", scale_bytes, 16)
                .map_err(LowBitMetalError::Metal)?;
            let input = arena
                .region(
                    "test_low_bit_input",
                    activations.len() * mem::size_of::<f16>(),
                    16,
                )
                .map_err(LowBitMetalError::Metal)?;
            let output = arena
                .region("test_low_bit_output", m * n * mem::size_of::<f16>(), 16)
                .map_err(LowBitMetalError::Metal)?;
            unsafe {
                arena
                    .write_region(&values, weights.packed_values())
                    .map_err(LowBitMetalError::Metal)?;
                let scale_destination = arena.host_ptr(&scales).cast::<u16>();
                for (index, scale) in weights.scales().iter().enumerate() {
                    scale_destination.add(index).write(scale.to_bits());
                }
                let input_destination = arena.host_ptr(&input).cast::<u16>();
                for (index, value) in activations.iter().enumerate() {
                    input_destination.add(index).write(value.to_bits());
                }
            }
            let projection = MetalLowBitProjectionOffsets::new(
                format,
                n,
                k,
                values.offset,
                values.size,
                scales.offset,
                scales.size,
            )?;
            let command_buffer = context
                .queue()
                .commandBuffer()
                .ok_or(LowBitMetalError::CommandEncoderUnavailable)?;
            projection.encode(
                &command_buffer,
                &pipelines,
                arena.buffer(),
                input.offset,
                output.offset,
                m,
            )?;
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            let expected = project_apple_low_bit_reference(&weights, &activations, m)?;
            let actual = unsafe {
                std::slice::from_raw_parts(arena.host_ptr(&output).cast::<u16>(), m * n)
                    .iter()
                    .map(|bits| f16::from_bits(*bits))
                    .collect::<Vec<_>>()
            };
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                let expected = expected.to_f32();
                let tolerance = 0.015 + expected.abs() * 0.002;
                assert!((actual.to_f32() - expected).abs() <= tolerance);
            }
        }
        Ok(())
    }

    #[test]
    fn encoder_rejects_short_ranges_and_unaligned_offsets() -> LowBitMetalResult<()> {
        let (context, pipelines) = test_context()?;
        let weights =
            quantize_apple_low_bit_reference(AppleLowBitWeightFormat::W4A16, 2, 33, &[1.0; 66])?;
        let projection = MetalLowBitProjection::new(&context, &weights)?;
        let short_input = shared_buffer(&context, 4, "short input")?;
        let output = shared_buffer(&context, 4, "output")?;
        let command_buffer = context
            .queue()
            .commandBuffer()
            .ok_or(LowBitMetalError::CommandEncoderUnavailable)?;

        assert!(matches!(
            projection.encode(&command_buffer, &pipelines, &short_input, 0, &output, 0, 1,),
            Err(LowBitMetalError::BufferRange {
                buffer: "activation",
                ..
            })
        ));
        assert!(matches!(
            projection.encode(&command_buffer, &pipelines, &short_input, 2, &output, 0, 1,),
            Err(LowBitMetalError::InvalidOffsetAlignment {
                buffer: "activation",
                offset: 2,
            })
        ));
        Ok(())
    }
}
