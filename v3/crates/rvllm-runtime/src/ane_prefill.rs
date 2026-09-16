//! Synchronized Metal page capture for ANE decode. This is a host-side adapter:
//! it does not initialize ANE or claim that a decoder consumed the snapshot.

use crate::paged_kv::APPLE_KV_PAGE_SIZE;
use crate::paged_prompt_cache::{KvPageIo, KvPageIoError};
use half::{bf16, f16};
use rvllm_core::{BlockId, ReqId, RvllmError, TokenId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillScalarType {
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillLayerShape {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
}

#[derive(Debug)]
pub struct PrefillLayerKv {
    pub shape: PrefillLayerShape,
    /// [token, KV head, dimension], K already rotated and V unrotated.
    pub keys: Vec<f16>,
    pub values: Vec<f16>,
}

#[derive(Debug)]
pub struct AnePrefillSnapshot {
    pub tokens: usize,
    pub source_dtype: PrefillScalarType,
    pub metal_numeric_abi: [u8; 32],
    pub layers: Vec<PrefillLayerKv>,
}

/// The first output token was sampled from the final Metal prefill row. If it
/// is not EOS, ANE consumes it at `cache.tokens` and appends exactly one KV row.
/// The prompt's last input token must not be decoded again.
#[derive(Debug)]
pub struct AneDecodeStart {
    pub req_id: ReqId,
    pub first_token: TokenId,
    pub cache: AnePrefillSnapshot,
    pub times: AnePrefillTimes,
}

/// Wall-clock scopes for the handoff. Waiting for a command buffer is not
/// equivalent to a GPU-only timer: it also includes scheduling delays.
#[derive(Clone, Copy, Debug)]
pub struct AnePrefillTimes {
    /// Reserve, encode, submit and collect the complete Metal prefill.
    pub metal_execution_ms: f64,
    /// Metal GPUStartTime/GPUEndTime interval, after completion. Still sensitive
    /// to device clocks and contention; absent when Metal supplies no timestamp.
    pub gpu_execution_ms: Option<f64>,
    /// Step wall time excluding explicit command-buffer waits; includes host
    /// scheduling and bookkeeping, not exclusively kernel encoding.
    pub host_non_wait_ms: f64,
    pub command_buffer_wait_ms: f64,
    /// Synchronous page capture and conversion into owned FP16 vectors.
    pub kv_capture_ms: f64,
}

impl AneDecodeStart {
    pub fn next_position(&self) -> usize {
        self.cache.tokens
    }
}

#[derive(Debug)]
pub enum AnePrefillError {
    Execution(RvllmError),
    Cache(KvPageIoError),
    Invalid(&'static str),
}

impl std::fmt::Display for AnePrefillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(error) => write!(f, "Metal prefill: {error}"),
            Self::Cache(error) => write!(f, "ANE prefill cache: {error}"),
            Self::Invalid(reason) => write!(f, "ANE prefill: {reason}"),
        }
    }
}

impl std::error::Error for AnePrefillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::Cache(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<RvllmError> for AnePrefillError {
    fn from(error: RvllmError) -> Self {
        Self::Execution(error)
    }
}

impl From<KvPageIoError> for AnePrefillError {
    fn from(error: KvPageIoError) -> Self {
        Self::Cache(error)
    }
}

/// Preserve the supplied logical page order, including a partial final page.
/// The backend must reject access while GPU commands own its KV memory.
pub(crate) fn capture_prefill(
    io: &mut dyn KvPageIo,
    pages: &[BlockId],
    tokens: usize,
    shapes: &[PrefillLayerShape],
    source_dtype: PrefillScalarType,
    metal_numeric_abi: [u8; 32],
) -> Result<AnePrefillSnapshot, KvPageIoError> {
    let page_size = APPLE_KV_PAGE_SIZE as usize;
    if tokens == 0 || tokens.div_ceil(page_size) != pages.len() || shapes.is_empty() {
        return Err(KvPageIoError::LayoutMismatch(
            "prefill token/page/layer count mismatch",
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(pages.len());
    if pages.iter().any(|page| !seen.insert(*page)) {
        return Err(KvPageIoError::LayoutMismatch(
            "prefill logical page table contains duplicate physical pages",
        ));
    }
    let mut expected_page_bytes = 0_usize;
    let mut widths = Vec::with_capacity(shapes.len());
    for shape in shapes {
        if shape.query_heads == 0
            || shape.kv_heads == 0
            || shape.head_dim == 0
            || shape.query_heads % shape.kv_heads != 0
            || shape.sliding_window == Some(0)
        {
            return Err(KvPageIoError::LayoutMismatch(
                "prefill layer has invalid attention geometry",
            ));
        }
        let width = shape
            .kv_heads
            .checked_mul(shape.head_dim)
            .ok_or(KvPageIoError::LayoutMismatch("prefill width overflow"))?;
        expected_page_bytes = width
            .checked_mul(page_size * 4)
            .and_then(|n| expected_page_bytes.checked_add(n))
            .ok_or(KvPageIoError::LayoutMismatch("prefill page size overflow"))?;
        widths.push(width);
    }
    let actual_page_bytes = io.page_bytes().ok_or(KvPageIoError::BackendUnavailable)?;
    if actual_page_bytes != expected_page_bytes {
        return Err(KvPageIoError::InvalidPageBytes {
            expected: expected_page_bytes,
            got: actual_page_bytes,
        });
    }
    let mut layers = Vec::with_capacity(shapes.len());
    for (&shape, &width) in shapes.iter().zip(&widths) {
        let count = width
            .checked_mul(tokens)
            .ok_or(KvPageIoError::LayoutMismatch(
                "prefill tensor size overflow",
            ))?;
        let allocate = || {
            let mut values = Vec::new();
            values.try_reserve_exact(count).map_err(|e| {
                KvPageIoError::DeviceCopyFailed(format!("allocate prefill snapshot: {e}"))
            })?;
            values.resize(count, f16::ZERO);
            Ok::<_, KvPageIoError>(values)
        };
        layers.push(PrefillLayerKv {
            shape,
            keys: allocate()?,
            values: allocate()?,
        });
    }
    for (logical_page, &physical_page) in pages.iter().enumerate() {
        let bytes = io.capture_page(physical_page)?;
        if bytes.len() != expected_page_bytes {
            return Err(KvPageIoError::InvalidPageBytes {
                expected: expected_page_bytes,
                got: bytes.len(),
            });
        }
        let start_token = logical_page * page_size;
        let valid_rows = (tokens - start_token).min(page_size);
        let mut offset = 0;
        for (layer, &width) in layers.iter_mut().zip(&widths) {
            let tensor_page_bytes = width * page_size * 2;
            for output in [&mut layer.keys, &mut layer.values] {
                let destination =
                    &mut output[start_token * width..(start_token + valid_rows) * width];
                let valid_bytes = &bytes[offset..offset + valid_rows * width * 2];
                for (dst, src) in destination.iter_mut().zip(valid_bytes.chunks_exact(2)) {
                    let bits = u16::from_le_bytes([src[0], src[1]]);
                    let value = match source_dtype {
                        PrefillScalarType::F16 => f16::from_bits(bits),
                        PrefillScalarType::Bf16 => f16::from_f32(bf16::from_bits(bits).to_f32()),
                    };
                    if !value.is_finite() {
                        return Err(KvPageIoError::LayoutMismatch(
                            "prefill KV value is nonfinite or outside FP16 range",
                        ));
                    }
                    *dst = value;
                }
                offset += tensor_page_bytes;
            }
        }
    }
    Ok(AnePrefillSnapshot {
        tokens,
        source_dtype,
        metal_numeric_abi,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::CowPageCopy;
    use std::sync::Arc;

    struct Pages {
        bytes: Vec<Arc<[u8]>>,
        busy: bool,
    }
    impl KvPageIo for Pages {
        fn page_bytes(&self) -> Option<usize> {
            self.bytes.first().map(|b| b.len())
        }
        fn capture_page(&mut self, page: BlockId) -> Result<Arc<[u8]>, KvPageIoError> {
            if self.busy {
                return Err(KvPageIoError::BackendBusy);
            }
            self.bytes
                .get(page.0 as usize)
                .cloned()
                .ok_or(KvPageIoError::InvalidPageId {
                    page: page.0,
                    total_pages: self.bytes.len() as u32,
                })
        }
        fn restore_page(&mut self, _: BlockId, _: &[u8]) -> Result<(), KvPageIoError> {
            unreachable!()
        }
        fn copy_page(&mut self, _: CowPageCopy) -> Result<(), KvPageIoError> {
            unreachable!()
        }
    }

    #[test]
    fn page_order_partial_tail_and_both_scalar_formats_survive_handoff() {
        let shapes = [
            PrefillLayerShape {
                query_heads: 4,
                kv_heads: 2,
                head_dim: 2,
                sliding_window: Some(1024),
            },
            PrefillLayerShape {
                query_heads: 4,
                kv_heads: 1,
                head_dim: 2,
                sliding_window: None,
            },
        ];
        for dtype in [PrefillScalarType::F16, PrefillScalarType::Bf16] {
            let mut pages = Pages {
                bytes: Vec::new(),
                busy: false,
            };
            for page in 0..3 {
                let mut bytes = Vec::new();
                for (layer, width) in [4, 2].into_iter().enumerate() {
                    for tensor in 0..2 {
                        for token in 0..32 {
                            for channel in 0..width {
                                // Poison unused rows of the last logical page.
                                let value = if page == 0 && token >= 3 {
                                    f32::NAN
                                } else {
                                    (page * 100 + layer * 30 + tensor * 10 + token + channel) as f32
                                        / 8.0
                                };
                                let bits = match dtype {
                                    PrefillScalarType::F16 => f16::from_f32(value).to_bits(),
                                    PrefillScalarType::Bf16 => bf16::from_f32(value).to_bits(),
                                };
                                bytes.extend_from_slice(&bits.to_le_bytes());
                            }
                        }
                    }
                }
                pages.bytes.push(bytes.into());
            }
            let snapshot = capture_prefill(
                &mut pages,
                &[BlockId(2), BlockId(0)],
                35,
                &shapes,
                dtype,
                [7; 32],
            )
            .unwrap();
            assert_eq!(snapshot.tokens, 35);
            assert_eq!(snapshot.metal_numeric_abi, [7; 32]);
            for (layer, kv) in snapshot.layers.iter().enumerate() {
                let width = kv.shape.kv_heads * kv.shape.head_dim;
                assert_eq!(kv.keys.len(), 35 * width);
                for token in 0..35 {
                    for channel in 0..width {
                        for (tensor, values) in [&kv.keys, &kv.values].into_iter().enumerate() {
                            let page = if token < 32 { 2 } else { 0 };
                            let expected =
                                (page * 100 + layer * 30 + tensor * 10 + token % 32 + channel)
                                    as f32
                                    / 8.0;
                            let expected = match dtype {
                                PrefillScalarType::F16 => f16::from_f32(expected),
                                PrefillScalarType::Bf16 => {
                                    f16::from_f32(bf16::from_f32(expected).to_f32())
                                }
                            };
                            assert_eq!(values[token * width + channel], expected);
                        }
                    }
                }
            }
            assert!(
                capture_prefill(&mut pages, &[BlockId(0)], 4, &shapes, dtype, [0; 32]).is_err()
            );
            assert!(capture_prefill(
                &mut pages,
                &[BlockId(2), BlockId(2)],
                35,
                &shapes,
                dtype,
                [0; 32]
            )
            .is_err());
            pages.busy = true;
            assert!(matches!(
                capture_prefill(&mut pages, &[BlockId(2)], 1, &shapes, dtype, [0; 32]),
                Err(KvPageIoError::BackendBusy)
            ));
        }
    }
}
