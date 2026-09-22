//! Source-level packed vector declaration, NOT verified compiled ANE strides.
//! The driver boundary currently trusts I/O sizes supplied by the caller. There
//! is intentionally no runtime selector or private-API consumer for this plan
//! until the local owner verifies compiled I/O descriptors for this graph.
#![forbid(unsafe_code)]
use half::f16;

#[derive(Clone, Copy, Debug)]
pub struct Packed32Declaration;

impl Packed32Declaration {
    pub const fn gemma12b() -> Self {
        Self
    }
    pub const fn shape(self) -> [usize; 4] {
        [1, 120, 1, 32]
    }
    /// Logical bytes under the declared contiguous-row layout, not a driver
    /// allocation measurement. No public constructor admits other geometry.
    pub const fn logical_bytes(self) -> usize {
        7680
    }

    pub fn pack(self, input: &[f16], output: &mut [u8]) -> Result<(), String> {
        if input.len() != 3840 || output.len() != self.logical_bytes() {
            return Err("packed32 declaration requires 3840 halves / 7680 bytes".into());
        }
        for (x, slot) in input.iter().zip(output.chunks_exact_mut(2)) {
            slot.copy_from_slice(&x.to_le_bytes());
        }
        Ok(())
    }

    pub fn unpack(self, input: &[u8], output: &mut [f16]) -> Result<(), String> {
        if input.len() != self.logical_bytes() || output.len() != 3840 {
            return Err("packed32 declaration requires 7680 bytes / 3840 halves".into());
        }
        for (slot, x) in input.chunks_exact(2).zip(output) {
            *x = f16::from_le_bytes([slot[0], slot[1]]);
        }
        Ok(())
    }
}
