//! Metal buffer arena: persistent buffer allocation with zero hot-path alloc.
//!
//! All inference buffers are pre-allocated from a single large MTLBuffer
//! (shared storage mode for unified memory). Sub-regions are carved out
//! via offset tracking. No allocation happens during command encoding.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx { backend: "metal-arena", op, device: "apple-silicon" }
}

/// A named sub-region within the arena buffer.
#[derive(Clone, Debug)]
pub struct MetalRegion {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Persistent Metal buffer arena. One large MTLBuffer, sub-allocated
/// at init time. Zero allocations during inference.
pub struct MetalBufferArena {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity: usize,
    cursor: usize,
    regions: Vec<MetalRegion>,
}

impl MetalBufferArena {
    /// Create a new arena backed by a single shared-mode MTLBuffer.
    ///
    /// Shared storage mode is optimal for Apple Silicon unified memory:
    /// both CPU and GPU access the same physical memory with no copies.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, capacity_bytes: usize) -> Result<Self> {
        // MTLResourceOptions::empty() gives shared storage mode + default hazard tracking
        let buffer = device
            .newBufferWithLength_options(capacity_bytes, MTLResourceOptions::empty())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::IoSurfaceFailed { bytes: capacity_bytes },
                    ctx("alloc"),
                )
            })?;

        tracing::info!(
            capacity_mb = capacity_bytes / (1024 * 1024),
            "Metal buffer arena allocated"
        );

        Ok(Self {
            buffer,
            capacity: capacity_bytes,
            cursor: 0,
            regions: Vec::with_capacity(64),
        })
    }

    /// Allocate a named region from the arena. Alignment must be a power
    /// of two. Returns the region descriptor with its offset.
    pub fn region(&mut self, name: &str, bytes: usize, align: usize) -> Result<MetalRegion> {
        assert!(align.is_power_of_two(), "alignment must be power of 2");
        let aligned_cursor = (self.cursor + align - 1) & !(align - 1);
        let end = aligned_cursor + bytes;
        if end > self.capacity {
            return Err(RvllmError::apple(
                AppleError::IoSurfaceFailed { bytes },
                ctx("region"),
            ));
        }
        let region = MetalRegion {
            name: name.to_owned(),
            offset: aligned_cursor,
            size: bytes,
        };
        self.cursor = end;
        self.regions.push(region.clone());
        Ok(region)
    }

    /// Get a raw CPU pointer to a region for host writes (weight loading).
    ///
    /// # Safety
    /// Caller must ensure no GPU commands are reading this region.
    pub unsafe fn host_ptr(&self, region: &MetalRegion) -> *mut u8 {
        let base = self.buffer.contents().as_ptr() as *mut u8;
        base.add(region.offset)
    }

    /// Write data to a region from host memory.
    ///
    /// # Safety
    /// Caller must ensure no GPU commands are reading this region.
    pub unsafe fn write_region(&self, region: &MetalRegion, data: &[u8]) -> Result<()> {
        if data.len() > region.size {
            return Err(RvllmError::apple(
                AppleError::IoSurfaceFailed { bytes: data.len() },
                ctx("write_region"),
            ));
        }
        let dst = self.host_ptr(region);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        Ok(())
    }

    /// Get the underlying MTLBuffer for binding to compute encoders.
    #[inline]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> { &*self.buffer }

    #[inline]
    pub fn buffer_retained(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> { &self.buffer }

    /// Total bytes allocated so far.
    #[inline]
    pub fn allocated(&self) -> usize { self.cursor }

    /// Total capacity.
    #[inline]
    pub fn capacity(&self) -> usize { self.capacity }

    /// Remaining bytes.
    #[inline]
    pub fn remaining(&self) -> usize { self.capacity - self.cursor }

    /// List all allocated regions (for debugging/audit).
    pub fn regions(&self) -> &[MetalRegion] { &self.regions }

    /// Reset the arena (reuse for a different model/config).
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.regions.clear();
    }
}

impl std::fmt::Debug for MetalBufferArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBufferArena")
            .field("capacity_mb", &(self.capacity / (1024 * 1024)))
            .field("allocated_mb", &(self.cursor / (1024 * 1024)))
            .field("regions", &self.regions.len())
            .finish()
    }
}
