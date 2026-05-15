use rvllm_core::DType;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct IoSurfaceTensorDesc {
    pub dtype: DType,
    pub channels: usize,
    pub spatial: usize,
}

impl IoSurfaceTensorDesc {
    #[must_use]
    pub const fn element_count(self) -> usize {
        self.channels * self.spatial
    }

    #[must_use]
    pub const fn bytes(self) -> usize {
        self.element_count() * self.dtype.bytes()
    }

    /// Byte-addressed IOSurface layout used by the ANE references:
    /// width=bytes, height=1, bytes-per-element=1, bytes-per-row=bytes.
    #[must_use]
    pub const fn byte_surface_shape(self) -> ByteSurfaceShape {
        let bytes = self.bytes();
        ByteSurfaceShape {
            width: bytes,
            height: 1,
            bytes_per_element: 1,
            bytes_per_row: bytes,
            alloc_size: bytes,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ByteSurfaceShape {
    pub width: usize,
    pub height: usize,
    pub bytes_per_element: usize,
    pub bytes_per_row: usize,
    pub alloc_size: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedField {
    pub name: String,
    pub desc: IoSurfaceTensorDesc,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedInputLayout {
    pub dtype: DType,
    pub channels: usize,
    pub spatial: usize,
    pub fields: Vec<PackedField>,
}

impl PackedInputLayout {
    #[must_use]
    pub fn pack_spatial(dtype: DType, channels: usize, fields: Vec<PackedField>) -> Option<Self> {
        if fields
            .iter()
            .any(|f| f.desc.dtype != dtype || f.desc.channels != channels)
        {
            return None;
        }
        let spatial = fields.iter().map(|f| f.desc.spatial).sum();
        Some(Self {
            dtype,
            channels,
            spatial,
            fields,
        })
    }

    #[must_use]
    pub const fn desc(&self) -> IoSurfaceTensorDesc {
        IoSurfaceTensorDesc {
            dtype: self.dtype,
            channels: self.channels,
            spatial: self.spatial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_surface_shape_matches_ane_runtime_convention() {
        let d = IoSurfaceTensorDesc {
            dtype: DType::F16,
            channels: 2048,
            spatial: 8,
        };
        let s = d.byte_surface_shape();
        assert_eq!(d.bytes(), 2048 * 8 * 2);
        assert_eq!(s.width, d.bytes());
        assert_eq!(s.height, 1);
        assert_eq!(s.bytes_per_element, 1);
        assert_eq!(s.bytes_per_row, d.bytes());
    }

    #[test]
    fn single_input_spatial_pack_requires_same_dtype_and_channels() {
        let fields = vec![
            PackedField {
                name: "activations".to_owned(),
                desc: IoSurfaceTensorDesc {
                    dtype: DType::F16,
                    channels: 1024,
                    spatial: 4,
                },
            },
            PackedField {
                name: "metadata".to_owned(),
                desc: IoSurfaceTensorDesc {
                    dtype: DType::F16,
                    channels: 1024,
                    spatial: 1,
                },
            },
        ];
        let layout = match PackedInputLayout::pack_spatial(DType::F16, 1024, fields) {
            Some(v) => v,
            None => panic!("layout should be valid"),
        };
        assert_eq!(layout.spatial, 5);
        assert_eq!(layout.desc().bytes(), 1024 * 5 * 2);
    }
}
