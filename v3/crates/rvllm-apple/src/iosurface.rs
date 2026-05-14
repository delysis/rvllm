use rvllm_core::{AppleCtx, AppleError, DType, Result, RvllmError};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct IoSurfaceTensorDesc {
    pub dtype: DType,
    pub channels: usize,
    pub spatial: usize,
}

impl IoSurfaceTensorDesc {
    #[must_use]
    pub const fn new(dtype: DType, channels: usize, spatial: usize) -> Self {
        Self {
            dtype,
            channels,
            spatial,
        }
    }

    #[must_use]
    pub const fn element_count(self) -> usize {
        self.channels * self.spatial
    }

    #[must_use]
    pub const fn bytes(self) -> usize {
        self.element_count() * self.dtype.bytes()
    }

    pub fn try_bytes(self) -> Result<usize> {
        let elements = self.channels.checked_mul(self.spatial).ok_or_else(|| {
            iosurface_err(
                AppleError::IoSurfaceSizeOverflow {
                    dtype: self.dtype,
                    channels: self.channels,
                    spatial: self.spatial,
                },
                "bytes",
            )
        })?;
        elements.checked_mul(self.dtype.bytes()).ok_or_else(|| {
            iosurface_err(
                AppleError::IoSurfaceSizeOverflow {
                    dtype: self.dtype,
                    channels: self.channels,
                    spatial: self.spatial,
                },
                "bytes",
            )
        })
    }

    pub fn validate(self) -> Result<()> {
        if self.channels == 0 || self.spatial == 0 {
            return Err(iosurface_err(
                AppleError::IoSurfaceInvalidDesc {
                    dtype: self.dtype,
                    channels: self.channels,
                    spatial: self.spatial,
                },
                "validate_desc",
            ));
        }
        self.try_bytes().map(|_| ())
    }

    #[must_use]
    pub const fn standalone_strides(self) -> PackedFieldStrides {
        let elem_bytes = self.dtype.bytes();
        PackedFieldStrides {
            channel_stride_bytes: self.spatial * elem_bytes,
            spatial_stride_bytes: elem_bytes,
        }
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PackedFieldStrides {
    pub channel_stride_bytes: usize,
    pub spatial_stride_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedField {
    pub name: String,
    pub desc: IoSurfaceTensorDesc,
}

impl PackedField {
    #[must_use]
    pub fn new(name: impl Into<String>, desc: IoSurfaceTensorDesc) -> Self {
        Self {
            name: name.into(),
            desc,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedFieldLayout {
    pub name: String,
    pub desc: IoSurfaceTensorDesc,
    pub spatial_offset: usize,
    pub base_byte_offset: usize,
    pub channel_stride_bytes: usize,
    pub spatial_stride_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedInputLayout {
    pub dtype: DType,
    pub channels: usize,
    pub spatial: usize,
    pub fields: Vec<PackedFieldLayout>,
}

impl PackedInputLayout {
    pub fn pack_spatial(dtype: DType, channels: usize, fields: Vec<PackedField>) -> Result<Self> {
        Self::single_input_spatial(dtype, channels, fields)
    }

    pub fn single_input_spatial(
        dtype: DType,
        channels: usize,
        fields: Vec<PackedField>,
    ) -> Result<Self> {
        if fields.is_empty() {
            return Err(iosurface_err(
                AppleError::IoSurfacePackEmpty,
                "pack_spatial",
            ));
        }

        let mut spatial = 0usize;
        let mut offsets = Vec::with_capacity(fields.len());
        for i in 0..fields.len() {
            let field = &fields[i];
            if field.name.is_empty() {
                return Err(iosurface_err(
                    AppleError::IoSurfacePackUnnamedField { field: i },
                    "pack_spatial",
                ));
            }
            if fields[..i].iter().any(|prior| prior.name == field.name) {
                return Err(iosurface_err(
                    AppleError::IoSurfacePackDuplicateField { field: i },
                    "pack_spatial",
                ));
            }
            field.desc.validate()?;
            if field.desc.dtype != dtype || field.desc.channels != channels {
                return Err(iosurface_err(
                    AppleError::IoSurfacePackFieldMismatch {
                        field: i,
                        expected_dtype: dtype,
                        actual_dtype: field.desc.dtype,
                        expected_channels: channels,
                        actual_channels: field.desc.channels,
                    },
                    "pack_spatial",
                ));
            }
            offsets.push(spatial);
            spatial = spatial.checked_add(field.desc.spatial).ok_or_else(|| {
                iosurface_err(
                    AppleError::IoSurfaceSizeOverflow {
                        dtype,
                        channels,
                        spatial: field.desc.spatial,
                    },
                    "pack_spatial",
                )
            })?;
        }

        let desc = IoSurfaceTensorDesc::new(dtype, channels, spatial);
        desc.validate()?;
        let elem_bytes = dtype.bytes();
        let channel_stride_bytes = spatial.checked_mul(elem_bytes).ok_or_else(|| {
            iosurface_err(
                AppleError::IoSurfaceSizeOverflow {
                    dtype,
                    channels,
                    spatial,
                },
                "pack_spatial",
            )
        })?;

        let mut packed_fields = Vec::with_capacity(fields.len());
        for (field, spatial_offset) in fields.into_iter().zip(offsets) {
            let base_byte_offset = spatial_offset.checked_mul(elem_bytes).ok_or_else(|| {
                iosurface_err(
                    AppleError::IoSurfaceSizeOverflow {
                        dtype,
                        channels,
                        spatial,
                    },
                    "pack_spatial",
                )
            })?;
            packed_fields.push(PackedFieldLayout {
                name: field.name,
                desc: field.desc,
                spatial_offset,
                base_byte_offset,
                channel_stride_bytes,
                spatial_stride_bytes: elem_bytes,
            });
        }

        Ok(Self {
            dtype,
            channels,
            spatial,
            fields: packed_fields,
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

    #[must_use]
    pub const fn surface_shape(&self) -> ByteSurfaceShape {
        self.desc().byte_surface_shape()
    }

    #[must_use]
    pub fn field(&self, name: &str) -> Option<&PackedFieldLayout> {
        self.fields.iter().find(|field| field.name == name)
    }

    #[must_use]
    pub fn field_byte_offset(&self, name: &str, channel: usize, spatial: usize) -> Option<usize> {
        let field = self.field(name)?;
        if channel >= field.desc.channels || spatial >= field.desc.spatial {
            return None;
        }
        let channel_offset = channel.checked_mul(field.channel_stride_bytes)?;
        let spatial_offset = spatial.checked_mul(field.spatial_stride_bytes)?;
        field
            .base_byte_offset
            .checked_add(channel_offset)?
            .checked_add(spatial_offset)
    }
}

fn iosurface_err(err: AppleError, op: &'static str) -> RvllmError {
    RvllmError::apple(
        err,
        AppleCtx {
            backend: "iosurface",
            op,
            device: "host",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{AppleError, RvllmError};

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
            PackedField::new("activations", IoSurfaceTensorDesc::new(DType::F16, 1024, 4)),
            PackedField::new("metadata", IoSurfaceTensorDesc::new(DType::F16, 1024, 1)),
        ];
        let layout = match PackedInputLayout::pack_spatial(DType::F16, 1024, fields) {
            Ok(v) => v,
            Err(e) => panic!("layout should be valid: {e}"),
        };
        assert_eq!(layout.spatial, 5);
        assert_eq!(layout.desc().bytes(), 1024 * 5 * 2);

        let err = match PackedInputLayout::pack_spatial(
            DType::F16,
            1024,
            vec![PackedField::new(
                "metadata",
                IoSurfaceTensorDesc::new(DType::F32, 1024, 1),
            )],
        ) {
            Ok(_) => panic!("layout should reject mixed dtype"),
            Err(e) => e,
        };
        match err {
            RvllmError::Apple {
                err:
                    AppleError::IoSurfacePackFieldMismatch {
                        field: 0,
                        expected_dtype: DType::F16,
                        actual_dtype: DType::F32,
                        expected_channels: 1024,
                        actual_channels: 1024,
                    },
                ..
            } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn single_input_spatial_pack_records_field_offsets_and_strides() {
        let fields = vec![
            PackedField::new("tokens", IoSurfaceTensorDesc::new(DType::F16, 4, 2)),
            PackedField::new("positions", IoSurfaceTensorDesc::new(DType::F16, 4, 1)),
            PackedField::new("scratch", IoSurfaceTensorDesc::new(DType::F16, 4, 3)),
        ];
        let layout = match PackedInputLayout::single_input_spatial(DType::F16, 4, fields) {
            Ok(v) => v,
            Err(e) => panic!("layout should be valid: {e}"),
        };

        assert_eq!(layout.desc(), IoSurfaceTensorDesc::new(DType::F16, 4, 6));
        assert_eq!(layout.surface_shape().alloc_size, 4 * 6 * 2);
        assert_eq!(layout.fields[0].spatial_offset, 0);
        assert_eq!(layout.fields[1].spatial_offset, 2);
        assert_eq!(layout.fields[2].spatial_offset, 3);
        assert_eq!(layout.fields[1].base_byte_offset, 2 * 2);
        assert_eq!(layout.fields[1].channel_stride_bytes, 6 * 2);
        assert_eq!(layout.fields[1].spatial_stride_bytes, 2);
        assert_eq!(
            layout.field_byte_offset("positions", 3, 0),
            Some(((3 * 6) + 2) * 2)
        );
    }
}
