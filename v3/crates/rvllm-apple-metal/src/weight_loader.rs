//! Weight loader: load safetensors model weights into Metal buffers.
//!
//! Handles BF16 → F16 conversion (Metal 3 / Apple9 has no native BF16
//! compute). Weights are loaded via mmap and converted in-place or
//! via the bf16_to_f16 Metal kernel.

use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::arena::{MetalBufferArena, MetalRegion};
use crate::context::MetalContext;
use rvllm_core::{AppleCtx, AppleError, DType, Result, RvllmError};

#[derive(Clone, Debug)]
pub struct SafetensorTensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub file: PathBuf,
    pub file_offset: usize,
    pub nbytes: usize,
}

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "metal-loader",
        op,
        device: "apple-silicon",
    }
}

/// Describes a loaded model's weight layout in the arena.
#[derive(Clone, Debug)]
pub struct MetalModelWeights {
    pub layers: Vec<MetalLayerWeightRegions>,
    pub final_norm: MetalRegion,
    pub lm_head: MetalRegion,
    pub rope_cos: MetalRegion,
    pub rope_sin: MetalRegion,
}

#[derive(Clone, Debug)]
pub struct MetalLayerWeightRegions {
    pub attn_norm: MetalRegion,
    pub qkv: MetalRegion,
    pub qkv_bias: Option<MetalRegion>,
    pub o_proj: MetalRegion,
    pub mlp_norm: MetalRegion,
    pub gate_up: MetalRegion,
    pub down_proj: MetalRegion,
}

#[derive(Clone, Debug)]
pub struct MetalLayerWeightNames {
    pub attn_norm: String,
    pub qkv: String,
    pub qkv_bias: Option<String>,
    pub o_proj: String,
    pub mlp_norm: String,
    pub gate_up: String,
    pub down_proj: String,
}

#[derive(Clone, Debug)]
pub struct MetalModelWeightNames {
    pub layers: Vec<MetalLayerWeightNames>,
    pub final_norm: String,
    pub lm_head: String,
    pub rope_cos: String,
    pub rope_sin: String,
}

pub fn bf16_to_f16_cpu(bf16_data: &[u8]) -> Vec<u8> {
    let count = bf16_data.len() / 2;
    let mut f16_data = vec![0u8; count * 2];

    for i in 0..count {
        let bf16_bits = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        let f16_val = half::f16::from_f32(f32_val);
        let f16_bits = f16_val.to_bits();
        f16_data[i * 2] = f16_bits as u8;
        f16_data[i * 2 + 1] = (f16_bits >> 8) as u8;
    }
    f16_data
}

pub fn parse_safetensors_index(model_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![("model.safetensors".to_owned(), single)]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(RvllmError::apple(
            AppleError::MetallibMissing { path: index_path },
            ctx("parse_index"),
        ));
    }

    let index_bytes = std::fs::read(&index_path).map_err(|e| RvllmError::Io {
        err: rvllm_core::IoError::from(&e),
        path: index_path.clone(),
        source: e,
    })?;

    let index: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|_| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "invalid index json",
            },
            ctx("parse_index"),
        )
    })?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing weight_map",
                },
                ctx("parse_index"),
            )
        })?;

    let mut files = Vec::new();
    let mut seen = HashSet::new();
    for (_tensor_name, file_val) in weight_map {
        if let Some(file_name) = file_val.as_str() {
            if seen.insert(file_name.to_owned()) {
                files.push((file_name.to_owned(), model_dir.join(file_name)));
            }
        }
    }
    Ok(files)
}

fn map_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F8_E4M3" | "F8E4M3" => DType::Fp8E4M3,
        _ => return None,
    })
}

fn dtype_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::Fp8E4M3 => 1,
        _ => 0,
    }
}

fn parse_safetensor_file(path: &Path) -> Result<Vec<SafetensorTensorInfo>> {
    let mut file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?
        .len() as usize;
    let mut header_len = [0u8; 8];
    if file.read_exact(&mut header_len).is_err() {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "safetensor file shorter than 8-byte prefix",
            },
            ctx("parse_safetensor_file"),
        ));
    }
    let header_bytes = u64::from_le_bytes(header_len) as usize;
    let payload_start = 8usize + header_bytes;
    if payload_start > file_len {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "safetensor header length exceeds file size",
            },
            ctx("parse_safetensor_file"),
        ));
    }
    let mut header_buf = vec![0u8; header_bytes];
    file.read_exact(&mut header_buf)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    let header_json = match std::str::from_utf8(&header_buf) {
        Ok(s) => s,
        Err(_) => {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor header is not utf-8",
                },
                ctx("parse_safetensor_file"),
            ))
        }
    };
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_json)
        .map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor header is not valid json",
                },
                ctx("parse_safetensor_file"),
            )
        })?;
    let mut out = Vec::new();
    for (name, meta) in header {
        if name == "__metadata__" {
            continue;
        }
        let obj = meta.as_object().ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor tensor metadata must be object",
                },
                ctx("parse_safetensor_file"),
            )
        })?;
        let dtype = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .and_then(map_dtype)
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "safetensor tensor missing or unsupported dtype",
                    },
                    ctx("parse_safetensor_file"),
                )
            })?;
        let shape = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "safetensor tensor missing shape",
                    },
                    ctx("parse_safetensor_file"),
                )
            })?
            .iter()
            .map(|x| {
                x.as_u64().map(|v| v as usize).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "safetensor tensor shape not integers",
                        },
                        ctx("parse_safetensor_file"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "safetensor tensor missing data_offsets",
                    },
                    ctx("parse_safetensor_file"),
                )
            })?;
        if offsets.len() != 2 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor tensor expects 2 data_offsets",
                },
                ctx("parse_safetensor_file"),
            ));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor offset start not integer",
                },
                ctx("parse_safetensor_file"),
            )
        })? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor offset end not integer",
                },
                ctx("parse_safetensor_file"),
            )
        })? as usize;
        if end < start {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor data_offsets are inverted",
                },
                ctx("parse_safetensor_file"),
            ));
        }
        let nbytes = end - start;
        let expected = dtype_bytes(dtype).saturating_mul(shape.iter().copied().product::<usize>());
        if expected != nbytes {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor byte length mismatch",
                },
                ctx("parse_safetensor_file"),
            ));
        }
        let offset = 8 + header_bytes + start;
        if offset.checked_add(nbytes).map_or(true, |v| v > file_len) {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "safetensor tensor offset out of file bounds",
                },
                ctx("parse_safetensor_file"),
            ));
        }
        out.push(SafetensorTensorInfo {
            name,
            dtype,
            shape,
            file: path.to_path_buf(),
            file_offset: offset,
            nbytes,
        });
    }
    Ok(out)
}

pub fn scan_safetensor_tensors(model_dir: &Path) -> Result<BTreeMap<String, SafetensorTensorInfo>> {
    let files = parse_safetensors_index(model_dir)?;
    let mut tensors = BTreeMap::new();
    for (_name, file) in files {
        let entries = parse_safetensor_file(&file)?;
        for entry in entries {
            if tensors.insert(entry.name.clone(), entry).is_some() {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "duplicate tensor name in safetensor files",
                    },
                    ctx("scan_safetensor_tensors"),
                ));
            }
        }
    }
    Ok(tensors)
}

pub fn load_safetensor_f16(model_dir: &Path, name: &str) -> Result<Vec<u8>> {
    let tensors = scan_safetensor_tensors(model_dir)?;
    let Some(info) = tensors.get(name) else {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "tensor not found",
            },
            ctx("load_safetensor_f16"),
        ));
    };
    let bytes = std::fs::read(&info.file).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: info.file.clone(),
        source,
    })?;
    let slice = &bytes[info.file_offset..info.file_offset + info.nbytes];
    Ok(match info.dtype {
        DType::F16 => slice.to_vec(),
        DType::Bf16 => bf16_to_f16_cpu(slice),
        DType::F32 | DType::Fp8E4M3 => {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unsupported dtype for metal f16 path",
                },
                ctx("load_safetensor_f16"),
            ))
        }
        _ => {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unsupported dtype for metal f16 path",
                },
                ctx("load_safetensor_f16"),
            ))
        }
    })
}

pub fn map_safetensor_to_arena(
    arena: &mut MetalBufferArena,
    model_dir: &Path,
    names: &[&str],
) -> Result<Vec<(String, MetalRegion)>> {
    let tensors = scan_safetensor_tensors(model_dir)?;
    map_safetensor_to_arena_from_tensors(arena, &tensors, names)
}

fn load_safetensor_entry_f16(entry: &SafetensorTensorInfo) -> Result<Vec<u8>> {
    let bytes = std::fs::read(&entry.file).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: entry.file.clone(),
        source,
    })?;
    let start = entry.file_offset;
    let end = entry.file_offset + entry.nbytes;
    if end > bytes.len() {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "tensor byte slice out of bounds",
            },
            ctx("load_safetensor_entry_f16"),
        ));
    }
    let slice = &bytes[start..end];
    Ok(match entry.dtype {
        DType::F16 => slice.to_vec(),
        DType::Bf16 => bf16_to_f16_cpu(slice),
        DType::F32 | DType::Fp8E4M3 => {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unsupported dtype for metal f16 path",
                },
                ctx("load_safetensor_entry_f16"),
            ))
        }
        _ => {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unsupported dtype for metal f16 path",
                },
                ctx("load_safetensor_entry_f16"),
            ))
        }
    })
}

pub fn map_safetensor_to_arena_from_tensors(
    arena: &mut MetalBufferArena,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    names: &[&str],
) -> Result<Vec<(String, MetalRegion)>> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    for &name in names {
        if !seen.insert(name.to_owned()) {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "duplicate tensor name in lookup list",
                },
                ctx("map_safetensor_to_arena"),
            ));
        }
        let entry = tensors.get(name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "tensor not found",
                },
                ctx("map_safetensor_to_arena"),
            )
        })?;
        let bytes = load_safetensor_entry_f16(entry)?;
        let region = arena.region(name, bytes.len(), 16)?;
        unsafe {
            arena.write_region(&region, &bytes)?;
        }
        out.push((name.to_string(), region));
    }
    Ok(out)
}

pub fn map_safetensor_to_model_weights(
    arena: &mut MetalBufferArena,
    model_dir: &Path,
    names: &MetalModelWeightNames,
) -> Result<MetalModelWeights> {
    let mut ordered = Vec::new();
    let mut seen = HashSet::new();
    let mut add_name =
        |name: &str, ordered: &mut Vec<String>, seen: &mut HashSet<String>| -> Result<()> {
            if !seen.insert(name.to_owned()) {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "duplicate tensor name in weight mapping manifest",
                    },
                    ctx("map_safetensor_to_model_weights"),
                ));
            }
            ordered.push(name.to_string());
            Ok(())
        };

    for layer in &names.layers {
        add_name(&layer.attn_norm, &mut ordered, &mut seen)?;
        add_name(&layer.qkv, &mut ordered, &mut seen)?;
        if let Some(name) = &layer.qkv_bias {
            add_name(name, &mut ordered, &mut seen)?;
        }
        add_name(&layer.o_proj, &mut ordered, &mut seen)?;
        add_name(&layer.mlp_norm, &mut ordered, &mut seen)?;
        add_name(&layer.gate_up, &mut ordered, &mut seen)?;
        add_name(&layer.down_proj, &mut ordered, &mut seen)?;
    }
    add_name(&names.final_norm, &mut ordered, &mut seen)?;
    add_name(&names.lm_head, &mut ordered, &mut seen)?;
    add_name(&names.rope_cos, &mut ordered, &mut seen)?;
    add_name(&names.rope_sin, &mut ordered, &mut seen)?;

    let refs = map_safetensor_to_arena(
        arena,
        model_dir,
        &ordered.iter().map(String::as_str).collect::<Vec<_>>(),
    )?;
    let mut region_by_name = BTreeMap::new();
    for (name, region) in refs {
        region_by_name.insert(name, region);
    }

    let mut take = |name: &str| -> Result<MetalRegion> {
        region_by_name.remove(name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing mapped tensor in weight map",
                },
                ctx("map_safetensor_to_model_weights"),
            )
        })
    };

    let mut layers = Vec::with_capacity(names.layers.len());
    for layer in &names.layers {
        layers.push(MetalLayerWeightRegions {
            attn_norm: take(&layer.attn_norm)?,
            qkv: take(&layer.qkv)?,
            qkv_bias: layer.qkv_bias.as_ref().map(|name| take(name)).transpose()?,
            o_proj: take(&layer.o_proj)?,
            mlp_norm: take(&layer.mlp_norm)?,
            gate_up: take(&layer.gate_up)?,
            down_proj: take(&layer.down_proj)?,
        });
    }

    let final_norm = take(&names.final_norm)?;
    let lm_head = take(&names.lm_head)?;
    let rope_cos = take(&names.rope_cos)?;
    let rope_sin = take(&names.rope_sin)?;

    Ok(MetalModelWeights {
        layers,
        final_norm,
        lm_head,
        rope_cos,
        rope_sin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::path::PathBuf;

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        p.push(format!(
            "rvllm-apple-metal-loader-{}-{}",
            std::process::id(),
            now,
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_safetensor_shard(dir: &Path, entries: &[(&str, DType, &[u8], &[usize])]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, data, shape) in entries {
            let start = payload.len();
            payload.extend_from_slice(data);
            let end = payload.len();

            let mut meta = serde_json::Map::new();
            let dt = match dtype {
                DType::F16 => "F16",
                DType::Bf16 => "BF16",
                DType::F32 => "F32",
                DType::Fp8E4M3 => "F8_E4M3",
                _ => "F16",
            };
            meta.insert("dtype".into(), serde_json::Value::String(dt.into()));
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|n| serde_json::Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number((start as u64).into()),
                    serde_json::Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), serde_json::Value::Object(meta));
        }
        let hjson = serde_json::to_string(&header).unwrap();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hjson.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hjson.as_bytes()).unwrap();
        f.write_all(&payload).unwrap();
        path
    }

    #[test]
    fn bf16_to_f16_converts_correctly() {
        let bf16_bytes = [0x80, 0x3F];
        let f16_bytes = bf16_to_f16_cpu(&bf16_bytes);
        let f16_val = half::f16::from_bits(u16::from_le_bytes([f16_bytes[0], f16_bytes[1]]));
        assert!((f16_val.to_f32() - 1.0).abs() < 0.001);
    }

    #[test]
    fn scans_and_loads_safetensor_f16_weights() {
        let dir = tempdir();
        let f16_vals = [f16::from_f32(1.0).to_bits(), f16::from_f32(-1.0).to_bits()];
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();
        let bf16_bytes = f16_to_bf16_bytes(f16::from_f32(2.0).to_bits());
        let _ = write_safetensor_shard(
            &dir,
            &[
                ("weight_f16", DType::F16, &f16_bytes, &[2]),
                ("weight_bf16", DType::Bf16, &bf16_bytes, &[2]),
            ],
        );

        let tensors = match scan_safetensor_tensors(&dir) {
            Ok(v) => v,
            Err(e) => panic!("unexpected scan error: {e}"),
        };
        assert!(tensors.contains_key("weight_f16"));
        assert!(tensors.contains_key("weight_bf16"));

        let w_f16 = match load_safetensor_f16(&dir, "weight_f16") {
            Ok(v) => v,
            Err(e) => panic!("unexpected f16 load error: {e}"),
        };
        assert_eq!(w_f16.len(), 4);
        assert_eq!(u16::from_le_bytes([w_f16[0], w_f16[1]]), f16_vals[0]);

        let w_bf16 = match load_safetensor_f16(&dir, "weight_bf16") {
            Ok(v) => v,
            Err(e) => panic!("unexpected bf16 load error: {e}"),
        };
        assert_eq!(w_bf16.len(), 4);
        assert_eq!(
            half::f16::from_bits(u16::from_le_bytes([w_bf16[0], w_bf16[1]])).to_f32(),
            2.0
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn maps_manifest_named_tensors_into_model_weights() {
        let dir = tempdir();
        let _ = write_safetensor_shard(
            &dir,
            &[
                ("layer0_attn_norm", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("layer0_qkv", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("layer0_o_proj", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("layer0_mlp_norm", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("layer0_gate_up", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("layer0_down_proj", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("final_norm", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("lm_head", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("rope_cos", DType::F16, &f16_to_vec(&[1.0]), &[1]),
                ("rope_sin", DType::F16, &f16_to_vec(&[1.0]), &[1]),
            ],
        );

        let names = MetalModelWeightNames {
            layers: vec![MetalLayerWeightNames {
                attn_norm: "layer0_attn_norm".to_owned(),
                qkv: "layer0_qkv".to_owned(),
                qkv_bias: None,
                o_proj: "layer0_o_proj".to_owned(),
                mlp_norm: "layer0_mlp_norm".to_owned(),
                gate_up: "layer0_gate_up".to_owned(),
                down_proj: "layer0_down_proj".to_owned(),
            }],
            final_norm: "final_norm".to_owned(),
            lm_head: "lm_head".to_owned(),
            rope_cos: "rope_cos".to_owned(),
            rope_sin: "rope_sin".to_owned(),
        };

        let mut context = match crate::context::MetalContext::new() {
            Ok(v) => v,
            Err(e) => panic!("unexpected metal context: {e}"),
        };
        let mut arena = match crate::arena::MetalBufferArena::new(context.device(), 1024) {
            Ok(v) => v,
            Err(e) => panic!("unexpected arena alloc: {e}"),
        };

        let model = match map_safetensor_to_model_weights(&mut arena, &dir, &names) {
            Ok(v) => v,
            Err(e) => panic!("unexpected map error: {e}"),
        };
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.layers[0].attn_norm.name, "layer0_attn_norm");
        assert_eq!(model.final_norm.name, "final_norm");
    }

    fn f16_to_vec(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
        }
        out
    }

    fn f16_to_bf16_bytes(raw_f16: u16) -> Vec<u8> {
        let f = f16::from_bits(raw_f16).to_f32();
        (f.to_bits() >> 16).to_le_bytes().to_vec()
    }
}
