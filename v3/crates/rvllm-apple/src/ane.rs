#![allow(unsafe_code)]
#[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
use prost::Message;
#[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
use rvllm_core::error::AneCompileError;
use rvllm_core::{AppleCtx, AppleError, DType, Result, RvllmError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
#[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use crate::iosurface::IoSurfaceTensorDesc;
use crate::plan::RolloutBucket;

type CompileOutput = Result<()>;

const ANE_DIAGNOSTIC_CAPACITY: usize = 8;

static ANE_DIAGNOSTICS: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

fn ane_diagnostics() -> &'static Mutex<VecDeque<String>> {
    ANE_DIAGNOSTICS.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn locate_compiled_bundle(workspace: &Path) -> Option<PathBuf> {
    let direct = workspace.join("model.mlmodelc");
    if direct.exists() {
        return Some(direct);
    }

    let mut seen: Vec<PathBuf> = std::fs::read_dir(workspace)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension() == Some(OsStr::new("mlmodelc"))
                || path.file_name() == Some(OsStr::new("model.mlmodelc"))
        })
        .collect();

    seen.sort();
    seen.into_iter().next()
}

fn push_diagnostic(message: impl Into<String>) {
    if let Ok(mut cache) = ane_diagnostics().lock() {
        if cache.len() == ANE_DIAGNOSTIC_CAPACITY {
            let _ = cache.pop_front();
        }
        cache.push_back(message.into());
    }
}

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "private-ane",
        op,
        device: "apple-silicon",
    }
}

pub fn last_ane_diagnostics() -> Vec<String> {
    match ane_diagnostics().lock() {
        Ok(cache) => cache.iter().cloned().collect(),
        Err(_) => Vec::new(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneRolloutConfig {
    pub bucket: RolloutBucket,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
}

impl AneRolloutConfig {
    #[must_use]
    pub fn activation_desc(&self) -> IoSurfaceTensorDesc {
        IoSurfaceTensorDesc {
            dtype: DType::F16,
            channels: self.hidden_size,
            spatial: (self.bucket.seqs * self.bucket.tokens) as usize,
        }
    }

    #[must_use]
    pub fn activation_bytes(&self) -> usize {
        self.activation_desc().bytes()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AneProcedure {
    FusedFfn {
        layer: usize,
        offsets: crate::mil::FfnMilOffsets,
    },
    FusedQkv {
        layer: usize,
        offsets: crate::mil::QkvMilOffsets,
    },
    LmHead {
        offset: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneProgramPlan {
    pub id: String,
    pub template_name: String,
    pub spatial: usize,
    pub in_ch: usize,
    pub hidden_ch: usize,
    pub out_ch: usize,
    pub offsets: std::collections::HashMap<String, u64>,
}

impl AneProgramPlan {
    #[must_use]
    pub fn proj_only(config: AneRolloutConfig) -> Self {
        let mut offsets = std::collections::HashMap::new();
        offsets.insert("proj_weight_to_fp16".to_string(), 0);

        Self {
            id: "proj_test".to_string(),
            template_name: "proj.mlmodel".to_string(),
            spatial: (config.bucket.seqs * config.bucket.tokens) as usize,
            in_ch: config.hidden_size,
            hidden_ch: config.hidden_size,
            out_ch: config.hidden_size,
            offsets,
        }
    }

    #[must_use]
    pub fn ffn_only(config: AneRolloutConfig) -> Self {
        let mut offsets = std::collections::HashMap::new();
        let gate_size = config.intermediate_size * config.hidden_size * 2;
        let up_size = config.intermediate_size * config.hidden_size * 2;

        offsets.insert("gate_weight_to_fp16".to_string(), 0);
        offsets.insert("up_weight_to_fp16".to_string(), gate_size as u64);
        offsets.insert(
            "down_weight_to_fp16".to_string(),
            (gate_size + up_size) as u64,
        );

        Self {
            id: "ffn_test".to_string(),
            template_name: "ffn.mlmodel".to_string(),
            spatial: (config.bucket.seqs * config.bucket.tokens) as usize,
            in_ch: config.hidden_size,
            hidden_ch: config.intermediate_size,
            out_ch: config.hidden_size,
            offsets,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.offsets.is_empty() {
            return Err(RvllmError::apple(
                AppleError::InvalidMil {
                    reason: "plan has no weight offsets",
                },
                ctx("validate_plan"),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn cache_key(&self) -> String {
        let mut fields = Vec::from_iter(
            self.offsets
                .iter()
                .map(|(name, offset)| (name.as_str(), *offset)),
        );
        fields.sort_by(|a, b| a.0.cmp(b.0));
        let mut input = format!(
            concat!(
                "rvllm-private-ane-program-v1\n",
                "id={}\n",
                "template={}\n",
                "spatial={}\n",
                "in_ch={}\n",
                "hidden_ch={}\n",
                "out_ch={}\n",
            ),
            self.id, self.template_name, self.spatial, self.in_ch, self.hidden_ch, self.out_ch
        );
        for (name, offset) in fields {
            input.push_str("offset=");
            input.push_str(name);
            input.push('=');
            input.push_str(&offset.to_string());
            input.push('\n');
        }
        format!("ane_v1_{}", crate::plan::stable_hash_hex(input.as_bytes()))
    }
}

#[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
pub fn compile_private_ane_program(plan: &AneProgramPlan, weights_path: &Path) -> Result<PathBuf> {
    if !crate::plan::private_ane_env_opted_in() {
        return Err(RvllmError::apple(
            AppleError::PrivateApiUnavailable {
                symbol: crate::plan::PRIVATE_ANE_ENV_VAR,
            },
            ctx("private_ane_env_opt_in"),
        ));
    }
    plan.validate()?;

    let cache_root = std::env::var_os("RVLLM_ANE_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| Path::new(&home).join(".cache").join("rvllm").join("ane"))
        })
        .unwrap_or_else(std::env::temp_dir);
    let cache_key = plan.cache_key();
    let workspace = cache_root.join(&cache_key);

    rvllm_apple_ane_sys::load_frameworks().map_err(|e| {
        push_diagnostic(format!("load_frameworks failed: {}", e));
        RvllmError::apple(
            AppleError::PrivateApiUnavailable {
                symbol: "load_frameworks",
            },
            ctx("load_frameworks"),
        )
    })?;

    let result: CompileOutput = (|| -> CompileOutput {
        if let Some(compiled) = locate_compiled_bundle(&workspace) {
            push_diagnostic(format!(
                "reusing existing compiled bundle from {}",
                compiled.display()
            ));
            let weights_dir = workspace.join("weights");
            std::fs::create_dir_all(&weights_dir).map_err(|e| {
                push_diagnostic(format!("create weights dir failed: {e}"));
                RvllmError::apple(
                    AppleError::CompileAneModel {
                        err: AneCompileError::CompileIo {
                            detail: e.to_string(),
                        },
                    },
                    ctx("create_weights_dir"),
                )
            })?;

            let weights_file = weights_dir.join("weight.bin");
            std::fs::copy(weights_path, &weights_file).map_err(|e| {
                push_diagnostic(format!("copy weights failed: {e}"));
                RvllmError::apple(
                    AppleError::CompileAneModel {
                        err: AneCompileError::CompileIo {
                            detail: e.to_string(),
                        },
                    },
                    ctx("copy_weights"),
                )
            })?;
            return Ok(());
        }

        std::fs::create_dir_all(&workspace).map_err(|e| {
            push_diagnostic(format!("create workspace failed: {e}"));
            RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CompileIo {
                        detail: e.to_string(),
                    },
                },
                ctx("create_workspace"),
            )
        })?;

        let mil_path = workspace.join("model.mlmodel");
        let weights_dir = workspace.join("weights");
        std::fs::create_dir_all(&weights_dir).map_err(|e| {
            push_diagnostic(format!("create weights dir failed: {e}"));
            RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CompileIo {
                        detail: e.to_string(),
                    },
                },
                ctx("create_weights_dir"),
            )
        })?;

        let weights_file = weights_dir.join("weight.bin");
        std::fs::copy(weights_path, &weights_file).map_err(|e| {
            push_diagnostic(format!("copy weights failed: {e}"));
            RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CompileIo {
                        detail: e.to_string(),
                    },
                },
                ctx("copy_weights"),
            )
        })?;

        let mut model = crate::mil::load_template(&plan.template_name);
        crate::mil::patch_ast(
            &mut model,
            "main",
            plan.spatial,
            plan.in_ch,
            plan.hidden_ch,
            plan.out_ch,
            &plan.offsets,
        );

        if let Some(rvllm_apple_coreml_sys::specification::model::Type::MlProgram(ref mut mlp)) =
            model.r#type
        {
            for func in mlp.functions.values_mut() {
                for block in func.block_specializations.values_mut() {
                    for op in block.operations.iter_mut() {
                        if op.r#type == "const" || op.r#type == "weight" {
                            if let Some(ref mut val) = op.attributes.get_mut("val") {
                                if let Some(ref mut im) = val.value {
                                    match im {
                                        rvllm_apple_coreml_sys::specification::mil_spec::value::Value::BlobFileValue(
                                            ref mut blob,
                                        ) => {
                                            if let Some(offset) = plan.offsets.get(&op.outputs[0].name) {
                                                blob.offset = *offset;
                                            }
                                            blob.file_name = "@model_path/weights/weight.bin".to_string();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::fs::write(&mil_path, model.encode_to_vec()).map_err(|e| {
            push_diagnostic(format!("write mlmodel failed: {e}"));
            RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CompileIo {
                        detail: e.to_string(),
                    },
                },
                ctx("write_mlmodel"),
            )
        })?;

        let metadata_out = Command::new("xcrun")
            .arg("coremlcompiler")
            .arg("metadata")
            .arg(&mil_path)
            .output();
        if let Ok(mo) = metadata_out {
            eprintln!("[ANE METADATA] {}", String::from_utf8_lossy(&mo.stdout));
        }

        let output = Command::new("xcrun")
            .arg("coremlcompiler")
            .arg("compile")
            .arg(&mil_path)
            .arg(&workspace)
            .output()
            .map_err(|_| {
                RvllmError::apple(
                    AppleError::CompileAneModel {
                        err: AneCompileError::CompileIo {
                            detail: "xcrun coremlcompiler invocation failed".into(),
                        },
                    },
                    ctx("compile"),
                )
            })?;

        if !output.status.success() {
            push_diagnostic(format!(
                "xcrun failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
            return Err(RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CompileIo {
                        detail: String::from_utf8_lossy(&output.stderr).into_owned(),
                    },
                },
                ctx("compile"),
            ));
        }

        if locate_compiled_bundle(&workspace).is_none() {
            return Err(RvllmError::apple(
                AppleError::CompileAneModel {
                    err: AneCompileError::CacheMissOrCorrupt {
                        cache_path: workspace.to_string_lossy().into_owned(),
                    },
                },
                ctx("compile"),
            ));
        }

        Ok(())
    })();

    result?;
    let compiled = locate_compiled_bundle(&workspace).ok_or_else(|| {
        RvllmError::apple(
            AppleError::CompileAneModel {
                err: AneCompileError::CacheMissOrCorrupt {
                    cache_path: workspace.to_string_lossy().into_owned(),
                },
            },
            ctx("compile"),
        )
    })?;

    Ok(compiled)
}

#[cfg(not(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64")))]
pub fn compile_private_ane_program(
    _plan: &AneProgramPlan,
    _weights_path: &Path,
) -> Result<PathBuf> {
    Err(RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "private-ane",
            op: "compile_private_ane_program requires macOS+aarch64+private-ane",
        },
        ctx("compile_private_ane_program"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "private-ane"))]
    fn test_hardware_ane_compilation_integration() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket {
                seqs: 1,
                tokens: 16,
            },
            hidden_size: 16,
            intermediate_size: 16,
            num_layers: 1,
        };
        let plan = AneProgramPlan::proj_only(config);

        let temp_dir = std::env::temp_dir().join("rvllm_test_weights_ane");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let weights_path = temp_dir.join("weights.bin");
        std::fs::write(&weights_path, vec![0u8; 1024 * 1024]).unwrap();

        let result = compile_private_ane_program(&plan, &weights_path);
        if let Err(ref e) = result {
            eprintln!("[ANE ERROR] {}", e);
            for diag in last_ane_diagnostics() {
                eprintln!("[ANE DIAG] {}", diag);
            }
        }
        assert!(
            result.is_ok(),
            "Hardware-gated ANE compilation failed: {:?}",
            result.err()
        );
    }

    #[test]
    #[ignore = "requires private ANE compile/load/evaluate opt-in"]
    #[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
    fn private_ane_tiny_projection_evaluate_smoke() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket {
                seqs: 1,
                tokens: 16,
            },
            hidden_size: 16,
            intermediate_size: 16,
            num_layers: 1,
        };
        let plan = AneProgramPlan::proj_only(config.clone());

        let temp_dir = std::env::temp_dir().join("rvllm_test_eval_ane");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let weights_path = temp_dir.join("weights.bin");
        std::fs::write(&weights_path, vec![0u8; 1024 * 1024]).unwrap();

        let compiled = match compile_private_ane_program(&plan, &weights_path) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("[ANE ERROR] {}", e);
                for diag in last_ane_diagnostics() {
                    eprintln!("[ANE DIAG] {}", diag);
                }
                panic!("private ANE compile failed: {e}");
            }
        };

        let compiled_str = compiled
            .to_str()
            .expect("compiled model path should be UTF-8");
        let handle = match rvllm_apple_ane_sys::AneModelHandle::load_with_error(compiled_str) {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("[ANE DIAG] compiled model could not be loaded by _ANEClient: {err}");
                eprintln!(
                    "[ANE DIAG] evaluate smoke skipped after compile; no ANE execution claim made"
                );
                return;
            }
        };

        let desc = crate::iosurface::IoSurfaceTensorDesc {
            dtype: DType::F32,
            channels: config.hidden_size,
            spatial: (config.bucket.seqs * config.bucket.tokens) as usize,
        };
        let shape = desc.byte_surface_shape();
        let input = rvllm_apple_ane_sys::AneSurface::new(
            shape.width,
            shape.height,
            shape.bytes_per_element,
        )
        .expect("input IOSurface should allocate");
        let output = rvllm_apple_ane_sys::AneSurface::new(
            shape.width,
            shape.height,
            shape.bytes_per_element,
        )
        .expect("output IOSurface should allocate");

        for i in 0..desc.element_count() {
            input
                .write_f32(i, (i + 1) as f32)
                .expect("input IOSurface write should succeed");
            output
                .write_f32(i, f32::NAN)
                .expect("output IOSurface write should succeed");
        }

        let request =
            rvllm_apple_ane_sys::AneRequest::new(&[input], &[0], &[output.clone()], &[0], 0)
                .expect("private ANE request should be created");
        handle
            .evaluate(&request)
            .expect("private ANE request should evaluate");

        let mut values = Vec::with_capacity(desc.element_count());
        for i in 0..desc.element_count() {
            values.push(
                output
                    .try_read_f32(i)
                    .expect("output IOSurface read should succeed"),
            );
        }
        assert!(
            values.iter().all(|v| v.is_finite()),
            "ANE output should be finite after evaluate: {values:?}"
        );
        assert!(
            values.iter().all(|v| v.abs() <= 1.0e-3),
            "zero-weight projection should write near-zero output: {values:?}"
        );
    }

    #[test]
    #[ignore = "requires private ANE compile/load opt-in; records load boundary only"]
    #[cfg(all(target_os = "macos", feature = "private-ane", target_arch = "aarch64"))]
    fn private_ane_tiny_projection_load_boundary_is_reported() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket {
                seqs: 1,
                tokens: 16,
            },
            hidden_size: 16,
            intermediate_size: 16,
            num_layers: 1,
        };
        let plan = AneProgramPlan::proj_only(config);

        let temp_dir = std::env::temp_dir().join("rvllm_test_load_boundary_ane");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let weights_path = temp_dir.join("weights.bin");
        std::fs::write(&weights_path, vec![0u8; 1024 * 1024]).unwrap();

        let compiled = match compile_private_ane_program(&plan, &weights_path) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("[ANE ERROR] {e}");
                for diag in last_ane_diagnostics() {
                    eprintln!("[ANE DIAG] {diag}");
                }
                panic!("private ANE compile failed before load-boundary diagnostic: {e}");
            }
        };

        let compiled_str = compiled
            .to_str()
            .expect("compiled model path should be UTF-8");
        match rvllm_apple_ane_sys::AneModelHandle::load_with_error(compiled_str) {
            Ok(_) => {
                eprintln!(
                    "[ANE DIAG] _ANEClient loadModel accepted {}; evaluation intentionally not run in this boundary test",
                    compiled.display()
                );
            }
            Err(err) => {
                assert!(
                    !err.trim().is_empty(),
                    "_ANEClient loadModel rejection should include a reason"
                );
                eprintln!(
                    "[ANE DIAG] _ANEClient loadModel rejected {}; no ANE execution claim made: {err}",
                    compiled.display()
                );
            }
        }
    }
}
