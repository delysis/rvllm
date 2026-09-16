//! Shipping-safe, deterministic public Core ML projection export gate.
//!
//! This module reads one dense `[out_features, in_features]` tensor directly
//! from a Hugging Face safetensors checkpoint and emits a Core ML neural-network
//! protobuf containing a single public 1x1 convolution. F16 payload bytes are
//! copied bit-for-bit and F32 values are preserved exactly. No platform runtime
//! or private Apple framework is linked by this module.

use half::{bf16, f16};
use prost::Message;
use rvllm_apple_coreml_runtime::{
    compile_model, CoreMlComputeUnits, CoreMlExecutionReport, CoreMlF32Input, CoreMlF32Output,
    PublicCoreMlModel,
};
use rvllm_apple_coreml_sys::specification::{
    array_feature_type, convolution_layer_params, feature_type, gelu_layer_params, model,
    neural_network_layer, ArrayFeatureType, ConvolutionLayerParams, FeatureDescription,
    FeatureType, GeluLayerParams, Model, ModelDescription, MultiplyBroadcastableLayerParams,
    NeuralNetwork, NeuralNetworkLayer, NeuralNetworkMultiArrayShapeMapping, ValidPadding,
    WeightParams,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

const EXPORTER_DOMAIN: &[u8] = b"rvllm.public-coreml.projection-export.v1\0";
const EXPORTER_FINGERPRINT_DOMAIN: &[u8] = b"rvllm.public-coreml.projection-exporter.v1";
const GATED_FFN_EXPORTER_DOMAIN: &[u8] = b"rvllm.public-coreml.gated-ffn-export.v1\0";
const GATED_FFN_EXPORTER_FINGERPRINT_DOMAIN: &[u8] = b"rvllm.public-coreml.gated-ffn-exporter.v1";
const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;

/// Checkpoint dtype accepted by the exact projection gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreMlProjectionWeightDtype {
    F16,
    Bf16,
    F32,
}

impl CoreMlProjectionWeightDtype {
    fn safetensors_name(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// A request to export one checkpoint projection as a standalone Core ML model.
#[derive(Clone, Debug)]
pub struct CoreMlProjectionExportRequest<'a> {
    pub model_dir: &'a Path,
    pub tensor_name: &'a str,
    pub out_features: usize,
    pub in_features: usize,
    /// Number of static token positions exposed as the convolution width.
    pub spatial: usize,
}

/// Deterministic CPU known-answer material bundled with an export.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlProjectionKnownAnswer {
    pub input: Vec<f32>,
    pub expected_output: Vec<f32>,
    pub input_sha256: String,
    pub expected_output_sha256: String,
}

/// A complete first-gate artifact. `model_bytes` is a public `.mlmodel`
/// protobuf and can be handed to a platform-owned public Core ML compiler.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlProjectionArtifact {
    pub schema_version: u32,
    pub tensor_name: String,
    pub source_dtype: CoreMlProjectionWeightDtype,
    pub out_features: usize,
    pub in_features: usize,
    pub spatial: usize,
    pub source_tensor_sha256: String,
    pub exporter_fingerprint_sha256: String,
    pub model_sha256: String,
    pub artifact_identity_sha256: String,
    pub model_bytes: Vec<u8>,
    pub known_answer: CoreMlProjectionKnownAnswer,
}

impl CoreMlProjectionArtifact {
    pub const SCHEMA_V1: u32 = 1;

    /// Decode and validate the emitted public model, then run its projection on
    /// the CPU and compare it with the export-time known answer.
    pub fn validate_cpu(&self) -> Result<CoreMlProjectionValidation, CoreMlProjectionError> {
        if self.schema_version != Self::SCHEMA_V1 {
            return Err(CoreMlProjectionError::Artifact(format!(
                "unsupported projection artifact schema {}",
                self.schema_version
            )));
        }
        let model_digest = sha256_hex(&self.model_bytes);
        if model_digest != self.model_sha256 {
            return Err(CoreMlProjectionError::Artifact(
                "Core ML model digest does not match the artifact manifest".to_string(),
            ));
        }
        if self.exporter_fingerprint_sha256 != exporter_fingerprint() {
            return Err(CoreMlProjectionError::Artifact(
                "projection exporter fingerprint is not recognized".to_string(),
            ));
        }
        let expected_identity = artifact_identity(
            &self.tensor_name,
            self.source_dtype,
            self.out_features,
            self.in_features,
            self.spatial,
            &self.source_tensor_sha256,
            &self.model_sha256,
            &self.known_answer.input_sha256,
            &self.known_answer.expected_output_sha256,
        );
        if expected_identity != self.artifact_identity_sha256 {
            return Err(CoreMlProjectionError::Artifact(
                "projection artifact identity does not match its contents".to_string(),
            ));
        }
        if sha256_f32(&self.known_answer.input) != self.known_answer.input_sha256
            || sha256_f32(&self.known_answer.expected_output)
                != self.known_answer.expected_output_sha256
        {
            return Err(CoreMlProjectionError::Artifact(
                "known-answer digest does not match its payload".to_string(),
            ));
        }

        let model = Model::decode(self.model_bytes.as_slice()).map_err(|error| {
            CoreMlProjectionError::Artifact(format!("invalid Core ML protobuf: {error}"))
        })?;
        validate_model_contract(&model, self.out_features, self.in_features, self.spatial)?;
        let (model_dtype, raw_weights) = extract_model_weights(&model)?;
        let expected_model_dtype = match self.source_dtype {
            CoreMlProjectionWeightDtype::F16 => CoreMlProjectionWeightDtype::F16,
            CoreMlProjectionWeightDtype::Bf16 | CoreMlProjectionWeightDtype::F32 => {
                CoreMlProjectionWeightDtype::F32
            }
        };
        if model_dtype != expected_model_dtype {
            return Err(CoreMlProjectionError::Artifact(
                "Core ML model weight encoding does not match the exact-export contract"
                    .to_string(),
            ));
        }
        let weights = decode_weights(model_dtype, &raw_weights)?;
        let reconstructed_source = encode_source_weights(self.source_dtype, &weights);
        if sha256_hex(&reconstructed_source) != self.source_tensor_sha256 {
            return Err(CoreMlProjectionError::Artifact(
                "Core ML model weights do not match the checkpoint tensor identity".to_string(),
            ));
        }
        let actual = cpu_projection(
            &weights,
            self.out_features,
            self.in_features,
            self.spatial,
            &self.known_answer.input,
        )?;
        if actual.len() != self.known_answer.expected_output.len() {
            return Err(CoreMlProjectionError::Artifact(
                "known-answer output length does not match the model output".to_string(),
            ));
        }
        let max_abs_error = actual
            .iter()
            .zip(&self.known_answer.expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        if max_abs_error != 0.0 {
            return Err(CoreMlProjectionError::KnownAnswer { max_abs_error });
        }
        Ok(CoreMlProjectionValidation {
            actual_output_sha256: sha256_f32(&actual),
            max_abs_error,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlProjectionValidation {
    pub actual_output_sha256: String,
    pub max_abs_error: f32,
}

/// Result of executing the exact projection artifact through public Core ML.
///
/// `execution_report` records only the requested compute units. Device
/// placement remains unverified until separate compute-plan and Instruments
/// evidence is attached.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlProjectionRuntimeValidation {
    pub compiled_model_path: PathBuf,
    pub actual_output_sha256: String,
    pub max_abs_error: f32,
    pub execution_report: CoreMlExecutionReport,
}

/// Exact checkpoint tensors and static shape for one gated feed-forward block.
///
/// The exported graph is deliberately limited to batch one and a static token
/// width. It is a public Core ML parity gate, not a production decoder route.
#[derive(Clone, Debug)]
pub struct CoreMlGatedFfnExportRequest<'a> {
    pub model_dir: &'a Path,
    pub gate_tensor_name: &'a str,
    pub up_tensor_name: &'a str,
    pub down_tensor_name: &'a str,
    pub hidden_features: usize,
    pub intermediate_features: usize,
    pub spatial: usize,
}

/// Source identity for one checkpoint tensor embedded in a gated FFN artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlGatedFfnTensorIdentity {
    pub tensor_name: String,
    pub source_dtype: CoreMlProjectionWeightDtype,
    pub source_tensor_sha256: String,
}

/// Exact five-layer public Core ML gated-FFN validation artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlGatedFfnArtifact {
    pub schema_version: u32,
    pub gate: CoreMlGatedFfnTensorIdentity,
    pub up: CoreMlGatedFfnTensorIdentity,
    pub down: CoreMlGatedFfnTensorIdentity,
    pub hidden_features: usize,
    pub intermediate_features: usize,
    pub spatial: usize,
    pub exporter_fingerprint_sha256: String,
    pub model_sha256: String,
    pub artifact_identity_sha256: String,
    pub model_bytes: Vec<u8>,
    pub known_answer: CoreMlProjectionKnownAnswer,
}

impl CoreMlGatedFfnArtifact {
    pub const SCHEMA_V1: u32 = 1;

    /// Validate graph topology, exact source weights, identity, and the
    /// deterministic CPU known answer without invoking an Apple runtime.
    pub fn validate_cpu(&self) -> Result<CoreMlProjectionValidation, CoreMlProjectionError> {
        if self.schema_version != Self::SCHEMA_V1 {
            return Err(CoreMlProjectionError::Artifact(format!(
                "unsupported gated FFN artifact schema {}",
                self.schema_version
            )));
        }
        if sha256_hex(&self.model_bytes) != self.model_sha256 {
            return Err(CoreMlProjectionError::Artifact(
                "gated FFN Core ML model digest does not match the artifact manifest".to_string(),
            ));
        }
        if self.exporter_fingerprint_sha256 != gated_ffn_exporter_fingerprint() {
            return Err(CoreMlProjectionError::Artifact(
                "gated FFN exporter fingerprint is not recognized".to_string(),
            ));
        }
        let expected_identity = gated_ffn_artifact_identity(
            &self.gate,
            &self.up,
            &self.down,
            self.hidden_features,
            self.intermediate_features,
            self.spatial,
            &self.model_sha256,
            &self.known_answer.input_sha256,
            &self.known_answer.expected_output_sha256,
        );
        if expected_identity != self.artifact_identity_sha256 {
            return Err(CoreMlProjectionError::Artifact(
                "gated FFN artifact identity does not match its contents".to_string(),
            ));
        }
        if sha256_f32(&self.known_answer.input) != self.known_answer.input_sha256
            || sha256_f32(&self.known_answer.expected_output)
                != self.known_answer.expected_output_sha256
        {
            return Err(CoreMlProjectionError::Artifact(
                "gated FFN known-answer digest does not match its payload".to_string(),
            ));
        }

        let model = Model::decode(self.model_bytes.as_slice()).map_err(|error| {
            CoreMlProjectionError::Artifact(format!("invalid gated FFN Core ML protobuf: {error}"))
        })?;
        validate_gated_ffn_model_contract(
            &model,
            self.hidden_features,
            self.intermediate_features,
            self.spatial,
        )?;
        let gate_weights = validated_gated_ffn_weights(
            &model,
            0,
            &self.gate,
            self.intermediate_features,
            self.hidden_features,
        )?;
        let up_weights = validated_gated_ffn_weights(
            &model,
            1,
            &self.up,
            self.intermediate_features,
            self.hidden_features,
        )?;
        let down_weights = validated_gated_ffn_weights(
            &model,
            4,
            &self.down,
            self.hidden_features,
            self.intermediate_features,
        )?;
        let actual = cpu_gated_ffn(
            &gate_weights,
            &up_weights,
            &down_weights,
            self.hidden_features,
            self.intermediate_features,
            self.spatial,
            &self.known_answer.input,
        )?;
        if actual.len() != self.known_answer.expected_output.len() {
            return Err(CoreMlProjectionError::Artifact(
                "gated FFN known-answer output length does not match the model output".to_string(),
            ));
        }
        let max_abs_error = actual
            .iter()
            .zip(&self.known_answer.expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        if max_abs_error != 0.0 {
            return Err(CoreMlProjectionError::KnownAnswer { max_abs_error });
        }
        Ok(CoreMlProjectionValidation {
            actual_output_sha256: sha256_f32(&actual),
            max_abs_error,
        })
    }
}

/// Public Core ML execution result for the gated-FFN parity gate.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreMlGatedFfnRuntimeValidation {
    pub compiled_model_path: PathBuf,
    pub actual_output_sha256: String,
    pub max_abs_error: f32,
    pub execution_report: CoreMlExecutionReport,
}

#[derive(Debug)]
pub enum CoreMlProjectionError {
    InvalidRequest(&'static str),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Index(String),
    MissingTensor(String),
    UnsupportedDtype {
        tensor: String,
        dtype: String,
    },
    Shape {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    CorruptTensor(String),
    ZeroWeights(String),
    Artifact(String),
    KnownAnswer {
        max_abs_error: f32,
    },
    Runtime(String),
}

impl fmt::Display for CoreMlProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => {
                write!(formatter, "invalid projection request: {message}")
            }
            Self::Io { path, source } => {
                write!(formatter, "failed to read {}: {source}", path.display())
            }
            Self::Index(message) => write!(formatter, "invalid safetensors index: {message}"),
            Self::MissingTensor(tensor) => {
                write!(formatter, "checkpoint tensor {tensor:?} was not found")
            }
            Self::UnsupportedDtype { tensor, dtype } => write!(
                formatter,
                "checkpoint tensor {tensor:?} has unsupported exact-export dtype {dtype}"
            ),
            Self::Shape {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "checkpoint tensor {tensor:?} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::CorruptTensor(message) => {
                write!(formatter, "corrupt safetensors payload: {message}")
            }
            Self::ZeroWeights(tensor) => write!(
                formatter,
                "checkpoint tensor {tensor:?} contains only zero weights; refusing scaffold export"
            ),
            Self::Artifact(message) => {
                write!(formatter, "invalid Core ML projection artifact: {message}")
            }
            Self::KnownAnswer { max_abs_error } => write!(
                formatter,
                "CPU projection known-answer mismatch (max absolute error {max_abs_error})"
            ),
            Self::Runtime(message) => {
                write!(formatter, "public Core ML projection failed: {message}")
            }
        }
    }
}

impl std::error::Error for CoreMlProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Export one exact dense checkpoint projection into a public Core ML model.
pub fn export_checkpoint_projection(
    request: CoreMlProjectionExportRequest<'_>,
) -> Result<CoreMlProjectionArtifact, CoreMlProjectionError> {
    validate_request(&request)?;
    let source = load_safetensors_tensor(request.model_dir, request.tensor_name)?;
    let expected_shape = vec![request.out_features, request.in_features];
    if source.shape != expected_shape {
        return Err(CoreMlProjectionError::Shape {
            tensor: request.tensor_name.to_string(),
            expected: expected_shape,
            actual: source.shape,
        });
    }
    let values = decode_weights(source.dtype, &source.bytes)?;
    if values.iter().all(|value| *value == 0.0) {
        return Err(CoreMlProjectionError::ZeroWeights(
            request.tensor_name.to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "tensor {:?} contains non-finite weights",
            request.tensor_name
        )));
    }

    let model = build_projection_model(
        source.dtype,
        &source.bytes,
        request.out_features,
        request.in_features,
        request.spatial,
    )?;
    let model_bytes = model.encode_to_vec();
    let source_tensor_sha256 = sha256_hex(&source.bytes);
    let model_sha256 = sha256_hex(&model_bytes);
    let input = deterministic_input(request.in_features, request.spatial);
    let expected_output = cpu_projection(
        &values,
        request.out_features,
        request.in_features,
        request.spatial,
        &input,
    )?;
    let input_sha256 = sha256_f32(&input);
    let expected_output_sha256 = sha256_f32(&expected_output);
    let artifact_identity_sha256 = artifact_identity(
        request.tensor_name,
        source.dtype,
        request.out_features,
        request.in_features,
        request.spatial,
        &source_tensor_sha256,
        &model_sha256,
        &input_sha256,
        &expected_output_sha256,
    );
    let artifact = CoreMlProjectionArtifact {
        schema_version: CoreMlProjectionArtifact::SCHEMA_V1,
        tensor_name: request.tensor_name.to_string(),
        source_dtype: source.dtype,
        out_features: request.out_features,
        in_features: request.in_features,
        spatial: request.spatial,
        source_tensor_sha256,
        exporter_fingerprint_sha256: exporter_fingerprint(),
        model_sha256,
        artifact_identity_sha256,
        model_bytes,
        known_answer: CoreMlProjectionKnownAnswer {
            input,
            expected_output,
            input_sha256,
            expected_output_sha256,
        },
    };
    artifact.validate_cpu()?;
    Ok(artifact)
}

/// Compile, load, and execute an exported projection using public Core ML.
///
/// The source model is materialized under `workspace`; Core ML owns the
/// returned compiled bundle. This is the nonzero projection promotion gate,
/// not a decoder backend and not evidence that the Neural Engine executed.
pub fn validate_public_coreml_projection(
    artifact: &CoreMlProjectionArtifact,
    workspace: &Path,
    requested_compute_units: CoreMlComputeUnits,
    absolute_tolerance: f32,
) -> Result<CoreMlProjectionRuntimeValidation, CoreMlProjectionError> {
    if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
        return Err(CoreMlProjectionError::InvalidRequest(
            "public Core ML absolute tolerance must be finite and non-negative",
        ));
    }
    artifact.validate_cpu()?;
    fs::create_dir_all(workspace).map_err(|source| CoreMlProjectionError::Io {
        path: workspace.to_path_buf(),
        source,
    })?;
    let source_model_path = workspace.join(format!(
        "projection-{}.mlmodel",
        artifact.artifact_identity_sha256
    ));
    fs::write(&source_model_path, &artifact.model_bytes).map_err(|source| {
        CoreMlProjectionError::Io {
            path: source_model_path.clone(),
            source,
        }
    })?;
    let compiled_model_path = compile_model(&source_model_path)
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    let model = PublicCoreMlModel::load(&compiled_model_path, requested_compute_units)
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    let actual = model
        .predict_f32(
            CoreMlF32Input {
                name: "x",
                shape: &[artifact.in_features as i64, 1, artifact.spatial as i64],
                values: &artifact.known_answer.input,
            },
            CoreMlF32Output {
                name: "y",
                shape: &[artifact.out_features as i64, 1, artifact.spatial as i64],
            },
        )
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    if actual.len() != artifact.known_answer.expected_output.len() {
        return Err(CoreMlProjectionError::Runtime(format!(
            "output length mismatch: expected {}, got {}",
            artifact.known_answer.expected_output.len(),
            actual.len()
        )));
    }
    let max_abs_error = actual
        .iter()
        .zip(&artifact.known_answer.expected_output)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if !max_abs_error.is_finite() || max_abs_error > absolute_tolerance {
        return Err(CoreMlProjectionError::KnownAnswer { max_abs_error });
    }
    Ok(CoreMlProjectionRuntimeValidation {
        compiled_model_path,
        actual_output_sha256: sha256_f32(&actual),
        max_abs_error,
        execution_report: model.execution_report(),
    })
}

/// Export an exact checkpoint gated FFN as a standalone public Core ML model.
///
/// The graph is fixed to `gate(x) -> GELU_TANH`, `up(x)`, elementwise
/// multiplication, and `down(...)`. No routing or execution claim is implied.
pub fn export_checkpoint_gated_ffn(
    request: CoreMlGatedFfnExportRequest<'_>,
) -> Result<CoreMlGatedFfnArtifact, CoreMlProjectionError> {
    validate_gated_ffn_request(&request)?;
    let gate_source = load_safetensors_tensor(request.model_dir, request.gate_tensor_name)?;
    let up_source = load_safetensors_tensor(request.model_dir, request.up_tensor_name)?;
    let down_source = load_safetensors_tensor(request.model_dir, request.down_tensor_name)?;
    validate_gated_ffn_source(
        request.gate_tensor_name,
        &gate_source,
        &[request.intermediate_features, request.hidden_features],
    )?;
    validate_gated_ffn_source(
        request.up_tensor_name,
        &up_source,
        &[request.intermediate_features, request.hidden_features],
    )?;
    validate_gated_ffn_source(
        request.down_tensor_name,
        &down_source,
        &[request.hidden_features, request.intermediate_features],
    )?;

    let gate_weights = decode_weights(gate_source.dtype, &gate_source.bytes)?;
    let up_weights = decode_weights(up_source.dtype, &up_source.bytes)?;
    let down_weights = decode_weights(down_source.dtype, &down_source.bytes)?;
    let model = build_gated_ffn_model(
        &gate_source,
        &up_source,
        &down_source,
        request.hidden_features,
        request.intermediate_features,
        request.spatial,
    )?;
    let model_bytes = model.encode_to_vec();
    let gate = gated_ffn_tensor_identity(request.gate_tensor_name, &gate_source);
    let up = gated_ffn_tensor_identity(request.up_tensor_name, &up_source);
    let down = gated_ffn_tensor_identity(request.down_tensor_name, &down_source);
    let input = deterministic_input(request.hidden_features, request.spatial);
    let expected_output = cpu_gated_ffn(
        &gate_weights,
        &up_weights,
        &down_weights,
        request.hidden_features,
        request.intermediate_features,
        request.spatial,
        &input,
    )?;
    let model_sha256 = sha256_hex(&model_bytes);
    let input_sha256 = sha256_f32(&input);
    let expected_output_sha256 = sha256_f32(&expected_output);
    let artifact_identity_sha256 = gated_ffn_artifact_identity(
        &gate,
        &up,
        &down,
        request.hidden_features,
        request.intermediate_features,
        request.spatial,
        &model_sha256,
        &input_sha256,
        &expected_output_sha256,
    );
    let artifact = CoreMlGatedFfnArtifact {
        schema_version: CoreMlGatedFfnArtifact::SCHEMA_V1,
        gate,
        up,
        down,
        hidden_features: request.hidden_features,
        intermediate_features: request.intermediate_features,
        spatial: request.spatial,
        exporter_fingerprint_sha256: gated_ffn_exporter_fingerprint(),
        model_sha256,
        artifact_identity_sha256,
        model_bytes,
        known_answer: CoreMlProjectionKnownAnswer {
            input,
            expected_output,
            input_sha256,
            expected_output_sha256,
        },
    };
    artifact.validate_cpu()?;
    Ok(artifact)
}

/// Compile and execute the exact gated-FFN artifact through public Core ML.
///
/// This remains a parity gate. The report records requested compute units and
/// intentionally never treats that request as evidence of ANE execution.
pub fn validate_public_coreml_gated_ffn(
    artifact: &CoreMlGatedFfnArtifact,
    workspace: &Path,
    requested_compute_units: CoreMlComputeUnits,
    absolute_tolerance: f32,
) -> Result<CoreMlGatedFfnRuntimeValidation, CoreMlProjectionError> {
    if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
        return Err(CoreMlProjectionError::InvalidRequest(
            "public Core ML gated FFN absolute tolerance must be finite and non-negative",
        ));
    }
    artifact.validate_cpu()?;
    fs::create_dir_all(workspace).map_err(|source| CoreMlProjectionError::Io {
        path: workspace.to_path_buf(),
        source,
    })?;
    let source_model_path = workspace.join(format!(
        "gated-ffn-{}.mlmodel",
        artifact.artifact_identity_sha256
    ));
    fs::write(&source_model_path, &artifact.model_bytes).map_err(|source| {
        CoreMlProjectionError::Io {
            path: source_model_path.clone(),
            source,
        }
    })?;
    let compiled_model_path = compile_model(&source_model_path)
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    let model = PublicCoreMlModel::load(&compiled_model_path, requested_compute_units)
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    let actual = model
        .predict_f32(
            CoreMlF32Input {
                name: "x",
                shape: &[
                    1,
                    artifact.hidden_features as i64,
                    1,
                    artifact.spatial as i64,
                ],
                values: &artifact.known_answer.input,
            },
            CoreMlF32Output {
                name: "y",
                shape: &[
                    1,
                    artifact.hidden_features as i64,
                    1,
                    artifact.spatial as i64,
                ],
            },
        )
        .map_err(|error| CoreMlProjectionError::Runtime(error.to_string()))?;
    if actual.len() != artifact.known_answer.expected_output.len() {
        return Err(CoreMlProjectionError::Runtime(format!(
            "gated FFN output length mismatch: expected {}, got {}",
            artifact.known_answer.expected_output.len(),
            actual.len()
        )));
    }
    let max_abs_error = actual
        .iter()
        .zip(&artifact.known_answer.expected_output)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if !max_abs_error.is_finite() || max_abs_error > absolute_tolerance {
        return Err(CoreMlProjectionError::KnownAnswer { max_abs_error });
    }
    Ok(CoreMlGatedFfnRuntimeValidation {
        compiled_model_path,
        actual_output_sha256: sha256_f32(&actual),
        max_abs_error,
        execution_report: model.execution_report(),
    })
}

fn validate_request(
    request: &CoreMlProjectionExportRequest<'_>,
) -> Result<(), CoreMlProjectionError> {
    if request.tensor_name.trim().is_empty() {
        return Err(CoreMlProjectionError::InvalidRequest(
            "tensor name is empty",
        ));
    }
    if request.in_features == 0 || request.out_features == 0 || request.spatial == 0 {
        return Err(CoreMlProjectionError::InvalidRequest(
            "in_features, out_features, and spatial must be nonzero",
        ));
    }
    for value in [request.in_features, request.out_features, request.spatial] {
        if i64::try_from(value).is_err() || u64::try_from(value).is_err() {
            return Err(CoreMlProjectionError::InvalidRequest(
                "projection dimensions exceed the Core ML protobuf ABI",
            ));
        }
    }
    request
        .out_features
        .checked_mul(request.in_features)
        .and_then(|value| value.checked_mul(request.spatial))
        .ok_or(CoreMlProjectionError::InvalidRequest(
            "projection dimensions overflow host address space",
        ))?;
    Ok(())
}

fn validate_gated_ffn_request(
    request: &CoreMlGatedFfnExportRequest<'_>,
) -> Result<(), CoreMlProjectionError> {
    let names = [
        request.gate_tensor_name,
        request.up_tensor_name,
        request.down_tensor_name,
    ];
    if names.iter().any(|name| name.trim().is_empty()) {
        return Err(CoreMlProjectionError::InvalidRequest(
            "gated FFN tensor names must be nonempty",
        ));
    }
    if names[0] == names[1] || names[0] == names[2] || names[1] == names[2] {
        return Err(CoreMlProjectionError::InvalidRequest(
            "gated FFN tensor names must be distinct",
        ));
    }
    if request.hidden_features == 0 || request.intermediate_features == 0 || request.spatial == 0 {
        return Err(CoreMlProjectionError::InvalidRequest(
            "hidden_features, intermediate_features, and spatial must be nonzero",
        ));
    }
    for value in [
        request.hidden_features,
        request.intermediate_features,
        request.spatial,
    ] {
        if i64::try_from(value).is_err() || u64::try_from(value).is_err() {
            return Err(CoreMlProjectionError::InvalidRequest(
                "gated FFN dimensions exceed the Core ML protobuf ABI",
            ));
        }
    }
    request
        .hidden_features
        .checked_mul(request.intermediate_features)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| {
            request
                .hidden_features
                .checked_mul(request.spatial)
                .and_then(|activation| value.checked_add(activation))
        })
        .ok_or(CoreMlProjectionError::InvalidRequest(
            "gated FFN dimensions overflow host address space",
        ))?;
    Ok(())
}

fn validate_gated_ffn_source(
    tensor_name: &str,
    source: &LoadedTensor,
    expected_shape: &[usize],
) -> Result<(), CoreMlProjectionError> {
    if source.shape != expected_shape {
        return Err(CoreMlProjectionError::Shape {
            tensor: tensor_name.to_string(),
            expected: expected_shape.to_vec(),
            actual: source.shape.clone(),
        });
    }
    let values = decode_weights(source.dtype, &source.bytes)?;
    if values.iter().all(|value| *value == 0.0) {
        return Err(CoreMlProjectionError::ZeroWeights(tensor_name.to_string()));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "tensor {tensor_name:?} contains non-finite weights"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct LoadedTensor {
    dtype: CoreMlProjectionWeightDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct SafetensorsTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn load_safetensors_tensor(
    model_dir: &Path,
    tensor_name: &str,
) -> Result<LoadedTensor, CoreMlProjectionError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let shard_path = if index_path.is_file() {
        let bytes = read(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_slice(&bytes)
            .map_err(|error| CoreMlProjectionError::Index(error.to_string()))?;
        let relative = index
            .weight_map
            .get(tensor_name)
            .ok_or_else(|| CoreMlProjectionError::MissingTensor(tensor_name.to_string()))?;
        validate_relative_shard_path(relative)?;
        model_dir.join(relative)
    } else {
        model_dir.join("model.safetensors")
    };
    if !shard_path.is_file() {
        return Err(CoreMlProjectionError::Io {
            path: shard_path,
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "shard is not a file"),
        });
    }
    let mut file = File::open(&shard_path).map_err(|source| CoreMlProjectionError::Io {
        path: shard_path.clone(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| CoreMlProjectionError::Io {
            path: shard_path.clone(),
            source,
        })?
        .len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|source| CoreMlProjectionError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header_len_u64 = u64::from_le_bytes(prefix);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        CoreMlProjectionError::CorruptTensor("header length exceeds host address space".to_string())
    })?;
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "header length {header_len} exceeds the {} byte limit",
            MAX_SAFETENSORS_HEADER_BYTES
        )));
    }
    let payload_start = 8_usize.checked_add(header_len).ok_or_else(|| {
        CoreMlProjectionError::CorruptTensor("header offset overflow".to_string())
    })?;
    if payload_start as u64 > file_len {
        return Err(CoreMlProjectionError::CorruptTensor(
            "header extends past the end of the shard".to_string(),
        ));
    }
    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|source| CoreMlProjectionError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|error| {
            CoreMlProjectionError::CorruptTensor(format!("invalid header JSON: {error}"))
        })?;
    let raw_meta = header
        .get(tensor_name)
        .ok_or_else(|| CoreMlProjectionError::MissingTensor(tensor_name.to_string()))?;
    let meta: SafetensorsTensorMeta =
        serde_json::from_value(raw_meta.clone()).map_err(|error| {
            CoreMlProjectionError::CorruptTensor(format!(
                "invalid metadata for tensor {tensor_name:?}: {error}"
            ))
        })?;
    let dtype = match meta.dtype.as_str() {
        "F16" => CoreMlProjectionWeightDtype::F16,
        "BF16" => CoreMlProjectionWeightDtype::Bf16,
        "F32" => CoreMlProjectionWeightDtype::F32,
        other => {
            return Err(CoreMlProjectionError::UnsupportedDtype {
                tensor: tensor_name.to_string(),
                dtype: other.to_string(),
            });
        }
    };
    let elements = meta.shape.iter().try_fold(1_usize, |total, value| {
        total.checked_mul(*value).ok_or_else(|| {
            CoreMlProjectionError::CorruptTensor(format!(
                "tensor {tensor_name:?} shape element count overflow"
            ))
        })
    })?;
    let expected_bytes = elements.checked_mul(dtype.bytes()).ok_or_else(|| {
        CoreMlProjectionError::CorruptTensor(format!("tensor {tensor_name:?} byte count overflow"))
    })?;
    let start = usize::try_from(meta.data_offsets[0]).map_err(|_| {
        CoreMlProjectionError::CorruptTensor(
            "tensor start offset exceeds host address space".to_string(),
        )
    })?;
    let end = usize::try_from(meta.data_offsets[1]).map_err(|_| {
        CoreMlProjectionError::CorruptTensor(
            "tensor end offset exceeds host address space".to_string(),
        )
    })?;
    if start > end || end.checked_sub(start) != Some(expected_bytes) {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "tensor {tensor_name:?} data offsets do not match dtype and shape"
        )));
    }
    let absolute_start = payload_start.checked_add(start).ok_or_else(|| {
        CoreMlProjectionError::CorruptTensor("tensor start offset overflow".to_string())
    })?;
    let absolute_end = payload_start.checked_add(end).ok_or_else(|| {
        CoreMlProjectionError::CorruptTensor("tensor end offset overflow".to_string())
    })?;
    if absolute_end as u64 > file_len {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "tensor {tensor_name:?} extends past the end of the shard"
        )));
    }
    file.seek(SeekFrom::Start(absolute_start as u64))
        .map_err(|source| CoreMlProjectionError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let mut tensor_bytes = vec![0_u8; expected_bytes];
    file.read_exact(&mut tensor_bytes)
        .map_err(|source| CoreMlProjectionError::Io {
            path: shard_path.clone(),
            source,
        })?;
    Ok(LoadedTensor {
        dtype,
        shape: meta.shape,
        bytes: tensor_bytes,
    })
}

fn validate_relative_shard_path(path: &str) -> Result<(), CoreMlProjectionError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CoreMlProjectionError::Index(
            "weight_map contains an unsafe shard path".to_string(),
        ));
    }
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>, CoreMlProjectionError> {
    fs::read(path).map_err(|source| CoreMlProjectionError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn build_projection_model(
    dtype: CoreMlProjectionWeightDtype,
    raw_weights: &[u8],
    out_features: usize,
    in_features: usize,
    spatial: usize,
) -> Result<Model, CoreMlProjectionError> {
    let input = feature("x", vec![in_features as i64, 1, spatial as i64]);
    let output = feature("y", vec![out_features as i64, 1, spatial as i64]);
    let weights = match dtype {
        CoreMlProjectionWeightDtype::F16 => WeightParams {
            float16_value: raw_weights.to_vec(),
            ..WeightParams::default()
        },
        CoreMlProjectionWeightDtype::Bf16 | CoreMlProjectionWeightDtype::F32 => {
            let values = decode_weights(dtype, raw_weights)?;
            WeightParams {
                float_value: values,
                ..WeightParams::default()
            }
        }
    };
    let convolution = ConvolutionLayerParams {
        output_channels: out_features as u64,
        kernel_channels: in_features as u64,
        n_groups: 1,
        kernel_size: vec![1, 1],
        stride: vec![1, 1],
        dilation_factor: vec![1, 1],
        is_deconvolution: false,
        has_bias: false,
        weights: Some(weights),
        convolution_padding_type: Some(convolution_layer_params::ConvolutionPaddingType::Valid(
            ValidPadding::default(),
        )),
        ..ConvolutionLayerParams::default()
    };
    let layer = NeuralNetworkLayer {
        name: "checkpoint_projection".to_string(),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        layer: Some(neural_network_layer::Layer::Convolution(convolution)),
        ..NeuralNetworkLayer::default()
    };
    let network = NeuralNetwork {
        layers: vec![layer],
        ..NeuralNetwork::default()
    };
    let description = ModelDescription {
        input: vec![input],
        output: vec![output],
        ..ModelDescription::default()
    };
    Ok(Model {
        specification_version: 4,
        description: Some(description),
        r#type: Some(model::Type::NeuralNetwork(network)),
        ..Model::default()
    })
}

fn build_gated_ffn_model(
    gate: &LoadedTensor,
    up: &LoadedTensor,
    down: &LoadedTensor,
    hidden_features: usize,
    intermediate_features: usize,
    spatial: usize,
) -> Result<Model, CoreMlProjectionError> {
    let gate_projection = dense_convolution_layer(
        "gated_ffn_gate_projection",
        "x",
        "gate",
        gate,
        intermediate_features,
        hidden_features,
    )?;
    let up_projection = dense_convolution_layer(
        "gated_ffn_up_projection",
        "x",
        "up",
        up,
        intermediate_features,
        hidden_features,
    )?;
    let gelu = NeuralNetworkLayer {
        name: "gated_ffn_gelu_tanh".to_string(),
        input: vec!["gate".to_string()],
        output: vec!["activated_gate".to_string()],
        layer: Some(neural_network_layer::Layer::Gelu(GeluLayerParams {
            mode: gelu_layer_params::GeluMode::TanhApproximation as i32,
        })),
        ..NeuralNetworkLayer::default()
    };
    let multiply = NeuralNetworkLayer {
        name: "gated_ffn_multiply".to_string(),
        input: vec!["activated_gate".to_string(), "up".to_string()],
        output: vec!["activated".to_string()],
        layer: Some(neural_network_layer::Layer::MultiplyBroadcastable(
            MultiplyBroadcastableLayerParams::default(),
        )),
        ..NeuralNetworkLayer::default()
    };
    let down_projection = dense_convolution_layer(
        "gated_ffn_down_projection",
        "activated",
        "y",
        down,
        hidden_features,
        intermediate_features,
    )?;
    let network = NeuralNetwork {
        layers: vec![
            gate_projection,
            up_projection,
            gelu,
            multiply,
            down_projection,
        ],
        array_input_shape_mapping: NeuralNetworkMultiArrayShapeMapping::ExactArrayMapping as i32,
        ..NeuralNetwork::default()
    };
    let description = ModelDescription {
        input: vec![feature(
            "x",
            vec![1, hidden_features as i64, 1, spatial as i64],
        )],
        output: vec![feature(
            "y",
            vec![1, hidden_features as i64, 1, spatial as i64],
        )],
        ..ModelDescription::default()
    };
    Ok(Model {
        specification_version: 4,
        description: Some(description),
        r#type: Some(model::Type::NeuralNetwork(network)),
        ..Model::default()
    })
}

fn dense_convolution_layer(
    layer_name: &str,
    input: &str,
    output: &str,
    source: &LoadedTensor,
    out_features: usize,
    in_features: usize,
) -> Result<NeuralNetworkLayer, CoreMlProjectionError> {
    let weights = match source.dtype {
        CoreMlProjectionWeightDtype::F16 => WeightParams {
            float16_value: source.bytes.clone(),
            ..WeightParams::default()
        },
        CoreMlProjectionWeightDtype::Bf16 | CoreMlProjectionWeightDtype::F32 => WeightParams {
            float_value: decode_weights(source.dtype, &source.bytes)?,
            ..WeightParams::default()
        },
    };
    Ok(NeuralNetworkLayer {
        name: layer_name.to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        layer: Some(neural_network_layer::Layer::Convolution(
            ConvolutionLayerParams {
                output_channels: out_features as u64,
                kernel_channels: in_features as u64,
                n_groups: 1,
                kernel_size: vec![1, 1],
                stride: vec![1, 1],
                dilation_factor: vec![1, 1],
                is_deconvolution: false,
                has_bias: false,
                weights: Some(weights),
                convolution_padding_type: Some(
                    convolution_layer_params::ConvolutionPaddingType::Valid(ValidPadding::default()),
                ),
                ..ConvolutionLayerParams::default()
            },
        )),
        ..NeuralNetworkLayer::default()
    })
}

fn feature(name: &str, shape: Vec<i64>) -> FeatureDescription {
    FeatureDescription {
        name: name.to_string(),
        r#type: Some(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape,
                data_type: array_feature_type::ArrayDataType::Float32 as i32,
                ..ArrayFeatureType::default()
            })),
            ..FeatureType::default()
        }),
        ..FeatureDescription::default()
    }
}

fn validate_model_contract(
    model: &Model,
    out_features: usize,
    in_features: usize,
    spatial: usize,
) -> Result<(), CoreMlProjectionError> {
    if model.specification_version != 4 {
        return Err(CoreMlProjectionError::Artifact(
            "projection model must use Core ML specification version 4".to_string(),
        ));
    }
    let description = model.description.as_ref().ok_or_else(|| {
        CoreMlProjectionError::Artifact("projection model has no description".to_string())
    })?;
    validate_feature(
        description.input.as_slice(),
        "x",
        &[in_features as i64, 1, spatial as i64],
    )?;
    validate_feature(
        description.output.as_slice(),
        "y",
        &[out_features as i64, 1, spatial as i64],
    )?;
    let Some(model::Type::NeuralNetwork(network)) = model.r#type.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "projection model is not a public neural network".to_string(),
        ));
    };
    if network.layers.len() != 1 {
        return Err(CoreMlProjectionError::Artifact(
            "projection model must contain exactly one layer".to_string(),
        ));
    }
    let layer = &network.layers[0];
    if layer.name != "checkpoint_projection"
        || layer.input != ["x"]
        || layer.output != ["y"]
        || layer.is_updatable
    {
        return Err(CoreMlProjectionError::Artifact(
            "projection layer interface does not match the exporter contract".to_string(),
        ));
    }
    let Some(neural_network_layer::Layer::Convolution(convolution)) = layer.layer.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "projection layer is not a convolution".to_string(),
        ));
    };
    if convolution.output_channels != out_features as u64
        || convolution.kernel_channels != in_features as u64
        || convolution.n_groups != 1
        || convolution.kernel_size != [1, 1]
        || convolution.stride != [1, 1]
        || convolution.is_deconvolution
        || convolution.has_bias
        || convolution.bias.is_some()
        || !matches!(
            convolution.convolution_padding_type,
            Some(convolution_layer_params::ConvolutionPaddingType::Valid(_))
        )
    {
        return Err(CoreMlProjectionError::Artifact(
            "projection convolution parameters do not match the exact dense contract".to_string(),
        ));
    }
    Ok(())
}

fn validate_gated_ffn_model_contract(
    model: &Model,
    hidden_features: usize,
    intermediate_features: usize,
    spatial: usize,
) -> Result<(), CoreMlProjectionError> {
    if model.specification_version != 4 || model.is_updatable {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN model must use non-updatable Core ML specification version 4".to_string(),
        ));
    }
    let description = model.description.as_ref().ok_or_else(|| {
        CoreMlProjectionError::Artifact("gated FFN model has no description".to_string())
    })?;
    if !description.functions.is_empty()
        || !description.default_function_name.is_empty()
        || !description.state.is_empty()
        || !description.training_input.is_empty()
        || !description.predicted_feature_name.is_empty()
        || !description.predicted_probabilities_name.is_empty()
    {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN model description contains unsupported auxiliary interfaces".to_string(),
        ));
    }
    validate_feature(
        description.input.as_slice(),
        "x",
        &[1, hidden_features as i64, 1, spatial as i64],
    )?;
    validate_feature(
        description.output.as_slice(),
        "y",
        &[1, hidden_features as i64, 1, spatial as i64],
    )?;
    let Some(model::Type::NeuralNetwork(network)) = model.r#type.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN model is not a public neural network".to_string(),
        ));
    };
    if network.layers.len() != 5
        || !network.preprocessing.is_empty()
        || network.update_params.is_some()
        || network.array_input_shape_mapping
            != NeuralNetworkMultiArrayShapeMapping::ExactArrayMapping as i32
    {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN model must use exact array mapping, exactly five layers, and no update/preprocessing state".to_string(),
        ));
    }
    validate_dense_convolution_layer(
        &network.layers[0],
        "gated_ffn_gate_projection",
        "x",
        "gate",
        intermediate_features,
        hidden_features,
    )?;
    validate_dense_convolution_layer(
        &network.layers[1],
        "gated_ffn_up_projection",
        "x",
        "up",
        intermediate_features,
        hidden_features,
    )?;
    let gelu = &network.layers[2];
    validate_layer_interface(gelu, "gated_ffn_gelu_tanh", &["gate"], &["activated_gate"])?;
    let Some(neural_network_layer::Layer::Gelu(parameters)) = gelu.layer.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN activation must be GELU".to_string(),
        ));
    };
    if parameters.mode != gelu_layer_params::GeluMode::TanhApproximation as i32 {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN GELU must use the tanh approximation".to_string(),
        ));
    }
    let multiply = &network.layers[3];
    validate_layer_interface(
        multiply,
        "gated_ffn_multiply",
        &["activated_gate", "up"],
        &["activated"],
    )?;
    if !matches!(
        multiply.layer.as_ref(),
        Some(neural_network_layer::Layer::MultiplyBroadcastable(_))
    ) {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN gate/up combination must use multiply-broadcastable".to_string(),
        ));
    }
    validate_dense_convolution_layer(
        &network.layers[4],
        "gated_ffn_down_projection",
        "activated",
        "y",
        hidden_features,
        intermediate_features,
    )
}

fn validate_layer_interface(
    layer: &NeuralNetworkLayer,
    expected_name: &str,
    expected_inputs: &[&str],
    expected_outputs: &[&str],
) -> Result<(), CoreMlProjectionError> {
    if layer.name != expected_name
        || layer
            .input
            .iter()
            .map(String::as_str)
            .ne(expected_inputs.iter().copied())
        || layer
            .output
            .iter()
            .map(String::as_str)
            .ne(expected_outputs.iter().copied())
        || layer.is_updatable
        || !layer.input_tensor.is_empty()
        || !layer.output_tensor.is_empty()
    {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} interface does not match the exporter contract"
        )));
    }
    Ok(())
}

fn validate_dense_convolution_layer(
    layer: &NeuralNetworkLayer,
    expected_name: &str,
    expected_input: &str,
    expected_output: &str,
    out_features: usize,
    in_features: usize,
) -> Result<(), CoreMlProjectionError> {
    validate_layer_interface(layer, expected_name, &[expected_input], &[expected_output])?;
    let Some(neural_network_layer::Layer::Convolution(convolution)) = layer.layer.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} is not a convolution"
        )));
    };
    let Some(convolution_layer_params::ConvolutionPaddingType::Valid(valid_padding)) =
        convolution.convolution_padding_type.as_ref()
    else {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} must use valid padding"
        )));
    };
    let valid_padding_is_zero = valid_padding
        .padding_amounts
        .as_ref()
        .map_or(true, |amounts| {
            amounts.border_amounts.len() == 2
                && amounts
                    .border_amounts
                    .iter()
                    .all(|edges| edges.start_edge_size == 0 && edges.end_edge_size == 0)
        });
    if convolution.output_channels != out_features as u64
        || convolution.kernel_channels != in_features as u64
        || convolution.n_groups != 1
        || convolution.kernel_size != [1, 1]
        || convolution.stride != [1, 1]
        || convolution.dilation_factor != [1, 1]
        || convolution.is_deconvolution
        || convolution.has_bias
        || convolution.bias.is_some()
        || !convolution.output_shape.is_empty()
        || !valid_padding_is_zero
    {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} parameters do not match the exact dense contract"
        )));
    }
    let weights = convolution.weights.as_ref().ok_or_else(|| {
        CoreMlProjectionError::Artifact(format!("gated FFN layer {expected_name:?} has no weights"))
    })?;
    if weights.quantization.is_some() || weights.is_updatable {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} weights must be immutable and unquantized"
        )));
    }
    let (dtype, raw_weights) = extract_convolution_weights(convolution)?;
    let expected_elements =
        out_features
            .checked_mul(in_features)
            .ok_or(CoreMlProjectionError::InvalidRequest(
                "gated FFN weight dimensions overflow",
            ))?;
    let actual_elements = raw_weights.len() / dtype.bytes();
    if raw_weights.len() % dtype.bytes() != 0 || actual_elements != expected_elements {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN layer {expected_name:?} weight length does not match its dimensions"
        )));
    }
    Ok(())
}

fn validate_feature(
    features: &[FeatureDescription],
    name: &str,
    expected_shape: &[i64],
) -> Result<(), CoreMlProjectionError> {
    if features.len() != 1 || features[0].name != name {
        return Err(CoreMlProjectionError::Artifact(format!(
            "projection model must expose exactly one {name:?} feature"
        )));
    }
    let feature_type = features[0]
        .r#type
        .as_ref()
        .ok_or_else(|| CoreMlProjectionError::Artifact(format!("feature {name:?} has no type")))?;
    let Some(feature_type::Type::MultiArrayType(array)) = feature_type.r#type.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(format!(
            "feature {name:?} is not a multi-array"
        )));
    };
    if feature_type.is_optional
        || array.shape != expected_shape
        || array.data_type != array_feature_type::ArrayDataType::Float32 as i32
        || array.shape_flexibility.is_some()
        || array.default_optional_value.is_some()
    {
        return Err(CoreMlProjectionError::Artifact(format!(
            "feature {name:?} optionality, shape, flexibility, default, or dtype does not match the projection contract"
        )));
    }
    Ok(())
}

fn extract_model_weights(
    model: &Model,
) -> Result<(CoreMlProjectionWeightDtype, Vec<u8>), CoreMlProjectionError> {
    let Some(model::Type::NeuralNetwork(network)) = model.r#type.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "projection model is not a neural network".to_string(),
        ));
    };
    let layer = network.layers.first().ok_or_else(|| {
        CoreMlProjectionError::Artifact("projection model has no layer".to_string())
    })?;
    let Some(neural_network_layer::Layer::Convolution(convolution)) = layer.layer.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "projection layer is not a convolution".to_string(),
        ));
    };
    extract_convolution_weights(convolution)
}

fn extract_convolution_weights(
    convolution: &ConvolutionLayerParams,
) -> Result<(CoreMlProjectionWeightDtype, Vec<u8>), CoreMlProjectionError> {
    let weights = convolution.weights.as_ref().ok_or_else(|| {
        CoreMlProjectionError::Artifact("projection convolution has no weights".to_string())
    })?;
    let has_f16 = !weights.float16_value.is_empty();
    let has_f32 = !weights.float_value.is_empty();
    if has_f16 == has_f32
        || !weights.raw_value.is_empty()
        || !weights.int8_raw_value.is_empty()
        || weights.quantization.is_some()
        || weights.is_updatable
    {
        return Err(CoreMlProjectionError::Artifact(
            "projection weights must use exactly one immutable, unquantized F16 or F32 field"
                .to_string(),
        ));
    }
    if has_f16 {
        Ok((
            CoreMlProjectionWeightDtype::F16,
            weights.float16_value.clone(),
        ))
    } else {
        let mut bytes = Vec::with_capacity(weights.float_value.len() * 4);
        for value in &weights.float_value {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok((CoreMlProjectionWeightDtype::F32, bytes))
    }
}

fn validated_gated_ffn_weights(
    model: &Model,
    layer_index: usize,
    source: &CoreMlGatedFfnTensorIdentity,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, CoreMlProjectionError> {
    let Some(model::Type::NeuralNetwork(network)) = model.r#type.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN model is not a neural network".to_string(),
        ));
    };
    let layer = network.layers.get(layer_index).ok_or_else(|| {
        CoreMlProjectionError::Artifact("gated FFN model is missing a projection layer".to_string())
    })?;
    let Some(neural_network_layer::Layer::Convolution(convolution)) = layer.layer.as_ref() else {
        return Err(CoreMlProjectionError::Artifact(
            "gated FFN projection layer is not a convolution".to_string(),
        ));
    };
    let (model_dtype, raw_weights) = extract_convolution_weights(convolution)?;
    let expected_model_dtype = match source.source_dtype {
        CoreMlProjectionWeightDtype::F16 => CoreMlProjectionWeightDtype::F16,
        CoreMlProjectionWeightDtype::Bf16 | CoreMlProjectionWeightDtype::F32 => {
            CoreMlProjectionWeightDtype::F32
        }
    };
    if model_dtype != expected_model_dtype {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN tensor {:?} model encoding does not match its exact-export dtype",
            source.tensor_name
        )));
    }
    let weights = decode_weights(model_dtype, &raw_weights)?;
    let expected_elements =
        out_features
            .checked_mul(in_features)
            .ok_or(CoreMlProjectionError::InvalidRequest(
                "gated FFN weight dimensions overflow",
            ))?;
    if weights.len() != expected_elements {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN tensor {:?} element count does not match its dimensions",
            source.tensor_name
        )));
    }
    let reconstructed_source = encode_source_weights(source.source_dtype, &weights);
    if sha256_hex(&reconstructed_source) != source.source_tensor_sha256 {
        return Err(CoreMlProjectionError::Artifact(format!(
            "gated FFN model weights do not match checkpoint tensor {:?}",
            source.tensor_name
        )));
    }
    Ok(weights)
}

fn decode_weights(
    dtype: CoreMlProjectionWeightDtype,
    bytes: &[u8],
) -> Result<Vec<f32>, CoreMlProjectionError> {
    if bytes.len() % dtype.bytes() != 0 {
        return Err(CoreMlProjectionError::CorruptTensor(format!(
            "{} payload length {} is not element-aligned",
            dtype.safetensors_name(),
            bytes.len()
        )));
    }
    match dtype {
        CoreMlProjectionWeightDtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        CoreMlProjectionWeightDtype::Bf16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        CoreMlProjectionWeightDtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
    }
}

fn encode_source_weights(dtype: CoreMlProjectionWeightDtype, values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.bytes());
    match dtype {
        CoreMlProjectionWeightDtype::F16 => {
            for value in values {
                bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
            }
        }
        CoreMlProjectionWeightDtype::Bf16 => {
            for value in values {
                bytes.extend_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
            }
        }
        CoreMlProjectionWeightDtype::F32 => {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    bytes
}

fn deterministic_input(in_features: usize, spatial: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(in_features * spatial);
    // Both the projection's `[C,H,W]` feature and the gated FFN's exact
    // `[B,C,H,W]` feature use channel-major contiguous values here.
    for channel in 0..in_features {
        for position in 0..spatial {
            let lane = ((position * 17 + channel * 13 + 5) % 29) as i32 - 14;
            input.push((lane as f32) / 16.0);
        }
    }
    input
}

fn cpu_projection(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    spatial: usize,
    input: &[f32],
) -> Result<Vec<f32>, CoreMlProjectionError> {
    let expected_weights =
        out_features
            .checked_mul(in_features)
            .ok_or(CoreMlProjectionError::InvalidRequest(
                "projection weight dimensions overflow",
            ))?;
    let expected_input =
        in_features
            .checked_mul(spatial)
            .ok_or(CoreMlProjectionError::InvalidRequest(
                "projection input dimensions overflow",
            ))?;
    if weights.len() != expected_weights || input.len() != expected_input {
        return Err(CoreMlProjectionError::Artifact(
            "CPU projection payload dimensions do not match the artifact".to_string(),
        ));
    }
    let output_len =
        out_features
            .checked_mul(spatial)
            .ok_or(CoreMlProjectionError::InvalidRequest(
                "projection output dimensions overflow",
            ))?;
    let mut output = vec![0.0_f32; output_len];
    for position in 0..spatial {
        for output_channel in 0..out_features {
            let weight_row =
                &weights[output_channel * in_features..(output_channel + 1) * in_features];
            let mut sum = 0.0_f32;
            for (input_channel, weight) in weight_row.iter().enumerate() {
                sum += weight * input[input_channel * spatial + position];
            }
            output[output_channel * spatial + position] = sum;
        }
    }
    Ok(output)
}

fn cpu_gated_ffn(
    gate_weights: &[f32],
    up_weights: &[f32],
    down_weights: &[f32],
    hidden_features: usize,
    intermediate_features: usize,
    spatial: usize,
    input: &[f32],
) -> Result<Vec<f32>, CoreMlProjectionError> {
    let gate = cpu_projection(
        gate_weights,
        intermediate_features,
        hidden_features,
        spatial,
        input,
    )?;
    let up = cpu_projection(
        up_weights,
        intermediate_features,
        hidden_features,
        spatial,
        input,
    )?;
    let mut activated = Vec::with_capacity(gate.len());
    for (gate, up) in gate.into_iter().zip(up) {
        let gelu = 0.5_f32
            * gate
            * (1.0_f32
                + (std::f32::consts::FRAC_2_SQRT_PI
                    * std::f32::consts::FRAC_1_SQRT_2
                    * (gate + 0.044_715_f32 * gate * gate * gate))
                    .tanh());
        activated.push(gelu * up);
    }
    cpu_projection(
        down_weights,
        hidden_features,
        intermediate_features,
        spatial,
        &activated,
    )
}

fn exporter_fingerprint() -> String {
    sha256_hex(EXPORTER_FINGERPRINT_DOMAIN)
}

#[allow(clippy::too_many_arguments)]
fn artifact_identity(
    tensor_name: &str,
    dtype: CoreMlProjectionWeightDtype,
    out_features: usize,
    in_features: usize,
    spatial: usize,
    source_tensor_sha256: &str,
    model_sha256: &str,
    input_sha256: &str,
    expected_output_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(EXPORTER_DOMAIN);
    hash_field(&mut hasher, tensor_name.as_bytes());
    hash_field(&mut hasher, dtype.safetensors_name().as_bytes());
    hash_field(&mut hasher, &(out_features as u64).to_le_bytes());
    hash_field(&mut hasher, &(in_features as u64).to_le_bytes());
    hash_field(&mut hasher, &(spatial as u64).to_le_bytes());
    hash_field(&mut hasher, source_tensor_sha256.as_bytes());
    hash_field(&mut hasher, model_sha256.as_bytes());
    hash_field(&mut hasher, input_sha256.as_bytes());
    hash_field(&mut hasher, expected_output_sha256.as_bytes());
    hex_lower(&hasher.finalize())
}

fn gated_ffn_tensor_identity(
    tensor_name: &str,
    source: &LoadedTensor,
) -> CoreMlGatedFfnTensorIdentity {
    CoreMlGatedFfnTensorIdentity {
        tensor_name: tensor_name.to_string(),
        source_dtype: source.dtype,
        source_tensor_sha256: sha256_hex(&source.bytes),
    }
}

fn gated_ffn_exporter_fingerprint() -> String {
    sha256_hex(GATED_FFN_EXPORTER_FINGERPRINT_DOMAIN)
}

#[allow(clippy::too_many_arguments)]
fn gated_ffn_artifact_identity(
    gate: &CoreMlGatedFfnTensorIdentity,
    up: &CoreMlGatedFfnTensorIdentity,
    down: &CoreMlGatedFfnTensorIdentity,
    hidden_features: usize,
    intermediate_features: usize,
    spatial: usize,
    model_sha256: &str,
    input_sha256: &str,
    expected_output_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(GATED_FFN_EXPORTER_DOMAIN);
    for source in [gate, up, down] {
        hash_field(&mut hasher, source.tensor_name.as_bytes());
        hash_field(
            &mut hasher,
            source.source_dtype.safetensors_name().as_bytes(),
        );
        hash_field(&mut hasher, source.source_tensor_sha256.as_bytes());
    }
    hash_field(&mut hasher, &(hidden_features as u64).to_le_bytes());
    hash_field(&mut hasher, &(intermediate_features as u64).to_le_bytes());
    hash_field(&mut hasher, &(spatial as u64).to_le_bytes());
    hash_field(&mut hasher, model_sha256.as_bytes());
    hash_field(&mut hasher, input_sha256.as_bytes());
    hash_field(&mut hasher, expected_output_sha256.as_bytes());
    hex_lower(&hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex_lower(&hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(1);

    fn test_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rvllm-coreml-projection-{}-{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap_or_else(|error| panic!("create test dir: {error}"));
        path
    }

    fn write_tensor(dir: &Path, name: &str, dtype: &str, shape: &[usize], payload: &[u8]) {
        let header = serde_json::json!({
            name: {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, payload.len()]
            }
        });
        let header_bytes = serde_json::to_vec(&header)
            .unwrap_or_else(|error| panic!("serialize test header: {error}"));
        let mut shard = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header_bytes);
        shard.extend_from_slice(payload);
        fs::write(dir.join("model.safetensors"), shard)
            .unwrap_or_else(|error| panic!("write test shard: {error}"));
    }

    fn write_tensors(dir: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, tensor_bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(tensor_bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header)
            .unwrap_or_else(|error| panic!("serialize multi-tensor test header: {error}"));
        let mut shard = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header_bytes);
        shard.extend_from_slice(&payload);
        fs::write(dir.join("model.safetensors"), shard)
            .unwrap_or_else(|error| panic!("write multi-tensor test shard: {error}"));
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn write_small_f16_gated_ffn(dir: &Path) {
        // hidden=2, intermediate=3. All three matrices are deliberately
        // nonzero and asymmetric so graph/weight reordering changes the KAT.
        write_tensors(
            dir,
            &[
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    "F16",
                    &[3, 2],
                    f16_bytes(&[0.5, -1.0, 1.5, 0.25, -0.75, 2.0]),
                ),
                (
                    "model.layers.0.mlp.up_proj.weight",
                    "F16",
                    &[3, 2],
                    f16_bytes(&[1.0, 0.5, -0.25, 2.0, 1.25, -1.5]),
                ),
                (
                    "model.layers.0.mlp.down_proj.weight",
                    "F16",
                    &[2, 3],
                    f16_bytes(&[0.75, -0.5, 1.0, -1.25, 0.25, 0.5]),
                ),
            ],
        );
    }

    fn small_gated_ffn_request(dir: &Path) -> CoreMlGatedFfnExportRequest<'_> {
        CoreMlGatedFfnExportRequest {
            model_dir: dir,
            gate_tensor_name: "model.layers.0.mlp.gate_proj.weight",
            up_tensor_name: "model.layers.0.mlp.up_proj.weight",
            down_tensor_name: "model.layers.0.mlp.down_proj.weight",
            hidden_features: 2,
            intermediate_features: 3,
            spatial: 2,
        }
    }

    fn refresh_gated_ffn_model_identity(artifact: &mut CoreMlGatedFfnArtifact, model: &Model) {
        artifact.model_bytes = model.encode_to_vec();
        artifact.model_sha256 = sha256_hex(&artifact.model_bytes);
        artifact.artifact_identity_sha256 = gated_ffn_artifact_identity(
            &artifact.gate,
            &artifact.up,
            &artifact.down,
            artifact.hidden_features,
            artifact.intermediate_features,
            artifact.spatial,
            &artifact.model_sha256,
            &artifact.known_answer.input_sha256,
            &artifact.known_answer.expected_output_sha256,
        );
    }

    fn gated_ffn_with_input_feature_mutation(
        artifact: &CoreMlGatedFfnArtifact,
        mutate: impl FnOnce(&mut FeatureType),
    ) -> CoreMlGatedFfnArtifact {
        let mut mutated = artifact.clone();
        let mut model = Model::decode(mutated.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode gated FFN for feature mutation: {error}"));
        let feature_type = model
            .description
            .as_mut()
            .and_then(|description| description.input.first_mut())
            .and_then(|feature| feature.r#type.as_mut())
            .unwrap_or_else(|| panic!("gated FFN input feature type"));
        mutate(feature_type);
        refresh_gated_ffn_model_identity(&mut mutated, &model);
        mutated
    }

    #[test]
    fn exports_nonzero_f16_checkpoint_weights_bit_exactly_and_validates() {
        let dir = test_dir();
        let values = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        write_tensor(
            &dir,
            "model.layers.0.self_attn.q_proj.weight",
            "F16",
            &[2, 3],
            &payload,
        );

        let artifact = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "model.layers.0.self_attn.q_proj.weight",
            out_features: 2,
            in_features: 3,
            spatial: 2,
        })
        .unwrap_or_else(|error| panic!("export projection: {error}"));
        let model = Model::decode(artifact.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode exported model: {error}"));
        let (dtype, exported) = extract_model_weights(&model)
            .unwrap_or_else(|error| panic!("extract exported weights: {error}"));
        assert_eq!(dtype, CoreMlProjectionWeightDtype::F16);
        assert_eq!(exported, payload);
        assert_eq!(
            artifact.known_answer.input,
            vec![
                -9.0 / 16.0,
                8.0 / 16.0,
                4.0 / 16.0,
                -8.0 / 16.0,
                -12.0 / 16.0,
                5.0 / 16.0
            ]
        );
        assert_eq!(
            artifact.known_answer.expected_output,
            vec![-2.03125, 1.375, -1.390625, 0.84375]
        );
        assert!(artifact
            .known_answer
            .expected_output
            .iter()
            .any(|value| *value != 0.0));
        let validation = artifact
            .validate_cpu()
            .unwrap_or_else(|error| panic!("validate artifact: {error}"));
        assert_eq!(validation.max_abs_error, 0.0);
        assert_eq!(
            validation.actual_output_sha256,
            artifact.known_answer.expected_output_sha256
        );
        let second = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "model.layers.0.self_attn.q_proj.weight",
            out_features: 2,
            in_features: 3,
            spatial: 2,
        })
        .unwrap_or_else(|error| panic!("repeat export: {error}"));
        assert_eq!(
            artifact.artifact_identity_sha256,
            second.artifact_identity_sha256
        );
        assert_eq!(artifact.model_bytes, second.model_bytes);
    }

    #[test]
    fn exports_exact_five_layer_gated_ffn_and_validates_deterministically() {
        let dir = test_dir();
        write_small_f16_gated_ffn(&dir);
        let artifact = export_checkpoint_gated_ffn(small_gated_ffn_request(&dir))
            .unwrap_or_else(|error| panic!("export gated FFN: {error}"));
        let validation = artifact
            .validate_cpu()
            .unwrap_or_else(|error| panic!("validate gated FFN: {error}"));
        assert_eq!(validation.max_abs_error, 0.0);
        assert_eq!(
            validation.actual_output_sha256,
            artifact.known_answer.expected_output_sha256
        );
        assert!(artifact
            .known_answer
            .expected_output
            .iter()
            .any(|value| *value != 0.0));

        let model = Model::decode(artifact.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode gated FFN: {error}"));
        let Some(model::Type::NeuralNetwork(network)) = &model.r#type else {
            panic!("gated FFN should be a neural network");
        };
        assert_eq!(network.layers.len(), 5);
        assert!(matches!(
            network.layers[2].layer.as_ref(),
            Some(neural_network_layer::Layer::Gelu(GeluLayerParams {
                mode
            })) if *mode == gelu_layer_params::GeluMode::TanhApproximation as i32
        ));
        assert!(matches!(
            network.layers[3].layer.as_ref(),
            Some(neural_network_layer::Layer::MultiplyBroadcastable(_))
        ));
        let source_payloads = [
            f16_bytes(&[0.5, -1.0, 1.5, 0.25, -0.75, 2.0]),
            f16_bytes(&[1.0, 0.5, -0.25, 2.0, 1.25, -1.5]),
            f16_bytes(&[0.75, -0.5, 1.0, -1.25, 0.25, 0.5]),
        ];
        for (layer_index, source_payload) in [0_usize, 1, 4].into_iter().zip(source_payloads) {
            let Some(neural_network_layer::Layer::Convolution(convolution)) =
                network.layers[layer_index].layer.as_ref()
            else {
                panic!("expected gated FFN convolution layer");
            };
            let (dtype, exported) = extract_convolution_weights(convolution)
                .unwrap_or_else(|error| panic!("extract gated FFN weights: {error}"));
            assert_eq!(dtype, CoreMlProjectionWeightDtype::F16);
            assert_eq!(exported, source_payload);
        }

        let second = export_checkpoint_gated_ffn(small_gated_ffn_request(&dir))
            .unwrap_or_else(|error| panic!("repeat gated FFN export: {error}"));
        assert_eq!(
            artifact.artifact_identity_sha256,
            second.artifact_identity_sha256
        );
        assert_eq!(artifact.model_bytes, second.model_bytes);
    }

    #[test]
    fn gated_ffn_rejects_shape_alias_and_graph_semantic_tampering() {
        let dir = test_dir();
        write_small_f16_gated_ffn(&dir);
        let mut aliased = small_gated_ffn_request(&dir);
        aliased.up_tensor_name = aliased.gate_tensor_name;
        assert!(matches!(
            export_checkpoint_gated_ffn(aliased),
            Err(CoreMlProjectionError::InvalidRequest(_))
        ));

        let mut wrong_shape = small_gated_ffn_request(&dir);
        wrong_shape.intermediate_features = 2;
        assert!(matches!(
            export_checkpoint_gated_ffn(wrong_shape),
            Err(CoreMlProjectionError::Shape { .. })
        ));

        let mut artifact = export_checkpoint_gated_ffn(small_gated_ffn_request(&dir))
            .unwrap_or_else(|error| panic!("export gated FFN: {error}"));
        let mut model = Model::decode(artifact.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode gated FFN: {error}"));
        let Some(model::Type::NeuralNetwork(network)) = model.r#type.as_mut() else {
            panic!("gated FFN should be a neural network");
        };
        let Some(neural_network_layer::Layer::Gelu(parameters)) = network.layers[2].layer.as_mut()
        else {
            panic!("gated FFN should contain GELU");
        };
        parameters.mode = gelu_layer_params::GeluMode::Exact as i32;
        refresh_gated_ffn_model_identity(&mut artifact, &model);
        let error = artifact
            .validate_cpu()
            .expect_err("semantic GELU mutation must fail even with refreshed digests");
        assert!(matches!(error, CoreMlProjectionError::Artifact(_)));
    }

    #[test]
    fn gated_ffn_rejects_refreshed_padding_and_feature_contract_tampering() {
        let dir = test_dir();
        write_small_f16_gated_ffn(&dir);
        let artifact = export_checkpoint_gated_ffn(small_gated_ffn_request(&dir))
            .unwrap_or_else(|error| panic!("export gated FFN: {error}"));

        let mut padded = artifact.clone();
        let mut padded_model = Model::decode(padded.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode gated FFN for padding mutation: {error}"));
        let Some(model::Type::NeuralNetwork(network)) = padded_model.r#type.as_mut() else {
            panic!("gated FFN should be a neural network");
        };
        let Some(neural_network_layer::Layer::Convolution(convolution)) =
            network.layers[0].layer.as_mut()
        else {
            panic!("gated FFN gate projection should be a convolution");
        };
        let Some(convolution_layer_params::ConvolutionPaddingType::Valid(valid)) =
            convolution.convolution_padding_type.as_mut()
        else {
            panic!("gated FFN gate projection should use valid padding");
        };
        valid.padding_amounts = Some(rvllm_apple_coreml_sys::specification::BorderAmounts {
            border_amounts: vec![
                rvllm_apple_coreml_sys::specification::border_amounts::EdgeSizes {
                    start_edge_size: 1,
                    end_edge_size: 0,
                },
                rvllm_apple_coreml_sys::specification::border_amounts::EdgeSizes::default(),
            ],
        });
        refresh_gated_ffn_model_identity(&mut padded, &padded_model);
        assert!(matches!(
            padded.validate_cpu(),
            Err(CoreMlProjectionError::Artifact(_))
        ));

        let optional = gated_ffn_with_input_feature_mutation(&artifact, |feature| {
            feature.is_optional = true;
        });
        assert!(matches!(
            optional.validate_cpu(),
            Err(CoreMlProjectionError::Artifact(_))
        ));

        let flexible = gated_ffn_with_input_feature_mutation(&artifact, |feature| {
            let Some(feature_type::Type::MultiArrayType(array)) = feature.r#type.as_mut() else {
                panic!("gated FFN input should be a multi-array");
            };
            array.shape_flexibility = Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(
                array_feature_type::EnumeratedShapes {
                    shapes: vec![array_feature_type::Shape {
                        shape: array.shape.clone(),
                    }],
                },
            ));
        });
        assert!(matches!(
            flexible.validate_cpu(),
            Err(CoreMlProjectionError::Artifact(_))
        ));

        let defaulted = gated_ffn_with_input_feature_mutation(&artifact, |feature| {
            let Some(feature_type::Type::MultiArrayType(array)) = feature.r#type.as_mut() else {
                panic!("gated FFN input should be a multi-array");
            };
            array.default_optional_value = Some(
                array_feature_type::DefaultOptionalValue::FloatDefaultValue(0.0),
            );
        });
        assert!(matches!(
            defaulted.validate_cpu(),
            Err(CoreMlProjectionError::Artifact(_))
        ));
    }

    #[test]
    #[ignore = "requires the public Core ML runtime on Apple Silicon"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn exact_gated_ffn_executes_through_public_coreml_without_ane_claim() {
        let dir = test_dir();
        write_small_f16_gated_ffn(&dir);
        let artifact = export_checkpoint_gated_ffn(small_gated_ffn_request(&dir))
            .unwrap_or_else(|error| panic!("export gated FFN: {error}"));
        let validation = validate_public_coreml_gated_ffn(
            &artifact,
            &dir.join("public-coreml-gated-ffn"),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            5.0e-3,
        )
        .unwrap_or_else(|error| panic!("execute exact public Core ML gated FFN: {error}"));
        assert!(validation.max_abs_error <= 5.0e-3);
        assert_eq!(
            validation.execution_report.requested_compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
        assert!(!validation.execution_report.neural_engine_execution_verified);
        assert!(validation
            .execution_report
            .honest_summary()
            .contains("not verified"));
    }

    #[test]
    #[ignore = "requires the public Core ML runtime on Apple Silicon"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn exact_nonzero_projection_executes_through_public_coreml_without_ane_claim() {
        let dir = test_dir();
        let values = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        write_tensor(&dir, "w", "F16", &[2, 3], &payload);
        let artifact = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "w",
            out_features: 2,
            in_features: 3,
            spatial: 2,
        })
        .unwrap_or_else(|error| panic!("export projection: {error}"));

        let validation = validate_public_coreml_projection(
            &artifact,
            &dir.join("public-coreml"),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            1.0e-3,
        )
        .unwrap_or_else(|error| panic!("execute exact public Core ML projection: {error}"));

        assert!(validation.max_abs_error <= 1.0e-3);
        assert_eq!(
            validation.execution_report.requested_compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
        assert!(!validation.execution_report.neural_engine_execution_verified);
        assert!(validation
            .execution_report
            .honest_summary()
            .contains("not verified"));
    }

    #[test]
    fn rejects_checkpoint_shape_mismatch() {
        let dir = test_dir();
        let payload = vec![0_u8; 12];
        write_tensor(&dir, "w", "F16", &[3, 2], &payload);
        let error = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "w",
            out_features: 2,
            in_features: 3,
            spatial: 1,
        })
        .expect_err("transposed checkpoint tensor must be rejected");
        assert!(matches!(error, CoreMlProjectionError::Shape { .. }));
    }

    #[test]
    fn widens_bf16_checkpoint_weights_to_f32_without_numerical_loss() {
        let dir = test_dir();
        let values = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        write_tensor(&dir, "w", "BF16", &[2, 3], &payload);
        let artifact = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "w",
            out_features: 2,
            in_features: 3,
            spatial: 1,
        })
        .unwrap_or_else(|error| panic!("export BF16 projection: {error}"));
        assert_eq!(artifact.source_dtype, CoreMlProjectionWeightDtype::Bf16);
        let model = Model::decode(artifact.model_bytes.as_slice())
            .unwrap_or_else(|error| panic!("decode BF16 projection: {error}"));
        let (model_dtype, model_bytes) = extract_model_weights(&model)
            .unwrap_or_else(|error| panic!("extract BF16 projection: {error}"));
        assert_eq!(model_dtype, CoreMlProjectionWeightDtype::F32);
        let model_values = decode_weights(model_dtype, &model_bytes)
            .unwrap_or_else(|error| panic!("decode widened BF16 weights: {error}"));
        assert_eq!(model_values, values);
        assert_eq!(
            encode_source_weights(artifact.source_dtype, &model_values),
            payload
        );
    }

    #[test]
    fn rejects_dtype_without_lossless_coreml_weight_mapping() {
        let dir = test_dir();
        let payload = vec![1_u8; 6];
        write_tensor(&dir, "w", "U8", &[2, 3], &payload);
        let error = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "w",
            out_features: 2,
            in_features: 3,
            spatial: 1,
        })
        .expect_err("U8 must not be silently interpreted as dense float weights");
        assert!(matches!(
            error,
            CoreMlProjectionError::UnsupportedDtype { .. }
        ));
    }

    #[test]
    fn artifact_validation_rejects_model_tampering() {
        let dir = test_dir();
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        write_tensor(&dir, "w", "F32", &[2, 2], &payload);
        let mut artifact = export_checkpoint_projection(CoreMlProjectionExportRequest {
            model_dir: &dir,
            tensor_name: "w",
            out_features: 2,
            in_features: 2,
            spatial: 1,
        })
        .unwrap_or_else(|error| panic!("export projection: {error}"));
        let last = artifact.model_bytes.len() - 1;
        artifact.model_bytes[last] ^= 1;
        let error = artifact
            .validate_cpu()
            .expect_err("tampered model must fail validation");
        assert!(matches!(error, CoreMlProjectionError::Artifact(_)));
    }
}
