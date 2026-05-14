use std::collections::BTreeMap;

use rvllm_core::{ConfigError, Result, RvllmError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const BENCH_SCHEMA_VERSION: &str = "rvllm.bench.energy.v1";

const REQUIRED_FIELDS: &[&str] = &[
    "schema_version",
    "workload",
    "workload.model",
    "workload.batch",
    "workload.prompt_tokens",
    "workload.generated_tokens",
    "workload.warmup_iters",
    "workload.measured_iters",
    "metrics",
    "metrics.tok_per_sec",
    "metrics.watts",
    "metrics.joules_per_token",
    "metrics.p50_ms",
    "metrics.p95_ms",
    "device",
    "device.backend",
    "device.device_name",
    "device.architecture",
    "device.accelerator",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub schema_version: String,
    pub workload: BenchWorkload,
    pub metrics: BenchMetrics,
    pub device: BenchDeviceMetadata,
}

impl BenchReport {
    #[must_use]
    pub fn new(
        workload: BenchWorkload,
        metrics: BenchMetrics,
        device: BenchDeviceMetadata,
    ) -> Self {
        Self {
            schema_version: BENCH_SCHEMA_VERSION.to_owned(),
            workload,
            metrics,
            device,
        }
    }

    pub fn from_json_value(value: Value) -> Result<Self> {
        validate_required_fields(&value)?;
        let report: Self = serde_json::from_value(value).map_err(|e| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "bench_report",
                    reason: e.to_string(),
                },
                "bench_report",
            )
        })?;
        report.validate()?;
        Ok(report)
    }

    pub fn from_json_str(body: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(body).map_err(|e| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "bench_report",
                    reason: e.to_string(),
                },
                "bench_report",
            )
        })?;
        Self::from_json_value(value)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BENCH_SCHEMA_VERSION {
            return Err(invalid_field(
                "schema_version",
                "unsupported bench schema version",
            ));
        }
        self.workload.validate()?;
        self.metrics.validate()?;
        self.device.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchWorkload {
    pub model: String,
    pub batch: u32,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub warmup_iters: u32,
    pub measured_iters: u32,
}

impl BenchWorkload {
    pub fn validate(&self) -> Result<()> {
        if self.model.is_empty() {
            return Err(invalid_field("workload.model", "model must not be empty"));
        }
        require_nonzero("workload.batch", self.batch)?;
        require_nonzero("workload.prompt_tokens", self.prompt_tokens)?;
        require_nonzero("workload.generated_tokens", self.generated_tokens)?;
        require_nonzero("workload.measured_iters", self.measured_iters)?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchMetrics {
    pub tok_per_sec: f64,
    pub watts: f64,
    pub joules_per_token: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

impl BenchMetrics {
    #[must_use]
    pub fn new(
        tok_per_sec: f64,
        watts: f64,
        joules_per_token: f64,
        p50_ms: f64,
        p95_ms: f64,
    ) -> Self {
        Self {
            tok_per_sec,
            watts,
            joules_per_token,
            p50_ms,
            p95_ms,
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_finite_positive("metrics.tok_per_sec", self.tok_per_sec)?;
        require_finite_positive("metrics.watts", self.watts)?;
        require_finite_positive("metrics.joules_per_token", self.joules_per_token)?;
        require_finite_nonnegative("metrics.p50_ms", self.p50_ms)?;
        require_finite_nonnegative("metrics.p95_ms", self.p95_ms)?;
        if self.p95_ms < self.p50_ms {
            return Err(invalid_field(
                "metrics.p95_ms",
                "p95_ms must be greater than or equal to p50_ms",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchDeviceMetadata {
    pub backend: String,
    pub device_name: String,
    pub architecture: String,
    pub accelerator: String,
    pub memory_bytes: Option<u64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl BenchDeviceMetadata {
    #[must_use]
    pub fn new(
        backend: impl Into<String>,
        device_name: impl Into<String>,
        architecture: impl Into<String>,
        accelerator: impl Into<String>,
    ) -> Self {
        Self {
            backend: backend.into(),
            device_name: device_name.into(),
            architecture: architecture.into(),
            accelerator: accelerator.into(),
            memory_bytes: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_nonempty("device.backend", &self.backend)?;
        require_nonempty("device.device_name", &self.device_name)?;
        require_nonempty("device.architecture", &self.architecture)?;
        require_nonempty("device.accelerator", &self.accelerator)?;
        Ok(())
    }
}

#[cfg(feature = "apple")]
impl From<&rvllm_apple::AppleAcceleratorTarget> for BenchDeviceMetadata {
    fn from(target: &rvllm_apple::AppleAcceleratorTarget) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "gpu_family".to_owned(),
            Value::String(format!("{:?}", target.gpu_family)),
        );
        metadata.insert(
            "npu_generation".to_owned(),
            Value::String(format!("{:?}", target.npu_generation)),
        );
        metadata.insert(
            "architecture_gen".to_owned(),
            Value::from(u64::from(target.architecture_gen)),
        );
        metadata.insert("has_nax".to_owned(), Value::from(target.has_nax));
        metadata.insert(
            "ane_cores".to_owned(),
            Value::from(u64::from(target.ane_cores)),
        );
        metadata.insert(
            "die_count".to_owned(),
            Value::from(u64::from(target.die_count)),
        );
        metadata.insert(
            "device_tier".to_owned(),
            Value::String(format!("{:?}", target.tier)),
        );

        Self {
            backend: "apple".to_owned(),
            device_name: target.device_name.clone(),
            architecture: std::env::consts::ARCH.to_owned(),
            accelerator: "metal+ane".to_owned(),
            memory_bytes: None,
            metadata,
        }
    }
}

fn validate_required_fields(value: &Value) -> Result<()> {
    for &field in REQUIRED_FIELDS {
        if lookup_path(value, field).is_none() {
            return Err(RvllmError::config(
                ConfigError::MissingField { name: field },
                field,
            ));
        }
    }
    Ok(())
}

fn lookup_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cursor = value;
    for segment in path.split('.') {
        cursor = cursor.get(segment)?;
    }
    Some(cursor)
}

fn require_nonempty(field: &'static str, value: &str) -> Result<()> {
    if value.is_empty() {
        Err(invalid_field(field, "value must not be empty"))
    } else {
        Ok(())
    }
}

fn require_nonzero(field: &'static str, value: u32) -> Result<()> {
    if value == 0 {
        Err(invalid_field(field, "value must be nonzero"))
    } else {
        Ok(())
    }
}

fn require_finite_positive(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_field(field, "value must be finite and positive"))
    }
}

fn require_finite_nonnegative(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_field(field, "value must be finite and nonnegative"))
    }
}

fn invalid_field(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_owned(),
        },
        field,
    )
}
