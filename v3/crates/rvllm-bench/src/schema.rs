use rvllm_core::{ConfigError, Result, RvllmError};
use serde::{Deserialize, Serialize};

pub const BENCH_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchReport {
    pub schema_version: u32,
    pub harness: BenchHarnessStub,
    pub device: BenchDeviceMetadata,
    pub run: BenchRunMetrics,
}

impl BenchReport {
    #[must_use]
    pub const fn new(
        harness: BenchHarnessStub,
        device: BenchDeviceMetadata,
        run: BenchRunMetrics,
    ) -> Self {
        Self {
            schema_version: BENCH_SCHEMA_VERSION,
            harness,
            device,
            run,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BENCH_SCHEMA_VERSION {
            return Err(invalid_field(
                "schema_version",
                format!(
                    "expected {BENCH_SCHEMA_VERSION}, got {}",
                    self.schema_version
                ),
            ));
        }
        self.harness.validate()?;
        self.device.validate()?;
        self.run.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchHarnessStub {
    pub name: String,
    pub kind: String,
}

impl BenchHarnessStub {
    #[must_use]
    pub fn new(name: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_non_empty("harness.name", &self.name)?;
        require_non_empty("harness.kind", &self.kind)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDeviceMetadata {
    pub backend: String,
    pub name: String,
    pub kind: String,
}

impl BenchDeviceMetadata {
    #[must_use]
    pub fn new(
        backend: impl Into<String>,
        name: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            backend: backend.into(),
            name: name.into(),
            kind: kind.into(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_non_empty("device.backend", &self.backend)?;
        require_non_empty("device.name", &self.name)?;
        require_non_empty("device.kind", &self.kind)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRunMetrics {
    pub batch: u32,
    pub iters: u32,
    pub generated_tokens: u64,
    pub tok_per_sec: f64,
    pub energy: BenchEnergyMetrics,
    pub latency_ms: BenchLatencyStats,
}

impl BenchRunMetrics {
    #[must_use]
    pub const fn new(
        batch: u32,
        iters: u32,
        generated_tokens: u64,
        tok_per_sec: f64,
        energy: BenchEnergyMetrics,
        latency_ms: BenchLatencyStats,
    ) -> Self {
        Self {
            batch,
            iters,
            generated_tokens,
            tok_per_sec,
            energy,
            latency_ms,
        }
    }

    pub fn from_elapsed_ns(
        batch: u32,
        iters: u32,
        generated_tokens: u64,
        elapsed_ns: u128,
        watts: f64,
        latency_ms: BenchLatencyStats,
    ) -> Result<Self> {
        if elapsed_ns == 0 {
            return Err(invalid_field("elapsed_ns", "must be greater than zero"));
        }
        if generated_tokens == 0 {
            return Err(invalid_field(
                "generated_tokens",
                "must be greater than zero",
            ));
        }

        let elapsed_s = elapsed_ns as f64 / 1.0e9;
        let tok_per_sec = generated_tokens as f64 / elapsed_s;
        let joules_per_token = watts * elapsed_s / generated_tokens as f64;
        let metrics = Self::new(
            batch,
            iters,
            generated_tokens,
            tok_per_sec,
            BenchEnergyMetrics::new(watts, joules_per_token),
            latency_ms,
        );
        metrics.validate()?;
        Ok(metrics)
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(invalid_field("run.batch", "must be greater than zero"));
        }
        if self.iters == 0 {
            return Err(invalid_field("run.iters", "must be greater than zero"));
        }
        if self.generated_tokens == 0 {
            return Err(invalid_field(
                "run.generated_tokens",
                "must be greater than zero",
            ));
        }
        require_finite_non_negative("run.tok_per_sec", self.tok_per_sec)?;
        self.energy.validate()?;
        self.latency_ms.validate()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchEnergyMetrics {
    pub watts: f64,
    pub joules_per_token: f64,
}

impl BenchEnergyMetrics {
    #[must_use]
    pub const fn new(watts: f64, joules_per_token: f64) -> Self {
        Self {
            watts,
            joules_per_token,
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_finite_non_negative("run.energy.watts", self.watts)?;
        require_finite_non_negative("run.energy.joules_per_token", self.joules_per_token)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchLatencyStats {
    pub p50: f64,
    pub p95: f64,
}

impl BenchLatencyStats {
    #[must_use]
    pub const fn new(p50: f64, p95: f64) -> Self {
        Self { p50, p95 }
    }

    pub fn validate(&self) -> Result<()> {
        require_finite_non_negative("run.latency_ms.p50", self.p50)?;
        require_finite_non_negative("run.latency_ms.p95", self.p95)?;
        if self.p95 < self.p50 {
            return Err(invalid_field(
                "run.latency_ms.p95",
                "must be greater than or equal to p50",
            ));
        }
        Ok(())
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<()> {
    if value.is_empty() {
        Err(invalid_field(field, "must not be empty"))
    } else {
        Ok(())
    }
}

fn require_finite_non_negative(field: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(invalid_field(field, "must be finite"));
    }
    if value < 0.0 {
        return Err(invalid_field(field, "must be non-negative"));
    }
    Ok(())
}

fn invalid_field(field: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.into(),
        },
        field,
    )
}
