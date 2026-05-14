use rvllm_core::{ConfigError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::schema::{BenchDeviceMetadata, BenchMetrics, BenchReport, BenchWorkload};

pub trait BenchHarness {
    fn run(&mut self) -> Result<BenchReport>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchHarnessConfig {
    pub workload: BenchWorkload,
    pub device: BenchDeviceMetadata,
}

impl BenchHarnessConfig {
    pub fn validate(&self) -> Result<()> {
        self.workload.validate()?;
        self.device.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StubBenchHarness {
    config: BenchHarnessConfig,
    metrics: Option<BenchMetrics>,
}

impl StubBenchHarness {
    #[must_use]
    pub fn new(config: BenchHarnessConfig) -> Self {
        Self {
            config,
            metrics: None,
        }
    }

    #[must_use]
    pub fn with_metrics(config: BenchHarnessConfig, metrics: BenchMetrics) -> Self {
        Self {
            config,
            metrics: Some(metrics),
        }
    }
}

impl BenchHarness for StubBenchHarness {
    fn run(&mut self) -> Result<BenchReport> {
        self.config.validate()?;
        let Some(metrics) = self.metrics.take() else {
            return Err(RvllmError::config(
                ConfigError::MissingField { name: "metrics" },
                "metrics",
            ));
        };
        metrics.validate()?;
        Ok(BenchReport::new(
            self.config.workload.clone(),
            metrics,
            self.config.device.clone(),
        ))
    }
}
