use rvllm_core::Result;

use crate::schema::{
    BenchDeviceMetadata, BenchHarnessStub, BenchLatencyStats, BenchReport, BenchRunMetrics,
};

#[derive(Clone, Debug, PartialEq)]
pub struct EnergyBenchHarness {
    pub harness: BenchHarnessStub,
    pub device: BenchDeviceMetadata,
}

impl EnergyBenchHarness {
    #[must_use]
    pub const fn new(harness: BenchHarnessStub, device: BenchDeviceMetadata) -> Self {
        Self { harness, device }
    }

    pub fn report(&self, run: BenchRunMetrics) -> Result<BenchReport> {
        let report = BenchReport::new(self.harness.clone(), self.device.clone(), run);
        report.validate()?;
        Ok(report)
    }

    pub fn report_from_measurement(&self, measurement: HarnessMeasurement) -> Result<BenchReport> {
        self.report(BenchRunMetrics::from_elapsed_ns(
            measurement.batch,
            measurement.iters,
            measurement.generated_tokens,
            measurement.elapsed_ns,
            measurement.watts,
            measurement.latency_ms,
        )?)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HarnessMeasurement {
    pub batch: u32,
    pub iters: u32,
    pub generated_tokens: u64,
    pub elapsed_ns: u128,
    pub watts: f64,
    pub latency_ms: BenchLatencyStats,
}

impl HarnessMeasurement {
    #[must_use]
    pub const fn new(
        batch: u32,
        iters: u32,
        generated_tokens: u64,
        elapsed_ns: u128,
        watts: f64,
        latency_ms: BenchLatencyStats,
    ) -> Self {
        Self {
            batch,
            iters,
            generated_tokens,
            elapsed_ns,
            watts,
            latency_ms,
        }
    }
}
