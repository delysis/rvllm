//! Verify rootless measurement capabilities without loading model weights.
#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rvllm_runtime::apple_measurement::{metal_counter_capabilities, PowerMonitor};
    use std::time::{Duration, Instant};
    let monitor = PowerMonitor::start(None)?;
    let phase = monitor.begin();
    let mut value = 1_u64;
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(100) {
        for _ in 0..1000 {
            value = std::hint::black_box(value.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
    }
    let busy = phase.finish(1);
    let phase = monitor.begin();
    std::thread::sleep(Duration::from_millis(1100));
    let idle = phase.finish(1);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
        "schema":"rvllm.apple_measurement_probe.v1", "cpu_work":busy,"sleep":idle,
        "metal_capabilities":metal_counter_capabilities(),"work_checksum":value,
        "claim":"Counter capability check only; no model inference or speedup claim."}))?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Apple measurement requires macOS");
}
