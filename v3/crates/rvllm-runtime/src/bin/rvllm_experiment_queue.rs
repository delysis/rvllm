//! A local, serial experiment queue. It never invokes a shell or repairs caches.
#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
#[path = "rvllm_compare_disaggregated/fair.rs"]
mod fair;
#[cfg(target_os = "macos")]
#[path = "rvllm_experiment_queue/queue.rs"]
mod queue;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    queue::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("The experiment power observer requires macOS.");
    std::process::exit(1);
}
