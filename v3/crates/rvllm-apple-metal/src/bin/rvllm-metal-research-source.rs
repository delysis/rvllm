//! Host-only source export. Does not create a device or execute a kernel.
#![forbid(unsafe_code)]
use rvllm_apple_metal::{MetalFloatType, MetalKernelOptions, MetalResearchCandidate};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--catalog") {
        if args.next().is_some() {
            return Err("--catalog accepts no additional arguments".into());
        }
        let mut output = std::io::stdout().lock();
        serde_json::to_writer_pretty(
            &mut output,
            &rvllm_apple_metal::research_catalog::catalog_json(),
        )?;
        output.write_all(b"\n")?;
        return Ok(());
    }
    let dtype = match first.as_deref() {
        Some("bf16") => MetalFloatType::Bf16,
        Some("f16") => MetalFloatType::F16,
        _ => return Err("usage: rvllm-metal-research-source bf16|f16 off|CANDIDATE".into()),
    };
    let research: MetalResearchCandidate =
        args.next().ok_or("explicit candidate required")?.parse()?;
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let source = rvllm_apple_metal::kernels::kernel_source_with_options(
        dtype,
        MetalKernelOptions {
            research,
            ..MetalKernelOptions::default()
        },
    );
    std::io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
