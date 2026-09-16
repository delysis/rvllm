use std::path::PathBuf;

use rvllm_apple_metal::{kernels, MetalFloatType};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dtype = match args.next().as_deref() {
        Some("f16") => MetalFloatType::F16,
        Some("bf16") => MetalFloatType::Bf16,
        _ => return Err("usage: emit_metal_kernels f16|bf16 <SOURCE> <MANIFEST>".to_owned()),
    };
    let source_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing Metal source output path".to_owned())?,
    );
    let manifest_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing pipeline manifest output path".to_owned())?,
    );
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }

    let source = kernels::kernel_source_for_float_type(dtype);
    std::fs::write(&source_path, source.as_bytes())
        .map_err(|error| format!("write Metal source {}: {error}", source_path.display()))?;
    let kernel_names = kernels::KERNEL_NAMES
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    let manifest = format!(
        "{{\n  \"schema\": \"rvllm.apple-metal.pipeline-manifest.v1\",\n  \"dtype\": \"{}\",\n  \"kv_page_tokens\": 32,\n  \"kernel_count\": {},\n  \"kernels\": [{}]\n}}\n",
        dtype.report_name(),
        kernels::KERNEL_NAMES.len(),
        kernel_names
    );
    std::fs::write(&manifest_path, manifest).map_err(|error| {
        format!(
            "write pipeline manifest {}: {error}",
            manifest_path.display()
        )
    })
}
