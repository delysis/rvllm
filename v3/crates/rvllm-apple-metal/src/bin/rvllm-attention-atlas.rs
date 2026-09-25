//! Explicit source/compile/oracle/benchmark stages. Never launches a worker.
#![forbid(unsafe_code)]
use rvllm_apple_metal::attention_atlas::{self as atlas, experiment as e, Error, Result};
use std::path::Path;
const USAGE:&str="usage:\n  rvllm-attention-atlas catalog OUT.json\n  rvllm-attention-atlas campaign ABS_NEW_DIR\n  rvllm-attention-atlas prepare SPEC.json ABS_NEW_DIR\n  rvllm-attention-atlas compile PREPARED ABS_NEW_DIR\n  rvllm-attention-atlas oracle PREPARED BUILD ABS_NEW_DIR\n  rvllm-attention-atlas bench PREPARED BUILD ORACLE.json ABS_NEW_DIR\n  rvllm-attention-atlas queue REQUEST.json OUT.json\n  rvllm-attention-atlas ane-layout SPEC.json OUT.json\nAll native stages require the local exclusive owner. Benchmark output is operator-only, never promotion evidence.";
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let a = std::env::args_os()
        .skip(1)
        .map(|s| {
            s.into_string()
                .map_err(|_| Error::new("arguments must be UTF8"))
        })
        .collect::<Result<Vec<_>>>()?;
    match a.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["catalog", out] => e::write_json(Path::new(out), &atlas::plan::catalog()),
        ["campaign", out] => e::campaign(Path::new(out)),
        ["prepare", spec, out] => e::prepare(Path::new(spec), Path::new(out)),
        ["compile", prepared, out] => e::compile(Path::new(prepared), Path::new(out)),
        ["oracle", prepared, build, out] => {
            native(Path::new(prepared), Path::new(build), None, Path::new(out))
        }
        ["bench", prepared, build, receipt, out] => native(
            Path::new(prepared),
            Path::new(build),
            Some(Path::new(receipt)),
            Path::new(out),
        ),
        ["queue", request, out] => e::queue(&e::read_json(Path::new(request))?, Path::new(out)),
        ["ane-layout", spec, out] => {
            let s: e::Spec = e::read_json(Path::new(spec))?;
            let plan = s.plan(atlas::Output::Bf16)?;
            e::write_json(
                Path::new(out),
                &atlas::ane::PackedLayout::new(plan, 64 * 1024 * 1024)?,
            )
        }
        _ => Err(Error::new(USAGE)),
    }
}
#[cfg(target_os = "macos")]
fn native(prepared: &Path, build: &Path, receipt: Option<&Path>, out: &Path) -> Result<()> {
    atlas::metal::run(prepared, build, receipt, out)
}
#[cfg(not(target_os = "macos"))]
fn native(_: &Path, _: &Path, _: Option<&Path>, _: &Path) -> Result<()> {
    Err(Error::new(
        "native Apple device stage unavailable on this host; no simulated success",
    ))
}
