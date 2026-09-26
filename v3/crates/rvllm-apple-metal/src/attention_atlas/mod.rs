//! Default-off, source-pinned Apple attention experiments.
//!
//! This is an operator candidate API, not a shipping selector or an implicit
//! replacement of the existing route referee. No model/cache ABI is changed.
//! Metadata and cache snapshots are immutable for the lifetime of a dispatch.
pub mod ane;
pub mod experiment;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod plan;
pub mod reference;
#[cfg(test)]
mod tests;

pub use plan::{CacheFormat, Candidate, Output, Plan, Shape};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Debug)]
pub struct Error(String);
impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub const COMMON: &str = include_str!("shaders/common.metal");
pub const MATRIX: &str = include_str!("shaders/matrix.metal");

/// Exact source frozen by the preparation command. It contains one candidate,
/// the atlas vector control, and the existing baseline kernel source generated
/// without environment-derived or research options. Native runs LOAD the frozen
/// metallib, never compile a source string or alter compilation options.
pub fn source(candidate: Candidate, dim: u32) -> Result<String> {
    candidate.validate()?;
    if !plan::catalog().contains(&candidate) || !matches!(dim, 256 | 512) {
        return Err(Error::new("candidate is outside the bounded catalog"));
    }
    let mut out = crate::kernels::kernel_source_with_options(
        crate::MetalFloatType::Bf16,
        crate::MetalKernelOptions::default(),
    )
    .into_owned();
    if dim == 512 {
        // Reuse every EXISTING cooperative control body, unmodified. Include
        // the common helper only once rather than concatenating eight sources
        // that each embed it. No upstream selector or threshold is changed.
        out.push_str(include_str!(
            "../research_shaders/global_decode_common.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_matrix_common.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_atlas_mma_r8k32p64t128.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_split_common.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_split_matrix_common.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_split_mma_r8k32s256t128.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r8p64t64.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r8p64t128.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r8p128t64.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r8p128t128.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r16p64t64.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r16p64t128.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r16p128t64.metal"
        ));
        out.push_str(include_str!(
            "../research_shaders/global_decode_r16p128t128.metal"
        ));
    }
    out.push_str("\n// BEGIN attention-atlas v1\n");
    out.push_str(COMMON);
    out.push_str(MATRIX);
    let c = candidate;
    match c.strategy {
        plan::Strategy::Matrix => out.push_str(&format!(
            "\nATLAS_MATRIX_ENTRY(atlas_candidate,{dim},{},{},{},{})\n",
            c.rows, c.keys, c.panel, c.threads
        )),
        _ => out.push_str(&format!(
            "\nATLAS_COOP_ENTRY(atlas_candidate,{dim},{},{},{},{},{},{})\n",
            c.rows,
            c.keys,
            c.panel,
            c.threads,
            c.softmax == plan::Softmax::PerKey,
            c.splits
        )),
    }
    out.push_str(&format!(
        "ATLAS_COOP_ENTRY(atlas_vector,{dim},1,1,64,32,true,1)\n"
    ));
    Ok(out)
}
