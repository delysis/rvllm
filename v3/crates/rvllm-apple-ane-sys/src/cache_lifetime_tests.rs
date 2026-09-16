//! Opt-in, bounded source-lifetime experiment. No requests or evaluations.
#![forbid(unsafe_code)]

use super::{compiled_model_exists, AneInMemoryProgram, AneProgramCachePolicy};
use std::rc::Rc;

const INPUT: usize = 4096;
const OUTPUT: usize = 3840;

fn fixture(tag: &str, retained: bool) -> (String, Vec<u8>) {
    let arm = if retained { "retained" } else { "immediate" };
    let mil = format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {INPUT}, 1, 1]> x) {{
        string pad_type = const()[name = string("pad_type"), val = string("valid")];
        tensor<int32, [2]> strides = const()[name = string("strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pad = const()[name = string("pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dilations = const()[name = string("dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 groups = const()[name = string("groups"), val = int32(1)];
        tensor<fp16, [{OUTPUT}, {INPUT}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{OUTPUT}, {INPUT}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {OUTPUT}, 1, 1]> y = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = W, x = x)[name = string("rvllm_lifetime_{tag}_{arm}")];
    }} -> (y);
}}
"#
    );
    let weight_bytes = INPUT * OUTPUT * 2;
    let mut blob = vec![0; 128 + weight_bytes];
    blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
    blob[72..80].copy_from_slice(&(weight_bytes as u64).to_le_bytes());
    blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
    let mut state = 0x91b7_c435_u32;
    for bytes in blob[128..].chunks_exact_mut(2) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Dense, finite FP16 weights, identical across arms. No zero/identity
        // special case that might produce a materially smaller lowered graph.
        let bits = 0x2400 | (state as u16 & 0x83ff);
        bytes.copy_from_slice(&bits.to_le_bytes());
    }
    (mil, blob)
}

#[test]
#[ignore = "explicit compile/verify phase; two bounded O44-sized compiles, zero evaluations"]
fn paired_source_lifetime() {
    let phase = std::env::var("RVLLM_ANE_CACHE_LIFETIME_PHASE").unwrap();
    assert!(matches!(phase.as_str(), "compile" | "verify"));
    let tag = std::env::var("RVLLM_ANE_CACHE_LIFETIME_TAG").unwrap();
    assert!(!tag.is_empty() && tag.len() <= 64);
    assert!(tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'));
    assert_eq!(super::compile_budget_used(), 0);
    for retained in [false, true] {
        let (mil, blob) = fixture(&tag, retained);
        let open = |policy| {
            AneInMemoryProgram::compile_with_staging_lifetime(
                &mil,
                &blob,
                &[INPUT * 64],
                &[OUTPUT * 64],
                policy,
                retained,
            )
        };
        if phase == "compile" {
            assert!(
                matches!(open(AneProgramCachePolicy::RequireExisting),
                Err(error) if error.contains("absent from")),
                "fixture must be cold"
            );
        }
        let policy = if phase == "compile" {
            AneProgramCachePolicy::ReuseOrCompileUpTo(2)
        } else {
            AneProgramCachePolicy::RequireExisting
        };
        let mut program = open(policy).unwrap();
        let path = program.inner._directory.0.clone();
        let exists_after_load = compiled_model_exists(&program.inner.model).unwrap();
        println!("lifetime phase={phase} retained={retained} model_id={} source_bytes={} cache_after_load={exists_after_load} data_present={} weight_present={}",
            program.inner.model_id().unwrap(), blob.len(), path.join("data").exists(), path.join("weights/weight.bin").exists());
        if phase == "compile" {
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        let exists_before_unload = compiled_model_exists(&program.inner.model).unwrap();
        println!("lifetime retained={retained} cache_before_unload={exists_before_unload} data_present={} weight_present={}",
            path.join("data").exists(), path.join("weights/weight.bin").exists());
        Rc::get_mut(&mut program.inner).unwrap().unload().unwrap();
        println!(
            "lifetime retained={retained} unload_success=true cache_after_unload={}",
            compiled_model_exists(&program.inner.model).unwrap()
        );
        drop(program);
        assert!(!path.exists());
        assert!(exists_after_load && exists_before_unload);
    }
    assert_eq!(
        super::compile_budget_used(),
        if phase == "compile" { 2 } else { 0 }
    );
}
