//! PR27 fixture adaptation and retained-evidence revalidation. No Metal/FFI here.
#![forbid(unsafe_code)]
use super::{abs, admit_driver, digest, fixtures, hash, library, load, plan, text, Result};
use fixtures::{Buffer, Fixture, Oracle};
use half::bf16;
use rvllm_apple_metal::MetalResearchCandidate as C;
use serde_json::{json, Value};
use std::{fs, path::Path};

pub fn is_new(arm: &str) -> bool {
    arm.parse::<C>()
        .ok()
        .is_some_and(|c| plan::next::FAMILY.contains(&c))
}
fn guard(payload: Vec<u8>) -> Vec<u8> {
    let mut bytes = vec![0xa5; 32];
    bytes.extend(payload);
    bytes.extend([0xa5; 32]);
    bytes
}
fn words(f: &Fixture, id: usize) -> Result<Vec<i32>> {
    let b = f
        .buffers
        .iter()
        .find(|b| b.id == id)
        .ok_or("buffer missing")?;
    Ok(b.bytes[32..b.bytes.len() - 32]
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn bf_values(f: &Fixture, id: usize) -> Result<Vec<f64>> {
    let b = f
        .buffers
        .iter()
        .find(|b| b.id == id)
        .ok_or("buffer missing")?;
    Ok(b.bytes[32..b.bytes.len() - 32]
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f64())
        .collect())
}
fn set_bf(f: &mut Fixture, id: usize, data: &[f64]) -> Result {
    let b = f
        .buffers
        .iter_mut()
        .find(|b| b.id == id)
        .ok_or("buffer missing")?;
    if b.bytes.len() != data.len() * 2 + 64 {
        return Err("replacement fixture size mismatch".into());
    }
    b.bytes = guard(
        data.iter()
            .flat_map(|v| bf16::from_f64(*v).to_bits().to_le_bytes())
            .collect(),
    );
    Ok(())
}
fn constant(f: &Fixture, index: u64) -> Result<usize> {
    let a = f.passes[0]["constants"]
        .as_array()
        .ok_or("constants missing")?;
    let v = a
        .iter()
        .find(|v| v["index"].as_u64() == Some(index))
        .ok_or("constant missing")?;
    Ok(v["word"].as_u64().ok_or("word missing")? as usize)
}

fn layout() -> Fixture {
    let mut values = (1..=64).map(|v| v as f64).collect::<Vec<_>>();
    for r in 0..8 {
        for c in 0..8 {
            values.push(
                (0..8)
                    .map(|k| (r * 8 + k + 1) as f64 * ((k + 1) * (c + 2)) as f64 / 16.0)
                    .sum(),
            );
        }
    }
    Fixture {
        buffers: vec![Buffer {
            id: 0,
            bytes: guard(vec![0xff; 512]),
            read_only: false,
        }],
        passes: vec![json!({"kernel":"pr27_fragment_layout_probe","grid":[1,1,1],
            "threads":[32,1,1],"bindings":[{"index":0,"id":0,"offset":32}],
            "constants":[],"sourceSharedBytes":512})],
        output: 0,
        width: 4,
        rows: 1,
        columns: 128,
        oracle: Oracle::Projection {
            values,
            absolute_sums: vec![0.0; 128],
            period: 128,
            exact: true,
            norm: false,
            gelu: false,
        },
        label: "pr27-fragment-layout-and-product".into(),
        intermediate: None,
    }
}
fn range(f: &mut Fixture) -> Result {
    let k = constant(f, 5)?;
    let m = f.rows;
    let n = f.columns;
    let exponents = [-40, -20, 0, 20, 40];
    let a = (0..m * k)
        .map(|i| ((i / k) % 7 + 1) as f64 / 8.0 * 2f64.powi(exponents[i % k % 5]))
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| (((i / k) % 11) as i32 - 5) as f64 / 16.0 * 2f64.powi(-exponents[i % k % 5]))
        .collect::<Vec<_>>();
    set_bf(f, 0, &a)?;
    set_bf(f, 1, &b)?;
    let values = (0..m * n)
        .map(|i| {
            k as f64 * ((i / n) % 7 + 1) as f64 / 8.0 * ((i % n % 11) as i32 - 5) as f64 / 16.0
        })
        .collect();
    f.oracle = Oracle::Projection {
        values,
        absolute_sums: vec![0.0; m * n],
        period: n,
        exact: true,
        norm: false,
        gelu: false,
    };
    f.intermediate = None;
    Ok(())
}
/// Deliberately not a four-coefficient V basis: every channel participates in
/// the complete FP64 reference. All finite fixture operands are exactly dyadic.
fn dense_attention(f: &mut Fixture, mixed_nan: bool) -> Result {
    let d = constant(f, 12)?;
    let kv = constant(f, 11)?;
    let block = constant(f, 13)?;
    let window = constant(f, 16)?;
    let positions = words(f, 7)?;
    let context = words(f, 5)?[0] as usize;
    let tables = words(f, 4)?;
    let mut q = bf_values(f, 0)?;
    let mut v = bf_values(f, 2)?;
    let k = bf_values(f, 1)?;
    if !mixed_nan {
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17 + i / 13) % 29) as f64 / 128.0 - 14.0 / 128.0;
        }
    }
    for t in 0..context {
        let page = tables[t / block];
        if page < 0 {
            continue;
        }
        for h in 0..kv {
            for dd in 0..d {
                let index = ((page as usize * block + t % block) * kv + h) * d + dd;
                v[index] = ((t * 13 + h * 7 + dd * 11 + dd / 9) % 61) as f64 / 64.0 - 30.0 / 64.0;
            }
        }
    }
    // One causally masked but physically present bad value in the shared tile:
    // first three query rows must stay finite; the last three must be poisoned.
    if mixed_nan {
        let t = positions[0] as usize + 3;
        for h in 0..kv {
            v[((tables[t / block] as usize * block + t % block) * kv + h) * d + 7] = f64::NAN;
        }
    }
    set_bf(f, 0, &q)?;
    set_bf(f, 2, &v)?;
    let mut values = vec![0.0; f.rows * 16 * d];
    for r in 0..f.rows {
        let end = positions[r] as usize + 1;
        let start = if window == 0 {
            0
        } else {
            end.saturating_sub(window)
        };
        for h in 0..16 {
            let kh = h / (16 / kv);
            let mut dots = Vec::new();
            for t in start..end {
                let page = tables[t / block];
                if page < 0 {
                    continue;
                }
                let base = ((page as usize * block + t % block) * kv + kh) * d;
                let dot = (0..d)
                    .map(|dd| q[(r * 16 + h) * d + dd] * k[base + dd])
                    .sum::<f64>();
                dots.push((base, dot));
            }
            let mx = dots.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
            let sum = dots.iter().map(|v| (v.1 - mx).exp()).sum::<f64>();
            let bad = dots
                .iter()
                .any(|(base, _)| v[*base..*base + d].iter().any(|v| !v.is_finite()));
            for dd in 0..d {
                values[(r * 16 + h) * d + dd] = if bad {
                    f64::NAN
                } else if sum == 0.0 {
                    0.0
                } else {
                    dots.iter()
                        .map(|(base, dot)| (*dot - mx).exp() / sum * v[*base + dd])
                        .sum()
                };
            }
        }
    }
    f.oracle = Oracle::Attention { values };
    Ok(())
}

pub fn fixture(arm: &str, role: &str, m: usize, kind: &str) -> Result<Fixture> {
    if role == "layout" {
        if !is_new(arm) || kind != "layout" || m != 6 {
            return Err("invalid layout request".into());
        }
        return Ok(layout());
    }
    let custom = ["range", "dense-v", "masked-nan-v"].contains(&kind);
    if custom && (!is_new(arm) || m != 6) {
        return Err("custom fixture is a short PR27 screen".into());
    }
    let base_kind = if custom { "structured" } else { kind };
    let mut f = if role.starts_with("attention-") {
        fixtures::attention(arm, role, m, base_kind)?
    } else {
        fixtures::projection(arm, role, m, base_kind)?
    };
    if let Ok(c) = arm.parse::<C>() {
        if let Some([tm, tn]) = plan::next::projection_tile(c) {
            let [gemm, qkv] = plan::projection_kernels(c).ok_or("missing projection kernels")?;
            let raw = plan::postnorm_kernels(c).ok_or("missing postnorm")?[0];
            for p in &mut f.passes {
                if [gemm.name(), qkv.name(), raw.name()]
                    .contains(&p["kernel"].as_str().unwrap_or(""))
                {
                    let n = p["constants"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|v| v["index"] == 4)
                        .unwrap()["word"]
                        .as_u64()
                        .unwrap() as usize;
                    // Grid uses the real fixture extent even for wrong-M refusal.
                    p["grid"] = json!([m.div_ceil(tm), n.div_ceil(tn), 1]);
                }
            }
        }
        if plan::next::query_tile(c).is_some() {
            for p in &mut f.passes {
                if [
                    plan::attention_kernel(c, 256).unwrap().name(),
                    plan::attention_kernel(c, 512).unwrap().name(),
                ]
                .contains(&p["kernel"].as_str().unwrap_or(""))
                {
                    p["grid"] = json!([m.div_ceil(8), 16, 1]);
                }
            }
        }
    }
    match kind {
        "range" => {
            if role.ends_with("-norm") || role == "gate-gelu" || role.starts_with("attention-") {
                return Err("invalid range role".into());
            }
            range(&mut f)?;
        }
        "dense-v" => dense_attention(&mut f, false)?,
        "masked-nan-v" => dense_attention(&mut f, true)?,
        _ => {}
    }
    f.label = format!("{arm}/{role}/{m}/{kind}");
    Ok(f)
}

pub fn add_preflight_buffer(f: &mut Fixture, arm: &str) {
    if is_new(arm) {
        f.buffers.push(Buffer {
            id: 30,
            bytes: guard(vec![0xff; 512]),
            read_only: false,
        });
    }
}
pub fn prepare_job(job: &mut Value, arm: &str) {
    job["schema"] = json!("rvllm.prefill-next.driver.v1");
    if is_new(arm) {
        job["preflight"].as_array_mut().unwrap().push(
            json!({"kernel":"pr27_fragment_layout_probe",
            "grid":[1,1,1],"threads":[32,1,1],"bindings":[{"index":0,"id":30,"offset":32}],
            "constants":[],"sourceSharedBytes":512}),
        );
    }
}
fn verify_layout_buffer(out: &Path) -> Result {
    let f = layout();
    let Oracle::Projection { values, .. } = f.oracle else {
        return Err("layout oracle".into());
    };
    for suffix in ["preflight", "first", "last"] {
        let raw = fs::read(out.join(format!("buffer-30-{suffix}.bin")))?;
        if raw.len() != 576
            || raw[..32]
                .iter()
                .chain(raw[544..].iter())
                .any(|v| *v != 0xa5)
        {
            return Err("layout output extent/guard".into());
        }
        for (r, v) in raw[32..544].chunks_exact(4).zip(&values) {
            if f32::from_le_bytes(r.try_into().unwrap()).to_bits() != (*v as f32).to_bits() {
                return Err("retained layout/product output mismatch".into());
            }
        }
    }
    Ok(())
}
pub fn verify(f: &Fixture, out: &Path, id: &str, kind: &str) -> Result<Value> {
    if f.buffers.iter().any(|b| b.id == 30) {
        verify_layout_buffer(out)?;
    }
    if kind != "masked-nan-v" {
        return f.verify(out, id);
    }
    let raw = fs::read(out.join(format!("buffer-{}-first.bin", f.output)))?;
    if raw != fs::read(out.join(format!("buffer-{}-last.bin", f.output)))?
        || raw.len() != f.rows * f.columns * 2 + 64
    {
        return Err("mixed-NaN extent/determinism failure".into());
    }
    if raw[..32]
        .iter()
        .chain(raw[raw.len() - 32..].iter())
        .any(|b| *b != 0xa5)
    {
        return Err("mixed-NaN guard failure".into());
    }
    let identity = fs::read(out.join("buffer-31-first.bin"))?;
    if identity.len() != 96 {
        return Err("identity extent".into());
    }
    let got_id = identity[32..64]
        .chunks_exact(4)
        .map(|v| format!("{:08x}", u32::from_le_bytes(v.try_into().unwrap())))
        .collect::<String>();
    if got_id != id {
        return Err("source identity mismatch".into());
    }
    let Oracle::Attention { values } = &f.oracle else {
        return Err("mixed oracle required".into());
    };
    let mut poisoned = 0;
    for (i, (raw, expected)) in raw[32..raw.len() - 32]
        .chunks_exact(2)
        .zip(values)
        .enumerate()
    {
        let got = bf16::from_bits(u16::from_le_bytes(raw.try_into().unwrap())).to_f64();
        if expected.is_nan() {
            if !got.is_nan() {
                return Err(format!("expected row poison at {i}").into());
            }
            poisoned += 1;
        } else if !got.is_finite() || (got - expected).abs() > 5e-5 + expected.abs() / 256.0 {
            return Err(format!(
                "masked bad V contaminated finite output {i}: {got} vs {expected}"
            )
            .into());
        }
    }
    if poisoned != 3 * 16 * (f.columns / 16) {
        return Err("mixed poison fixture no longer has three bad rows".into());
    }
    Ok(
        json!({"complete_elements":true,"elements":f.rows*f.columns,"poisoned_elements":poisoned,"bitwise_repeat":true}),
    )
}

/// Rebuild the exact fixture and re-read the retained job, outputs and driver
/// report. A manually edited `status: pass` is not sufficient timing admission.
pub fn verify_retained(
    path: &Path,
    arm: &str,
    role: &str,
    m: u32,
    kind: &str,
    driver: &Path,
    libs: &Path,
    referee_sha256: &str,
) -> Result {
    let c = load(path)?;
    let out = path.parent().ok_or("cell parent missing")?;
    let (lib, id, libsha) = library(arm, libs)?;
    if c["schema"] != "rvllm.prefill-next.cell.v1"
        || c["base"] != super::BASE
        || c["status"] != "pass"
        || c["arm"] != arm
        || c["role"] != role
        || c["m"] != m
        || c["fixture"] != kind
        || c["source_body_sha256"] != id
        || c["library_sha256"] != libsha
        || c["driver_sha256"] != hash(driver)?
        || c["referee_sha256"] != referee_sha256
    {
        return Err("retained cell identity mismatch".into());
    }
    let mut f = fixture(arm, role, m as usize, kind)?;
    add_preflight_buffer(&mut f, arm);
    f.buffers.push(Buffer {
        id: 31,
        bytes: guard(vec![0xff; 32]),
        read_only: false,
    });
    let job_path = out.join("inputs/job.json");
    let job = load(&job_path)?;
    if job["schema"] != "rvllm.prefill-next.driver.v1"
        || job["preflight"].as_array().map(Vec::len) != Some(if is_new(arm) { 2 } else { 1 })
    {
        return Err("retained preflight omitted".into());
    }
    if c["job_sha256"] != hash(&job_path)?
        || job["passes"] != json!(f.passes)
        || job["library"] != json!(lib)
        || job["sourceBodySha256"] != id
        || job["warmup"] != 20
        || job["repeats"] != 9
    {
        return Err("retained job changed work".into());
    }
    for b in &f.buffers {
        if hash(&out.join(format!("inputs/input-{}.bin", b.id)))? != digest(&b.bytes) {
            return Err("retained input changed".into());
        }
    }
    let driver_path = out.join("driver.json");
    if c["driver_receipt_sha256"] != hash(&driver_path)? {
        return Err("retained driver receipt changed".into());
    }
    let raw = load(&driver_path)?;
    admit_driver(&raw, &job, &f)?;
    if !json_roundtrip_equal(&c["samples"], &raw["samples"]) {
        return Err("timing arrays changed".into());
    }
    let expected = verify(&f, out, &id, kind)?;
    let intermediate = if f.intermediate.is_some() {
        json!(hash(&out.join("buffer-2-first.bin"))?)
    } else {
        Value::Null
    };
    if c["intermediate_output_sha256"] != intermediate {
        return Err("intermediate output changed".into());
    }
    if !json_roundtrip_equal(&c["oracle"], &expected) {
        return Err("retained oracle summary changed".into());
    }
    if hash(&abs(text(&c, "output_file")?)?)? != text(&c, "output_sha256")? {
        return Err("retained raw output changed".into());
    }
    Ok(())
}

/// JSON decimal parsing can round a recorded finite f64 by one ULP. Raw
/// outputs, numerical oracles, driver identities and hashes are rechecked
/// separately; this only admits serialization round-trip noise.
fn json_roundtrip_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            let (Some(a), Some(b)) = (a.as_f64(), b.as_f64()) else {
                return false;
            };
            a.is_finite() && b.is_finite() && a.to_bits().abs_diff(b.to_bits()) <= 1
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| json_roundtrip_equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| json_roundtrip_equal(a, b)))
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn oracle_summary_allows_only_one_serialization_ulp() {
        let a = json!({"complete_elements": true, "elements": 65_280,
            "max_abs": 0.003890753692882276_f64,
            "relative_l2": 0.0015692040879649325_f64,
            "poison_expected": false, "bitwise_repeat": true});
        let mut b = a.clone();
        b["relative_l2"] = json!(0.0015692040879649323_f64);
        assert!(json_roundtrip_equal(&a, &b));
        b["relative_l2"] = json!(0.001569204087964931_f64);
        assert!(!json_roundtrip_equal(&a, &b));
        b = a.clone();
        b["elements"] = json!(65_279);
        assert!(!json_roundtrip_equal(&a, &b));
        b = a.clone();
        b["max_abs"] = json!(f64::NAN);
        assert!(!json_roundtrip_equal(&a, &b));
        let one = json!([{"gpu_ns": 0.0015692040879649325_f64,
            "dispatches": ["required_kernel"], "completed": true}]);
        let two = json!([{"gpu_ns": 0.0015692040879649323_f64,
            "dispatches": ["different_kernel"], "completed": true}]);
        assert!(!json_roundtrip_equal(&one, &two));
    }

    #[test]
    #[ignore = "read-only audit of an explicitly supplied retained screen"]
    fn retained_oracle_json_roundtrip_audit() {
        let screen_path = PathBuf::from(
            std::env::var_os("RVLLM_PREFILL27_ORACLE_SCREEN")
                .expect("RVLLM_PREFILL27_ORACLE_SCREEN is required"),
        );
        let screen = load(&screen_path).expect("screen JSON");
        for entry in screen["completed"].as_array().expect("completed cells") {
            let cell_path = PathBuf::from(entry["path"].as_str().expect("cell path"));
            let cell = load(&cell_path).expect("cell JSON");
            let driver = load(&cell_path.parent().expect("cell parent").join("driver.json"))
                .expect("driver JSON");
            assert!(
                json_roundtrip_equal(&cell["samples"], &driver["samples"]),
                "{}: recorded timing array changed beyond JSON round-trip",
                cell_path.display()
            );
            let arm = cell["arm"].as_str().expect("arm");
            let role = cell["role"].as_str().expect("role");
            let m = cell["m"].as_u64().expect("M") as usize;
            let kind = cell["fixture"].as_str().expect("fixture");
            let source = cell["source_body_sha256"].as_str().expect("source");
            let mut fixture = fixture(arm, role, m, kind).expect("fixture regeneration");
            add_preflight_buffer(&mut fixture, arm);
            let actual = verify(
                &fixture,
                cell_path.parent().expect("cell parent"),
                source,
                kind,
            )
            .expect("retained output verification");
            if !json_roundtrip_equal(&cell["oracle"], &actual) {
                panic!(
                    "{}: stored oracle {} differs from recomputed {}",
                    cell_path.display(),
                    cell["oracle"],
                    actual
                );
            }
        }
    }
    #[test]
    fn layout_reference_has_two_complete_nontrivial_matrices() {
        let f = layout();
        let Oracle::Projection { values, .. } = f.oracle else {
            panic!()
        };
        assert_eq!(values.len(), 128);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[63], 64.0);
        assert_eq!(values[64], 25.5);
        assert!(values[127] > values[64]);
    }
    #[test]
    fn new_ownership_is_not_a_name_prefix_guess() {
        assert!(is_new("metal-prefill-wide64"));
        assert!(!is_new("metal-prefill-pipeline32x64"));
        assert!(!is_new("metal-prefill-auto"));
    }
    #[test]
    fn fixture_grid_and_fp32_raw_boundary_follow_new_planner() {
        let f = fixture("metal-prefill-wide128", "qkv-local", 6, "structured").unwrap();
        assert_eq!(f.passes[0]["grid"], json!([1, 64, 1]));
        assert_eq!(f.width, 4);
    }
    #[test]
    fn masked_bad_value_oracle_keeps_earlier_queries_finite() {
        let f = fixture(
            "metal-prefill-mma8k32",
            "attention-local",
            6,
            "masked-nan-v",
        )
        .unwrap();
        let Oracle::Attention { values } = f.oracle else {
            panic!()
        };
        assert!(values[..3 * 16 * 256].iter().all(|v| v.is_finite()));
        assert!(values[3 * 16 * 256..].iter().all(|v| v.is_nan()));
    }
}
