//! Offline, safe-Rust prefill source exporter, gated serial referee and timer.
//! This executable is never imported by the inference hot path.
#![forbid(unsafe_code)]
#[path = "prefill_round/fixtures.rs"]
mod fixtures;
#[path = "prefill_round/queue.rs"]
mod queue;
use rvllm_apple_metal::{
    kernels, prefill_round as plan, MetalFloatType, MetalKernelOptions, MetalResearchCandidate,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
type Error = Box<dyn std::error::Error>;
type Result<T = ()> = std::result::Result<T, Error>;
const BASE: &str = "fb5f169cae18eb492852e6bf668130e80c347cfb";
const ARMS: [&str; 8] = [
    "normal",
    "mma32",
    "metal-mma32-load4",
    "simd-attention",
    "metal-prefill-load4-control",
    "metal-prefill-pipeline32x64",
    "metal-prefill-q4k16",
    "metal-prefill-pipeline32x64-q4k16",
];
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn load(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn save(path: &Path, value: &Value) -> Result {
    let tmp = path.with_extension("partial");
    let mut f = fs::File::create_new(&tmp)?;
    f.write_all(&serde_json::to_vec_pretty(value)?)?;
    f.sync_all()?;
    fs::rename(tmp, path)?;
    Ok(())
}
fn abs(s: &str) -> Result<PathBuf> {
    let p = PathBuf::from(s);
    if !p.is_absolute() {
        return Err("artifact/executable paths must be absolute".into());
    }
    Ok(p)
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}").into())
}
fn source(arm: &str) -> Result<(String, String)> {
    if !ARMS.contains(&arm) {
        return Err("unknown arm".into());
    }
    let selected = if ["normal", "mma32", "simd-attention"].contains(&arm) {
        MetalResearchCandidate::Off
    } else {
        arm.parse()?
    };
    let options = MetalKernelOptions {
        research: selected,
        ..MetalKernelOptions::default()
    };
    let body = kernels::kernel_source_with_options(MetalFloatType::Bf16, options).into_owned();
    let id = digest(body.as_bytes());
    let words = (0..8)
        .map(|i| format!("0x{}u", &id[i * 8..i * 8 + 8]))
        .collect::<Vec<_>>()
        .join(",");
    let emitted=format!("{body}\n// Source-body SHA256, independently read back before every operator arm.\nkernel void pr26_export_identity(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{\n    const uint identity[8] = {{{words}}};\n    if (i < 8u) out[i] = identity[i];\n}}\n");
    Ok((emitted, id))
}
fn export_all(out: &Path) -> Result {
    fs::create_dir(out)?;
    for arm in ARMS {
        let (msl, id) = source(arm)?;
        fs::write(out.join(format!("{arm}.metal")), msl.as_bytes())?;
        save(
            &out.join(format!("{arm}.source.json")),
            &json!({"schema":"rvllm.prefill-round.source.v1",
            "base":BASE,"arm":arm,"source_body_sha256":id,"msl_sha256":digest(msl.as_bytes()),
            "dtype":"bfloat16","production_promotion":false}),
        )?;
    }
    save(
        &out.join("catalog.json"),
        &rvllm_apple_metal::research_catalog::catalog_json(),
    )
}
struct Lock(PathBuf);
impl Lock {
    fn acquire(path: &Path) -> Result<Self> {
        let mut f = fs::File::create_new(path).map_err(|e| {
            format!(
                "exclusive experiment lock {}: {e}; never auto-delete another owner's lock",
                path.display()
            )
        })?;
        writeln!(f, "{}", std::process::id())?;
        f.sync_all()?;
        Ok(Self(path.to_path_buf()))
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
struct OwnedChild(Child);
// Match the existing accelerator queue: never kill a Metal owner and then
// release its cooperative lock while device work may still be running.
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.wait();
    }
}
fn child(driver: &Path, job: &Path, out: &Path) -> Result {
    let stdout = fs::File::create_new(out.join("driver.stdout"))?;
    let stderr = fs::File::create_new(out.join("driver.stderr"))?;
    let mut cmd = Command::new(driver);
    cmd.arg(job)
        .arg(out)
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("LANG", "C")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .stdin(Stdio::null());
    // No inherited Metal, RVLLM, DYLD or timing selectors. OS defaults otherwise.
    let mut process = OwnedChild(cmd.spawn()?);
    let started = Instant::now();
    let mut overdue = false;
    loop {
        if let Some(status) = process.0.try_wait()? {
            if overdue {
                return Err(
                    "operator exceeded 600 seconds; retained child until safe exit, no retry"
                        .into(),
                );
            }
            if status.success() {
                return Ok(());
            }
            return Err(format!("operator process exited {status}").into());
        }
        if !overdue && started.elapsed() > Duration::from_secs(600) {
            overdue = true;
            fs::write(out.join("OVERDUE"),"Deadline exceeded; awaiting Metal child exit, not killing or releasing ownership.\n")?;
        }
        thread::sleep(Duration::from_millis(25));
    }
}
fn library(arm: &str, dir: &Path) -> Result<(PathBuf, String, String)> {
    let manifest = load(&dir.join(format!("{arm}.source.json")))?;
    let (expected, id) = source(arm)?;
    if manifest["base"] != BASE
        || manifest["arm"] != arm
        || manifest["source_body_sha256"] != id
        || manifest["msl_sha256"] != digest(expected.as_bytes())
        || hash(&dir.join(format!("{arm}.metal")))? != digest(expected.as_bytes())
    {
        return Err("source manifest is not this executable's exact source".into());
    }
    let path = fs::canonicalize(dir.join(format!("{arm}.metallib")))?;
    let sha = hash(&path)?;
    Ok((path, id, sha))
}
fn admit_driver(v: &Value, job: &Value, fixture: &fixtures::Fixture) -> Result {
    if v["schema"] != "rvllm.prefill-round.driver-receipt.v1" || v["production_promotion"] != false
    {
        return Err("invalid driver schema/claim".into());
    }
    if v["preflight"]["completed"] != true || v["source_identity_verified_before_operator"] != true
    {
        return Err("source identity was not verified before operator dispatch".into());
    }
    let expected = job["passes"]
        .as_array()
        .ok_or("passes missing")?
        .iter()
        .map(|p| p["kernel"].clone())
        .collect::<Vec<_>>();
    let samples = v["samples"].as_array().ok_or("samples missing")?;
    if samples.len() != 9 {
        return Err("wrong sample count".into());
    }
    for row in samples.iter().chain(std::iter::once(&v["correctness"])) {
        if row["completed"] != true
            || row["dispatches"] != json!(expected)
            || row["library_load_calls"] != 0
            || row["source_compile_calls"] != 0
            || row["pipeline_creation_calls"] != 0
        {
            return Err(
                "missing dispatch, incomplete GPU work, or compile during timed interval".into(),
            );
        }
    }
    for key in ["first_readbacks", "last_readbacks"] {
        let rows = v[key].as_array().ok_or("readbacks missing")?;
        if rows.len() != fixture.buffers.len() {
            return Err("incomplete read-only/guard checks".into());
        }
        for b in &fixture.buffers {
            let hits = rows
                .iter()
                .filter(|r| r["id"].as_u64() == Some(b.id as u64))
                .collect::<Vec<_>>();
            if hits.len() != 1 {
                return Err("duplicate/missing readback".into());
            }
            let row = hits[0];
            if row["guards_ok"] != true
                || row["bytes"].as_u64() != Some(b.bytes.len() as u64)
                || row["read_only"] != b.read_only
            {
                return Err("invalid guard/extent metadata".into());
            }
            if b.read_only && (row["unchanged"] != true || row["sha256"] != digest(&b.bytes)) {
                return Err("read-only Metal input mutation".into());
            }
        }
    }
    Ok(())
}
fn cell(
    arm: &str,
    role: &str,
    m: usize,
    kind: &str,
    driver: &Path,
    libdir: &Path,
    out: &Path,
) -> Result<Value> {
    fs::create_dir(out)?;
    let result = (|| -> Result<Value> {
        let (lib, id, libsha) = library(arm, libdir)?;
        let driversha = hash(driver)?;
        let mut f = if role.starts_with("attention-") {
            fixtures::attention(arm, role, m, kind)?
        } else {
            fixtures::projection(arm, role, m, kind)?
        };
        let inputs = out.join("inputs");
        fs::create_dir(&inputs)?;
        let job = f.write(&inputs, &lib, &id)?;
        child(driver, &inputs.join("job.json"), out)?;
        let raw = load(&out.join("driver.json"))?;
        admit_driver(&raw, &job, &f)?;
        let oracle = f.verify(out, &id)?;
        let mut fixture_hash = Sha256::new();
        for b in f.buffers.iter().filter(|b| b.read_only) {
            fixture_hash.update((b.id as u64).to_le_bytes());
            fixture_hash.update((b.bytes.len() as u64).to_le_bytes());
            fixture_hash.update(&b.bytes);
        }
        let fixture_hash = format!("{:x}", fixture_hash.finalize());
        let output_hash = hash(&out.join(format!("buffer-{}-first.bin", f.output)))?;
        // Re-open every sealed input after the child. Never trust only a filename.
        for b in &f.buffers {
            if hash(&inputs.join(format!("input-{}.bin", b.id)))? != digest(&b.bytes) {
                return Err("host fixture changed during trial".into());
            }
        }
        if hash(&lib)? != libsha || hash(driver)? != driversha {
            return Err("driver/library changed during trial".into());
        }
        let times = raw["samples"].as_array().unwrap();
        let timer_valid = times.iter().all(|v| {
            v["gpu_timer_valid"] == true
                && v["gpu_ns"]
                    .as_f64()
                    .is_some_and(|t| t.is_finite() && t > 0.0)
        });
        Ok(
            json!({"schema":"rvllm.prefill-round.cell.v1","status":"pass","base":BASE,
            "arm":arm,"role":role,"m":m,"fixture":kind,"library_sha256":libsha,
            "driver_sha256":driversha,"referee_sha256":hash(&std::env::current_exe()?)?,
            "source_body_sha256":id,"fixture_read_only_sha256":fixture_hash,"output_sha256":output_hash,"driver_receipt_sha256":hash(&out.join("driver.json"))?,
            "job_sha256":hash(&inputs.join("job.json"))?,"oracle":oracle,
            "output_buffer_id":f.output,"output_file":out.join(format!("buffer-{}-first.bin",f.output)),
            "timer_valid":timer_valid,"samples":raw["samples"],"device":raw["device"],
            "conditions":raw["conditions"],"production_promotion":false}),
        )
    })();
    match result {
        Ok(report) => {
            save(&out.join("cell.json"), &report)?;
            Ok(report)
        }
        Err(e) => {
            save(
                &out.join("cell.json"),
                &json!({"schema":"rvllm.prefill-round.cell.v1","status":"failed",
            "base":BASE,"arm":arm,"role":role,"m":m,"fixture":kind,"error":e.to_string(),"production_promotion":false}),
            )?;
            Err(e)
        }
    }
}
fn kinds(arm: &str, role: &str, m: u32) -> Vec<&'static str> {
    let new = arm.starts_with("metal-prefill-");
    let attn = role.starts_with("attention-");
    let mut kinds = vec!["structured"];
    if m <= 17 {
        kinds.push("periodic");
    }
    if attn {
        kinds.push("newest");
        if arm != "normal" {
            kinds.push("holes");
            kinds.push("all-holes");
        }
        if new && m == 6 {
            kinds.extend(["bad-page", "bad-metadata", "wrong-threads", "wrong-m"]);
        }
    } else if new && m == 6 && !role.ends_with("-norm") && role != "gate-gelu" {
        kinds.extend(["wrong-threads", "wrong-m"]);
    }
    kinds
}
fn screen(arm: &str, role: &str, limit: u32, driver: &Path, libdir: &Path, out: &Path) -> Result {
    if !plan::RUNGS.contains(&limit) {
        return Err("limit must be a published correctness rung".into());
    }
    fs::create_dir(out)?;
    let mut completed = Vec::new();
    let mut report = json!({"schema":"rvllm.prefill-round.screen.v1","base":BASE,"arm":arm,"role":role,
        "limit":limit,"status":"running","production_promotion":false});
    for m in plan::RUNGS.into_iter().filter(|m| *m <= limit) {
        for kind in kinds(arm, role, m) {
            let dir = out.join(format!("m{m}-{kind}"));
            match cell(arm,role,m as usize,kind,driver,libdir,&dir) {
                Ok(_)=>completed.push(json!({"m":m,"fixture":kind,"path":dir.join("cell.json"),"sha256":hash(&dir.join("cell.json"))?})),
                Err(e)=>{report["status"]=json!("failed");report["error"]=json!(e.to_string());report["completed"]=json!(completed);save(&out.join("screen.json"),&report)?;return Err(e);}
            }
            report["completed"] = json!(completed);
            save(&out.join("screen.json"), &report)?;
        }
    }
    report["status"] = json!("pass");
    save(&out.join("screen.json"), &report)
}
fn admitted_screen(
    path: &Path,
    arm: &str,
    role: &str,
    m: u32,
    driver: &Path,
    libdir: &Path,
) -> Result {
    let report = load(path)?;
    if report["schema"] != "rvllm.prefill-round.screen.v1"
        || report["status"] != "pass"
        || report["base"] != BASE
        || report["arm"] != arm
        || report["role"] != role
        || report["limit"].as_u64().unwrap_or(0) < m as u64
    {
        return Err("timing requires this arm/role's completed short-to-long screen".into());
    }
    let completed = report["completed"].as_array().ok_or("no completed cells")?;
    let (_, id, libsha) = library(arm, libdir)?;
    let driversha = hash(driver)?;
    for rung in plan::RUNGS.into_iter().filter(|n| *n <= m) {
        for kind in kinds(arm, role, rung) {
            let hits = completed
                .iter()
                .filter(|v| v["m"] == rung && v["fixture"] == kind)
                .collect::<Vec<_>>();
            if hits.len() != 1 {
                return Err("screen skipped or duplicated a required correctness fixture".into());
            }
            let path = abs(text(hits[0], "path")?)?;
            if hash(&path)? != text(hits[0], "sha256")? {
                return Err("screen cell receipt changed".into());
            }
            let cell = load(&path)?;
            if cell["status"] != "pass"
                || cell["arm"] != arm
                || cell["role"] != role
                || cell["m"] != rung
                || cell["fixture"] != kind
                || cell["source_body_sha256"] != id
                || cell["library_sha256"] != libsha
                || cell["driver_sha256"] != driversha
                || cell["referee_sha256"] != hash(&std::env::current_exe()?)?
            {
                return Err("screen identity changed".into());
            }
        }
    }
    Ok(())
}
fn median(xs: &[f64]) -> Result<f64> {
    if xs.is_empty() || xs.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("nonpositive/nonfinite timer".into());
    }
    let mut values = xs.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    Ok(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    })
}
fn compare(
    a: &str,
    b: &str,
    role: &str,
    m: u32,
    sa: &Path,
    sb: &Path,
    driver: &Path,
    libs: &Path,
    out: &Path,
) -> Result {
    if ![256, 512, 1024, 2048].contains(&m) || a == b {
        return Err("comparison needs two distinct arms and a target M".into());
    }
    admitted_screen(sa, a, role, m, driver, libs)?;
    admitted_screen(sb, b, role, m, driver, libs)?;
    fs::create_dir(out)?;
    let mut runs: Vec<Value> = Vec::new();
    // Eight independent driver processes. Both orders retained, never pick a favorable pair.
    for (i, arm) in [a, b, b, a, b, a, a, b].into_iter().enumerate() {
        let dir = out.join(format!("{i:02}-{arm}"));
        let receipt = cell(arm, role, m as usize, "structured", driver, libs, &dir)?;
        if receipt["timer_valid"] != true {
            return Err(
                "invalid GPU timer: correctness retained, timing comparison rejected".into(),
            );
        }
        if let Some(first) = runs.first() {
            if first["fixture_read_only_sha256"] != receipt["fixture_read_only_sha256"] {
                return Err("timing arms used different input bytes".into());
            }
            let schedule_pair = [a, b].iter().all(|name| {
                name.parse::<MetalResearchCandidate>()
                    .ok()
                    .is_some_and(|c| plan::projection_kernels(c).is_some())
            });
            if schedule_pair && first["output_sha256"] != receipt["output_sha256"] {
                return Err("matched prefill schedules were not bitwise equal".into());
            }
        }
        let samples = receipt["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["gpu_ns"].as_f64().ok_or("timer missing"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        runs.push(json!({"arm":arm,"fixture_read_only_sha256":receipt["fixture_read_only_sha256"],"output_sha256":receipt["output_sha256"],"median_gpu_ns":median(&samples)?,"cell":dir.join("cell.json"),"sha256":hash(&dir.join("cell.json"))?}));
        save(
            &out.join("comparison.json"),
            &json!({"status":"running","runs":runs,"production_promotion":false}),
        )?;
    }
    let med = |arm| -> Result<f64> {
        median(
            &runs
                .iter()
                .filter(|r| r["arm"] == arm)
                .map(|r| r["median_gpu_ns"].as_f64().unwrap())
                .collect::<Vec<_>>(),
        )
    };
    save(
        &out.join("comparison.json"),
        &json!({"schema":"rvllm.prefill-round.comparison.v1","status":"pass",
        "base":BASE,"a":a,"b":b,"role":role,"m":m,"order":"ABBA-BAAB","runs":runs,
        "descriptive_a_over_b":med(a)?/med(b)?,"conditions_qualified":false,
        "claim":"independently timed synthetic operator only; not an MLX/model or production speedup",
        "production_promotion":false}),
    )
}
fn usage() -> Error {
    "usage:\n rvllm-prefill-round source-all ABS_OUTPUT_DIR\n rvllm-prefill-round screen ARM ROLE LIMIT ABS_DRIVER ABS_LIBRARY_DIR ABS_OUTPUT_DIR ABS_LOCK_FILE\n rvllm-prefill-round compare A B ROLE M ABS_SCREEN_A_JSON ABS_SCREEN_B_JSON ABS_DRIVER ABS_LIBRARY_DIR ABS_OUTPUT_DIR ABS_LOCK_FILE\n Additional commands: screen-jobs ABS_CONFIG_JSON; timing-jobs ABS_CONFIG_JSON;\n sample ARM ROLE M ABS_SCREEN_JSON ABS_DRIVER ABS_LIBS ABS_OUT ABS_LOCK;\n adjudicate ABS_PLAN_JSON ABS_OUT_JSON\nRoles: qkv-local qkv-global gate-up gate-gelu o-local o-global down o-local-norm o-global-norm down-norm attention-local attention-global".into()
}
fn run() -> Result {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    match a.first().map(String::as_str) {
        Some("source-all") if a.len() == 2 => export_all(&abs(&a[1])?),
        Some("screen-jobs") if a.len() == 2 => queue::screen_jobs(&abs(&a[1])?),
        Some("timing-jobs") if a.len() == 2 => queue::timing_jobs(&abs(&a[1])?),
        Some("adjudicate") if a.len() == 3 => queue::adjudicate(&abs(&a[1])?, &abs(&a[2])?),
        Some("sample") if a.len() == 9 => {
            let _lock = Lock::acquire(&abs(&a[8])?)?;
            let m = a[3].parse()?;
            if ![256, 512, 1024, 2048].contains(&m) {
                return Err("sample M must be a target prefill rung".into());
            }
            let driver = abs(&a[5])?;
            let libs = abs(&a[6])?;
            admitted_screen(&abs(&a[4])?, &a[1], &a[2], m, &driver, &libs)?;
            let receipt = cell(
                &a[1],
                &a[2],
                m as usize,
                "structured",
                &driver,
                &libs,
                &abs(&a[7])?,
            )?;
            if receipt["timer_valid"] != true {
                return Err("invalid GPU timer; raw correctness receipt retained".into());
            }
            Ok(())
        }
        Some("screen") if a.len() == 8 => {
            let _lock = Lock::acquire(&abs(&a[7])?)?;
            screen(
                &a[1],
                &a[2],
                a[3].parse()?,
                &abs(&a[4])?,
                &abs(&a[5])?,
                &abs(&a[6])?,
            )
        }
        Some("compare") if a.len() == 11 => {
            let _lock = Lock::acquire(&abs(&a[10])?)?;
            compare(
                &a[1],
                &a[2],
                &a[3],
                a[4].parse()?,
                &abs(&a[5])?,
                &abs(&a[6])?,
                &abs(&a[7])?,
                &abs(&a[8])?,
                &abs(&a[9])?,
            )
        }
        _ => Err(usage()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("prefill referee: {e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timers_fail_closed() {
        assert!(median(&[]).is_err());
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(median(&[v]).is_err());
        }
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
    }
    #[test]
    fn short_before_long_and_negative_controls_are_explicit() {
        assert_eq!(&plan::RUNGS[..3], &[6, 17, 64]);
        assert!(kinds(ARMS[4], "qkv-local", 6).contains(&"wrong-m"));
        assert!(!kinds("normal", "attention-local", 6).contains(&"bad-page"));
        assert!(!kinds("normal", "attention-local", 6).contains(&"holes"));
    }
    #[test]
    fn exports_embed_exact_body_identity() {
        for arm in ARMS {
            let (msl, id) = source(arm).unwrap();
            assert_eq!(id.len(), 64);
            assert_eq!(msl.matches("kernel void pr26_export_identity(").count(), 1);
            assert!(msl.contains(&id[..8]));
        }
    }
    #[test]
    fn output_paths_are_explicit() {
        assert!(abs("relative").is_err());
        assert!(abs("/tmp/research").is_ok());
    }
}
