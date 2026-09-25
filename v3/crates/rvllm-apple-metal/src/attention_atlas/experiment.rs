//! File-bound preparation and compatibility with rvllm.experiment_job.v1.
//! This module never starts the queue, changes STOP, or manages device owners.
#![forbid(unsafe_code)]
use super::{
    plan, reference, sha256, source, CacheFormat, Candidate, Error, Output, Plan, Result, Shape,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const MAX_JSON: u64 = 4 * 1024 * 1024;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub path: PathBuf,
    pub sha256: String,
}
impl Pin {
    pub fn file(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        if !path.is_file() {
            return Err(Error::new("pin is not a regular file"));
        }
        Ok(Self {
            sha256: digest(&path)?,
            path,
        })
    }
    pub fn verify(&self) -> Result<()> {
        if !self.path.is_absolute()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|x| x.is_ascii_hexdigit())
            || digest(&self.path)? != self.sha256.to_ascii_lowercase()
        {
            return Err(Error::new(format!(
                "changed/missing pin: {}",
                self.path.display()
            )));
        }
        Ok(())
    }
}
pub fn digest(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        hasher.update(&b[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let mut bytes = vec![];
    File::open(path)?
        .take(MAX_JSON + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_JSON {
        return Err(Error::new("JSON size bound exceeded"));
    }
    // Typed structs use deny_unknown_fields and serde rejects duplicate fields.
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
pub fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)
}
pub fn fresh(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        return Err(Error::new("output directory must be absolute"));
    }
    fs::create_dir(path)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    ExistingDefault,
    ExistingPrefillSimd,
    AtlasVector,
    #[serde(rename = "existing_global_r8p64t64")]
    ExistingGlobalR8P64T64,
    #[serde(rename = "existing_global_r8p64t128")]
    ExistingGlobalR8P64T128,
    #[serde(rename = "existing_global_r8p128t64")]
    ExistingGlobalR8P128T64,
    #[serde(rename = "existing_global_r8p128t128")]
    ExistingGlobalR8P128T128,
    #[serde(rename = "existing_global_r16p64t64")]
    ExistingGlobalR16P64T64,
    #[serde(rename = "existing_global_r16p64t128")]
    ExistingGlobalR16P64T128,
    #[serde(rename = "existing_global_r16p128t64")]
    ExistingGlobalR16P128T64,
    #[serde(rename = "existing_global_r16p128t128")]
    ExistingGlobalR16P128T128,
    #[serde(rename = "current_matrix_r8k32p64t128")]
    CurrentMatrixR8K32P64T128,
}
impl Control {
    pub fn global_tile(self) -> Option<crate::attention_global_decode::DecodeTile> {
        let (rows, panel, threads) = match self {
            Self::ExistingGlobalR8P64T64 => (8, 64, 64),
            Self::ExistingGlobalR8P64T128 => (8, 64, 128),
            Self::ExistingGlobalR8P128T64 => (8, 128, 64),
            Self::ExistingGlobalR8P128T128 => (8, 128, 128),
            Self::ExistingGlobalR16P64T64 => (16, 64, 64),
            Self::ExistingGlobalR16P64T128 => (16, 64, 128),
            Self::ExistingGlobalR16P128T64 => (16, 128, 64),
            Self::ExistingGlobalR16P128T128 => (16, 128, 128),
            Self::CurrentMatrixR8K32P64T128 => {
                return Some(crate::attention_global_decode::DecodeTile {
                    rows: 8,
                    keys: 32,
                    panel: 64,
                    threads: 128,
                    per_tile_softmax: true,
                    simd_matrix: true,
                });
            }
            _ => return None,
        };
        Some(crate::attention_global_decode::DecodeTile {
            rows,
            keys: 8,
            panel,
            threads,
            per_tile_softmax: false,
            simd_matrix: false,
        })
    }
    pub fn global_name(self) -> Option<&'static str> {
        match self {
            Self::ExistingGlobalR8P64T64 => Some("research_global_d512_r8p64t64"),
            Self::ExistingGlobalR8P64T128 => Some("research_global_d512_r8p64t128"),
            Self::ExistingGlobalR8P128T64 => Some("research_global_d512_r8p128t64"),
            Self::ExistingGlobalR8P128T128 => Some("research_global_d512_r8p128t128"),
            Self::ExistingGlobalR16P64T64 => Some("research_global_d512_r16p64t64"),
            Self::ExistingGlobalR16P64T128 => Some("research_global_d512_r16p64t128"),
            Self::ExistingGlobalR16P128T64 => Some("research_global_d512_r16p128t64"),
            Self::ExistingGlobalR16P128T128 => Some("research_global_d512_r16p128t128"),
            Self::CurrentMatrixR8K32P64T128 => Some("research_global_d512_atlas_mma_r8k32p64t128"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Input {
    Synthetic {
        fixture: reference::Fixture,
    },
    /// Exact 12 input files in ABI buffer order 0..11. Slots 5..7 are reserved
    /// and must be empty files, not trusted output or alleged oracle evidence.
    Captured {
        buffers: Vec<Pin>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub schema: String,
    pub id: String,
    pub candidate: Candidate,
    pub shape: Shape,
    pub cache: CacheFormat,
    pub control: Control,
    pub input: Input,
    pub arena_limit_bytes: usize,
    pub max_oracle_fmas: u64,
    pub fp32_tolerance: reference::Tolerance,
    pub bf16_tolerance: reference::Tolerance,
    pub timing: Timing,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub warmups: u32,
    pub blocks: u32,
    pub max_drift_fraction: f64,
}
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 96
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
impl Spec {
    pub fn plan(&self, output: Output) -> Result<Plan> {
        if self.schema != "rvllm.attention_atlas.spec.v1"
            || !valid_id(&self.id)
            || !plan::catalog().contains(&self.candidate)
            || self.arena_limit_bytes as u64 > 16u64 * 1024 * 1024 * 1024
            || self.arena_limit_bytes == 0
            || self.max_oracle_fmas == 0
            || !(2..=16).contains(&self.timing.warmups)
            || !(2..=64).contains(&self.timing.blocks)
            || !self.timing.max_drift_fraction.is_finite()
            || self.timing.max_drift_fraction <= 0.0
            || self.timing.max_drift_fraction > 0.05
        {
            return Err(Error::new(
                "invalid experiment identity, catalog, budgets or predeclared timing plan",
            ));
        }
        if self.control.global_tile().is_some()
            && (self.shape.queries != 1 || self.shape.dim != 512 || self.shape.kv_heads != 1)
        {
            return Err(Error::new(
                "existing cooperative control admits D512/g16 one-token decode only",
            ));
        }
        self.fp32_tolerance.validate()?;
        self.bf16_tolerance.validate()?;
        let p = Plan::new(self.candidate, self.shape, self.cache, output)?;
        plan::Layout::new(p, self.arena_limit_bytes)?;
        Ok(p)
    }
    pub fn input_pins(&self) -> Vec<Pin> {
        match &self.input {
            Input::Synthetic { .. } => vec![],
            Input::Captured { buffers } => buffers.clone(),
        }
    }
    pub fn data(&self) -> Result<reference::Data> {
        let plan = self.plan(Output::F32)?;
        match &self.input {
            Input::Synthetic { fixture } => {
                reference::generate(plan, fixture, self.arena_limit_bytes)
            }
            Input::Captured { buffers } => {
                if buffers.len() != 12 {
                    return Err(Error::new("capture requires exactly 12 pins in ABI order"));
                }
                let lens = plan.buffer_lengths()?;
                let mut b = Vec::new();
                for (i, pin) in buffers.iter().enumerate() {
                    pin.verify()?;
                    let len = fs::metadata(&pin.path)?.len();
                    let expected = if (5..=7).contains(&i) {
                        0
                    } else {
                        lens[i] as u64
                    };
                    if len != expected {
                        return Err(Error::new("captured byte length differs from binding ABI"));
                    }
                    let bytes = fs::read(&pin.path)?;
                    if sha256(&bytes) != pin.sha256.to_ascii_lowercase() {
                        return Err(Error::new("captured bytes do not match pin"));
                    }
                    b.push(bytes);
                }
                let u16s = |i: usize| {
                    b[i].chunks_exact(2)
                        .map(|x| u16::from_le_bytes([x[0], x[1]]))
                        .collect::<Vec<_>>()
                };
                let i32s = |i: usize| {
                    b[i].chunks_exact(4)
                        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                let f32s = |i: usize| {
                    b[i].chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                let raw = self.cache != CacheFormat::SeparateBf16;
                let d = reference::Data {
                    q: u16s(0),
                    k: u16s(1),
                    v: if raw { vec![] } else { u16s(2) },
                    pages: i32s(3),
                    positions: i32s(4),
                    factor: if raw { f32s(8) } else { vec![] },
                    gamma: if raw { u16s(9) } else { vec![] },
                    cos: if raw { f32s(10) } else { vec![] },
                    sin: if raw { f32s(11) } else { vec![] },
                };
                d.validate(plan)?;
                for pin in buffers {
                    pin.verify()?;
                }
                Ok(d)
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub schema: String,
    pub spec: Pin,
    pub source: Pin,
    pub executable: Pin,
}
impl Prepared {
    pub fn load(dir: &Path) -> Result<(Self, Spec)> {
        let p: Self = read_json(&dir.join("prepared.json"))?;
        if p.schema != "rvllm.attention_atlas.prepared.v1" {
            return Err(Error::new("prepared schema mismatch"));
        }
        p.verify()?;
        let spec: Spec = read_json(&p.spec.path)?;
        spec.plan(Output::F32)?;
        if sha256(source(spec.candidate, spec.shape.dim)?.as_bytes()) != p.source.sha256 {
            return Err(Error::new(
                "frozen shader does not match this executable's embedded source",
            ));
        }
        for pin in spec.input_pins() {
            pin.verify()?;
        }
        Ok((p, spec))
    }
    pub fn verify(&self) -> Result<()> {
        self.spec.verify()?;
        self.source.verify()?;
        self.executable.verify()?;
        if Pin::file(&std::env::current_exe()?)?.sha256 != self.executable.sha256 {
            return Err(Error::new("runner executable identity changed"));
        }
        Ok(())
    }
}
pub fn prepare(spec_path: &Path, out: &Path) -> Result<()> {
    let spec: Spec = read_json(spec_path)?;
    spec.plan(Output::F32)?;
    for pin in spec.input_pins() {
        pin.verify()?;
    }
    fresh(out)?;
    // Freeze exactly the typed value validated above, not a second path read.
    write_json(&out.join("spec.json"), &spec)?;
    write_new(
        &out.join("candidate.metal"),
        source(spec.candidate, spec.shape.dim)?.as_bytes(),
    )?;
    write_json(&out.join("plan.json"), &spec.plan(Output::F32)?)?;
    write_json(
        &out.join("prepared.json"),
        &Prepared {
            schema: "rvllm.attention_atlas.prepared.v1".into(),
            spec: Pin::file(&out.join("spec.json"))?,
            source: Pin::file(&out.join("candidate.metal"))?,
            executable: Pin::file(&std::env::current_exe()?)?,
        },
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub schema: String,
    pub prepared: Pin,
    pub source: Pin,
    pub executable: Pin,
    pub library: Pin,
    pub compiler: Pin,
    pub linker: Pin,
    pub driver: Pin,
    pub sdk_path: PathBuf,
    pub sdk_version: String,
    pub flags: Vec<String>,
    pub success: bool,
}
impl Build {
    pub fn load(prepared_dir: &Path, build_dir: &Path) -> Result<Self> {
        let (p, _) = Prepared::load(prepared_dir)?;
        let b: Self = read_json(&build_dir.join("build.json"))?;
        if b.schema != "rvllm.attention_atlas.build.v1"
            || !b.success
            || b.prepared != Pin::file(&prepared_dir.join("prepared.json"))?
            || b.source != p.source
            || b.executable != p.executable
            || !b.sdk_path.is_absolute()
            || b.sdk_version.is_empty()
            || b.flags != ["-std=metal3.1", "-fno-fast-math"]
        {
            return Err(Error::new("mismatched/failed build receipt"));
        }
        b.library.verify()?;
        b.compiler.verify()?;
        b.linker.verify()?;
        b.driver.verify()?;
        Ok(b)
    }
}
/// Explicit local-owner compilation. This command is NEVER called by native
/// oracle/timing. Stdout/stderr and any partially created files are preserved.
pub fn compile(prepared_dir: &Path, out: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Err(Error::new(
            "Metal compilation requires the local macOS owner",
        ));
    }
    let (p, _) = Prepared::load(prepared_dir)?;
    fresh(out)?;
    let result = (|| {
        let find = |tool: &str| -> Result<PathBuf> {
            let r = Command::new("/usr/bin/xcrun")
                .args(["--sdk", "macosx", "--find", tool])
                .output()?;
            if !r.status.success() {
                return Err(Error::new(format!("xcrun cannot locate {tool}")));
            }
            Ok(PathBuf::from(
                String::from_utf8(r.stdout)
                    .map_err(|_| Error::new("tool path is not UTF8"))?
                    .trim(),
            ))
        };
        let compiler = Pin::file(&find("metal")?)?;
        let linker = Pin::file(&find("metallib")?)?;
        let driver = Pin::file(Path::new("/usr/bin/xcrun"))?;
        let sdk_info = |flag: &str| -> Result<String> {
            let r = Command::new(&driver.path)
                .args(["--sdk", "macosx", flag])
                .output()?;
            if !r.status.success() {
                return Err(Error::new("xcrun SDK query failed"));
            }
            let text =
                String::from_utf8(r.stdout).map_err(|_| Error::new("SDK query is not UTF8"))?;
            let text = text.trim().to_owned();
            if text.is_empty() {
                return Err(Error::new("empty SDK identity"));
            }
            Ok(text)
        };
        let sdk_path = PathBuf::from(sdk_info("--show-sdk-path")?);
        let sdk_version = sdk_info("--show-sdk-version")?;
        let air = out.join("candidate.air");
        let lib = out.join("candidate.metallib");
        let run = |tool: &Pin, args: Vec<std::ffi::OsString>, label: &str| -> Result<()> {
            tool.verify()?;
            driver.verify()?;
            if Pin::file(&find(label)?)? != *tool {
                return Err(Error::new("xcrun tool selection changed"));
            }
            let stdout = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join(format!("{label}.stdout")))?;
            let stderr = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join(format!("{label}.stderr")))?;
            write_json(
                &out.join(format!("{label}.command.json")),
                &serde_json::json!({
                    "driver":driver,"sdk":"macosx","tool":tool,"args":args.iter().map(|s|s.to_string_lossy().into_owned()).collect::<Vec<_>>()
                }),
            )?;
            let status = Command::new(&driver.path)
                .args(["--sdk", "macosx", label])
                .args(&args)
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr))
                .status()?;
            tool.verify()?;
            driver.verify()?;
            if Pin::file(&find(label)?)? != *tool {
                return Err(Error::new("xcrun tool selection changed"));
            }
            if !status.success() {
                return Err(Error::new(format!("{label} failed: {status}")));
            }
            Ok(())
        };
        run(
            &compiler,
            vec![
                "-std=metal3.1".into(),
                "-fno-fast-math".into(),
                "-c".into(),
                p.source.path.as_os_str().into(),
                "-o".into(),
                air.as_os_str().into(),
            ],
            "metal",
        )?;
        run(
            &linker,
            vec![air.as_os_str().into(), "-o".into(), lib.as_os_str().into()],
            "metallib",
        )?;
        p.verify()?;
        if PathBuf::from(sdk_info("--show-sdk-path")?) != sdk_path
            || sdk_info("--show-sdk-version")? != sdk_version
        {
            return Err(Error::new("SDK selection changed during compilation"));
        }
        write_json(
            &out.join("build.json"),
            &Build {
                schema: "rvllm.attention_atlas.build.v1".into(),
                prepared: Pin::file(&prepared_dir.join("prepared.json"))?,
                source: p.source,
                executable: p.executable,
                library: Pin::file(&lib)?,
                compiler,
                linker,
                driver,
                sdk_path,
                sdk_version,
                flags: vec!["-std=metal3.1".into(), "-fno-fast-math".into()],
                success: true,
            },
        )
    })();
    if let Err(e) = &result {
        write_json(
            &out.join("failure.json"),
            &serde_json::json!({"stage":"compile","error":e.to_string(),"passed":false}),
        )?;
    }
    result
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conditions {
    pub power_source: String,
    pub low_power_mode: Option<bool>,
    pub pmset_power_mode: Option<u64>,
    pub thermal_state: Option<u64>,
    pub minimum_free_bytes: u64,
    pub disk_path: PathBuf,
    pub quiet_process_names: Vec<String>,
    pub observe_process_names: Vec<String>,
    pub idle_llama_servers: Vec<IdleServer>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdleServer {
    pub pid: u32,
    pub port: u16,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueRequest {
    pub id: String,
    pub stage: String,
    pub prepared_dir: PathBuf,
    pub build_dir: Option<PathBuf>,
    pub oracle_receipt: Option<PathBuf>,
    pub cwd: PathBuf,
    pub conditions: Conditions,
    pub stable_seconds: u64,
    pub max_wait_seconds: u64,
    pub max_run_seconds: u64,
}
#[derive(Serialize)]
struct Invocation {
    executable: Pin,
    cwd: PathBuf,
    args: Vec<String>,
    env: BTreeMap<String, String>,
}
#[derive(Serialize)]
struct QueueJob {
    schema: &'static str,
    id: String,
    purpose: &'static str,
    command: Invocation,
    inputs: Vec<Pin>,
    after: Vec<String>,
    conditions: Conditions,
    stable_seconds: u64,
    max_wait_seconds: u64,
    max_run_seconds: u64,
}
pub fn queue(request: &QueueRequest, out: &Path) -> Result<()> {
    let r = request;
    let c = &r.conditions;
    if !valid_id(&r.id)
        || !matches!(r.stage.as_str(), "compile" | "oracle" | "bench")
        || !r.cwd.is_absolute()
        || !r.cwd.is_dir()
        || !r.prepared_dir.is_absolute()
        || !matches!(c.power_source.as_str(), "ac" | "battery")
        || c.pmset_power_mode.is_some_and(|mode| mode > 2)
        || c.thermal_state.is_some_and(|state| state > 3)
        || !c.disk_path.is_absolute()
        || !c.disk_path.is_dir()
        || c.minimum_free_bytes < 16 * 1024 * 1024 * 1024
        || r.stable_seconds > 600
        || r.max_wait_seconds > 86400
        || !(1..=3600).contains(&r.max_run_seconds)
        || c.idle_llama_servers.len() > 4
        || c.idle_llama_servers
            .iter()
            .any(|s| s.pid == 0 || s.port == 0)
        || c.quiet_process_names
            .iter()
            .chain(&c.observe_process_names)
            .any(|s| s.is_empty() || s.contains('/'))
    {
        return Err(Error::new(
            "invalid queue identity, explicit power stratum or bounds",
        ));
    }
    let (p, spec) = Prepared::load(&r.prepared_dir)?;
    let mut pins = vec![
        Pin::file(&r.prepared_dir.join("prepared.json"))?,
        p.spec.clone(),
        p.source.clone(),
    ];
    pins.extend(spec.input_pins());
    let mut args = vec![
        r.stage.clone(),
        r.prepared_dir.to_string_lossy().into_owned(),
    ];
    if r.stage != "compile" {
        let dir = r
            .build_dir
            .as_ref()
            .ok_or_else(|| Error::new("successful build directory is required"))?;
        let b = Build::load(&r.prepared_dir, dir)?;
        pins.extend([
            Pin::file(&dir.join("build.json"))?,
            b.library,
            b.compiler,
            b.linker,
            b.driver,
        ]);
        args.push(dir.to_string_lossy().into_owned());
    }
    if r.stage == "bench" {
        let path = r
            .oracle_receipt
            .as_ref()
            .ok_or_else(|| Error::new("successful exact-workload oracle receipt is required"))?;
        validate_oracle(path, &r.prepared_dir, r.build_dir.as_ref().unwrap())?;
        pins.push(Pin::file(path)?);
        args.push(path.to_string_lossy().into_owned());
    } else if r.oracle_receipt.is_some() {
        return Err(Error::new("oracle receipt only belongs to bench stage"));
    }
    if r.stage == "compile" && r.build_dir.is_some() {
        return Err(Error::new("compile cannot consume an old build"));
    }
    // The existing queue expands {output} in args only. No env indirection.
    args.push("{output}/atlas".into());
    let job = QueueJob {
        schema: "rvllm.experiment_job.v1",
        id: r.id.clone(),
        purpose: if r.stage == "bench" {
            "exploratory_timing"
        } else {
            "preparation"
        },
        command: Invocation {
            executable: p.executable,
            cwd: r.cwd.clone(),
            args,
            env: BTreeMap::new(),
        },
        inputs: pins,
        after: vec![],
        conditions: c.clone(),
        stable_seconds: r.stable_seconds,
        max_wait_seconds: r.max_wait_seconds,
        max_run_seconds: r.max_run_seconds,
    };
    write_json(out, &job)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleReceipt {
    pub schema: String,
    pub prepared: Pin,
    pub build: Pin,
    pub executable: Pin,
    pub detail: Pin,
    pub passed: bool,
    pub scope: String,
    pub full_route_qualified: bool,
    pub promotion_eligible: bool,
}
pub fn validate_oracle(path: &Path, prepared: &Path, build: &Path) -> Result<OracleReceipt> {
    let r: OracleReceipt = read_json(path)?;
    if r.schema != "rvllm.attention_atlas.oracle.v1"
        || !r.passed
        || r.scope != "operator_only"
        || r.full_route_qualified
        || r.promotion_eligible
        || r.prepared != Pin::file(&prepared.join("prepared.json"))?
        || r.build != Pin::file(&build.join("build.json"))?
        || r.executable.sha256 != Pin::file(&std::env::current_exe()?)?.sha256
    {
        return Err(Error::new("missing/mismatched operator oracle receipt"));
    }
    for pin in [&r.prepared, &r.build, &r.executable, &r.detail] {
        pin.verify()?;
    }
    Ok(r)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub index: u32,
    pub arm: String,
    pub host_ns: u64,
    pub gpu_ns: u64,
    pub encoded_dispatches: u32,
    pub completed_dispatches: u32,
}
#[derive(Clone, Debug, Serialize)]
pub struct Score {
    pub median_control_gpu_ns: f64,
    pub median_candidate_gpu_ns: f64,
    pub descriptive_ratio: f64,
    pub baseline_drift: f64,
    pub drift_passed: bool,
    pub full_route_qualified: bool,
    pub promotion_eligible: bool,
}
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 0 {
        (xs[n / 2 - 1] + xs[n / 2]) * 0.5
    } else {
        xs[n / 2]
    }
}
pub fn score(samples: &[Sample], timing: &Timing) -> Result<Score> {
    if !(2..=64).contains(&timing.blocks)
        || !timing.max_drift_fraction.is_finite()
        || timing.max_drift_fraction <= 0.0
        || timing.max_drift_fraction > 0.05
        || samples.len() != timing.blocks as usize * 4
    {
        return Err(Error::new("incomplete ABBA sequence; no drop/retry"));
    }
    for (i, s) in samples.iter().enumerate() {
        let expected = if i % 4 == 0 || i % 4 == 3 {
            "control"
        } else {
            "candidate"
        };
        if s.index != i as u32
            || s.arm != expected
            || s.gpu_ns == 0
            || s.host_ns == 0
            || s.encoded_dispatches == 0
            || s.encoded_dispatches != s.completed_dispatches
        {
            return Err(Error::new(
                "invalid timing order, duration or completion count",
            ));
        }
    }
    let controls = samples
        .iter()
        .filter(|s| s.arm == "control")
        .map(|s| s.gpu_ns as f64)
        .collect::<Vec<_>>();
    let candidates = samples
        .iter()
        .filter(|s| s.arm == "candidate")
        .map(|s| s.gpu_ns as f64)
        .collect::<Vec<_>>();
    let mid = controls.len() / 2;
    let early = median(controls[..mid].to_vec());
    let late = median(controls[mid..].to_vec());
    let drift = early.max(late) / early.min(late) - 1.0;
    let a = median(controls);
    let b = median(candidates);
    Ok(Score {
        median_control_gpu_ns: a,
        median_candidate_gpu_ns: b,
        descriptive_ratio: a / b,
        baseline_drift: drift,
        drift_passed: drift <= timing.max_drift_fraction,
        full_route_qualified: false,
        promotion_eligible: false,
    })
}

/// Materialize the bounded candidate/workload matrix as SPECS only. No device,
/// source compilation, worker or queue mutation is performed. Operators choose
/// which specimens to prepare; these synthetic cases do not qualify a model.
pub fn campaign(out: &Path) -> Result<()> {
    fresh(out)?;
    let mut index = Vec::new();
    for candidate in plan::catalog() {
        for (label, local, q, live, raw) in [
            ("global-decode", false, 1, 1025, false),
            ("local-decode", true, 1, 1025, false),
            ("global-verify", false, 7, 129, false),
            ("local-verify", true, 7, 1025, false),
            ("local-prefill", true, 33, 1025, false),
            ("global-prefill", false, 33, 257, false),
            ("raw-global-decode", false, 1, 129, true),
        ] {
            if q > 8 && candidate.splits > 1 {
                continue;
            }
            if candidate.strategy == plan::Strategy::Cooperative
                && candidate.rows < 8
                && (!local || q > 8)
            {
                continue;
            }
            let shape = Shape {
                queries: q,
                live_keys: live,
                kv_heads: if local { 8 } else { 1 },
                dim: if local { 256 } else { 512 },
                window: if local { 1024 } else { 0 },
                page_size: 32,
                max_blocks: live.div_ceil(32) + 2,
                physical_blocks: live.div_ceil(32) + 2,
            };
            let control = if !local && q == 1 {
                Control::ExistingGlobalR16P64T128
            } else {
                Control::ExistingDefault
            };
            let spec = Spec {
                schema: "rvllm.attention_atlas.spec.v1".into(),
                id: format!("{}-{label}", candidate.name()),
                candidate,
                shape,
                cache: if raw {
                    CacheFormat::BaseBf16FactorF32RopeV1
                } else {
                    CacheFormat::SeparateBf16
                },
                control,
                input: Input::Synthetic {
                    fixture: reference::Fixture {
                        seed: 0x61746c6173,
                        first_position: live - q,
                        pattern: reference::Pattern::Mixed,
                    },
                },
                arena_limit_bytes: 1024 * 1024 * 1024,
                max_oracle_fmas: 1_000_000_000,
                fp32_tolerance: reference::Tolerance {
                    max_abs: 0.0002,
                    rel_l2: 0.0002,
                },
                bf16_tolerance: reference::Tolerance {
                    max_abs: 0.02,
                    rel_l2: 0.01,
                },
                timing: Timing {
                    warmups: 2,
                    blocks: 7,
                    max_drift_fraction: 0.05,
                },
            };
            spec.plan(Output::F32)?;
            let file = format!("{}.json", spec.id);
            write_json(&out.join(&file), &spec)?;
            index.push(file);
        }
    }
    write_json(
        &out.join("index.json"),
        &serde_json::json!({"schema":"rvllm.attention_atlas.campaign.v1",
        "spec_files":index,"native_execution":false,"qualification":false,"promotion":false}),
    )
}
