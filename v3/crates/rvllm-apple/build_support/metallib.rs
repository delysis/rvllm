use std::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CiBehavior {
    Build,
    SkipUnlessOptedIn,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MetallibSkipReason {
    NonMacosTarget,
    CiDefault,
    NoMetalSources,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetallibBuildPlan {
    Compile(MetallibCompilePlan),
    Skip(MetallibSkipReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetallibCompilePlan {
    pub metal_root: PathBuf,
    pub metal_sources: Vec<PathBuf>,
    pub air_outputs: Vec<PathBuf>,
    pub metallib_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetallibBuildEnv<'a> {
    pub manifest_dir: &'a Path,
    pub out_dir: &'a Path,
    pub target_os: &'a str,
    pub ci: bool,
    pub ci_behavior: CiBehavior,
    pub metal_sources: Vec<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetallibBuildPlanError {
    SourceOutsideMetalRoot {
        source: PathBuf,
        metal_root: PathBuf,
    },
    SourceNotMetal {
        source: PathBuf,
    },
}

impl fmt::Display for MetallibBuildPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceOutsideMetalRoot { source, metal_root } => write!(
                f,
                "Metal source {source:?} is outside metal source root {metal_root:?}"
            ),
            Self::SourceNotMetal { source } => {
                write!(
                    f,
                    "Metal source {source:?} does not use the .metal extension"
                )
            }
        }
    }
}

pub fn plan_metallib_build(
    env: MetallibBuildEnv<'_>,
) -> Result<MetallibBuildPlan, MetallibBuildPlanError> {
    if env.target_os != "macos" {
        return Ok(MetallibBuildPlan::Skip(MetallibSkipReason::NonMacosTarget));
    }

    if env.ci && env.ci_behavior == CiBehavior::SkipUnlessOptedIn {
        return Ok(MetallibBuildPlan::Skip(MetallibSkipReason::CiDefault));
    }

    if env.metal_sources.is_empty() {
        return Ok(MetallibBuildPlan::Skip(MetallibSkipReason::NoMetalSources));
    }

    let metal_root = env.manifest_dir.join("metal");
    let air_dir = env.out_dir.join("metal-air");
    let mut air_outputs = Vec::with_capacity(env.metal_sources.len());

    for source in &env.metal_sources {
        if source.extension().and_then(|ext| ext.to_str()) != Some("metal") {
            return Err(MetallibBuildPlanError::SourceNotMetal {
                source: source.clone(),
            });
        }

        let relative = source.strip_prefix(&metal_root).map_err(|_| {
            MetallibBuildPlanError::SourceOutsideMetalRoot {
                source: source.clone(),
                metal_root: metal_root.clone(),
            }
        })?;
        if relative
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(MetallibBuildPlanError::SourceOutsideMetalRoot {
                source: source.clone(),
                metal_root: metal_root.clone(),
            });
        }
        air_outputs.push(air_dir.join(relative).with_extension("air"));
    }

    Ok(MetallibBuildPlan::Compile(MetallibCompilePlan {
        metal_root,
        metal_sources: env.metal_sources,
        air_outputs,
        metallib_path: env.out_dir.join("rvllm_apple.metallib"),
    }))
}

pub fn discover_metal_sources(metal_root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut sources = Vec::new();
    if !metal_root.exists() {
        return Ok(sources);
    }
    collect_metal_sources(metal_root, &mut sources)?;
    sources.sort();
    Ok(sources)
}

fn collect_metal_sources(dir: &Path, sources: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_metal_sources(&path, sources)?;
        } else if file_type.is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("metal")
        {
            sources.push(path);
        }
    }
    Ok(())
}
