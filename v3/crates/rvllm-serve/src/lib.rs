// rvllm-serve — scaffold only.
//   pub mod http;    // tokio HTTP loop
//   pub mod openai;  // /v1/completions and /v1/chat/completions handlers

use std::path::PathBuf;

use rvllm_core::{AppleCtx, AppleError, ConfigError, Result, RvllmError};

#[cfg(feature = "apple")]
use rvllm_runtime::AppleBackendMode;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServeConfig {
    pub model: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub apple: Option<AppleServeConfig>,
}

#[cfg(feature = "apple")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleServeConfig {
    pub mode: AppleBackendMode,
    pub rollout_tokens: Option<u32>,
    pub private_ane_opt_in: bool,
}

#[cfg(not(feature = "apple"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleServeConfig {
    _private: (),
}

pub fn parse_serve_args<I, S>(args: I) -> Result<ServeConfig>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut parser = ArgParser::new(args);
    let mut model = None;
    let mut host = String::from("0.0.0.0");
    let mut port = 8000u16;
    let mut apple = AppleParseState::new();

    while let Some(arg) = parser.next() {
        match split_arg(&arg) {
            ("--model", Some(value)) => model = Some(PathBuf::from(value)),
            ("--model", None) => model = Some(PathBuf::from(parser.value("--model")?)),
            ("--host", Some(value)) => host = value.to_owned(),
            ("--host", None) => host = parser.value("--host")?,
            ("--port", Some(value)) => port = parse_u16("--port", value)?,
            ("--port", None) => port = parse_u16("--port", &parser.value("--port")?)?,
            ("--apple-mode", Some(value)) => apple.set_mode(value)?,
            ("--apple-mode", None) => apple.set_mode(&parser.value("--apple-mode")?)?,
            ("--apple-rollout-tokens", Some(value)) => apple.set_rollout_tokens(value)?,
            ("--apple-rollout-tokens", None) => {
                apple.set_rollout_tokens(&parser.value("--apple-rollout-tokens")?)?;
            }
            ("--apple-private-ane-opt-in", None) => apple.set_private_ane_opt_in()?,
            ("--apple-private-ane-opt-in", Some(_)) => {
                return Err(invalid(
                    "--apple-private-ane-opt-in",
                    "flag does not take a value",
                ));
            }
            (flag, _) if flag.starts_with("--apple-") => return Err(unknown_apple_flag(flag)),
            (flag, _) => return Err(invalid("cli", format!("unknown serve flag {flag:?}"))),
        }
    }

    Ok(ServeConfig {
        model,
        host,
        port,
        apple: apple.finish()?,
    })
}

pub fn apple_execution_unavailable() -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "rvllm-serve",
            op: "serve",
        },
        apple_ctx("serve"),
    )
}

struct ArgParser {
    args: std::vec::IntoIter<String>,
}

impl ArgParser {
    fn new<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args: Vec<String> = args.into_iter().map(Into::into).collect();
        if !args.is_empty() {
            args.remove(0);
        }
        Self {
            args: args.into_iter(),
        }
    }

    fn next(&mut self) -> Option<String> {
        self.args.next()
    }

    fn value(&mut self, flag: &'static str) -> Result<String> {
        match self.args.next() {
            Some(value) if !value.starts_with("--") => Ok(value),
            Some(value) => Err(invalid(flag, format!("expected value, got flag {value:?}"))),
            None => Err(invalid(flag, "missing value")),
        }
    }
}

fn split_arg(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((flag, value)) => (flag, Some(value)),
        None => (arg, None),
    }
}

fn parse_u16(flag: &'static str, value: &str) -> Result<u16> {
    value
        .parse()
        .map_err(|_| invalid(flag, format!("expected u16, got {value:?}")))
}

#[cfg(feature = "apple")]
fn parse_u32(flag: &'static str, value: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|_| invalid(flag, format!("expected u32, got {value:?}")))
}

fn invalid(field: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.into(),
        },
        field,
    )
}

fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "rvllm-serve",
        op,
        device: "apple-silicon",
    }
}

#[cfg(not(feature = "apple"))]
fn apple_feature_unavailable(op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "rvllm-apple",
            op,
        },
        apple_ctx(op),
    )
}

#[cfg(feature = "apple")]
fn unknown_apple_flag(flag: &str) -> RvllmError {
    invalid("cli", format!("unknown Apple serve flag {flag:?}"))
}

#[cfg(not(feature = "apple"))]
fn unknown_apple_flag(_flag: &str) -> RvllmError {
    apple_feature_unavailable("cli")
}

#[cfg(feature = "apple")]
struct AppleParseState {
    mode: Option<AppleBackendMode>,
    rollout_tokens: Option<u32>,
    private_ane_opt_in: bool,
}

#[cfg(feature = "apple")]
impl AppleParseState {
    const fn new() -> Self {
        Self {
            mode: None,
            rollout_tokens: None,
            private_ane_opt_in: false,
        }
    }

    fn set_mode(&mut self, value: &str) -> Result<()> {
        self.mode = Some(AppleBackendMode::from_flag(value).ok_or_else(|| {
            invalid(
                "--apple-mode",
                format!("unknown Apple backend mode {value:?}"),
            )
        })?);
        Ok(())
    }

    fn set_rollout_tokens(&mut self, value: &str) -> Result<()> {
        let tokens = parse_u32("--apple-rollout-tokens", value)?;
        if tokens == 0 {
            return Err(invalid(
                "--apple-rollout-tokens",
                "must be greater than zero",
            ));
        }
        self.rollout_tokens = Some(tokens);
        Ok(())
    }

    fn set_private_ane_opt_in(&mut self) -> Result<()> {
        self.private_ane_opt_in = true;
        Ok(())
    }

    fn finish(self) -> Result<Option<AppleServeConfig>> {
        match self.mode {
            None if self.rollout_tokens.is_none() && !self.private_ane_opt_in => Ok(None),
            None => Err(invalid("--apple-mode", "required when Apple flags are set")),
            Some(mode) if mode.requires_private_ane() && self.rollout_tokens.is_none() => {
                Err(invalid(
                    "--apple-rollout-tokens",
                    "required for private ANE Apple modes",
                ))
            }
            Some(mode) => Ok(Some(AppleServeConfig {
                mode,
                rollout_tokens: self.rollout_tokens,
                private_ane_opt_in: self.private_ane_opt_in,
            })),
        }
    }
}

#[cfg(not(feature = "apple"))]
struct AppleParseState;

#[cfg(not(feature = "apple"))]
impl AppleParseState {
    const fn new() -> Self {
        Self
    }

    fn set_mode(&mut self, _value: &str) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn set_rollout_tokens(&mut self, _value: &str) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn set_private_ane_opt_in(&mut self) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn finish(self) -> Result<Option<AppleServeConfig>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "apple")]
    #[test]
    fn parses_apple_serve_mode_without_touching_hardware() {
        use rvllm_runtime::AppleBackendMode;

        let cfg = match parse_serve_args([
            "rvllm-server",
            "--model",
            "/models/gemma-4-e2b",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--apple-mode",
            "metal-prefill-ane-ffn-rollout",
            "--apple-rollout-tokens",
            "4",
            "--apple-private-ane-opt-in",
        ]) {
            Ok(cfg) => cfg,
            Err(e) => panic!("unexpected parse error: {e}"),
        };

        assert_eq!(
            cfg.model.as_deref(),
            Some(std::path::Path::new("/models/gemma-4-e2b"))
        );
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8081);
        let apple = match cfg.apple {
            Some(apple) => apple,
            None => panic!("expected apple config"),
        };
        assert_eq!(apple.mode, AppleBackendMode::MetalPrefillAneFfnRollout);
        assert_eq!(apple.rollout_tokens, Some(4));
        assert!(apple.private_ane_opt_in);
    }

    #[cfg(not(feature = "apple"))]
    #[test]
    fn apple_serve_flags_require_apple_feature() {
        let err = match parse_serve_args(["rvllm-server", "--apple-mode", "metal-only"]) {
            Ok(_) => panic!("apple flags should be rejected when feature is disabled"),
            Err(err) => err,
        };

        assert!(format!("{err}").contains("FeatureNotAvailable"));
    }
}
