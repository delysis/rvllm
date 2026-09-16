use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(all(
    feature = "macos-private-ane-research",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod gemma_disaggregated;

#[cfg(feature = "apple-bench")]
pub mod continuous_bench;
#[cfg(feature = "apple-bench")]
pub mod prompt_cache_bench;

pub const SERVER_SCHEMA: &str = "rvllm.openai_completions.v1";
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 1024 * 1024;
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(30);
static NEXT_COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

/// Runs the same bounded, continuous Metal worker used by the HTTP server.
/// This surface is deliberately available only in development/benchmark
/// builds so shipping applications do not acquire a second serving API.
#[cfg(all(feature = "apple-bench", target_os = "macos"))]
pub fn run_apple_continuous_benchmark(
    config: ServerConfig,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    batch: u32,
    iters: u32,
    warmup: u32,
    cache_enabled: bool,
) -> Result<Value, String> {
    let backend = metal_direct::PreparedMetalBackend::new(&config)?;
    let health = CompletionBackend::health(&backend);
    let mut request = rvllm_runtime::GenerateRequest::new(
        prompt_token_ids
            .into_iter()
            .map(rvllm_core::TokenId)
            .collect(),
        u32::try_from(max_new_tokens)
            .map_err(|_| format!("max_new_tokens must be at most {}", u32::MAX))?,
    );
    if !cache_enabled {
        request.cache_policy = rvllm_runtime::CachePolicy::Disabled;
    }
    let mut record = backend.run_continuous_benchmark(
        request,
        usize::try_from(batch).map_err(|_| "batch does not fit usize".to_owned())?,
        usize::try_from(iters).map_err(|_| "iters does not fit usize".to_owned())?,
        usize::try_from(warmup).map_err(|_| "warmup does not fit usize".to_owned())?,
    )?;
    if let Some(object) = record.as_object_mut() {
        object.insert("backend".to_owned(), Value::String("apple".to_owned()));
        object.insert(
            "backend_profile".to_owned(),
            Value::String("apple".to_owned()),
        );
        object.insert("worker_health".to_owned(), health);
    }
    Ok(record)
}

/// Runs the exact 512-token Apple prompt-cache promotion gate against the
/// production continuous Metal worker. T1 is mandatory; macOS T2 is measured
/// through a separate warm-only worker so a T1 hit cannot mask restore cost.
#[cfg(all(feature = "apple-bench", target_os = "macos"))]
pub fn run_apple_prompt_cache_gate(
    config: ServerConfig,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
) -> Result<Value, String> {
    use prompt_cache_bench::{ExpectedCacheTier, PromptCacheGateConfig};
    use rvllm_runtime::{AppleEngineConfig, CachePolicy};

    let prompt_tokens = prompt_token_ids
        .into_iter()
        .map(rvllm_core::TokenId)
        .collect::<Vec<_>>();
    let max_output_tokens = u32::try_from(max_new_tokens)
        .map_err(|_| format!("max_new_tokens must be at most {}", u32::MAX))?;

    let mut hot_config = AppleEngineConfig::default();
    hot_config.cache_policy = CachePolicy::MemoryOnly;
    hot_config.warm_cache_bytes = 0;
    let hot_backend =
        metal_direct::PreparedMetalBackend::new_with_engine_config(&config, hot_config)?;
    let hot_health = CompletionBackend::health(&hot_backend);
    let hot = hot_backend.run_prompt_cache_gate(PromptCacheGateConfig {
        prompt_tokens: prompt_tokens.clone(),
        max_output_tokens,
        expected_tier: ExpectedCacheTier::Hot,
    })?;
    drop(hot_backend);

    let mut warm_config = AppleEngineConfig::default();
    warm_config.cache_policy = CachePolicy::MemoryOnly;
    warm_config.hot_cache_bytes = 0;
    let warm_backend =
        metal_direct::PreparedMetalBackend::new_with_engine_config(&config, warm_config)?;
    let warm_health = CompletionBackend::health(&warm_backend);
    let warm = warm_backend.run_prompt_cache_gate(PromptCacheGateConfig {
        prompt_tokens,
        max_output_tokens,
        expected_tier: ExpectedCacheTier::WarmOrRecompute,
    })?;

    let t1_passed = hot["passed"].as_bool().unwrap_or(false);
    let t2_safe = warm["passed"].as_bool().unwrap_or(false);
    Ok(serde_json::json!({
        "schema": "rvllm.apple_prompt_cache_promotion.v1",
        "model_dir": config.model_dir,
        "t1": hot,
        "t2": warm,
        "t1_worker_health": hot_health,
        "t2_worker_health": warm_health,
        "t1_promotion_eligible": t1_passed,
        "t2_safe_or_promotion_eligible": t2_safe,
        "passed": t1_passed && t2_safe,
        "claim": if t1_passed {
            "T1 promotion gate passed on this run; T2 is promoted only when its own record says promotion_eligible"
        } else {
            "prompt-cache promotion gate did not pass"
        },
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerBackendMode {
    Subprocess,
    MetalDirect,
    MetalAne,
}

impl ServerBackendMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Subprocess => "subprocess",
            Self::MetalDirect => "metal-direct",
            Self::MetalAne => "metal-prefill-ane-decode",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerConfig {
    pub addr: String,
    pub model_dir: PathBuf,
    pub infer_bin: PathBuf,
    pub backend: ServerBackendMode,
    pub max_new_tokens: usize,
    pub max_total_tokens: Option<usize>,
    pub large_model_opt_in: bool,
    pub metallib_bf16: Option<PathBuf>,
    pub ane_compile_budget: usize,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct CompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    model: String,
    schema: &'static str,
    choices: Vec<CompletionChoice>,
    usage: CompletionUsage,
    backend_report: Value,
}

trait CompletionBackend: Send + Sync {
    fn shutdown(&self) {}
    fn validate_request(&self, _prompt: &str, _count: usize) -> Result<(), String> {
        Ok(())
    }
    fn backend_name(&self) -> &'static str;
    fn model_label(&self) -> String;
    fn health(&self) -> Value;
    fn complete(&self, prompt: &str, max_new_tokens: usize) -> Result<Value, String>;

    /// Produces text deltas as they become available, followed by exactly one
    /// terminal event. Legacy backends may use the default buffered adapter;
    /// the Apple engine overrides this with real token-by-token delivery.
    fn stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        emit: &mut dyn FnMut(BackendStreamEvent) -> Result<(), String>,
    ) -> Result<(), String> {
        let report = self.complete(prompt, max_new_tokens)?;
        let text = report
            .get("generated_text")
            .and_then(Value::as_str)
            .ok_or_else(|| "inference report missing generated_text".to_owned())?
            .to_owned();
        if !text.is_empty() {
            emit(BackendStreamEvent::TextDelta(text))?;
        }
        let finish_reason = report
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        emit(BackendStreamEvent::Finished {
            finish_reason,
            report,
        })
    }
}

#[derive(Debug)]
enum BackendStreamEvent {
    TextDelta(String),
    Finished {
        finish_reason: String,
        report: Value,
    },
}

pub fn parse_args_from<I, S>(args: I) -> Result<ServerConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut addr = "127.0.0.1:8080".to_owned();
    let mut model_dir = None;
    let mut infer_bin = PathBuf::from("rvllm_metal_infer");
    let mut backend = ServerBackendMode::Subprocess;
    let mut max_new_tokens = 1usize;
    let mut max_total_tokens = None;
    let mut large_model_opt_in = false;
    let mut metallib_bf16 = None;
    let mut ane_compile_budget = 0;

    let mut iter = args.into_iter().map(Into::into);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--addr" => {
                addr = iter
                    .next()
                    .ok_or_else(|| "--addr requires a value".to_owned())?;
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--model-dir requires a value".to_owned())?,
                ));
            }
            "--infer-bin" => {
                infer_bin = PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--infer-bin requires a value".to_owned())?,
                );
            }
            "--backend" => {
                let raw = iter
                    .next()
                    .ok_or_else(|| "--backend requires a value".to_owned())?;
                backend = parse_backend_mode(&raw)?;
            }
            "--max-new-tokens" => {
                max_new_tokens = parse_positive_usize(
                    "--max-new-tokens",
                    &iter
                        .next()
                        .ok_or_else(|| "--max-new-tokens requires a value".to_owned())?,
                )?;
                validate_u32_count("--max-new-tokens", max_new_tokens)?;
            }
            "--max-total-tokens" => {
                max_total_tokens = Some(parse_max_total_tokens(
                    "--max-total-tokens",
                    &iter
                        .next()
                        .ok_or_else(|| "--max-total-tokens requires a value".to_owned())?,
                )?);
            }
            "--large-model-opt-in" => large_model_opt_in = true,
            "--metallib-bf16" => {
                metallib_bf16 = Some(PathBuf::from(
                    iter.next().ok_or("--metallib-bf16 requires a value")?,
                ))
            }
            "--ane-compile-budget" => {
                ane_compile_budget = iter
                    .next()
                    .ok_or("--ane-compile-budget requires a value")?
                    .parse::<usize>()
                    .map_err(|error| error.to_string())?
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
    }

    let model_dir = model_dir.ok_or_else(|| "--model-dir is required".to_owned())?;
    if max_new_tokens == 0 {
        return Err("--max-new-tokens must be positive".to_owned());
    }
    if backend == ServerBackendMode::MetalAne {
        if metallib_bf16.is_none() {
            return Err("--metallib-bf16 is required for metal-ane".into());
        }
        if !matches!(max_total_tokens.unwrap_or(1024), 64 | 1024) {
            return Err("metal-ane requires --max-total-tokens 64 or 1024".into());
        }
        if ane_compile_budget > 16 {
            return Err("--ane-compile-budget must be 0..=16".into());
        }
    } else if metallib_bf16.is_some() || ane_compile_budget != 0 {
        return Err("--metallib-bf16 and --ane-compile-budget apply only to metal-ane".into());
    }

    Ok(ServerConfig {
        addr,
        model_dir,
        infer_bin,
        backend,
        max_new_tokens,
        max_total_tokens,
        large_model_opt_in,
        metallib_bf16,
        ane_compile_budget,
    })
}

pub fn usage() -> String {
    "usage: rvllm-server --model-dir <DIR> [--addr HOST:PORT] \
     [--backend subprocess|metal-engine|metal-direct|metal-ane] [--infer-bin PATH] [--max-new-tokens N] \
     [--max-total-tokens N] [--large-model-opt-in] [--metallib-bf16 PATH] [--ane-compile-budget 0..16]"
        .to_owned()
}

fn parse_backend_mode(raw: &str) -> Result<ServerBackendMode, String> {
    match raw {
        "subprocess" => Ok(ServerBackendMode::Subprocess),
        "metal-engine" | "metal-direct" => Ok(ServerBackendMode::MetalDirect),
        "metal-ane" => Ok(ServerBackendMode::MetalAne),
        other => Err(format!(
            "--backend must be subprocess, metal-engine, metal-direct or metal-ane; got {other:?}"
        )),
    }
}

fn parse_positive_usize(flag: &str, raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("{flag} must be a positive integer: {err}"))?;
    if value == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(value)
}

fn parse_max_total_tokens(flag: &str, raw: &str) -> Result<usize, String> {
    let value = parse_positive_usize(flag, raw)?;
    validate_u32_count(flag, value)?;
    Ok(value)
}

fn validate_u32_count(label: &str, value: usize) -> Result<(), String> {
    if u32::try_from(value).is_err() {
        return Err(format!("{label} must be at most {}", u32::MAX));
    }
    Ok(())
}

pub fn run_server(config: ServerConfig) -> Result<(), String> {
    run_server_until(config, Arc::new(AtomicBool::new(false)))
}

/// Embedders own the shutdown signal. The binary installs process handlers;
/// this library never changes global signal handling or power configuration.
pub fn run_server_until(config: ServerConfig, shutdown: Arc<AtomicBool>) -> Result<(), String> {
    let listener =
        TcpListener::bind(&config.addr).map_err(|err| format!("bind {}: {err}", config.addr))?;
    let backend = build_completion_backend(&config)?;
    eprintln!(
        "rvllm-server listening on {} with backend {}",
        config.addr,
        backend.backend_name()
    );
    serve_until(config, backend, listener, shutdown)
}

fn serve_until(
    config: ServerConfig,
    backend: Arc<dyn CompletionBackend>,
    listener: TcpListener,
    shutdown: Arc<AtomicBool>,
) -> Result<(), String> {
    listener
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    let config = Arc::new(config);
    let mut connections: Vec<(std::thread::JoinHandle<()>, TcpStream)> = Vec::new();
    let outcome = (|| {
        while !shutdown.load(Ordering::Acquire) {
            let mut index = 0;
            while index < connections.len() {
                if connections[index].0.is_finished() {
                    let (thread, _) = connections.swap_remove(index);
                    let _ = thread.join();
                } else {
                    index += 1;
                }
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_read_timeout(Some(HTTP_IO_TIMEOUT))
                        .map_err(|err| format!("set request read timeout: {err}"))?;
                    stream
                        .set_write_timeout(Some(HTTP_IO_TIMEOUT))
                        .map_err(|err| format!("set response write timeout: {err}"))?;
                    if connections.len() >= 64 {
                        let _ = write_json_response(
                            &mut stream,
                            503,
                            &serde_json::json!({"error":{"message":"connection capacity reached"}}),
                        );
                        continue;
                    }
                    let config = Arc::clone(&config);
                    let backend = Arc::clone(&backend);
                    let socket = stream.try_clone().map_err(|error| error.to_string())?;
                    let thread = std::thread::Builder::new()
                        .name("rvllm-http".to_owned())
                        .spawn(move || {
                            if let Err(err) = handle_stream(&config, backend.as_ref(), &mut stream)
                            {
                                eprintln!("connection error: {err}");
                            }
                            let _ = stream.shutdown(Shutdown::Both);
                        })
                        .map_err(|err| format!("spawn HTTP connection handler: {err}"))?;
                    connections.push((thread, socket));
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(format!("accept connection: {err}")),
            }
        }
        Ok(())
    })();
    // Close network waits before joining the accelerator owner. This also
    // drops/cancels disconnected request handles at the HTTP boundary.
    for (_, socket) in &connections {
        let _ = socket.shutdown(Shutdown::Both);
    }
    backend.shutdown();
    for (thread, _) in connections {
        let _ = thread.join();
    }
    outcome
}

fn build_completion_backend(config: &ServerConfig) -> Result<Arc<dyn CompletionBackend>, String> {
    match config.backend {
        ServerBackendMode::Subprocess => Ok(Arc::new(SubprocessBackend {
            config: config.clone(),
        })),
        ServerBackendMode::MetalDirect => build_metal_direct_backend(config),
        ServerBackendMode::MetalAne => build_gemma_disaggregated_backend(config),
    }
}

#[cfg(all(
    feature = "macos-private-ane-research",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn build_gemma_disaggregated_backend(
    config: &ServerConfig,
) -> Result<Arc<dyn CompletionBackend>, String> {
    Ok(Arc::new(gemma_disaggregated::GemmaBackend::new(config)?))
}

#[cfg(not(all(
    feature = "macos-private-ane-research",
    target_os = "macos",
    target_arch = "aarch64"
)))]
fn build_gemma_disaggregated_backend(
    _: &ServerConfig,
) -> Result<Arc<dyn CompletionBackend>, String> {
    Err("metal-ane requires Apple Silicon macOS and the macos-private-ane-research feature".into())
}

fn handle_stream(
    config: &ServerConfig,
    backend: &dyn CompletionBackend,
    stream: &mut TcpStream,
) -> Result<(), String> {
    let request = match read_http_request(stream) {
        Ok(request) => request,
        Err(err) => {
            return write_json_response(
                stream,
                400,
                &serde_json::json!({"error": {"message": err}}),
            );
        }
    };
    if request.method == "POST" && request.path == "/v1/completions" {
        let body = match parse_completion_request(config, &request) {
            Ok(body) => body,
            Err(response) => return write_json_response(stream, response.status, &response.body),
        };
        if body.stream {
            if let Err(error) = backend.validate_request(&body.prompt, body.max_tokens.unwrap()) {
                return write_json_response(
                    stream,
                    400,
                    &serde_json::json!({"error":{"message":error}}),
                );
            }
            return write_completion_stream(stream, backend, &body);
        }
    }
    let response = match route_request(config, backend, &request) {
        Ok(response) => response,
        Err(err) => HttpResponse {
            status: 500,
            body: serde_json::json!({"error": {"message": err}}),
        },
    };
    write_json_response(stream, response.status, &response.body)
}

#[derive(Debug, PartialEq, Eq)]
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

#[derive(Debug, PartialEq)]
struct HttpResponse {
    status: u16,
    body: Value,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    read_http_request_from(stream)
}

fn read_http_request_from(reader: &mut impl Read) -> Result<HttpRequest, String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = reader
            .read(&mut tmp)
            .map_err(|err| format!("read request: {err}"))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if buf.len() > MAX_HTTP_HEADER_BYTES {
            return Err("request header too large".to_owned());
        }
    }
    let header_end = buf
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP request: missing header terminator".to_owned())?;
    if header_end > MAX_HTTP_HEADER_BYTES {
        return Err("request header too large".to_owned());
    }
    let header = String::from_utf8(buf[..header_end].to_vec())
        .map_err(|err| format!("request header is not UTF-8: {err}"))?;
    let mut lines = header.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "malformed HTTP request: missing request line".to_owned())?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "malformed HTTP request: missing method".to_owned())?
        .to_owned();
    let path = parts
        .next()
        .ok_or_else(|| "malformed HTTP request: missing path".to_owned())?
        .to_owned();
    let version = parts
        .next()
        .ok_or_else(|| "malformed HTTP request: missing HTTP version".to_owned())?;
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") || parts.next().is_some() {
        return Err("malformed HTTP request line".to_owned());
    }

    let mut content_length = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err(format!("malformed HTTP header: {line:?}"));
        };
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err("multiple Content-Length headers are not supported".to_owned());
            }
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|err| format!("invalid Content-Length: {err}"))?,
            );
        }
        if name.eq_ignore_ascii_case("transfer-encoding")
            && !value.trim().eq_ignore_ascii_case("identity")
        {
            return Err("Transfer-Encoding is not supported".to_owned());
        }
    }
    let content_length = content_length.unwrap_or(0);
    if content_length > MAX_HTTP_BODY_BYTES {
        return Err(format!(
            "request body too large: {content_length} bytes exceeds {MAX_HTTP_BODY_BYTES}"
        ));
    }

    let body_start = header_end + 4;
    let mut body = buf[body_start..].to_vec();
    while body.len() < content_length {
        let n = reader
            .read(&mut tmp)
            .map_err(|err| format!("read request body: {err}"))?;
        if n == 0 {
            return Err(format!(
                "truncated HTTP request body: expected {content_length} bytes, received {}",
                body.len()
            ));
        }
        body.extend_from_slice(&tmp[..n]);
        if body.len() > MAX_HTTP_BODY_BYTES {
            return Err("request body too large".to_owned());
        }
    }
    body.truncate(content_length);
    let body =
        String::from_utf8(body).map_err(|err| format!("request body is not UTF-8: {err}"))?;

    Ok(HttpRequest { method, path, body })
}

fn route_request(
    config: &ServerConfig,
    backend: &dyn CompletionBackend,
    request: &HttpRequest,
) -> Result<HttpResponse, String> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/healthz") => {
            let health = backend.health();
            let ready = health["ready"].as_bool().unwrap_or(false);
            Ok(HttpResponse {
                status: if ready { 200 } else { 503 },
                body: serde_json::json!({
                    "status": if ready { "ok" } else { "unavailable" },
                    "schema": "rvllm.health.v1",
                    "model_dir": config.model_dir,
                    "backend": backend.backend_name(),
                    "backend_status": health,
                }),
            })
        }
        ("POST", "/v1/completions") => complete_request(config, backend, request),
        _ => Ok(HttpResponse {
            status: 404,
            body: serde_json::json!({"error": {"message": "not found"}}),
        }),
    }
}

fn complete_request(
    config: &ServerConfig,
    backend: &dyn CompletionBackend,
    request: &HttpRequest,
) -> Result<HttpResponse, String> {
    let body = match parse_completion_request(config, request) {
        Ok(body) => body,
        Err(response) => return Ok(response),
    };
    let max_tokens = body.max_tokens.unwrap_or(config.max_new_tokens);
    if let Err(error) = backend.validate_request(&body.prompt, max_tokens) {
        return Ok(HttpResponse {
            status: 400,
            body: serde_json::json!({"error":{"message":error}}),
        });
    }
    let backend_report = backend.complete(&body.prompt, max_tokens)?;
    let response = completion_response(backend.model_label(), backend_report)?;
    Ok(HttpResponse {
        status: 200,
        body: serde_json::to_value(response)
            .map_err(|err| format!("serialize completion response: {err}"))?,
    })
}

fn parse_completion_request(
    config: &ServerConfig,
    request: &HttpRequest,
) -> Result<CompletionRequest, HttpResponse> {
    let mut body: CompletionRequest = match serde_json::from_str(&request.body) {
        Ok(body) => body,
        Err(err) => {
            return Err(HttpResponse {
                status: 400,
                body: serde_json::json!({
                    "error": {"message": format!("parse completion request JSON: {err}")}
                }),
            });
        }
    };
    if body.prompt.trim().is_empty() {
        return Err(HttpResponse {
            status: 400,
            body: serde_json::json!({"error": {"message": "prompt must not be empty"}}),
        });
    }
    let max_tokens = body.max_tokens.unwrap_or(config.max_new_tokens);
    if max_tokens == 0 {
        return Err(HttpResponse {
            status: 400,
            body: serde_json::json!({"error": {"message": "max_tokens must be positive"}}),
        });
    }
    if u32::try_from(max_tokens).is_err() {
        return Err(HttpResponse {
            status: 400,
            body: serde_json::json!({
                "error": {"message": format!("max_tokens must be at most {}", u32::MAX)}
            }),
        });
    }
    body.max_tokens = Some(max_tokens);
    Ok(body)
}

struct SubprocessBackend {
    config: ServerConfig,
}

impl CompletionBackend for SubprocessBackend {
    fn backend_name(&self) -> &'static str {
        ServerBackendMode::Subprocess.as_str()
    }

    fn model_label(&self) -> String {
        self.config.model_dir.display().to_string()
    }

    fn health(&self) -> Value {
        serde_json::json!({
            "ready": true,
            "infer_bin": self.config.infer_bin,
            "prepares_per_request": true,
        })
    }

    fn complete(&self, prompt: &str, max_new_tokens: usize) -> Result<Value, String> {
        let mut cmd = Command::new(&self.config.infer_bin);
        cmd.arg("--model-dir")
            .arg(&self.config.model_dir)
            .arg("--prompt")
            .arg(prompt)
            .arg("--max-new-tokens")
            .arg(max_new_tokens.to_string())
            .arg("--json");
        if let Some(max_total_tokens) = self.config.max_total_tokens {
            cmd.arg("--max-total-tokens")
                .arg(max_total_tokens.to_string());
        }
        if self.config.large_model_opt_in {
            cmd.arg("--large-model-opt-in");
        }
        let output = cmd
            .output()
            .map_err(|err| format!("launch {}: {err}", self.config.infer_bin.display()))?;
        if !output.status.success() {
            return Err(format!(
                "inference command failed with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        serde_json::from_slice(&output.stdout).map_err(|err| {
            format!(
                "parse inference command JSON from {}: {err}; stdout={}",
                self.config.infer_bin.display(),
                String::from_utf8_lossy(&output.stdout)
            )
        })
    }
}

fn completion_response(model_label: String, report: Value) -> Result<CompletionResponse, String> {
    let generated_text = report
        .get("generated_text")
        .and_then(Value::as_str)
        .ok_or_else(|| "inference report missing generated_text".to_owned())?
        .to_owned();
    let finish_reason = report
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let prompt_tokens = report
        .get("prompt_token_ids")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let completion_tokens = report
        .get("generated_token_ids")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let total_tokens = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or_else(|| "completion usage token count overflow".to_owned())?;
    Ok(CompletionResponse {
        id: next_completion_id(),
        object: "text_completion",
        model: model_label,
        schema: SERVER_SCHEMA,
        choices: vec![CompletionChoice {
            index: 0,
            text: generated_text,
            finish_reason,
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
        backend_report: report,
    })
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: &Value) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let body = serde_json::to_string_pretty(body)
        .map_err(|err| format!("serialize HTTP response body: {err}"))?;
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|err| format!("write response: {err}"))
}

fn write_completion_stream(
    writer: &mut impl Write,
    backend: &dyn CompletionBackend,
    request: &CompletionRequest,
) -> Result<(), String> {
    let max_tokens = request
        .max_tokens
        .ok_or_else(|| "validated completion request is missing max_tokens".to_owned())?;
    let completion_id = next_completion_id();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let model = backend.model_label();
    writer
        .write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nx-accel-buffering: no\r\nconnection: close\r\n\r\n",
        )
        .map_err(|err| format!("write SSE response headers: {err}"))?;

    let result = backend.stream(&request.prompt, max_tokens, &mut |event| match event {
        BackendStreamEvent::TextDelta(text) => write_sse_json(
            writer,
            &serde_json::json!({
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "logprobs": null,
                    "finish_reason": null,
                }],
            }),
        ),
        BackendStreamEvent::Finished {
            finish_reason,
            report,
        } => write_sse_json(
            writer,
            &serde_json::json!({
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": "",
                    "logprobs": null,
                    "finish_reason": finish_reason,
                }],
                "backend_report": report,
            }),
        ),
    });

    match result {
        Ok(()) => write_sse_done(writer),
        Err(error) => {
            // Once streaming headers have been sent the status cannot change.
            // Send an OpenAI-shaped error when the client is still connected;
            // a broken writer returns immediately and drops the engine request,
            // which is the cancellation signal.
            write_sse_json(writer, &serde_json::json!({"error": {"message": error}}))?;
            write_sse_done(writer)
        }
    }
}

fn write_sse_json(writer: &mut impl Write, value: &Value) -> Result<(), String> {
    let payload =
        serde_json::to_string(value).map_err(|err| format!("serialize SSE event: {err}"))?;
    writer
        .write_all(format!("data: {payload}\n\n").as_bytes())
        .map_err(|err| format!("write SSE event: {err}"))?;
    writer
        .flush()
        .map_err(|err| format!("flush SSE event: {err}"))
}

fn write_sse_done(writer: &mut impl Write) -> Result<(), String> {
    writer
        .write_all(b"data: [DONE]\n\n")
        .map_err(|err| format!("write SSE completion marker: {err}"))?;
    writer
        .flush()
        .map_err(|err| format!("flush SSE completion marker: {err}"))
}

fn next_completion_id() -> String {
    let completion_id = NEXT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed);
    format!("cmpl-rvllm-{}-{completion_id}", std::process::id())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
mod metal_direct {
    use super::*;
    use rvllm_core::TokenId;
    #[cfg(target_os = "macos")]
    use rvllm_runtime::apple_metal_backend::MetalModelCapacity;
    #[cfg(target_os = "macos")]
    use rvllm_runtime::EngineHandle;
    use rvllm_runtime::{
        ActiveRequest, AppleEngineConfig, BackendKind, BackendReport, ContinuousInferenceWorker,
        ContinuousStepOutput, FinishReason, GenerateRequest, InferenceError, MemoryPressure,
        TokenEvent,
    };
    #[cfg(target_os = "macos")]
    use std::ffi::OsString;
    #[cfg(target_os = "macos")]
    use std::sync::Mutex;

    #[cfg(target_os = "macos")]
    const CLAIM: &str =
        "Apple Metal server completion workflow; not production-ready until acceptance gates pass";
    #[cfg(target_os = "macos")]
    const JSON_SCHEMA: &str = "rvllm.apple_metal_server_completion.v1";
    #[cfg(target_os = "macos")]
    const LARGE_MODEL_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";
    #[cfg(target_os = "macos")]
    const MAX_TOTAL_TOKENS_ENV: &str = "RVLLM_METAL_MAX_TOTAL_TOKENS";
    #[cfg(target_os = "macos")]
    const MAX_BATCH_TOKENS_ENV: &str = "RVLLM_METAL_MAX_BATCH_TOKENS";
    #[cfg(target_os = "macos")]
    const MAX_BATCH_SEQUENCES_ENV: &str = "RVLLM_METAL_MAX_BATCH_SEQUENCES";
    const METAL_PREFILL_TOKEN_BUDGET: usize = 128;
    #[cfg(target_os = "macos")]
    const DEFAULT_MAX_METAL_TOTAL_TOKENS: usize = 2048;

    #[cfg(target_os = "macos")]
    pub struct PreparedMetalBackend {
        model_dir: PathBuf,
        tokenizer: tokenizers::Tokenizer,
        engine: EngineHandle,
        max_supported_total_tokens: usize,
        large_model_opt_in: bool,
        worker_health: Arc<Mutex<Option<WorkerHealth>>>,
        engine_config: AppleEngineConfig,
        _large_model_env: EnvGuard,
        _max_tokens_env: EnvGuard,
        _max_batch_tokens_env: EnvGuard,
        _max_batch_sequences_env: EnvGuard,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug)]
    struct WorkerHealth {
        prepare_ms: f64,
        arena_bytes: usize,
        debug_sync: bool,
        max_batch_tokens: usize,
        physical_kv_pages: u32,
        kv_page_size: u32,
        http_cache_scope: &'static str,
        capacity: MetalModelCapacity,
    }

    #[cfg(target_os = "macos")]
    struct RuntimeContinuousWorker(Box<dyn ContinuousInferenceWorker>);

    #[cfg(target_os = "macos")]
    impl ContinuousInferenceWorker for RuntimeContinuousWorker {
        fn admit(
            &mut self,
            request_id: rvllm_core::ReqId,
            request: &GenerateRequest,
            cancellation: &std::sync::atomic::AtomicBool,
        ) -> Result<(), InferenceError> {
            self.0.admit(request_id, request, cancellation)
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            self.0.step(active)
        }

        fn abort(&mut self, request_id: rvllm_core::ReqId) {
            self.0.abort(request_id);
        }

        fn handle_memory_pressure(
            &mut self,
            pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            self.0.handle_memory_pressure(pressure)
        }
    }

    #[cfg(target_os = "macos")]
    impl PreparedMetalBackend {
        pub fn new(config: &ServerConfig) -> Result<Self, String> {
            Self::new_with_engine_config(config, AppleEngineConfig::default())
        }

        pub(super) fn new_with_engine_config(
            config: &ServerConfig,
            engine_config: AppleEngineConfig,
        ) -> Result<Self, String> {
            if !config.model_dir.is_dir() {
                return Err(format!(
                    "model path does not exist or is not a directory: {}",
                    config.model_dir.display()
                ));
            }
            let tokenizer = load_tokenizer(&config.model_dir)?;
            let env_opt_in = std::env::var(LARGE_MODEL_ENV).ok().as_deref() == Some("1");
            let effective_large_opt_in = config.large_model_opt_in || env_opt_in;

            let max_supported_total_tokens = config
                .max_total_tokens
                .unwrap_or(DEFAULT_MAX_METAL_TOTAL_TOKENS);
            let large_model_env =
                EnvGuard::set_if(LARGE_MODEL_ENV, config.large_model_opt_in && !env_opt_in);
            let max_tokens_env =
                EnvGuard::set_value(MAX_TOTAL_TOKENS_ENV, max_supported_total_tokens.to_string());
            let max_batch_tokens_env =
                EnvGuard::set_value(MAX_BATCH_TOKENS_ENV, METAL_PREFILL_TOKEN_BUDGET.to_string());
            let max_batch_sequences_env = EnvGuard::set_value(
                MAX_BATCH_SEQUENCES_ENV,
                engine_config.maximum_concurrency.to_string(),
            );
            let local_cache_scope = is_local_http_cache_scope(&config.addr);
            let worker_health = Arc::new(Mutex::new(None));
            let worker_health_for_factory = Arc::clone(&worker_health);
            let worker_config = rvllm_runtime::DevelopmentMetalWorkerConfig {
                model_dir: config.model_dir.clone(),
                max_supported_total_tokens,
                rollout_tokens: config.max_new_tokens.max(1),
                cache_namespace: "local-http-session".to_owned(),
                enable_memory_cache: local_cache_scope,
            };
            let engine = EngineHandle::spawn_continuous_with_factory(
                engine_config.clone(),
                move |engine_config| {
                    let (worker, runtime_health) =
                        rvllm_runtime::create_development_apple_metal_worker(
                            engine_config,
                            &worker_config,
                        )?;
                    let health = WorkerHealth {
                        prepare_ms: runtime_health.prepare_ms,
                        arena_bytes: runtime_health.arena_bytes,
                        debug_sync: runtime_health.debug_sync,
                        max_batch_tokens: runtime_health.max_batch_tokens,
                        physical_kv_pages: runtime_health.physical_kv_pages,
                        kv_page_size: runtime_health.kv_page_size,
                        http_cache_scope: if local_cache_scope {
                            "local-session"
                        } else {
                            "disabled-non-loopback"
                        },
                        capacity: runtime_health.capacity,
                    };
                    *worker_health_for_factory
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(health);
                    Ok(RuntimeContinuousWorker(worker))
                },
            )
            .map_err(|err| format!("start shared Metal inference worker: {err}"))?;

            Ok(Self {
                model_dir: config.model_dir.clone(),
                tokenizer,
                engine,
                max_supported_total_tokens,
                large_model_opt_in: effective_large_opt_in,
                worker_health,
                engine_config,
                _large_model_env: large_model_env,
                _max_tokens_env: max_tokens_env,
                _max_batch_tokens_env: max_batch_tokens_env,
                _max_batch_sequences_env: max_batch_sequences_env,
            })
        }

        #[cfg(feature = "apple-bench")]
        pub(super) fn run_continuous_benchmark(
            &self,
            request: GenerateRequest,
            batch: usize,
            iters: usize,
            warmup: usize,
        ) -> Result<Value, String> {
            crate::continuous_bench::run_engine_benchmark(
                &self.engine,
                crate::continuous_bench::ContinuousBenchConfig {
                    request,
                    batch,
                    iters,
                    warmup,
                },
            )
        }

        #[cfg(feature = "apple-bench")]
        pub(super) fn run_prompt_cache_gate(
            &self,
            config: crate::prompt_cache_bench::PromptCacheGateConfig,
        ) -> Result<Value, String> {
            crate::prompt_cache_bench::run_prompt_cache_gate(&self.engine, config)
        }

        fn generate_with(
            &self,
            prompt: &str,
            max_new_tokens: usize,
            mut emit_delta: impl FnMut(String) -> Result<(), String>,
        ) -> Result<Value, String> {
            let prompt_token_ids = tokenize_prompt(&self.tokenizer, prompt)?;
            validate_token_budget(
                prompt_token_ids.len(),
                max_new_tokens,
                self.max_supported_total_tokens,
            )?;
            let request = GenerateRequest::new(
                prompt_token_ids.iter().copied().map(TokenId).collect(),
                u32::try_from(max_new_tokens)
                    .map_err(|_| format!("max_new_tokens must be at most {}", u32::MAX))?,
            );
            let mut handle = self
                .engine
                .submit(request)
                .map_err(|err| format!("submit inference request: {err}"))?;
            let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
            let mut streamed_text = String::new();
            let (finish_reason, backend_report) = loop {
                match handle
                    .recv()
                    .map_err(|err| format!("receive inference event: {err}"))?
                {
                    TokenEvent::Token { token_id, text, .. } => {
                        generated_token_ids.push(token_id.raw());
                        if let Some(text) = text {
                            streamed_text.push_str(&text);
                            if !text.is_empty() {
                                emit_delta(text.to_string())?;
                            }
                        }
                    }
                    TokenEvent::Finished {
                        finish_reason,
                        report,
                        ..
                    } => break (finish_reason_label(finish_reason), report),
                }
            };
            let generated_text = decode_text(&self.tokenizer, &generated_token_ids, "generated")?;
            // Decoder deltas are expected to reconstruct the exact final text.
            // Fail closed if a tokenizer normalization edge case violates that.
            if streamed_text != generated_text {
                return Err(format!(
                    "incremental tokenizer output mismatch: streamed {} bytes, final decode {} bytes",
                    streamed_text.len(),
                    generated_text.len()
                ));
            }
            Ok(serde_json::json!({
                "schema": JSON_SCHEMA,
                "claim": CLAIM,
                "model_dir": self.model_dir,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "max_new_tokens": max_new_tokens,
                "generated_token_ids": generated_token_ids,
                "generated_text": generated_text,
                "finish_reason": finish_reason,
                "backend_report": backend_report_json(&backend_report),
            }))
        }
    }

    impl CompletionBackend for PreparedMetalBackend {
        fn shutdown(&self) {
            self.engine.shutdown();
        }
        fn backend_name(&self) -> &'static str {
            ServerBackendMode::MetalDirect.as_str()
        }

        fn model_label(&self) -> String {
            self.model_dir.display().to_string()
        }

        fn health(&self) -> Value {
            let health = self
                .worker_health
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let worker_available = self.engine.is_available();
            serde_json::json!({
                "ready": health.is_some() && worker_available,
                "worker_available": worker_available,
                "prepared_once": true,
                "prepare_ms": health.as_ref().map(|value| value.prepare_ms),
                "arena_bytes": health.as_ref().map(|value| value.arena_bytes),
                "max_supported_total_tokens": self.max_supported_total_tokens,
                "large_model_opt_in": self.large_model_opt_in,
                "debug_sync": health.as_ref().map(|value| value.debug_sync),
                "max_batch_tokens": health.as_ref().map(|value| value.max_batch_tokens),
                "physical_kv_pages": health.as_ref().map(|value| value.physical_kv_pages),
                "kv_page_size": health.as_ref().map(|value| value.kv_page_size),
                "http_prompt_cache_scope": health.as_ref().map(|value| value.http_cache_scope),
                "kv_pool_sizing": "working-set-budget-after-weights-three-scratch-slots-and-metadata",
                "budget_driven_kv_capacity": true,
                "recommended_working_set_bytes": health.as_ref().map(|value| value.capacity.recommended_working_set_bytes),
                "working_set_reserve_bytes": health.as_ref().map(|value| value.capacity.reserve_bytes),
                "usable_working_set_bytes": health.as_ref().map(|value| value.capacity.usable_bytes),
                "weights_bytes": health.as_ref().map(|value| value.capacity.weights_bytes),
                "scratch_slot_bytes": health.as_ref().map(|value| value.capacity.scratch_slot_bytes),
                "three_slot_scratch_budget_bytes": health.as_ref().map(|value| value.capacity.scratch_budget_bytes),
                "metadata_bytes": health.as_ref().map(|value| value.capacity.metadata_bytes),
                "kv_budget_bytes": health.as_ref().map(|value| value.capacity.kv_budget_bytes),
                "kv_page_bytes": health.as_ref().map(|value| value.capacity.kv_page_bytes),
                "allocated_kv_bytes": health.as_ref().map(|value| value.capacity.allocated_kv_bytes),
                "prepared_arena_bytes": health.as_ref().map(|value| value.capacity.prepared_arena_bytes),
                "max_useful_kv_pages": health.as_ref().map(|value| value.capacity.max_useful_kv_pages),
                "admission_required_kv_pages": health.as_ref().map(|value| value.capacity.admission_required_kv_pages),
                "metal_float_type": health.as_ref().map(|value| value.capacity.metal_float_type),
                "kv_storage_format": health.as_ref().map(|value| value.capacity.kv_storage_format),
                "experimental_kv_int8_opt_in": health.as_ref().map(|value| value.capacity.experimental_kv_int8_opt_in),
                "experimental_kv_int8_active": health.as_ref().map(|value| value.capacity.experimental_kv_int8_active),
                "low_bit_projection_count": health.as_ref().map(|value| value.capacity.low_bit_projection_count),
                "low_bit_weight_bytes": health.as_ref().map(|value| value.capacity.low_bit_weight_bytes),
                "numeric_abi_version": health.as_ref().map(|value| value.capacity.numeric_abi_version),
                "shared_bounded_ingress": true,
                "ingress_queue_capacity": self.engine_config.ingress_queue_capacity,
                "configured_request_concurrency": self.engine_config.maximum_concurrency,
                "worker_active_limit": self.engine_config.maximum_concurrency,
                "worker_adapter": "engine-paged-kv-continuous",
                "continuous_batching": true,
                "paged_kv": true,
            })
        }

        fn complete(&self, prompt: &str, max_new_tokens: usize) -> Result<Value, String> {
            self.generate_with(prompt, max_new_tokens, |_| Ok(()))
        }

        fn stream(
            &self,
            prompt: &str,
            max_new_tokens: usize,
            emit: &mut dyn FnMut(BackendStreamEvent) -> Result<(), String>,
        ) -> Result<(), String> {
            let report = self.generate_with(prompt, max_new_tokens, |delta| {
                emit(BackendStreamEvent::TextDelta(delta))
            })?;
            let finish_reason = report["finish_reason"]
                .as_str()
                .unwrap_or("unknown")
                .to_owned();
            emit(BackendStreamEvent::Finished {
                finish_reason,
                report,
            })
        }
    }

    fn finish_reason_label(reason: FinishReason) -> &'static str {
        match reason {
            FinishReason::EndOfSequence => "stop",
            FinishReason::Length => "length",
            FinishReason::StopToken => "stop",
        }
    }

    fn backend_report_json(report: &BackendReport) -> Value {
        serde_json::json!({
            "selected_route": match report.selected_backend {
                BackendKind::Metal => "metal",
                BackendKind::CoreMl => "coreml",
                BackendKind::MetalPrefillAneDecode => "metal-prefill-ane-decode",
            },
            "cache_tier": format!("{:?}", report.cache_tier).to_lowercase(),
            "matched_tokens": report.matched_cache_tokens,
            "saved_tokens": report.saved_prefill_tokens,
            "queue_ms": ms(report.queue_time),
            "batch_size": report.batch_size,
            "padding_tokens": report.padding_tokens,
            "prefill_ms": ms(report.prefill_time),
            "decode_ms": ms(report.decode_time),
            "fallback": report.fallback.as_ref().map(|fallback| serde_json::json!({
                "from": format!("{:?}", fallback.from).to_lowercase(),
                "to": format!("{:?}", fallback.to).to_lowercase(),
                "reason": fallback.reason.as_ref(),
            })),
            "resident_memory_bytes": report.resident_memory_bytes,
            "thermal_state": format!("{:?}", report.thermal_state).to_lowercase(),
        })
    }

    fn load_tokenizer(model_dir: &std::path::Path) -> Result<tokenizers::Tokenizer, String> {
        let path = model_dir.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&path)
            .map_err(|err| format!("load tokenizer {}: {err}", path.display()))
    }

    fn tokenize_prompt(
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
    ) -> Result<Vec<u32>, String> {
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|err| format!("tokenize prompt: {err}"))?;
        let mut token_ids = encoding.get_ids().to_vec();
        token_ids.insert(0, 2);
        if token_ids.is_empty() {
            return Err("prompt produced zero token IDs".to_owned());
        }
        Ok(token_ids)
    }

    pub(super) fn validate_token_budget(
        prompt_len: usize,
        max_new_tokens: usize,
        max_total_tokens: usize,
    ) -> Result<(), String> {
        if prompt_len == 0 {
            return Err("Metal prompt must contain at least one token".to_owned());
        }
        if max_new_tokens == 0 {
            return Err("Metal max_new_tokens must be positive".to_owned());
        }
        let total_tokens = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| "Metal token budget overflow".to_owned())?;
        if u32::try_from(total_tokens).is_err() {
            return Err(format!(
                "Metal token positions must fit in u32; got {total_tokens} total tokens"
            ));
        }
        if total_tokens > max_total_tokens {
            return Err(format!(
                "Metal workflow supports prompt token count + max_new_tokens <= configured max_total_tokens ({max_total_tokens}); got {} + {}",
                prompt_len, max_new_tokens
            ));
        }
        Ok(())
    }

    pub(super) fn is_local_http_cache_scope(addr: &str) -> bool {
        addr.parse::<std::net::SocketAddr>()
            .is_ok_and(|socket| socket.ip().is_loopback())
            || addr
                .strip_prefix("localhost:")
                .is_some_and(|port| port.parse::<u16>().is_ok())
    }

    fn decode_text(
        tokenizer: &tokenizers::Tokenizer,
        token_ids: &[u32],
        label: &str,
    ) -> Result<String, String> {
        tokenizer
            .decode(token_ids, true)
            .map_err(|err| format!("decode {label} token IDs: {err}"))
    }

    fn ms(duration: std::time::Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    struct EnvGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn set_if(name: &'static str, enabled: bool) -> Self {
            if enabled {
                Self::set_value(name, "1".to_owned())
            } else {
                Self {
                    name,
                    previous: None,
                }
            }
        }

        fn set_value(name: &'static str, value: String) -> Self {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            Self { name, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.as_ref() {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn build_metal_direct_backend(config: &ServerConfig) -> Result<Arc<dyn CompletionBackend>, String> {
    Ok(Arc::new(metal_direct::PreparedMetalBackend::new(config)?))
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
fn build_metal_direct_backend(
    _config: &ServerConfig,
) -> Result<Arc<dyn CompletionBackend>, String> {
    Err("metal-direct backend requires rvllm-serve --features apple on macOS".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::time::Instant;

    fn config() -> ServerConfig {
        ServerConfig {
            addr: "127.0.0.1:0".to_owned(),
            model_dir: PathBuf::from("/tmp/model"),
            infer_bin: PathBuf::from("rvllm_metal_infer"),
            backend: ServerBackendMode::Subprocess,
            max_new_tokens: 2,
            max_total_tokens: Some(16),
            large_model_opt_in: true,
            metallib_bf16: None,
            ane_compile_budget: 0,
        }
    }

    struct StaticBackend {
        calls: AtomicUsize,
        stopped: AtomicBool,
        report: Value,
    }

    impl CompletionBackend for StaticBackend {
        fn shutdown(&self) {
            self.stopped.store(true, Ordering::Release);
        }

        fn validate_request(&self, _prompt: &str, count: usize) -> Result<(), String> {
            if count > 16 {
                Err("test context capacity exceeded".into())
            } else {
                Ok(())
            }
        }

        fn backend_name(&self) -> &'static str {
            "test"
        }

        fn model_label(&self) -> String {
            "test-model".to_owned()
        }

        fn health(&self) -> Value {
            serde_json::json!({"ready": !self.stopped.load(Ordering::Acquire)})
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> Result<Value, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.report.clone())
        }
    }

    fn static_backend() -> StaticBackend {
        StaticBackend {
            calls: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            report: serde_json::json!({
                "generated_text": ",",
                "finish_reason": "length",
                "prompt_token_ids": [2, 9259],
                "generated_token_ids": [236764]
            }),
        }
    }

    #[test]
    fn gemma_args_require_qualified_capacity_and_explicit_metallib() {
        let base = ["--model-dir", "/models/gemma", "--backend", "metal-ane"];
        assert!(parse_args_from(base).unwrap_err().contains("metallib"));
        let args: Vec<_> = base
            .into_iter()
            .chain(["--metallib-bf16", "/tmp/model.metallib"])
            .collect();
        let cfg = parse_args_from(args.clone()).unwrap();
        assert_eq!(cfg.backend, ServerBackendMode::MetalAne);
        assert_eq!(cfg.ane_compile_budget, 0);
        for tail in [
            ["--max-total-tokens", "128"],
            ["--ane-compile-budget", "17"],
        ] {
            assert!(parse_args_from(args.iter().copied().chain(tail)).is_err());
        }
    }

    #[test]
    fn server_shutdown_closes_idle_connections_and_waits_for_backend() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let backend = Arc::new(static_backend());
        let shutdown = Arc::new(AtomicBool::new(false));
        let owner_backend = backend.clone();
        let owner_signal = shutdown.clone();
        let (done, completion) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || {
            done.send(serve_until(config(), owner_backend, listener, owner_signal))
                .unwrap();
        });
        let mut idle = TcpStream::connect(address).unwrap();
        idle.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        // A later health request proves the accept loop has admitted the idle
        // connection, whose handler is blocked reading an incomplete header.
        let mut health = TcpStream::connect(address).unwrap();
        health
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        health.write_all(b"GET /healthz HTTP/1.1\r\n\r\n").unwrap();
        let mut response = String::new();
        health.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        shutdown.store(true, Ordering::Release);
        completion
            .recv_timeout(Duration::from_secs(2))
            .expect("bounded shutdown")
            .unwrap();
        owner.join().unwrap();
        assert!(backend.stopped.load(Ordering::Acquire));
        // A handler can have buffered a final 400 when shutdown wakes its
        // incomplete-header read. Either way the peer must reach EOF promptly.
        idle.read_to_end(&mut Vec::new()).unwrap();
        let response = route_request(
            &config(),
            backend.as_ref(),
            &HttpRequest {
                method: "GET".into(),
                path: "/healthz".into(),
                body: String::new(),
            },
        )
        .unwrap();
        assert_eq!(response.status, 503);
    }

    #[test]
    fn over_capacity_requests_fail_before_json_or_stream_generation() {
        for stream in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (mut server, _) = listener.accept().unwrap();
            let body =
                serde_json::json!({"prompt":"Hello", "max_tokens":17, "stream":stream}).to_string();
            write!(
                client,
                "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            let backend = static_backend();
            handle_stream(&config(), &backend, &mut server).unwrap();
            server.shutdown(Shutdown::Both).unwrap();
            let mut response = String::new();
            client.read_to_string(&mut response).unwrap();
            assert!(response.starts_with("HTTP/1.1 400"));
            assert!(!response.contains("text/event-stream"));
            assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn parses_server_args() {
        let cfg = parse_args_from([
            "--addr",
            "127.0.0.1:9000",
            "--model-dir",
            "/models/gemma",
            "--infer-bin",
            "/tmp/rvllm_metal_infer",
            "--backend",
            "metal-direct",
            "--max-new-tokens",
            "4",
            "--max-total-tokens",
            "32",
            "--large-model-opt-in",
        ])
        .expect("parse args");
        assert_eq!(cfg.addr, "127.0.0.1:9000");
        assert_eq!(cfg.model_dir, PathBuf::from("/models/gemma"));
        assert_eq!(cfg.infer_bin, PathBuf::from("/tmp/rvllm_metal_infer"));
        assert_eq!(cfg.backend, ServerBackendMode::MetalDirect);
        assert_eq!(cfg.max_new_tokens, 4);
        assert_eq!(cfg.max_total_tokens, Some(32));
        assert!(cfg.large_model_opt_in);
    }

    #[test]
    fn server_args_reject_token_counts_that_do_not_fit_runtime_metadata() {
        let too_large = (u32::MAX as u64 + 1).to_string();
        let err = parse_args_from([
            "--model-dir",
            "/models/gemma",
            "--max-total-tokens",
            &too_large,
        ])
        .expect_err("oversized token count must fail");
        assert!(err.contains("at most"));
    }

    #[test]
    fn http_reader_rejects_truncated_body() {
        let raw = b"POST /v1/completions HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}";
        let err = read_http_request_from(&mut Cursor::new(raw))
            .expect_err("truncated request body must fail");
        assert!(err.contains("truncated"));
    }

    #[test]
    fn http_reader_rejects_oversized_body_before_allocating_it() {
        let raw = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_HTTP_BODY_BYTES + 1
        );
        let err = read_http_request_from(&mut Cursor::new(raw.as_bytes()))
            .expect_err("oversized request body must fail");
        assert!(err.contains("too large"));
    }

    #[test]
    fn http_reader_rejects_transfer_encoding() {
        let raw = b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
        let err = read_http_request_from(&mut Cursor::new(raw))
            .expect_err("chunked request must fail clearly");
        assert!(err.contains("Transfer-Encoding"));
    }

    #[test]
    fn health_route_reports_ready_json() {
        let req = HttpRequest {
            method: "GET".to_owned(),
            path: "/healthz".to_owned(),
            body: String::new(),
        };
        let mut backend = static_backend();
        let resp = route_request(&config(), &mut backend, &req).expect("route health");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body["status"], "ok");
        assert_eq!(resp.body["backend"], "test");
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completion_response_maps_backend_report_to_openai_shape() {
        let report = serde_json::json!({
            "generated_text": ",",
            "finish_reason": "length",
            "prompt_token_ids": [2, 9259],
            "generated_token_ids": [236764]
        });
        let response =
            completion_response("test-model".to_owned(), report).expect("completion response");
        assert_eq!(response.schema, SERVER_SCHEMA);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.choices[0].text, ",");
        assert_eq!(response.usage.prompt_tokens, 2);
        assert_eq!(response.usage.completion_tokens, 1);
        assert_eq!(response.usage.total_tokens, 3);
    }

    #[test]
    fn completion_route_uses_backend_once() {
        let req = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/completions".to_owned(),
            body: serde_json::json!({"prompt": "Hello", "max_tokens": 1}).to_string(),
        };
        let mut backend = static_backend();
        let resp = route_request(&config(), &mut backend, &req).expect("route completion");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body["schema"], SERVER_SCHEMA);
        assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn completion_route_rejects_empty_prompt_without_launching_backend() {
        let req = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/completions".to_owned(),
            body: serde_json::json!({"prompt": ""}).to_string(),
        };
        let mut backend = static_backend();
        let resp = route_request(&config(), &mut backend, &req).expect("route completion");
        assert_eq!(resp.status, 400);
        assert!(resp.body["error"]["message"]
            .as_str()
            .expect("error")
            .contains("prompt"));
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completion_route_rejects_malformed_json_as_bad_request() {
        let req = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/completions".to_owned(),
            body: "{".to_owned(),
        };
        let mut backend = static_backend();
        let resp = route_request(&config(), &mut backend, &req).expect("route completion");
        assert_eq!(resp.status, 400);
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completion_stream_uses_openai_sse_frames_and_done_marker() {
        let backend = static_backend();
        let request = CompletionRequest {
            prompt: "Hello".to_owned(),
            max_tokens: Some(1),
            stream: true,
        };
        let mut output = Vec::new();
        write_completion_stream(&mut output, &backend, &request).expect("write SSE stream");
        let output = String::from_utf8(output).expect("UTF-8 response");
        assert!(output.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(output.contains("content-type: text/event-stream\r\n"));
        let body = output
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("HTTP response body");
        let frames = body
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(frames.last(), Some(&"[DONE]"));
        let first: Value = serde_json::from_str(frames[0]).expect("first SSE JSON");
        assert_eq!(first["object"], "text_completion");
        assert_eq!(first["choices"][0]["text"], ",");
        assert!(first["choices"][0]["finish_reason"].is_null());
        let terminal: Value = serde_json::from_str(frames[1]).expect("terminal SSE JSON");
        assert_eq!(terminal["choices"][0]["text"], "");
        assert_eq!(terminal["choices"][0]["finish_reason"], "length");
    }

    struct CancellationWorker {
        observed_cancellation: Arc<AtomicBool>,
    }

    impl rvllm_runtime::InferenceWorker for CancellationWorker {
        fn generate(
            &mut self,
            _request: &rvllm_runtime::GenerateRequest,
            output: &mut dyn rvllm_runtime::TokenEmitter,
        ) -> Result<rvllm_runtime::GenerationOutcome, rvllm_runtime::InferenceError> {
            for _ in 0..64 {
                if let Err(error) = output.emit_token(rvllm_core::TokenId(9), Some("x".into())) {
                    self.observed_cancellation.store(true, Ordering::Release);
                    return Err(error);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(rvllm_runtime::GenerationOutcome::length_limited(
                rvllm_runtime::BackendReport::new(rvllm_runtime::BackendKind::Metal),
            ))
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: rvllm_runtime::MemoryPressure,
        ) -> Result<(), rvllm_runtime::InferenceError> {
            Ok(())
        }
    }

    struct EngineTestBackend {
        engine: rvllm_runtime::EngineHandle,
    }

    impl CompletionBackend for EngineTestBackend {
        fn backend_name(&self) -> &'static str {
            "test-engine"
        }

        fn model_label(&self) -> String {
            "test-model".to_owned()
        }

        fn health(&self) -> Value {
            serde_json::json!({"ready": true})
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> Result<Value, String> {
            Err("test backend is streaming-only".to_owned())
        }

        fn stream(
            &self,
            _prompt: &str,
            max_new_tokens: usize,
            emit: &mut dyn FnMut(BackendStreamEvent) -> Result<(), String>,
        ) -> Result<(), String> {
            let mut handle = self
                .engine
                .submit(rvllm_runtime::GenerateRequest::new(
                    vec![rvllm_core::TokenId(1)],
                    max_new_tokens as u32,
                ))
                .map_err(|error| error.to_string())?;
            loop {
                match handle.recv().map_err(|error| error.to_string())? {
                    rvllm_runtime::TokenEvent::Token { text, .. } => {
                        let result = emit(BackendStreamEvent::TextDelta(
                            text.unwrap_or_default().to_string(),
                        ));
                        if let Err(error) = result {
                            handle.cancel();
                            return Err(error);
                        }
                    }
                    rvllm_runtime::TokenEvent::Finished { .. } => {
                        return emit(BackendStreamEvent::Finished {
                            finish_reason: "length".to_owned(),
                            report: serde_json::json!({}),
                        });
                    }
                }
            }
        }
    }

    struct FailAfterHeader {
        writes_remaining: usize,
    }

    impl Write for FailAfterHeader {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.writes_remaining == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "client disconnected",
                ));
            }
            self.writes_remaining -= 1;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn streaming_disconnect_cancels_engine_request() {
        let observed_cancellation = Arc::new(AtomicBool::new(false));
        let observed_by_worker = Arc::clone(&observed_cancellation);
        let mut engine_config = rvllm_runtime::AppleEngineConfig::default();
        engine_config.event_queue_capacity = 1;
        let engine = rvllm_runtime::EngineHandle::spawn_with_factory(engine_config, move |_| {
            Ok(CancellationWorker {
                observed_cancellation: observed_by_worker,
            })
        })
        .expect("start test engine");
        let backend = EngineTestBackend { engine };
        let request = CompletionRequest {
            prompt: "Hello".to_owned(),
            max_tokens: Some(64),
            stream: true,
        };
        let mut writer = FailAfterHeader {
            writes_remaining: 1,
        };
        let error = write_completion_stream(&mut writer, &backend, &request)
            .expect_err("broken client writer must fail");
        assert!(error.contains("SSE"));

        let deadline = Instant::now() + Duration::from_secs(1);
        while !observed_cancellation.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(observed_cancellation.load(Ordering::Acquire));
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn metal_token_budget_rejects_integer_overflow() {
        let err = metal_direct::validate_token_budget(usize::MAX, 1, usize::MAX)
            .expect_err("overflowing token budget must fail");
        assert!(err.contains("overflow"));
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn http_prompt_cache_is_scoped_only_to_loopback_sessions() {
        assert!(metal_direct::is_local_http_cache_scope("127.0.0.1:8080"));
        assert!(metal_direct::is_local_http_cache_scope("[::1]:8080"));
        assert!(metal_direct::is_local_http_cache_scope("localhost:8080"));
        assert!(!metal_direct::is_local_http_cache_scope("0.0.0.0:8080"));
        assert!(!metal_direct::is_local_http_cache_scope("192.0.2.1:8080"));
    }
}
