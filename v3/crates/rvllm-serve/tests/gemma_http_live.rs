//! Explicit, local-only acceptance of the signed server binary and real model.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server(Child);
impl Server {
    fn stop(&mut self) -> std::process::ExitStatus {
        if let Some(status) = self.0.try_wait().unwrap() {
            return status;
        }
        assert!(Command::new("/bin/kill")
            .args(["-TERM", &self.0.id().to_string()])
            .status()
            .unwrap()
            .success());
        // Never force-kill a process that may own an in-flight ANE request.
        self.0.wait().unwrap()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

fn socket(addr: &str, method: &str, path: &str, body: &str) -> TcpStream {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    stream
}

fn response(mut socket: TcpStream) -> (u16, Value) {
    let mut text = String::new();
    socket.read_to_string(&mut text).unwrap();
    let (header, body) = text.split_once("\r\n\r\n").unwrap();
    let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}

fn complete(addr: &str, prompt: &str, count: usize) -> Value {
    let (status, value) = response(socket(
        addr,
        "POST",
        "/v1/completions",
        &json!({"prompt":prompt,"max_tokens":count}).to_string(),
    ));
    assert_eq!(status, 200, "{value}");
    value
}

fn assert_reference(response: &Value, reference: &Value) {
    assert_eq!(response["prompt_token_ids"], reference["prompt_token_ids"]);
    assert_eq!(
        response["generated_token_ids"],
        reference["generated_tokens"]
    );
    assert_eq!(response["generated_text"], reference["generated_text"]);
    assert_eq!(
        response["backend_report"]["selected_route"],
        "metal-prefill-ane-decode"
    );
    assert!(response["backend_report"]["fallback"].is_null());
}

fn journal(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn boot() -> String {
    String::from_utf8(
        Command::new("/usr/sbin/sysctl")
            .args(["-n", "kern.boottime"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
}

#[test]
#[ignore = "real Gemma HTTP timing: two warmups, seven identical requests, sampled power, no driver journal"]
fn real_server_repeated_request_measurements() {
    let env_path = |name| PathBuf::from(std::env::var(name).unwrap());
    let output = env_path("RVLLM_HTTP_RECEIPT");
    fs::create_dir(&output).unwrap();
    let reference: Value =
        serde_json::from_slice(&fs::read(env_path("RVLLM_REFERENCE_B")).unwrap()).unwrap();
    let prompt = reference["user_text"].as_str().unwrap();
    let count = reference["generated_tokens"].as_array().unwrap().len();
    let prior = journal(&env_path("RVLLM_HTTP_KNOWN_MODELS_JOURNAL"));
    let models: BTreeSet<_> = prior
        .iter()
        .filter(|r| r["stage"] == "load_completed")
        .map(|r| r["model_id"].as_str().unwrap())
        .collect();
    assert_eq!(models.len(), 162);
    assert!(models
        .iter()
        .all(|id| !std::env::temp_dir().join(id).exists()));
    let boot_before = boot();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    drop(listener);
    let log = File::create(output.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(env_path("RVLLM_HTTP_SERVER_BIN"))
            .args(["--addr", &addr, "--backend", "metal-ane", "--model-dir"])
            .arg(env_path("RVLLM_GEMMA4_MODEL_DIR"))
            .arg("--metallib-bf16")
            .arg(env_path("RVLLM_METALLIB_BF16"))
            .args(["--max-total-tokens", "1024"])
            .env_remove("RVLLM_ANE_DIAGNOSTIC_JOURNAL")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited during preparation"
        );
        if fs::read_to_string(output.join("server.log"))
            .unwrap()
            .contains("rvllm-server listening")
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(100));
    }
    let (status, health) = response(socket(&addr, "GET", "/healthz", ""));
    assert_eq!(status, 200);
    assert_eq!(health["backend_status"]["ane_compile_budget_used"], 0);
    let mut trials = Vec::new();
    for index in 0..9 {
        let start = Instant::now();
        let response = complete(&addr, prompt, count);
        let client_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_reference(&response["backend_report"], &reference);
        let report = &response["backend_report"]["backend_report"];
        let measurement = &report["measurement"];
        let steps = measurement["steps"].as_array().unwrap();
        assert_eq!(steps.len(), count - 1);
        let stratum = &measurement["prefill"]["comparison_stratum"];
        let mut phases = vec![&measurement["prefill"], &measurement["import"]];
        phases.extend(steps.iter().map(|step| &step["measurement"]));
        let eligible = phases.iter().all(|phase| {
            phase["sampled_controls_eligible"] == true
                && !stratum.is_null()
                && phase["comparison_stratum"] == *stratum
        });
        trials.push(json!({"index":index,"warmup":index < 2,"client_wall_ms":client_wall_ms,
            "sampled_controls_eligible":eligible,"comparison_stratum":stratum,
            "ane_decode_steps":steps.len(),"ane_steps_per_second":steps.len() as f64*1000.0/report["decode_ms"].as_f64().unwrap(),
            "response":response}));
        fs::write(
            output.join("trials.json"),
            serde_json::to_vec_pretty(&trials).unwrap(),
        )
        .unwrap();
    }
    let status = server.stop();
    let remaining: Vec<_> = models
        .iter()
        .filter(|id| std::env::temp_dir().join(id).exists())
        .collect();
    let receipt = json!({"schema":"rvllm.gemma12b_http_measurements.v1","health":health,
        "driver_journal":false,"trials":trials,"server_exit_success":status.success(),
        "boot_unchanged":boot_before == boot(),"remaining_owned_staging_directories":remaining,
        "claim":"Repeated identical workload characterization. No before/after speedup. Compare only matching eligible sampled strata; CPU cycles exclude GPU and ANE."});
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    assert!(status.success());
    assert!(remaining.is_empty());
    assert_eq!(boot_before, boot());
}

#[test]
#[ignore = "real Gemma 4 12B HTTP/SSE/cancellation/SIGTERM; explicit binary, model, metallib, references, output directory"]
fn real_server_streams_recovers_and_shuts_down() {
    let env_path = |name| PathBuf::from(std::env::var(name).unwrap());
    let binary = env_path("RVLLM_HTTP_SERVER_BIN");
    let model = env_path("RVLLM_GEMMA4_MODEL_DIR");
    let metallib = env_path("RVLLM_METALLIB_BF16");
    let output = env_path("RVLLM_HTTP_RECEIPT");
    fs::create_dir(&output).unwrap();
    let load = |name| serde_json::from_slice::<Value>(&fs::read(env_path(name)).unwrap()).unwrap();
    let a = load("RVLLM_REFERENCE_A");
    let b = load("RVLLM_REFERENCE_B");
    let prompt_a = "Reply with just the capital of France.";
    let prompt_b = b["user_text"].as_str().unwrap();
    let count_b = b["generated_tokens"].as_array().unwrap().len();
    let boot_before = boot();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    drop(listener);
    let phases = output.join("driver-phases.jsonl");
    let log = File::create(output.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(&binary)
            .args(["--addr", &addr, "--backend", "metal-ane", "--model-dir"])
            .arg(model)
            .arg("--metallib-bf16")
            .arg(metallib)
            .args(["--max-total-tokens", "1024", "--max-new-tokens", "16"])
            .env("RVLLM_ANE_DIAGNOSTIC_JOURNAL", &phases)
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(600);
    // The port is bound before device setup; wait for the ready log so HTTP
    // reads don't obscure an initialization failure or consume 120 seconds.
    loop {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited during preparation; inspect server.log"
        );
        if fs::read_to_string(output.join("server.log"))
            .unwrap()
            .contains("rvllm-server listening")
        {
            break;
        }
        assert!(Instant::now() < deadline, "server readiness timed out");
        std::thread::sleep(Duration::from_millis(100));
    }
    let (status, health) = response(socket(&addr, "GET", "/healthz", ""));
    assert_eq!(status, 200);
    assert_eq!(health["backend_status"]["ane_compile_budget_used"], 0);
    let first = complete(&addr, prompt_a, 16);
    assert_reference(&first["backend_report"], &a);

    // SSE headers are sent before inference. Queue A while B is in flight.
    let stream = socket(
        &addr,
        "POST",
        "/v1/completions",
        &json!({"prompt":prompt_b,"max_tokens":count_b,"stream":true}).to_string(),
    );
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(!line.is_empty());
        if line == "\r\n" {
            break;
        }
    }
    let queued_addr = addr.clone();
    let queued = std::thread::spawn(move || complete(&queued_addr, prompt_a, 16));
    let mut text = String::new();
    let mut terminal = None;
    let mut deltas = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(!line.is_empty(), "SSE ended without DONE");
        let Some(data) = line.trim_end().strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let event: Value = serde_json::from_str(data).unwrap();
        assert!(event.get("error").is_none(), "{event}");
        if let Some(delta) = event["choices"][0]["text"].as_str() {
            text.push_str(delta);
            if !delta.is_empty() {
                deltas += 1;
            }
        }
        if let Some(report) = event.get("backend_report") {
            terminal = Some(report.clone());
        }
    }
    assert!(deltas > 1);
    assert_eq!(text, b["generated_text"].as_str().unwrap());
    let terminal = terminal.unwrap();
    assert_reference(&terminal, &b);
    let queued = queued.join().unwrap();
    assert_reference(&queued["backend_report"], &a);

    for stream in [false, true] {
        let (status, value) = response(socket(
            &addr,
            "POST",
            "/v1/completions",
            &json!({"prompt":prompt_a,"max_tokens":1024,"stream":stream}).to_string(),
        ));
        assert_eq!(status, 400, "{value}");
    }

    let eval_before = journal(&phases)
        .iter()
        .filter(|r| r["stage"] == "evaluate_completed")
        .count();
    let mut interrupted = BufReader::new(socket(
        &addr,
        "POST",
        "/v1/completions",
        &json!({"prompt":prompt_b,"max_tokens":count_b,"stream":true}).to_string(),
    ));
    let mut received = 0;
    while received < 2 {
        line.clear();
        interrupted.read_line(&mut line).unwrap();
        assert!(!line.is_empty());
        if let Some(data) = line.trim_end().strip_prefix("data: ") {
            let event: Value = serde_json::from_str(data).unwrap();
            if event["choices"][0]["text"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
            {
                received += 1;
            }
        }
    }
    interrupted.get_ref().shutdown(Shutdown::Both).unwrap();
    drop(interrupted);
    let recovered = complete(&addr, prompt_a, 16);
    assert_reference(&recovered["backend_report"], &a);
    let eval_after = journal(&phases)
        .iter()
        .filter(|r| r["stage"] == "evaluate_completed")
        .count();
    // A consumes one ANE step (208 evaluations); B must have started decode,
    // then cancelled before all nine of its decode inputs completed.
    let interrupted_evaluations = eval_after - eval_before - 208;
    assert!(interrupted_evaluations >= 208 && interrupted_evaluations < (count_b - 1) * 208);
    let fresh = complete(&addr, prompt_b, count_b);
    assert_reference(&fresh["backend_report"], &b);

    let stopped = server.stop();
    let records = journal(&phases);
    let mut counts = BTreeMap::<String, usize>::new();
    let mut models = BTreeSet::new();
    for record in &records {
        let stage = record["stage"].as_str().unwrap();
        *counts.entry(stage.into()).or_default() += 1;
        if stage == "load_completed" {
            models.insert(record["model_id"].as_str().unwrap());
        }
    }
    let remaining: Vec<_> = models
        .iter()
        .filter(|id| std::env::temp_dir().join(id).exists())
        .collect();
    let receipt = json!({"schema":"rvllm.gemma12b_http_qualification.v1", "server_binary":binary,
        "server_exit_success":stopped.success(), "health":health, "json_first":first,
        "sse_terminal":terminal,"sse_text":text,"sse_text_deltas":deltas,"queued":queued,
        "recovered":recovered,"fresh":fresh,"interrupted_evaluations":interrupted_evaluations,
        "driver_events":counts,"remaining_owned_staging_directories":remaining,
        "boot_unchanged":boot_before == boot(),
        "timing_claim":"Durable driver journal; correctness and lifecycle qualification, not a speed comparison."});
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    assert!(stopped.success());
    assert_eq!(counts.get("cache_hit"), Some(&162));
    assert_eq!(counts.get("unload_completed"), Some(&162));
    assert!(!counts.contains_key("compile_begin"));
    assert!(!counts.contains_key("unload_failed"));
    assert!(remaining.is_empty());
    assert_eq!(boot_before, boot());
}
