use std::process::ExitCode;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn main() -> ExitCode {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", rvllm_serve::usage());
        return ExitCode::SUCCESS;
    }
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal = shutdown.clone();
    let outcome = rvllm_serve::parse_args_from(raw_args).and_then(|config| {
        ctrlc::try_set_handler(move || signal.store(true, Ordering::Release))
            .map_err(|error| format!("install shutdown handler: {error}"))?;
        rvllm_serve::run_server_until(config, shutdown)
    });
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            eprintln!("{}", rvllm_serve::usage());
            ExitCode::FAILURE
        }
    }
}
