fn main() {
    // rvllm-server entry point. Phase D.
    let cfg = match rvllm_serve::parse_serve_args(std::env::args()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("rvllm-server: {e}");
            std::process::exit(1);
        }
    };
    if cfg.apple.is_some() {
        eprintln!(
            "rvllm-server: {}",
            rvllm_serve::apple_execution_unavailable()
        );
        std::process::exit(1);
    }
    eprintln!("rvllm-server: not yet implemented (Phase D)");
    std::process::exit(1);
}
