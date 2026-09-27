//! Seal the dense text projections in the pinned original Gemma 4 12B-it checkpoint.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const METALLIB_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const ROLES: [(&str, usize); 7] = [
    ("q_proj", 48),
    ("k_proj", 48),
    ("v_proj", 40),
    ("o_proj", 48),
    ("gate_proj", 48),
    ("up_proj", 48),
    ("down_proj", 48),
];
const EXPECTED_ELEMENTS: u64 = 10_899_947_520;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Serialize)]
struct Selector {
    tensor_name: String,
    shape: [u64; 2],
    w4_argument: String,
    w8_argument: String,
}

#[derive(Serialize)]
struct Plan {
    schema: &'static str,
    model_sha256: &'static str,
    config_sha256: &'static str,
    tokenizer_sha256: &'static str,
    donor_metallib_sha256: &'static str,
    group_size: u64,
    text_projection_elements: u64,
    selectors: Vec<Selector>,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 6 {
        eprintln!("usage: rvllm_gemma4_lowbit_full_selectors MODEL.safetensors CONFIG.json TOKENIZER.json DONOR.core.metallib NEW_ABSOLUTE_PLAN.json");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[1..]) {
        eprintln!("rvllm_gemma4_lowbit_full_selectors: {error}");
        std::process::exit(1);
    }
}

fn verify_sha(path: &Path, expected: &str) -> Result<()> {
    let actual =
        Sha256Digest::file(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if actual.as_str() != expected {
        return Err(format!(
            "{}: SHA-256 differs from sealed input",
            path.display()
        ));
    }
    Ok(())
}

fn projection_role(name: &str) -> Option<&'static str> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.len() != 7
        || parts[0..3] != ["model", "language_model", "layers"]
        || parts[3]
            .parse::<usize>()
            .ok()
            .filter(|layer| *layer < 48)
            .is_none()
        || parts[6] != "weight"
    {
        return None;
    }
    let family = parts[4];
    let role = parts[5];
    let valid = match family {
        "self_attn" => matches!(role, "q_proj" | "k_proj" | "v_proj" | "o_proj"),
        "mlp" => matches!(role, "gate_proj" | "up_proj" | "down_proj"),
        _ => false,
    };
    if !valid {
        return None;
    }
    ROLES
        .iter()
        .find_map(|(name, _)| (*name == role).then_some(*name))
}

fn read_header(path: &Path) -> Result<Value> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let file_len = file.metadata().map_err(|error| error.to_string())?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err("invalid safetensors header length".into());
    }
    let mut header = vec![0_u8; usize::try_from(header_len).map_err(|e| e.to_string())?];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    parse_strict_json(&header).map_err(|error| error.to_string())
}

fn plan_from_header(header: &Value) -> Result<Plan> {
    let tensors = header
        .as_object()
        .ok_or("safetensors header is not an object")?;
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut selectors = Vec::new();
    let mut elements = 0_u64;
    for (name, metadata) in tensors {
        let Some(role) = projection_role(name) else {
            continue;
        };
        if metadata["dtype"] != "BF16" {
            return Err(format!("{name}: expected BF16"));
        }
        let shape = metadata["shape"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing shape"))?;
        if shape.len() != 2 {
            return Err(format!("{name}: expected matrix"));
        }
        let rows = shape[0]
            .as_u64()
            .ok_or_else(|| format!("{name}: invalid rows"))?;
        let k = shape[1]
            .as_u64()
            .ok_or_else(|| format!("{name}: invalid K"))?;
        if rows == 0 || k == 0 || k % 32 != 0 {
            return Err(format!("{name}: invalid group-32 shape"));
        }
        elements = elements
            .checked_add(rows.checked_mul(k).ok_or("element overflow")?)
            .ok_or("element overflow")?;
        *counts.entry(role).or_default() += 1;
        selectors.push(Selector {
            tensor_name: name.clone(),
            shape: [rows, k],
            w4_argument: format!("{name}=w4a16-group32"),
            w8_argument: format!("{name}=w8a16-group32"),
        });
    }
    for (role, expected) in ROLES {
        if counts.get(role).copied().unwrap_or_default() != expected {
            return Err(format!(
                "{role}: expected {expected}, found {}",
                counts.get(role).copied().unwrap_or_default()
            ));
        }
    }
    if selectors.len() != 328 || elements != EXPECTED_ELEMENTS {
        return Err(format!(
            "unexpected projection count or elements: {} / {elements}",
            selectors.len()
        ));
    }
    selectors.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    Ok(Plan {
        schema: "rvllm.gemma4.lowbit-full-text-selectors.v1",
        model_sha256: MODEL_SHA,
        config_sha256: CONFIG_SHA,
        tokenizer_sha256: TOKENIZER_SHA,
        donor_metallib_sha256: METALLIB_SHA,
        group_size: 32,
        text_projection_elements: elements,
        selectors,
    })
}

fn run(args: &[PathBuf]) -> Result<()> {
    let [model, config, tokenizer, metallib, output] = args else {
        return Err("expected five arguments".into());
    };
    if !output.is_absolute() || output.exists() {
        return Err("output must be a new absolute path".into());
    }
    verify_sha(model, MODEL_SHA)?;
    verify_sha(config, CONFIG_SHA)?;
    verify_sha(tokenizer, TOKENIZER_SHA)?;
    verify_sha(metallib, METALLIB_SHA)?;
    let plan = plan_from_header(&read_header(model)?)?;
    let mut bytes = serde_json::to_vec_pretty(&plan).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| format!("{}: {error}", output.display()))?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_are_exact_and_reject_experts() {
        assert_eq!(
            projection_role("model.language_model.layers.0.self_attn.q_proj.weight"),
            Some("q_proj")
        );
        assert_eq!(
            projection_role("model.language_model.layers.47.mlp.down_proj.weight"),
            Some("down_proj")
        );
        assert_eq!(
            projection_role("model.language_model.layers.48.mlp.down_proj.weight"),
            None
        );
        assert_eq!(
            projection_role("model.language_model.layers.0.mlp.experts.0.down_proj.weight"),
            None
        );
        assert_eq!(
            projection_role("model.language_model.embed_tokens.weight"),
            None
        );
    }

    #[test]
    fn incomplete_header_fails_closed() {
        let header = serde_json::json!({"model.language_model.layers.0.self_attn.q_proj.weight": {"dtype":"BF16", "shape":[32,32]}});
        assert!(plan_from_header(&header).is_err());
    }
}
