//! Seal human-authored Gemma 4 continuation targets before device trials.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

const SOURCE_SCHEMA: &str = "rvllm.gemma4_heldout_text.v1";
const OUTPUT_SCHEMA: &str = "rvllm.gemma4_heldout_tokens.v1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    schema: String,
    provenance: String,
    cases: Vec<TextCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextCase {
    id: String,
    prompt: String,
    continuation: String,
}

#[derive(Serialize)]
struct Output {
    schema: &'static str,
    source_sha256: Sha256Digest,
    tokenizer_sha256: Sha256Digest,
    provenance: String,
    cases: Vec<TokenCase>,
}

#[derive(Serialize)]
struct TokenCase {
    id: String,
    prompt: String,
    continuation: String,
    prompt_token_ids: Vec<u32>,
    target_token_ids: Vec<u32>,
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        eprintln!("usage: rvllm_gemma4_heldout_tokens TOKENIZER.json SOURCE.json");
        std::process::exit(2);
    }
    match run(Path::new(&args[1]), Path::new(&args[2])) {
        Ok(output) => println!("{}", serde_json::to_string_pretty(&output).unwrap()),
        Err(error) => {
            eprintln!("rvllm_gemma4_heldout_tokens: {error}");
            std::process::exit(1);
        }
    }
}

fn run(tokenizer_path: &Path, source_path: &Path) -> Result<Output, String> {
    let source_bytes = fs::read(source_path)
        .map_err(|error| format!("read {}: {error}", source_path.display()))?;
    let source: Source = parse_strict_json(&source_bytes)
        .map_err(|error| format!("parse {}: {error}", source_path.display()))?;
    if source.schema != SOURCE_SCHEMA || source.provenance.trim().is_empty() {
        return Err("invalid source schema or provenance".into());
    }
    if source.cases.is_empty() {
        return Err("at least one case is required".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|error| format!("load {}: {error}", tokenizer_path.display()))?;
    let mut seen = std::collections::BTreeSet::new();
    let mut cases = Vec::with_capacity(source.cases.len());
    for case in source.cases {
        if case.id.trim().is_empty()
            || !seen.insert(case.id.clone())
            || case.prompt.is_empty()
            || case.continuation.is_empty()
        {
            return Err(format!(
                "case {} has duplicate/empty identity or text",
                case.id
            ));
        }
        let prompt_ids = encode_with_bos(&tokenizer, &case.prompt)?;
        let full_text = format!("{}{}", case.prompt, case.continuation);
        let full_ids = encode_with_bos(&tokenizer, &full_text)?;
        let target_ids = target_suffix(&prompt_ids, &full_ids)?;
        if !(128..=1024).contains(&prompt_ids.len()) || !(1..=64).contains(&target_ids.len()) {
            return Err(format!(
                "case {} requires 128..=1024 prompt and 1..=64 target tokens, got {}/{}",
                case.id,
                prompt_ids.len(),
                target_ids.len()
            ));
        }
        cases.push(TokenCase {
            id: case.id,
            prompt: case.prompt,
            continuation: case.continuation,
            prompt_token_ids: prompt_ids,
            target_token_ids: target_ids,
        });
    }
    Ok(Output {
        schema: OUTPUT_SCHEMA,
        source_sha256: Sha256Digest::file(source_path).map_err(|error| error.to_string())?,
        tokenizer_sha256: Sha256Digest::file(tokenizer_path).map_err(|error| error.to_string())?,
        provenance: source.provenance,
        cases,
    })
}

fn encode_with_bos(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| format!("tokenize text: {error}"))?;
    let mut ids = Vec::with_capacity(encoded.len() + 1);
    ids.push(2);
    ids.extend_from_slice(encoded.get_ids());
    Ok(ids)
}

fn target_suffix(prompt: &[u32], full: &[u32]) -> Result<Vec<u32>, String> {
    full.strip_prefix(prompt)
        .filter(|suffix| !suffix.is_empty())
        .map(<[u32]>::to_vec)
        .ok_or_else(|| "continuation changed prompt tokenization or added no tokens".into())
}

#[cfg(test)]
mod tests {
    use super::target_suffix;

    #[test]
    fn stable_prefix_yields_only_new_targets() {
        assert_eq!(target_suffix(&[2, 4, 5], &[2, 4, 5, 6, 7]).unwrap(), [6, 7]);
    }

    #[test]
    fn rejects_boundary_retokenization() {
        assert!(target_suffix(&[2, 4, 5], &[2, 4, 8, 6]).is_err());
    }

    #[test]
    fn rejects_empty_continuation_tokens() {
        assert!(target_suffix(&[2, 4, 5], &[2, 4, 5]).is_err());
    }
}
