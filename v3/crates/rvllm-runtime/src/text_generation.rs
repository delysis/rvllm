//! Text input and incremental output for local inference.

use sha2::{Digest, Sha256};
use std::path::Path;

/// Encode one user text turn using the qualified Gemma 4 non-thinking template.
/// This deliberately supports neither message histories nor tools/multimodal input.
/// A changed template must be reviewed instead of silently using stale formatting.
pub fn encode_gemma4_user_prompt(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    content: &str,
) -> Result<Vec<u32>, String> {
    let template = std::fs::read(model_dir.join("chat_template.jinja"))
        .map_err(|e| format!("read Gemma 4 chat template: {e}"))?;
    let digest = format!("{:x}", Sha256::digest(&template));
    if digest != "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4" {
        return Err(
            "this chat template has not been qualified for single-user Gemma 4 text input".into(),
        );
    }
    // Jinja uses Python str.strip(), which additionally treats U+001C..U+001F
    // as whitespace. Preserve every interior character exactly.
    let content =
        content.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
    if content.is_empty() {
        return Err("prompt must contain text".into());
    }
    let rendered =
        format!("<bos><|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    let encoded = tokenizer
        .encode(rendered, false)
        .map_err(|e| e.to_string())?;
    if encoded.get_ids().first().copied() != tokenizer.token_to_id("<bos>") {
        return Err("Gemma 4 tokenizer did not encode the template BOS".into());
    }
    Ok(encoded.get_ids().to_vec())
}

/// Retain tokenizer context across output tokens, including incomplete UTF-8.
#[derive(Default)]
pub struct IncrementalTextDecoder {
    retained_ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    read_index: usize,
}

impl IncrementalTextDecoder {
    pub fn step(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        token_id: u32,
    ) -> tokenizers::Result<Option<String>> {
        self.read_index = self.prefix_index;
        tokenizers::tokenizer::step_decode_stream(
            tokenizer,
            token_id,
            true,
            &mut self.retained_ids,
            &mut self.prefix,
            &mut self.prefix_index,
            &mut self.read_index,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RVLLM_GEMMA4_MODEL_DIR; reads tokenizer/template only, no accelerator access"]
    fn qualified_checkpoint_text_encoding_and_streaming() {
        let model = std::path::PathBuf::from(
            std::env::var_os("RVLLM_GEMMA4_MODEL_DIR").expect("checkpoint directory"),
        );
        let tokenizer = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).unwrap();
        let cases = [
            (
                "Reply with just the capital of France.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-capital.json"),
            ),
            (
                "Compute 17 × 19. Reply with only the integer.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-arithmetic.json"),
            ),
            (
                "Reply with only the result: 25 plus 17.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-addition.json"),
            ),
            (
                "Reply with one word: the opposite of hot.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-opposite.json"),
            ),
            (
                "What is the chemical symbol for gold? Reply with only its two letters.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-gold.json"),
            ),
            (
                "Return just the acronym for Central Processing Unit.",
                include_str!("../tests/reference/gemma4-12b-hf-chat-cpu.json"),
            ),
        ];
        for (text, fixture) in cases {
            let reference: serde_json::Value = serde_json::from_str(fixture).unwrap();
            let expected: Vec<u32> =
                serde_json::from_value(reference["prompt_token_ids"].clone()).unwrap();
            assert_eq!(
                encode_gemma4_user_prompt(&model, &tokenizer, text).unwrap(),
                expected
            );
            assert_eq!(
                encode_gemma4_user_prompt(&model, &tokenizer, &format!("\u{1c}\n {text}\t\u{1f}"))
                    .unwrap(),
                expected
            );
        }
        assert!(encode_gemma4_user_prompt(&model, &tokenizer, "\t\n").is_err());
        let unknown = tempfile::tempdir().unwrap();
        std::fs::write(
            unknown.path().join("chat_template.jinja"),
            "different template",
        )
        .unwrap();
        assert!(encode_gemma4_user_prompt(unknown.path(), &tokenizer, "hello").is_err());
        for text in [
            "A café ☕ — 東京 🦀 🫟𐍈\n".repeat(1),
            "Some words, punctuation, and spaces. ".repeat(100),
        ] {
            let ids = tokenizer.encode(text.clone(), false).unwrap();
            let expected = tokenizer.decode(ids.get_ids(), true).unwrap();
            assert_eq!(expected, text);
            let mut decoder = IncrementalTextDecoder::default();
            let mut emitted = String::new();
            for &token in ids.get_ids() {
                if let Some(chunk) = decoder.step(&tokenizer, token).unwrap() {
                    emitted.push_str(&chunk);
                }
            }
            let tail = expected
                .strip_prefix(&emitted)
                .expect("stream prefix matches complete decode");
            emitted.push_str(tail);
            assert_eq!(emitted, text);
        }
    }
}
