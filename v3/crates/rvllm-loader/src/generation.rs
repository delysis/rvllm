//! Generation stop tokens belong to the checkpoint. BOS and newline token
//! numbers are not portable EOS defaults, even between Gemma releases.

use serde_json::Value;
use std::path::Path;

pub fn load_eos_token_ids(model_dir: &Path) -> Result<Vec<u32>, String> {
    load_eos_token_ids_from_files(
        Some(&model_dir.join("generation_config.json")),
        &model_dir.join("config.json"),
    )
}

/// Packaged callers pass only metadata paths authenticated by their manifest.
pub fn load_eos_token_ids_from_files(
    generation_config: Option<&Path>,
    model_config: &Path,
) -> Result<Vec<u32>, String> {
    if let Some(path) = generation_config {
        match std::fs::read(path) {
            Ok(bytes) => {
                let value: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                if let Some(ids) = parse_eos_token_ids(&value)? {
                    return Ok(ids);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("{}: {error}", path.display())),
        }
    }
    let bytes =
        std::fs::read(model_config).map_err(|e| format!("{}: {e}", model_config.display()))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", model_config.display()))?;
    if let Some(ids) = parse_eos_token_ids(&value)? {
        return Ok(ids);
    }
    Ok(parse_eos_token_ids(&value["text_config"])?.unwrap_or_default())
}

fn parse_eos_token_ids(config: &Value) -> Result<Option<Vec<u32>>, String> {
    let Some(value) = config.get("eos_token_id").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let values = match value {
        Value::Array(values) => values.as_slice(),
        value => std::slice::from_ref(value),
    };
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let id = value
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .ok_or("eos_token_id must contain unsigned 32-bit integer token IDs")?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(Some(ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eos_values_preserve_checkpoint_meaning() {
        assert_eq!(
            parse_eos_token_ids(&json!({"eos_token_id": [1, 106, 50, 106]})).unwrap(),
            Some(vec![1, 106, 50])
        );
        assert_eq!(
            parse_eos_token_ids(&json!({"eos_token_id": 106})).unwrap(),
            Some(vec![106])
        );
        assert_eq!(
            parse_eos_token_ids(&json!({"eos_token_id": []})).unwrap(),
            Some(vec![])
        );
        assert_eq!(parse_eos_token_ids(&json!({})).unwrap(), None);
        for value in [
            json!(-1),
            json!(4294967296_u64),
            json!("106"),
            json!([1, null]),
        ] {
            assert!(parse_eos_token_ids(&json!({"eos_token_id": value})).is_err());
        }
    }

    #[test]
    fn generation_config_overrides_model_and_undeclared_files_are_ignored() {
        let root = std::env::temp_dir().join(format!("rvllm-eos-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let generation = root.join("generation_config.json");
        let config = root.join("config.json");
        std::fs::write(
            &config,
            r#"{"eos_token_id":[1,106],"text_config":{"eos_token_id":1}}"#,
        )
        .unwrap();
        std::fs::write(&generation, r#"{"eos_token_id":[1,106,50]}"#).unwrap();
        assert_eq!(load_eos_token_ids(&root).unwrap(), vec![1, 106, 50]);
        assert_eq!(
            load_eos_token_ids_from_files(None, &config).unwrap(),
            vec![1, 106]
        );
        std::fs::remove_file(&generation).unwrap();
        assert_eq!(load_eos_token_ids(&root).unwrap(), vec![1, 106]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
