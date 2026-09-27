//! Compare two sealed post-layer Metal route observations without rerunning a model.
use half::{bf16, f16};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, path::Path};

fn read_trace(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("missing {key}"))
}

fn u16_bits(summary: &Value) -> Result<Option<Vec<u16>>, String> {
    let Some(hex) = summary.get("raw_u16_hex").and_then(Value::as_str) else {
        return Ok(None);
    };
    let shape = required(summary, "shape")?
        .as_array()
        .ok_or("shape is not an array")?;
    let count = shape.iter().try_fold(1usize, |product, dim| {
        let dim = dim.as_u64().ok_or("invalid shape dimension")?;
        let dim = usize::try_from(dim).map_err(|_| "shape dimension overflow")?;
        product
            .checked_mul(dim)
            .ok_or("shape element count overflow")
    })?;
    if hex.len() != count.checked_mul(4).ok_or("hex length overflow")? {
        return Err("raw_u16_hex length does not match shape".to_owned());
    }
    let bits = hex
        .as_bytes()
        .chunks_exact(4)
        .map(|chunk| {
            let text = std::str::from_utf8(chunk).map_err(|_| "raw hex is not UTF-8")?;
            u16::from_str_radix(text, 16).map_err(|_| "invalid raw u16 hex")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut digest = Sha256::new();
    for bit in &bits {
        digest.update(bit.to_le_bytes());
    }
    let actual = format!("{:x}", digest.finalize());
    let expected = required(summary, "sha256_le_u16")?
        .as_str()
        .ok_or("digest is not a string")?;
    if actual != expected {
        return Err("raw bits do not match sealed SHA-256".to_owned());
    }
    Ok(Some(bits))
}

fn compare(on: &Value, off: &Value) -> Result<Value, String> {
    for key in [
        "schema",
        "observation_mode",
        "float_type",
        "phase",
        "layer",
        "num_tokens",
        "kv_cache_rows",
    ] {
        if required(on, key)? != required(off, key)? {
            return Err(format!("trace metadata differs at {key}"));
        }
    }
    if on["schema"] != "rvllm.gemma4_metal_layer_trace.v1"
        || on["observation_mode"] != "post_layer_existing_buffers_per_layer_sync"
    {
        return Err("inputs are not post-layer route observations".to_owned());
    }
    let decode_half: fn(u16) -> f32 = match on["float_type"].as_str() {
        Some("f16") => |bits| f16::from_bits(bits).to_f32(),
        Some("bf16") => |bits| bf16::from_bits(bits).to_f32(),
        _ => return Err("unsupported or missing float_type".to_owned()),
    };
    let on_fields = required(on, "summaries")?
        .as_object()
        .ok_or("on summaries are not an object")?;
    let off_fields = required(off, "summaries")?
        .as_object()
        .ok_or("off summaries are not an object")?;
    let keys: BTreeSet<&String> = on_fields.keys().collect();
    if keys != off_fields.keys().collect() {
        return Err("summary field sets differ".to_owned());
    }
    let mut fields = Map::new();
    for key in keys {
        let a = &on_fields[key];
        let b = &off_fields[key];
        if required(a, "shape")? != required(b, "shape")? {
            return Err(format!("{key}: shapes differ"));
        }
        let digest_equal = required(a, "sha256_le_u16")? == required(b, "sha256_le_u16")?;
        let raw = match (u16_bits(a)?, u16_bits(b)?) {
            (Some(a_bits), Some(b_bits)) => {
                if a_bits.len() != b_bits.len() {
                    return Err(format!("{key}: raw lengths differ"));
                }
                let mut first = None;
                let mut changed = 0usize;
                let mut max_abs_delta = 0.0f32;
                for (index, (&a_bit, &b_bit)) in a_bits.iter().zip(&b_bits).enumerate() {
                    if a_bit != b_bit {
                        changed += 1;
                        first.get_or_insert(json!({
                            "index": index,
                            "on_bits": format!("{a_bit:04x}"),
                            "off_bits": format!("{b_bit:04x}"),
                            "on_value": decode_half(a_bit),
                            "off_value": decode_half(b_bit)
                        }));
                        let delta = (decode_half(a_bit) - decode_half(b_bit)).abs();
                        if delta.is_finite() {
                            max_abs_delta = max_abs_delta.max(delta);
                        }
                    }
                }
                if digest_equal != (changed == 0) {
                    return Err(format!("{key}: digest/raw equality disagree"));
                }
                json!({"element_count":a_bits.len(), "changed_elements":changed,
                    "first_difference":first, "max_finite_abs_delta":max_abs_delta})
            }
            (None, None) => Value::Null,
            _ => return Err(format!("{key}: raw capture is missing in one arm")),
        };
        fields.insert(
            key.clone(),
            json!({"digest_equal":digest_equal, "raw_comparison":raw}),
        );
    }
    Ok(json!({"schema":"rvllm.metal_route_trace_comparison.v1",
        "phase":on["phase"], "layer":on["layer"],
        "float_type":on["float_type"],
        "num_tokens":on["num_tokens"], "kv_cache_rows":on["kv_cache_rows"],
        "fields":fields,
        "claim":"Post-layer stored tensor comparison, not a numerical reference or timing result."}))
}

fn main() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let on = args
        .next()
        .ok_or("usage: rvllm_metal_trace_compare ON.json OFF.json")?;
    let off = args
        .next()
        .ok_or("usage: rvllm_metal_trace_compare ON.json OFF.json")?;
    if args.next().is_some() {
        return Err("usage: rvllm_metal_trace_compare ON.json OFF.json".to_owned());
    }
    let output = compare(&read_trace(Path::new(&on))?, &read_trace(Path::new(&off))?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_raw_digest() {
        let summary = json!({"shape":[1,1], "raw_u16_hex":"3c00", "sha256_le_u16":"bad"});
        assert!(u16_bits(&summary).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn rejects_mismatched_metadata() {
        let mut a = json!({"schema":"rvllm.gemma4_metal_layer_trace.v1",
            "observation_mode":"post_layer_existing_buffers_per_layer_sync",
            "float_type":"bf16",
            "phase":"decode", "layer":5, "num_tokens":1, "kv_cache_rows":513,
            "summaries":{}});
        let b = a.clone();
        a["layer"] = json!(6);
        assert!(compare(&a, &b).unwrap_err().contains("layer"));
    }

    #[test]
    fn decodes_a_one_bit_bf16_difference_without_using_f16() {
        let trace = |bits: u16| {
            let digest = format!("{:x}", Sha256::digest(bits.to_le_bytes()));
            json!({"schema":"rvllm.gemma4_metal_layer_trace.v1",
                "observation_mode":"post_layer_existing_buffers_per_layer_sync",
                "float_type":"bf16", "phase":"decode", "layer":5,
                "num_tokens":1, "kv_cache_rows":513,
                "summaries":{"attention_output":{
                    "shape":[1,1], "raw_u16_hex":format!("{bits:04x}"),
                    "sha256_le_u16":digest}}})
        };
        let result = compare(&trace(0xb755), &trace(0xb754)).unwrap();
        let difference = &result["fields"]["attention_output"]["raw_comparison"];
        assert_eq!(difference["changed_elements"], 1);
        assert_eq!(
            difference["first_difference"]["on_value"],
            -0.000012695789337158203
        );
        assert_eq!(
            difference["first_difference"]["off_value"],
            -0.000012636184692382812
        );
        assert_eq!(difference["max_finite_abs_delta"], 5.960464477539063e-8);
    }
}
