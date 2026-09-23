//! One deliberately bounded interpreter for the emitted host-test MIL subset.
//! It models graph wiring and explicit FP16 materialization, not ANE lowering.
#![forbid(unsafe_code)]
use super::*;
use std::collections::BTreeSet;

#[derive(Clone)]
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f16>,
}

fn count(shape: &[usize]) -> TestResult<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("empty tensor shape".into());
    }
    shape
        .iter()
        .try_fold(1_usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| "shape overflow".into())
}

fn call(expression: &str) -> TestResult<(&str, BTreeMap<&str, &str>)> {
    let (op, tail) = expression.split_once('(').ok_or("operation syntax")?;
    let args = tail.split_once(")[").ok_or("operation attributes")?.0;
    let mut depth = 0_usize;
    let mut start = 0;
    let mut result = BTreeMap::new();
    for (index, ch) in args
        .char_indices()
        .chain(std::iter::once((args.len(), ',')))
    {
        match ch {
            '(' => depth = depth.checked_add(1).ok_or("argument depth")?,
            ')' => depth = depth.checked_sub(1).ok_or("unbalanced arguments")?,
            ',' if depth == 0 => {
                let (key, value) = args[start..index]
                    .trim()
                    .split_once(" = ")
                    .ok_or("named argument")?;
                if key.is_empty() || value.is_empty() || result.insert(key, value).is_some() {
                    return Err("duplicate or empty argument".into());
                }
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("unbalanced arguments".into());
    }
    Ok((op, result))
}

fn exact_keys(args: &BTreeMap<&str, &str>, expected: &[&str]) -> TestResult {
    if args.len() != expected.len() || expected.iter().any(|name| !args.contains_key(name)) {
        return Err("unexpected or missing MIL argument".into());
    }
    Ok(())
}

fn integer_vector(expression: &str) -> TestResult<Vec<usize>> {
    expression
        .split_once(">([")
        .ok_or("integer vector data")?
        .1
        .split_once("])")
        .ok_or("integer vector end")?
        .0
        .split(',')
        .map(number)
        .collect()
}

pub(super) fn interpret(source: &Int8CandidateSource, input: &[f16]) -> TestResult<Vec<f16>> {
    let weights = decode_constants(&source.mil, &source.blob)?;
    let mut values = BTreeMap::<String, Tensor>::new();
    let mut ints = BTreeMap::<String, Vec<usize>>::new();
    let mut bools = BTreeMap::<String, bool>::new();
    let mut strings = BTreeMap::<String, String>::new();
    let mut seen = BTreeSet::<String>::new();
    let mut inputs = 0;
    let mut returns = 0;
    for line in source.mil.lines() {
        let line = line.trim();
        if line.starts_with("func main<") {
            if !line.ends_with("> x) {")
                || !line
                    .split_once(">(")
                    .ok_or("function input")?
                    .1
                    .starts_with("tensor<fp16,")
            {
                return Err("single-input fp16 declaration".into());
            }
            let shape = dimensions(line)?;
            if count(&shape)? != input.len() || inputs != 0 {
                return Err("input shape or count".into());
            }
            values.insert(
                "x".into(),
                Tensor {
                    shape,
                    data: input.to_vec(),
                },
            );
            seen.insert("x".into());
            inputs += 1;
            continue;
        }
        if line.starts_with("} ->") {
            if line != "} -> (y);" || returns != 0 {
                return Err("single-output declaration".into());
            }
            returns += 1;
            continue;
        }
        if !line.starts_with("tensor<")
            && !line.starts_with("fp16 ")
            && !line.starts_with("int32 ")
            && !line.starts_with("bool ")
            && !line.starts_with("string ")
        {
            if inputs == 1 && returns == 0 && line.contains(" = ") {
                return Err("unsupported MIL statement".into());
            }
            continue; // Program/build-info/braces, not executable statements.
        }
        let (left, expression) = line.split_once(" = ").ok_or("MIL assignment")?;
        let name = left.split_whitespace().last().ok_or("MIL name")?;
        if inputs != 1 || returns != 0 || !seen.insert(name.to_owned()) {
            return Err("duplicate MIL definition or statement outside function".into());
        }
        if expression.starts_with("constexpr_affine_dequantize()") {
            if !weights.contains_key(name) {
                return Err("constant descriptor lookup".into());
            }
            continue;
        }
        if expression.starts_with("const()[") {
            if let Some(tail) = expression.split_once("val = fp16(").map(|v| v.1) {
                if !left.starts_with("fp16 ") {
                    return Err("scalar fp16 declaration".into());
                }
                let scalar = tail
                    .split(')')
                    .next()
                    .ok_or("scalar")?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?;
                if !scalar.is_finite() || !f16::from_f32(scalar).is_finite() {
                    return Err("nonfinite fp16 scalar".into());
                }
                values.insert(
                    name.into(),
                    Tensor {
                        shape: vec![],
                        data: vec![f16::from_f32(scalar)],
                    },
                );
            } else if expression.contains("val = tensor<int32,") {
                if !left.starts_with("tensor<int32,") {
                    return Err("integer tensor declaration".into());
                }
                let v = integer_vector(expression)?;
                if count(&dimensions(left)?)? != v.len() {
                    return Err("integer constant shape".into());
                }
                ints.insert(name.into(), v);
            } else if let Some(tail) = expression.split_once("val = int32(").map(|v| v.1) {
                if !left.starts_with("int32 ") {
                    return Err("scalar int32 declaration".into());
                }
                ints.insert(
                    name.into(),
                    vec![number(tail.split(')').next().ok_or("integer")?)?],
                );
            } else if let Some(tail) = expression.split_once("val = bool(").map(|v| v.1) {
                if !left.starts_with("bool ") {
                    return Err("scalar bool declaration".into());
                }
                let value = tail.split(')').next().ok_or("bool")?;
                bools.insert(
                    name.into(),
                    match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err("bool constant".into()),
                    },
                );
            } else if let Some(tail) = expression.split_once("val = string(\"").map(|v| v.1) {
                if !left.starts_with("string ") {
                    return Err("scalar string declaration".into());
                }
                strings.insert(
                    name.into(),
                    tail.split_once("\")").ok_or("string constant")?.0.into(),
                );
            } else {
                return Err("unsupported MIL constant".into());
            }
            continue;
        }
        if !left.starts_with("tensor<fp16,") {
            return Err("materialized result must remain fp16".into());
        }
        let shape = dimensions(left)?;
        let expected_count = count(&shape)?;
        let (op, args) = call(expression)?;
        let get = |key: &str| -> TestResult<&Tensor> {
            values
                .get(*args.get(key).ok_or("argument lookup")?)
                .ok_or_else(|| "tensor lookup".into())
        };
        let int_arg = |key: &str| -> TestResult<&[usize]> {
            ints.get(*args.get(key).ok_or("integer argument")?)
                .map(Vec::as_slice)
                .ok_or_else(|| "integer lookup".into())
        };
        let data = match op {
            "conv" => {
                exact_keys(
                    &args,
                    &[
                        "dilations",
                        "groups",
                        "pad",
                        "pad_type",
                        "strides",
                        "weight",
                        "x",
                    ],
                )?;
                if int_arg("dilations")? != [1, 1]
                    || int_arg("groups")? != [1]
                    || int_arg("pad")? != [0, 0, 0, 0]
                    || int_arg("strides")? != [1, 1]
                    || strings
                        .get(*args.get("pad_type").ok_or("pad_type")?)
                        .map(String::as_str)
                        != Some("valid")
                {
                    return Err("convolution contract".into());
                }
                let w = weights
                    .get(*args.get("weight").ok_or("weight")?)
                    .ok_or("weight lookup")?;
                let x = get("x")?;
                if x.shape != [1, w.1, 1, 1] || shape != [1, w.0, 1, 1] {
                    return Err("projection tensor shape".into());
                }
                projection(&w.2, &x.data)
            }
            "reshape" => {
                exact_keys(&args, &["shape", "x"])?;
                let x = get("x")?;
                if int_arg("shape")? != shape.as_slice() || expected_count != x.data.len() {
                    return Err("reshape shape constant or element count".into());
                }
                x.data.clone()
            }
            "slice_by_size" => {
                exact_keys(&args, &["begin", "size", "x"])?;
                let x = get("x")?;
                let begin = int_arg("begin")?;
                let size = int_arg("size")?;
                if size != shape.as_slice()
                    || begin.len() != size.len()
                    || x.shape.len() != size.len()
                    || begin
                        .iter()
                        .zip(size)
                        .zip(&x.shape)
                        .any(|((&b, &n), &d)| b.checked_add(n).map_or(true, |end| end > d))
                {
                    return Err("slice bounds".into());
                }
                let mut output = Vec::with_capacity(expected_count);
                for linear in 0..expected_count {
                    let mut remaining = linear;
                    let mut offset = 0_usize;
                    let mut stride = 1_usize;
                    for dim in (0..size.len()).rev() {
                        let coordinate = remaining % size[dim];
                        remaining /= size[dim];
                        let index = begin[dim]
                            .checked_add(coordinate)
                            .and_then(|n| n.checked_mul(stride))
                            .ok_or("slice index overflow")?;
                        offset = offset.checked_add(index).ok_or("slice offset overflow")?;
                        stride = stride
                            .checked_mul(x.shape[dim])
                            .ok_or("slice stride overflow")?;
                    }
                    output.push(*x.data.get(offset).ok_or("slice storage bounds")?);
                }
                output
            }
            "concat" => {
                exact_keys(&args, &["axis", "interleave", "values"])?;
                if int_arg("axis")? != [1]
                    || bools.get(*args.get("interleave").ok_or("interleave")?) != Some(&false)
                    || shape.len() != 4
                    || shape[0] != 1
                    || shape[2..] != [1, 1]
                {
                    return Err("concat contract".into());
                }
                let names = args
                    .get("values")
                    .ok_or("concat inputs")?
                    .strip_prefix('(')
                    .and_then(|v| v.strip_suffix(')'))
                    .ok_or("concat tuple")?;
                let mut output = Vec::new();
                for input_name in names.split(',') {
                    let x = values
                        .get(input_name.trim())
                        .ok_or("concat tensor lookup")?;
                    if x.shape.len() != 4
                        || x.shape[0] != 1
                        || x.shape[2..] != [1, 1]
                        || output
                            .len()
                            .checked_add(x.data.len())
                            .map_or(true, |n| n > expected_count)
                    {
                        return Err("concat shape".into());
                    }
                    output.extend_from_slice(&x.data);
                }
                output
            }
            "mul" | "add" | "tanh" => {
                exact_keys(&args, if op == "tanh" { &["x"] } else { &["x", "y"] })?;
                let x = get("x")?;
                let y = if op == "tanh" { None } else { Some(get("y")?) };
                if x.shape != shape || y.is_some_and(|y| !y.shape.is_empty() && y.shape != shape) {
                    return Err("elementwise shape".into());
                }
                x.data
                    .iter()
                    .enumerate()
                    .map(|(index, x)| {
                        let a = x.to_f32();
                        let b = y
                            .map(|y| y.data[if y.shape.is_empty() { 0 } else { index }].to_f32())
                            .unwrap_or(0.0);
                        f16::from_f32(match op {
                            "mul" => a * b,
                            "add" => a + b,
                            _ => a.tanh(),
                        })
                    })
                    .collect()
            }
            _ => return Err(format!("unsupported MIL operation: {op}")),
        };
        if data.len() != expected_count {
            return Err("declared output size".into());
        }
        values.insert(name.into(), Tensor { shape, data });
    }
    if inputs != 1 || returns != 1 {
        return Err("function input/output count".into());
    }
    values
        .remove("y")
        .map(|y| y.data)
        .ok_or_else(|| "missing output".into())
}
