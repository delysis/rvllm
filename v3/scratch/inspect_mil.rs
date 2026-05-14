extern crate prost;
extern crate rvllm_apple_coreml_sys;
use prost::Message;
use rvllm_apple_coreml_sys::specification::{Model, model, mil_spec::value_type};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let model = Model::decode(&bytes[..]).unwrap();
    if let Some(model::Type::MlProgram(ref program)) = model.r#type {
        for (name, func) in &program.functions {
            println!("Function: {}", name);
            for (spec_name, spec) in &func.block_specializations {
                 println!("  Specialization: {}", spec_name);
                 for op in &spec.operations {
                     print!("    Op: {} ", op.r#type);
                     for output in &op.outputs {
                         if let Some(ref val_type) = output.r#type {
                             if let Some(value_type::Type::TensorType(ref tensor)) = val_type.r#type {
                                 print!("Output: {:?} ", tensor.dimensions);
                             }
                         }
                     }
                     println!();
                 }
            }
        }
    }
}
