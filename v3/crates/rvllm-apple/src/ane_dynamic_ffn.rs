//! One compiled FFN graph with independently resident per-layer weight inputs.
//! Only the token activation is copied on a decode step. Compilation and
//! packing belong to initialization, never to the token loop.

use crate::ane_ffn_layout::PackedFfnLayout;
use half::f16;
use rvllm_apple_ane_sys::{AneInMemoryKernel, AneInMemoryProgram};
use std::time::{Duration, Instant};

/// Application stage durations, including any scheduler wait in each call.
#[derive(Debug)]
pub struct FfnStageTimes {
    pub input: Duration,
    pub evaluate: Duration,
    pub output: Duration,
}

pub struct AneDynamicFfnProgram {
    layout: PackedFfnLayout,
    program: AneInMemoryProgram,
}

impl AneDynamicFfnProgram {
    pub fn compile(hidden: usize, intermediate: usize) -> Result<Self, String> {
        let layout = PackedFfnLayout::new(hidden, intermediate)?;
        let program = AneInMemoryProgram::compile(
            &layout.mil(),
            &[],
            layout.input_bytes(),
            layout.output_bytes(),
        )?;
        Ok(Self { layout, program })
    }

    /// Gate/up are [intermediate, hidden], down is [hidden, intermediate].
    /// Pack and stage the weights once, then discard the staging allocation.
    /// The returned layer owns its request and keeps the graph loaded even
    /// after the original program handle is dropped.
    pub fn create_layer(
        &self,
        gate: &[f16],
        up: &[f16],
        down: &[f16],
    ) -> Result<AneDynamicFfn, String> {
        let weights = self.layout.pack_weights(gate, up, down)?;
        let mut kernel = self.program.create_request()?;
        kernel.write_input(&weights)?;
        Ok(AneDynamicFfn {
            kernel,
            layout: self.layout,
            input: vec![0; self.layout.hidden() * 2],
            output: vec![0; self.layout.output_bytes()],
        })
    }
}

pub struct AneDynamicFfn {
    kernel: AneInMemoryKernel,
    layout: PackedFfnLayout,
    input: Vec<u8>,
    output: Vec<u8>,
}

impl AneDynamicFfn {
    /// Evaluate one token without compiling, allocating, or rewriting weights.
    pub fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        self.stage_input(input, output.len())?;
        self.kernel.evaluate()?;
        self.collect_output(output)
    }

    pub fn project_profiled(
        &mut self,
        input: &[f16],
        output: &mut [f16],
    ) -> Result<FfnStageTimes, String> {
        let start = Instant::now();
        self.stage_input(input, output.len())?;
        let input_done = Instant::now();
        self.kernel.evaluate()?;
        let evaluation_done = Instant::now();
        self.collect_output(output)?;
        Ok(FfnStageTimes {
            input: input_done - start,
            evaluate: evaluation_done - input_done,
            output: evaluation_done.elapsed(),
        })
    }

    fn stage_input(&mut self, input: &[f16], output_len: usize) -> Result<(), String> {
        if input.len() != self.layout.hidden()
            || output_len != self.layout.hidden()
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("dynamic FFN requires one finite hidden-size input and output".into());
        }
        for (value, slot) in input.iter().zip(self.input.chunks_exact_mut(2)) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel
            .write_tensor_strided(0, 0, self.layout.row_bytes(), 2, &self.input)
    }

    fn collect_output(&mut self, output: &mut [f16]) -> Result<(), String> {
        self.kernel.read_output(&mut self.output)?;
        for (value, row) in output.iter_mut().zip(self.output.chunks_exact(64)) {
            *value = f16::from_le_bytes([row[0], row[1]]);
            if !value.is_finite() {
                return Err("dynamic FFN produced a nonfinite FP16 output".into());
            }
        }
        Ok(())
    }

    pub fn resident_input_bytes(&self) -> usize {
        self.layout.input_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(gate: &[f16], up: &[f16], down: &[f16], x: &[f16]) -> Vec<f16> {
        let dot = |w: &[f16], x: &[f16]| {
            f16::from_f32(w.iter().zip(x).map(|(a, b)| a.to_f32() * b.to_f32()).sum())
        };
        let gated: Vec<_> = gate
            .chunks_exact(x.len())
            .zip(up.chunks_exact(x.len()))
            .map(|(g, u)| {
                let g = dot(g, x).to_f32();
                let u = dot(u, x).to_f32();
                let activated = f16::from_f32(
                    0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
                );
                f16::from_f32(activated.to_f32() * u)
            })
            .collect();
        down.chunks_exact(gated.len())
            .map(|w| dot(w, &gated))
            .collect()
    }

    fn compare(actual: &[f16], expected: &[f16]) {
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for (a, e) in actual.iter().zip(expected) {
            assert!(a.is_finite());
            let error = a.to_f64() - e.to_f64();
            squared_error += error * error;
            squared_reference += e.to_f64() * e.to_f64();
            assert!(error.abs() < 0.005 + 0.02 * e.to_f64().abs());
        }
        assert!(
            squared_reference > 0.01,
            "fixture must exercise nonzero outputs"
        );
        let relative_error = (squared_error / squared_reference).sqrt();
        assert!(
            relative_error < 0.02,
            "relative_error={relative_error}, actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn invalid_dimensions_are_rejected_before_private_api_access() {
        assert!(AneDynamicFfnProgram::compile(0, 128).is_err());
        assert!(AneDynamicFfnProgram::compile(64, 65536).is_err());
    }

    #[test]
    #[ignore = "diagnostic only: four small ANE graphs expose FFN intermediates; not a numerical acceptance test"]
    fn hardware_reports_ffn_intermediate_error() {
        let (hidden, intermediate) = (64, 128);
        let count = hidden * intermediate;
        let gate: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 7 + 3) % 23) as f32 / 32.0 - 0.35))
            .collect();
        let up: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 13 + 7) % 29) as f32 / 32.0 - 0.45))
            .collect();
        let down: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 11 + 3) % 31) as f32 / 128.0 - 0.12))
            .collect();
        let input: Vec<_> = (0..hidden)
            .map(|i| f16::from_f32((i % 17) as f32 / 16.0 - 0.5))
            .collect();
        let project = |weights: &[f16]| -> Vec<f16> {
            weights
                .chunks_exact(hidden)
                .map(|row| {
                    f16::from_f32(
                        row.iter()
                            .zip(&input)
                            .map(|(a, b)| a.to_f32() * b.to_f32())
                            .sum(),
                    )
                })
                .collect()
        };
        let g = project(&gate);
        let u = project(&up);
        let a: Vec<_> = g
            .iter()
            .map(|v| {
                let v = v.to_f32();
                f16::from_f32(0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v * v * v)).tanh()))
            })
            .collect();
        let gated: Vec<_> = a
            .iter()
            .zip(&u)
            .map(|(a, u)| f16::from_f32(a.to_f32() * u.to_f32()))
            .collect();
        let layout = PackedFfnLayout::new(hidden, intermediate).unwrap();
        let mut packed = layout.pack_weights(&gate, &up, &down).unwrap();
        for (row, value) in packed.chunks_exact_mut(layout.row_bytes()).zip(&input) {
            row[..2].copy_from_slice(&value.to_le_bytes());
        }
        for (stage, expected) in [("gate", g), ("up", u), ("activated", a), ("gated", gated)] {
            let mil = layout.mil();
            let old = format!("tensor<fp16, [1, {hidden}, 1, 1]> y = reshape(x = result, shape = ro)[name = string(\"y\")];");
            assert_eq!(mil.matches(&old).count(), 1);
            let output = format!("tensor<int32, [4]> diagnostic_shape = const()[name = string(\"diagnostic_shape\"), val = tensor<int32, [4]>([1, {intermediate}, 1, 1])];\n        tensor<fp16, [1, {intermediate}, 1, 1]> y = reshape(x = {stage}, shape = diagnostic_shape)[name = string(\"y\")];");
            let mut kernel = AneInMemoryKernel::compile(
                &mil.replace(&old, &output),
                &[],
                layout.input_bytes(),
                intermediate * 64,
            )
            .unwrap();
            kernel.write_input(&packed).unwrap();
            kernel.evaluate().unwrap();
            let mut bytes = vec![0; intermediate * 64];
            kernel.read_output(&mut bytes).unwrap();
            let actual: Vec<_> = bytes
                .chunks_exact(64)
                .map(|row| f16::from_le_bytes([row[0], row[1]]).to_f32())
                .collect();
            assert!(actual.iter().all(|v| v.is_finite()));
            let expected: Vec<_> = expected.iter().map(|v| v.to_f32()).collect();
            let squared_error: f32 = actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e) * (a - e))
                .sum();
            let squared_reference: f32 = expected.iter().map(|e| e * e).sum();
            let max_absolute_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "{}",
                serde_json::json!({"stage":stage,"relative_error":(squared_error/squared_reference).sqrt(),"max_absolute_error":max_absolute_error,"actual":actual,"expected":expected})
            );
        }
    }

    #[test]
    #[ignore = "executes private ANE API; one graph, two weight sets, seven evaluations"]
    fn hardware_shared_program_keeps_weights_and_request_lifetimes_independent() {
        let (hidden, intermediate) = (64, 128);
        let count = hidden * intermediate;
        let gate: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 7 + 3) % 23) as f32 / 32.0 - 0.35))
            .collect();
        let up: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 13 + 7) % 29) as f32 / 32.0 - 0.45))
            .collect();
        let down: Vec<_> = (0..count)
            .map(|i| f16::from_f32(((i * 11 + 3) % 31) as f32 / 128.0 - 0.12))
            .collect();
        let negative_down: Vec<_> = down.iter().map(|w| -*w).collect();
        let inputs: Vec<Vec<_>> = (0..4)
            .map(|step| {
                (0..hidden)
                    .map(|i| f16::from_f32(((i + step * 3) % 17) as f32 / 16.0 - 0.5))
                    .collect()
            })
            .collect();
        let references: Vec<_> = inputs
            .iter()
            .map(|input| reference(&gate, &up, &down, input))
            .collect();
        let negative_references: Vec<_> = inputs
            .iter()
            .map(|input| reference(&gate, &up, &negative_down, input))
            .collect();
        // Ensure every fixture is finite and has enough signal to distinguish
        // the two resident weight sets before touching the private framework.
        for expected in references.iter().chain(&negative_references) {
            compare(expected, expected);
        }
        let program = AneDynamicFfnProgram::compile(hidden, intermediate).unwrap();
        let mut first = program.create_layer(&gate, &up, &down).unwrap();
        let mut second = program.create_layer(&gate, &up, &negative_down).unwrap();
        drop(program);
        let mut actual = vec![f16::ZERO; hidden];
        for step in 0..3 {
            first.project(&inputs[step], &mut actual).unwrap();
            compare(&actual, &references[step]);
            second.project(&inputs[step], &mut actual).unwrap();
            compare(&actual, &negative_references[step]);
        }
        drop(first);
        second.project(&inputs[3], &mut actual).unwrap();
        compare(&actual, &negative_references[3]);
    }
}
