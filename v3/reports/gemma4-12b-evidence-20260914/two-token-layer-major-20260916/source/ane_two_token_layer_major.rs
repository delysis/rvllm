//! Layer-major two-token reference using the existing S1 ANE requests.
//! Projection pairs are adjacent; attention remains causal and sequential.
#![forbid(unsafe_code)]

use super::*;
use crate::gemma_ane_decode::{
    add_residual_f16, decode_f16, rms_norm_f16_in_place, scale_layer_f16, AneDecodeTimes,
    HEAD_ROWS, HIDDEN,
};
use half::f16;
use std::os::unix::fs::FileExt;

fn project_pair(
    input: &[f16],
    output: &mut [f16],
    input_width: usize,
    output_width: usize,
    mut project: impl FnMut(&[f16], &mut [f16]) -> Result<(), String>,
) -> Result<(), String> {
    if input_width == 0
        || output_width == 0
        || input_width.checked_mul(2) != Some(input.len())
        || output_width.checked_mul(2) != Some(output.len())
    {
        return Err("two-token projection shape mismatch".into());
    }
    for (input, output) in input
        .chunks_exact(input_width)
        .zip(output.chunks_exact_mut(output_width))
    {
        project(input, output)?;
    }
    Ok(())
}

impl GemmaAneDecode {
    /// Diagnostic layer-major schedule. No S2 graph is loaded or evaluated.
    /// Uses the same fail-stop transaction and unwrapped capacity restriction
    /// as the serial reference. Timing fields are unavailable (all zero);
    /// journaled correctness execution must not be used as a timing trial.
    pub fn begin_two_token_layer_major_reference(
        &mut self,
        anchor: TokenId,
        draft: TokenId,
    ) -> Result<PendingTwoToken<'_>, String> {
        self.begin_layer_major_observed(anchor, draft, &mut |_, _, _| Ok(()))
    }

    fn begin_layer_major_observed(
        &mut self,
        anchor: TokenId,
        draft: TokenId,
        observer: &mut impl FnMut(usize, usize, &[f16]) -> Result<(), String>,
    ) -> Result<PendingTwoToken<'_>, String> {
        let positions = self.reference_positions(anchor, draft)?;
        self.next_position = None;
        let predictions = self.layer_major_pair([anchor, draft], positions, observer)?;
        Ok(PendingTwoToken {
            decoder: self,
            positions,
            draft,
            predictions,
        })
    }

    fn layer_major_pair(
        &mut self,
        tokens: [TokenId; 2],
        positions: Positions,
        observer: &mut impl FnMut(usize, usize, &[f16]) -> Result<(), String>,
    ) -> Result<[AneDecodedToken; 2], String> {
        // One owned token-major scratch allocation per tensor, bounded by the
        // already validated 12B geometries. Resizing below never grows capacity.
        let mut hidden = vec![f16::ZERO; 2 * HIDDEN];
        let mut normalized = vec![f16::ZERO; 2 * HIDDEN];
        let mut branch = vec![f16::ZERO; 2 * HIDDEN];
        let mut projected = Vec::with_capacity(2 * 8704);
        let mut value = vec![f16::ZERO; 2048];
        let mut attended = Vec::with_capacity(2 * 8192);
        let mut head_output = vec![f16::ZERO; 2 * HEAD_ROWS];
        for (token, hidden) in tokens.into_iter().zip(hidden.chunks_exact_mut(HIDDEN)) {
            let offset = self.embedding_info.file_offset + token.raw() as usize * HIDDEN * 2;
            self.embedding_file
                .read_exact_at(&mut self.embedding_bytes, offset as u64)
                .map_err(|e| e.to_string())?;
            decode_f16(&self.embedding_bytes, self.embedding_info.dtype, hidden)?;
        }
        scale_layer_f16(&mut hidden, f16::from_f32((HIDDEN as f32).sqrt()).to_f32())?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if layer.attention.tokens_seen() != positions.start {
                return Err("layer-major reference KV positions disagree".into());
            }
            let q_width = layer.shape.query_heads * layer.shape.head_dim;
            let kv_width = layer.shape.kv_heads * layer.shape.head_dim;
            let projected_width = q_width + kv_width * if layer.shared_value { 1 } else { 2 };
            projected.resize(2 * projected_width, f16::ZERO);
            attended.resize(2 * q_width, f16::ZERO);
            normalized.copy_from_slice(&hidden);
            rms_norm_f16_in_place(
                &mut normalized,
                HIDDEN,
                Some(&layer.input_norm),
                self.epsilon,
            )?;
            project_pair(
                &normalized,
                &mut projected,
                HIDDEN,
                projected_width,
                |input, output| layer.qkv.project(input, output),
            )?;

            for lane in 0..2 {
                let projection =
                    &mut projected[lane * projected_width..(lane + 1) * projected_width];
                let (query, kv) = projection.split_at_mut(q_width);
                let (key, projected_value) = kv.split_at_mut(kv_width);
                let value = &mut value[..kv_width];
                // Global V is raw K, copied before key normalization or RoPE.
                value.copy_from_slice(if layer.shared_value {
                    key
                } else {
                    projected_value
                });
                rms_norm_f16_in_place(
                    query,
                    layer.shape.head_dim,
                    Some(&layer.query_norm),
                    self.epsilon,
                )?;
                rms_norm_f16_in_place(
                    key,
                    layer.shape.head_dim,
                    Some(&layer.key_norm),
                    self.epsilon,
                )?;
                rms_norm_f16_in_place(value, layer.shape.head_dim, None, self.epsilon)?;
                let rope = if layer.shared_value {
                    &mut self.global_rope
                } else {
                    &mut self.sliding_rope
                };
                let position = positions.start + lane;
                rope.apply_f16(query, position as u32)?;
                rope.apply_f16(key, position as u32)?;
                // Lane zero cannot attend to the draft: its append/evaluate
                // completes before lane one's K/V is written to this request.
                layer.attention.decode(
                    query,
                    key,
                    value,
                    &mut attended[lane * q_width..(lane + 1) * q_width],
                )?;
            }

            project_pair(&attended, &mut branch, q_width, HIDDEN, |input, output| {
                layer.output.project(input, output)
            })?;
            rms_norm_f16_in_place(
                &mut branch,
                HIDDEN,
                Some(&layer.post_attention_norm),
                self.epsilon,
            )?;
            add_residual_f16(&mut hidden, &branch)?;
            normalized.copy_from_slice(&hidden);
            rms_norm_f16_in_place(
                &mut normalized,
                HIDDEN,
                Some(&layer.pre_ffn_norm),
                self.epsilon,
            )?;
            project_pair(&normalized, &mut branch, HIDDEN, HIDDEN, |input, output| {
                layer.ffn.project(input, output)
            })?;
            rms_norm_f16_in_place(
                &mut branch,
                HIDDEN,
                Some(&layer.post_ffn_norm),
                self.epsilon,
            )?;
            add_residual_f16(&mut hidden, &branch)?;
            scale_layer_f16(&mut hidden, layer.scalar)?;
            for (lane, hidden) in hidden.chunks_exact(HIDDEN).enumerate() {
                observer(lane, index, hidden)?;
            }
        }
        rms_norm_f16_in_place(&mut hidden, HIDDEN, Some(&self.final_norm), self.epsilon)?;
        let mut top_five = [[(u32::MAX, f32::NEG_INFINITY); 5]; 2];
        for (tile, head) in self.head.iter_mut().enumerate() {
            project_pair(
                &hidden,
                &mut head_output,
                HIDDEN,
                HEAD_ROWS,
                |input, output| head.project(input, output),
            )?;
            for (top, output) in top_five.iter_mut().zip(head_output.chunks_exact(HEAD_ROWS)) {
                for (row, value) in output.iter().enumerate() {
                    if !value.is_finite() {
                        return Err("layer-major vocabulary projection is nonfinite".into());
                    }
                    let divided = f16::from_f32(value.to_f32() / self.softcap);
                    let squashed = f16::from_f32(divided.to_f32().tanh());
                    let logit = f16::from_f32(squashed.to_f32() * self.softcap).to_f32();
                    if let Some(rank) = top.iter().position(|&(_, best)| logit > best) {
                        top.copy_within(rank..4, rank + 1);
                        top[rank] = ((tile * HEAD_ROWS + row) as u32, logit);
                    }
                }
            }
        }
        Ok(std::array::from_fn(|lane| AneDecodedToken {
            token: TokenId(top_five[lane][0].0),
            position: positions.start + lane,
            top_five: top_five[lane],
            times: AneDecodeTimes::default(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::live_tests::{assert_poisoned, load_snapshot, record, signature};
    use super::*;
    use crate::gemma_ane_decode::AneWeightPlan;
    use serde_json::json;
    use std::fs::File;
    use std::path::Path;

    #[test]
    fn paired_projection_validates_before_calls_and_keeps_lanes_independent() {
        let input = [
            f16::ONE,
            f16::from_f32(2.0),
            f16::from_f32(7.0),
            f16::from_f32(9.0),
        ];
        let mut output = [f16::ZERO; 2];
        let mut calls = 0;
        project_pair(&input, &mut output, 2, 1, |input, output| {
            calls += 1;
            output[0] = f16::from_f32(input[0].to_f32() + input[1].to_f32());
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(output.map(f16::to_f32), [3.0, 16.0]);
        for (input_width, output_width) in [(0, 1), (2, 0), (1, 1), (2, 2), (usize::MAX, 1)] {
            assert!(
                project_pair(&input, &mut output, input_width, output_width, |_, _| {
                    panic!("malformed pair must not reach a projection")
                })
                .is_err()
            );
        }
    }

    #[test]
    fn first_projection_failure_stops_before_second_lane() {
        let mut calls = 0;
        let result = project_pair(&[f16::ONE; 2], &mut [f16::ZERO; 2], 1, 1, |_, _| {
            calls += 1;
            Err("injected projection failure".into())
        });
        assert_eq!(calls, 1);
        assert_eq!(result.unwrap_err(), "injected projection failure");
    }

    #[test]
    #[ignore = "Gemma 4 12B layer-major S1 reference; 162 cache-only loads, no compilation or timing"]
    fn layer_major_matches_serial_layers_and_transaction_continuations() {
        assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
        let model = std::env::var("RVLLM_GEMMA4_MODEL_DIR").unwrap();
        let snapshot_path = std::env::var("RVLLM_TWO_TOKEN_SNAPSHOT").unwrap();
        let mut receipt =
            File::create_new(std::env::var("RVLLM_TWO_TOKEN_RECEIPT").unwrap()).unwrap();
        let (snapshot, anchor, snapshot_hash) = load_snapshot(Path::new(&snapshot_path));
        record(
            &mut receipt,
            json!({"stage":"fixture","snapshot_report_sha256":snapshot_hash,
            "prompt_tokens":84,"weight_plan":"static-int8-ffn-cached",
            "timing_claim":false,"reference_timing_fields_unavailable":true}),
        );
        let mut decoder = GemmaAneDecode::load_with_weight_plan(
            Path::new(&model),
            1024,
            AneWeightPlan::StaticInt8FfnCached,
        )
        .unwrap();
        assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
        decoder.import_prefill(&snapshot).unwrap();
        let mut states: [Vec<Vec<f16>>; 2] = std::array::from_fn(|_| Vec::new());
        let first = decoder
            .decode_with_observer(anchor, &mut |index, hidden| {
                assert_eq!(index, states[0].len());
                states[0].push(hidden.to_vec());
                Ok(())
            })
            .unwrap();
        let second = decoder
            .decode_with_observer(first.token, &mut |index, hidden| {
                assert_eq!(index, states[1].len());
                states[1].push(hidden.to_vec());
                Ok(())
            })
            .unwrap();
        let third = decoder.decode(second.token).unwrap();
        let expected = [signature(&first), signature(&second), signature(&third)];
        record(
            &mut receipt,
            json!({"stage":"serial","predictions":expected}),
        );

        decoder.import_prefill(&snapshot).unwrap();
        let mut comparisons = 0;
        let pending = decoder
            .begin_layer_major_observed(anchor, first.token, &mut |lane, index, hidden| {
                assert_eq!((index, lane), (comparisons / 2, comparisons % 2));
                assert_eq!(hidden.len(), HIDDEN);
                for (coordinate, (a, b)) in hidden.iter().zip(&states[lane][index]).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "lane {lane} layer {index} coordinate {coordinate}"
                    );
                }
                comparisons += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(comparisons, 2 * LAYERS);
        assert_eq!(signature(&pending.predictions()[0]), expected[0]);
        assert_eq!(signature(&pending.predictions()[1]), expected[1]);
        let accepted = pending.resolve(true).unwrap();
        assert!(accepted.draft_accepted);
        assert_eq!(accepted.next_position, 86);
        assert_eq!(
            accepted
                .predictions
                .iter()
                .map(signature)
                .collect::<Vec<_>>(),
            expected[..2]
        );
        assert_eq!(
            signature(&decoder.decode(second.token).unwrap()),
            expected[2]
        );
        record(
            &mut receipt,
            json!({"stage":"accepted-and-continuation","passed":true,
            "layer_bit_comparisons":comparisons,"fp16_values_compared":comparisons*HIDDEN}),
        );

        decoder.import_prefill(&snapshot).unwrap();
        let wrong = TokenId((first.token.raw() + 1) % VOCAB as u32);
        let rejected = decoder
            .begin_two_token_layer_major_reference(anchor, wrong)
            .unwrap()
            .resolve(true)
            .unwrap();
        assert!(!rejected.draft_accepted);
        assert_eq!(rejected.next_position, 85);
        assert_eq!(
            rejected
                .predictions
                .iter()
                .map(signature)
                .collect::<Vec<_>>(),
            expected[..1]
        );
        assert_eq!(
            signature(&decoder.decode(first.token).unwrap()),
            expected[1]
        );
        assert_eq!(
            signature(&decoder.decode(second.token).unwrap()),
            expected[2]
        );
        record(
            &mut receipt,
            json!({"stage":"rejected-replaced-and-continuation","passed":true}),
        );

        decoder.import_prefill(&snapshot).unwrap();
        let stopped = decoder
            .begin_two_token_layer_major_reference(anchor, first.token)
            .unwrap()
            .resolve(false)
            .unwrap();
        assert!(!stopped.draft_accepted);
        assert_eq!(stopped.next_position, 85);
        assert_eq!(
            stopped
                .predictions
                .iter()
                .map(signature)
                .collect::<Vec<_>>(),
            expected[..1]
        );
        assert_eq!(
            signature(&decoder.decode(first.token).unwrap()),
            expected[1]
        );
        record(
            &mut receipt,
            json!({"stage":"one-output-boundary","passed":true}),
        );

        decoder.import_prefill(&snapshot).unwrap();
        drop(
            decoder
                .begin_two_token_layer_major_reference(anchor, first.token)
                .unwrap(),
        );
        assert_poisoned(&mut decoder, anchor);
        record(
            &mut receipt,
            json!({"stage":"unresolved-drop-poisoned","passed":true}),
        );

        for fail_after in [1, 2, 2 * LAYERS - 1] {
            decoder.import_prefill(&snapshot).unwrap();
            let mut observed = 0;
            let result = decoder.begin_layer_major_observed(anchor, first.token, &mut |_, _, _| {
                observed += 1;
                if observed == fail_after {
                    Err("injected layer-major observer failure".into())
                } else {
                    Ok(())
                }
            });
            assert!(
                matches!(result,Err(error) if error == "injected layer-major observer failure")
            );
            assert_eq!(observed, fail_after);
            let completed_layers = (fail_after + 1) / 2;
            assert!(decoder.layers[..completed_layers]
                .iter()
                .all(|l| l.attention.tokens_seen() == 86));
            assert!(decoder.layers[completed_layers..]
                .iter()
                .all(|l| l.attention.tokens_seen() == 84));
            assert_poisoned(&mut decoder, anchor);
            record(
                &mut receipt,
                json!({"stage":"execution-error-poisoned","callbacks":fail_after,"passed":true}),
            );
        }
        decoder.import_prefill(&snapshot).unwrap();
        assert_eq!(signature(&decoder.decode(anchor).unwrap()), expected[0]);
        assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
        drop(decoder);
        record(
            &mut receipt,
            json!({"stage":"complete","compiler_calls":0,"reimport_recovered":true,
            "timing_claim":false,"claim":"Layer-major two-token schedule matches serial INT8 layer bits and committed greedy continuation at captured Metal position 84. Existing S1 graphs only; no batched arithmetic or drafter."}),
        );
    }
}
