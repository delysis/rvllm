use rvllm_apple::{
    build_weight_blob_fp16_named, compile_private_ane_mil, dense_1x1_conv_mil, fused_ffn_mil,
    AneProcedure, FfnMilOffsets,
};
use rvllm_core::{AppleError, RvllmError};

fn require_private_ane_success(name: &str, result: rvllm_core::Result<()>) {
    if let Err(err) = result {
        match &err {
            RvllmError::Apple {
                err:
                    AppleError::FeatureNotAvailable { .. }
                    | AppleError::PrivateApiUnavailable { .. }
                    | AppleError::MilCompileFailed { .. },
                ..
            } => {
                panic!("{name} failed with typed Apple hardware error: {err}");
            }
            other => {
                panic!("{name} failed with unexpected non-Apple error: {other:?}");
            }
        }
    }
}

#[test]
#[ignore = "requires Apple Silicon private ANE hardware/compiler access"]
fn private_ane_dense_projection_compile_smoke() {
    let in_ch = 4;
    let out_ch = 3;
    let spatial = 2;
    let weights = vec![
        0.25, -0.50, 0.75, 1.00, -1.25, 1.50, -1.75, 2.00, 2.25, -2.50, 2.75, -3.00,
    ];
    let (weight_blob, descs) = build_weight_blob_fp16_named(&[("dense", &weights)]);
    let mil = dense_1x1_conv_mil(
        "dense_projection_smoke",
        in_ch,
        out_ch,
        spatial,
        descs[0].data_offset,
    );

    require_private_ane_success(
        "private_ane_dense_projection_compile_smoke",
        compile_private_ane_mil(
            &AneProcedure::DenseProjection {
                name: "dense_projection_smoke".to_owned(),
            },
            &mil,
            &weight_blob,
        ),
    );
}

#[test]
#[ignore = "requires Apple Silicon private ANE hardware/compiler access"]
fn private_ane_fused_ffn_compile_smoke() {
    let dim = 4;
    let hidden_dim = 8;
    let spatial = 2;
    let gate = vec![0.125f32; hidden_dim * dim];
    let up = vec![0.25f32; hidden_dim * dim];
    let down = vec![0.5f32; dim * hidden_dim];
    let (weight_blob, descs) =
        build_weight_blob_fp16_named(&[("gate", &gate), ("up", &up), ("down", &down)]);
    let offsets = FfnMilOffsets {
        gate: descs[0].data_offset,
        up: descs[1].data_offset,
        down: descs[2].data_offset,
    };
    let mil = fused_ffn_mil(dim, hidden_dim, spatial, offsets);

    require_private_ane_success(
        "private_ane_fused_ffn_compile_smoke",
        compile_private_ane_mil(&AneProcedure::FusedFfn { layer: 0 }, &mil, &weight_blob),
    );
}
