use super::*;
use crate::MetalKernelOptions;
const ARENA: usize = 4 * 1024 * 1024 * 1024;
const OUT: usize = 3 * 1024 * 1024 * 1024;
fn owner(global: bool, tokens: u32, decode: bool) -> Policy {
    Policy {
        selected: MetalResearchCandidate::Donor12bSg8,
        dtype: Some(MetalFloatType::Bf16),
        quantized_accumulation: false,
        decode,
        model: Gemma12bResearchShape {
            tokens,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: if global { 1 } else { 8 },
            head_dim: if global { 512 } else { 256 },
            attention_window: if global { 0 } else { 1024 },
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        },
    }
}
fn w(role: Role, n: u32, k: u32, slot: usize) -> Weight {
    Weight {
        role,
        format: Format::W4A16,
        n,
        k,
        values: (slot + 1) * 256 * 1024 * 1024,
        scales: (slot + 1) * 256 * 1024 * 1024 + 128 * 1024 * 1024,
    }
}
fn projection(m: u32) -> ProjectionRequest {
    ProjectionRequest {
        selected: MetalResearchCandidate::Donor12bSg8,
        dtype: Some(MetalFloatType::Bf16),
        quantized_accumulation: false,
        shape: [m, 3840, 15360],
        activation: 0,
        native_weights: 256 * 1024 * 1024,
        low_bit: Some(w(Role::DenseDownProjection, 3840, 15360, 0)),
        output: OUT,
        output_stride: 3848,
        output_column: 4,
        output_f32: false,
        arena_bytes: ARENA,
    }
}
#[test]
fn donor_registry_is_unique_complete_and_off_by_default() {
    assert_eq!(
        MetalKernelOptions::default().research,
        MetalResearchCandidate::Off
    );
    let mut names = std::collections::HashSet::new();
    for candidate in [
        MetalResearchCandidate::Donor12bSg8,
        MetalResearchCandidate::Donor12bSg4,
    ] {
        assert_eq!(candidate.name().parse(), Ok(candidate));
        assert!(candidate.explicit_storage_abi());
        assert_eq!(candidate.kernels().len(), 12);
        for operation in [
            Operation::W4,
            Operation::W8,
            Operation::BatchW4,
            Operation::BatchW8,
            Operation::GateW4,
            Operation::GateW8,
            Operation::QkvW4,
            Operation::QkvW8,
            Operation::NativeGate,
            Operation::NativeProjection,
            Operation::LocalAttention,
            Operation::GlobalAttention,
        ] {
            let k = kernel(candidate, operation).unwrap();
            assert!(names.insert(k.name()));
            assert!(candidate.source().contains(k.name()));
            assert!(crate::research::launch_fits(
                32,
                1024,
                k.limits().1,
                32768,
                k.limits().0,
                k.limits().1
            ));
            assert!(!crate::research::launch_fits(
                16,
                1024,
                0,
                32768,
                k.limits().0,
                k.limits().1
            ));
        }
    }
    assert_eq!(names.len(), 24);
    assert!(kernel(MetalResearchCandidate::Off, Operation::W4).is_none());
}
#[test]
fn model_phase_dtype_and_near_misses_fail_closed() {
    for global in [false, true] {
        let p = owner(global, 1, true);
        assert!(p.allowed());
        assert!(!Policy { dtype: None, ..p }.allowed());
        assert!(!Policy {
            dtype: Some(MetalFloatType::F16),
            ..p
        }
        .allowed());
        assert!(!Policy {
            quantized_accumulation: true,
            ..p
        }
        .allowed());
        assert!(!Policy {
            selected: MetalResearchCandidate::Off,
            ..p
        }
        .allowed());
        for shape in [
            Gemma12bResearchShape {
                tokens: 2,
                ..p.model
            },
            Gemma12bResearchShape {
                hidden: 1536,
                ..p.model
            },
            Gemma12bResearchShape {
                layers: 35,
                ..p.model
            },
            Gemma12bResearchShape {
                intermediate: 6144,
                ..p.model
            },
            Gemma12bResearchShape {
                heads: 8,
                ..p.model
            },
            Gemma12bResearchShape { ple: 1, ..p.model },
            Gemma12bResearchShape {
                moe_experts: 8,
                ..p.model
            },
            Gemma12bResearchShape {
                head_dim: 128,
                ..p.model
            },
        ] {
            assert!(!Policy { model: shape, ..p }.allowed());
        }
    }
    for m in [1, 2, 7, 8, 9, 17, 127, 128] {
        assert!(owner(false, m, false).allowed());
    }
    assert!(!owner(false, 129, false).allowed());
    let p = owner(false, 1, true);
    assert!(!Policy {
        model: Gemma12bResearchShape {
            attention_window: 0,
            ..p.model
        },
        ..p
    }
    .allowed());
}
#[test]
fn projection_tiles_cover_token_and_row_tails_without_reinterpretation() {
    for candidate in [
        MetalResearchCandidate::Donor12bSg8,
        MetalResearchCandidate::Donor12bSg4,
    ] {
        for m in [1, 2, 7, 8, 9, 15, 17, 127, 128] {
            for format in [Format::W4A16, Format::W8A16] {
                let mut r = projection(m);
                r.selected = candidate;
                r.low_bit.as_mut().unwrap().format = format;
                let p = r.plan().unwrap();
                let rows = simdgroups(candidate).unwrap() * if m == 1 { 4 } else { 2 };
                assert_eq!(p.grid, [3840 / rows, (m as usize).div_ceil(8), 1]);
                assert_eq!(p.params, [m, 3840, 15360, 3848, 4, 0, 0, 0]);
                assert_eq!(p.buffer_count, 4);
                assert_eq!(p.threads()[0], simdgroups(candidate).unwrap() * 32);
            }
        }
    }
}
#[test]
fn projection_refuses_overflow_overlap_stride_role_and_dtype() {
    let r = projection(1);
    assert!(r.plan().is_some());
    for bad in [
        ProjectionRequest { output: 0, ..r },
        ProjectionRequest { activation: 1, ..r },
        ProjectionRequest {
            output: usize::MAX - 1,
            ..r
        },
        ProjectionRequest {
            arena_bytes: OUT + 1,
            ..r
        },
        ProjectionRequest {
            output_stride: 3843,
            ..r
        },
        ProjectionRequest {
            output_column: u32::MAX,
            ..r
        },
        ProjectionRequest {
            shape: [0, 3840, 15360],
            ..r
        },
        ProjectionRequest {
            shape: [129, 3840, 15360],
            ..r
        },
        ProjectionRequest {
            dtype: Some(MetalFloatType::F16),
            ..r
        },
        ProjectionRequest {
            quantized_accumulation: true,
            ..r
        },
    ] {
        assert!(bad.plan().is_none());
    }
    let mut bad = r;
    bad.low_bit.as_mut().unwrap().role = Role::OutputProjection;
    assert!(bad.plan().is_none());
    let mut bad = r;
    bad.low_bit.as_mut().unwrap().scales = bad.low_bit.unwrap().values;
    assert!(bad.plan().is_none());
}
#[test]
fn every_projection_role_has_explicit_12b_shapes() {
    for (role, n, k) in [
        (Role::QueryProjection, 4096, 3840),
        (Role::QueryProjection, 8192, 3840),
        (Role::KeyProjection, 512, 3840),
        (Role::ValueProjection, 2048, 3840),
        (Role::OutputProjection, 3840, 4096),
        (Role::OutputProjection, 3840, 8192),
        (Role::DenseGateProjection, 15360, 3840),
        (Role::DenseUpProjection, 15360, 3840),
        (Role::DenseDownProjection, 3840, 15360),
        (Role::LmHead, 262144, 3840),
    ] {
        assert!(projection_role_matches(role, n, k));
        assert!(!projection_role_matches(role, n + 4, k));
        assert!(!projection_role_matches(role, n, k + 256));
    }
}
#[test]
fn native_qkv_retains_true_f32_storage_and_checks_its_full_span() {
    let r = ProjectionRequest {
        shape: [9, 9216, 3840],
        low_bit: None,
        output_stride: 9216,
        output_column: 0,
        output_f32: true,
        ..projection(9)
    };
    let p = r.plan().unwrap();
    assert_eq!(p.params[5], 1);
    assert_eq!(p.buffer_count, 3);
    let end = OUT + 9 * 9216 * 4;
    assert!(ProjectionRequest {
        arena_bytes: end,
        ..r
    }
    .plan()
    .is_some());
    assert!(ProjectionRequest {
        arena_bytes: end - 1,
        ..r
    }
    .plan()
    .is_none());
    assert!(ProjectionRequest {
        output: OUT + 2,
        ..r
    }
    .plan()
    .is_none());
}
#[test]
fn gate_requires_same_format_role_and_materialized_trace_fallback() {
    let r = GateRequest {
        policy: owner(false, 1, true),
        capture_gate_up: false,
        activation: 0,
        native_weights: 256 * 1024 * 1024,
        low_bit: None,
        output: OUT,
        arena_bytes: ARENA,
    };
    assert_eq!(
        r.plan().unwrap().kernel,
        kernel(r.policy.selected, Operation::NativeGate).unwrap()
    );
    assert!(GateRequest {
        capture_gate_up: true,
        ..r
    }
    .plan()
    .is_none());
    assert!(GateRequest {
        policy: owner(false, 1, false),
        ..r
    }
    .plan()
    .is_none());
    assert!(GateRequest { output: 0, ..r }.plan().is_none());
    let g = w(Role::DenseGateProjection, 15360, 3840, 0);
    let u = w(Role::DenseUpProjection, 15360, 3840, 1);
    let low = GateRequest {
        low_bit: Some([g, u]),
        ..r
    };
    assert!(low.plan().is_some());
    assert!(GateRequest {
        low_bit: Some([
            g,
            Weight {
                format: Format::W8A16,
                ..u
            }
        ]),
        ..r
    }
    .plan()
    .is_none());
    assert!(GateRequest {
        low_bit: Some([
            g,
            Weight {
                role: Role::DenseGateProjection,
                ..u
            }
        ]),
        ..r
    }
    .plan()
    .is_none());
}
#[test]
fn global_raw_k_reuse_requires_actual_byte_identity_not_shape() {
    let q = w(Role::QueryProjection, 8192, 3840, 0);
    let k = w(Role::KeyProjection, 512, 3840, 1);
    let v = Weight {
        role: Role::ValueProjection,
        ..k
    };
    let r = QkvRequest {
        policy: owner(true, 1, true),
        capture_projection: false,
        skip_kv: false,
        activation: 0,
        weights: [q, k, v],
        output: OUT,
        arena_bytes: ARENA,
    };
    let p = r.plan().unwrap();
    assert_eq!(p.params[6], 1);
    assert_eq!(p.grid[0], 8704 / 32);
    let different = QkvRequest {
        weights: [q, k, w(Role::ValueProjection, 512, 3840, 2)],
        ..r
    }
    .plan()
    .unwrap();
    assert_eq!(different.params[6], 0);
    assert_eq!(different.grid[0], 9216 / 32);
    assert!(QkvRequest { skip_kv: true, ..r }.plan().is_none());
    assert!(QkvRequest {
        capture_projection: true,
        ..r
    }
    .plan()
    .is_none());
    assert!(QkvRequest {
        output: k.values,
        ..r
    }
    .plan()
    .is_none());
}
#[test]
fn local_qkv_does_not_inherit_e2b_shared_kv_heads() {
    let r = QkvRequest {
        policy: owner(false, 1, true),
        capture_projection: false,
        skip_kv: false,
        activation: 0,
        weights: [
            w(Role::QueryProjection, 4096, 3840, 0),
            w(Role::KeyProjection, 2048, 3840, 1),
            w(Role::ValueProjection, 2048, 3840, 2),
        ],
        output: OUT,
        arena_bytes: ARENA,
    };
    let p = r.plan().unwrap();
    assert_eq!(&p.params[1..3], &[4096, 2048]);
    assert_eq!(p.grid[0], 8192 / 32);
    assert_eq!(p.params[6], 0);
}
#[test]
fn paged_attention_checks_actual_cache_spans_and_resources() {
    for global in [false, true] {
        let r = AttentionRequest {
            policy: owner(global, 1, true),
            offsets: [
                0,
                256 * 1024 * 1024,
                512 * 1024 * 1024,
                OUT,
                8192 * 1024,
                8192 * 1024 + 4096,
                8192 * 1024 + 4100,
            ],
            block_size: 16,
            max_blocks: 128,
            num_blocks: 128,
            scale: 1.0,
            arena_bytes: ARENA,
        };
        let p = r.plan().unwrap();
        assert_eq!(p.grid, [16, 1, 1]);
        assert_eq!(p.params[3], if global { 0 } else { 1024 });
        assert_eq!(p.threads()[0], if global { 256 } else { 512 });
        assert!(AttentionRequest {
            scale: 1.0 / 16.0,
            ..r
        }
        .plan()
        .is_none());
        assert!(AttentionRequest { block_size: 0, ..r }.plan().is_none());
        assert!(AttentionRequest {
            max_blocks: u32::MAX,
            ..r
        }
        .plan()
        .is_none());
        let mut bad = r;
        bad.offsets[3] = bad.offsets[1];
        assert!(bad.plan().is_none());
    }
}
#[test]
fn source_keeps_explicit_scale_type_and_bf16_rounding_boundary() {
    let source = MetalResearchCandidate::Donor12bSg8.source();
    assert!(source.contains("device const half *sc"));
    assert!(source.contains("d12_load(d12_round"));
    assert!(source.contains("k0 / 32"));
    assert!(source.contains("head/(16/KV)"));
    for dtype in [MetalFloatType::F16, MetalFloatType::Bf16] {
        let source = crate::kernels::kernel_source_with_options(
            dtype,
            MetalKernelOptions {
                research: MetalResearchCandidate::Donor12bSg8,
                ..MetalKernelOptions::default()
            },
        );
        assert!(source.contains("device const half *sc"));
        assert!(source.ends_with(MetalResearchCandidate::Donor12bSg8.source()));
    }
}

#[test]
fn bf16_package_admission_requires_explicit_donor_and_fp32_accumulation() {
    let off = MetalKernelOptions::default();
    assert!(!bf16_sidecars_allowed(off));
    let on = MetalKernelOptions {
        research: MetalResearchCandidate::Donor12bSg8,
        ..off
    };
    assert!(bf16_sidecars_allowed(on));
    assert!(!bf16_sidecars_allowed(MetalKernelOptions {
        quantized_bf16_accumulation: true,
        ..on
    }));
}
