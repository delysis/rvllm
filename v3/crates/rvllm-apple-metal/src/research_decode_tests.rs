use super::reference::*;
use super::*;
use crate::research_evidence::{ResearchDispatchSnapshot, ResearchKernel};
use crate::{MetalKernelOptions, MetalResearchCandidate as Candidate};
use rvllm_apple::{AppleLowBitTensorRole as Role, AppleLowBitWeightFormat as Format};

pub(crate) fn request() -> GateUpRequest {
    GateUpRequest {
        selected: Candidate::FfnBf16R4Sg2,
        dtype: Some(MetalFloatType::Bf16),
        decode: true,
        quantized_accumulation: false,
        capture_gate_up: false,
        has_low_bit_gate_or_up: false,
        model: Gemma12bResearchShape {
            tokens: 1,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: 1,
            head_dim: 512,
            attention_window: 0,
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        },
        offsets: [64, 7808, 235937472],
        arena_bytes: 235968256,
    }
}
#[test]
fn exact_shape_route_and_each_refusal_boundary() {
    let good = request();
    let plan = good.plan().unwrap();
    assert_eq!(plan.grid, [1920, 1, 1]);
    assert_eq!(plan.threads, [64, 1, 1]);
    assert_eq!(plan.params, [1, 3840, 15360]);
    for i in 0..24 {
        let mut bad = good;
        match i {
            0 => bad.selected = Candidate::Off,
            1 => bad.dtype = None,
            2 => bad.dtype = Some(MetalFloatType::F16),
            3 => bad.decode = false,
            4 => bad.quantized_accumulation = true,
            5 => bad.capture_gate_up = true,
            6 => bad.has_low_bit_gate_or_up = true,
            7 => bad.model.tokens = 2,
            8 => bad.model.hidden = 3839,
            9 => bad.model.intermediate = 15359,
            10 => bad.model.layers = 47,
            11 => bad.model.heads = 8,
            12 => bad.model.kv_heads = 2,
            13 => bad.model.head_dim = 256,
            14 => bad.model.attention_window = 1024,
            15 => bad.model.moe_experts = 1,
            16 => bad.model.moe_top_k = 1,
            17 => bad.model.moe_intermediate = 1,
            18 => bad.model.ple = 1,
            19 => bad.offsets[0] += 1,
            20 => bad.offsets[1] = usize::MAX - 2,
            21 => bad.offsets[2] = bad.offsets[0],
            22 => bad.offsets[2] = bad.offsets[1],
            _ => bad.arena_bytes = bad.offsets[2] + 30719,
        }
        assert!(bad.plan().is_none(), "refusal case {i}");
    }
    for _ in 0..1024 {
        assert_eq!(good.plan(), Some(plan));
    }
    let mut local = good;
    local.model.kv_heads = 8;
    local.model.head_dim = 256;
    local.model.attention_window = 1024;
    assert!(local.plan().is_some());
    assert_eq!(gate_up_encoder_count(true), 1);
    assert_eq!(gate_up_encoder_count(false), 2);
}
#[test]
fn qmv_role_format_shape_are_separate_requirements() {
    assert_eq!(ResearchKernel::QmvW4G32R4Sg8K8.limits(), (256, 0));
    assert_eq!(ResearchKernel::QmvW4G32R4Sg8K8.qmv_output_rows(), Some(32));
    assert_eq!(ResearchKernel::QmvW8G32R4Sg8K8.limits(), (256, 0));
    assert_eq!(ResearchKernel::QmvW8G32R4Sg8K8.qmv_output_rows(), Some(32));
    assert_eq!(ResearchKernel::QmvW4G32R8Sg2.qmv_output_rows(), Some(16));
    for (selector, format, role, k) in [
        (
            Candidate::QmvW4G32R8Sg2,
            Format::W4A16,
            Role::DenseDownProjection,
            15360,
        ),
        (
            Candidate::QmvW4G32R4Sg8K8,
            Format::W4A16,
            Role::DenseDownProjection,
            15360,
        ),
        (
            Candidate::QmvW8G32R8Sg2,
            Format::W8A16,
            Role::OutputProjection,
            4096,
        ),
        (
            Candidate::QmvW8G32R4Sg8K8,
            Format::W8A16,
            Role::OutputProjection,
            4096,
        ),
        (
            Candidate::QmvW8G32R8Sg2,
            Format::W8A16,
            Role::OutputProjection,
            8192,
        ),
        (
            Candidate::QmvW8G32R4Sg8K8,
            Format::W8A16,
            Role::OutputProjection,
            8192,
        ),
    ] {
        assert!(qmv_decode_contract(selector, format, role, 1, 3840, k));
        for (m, n, badk) in [(0, 3840, k), (2, 3840, k), (1, 3839, k), (1, 3840, k - 1)] {
            assert!(!qmv_decode_contract(selector, format, role, m, n, badk));
        }
        assert!(!qmv_decode_contract(
            Candidate::Off,
            format,
            role,
            1,
            3840,
            k
        ));
        assert!(!qmv_decode_contract(
            selector,
            format,
            Role::QueryProjection,
            1,
            3840,
            k
        ));
        let other = if format == Format::W4A16 {
            Format::W8A16
        } else {
            Format::W4A16
        };
        assert!(!qmv_decode_contract(selector, other, role, 1, 3840, k));
    }
}
#[test]
fn exact_ledger_rejects_missing_extra_reset_overflow() {
    let kernel = ResearchKernel::FfnBf16R4Sg2;
    let zero = ResearchDispatchSnapshot::default();
    assert!(zero.verify_exact(kernel, 0).is_ok());
    assert!(zero.verify_exact(kernel, 1).is_err());
    let mut after = zero;
    after.counts[kernel as usize] = 3;
    assert!(after
        .checked_since(zero)
        .unwrap()
        .verify_exact(kernel, 3)
        .is_ok());
    assert!(zero.checked_since(after).is_err());
    after.counts[ResearchKernel::QmvW4G32R8Sg2 as usize] = 1;
    assert!(after.verify_exact(kernel, 3).is_err());
    after.overflowed = true;
    assert!(after.verify_exact(kernel, 3).is_err());
}
#[test]
fn generated_qmv_source_retains_fp16_scale_abi_and_defaults() {
    let defaults = MetalKernelOptions::default();
    assert_eq!(defaults.research, Candidate::Off);
    let off = crate::kernels::kernel_source_with_options(MetalFloatType::Bf16, defaults);
    for c in [
        Candidate::FfnBf16R4Sg2,
        Candidate::QmvW4G32R8Sg2,
        Candidate::QmvW4G32R4Sg8K8,
        Candidate::QmvW8G32R8Sg2,
        Candidate::QmvW8G32R4Sg8K8,
        Candidate::GlobalD512ShortR4T128,
    ] {
        let source = crate::kernels::kernel_source_with_options(
            MetalFloatType::Bf16,
            MetalKernelOptions {
                research: c,
                ..defaults
            },
        );
        assert!(!off.contains(c.kernels()[0].name()));
        assert_eq!(
            source
                .matches(&format!("kernel void {}(", c.kernels()[0].name()))
                .count(),
            1
        );
        if matches!(
            c,
            Candidate::QmvW4G32R8Sg2
                | Candidate::QmvW4G32R4Sg8K8
                | Candidate::QmvW8G32R8Sg2
                | Candidate::QmvW8G32R4Sg8K8
        ) {
            assert!(source.contains("device const half *scales"));
            assert!(!source.contains("device const bfloat *scales"));
        }
    }
}
#[test]
fn independent_signed_group32_boundary_and_tail_oracles() {
    for bits in [4, 8] {
        for k in [1, 31, 32, 33, 63, 64, 65, 96] {
            let f = Group32Fixture::new(bits, 17, k);
            let a = f.output();
            assert_eq!(a, f.output());
            assert_eq!(a.len(), 17);
            assert!(a.iter().all(|&x| widen(x).is_finite()));
        }
    }
    // Negative endpoint and nibble order, followed by a distinct scale group.
    let mut f = Group32Fixture::new(4, 1, 33);
    f.values.fill(0);
    f.values[0] = 0x78;
    f.values[16] = 1;
    f.scales = vec![
        half::f16::from_f32(1.0).to_bits(),
        half::f16::from_f32(2.0).to_bits(),
    ];
    f.x.fill(bf16(1.0));
    assert_eq!(
        group32_fp64(4, 1, 33, &f.values, &f.scales, &f.x),
        vec![1.0]
    );
    let v = [0x80, 0x7f];
    let x = [bf16(1.0); 2];
    let s = [half::f16::from_f32(1.0).to_bits()];
    assert_eq!(group32_fp64(8, 1, 2, &v, &s, &x), vec![-1.0]);
}
#[test]
fn fusion_oracle_retains_intermediate_rounding_and_gelu_tails() {
    assert_eq!(activate(-6.0, 3.0), bf16(0.0));
    assert_eq!(activate(6.0, 3.0), bf16(18.0));
    assert_eq!(bf16(f64::NAN) & 0x7fc0, 0x7fc0);
    let x = [bf16(1.0), bf16(2.0)];
    let w = [bf16(1.0), bf16(2.0), bf16(-1.0), bf16(1.0)];
    assert_eq!(dense_gate_up(&x, &w, 1), vec![bf16(5.0)]);
    let values = sparse_gate_up();
    assert_eq!(values.len(), 15360);
    assert_eq!(values, sparse_gate_up());
    assert!(values.iter().any(|&x| widen(x) > 1.0));
    assert!(values.iter().any(|&x| widen(x) < -1.0));
}
