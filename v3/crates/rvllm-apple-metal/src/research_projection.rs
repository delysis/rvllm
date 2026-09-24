//! One checked launch description for all explicit research projections/norms.
//! This module never acquires a device, allocates scratch, or records dispatch.
#![forbid(unsafe_code)]

use crate::research::MetalResearchCandidate;
use crate::research::{buffer_span, disjoint_writes, matrix_bytes, projection_buffers_fit};
use crate::research_evidence::ResearchKernel;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackReason {
    NotThisOperation,
    ModelPhaseOrTrace,
    StoragePrecision,
    ProjectionScale,
    Shape,
    Alignment,
    BufferOrAlias,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionPlan {
    pub kernel: ResearchKernel,
    pub tile_m: usize,
    pub tile_n: usize,
}

/// Compile-time geometry of the vector-loaded, scratch-reusing tile family.
/// This describes source layout only; it is not a measured occupancy claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Load4Tile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub groups_m: usize,
    pub groups_n: usize,
}

pub const fn load4_tile(candidate: MetalResearchCandidate) -> Option<Load4Tile> {
    use MetalResearchCandidate::*;
    let [m, n, k, groups_m, groups_n] = match candidate {
        Load4M16N32K64 => [16, 32, 64, 1, 2],
        Load4M16N64K64 => [16, 64, 64, 1, 4],
        Load4M32N32K64 => [32, 32, 64, 2, 2],
        Load4M32N64K32 => [32, 64, 32, 2, 2],
        Load4M32N64K64 => [32, 64, 64, 2, 2],
        Load4M32N64K128 => [32, 64, 128, 2, 2],
        Load4M64N64K64 => [64, 64, 64, 2, 2],
        _ => return None,
    };
    Some(Load4Tile {
        m,
        n,
        k,
        groups_m,
        groups_n,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionRequest {
    pub candidate: MetalResearchCandidate,
    /// Supplied only by the incumbent full-model/prefill and trace predicate.
    /// Matrix dimensions alone do not establish the model/phase contract.
    pub full_prefill: bool,
    pub native_bf16: bool,
    pub alpha: f32,
    pub beta: f32,
    pub shape: [u32; 3],
    pub output_f32: bool,
    pub offsets: [usize; 3],
    pub arena_bytes: usize,
}

impl ProjectionRequest {
    pub fn plan(self) -> Result<ProjectionPlan, FallbackReason> {
        use MetalResearchCandidate::*;
        let (tile_m, tile_n) =
            projection_tile(self.candidate).ok_or(FallbackReason::NotThisOperation)?;
        let [gemm, qkv] = self.candidate.kernels() else {
            return Err(FallbackReason::NotThisOperation);
        };
        if !self.full_prefill {
            return Err(FallbackReason::ModelPhaseOrTrace);
        }
        if !self.native_bf16 {
            return Err(FallbackReason::StoragePrecision);
        }
        if self.alpha != 1.0 || self.beta != 0.0 {
            return Err(FallbackReason::ProjectionScale);
        }
        let [m, n, k] = self.shape;
        let spec = self.candidate.spec();
        if !(spec.min_tokens..=spec.max_tokens).contains(&m)
            || !crate::research_next::prefetch_projection_shape(m, n, k, self.output_f32)
        {
            return Err(FallbackReason::Shape);
        }
        let tile = load4_tile(self.candidate);
        if tile.is_some_and(|t| k as usize % t.k != 0 || n as usize % t.n != 0) {
            return Err(FallbackReason::Shape);
        }
        if (self.candidate == Mma32Load4 || tile.is_some())
            && (self.offsets[0] % 8 != 0 || self.offsets[1] % 8 != 0)
        {
            return Err(FallbackReason::Alignment);
        }
        if !projection_buffers_fit(
            self.offsets,
            self.shape,
            if self.output_f32 { 4 } else { 2 },
            self.arena_bytes,
        ) {
            return Err(FallbackReason::BufferOrAlias);
        }
        Ok(ProjectionPlan {
            kernel: if self.output_f32 { *qkv } else { *gemm },
            tile_m,
            tile_n,
        })
    }
}

/// Intrinsic output tile, not runtime shape admission. Component oracles use
/// this geometry with their own explicitly guarded synthetic tail fixtures.
pub fn projection_tile(candidate: MetalResearchCandidate) -> Option<(usize, usize)> {
    use MetalResearchCandidate::*;
    match candidate {
        ShortMma16x64 => Some((16, 64)),
        LongMma32x64 => Some((32, 64)),
        Mma32Prefetch | Mma32F32 | Mma32Load4 => Some((32, 32)),
        _ => load4_tile(candidate).map(|tile| (tile.m, tile.n)),
    }
}

/// Preserve exact in-place normalization, never partial aliases or a gamma
/// overlap. Explicit bounds are checked before a device can observe the plan.
pub fn postnorm_buffers_fit(
    offsets: [usize; 3],
    tokens: u32,
    hidden: u32,
    arena_bytes: usize,
) -> bool {
    if !(6..=1024).contains(&tokens) || hidden != 3840 {
        return false;
    }
    let checked = || {
        let bytes = matrix_bytes(tokens, hidden, 2)?;
        let input = buffer_span(offsets[0], bytes, arena_bytes)?;
        let output = buffer_span(offsets[1], bytes, arena_bytes)?;
        let gamma = buffer_span(offsets[2], matrix_bytes(1, hidden, 2)?, arena_bytes)?;
        let input_ok = input == output || disjoint_writes(&[input], &[output.clone()]);
        Some(input_ok && disjoint_writes(&[gamma], &[output]))
    };
    checked().unwrap_or(false)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PostnormPlan {
    pub kernel: ResearchKernel,
    pub binds_token_count: bool,
}

pub fn postnorm_plan(
    candidate: MetalResearchCandidate,
    full_prefill: bool,
    native_bf16: bool,
    offsets: [usize; 3],
    tokens: u32,
    hidden: u32,
    eps: f32,
    arena_bytes: usize,
) -> Result<PostnormPlan, FallbackReason> {
    let (kernel, binds_token_count) = match candidate {
        MetalResearchCandidate::RmsSimd32 => (ResearchKernel::Rms32, true),
        MetalResearchCandidate::RmsnormSimd256 => (ResearchKernel::Rms256, false),
        _ => return Err(FallbackReason::NotThisOperation),
    };
    if !full_prefill {
        return Err(FallbackReason::ModelPhaseOrTrace);
    }
    if !native_bf16 {
        return Err(FallbackReason::StoragePrecision);
    }
    if eps != 1.0e-6 || !(6..=1024).contains(&tokens) || hidden != 3840 {
        return Err(FallbackReason::Shape);
    }
    if !postnorm_buffers_fit(offsets, tokens, hidden, arena_bytes) {
        return Err(FallbackReason::BufferOrAlias);
    }
    Ok(PostnormPlan {
        kernel,
        binds_token_count,
    })
}

/// Component-fixture expectation for the *unchanged* role-specific prefetch
/// entry points. A refused dispatch is a negative control, not arithmetic
/// coverage. This helper exists only in test builds; runtime admission above
/// and the shader's early-return guards remain unchanged.
#[cfg(test)]
pub(crate) fn prefetch_fixture_expectation(
    kernel: &str,
    shape: [u32; 3],
) -> Result<bool, &'static str> {
    let output_f32 = match kernel {
        "research_gemm_mma32_prefetch" => false,
        "research_qkv_mma32_prefetch" => true,
        "qkv_project_f32_mma32" | "gemm_f16_mma32" => return Ok(true),
        _ => return Err("unknown kernel in prefetch component fixture"),
    };
    let [m, n, k] = shape;
    Ok(crate::research_next::prefetch_projection_shape(
        m, n, k, output_f32,
    ))
}

/// Inspect all payload and guard bytes. Numerical paths must write finite
/// results; expected refusals must leave *every* poison byte untouched. Caller
/// must save diagnostic bytes before propagating an error from this function.
#[cfg(test)]
pub(crate) fn validate_fixture_bytes(
    guarded: &[u8],
    rows: u32,
    columns: u32,
    output_f32: bool,
    numerical: bool,
) -> Result<(), String> {
    let width = if output_f32 { 4 } else { 2 };
    let bytes = matrix_bytes(rows, columns, width)
        .and_then(|n| n.checked_add(64))
        .ok_or("invalid fixture output size")?;
    if guarded.len() != bytes {
        return Err("fixture output byte count mismatch".into());
    }
    if guarded[..32]
        .iter()
        .chain(&guarded[bytes - 32..])
        .any(|&byte| byte != 0xa5)
    {
        return Err("fixture guard bytes changed".into());
    }
    let payload = &guarded[32..bytes - 32];
    if !numerical {
        return if payload.iter().all(|&byte| byte == 0xff) {
            Ok(())
        } else {
            Err("guard-refused dispatch modified its output".into())
        };
    }
    for (index, raw) in payload.chunks_exact(width).enumerate() {
        let value = if output_f32 {
            f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
        } else {
            half::bf16::from_le_bytes([raw[0], raw[1]]).to_f32()
        };
        if !value.is_finite() {
            return Err(format!(
                "non-finite numerical output at [{},{}], raw={raw:02x?}, entire_payload_untouched={}",
                index / columns as usize,
                index % columns as usize,
                payload.iter().all(|&byte| byte == 0xff),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        candidate: MetalResearchCandidate,
        m: u32,
        n: u32,
        k: u32,
        fp32: bool,
    ) -> ProjectionRequest {
        let a = m as usize * k as usize * 2;
        let b = n as usize * k as usize * 2;
        ProjectionRequest {
            candidate,
            full_prefill: true,
            native_bf16: true,
            alpha: 1.0,
            beta: 0.0,
            shape: [m, n, k],
            output_f32: fp32,
            offsets: [0, a, a + b],
            arena_bytes: a + b + m as usize * n as usize * if fp32 { 4 } else { 2 },
        }
    }

    #[test]
    fn old_projection_routes_retain_their_exact_shape_sets() {
        for candidate in [
            MetalResearchCandidate::ShortMma16x64,
            MetalResearchCandidate::Mma32Prefetch,
        ] {
            for m in [0, 1, 5, 6, 15, 16, 17, 32, 63, 64, 65, 84, 1024, 1025] {
                for (n, k, fp32) in [
                    (8192, 3840, true),
                    (9216, 3840, true),
                    (30720, 3840, false),
                    (3840, 4096, false),
                    (3840, 8192, false),
                    (3840, 15360, false),
                ] {
                    let expected = match candidate {
                        MetalResearchCandidate::ShortMma16x64 => {
                            crate::research::short_mma_shape(m, n, k, fp32)
                        }
                        _ => crate::research_next::prefetch_projection_shape(m, n, k, fp32),
                    };
                    assert_eq!(request(candidate, m, n, k, fp32).plan().is_ok(), expected);
                }
            }
        }
    }

    #[test]
    fn every_projection_requires_identity_precision_scale_and_nonaliasing() {
        for candidate in crate::research_catalog::ALL_CANDIDATES
            .into_iter()
            .filter(|&c| projection_tile(c).is_some())
        {
            let good = request(candidate, 64, 8192, 3840, true);
            let plan = good.plan().unwrap();
            assert_eq!(plan.kernel.owner(), candidate);
            assert_eq!(
                ProjectionRequest {
                    full_prefill: false,
                    ..good
                }
                .plan(),
                Err(FallbackReason::ModelPhaseOrTrace)
            );
            assert_eq!(
                ProjectionRequest {
                    native_bf16: false,
                    ..good
                }
                .plan(),
                Err(FallbackReason::StoragePrecision)
            );
            for alpha in [0.0, 0.5, f32::NAN, f32::INFINITY] {
                assert_eq!(
                    ProjectionRequest { alpha, ..good }.plan(),
                    Err(FallbackReason::ProjectionScale)
                );
            }
            assert_eq!(
                ProjectionRequest { beta: 1.0, ..good }.plan(),
                Err(FallbackReason::ProjectionScale)
            );
            assert!(ProjectionRequest {
                output_f32: false,
                ..good
            }
            .plan()
            .is_err());
            assert!(ProjectionRequest {
                arena_bytes: good.arena_bytes - 1,
                ..good
            }
            .plan()
            .is_err());
            assert!(ProjectionRequest {
                offsets: [0, good.offsets[1], 0],
                ..good
            }
            .plan()
            .is_err());
            assert!(ProjectionRequest {
                shape: [u32::MAX, 8192, 3840],
                ..good
            }
            .plan()
            .is_err());
        }
    }

    #[test]
    fn long_tile_and_vector_alignment_are_independent_admission_axes() {
        let long = request(MetalResearchCandidate::LongMma32x64, 63, 30720, 3840, false);
        assert_eq!(long.plan(), Err(FallbackReason::Shape));
        assert!(
            request(MetalResearchCandidate::Mma32Load4, 6, 30720, 3840, false)
                .plan()
                .is_ok()
        );
        let mut v = request(MetalResearchCandidate::Mma32Load4, 64, 8192, 3840, true);
        v.offsets[0] = 2;
        assert_eq!(v.plan(), Err(FallbackReason::Alignment));
        let long = request(MetalResearchCandidate::LongMma32x64, 84, 8192, 3840, true)
            .plan()
            .unwrap();
        assert_eq!(
            (long.tile_m, long.tile_n, long.kernel.limits()),
            (32, 64, (128, 14336))
        );
    }

    #[test]
    fn rms_alias_eps_and_full_model_gates_are_shared_by_both_reductions() {
        let bytes = 6 * 3840 * 2;
        let end = bytes + 3840 * 2;
        for candidate in [
            MetalResearchCandidate::RmsSimd32,
            MetalResearchCandidate::RmsnormSimd256,
        ] {
            assert!(
                postnorm_plan(candidate, true, true, [0, 0, bytes], 6, 3840, 1e-6, end).is_ok()
            );
            for offsets in [[0, 2, bytes], [0, 0, 0], [usize::MAX, 0, bytes]] {
                assert!(postnorm_plan(candidate, true, true, offsets, 6, 3840, 1e-6, end).is_err());
            }
            for eps in [0.0, -1.0, 1e-5, f32::NAN, f32::INFINITY] {
                assert!(
                    postnorm_plan(candidate, true, true, [0, 0, bytes], 6, 3840, eps, end).is_err()
                );
            }
            assert!(
                postnorm_plan(candidate, false, true, [0, 0, bytes], 6, 3840, 1e-6, end).is_err()
            );
            assert!(
                postnorm_plan(candidate, true, false, [0, 0, bytes], 6, 3840, 1e-6, end).is_err()
            );
        }
    }

    #[test]
    fn prefetch_synthetic_tail_is_a_refusal_not_a_numerical_success() {
        for name in [
            "research_gemm_mma32_prefetch",
            "research_qkv_mma32_prefetch",
        ] {
            assert_eq!(prefetch_fixture_expectation(name, [63, 67, 35]), Ok(false));
        }
        assert!(f32::from_bits(u32::MAX).is_nan());
        assert!(half::bf16::from_bits(u16::MAX).is_nan());
    }

    #[test]
    fn prefetch_production_roles_have_five_positive_and_seven_negative_arms() {
        let cases = [
            ([63, 67, 35], [false, false]),
            ([84, 8192, 3840], [false, true]),
            ([652, 9216, 3840], [false, true]),
            ([652, 30720, 3840], [true, false]),
            ([652, 3840, 8192], [true, false]),
            ([1024, 3840, 15360], [true, false]),
        ];
        let mut positive = [0, 0];
        for (shape, expected) in cases {
            for (index, name) in [
                "research_gemm_mma32_prefetch",
                "research_qkv_mma32_prefetch",
            ]
            .iter()
            .enumerate()
            {
                let actual = prefetch_fixture_expectation(name, shape).unwrap();
                assert_eq!(actual, expected[index]);
                positive[index] += usize::from(actual);
            }
        }
        assert_eq!(positive, [3, 2]);
        assert_eq!(12 - positive.iter().sum::<usize>(), 7);
    }

    #[test]
    fn refusal_validation_rejects_even_one_written_byte() {
        for output_f32 in [false, true] {
            let size = if output_f32 { 8 } else { 4 };
            let mut raw = vec![0xa5; 64 + size];
            raw[32..32 + size].fill(0xff);
            assert!(validate_fixture_bytes(&raw, 1, 2, output_f32, false).is_ok());
            for byte in 32..32 + size {
                raw[byte] = 0;
                assert!(validate_fixture_bytes(&raw, 1, 2, output_f32, false).is_err());
                raw[byte] = 0xff;
            }
        }
    }

    #[test]
    fn numerical_validation_never_accepts_untouched_or_partial_poison() {
        for output_f32 in [false, true] {
            let width = if output_f32 { 4 } else { 2 };
            let mut raw = vec![0xa5; 64 + 2 * width];
            raw[32..32 + 2 * width].fill(0xff);
            assert!(validate_fixture_bytes(&raw, 1, 2, output_f32, true).is_err());
            raw[32..32 + width].fill(0);
            let error = validate_fixture_bytes(&raw, 1, 2, output_f32, true).unwrap_err();
            assert!(error.contains("[0,1]"));
            assert!(error.contains("entire_payload_untouched=false"));
            raw[32..32 + 2 * width].fill(0);
            assert!(validate_fixture_bytes(&raw, 1, 2, output_f32, true).is_ok());
        }
    }

    #[test]
    fn fixture_size_and_canaries_are_checked_before_decoding() {
        assert!(validate_fixture_bytes(&[], 0, 1, true, true).is_err());
        assert!(validate_fixture_bytes(&[], u32::MAX, u32::MAX, true, true).is_err());
        let mut raw = vec![0xa5; 68];
        raw[32..36].fill(0);
        assert!(validate_fixture_bytes(&raw[..67], 1, 1, true, true).is_err());
        for byte in [0, 31, 36, 67] {
            raw[byte] = 0;
            assert!(validate_fixture_bytes(&raw, 1, 1, true, true).is_err());
            raw[byte] = 0xa5;
        }
    }

    #[test]
    fn prefetch_fixture_has_no_unknown_or_wrong_role_fallback() {
        assert!(prefetch_fixture_expectation("unreviewed", [84, 8192, 3840]).is_err());
        for m in [0, 1, 5, 1025, u32::MAX] {
            assert_eq!(
                prefetch_fixture_expectation("research_qkv_mma32_prefetch", [m, 8192, 3840]),
                Ok(false)
            );
        }
        for name in ["qkv_project_f32_mma32", "gemm_f16_mma32"] {
            assert_eq!(prefetch_fixture_expectation(name, [63, 67, 35]), Ok(true));
        }
    }

    #[test]
    fn load4_tiles_admit_exact_roles_and_mask_only_the_prompt_tail() {
        for candidate in crate::research_catalog::ALL_CANDIDATES {
            let Some(tile) = load4_tile(candidate) else {
                continue;
            };
            let spec = candidate.spec();
            for m in [0, 1, 5, 6, 15, 16, 17, 21, 31, 32, 33, 63, 64, 65, 84, 652, 1024, 1025] {
                for (n, k, fp32) in [
                    (8192, 3840, true),
                    (9216, 3840, true),
                    (30720, 3840, false),
                    (3840, 4096, false),
                    (3840, 8192, false),
                    (3840, 15360, false),
                ] {
                    assert_eq!(n as usize % tile.n, 0);
                    assert_eq!(k as usize % tile.k, 0);
                    assert_eq!(
                        request(candidate, m, n, k, fp32).plan().is_ok(),
                        (spec.min_tokens..=1024).contains(&m)
                    );
                    assert_eq!(
                        request(candidate, m, n, k, !fp32).plan(),
                        Err(FallbackReason::Shape)
                    );
                }
            }
            for (n, k) in [(67, 35), (8191, 3840), (8192, 3839)] {
                assert_eq!(
                    request(candidate, 84, n, k, true).plan(),
                    Err(FallbackReason::Shape)
                );
            }
        }
    }

    #[test]
    fn load4_tiles_check_each_vector_pointer_and_every_byte_of_the_output() {
        for candidate in crate::research_catalog::ALL_CANDIDATES {
            if load4_tile(candidate).is_none() {
                continue;
            }
            let good = request(candidate, 84, 8192, 3840, true);
            for input in 0..2 {
                for residue in [2, 4, 6] {
                    let mut bad = good;
                    bad.offsets[input] += residue;
                    assert_eq!(bad.plan(), Err(FallbackReason::Alignment));
                }
            }
            for offsets in [[0, good.offsets[1], 2], [0, good.offsets[1], good.offsets[1]]] {
                assert_eq!(
                    ProjectionRequest { offsets, ..good }.plan(),
                    Err(FallbackReason::BufferOrAlias)
                );
            }
            for beta in [f32::NAN, f32::INFINITY, -1.0] {
                assert_eq!(
                    ProjectionRequest { beta, ..good }.plan(),
                    Err(FallbackReason::ProjectionScale)
                );
            }
        }
    }

    #[test]
    fn load4_staging_and_fragment_ownership_are_bijective_with_bounded_scratch() {
        for candidate in crate::research_catalog::ALL_CANDIDATES {
            let Some(t) = load4_tile(candidate) else {
                continue;
            };
            let threads = 32 * t.groups_m * t.groups_n;
            let bytes = (2 * (t.m + t.n) * t.k).max(4 * t.m * t.n);
            for kernel in candidate.kernels() {
                assert_eq!(kernel.limits(), (threads, bytes));
                assert!(crate::research::launch_fits(32, threads, bytes, bytes, threads, bytes));
                assert!(!crate::research::launch_fits(32, threads, bytes, bytes - 1, threads, bytes));
            }
            for rows in [t.m, t.n] {
                let mut writes = vec![0_u8; rows * t.k];
                for tid in 0..threads {
                    for v in (tid..rows * t.k / 4).step_by(threads) {
                        let row = v / (t.k / 4);
                        let col = 4 * (v % (t.k / 4));
                        for lane in 0..4 {
                            writes[row * t.k + col + lane] += 1;
                        }
                    }
                }
                assert!(writes.iter().all(|&n| n == 1));
            }
            let mut writes = vec![0_u8; t.m * t.n];
            let sm = t.m / t.groups_m;
            let sn = t.n / t.groups_n;
            for sg in 0..t.groups_m * t.groups_n {
                for i in 0..sm / 8 {
                    for j in 0..sn / 8 {
                        for row in 0..8 {
                            for col in 0..8 {
                                let r = (sg / t.groups_n) * sm + i * 8 + row;
                                let c = (sg % t.groups_n) * sn + j * 8 + col;
                                writes[r * t.n + c] += 1;
                            }
                        }
                    }
                }
            }
            assert!(writes.iter().all(|&n| n == 1));
            for k in [3840, 4096, 8192, 15360] {
                let order: Vec<_> = (0..k)
                    .step_by(t.k)
                    .flat_map(|kb| (0..t.k).step_by(8).map(move |kk| kb + kk))
                    .collect();
                assert_eq!(order, (0..k).step_by(8).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn load4_export_preserves_native_operands_and_both_output_storage_contracts() {
        for candidate in crate::research_catalog::ALL_CANDIDATES {
            if load4_tile(candidate).is_none() {
                continue;
            }
            let source = crate::kernels::kernel_source_with_options(
                crate::MetalFloatType::Bf16,
                crate::MetalKernelOptions {
                    research: candidate,
                    ..crate::MetalKernelOptions::default()
                },
            );
            let extra = source.split("// load4-tiled:").nth(1).unwrap();
            assert!(!extra.contains("half"));
            assert!(extra.contains("simdgroup_matrix<bfloat, 8, 8>"));
            assert!(extra.contains("simdgroup_float8x8 acc[RM][RN]"));
            assert!(extra.contains("device const vec<bfloat, 4>"));
            assert!(extra.contains("else C[output] = bf16_sat(value)"));
            assert!(extra.contains("device float *C [[buffer(2)]]"));
            assert_eq!(extra.matches("kernel void ").count(), 2);
        }
    }
}
