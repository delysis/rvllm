//! Host-only admission and storage contracts for the second candidate batch.
#![forbid(unsafe_code)]

/// Same narrow 12B projection roles as the incumbent MMA32, not a generic GEMM.
/// Callers must additionally carry full-model/prefill eligibility and queried
/// device/PSO limits; dimensions alone are not model identity.
pub fn prefetch_projection_shape(m: u32, n: u32, k: u32, f32_output: bool) -> bool {
    (6..=1024).contains(&m)
        && if f32_output {
            k == 3840 && matches!(n, 8192 | 9216)
        } else {
            (n == 30720 && k == 3840) || (n == 3840 && matches!(k, 4096 | 8192 | 15360))
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalFloatType, MetalKernelOptions, MetalResearchCandidate};

    #[test]
    fn prefetch_roles_and_boundaries_are_explicit() {
        for m in [6, 21, 64, 84, 256, 1024] {
            for (n, k, f32_output) in [
                (8192, 3840, true),
                (9216, 3840, true),
                (30720, 3840, false),
                (3840, 4096, false),
                (3840, 8192, false),
                (3840, 15360, false),
            ] {
                assert!(prefetch_projection_shape(m, n, k, f32_output));
                assert!(!prefetch_projection_shape(m, n + 1, k, f32_output));
                assert!(!prefetch_projection_shape(m, n, k + 1, f32_output));
            }
        }
        for m in [0, 1, 5, 1025, u32::MAX] {
            assert!(!prefetch_projection_shape(m, 8192, 3840, true));
        }
        assert!(!prefetch_projection_shape(64, 262144, 3840, false));
        assert!(!prefetch_projection_shape(64, 8192, 3840, false));
        assert!(!prefetch_projection_shape(64, 30720, 3840, true));
    }

    #[test]
    fn cooperative_register_prefetch_covers_a_tile_once_with_no_last_k_read() {
        let mut written = [0_u8; 1024];
        for tid in 0..128 {
            for i in 0..8 {
                written[tid + i * 128] += 1;
            }
        }
        assert!(written.into_iter().all(|n| n == 1));
        for k in [3840, 4096, 8192, 15360] {
            let mut staged = vec![0_u8; k];
            for d in 0..32 {
                staged[d] += 1;
            }
            for kb in (0..k).step_by(32) {
                if kb + 32 < k {
                    for d in 0..32 {
                        staged[kb + 32 + d] += 1;
                    }
                }
            }
            assert!(staged.into_iter().all(|n| n == 1));
        }
        for m in [6_usize, 21, 64, 84, 1024] {
            let covered: Vec<_> = (0..m.div_ceil(32))
                .flat_map(|tile| {
                    (0..32)
                        .map(move |row| tile * 32 + row)
                        .filter(move |&r| r < m)
                })
                .collect();
            assert_eq!(covered, (0..m).collect::<Vec<_>>());
        }
    }

    #[test]
    fn prefetch_source_keeps_dtype_rounding_and_metal31_vector_grid_abi() {
        for dtype in [MetalFloatType::F16, MetalFloatType::Bf16] {
            let source = crate::kernels::kernel_source_with_options(
                dtype,
                MetalKernelOptions {
                    research: MetalResearchCandidate::Mma32Prefetch,
                    ..MetalKernelOptions::default()
                },
            );
            assert!(source.contains("kernel void research_qkv_mma32_prefetch"));
            assert!(source.contains("device float *C [[buffer(2)]]"));
            assert!(source.contains("uint3 threads [[threads_per_threadgroup]]"));
            assert!(source.contains("if (more)"));
            assert!(source.contains("next_a[8], next_b[8]"));
        }
    }
}

/// Eight 512-wide or sixteen 256-wide K/V rows fit within a 17 KiB bound.
/// This is a source budget, not a claim about driver allocation or occupancy.
pub fn temporal_kv_rows(head_dim: u32) -> Option<usize> {
    match head_dim {
        256 => Some(16),
        512 => Some(8),
        _ => None,
    }
}

pub fn temporal_threadgroup_bytes(head_dim: u32) -> Option<usize> {
    temporal_kv_rows(head_dim)?.checked_mul(4 * head_dim as usize + 4)
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    #[test]
    fn query_tiles_cover_active_rows_exactly_and_every_simd_group_reaches_barriers() {
        for queries in [64_usize, 65, 84, 255, 256, 1023, 1024] {
            let mut visits = vec![0; queries];
            let mut inactive = 0;
            for group in 0..queries.div_ceil(4) {
                for simd in 0..4 {
                    let row = group * 4 + simd;
                    if row < queries {
                        visits[row] += 1;
                    } else {
                        inactive += 1;
                    }
                }
            }
            assert!(visits.into_iter().all(|n| n == 1));
            assert_eq!(inactive, queries.div_ceil(4) * 4 - queries);
        }
        assert_eq!(temporal_threadgroup_bytes(256), Some(16_448));
        assert_eq!(temporal_threadgroup_bytes(512), Some(16_416));
        assert_eq!(temporal_threadgroup_bytes(128), None);
    }

    #[test]
    fn temporal_shape_and_generated_sources_keep_global_and_sliding_distinct() {
        use crate::{MetalFloatType, MetalKernelOptions, MetalResearchCandidate};
        let source = crate::kernels::kernel_source_with_options(
            MetalFloatType::Bf16,
            MetalKernelOptions {
                research: MetalResearchCandidate::AttentionQ4,
                ..MetalKernelOptions::default()
            },
        );
        assert!(source.contains("research_attn_q4_body<256, 16>"));
        assert!(source.contains("research_attn_q4_body<512, 8>"));
        assert!(source.contains("float maximum = -INFINITY, denominator = 0.0f"));
        assert!(source.contains("kv_head = group.y / (heads / kv_heads)"));
        assert!(source.contains("scale != 1.0f"));
        assert!(source.contains("poisoned || !isfinite(value) ? bfloat(NAN) : bf16_sat(value)"));
        assert!(!source.contains("half query_lane"));
        // Static return sites are before the first barrier. Per-query mask
        // branches contain no barrier and leave inactive SIMD groups alive.
        let body = source
            .split("static inline void research_attn_q4_body")
            .nth(1)
            .unwrap();
        let main_body = body
            .split("kernel void research_attn_q4_d256")
            .next()
            .unwrap();
        assert!(!main_body
            .split("threadgroup_barrier")
            .skip(1)
            .any(|s| s.contains("return;")));
    }
}

/// [input, output, real gamma]. Only exact in-place or disjoint output is
/// admissible; partial row overlap can corrupt another lane's unread values.
pub fn rms32_buffers_fit(offsets: [usize; 3], tokens: u32, hidden: u32, arena: usize) -> bool {
    if hidden != 3840 || !(6..=1024).contains(&tokens) {
        return false;
    }
    let Some(bytes) = crate::research::matrix_bytes(tokens, hidden, 2) else {
        return false;
    };
    let Some(input) = crate::research::buffer_span(offsets[0], bytes, arena) else {
        return false;
    };
    let Some(output) = crate::research::buffer_span(offsets[1], bytes, arena) else {
        return false;
    };
    let Some(gamma) = crate::research::buffer_span(offsets[2], hidden as usize * 2, arena) else {
        return false;
    };
    crate::research::disjoint_writes(&[gamma], &[output.clone()])
        && (input == output || crate::research::disjoint_writes(&[input], &[output]))
}

#[cfg(test)]
mod rms_tests {
    use super::*;

    #[test]
    fn rms_in_place_alias_and_extent_rules_are_checked() {
        let n = 64 * 3840 * 2;
        let gamma = 3840 * 2;
        assert!(rms32_buffers_fit([0, 0, n], 64, 3840, n + gamma));
        assert!(rms32_buffers_fit([0, n, 2 * n], 64, 3840, 2 * n + gamma));
        assert!(!rms32_buffers_fit([0, 2, n], 64, 3840, n + gamma));
        assert!(!rms32_buffers_fit([0, 0, 0], 64, 3840, n + gamma));
        assert!(!rms32_buffers_fit(
            [0, n, 2 * n],
            64,
            3840,
            2 * n + gamma - 1
        ));
        assert!(!rms32_buffers_fit(
            [usize::MAX, n, 2 * n],
            64,
            3840,
            usize::MAX
        ));
        assert!(!rms32_buffers_fit([0, n, 2 * n], 65, 3840, 2 * n + gamma));
        assert!(!rms32_buffers_fit([0, n, 2 * n], 64, 4096, usize::MAX));
    }

    #[test]
    fn rms_reduction_covers_each_feature_once_and_retains_real_gamma() {
        let mut hits = vec![0; 3840];
        for lane in 0..32 {
            for d in (lane..3840).step_by(32) {
                hits[d] += 1;
            }
        }
        assert!(hits.into_iter().all(|x| x == 1));
        let source = include_str!("research_shaders/rms_simd32.metal");
        assert!(source.contains("float sum = 0.0f"));
        assert!(source.contains("simd_sum(sum) / float(hidden) + eps"));
        assert!(source.contains("f16_sat(value * inverse * float(gamma[d]))"));
        assert!(!source.contains("1.0f + float(gamma"));
        assert!(crate::research::launch_fits(32, 32, 0, 16384, 32, 0));
        assert!(!crate::research::launch_fits(64, 128, 0, 16384, 32, 0));
    }
}

#[cfg(test)]
mod identity_tests {
    use crate::research::{Gemma12bResearchShape, MetalResearchCandidate};

    fn shape(tokens: u32) -> Gemma12bResearchShape {
        Gemma12bResearchShape {
            tokens,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            attention_window: 1024,
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        }
    }

    #[test]
    fn new_routes_reject_near_miss_models_and_preserve_the_attention_window_boundary() {
        for candidate in [
            MetalResearchCandidate::Mma32Prefetch,
            MetalResearchCandidate::AttentionQ4,
            MetalResearchCandidate::RmsSimd32,
        ] {
            let good = shape(64);
            assert!(good.supports(candidate));
            for bad in [
                Gemma12bResearchShape {
                    hidden: 5376,
                    ..good
                },
                Gemma12bResearchShape { layers: 60, ..good },
                Gemma12bResearchShape { heads: 32, ..good },
                Gemma12bResearchShape {
                    kv_heads: 4,
                    ..good
                },
                Gemma12bResearchShape {
                    head_dim: 128,
                    ..good
                },
                Gemma12bResearchShape {
                    intermediate: 16384,
                    ..good
                },
                Gemma12bResearchShape { ple: 1, ..good },
                Gemma12bResearchShape {
                    moe_experts: 1,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_top_k: 1,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_intermediate: 1,
                    ..good
                },
                Gemma12bResearchShape {
                    tokens: 1025,
                    ..good
                },
            ] {
                assert!(!bad.supports(candidate));
            }
            let unusual_window = Gemma12bResearchShape {
                attention_window: 17,
                ..good
            };
            assert_eq!(
                unusual_window.supports(candidate),
                candidate != MetalResearchCandidate::AttentionQ4
            );
            assert!(Gemma12bResearchShape {
                kv_heads: 1,
                head_dim: 512,
                attention_window: 0,
                ..good
            }
            .supports(candidate));
            assert_eq!(
                shape(6).supports(candidate),
                candidate != MetalResearchCandidate::AttentionQ4
            );
            assert!(!shape(5).supports(candidate));
        }
    }
}

/// The current disaggregated route supports at most 1024 retained positions.
/// Do not let a short query batch with a much larger prefix enter this trial.
pub fn temporal_context_capacity_fits(block_size: u32, max_blocks: u32) -> bool {
    block_size != 0
        && max_blocks != 0
        && max_blocks
            .checked_mul(block_size)
            .is_some_and(|n| n <= 1024)
}

#[cfg(test)]
mod temporal_capacity_tests {
    #[test]
    fn long_prefix_and_integer_overflow_fail_closed_before_encoding() {
        use super::temporal_context_capacity_fits;
        assert!(temporal_context_capacity_fits(16, 64));
        assert!(temporal_context_capacity_fits(32, 32));
        assert!(!temporal_context_capacity_fits(16, 65));
        assert!(!temporal_context_capacity_fits(0, 64));
        assert!(!temporal_context_capacity_fits(16, 0));
        assert!(!temporal_context_capacity_fits(u32::MAX, 2));
    }
}
