use super::reference::{self, Fixture};
use super::*;

#[test]
fn short_capacity_is_host_refused_not_an_encoded_no_write_success() {
    let f = Fixture::for_tile(512, 32, SHORT_R4T128);
    let plan = DecodePlan::new(SHORT_R4T128, f.shape, DecodeOutput::Bf16).unwrap();
    assert_eq!(plan.grid, [4, 1, 1]);
    assert_eq!(plan.threads, [128, 1, 1]);
    assert_eq!(plan.threadgroup_bytes, 2048);
    assert_eq!(plan.scratch_bytes, 0);
    for _ in 0..1024 {
        assert_eq!(
            DecodePlan::new(SHORT_R4T128, f.shape, DecodeOutput::Bf16),
            Some(plan)
        );
    }
    let mut shape = f.shape;
    shape.max_blocks += 1;
    assert!(DecodePlan::new(SHORT_R4T128, shape, DecodeOutput::Bf16).is_none());
    shape.block_size = u32::MAX;
    assert!(DecodePlan::new(SHORT_R4T128, shape, DecodeOutput::Bf16).is_none());
    // L512 with seven-token blocks needs 518 capacity: conservative refusal.
    let odd = Fixture::for_tile(512, 7, SHORT_R4T128);
    assert!(DecodePlan::new(SHORT_R4T128, odd.shape, DecodeOutput::Bf16).is_none());
}
#[test]
fn short_independent_fp64_at_both_screen_lengths() {
    for length in [256, 512] {
        let f = Fixture::for_tile(length, 32, SHORT_R4T128);
        let p = DecodePlan::new(SHORT_R4T128, f.shape, DecodeOutput::F32).unwrap();
        let a = reference::output_f32(&f, p).unwrap();
        let b = reference::output_f64(&f, p).unwrap();
        assert_eq!(a, reference::output_f32(&f, p).unwrap());
        let mut error2 = 0.0;
        let mut norm2 = 0.0;
        for (&x, &y) in a.iter().zip(&b) {
            assert!(x.is_finite() && (f64::from(x) - y).abs() <= 5e-4);
            error2 += (f64::from(x) - y).powi(2);
            norm2 += y * y;
        }
        assert!(error2.sqrt() / norm2.sqrt().max(1e-30) <= 1e-4);
    }
}
#[test]
fn paged_holes_newest_owner_prefix_and_rollback_are_visibility_not_contiguity() {
    let mut f = Fixture::for_tile(65, 7, SHORT_R4T128);
    let p = DecodePlan::new(SHORT_R4T128, f.shape, DecodeOutput::F32).unwrap();
    let original = reference::output_f64(&f, p).unwrap();
    // Last logical token is in a reversed, noncontiguous physical owner page.
    let base = physical_base(f.shape, &f.table, 64).unwrap();
    f.v[base..base + 512].fill(round_bf16(10.0));
    let newest = reference::output_f64(&f, p).unwrap();
    assert_ne!(original, newest);
    // Position excludes speculative payload, even if context still includes it.
    f.position = 30;
    for t in 31..65 {
        let b = physical_base(f.shape, &f.table, t).unwrap();
        f.k[b..b + 512].fill(0x7fc1);
        f.v[b..b + 512].fill(0x7fc1);
    }
    let prefix = reference::output_f64(&f, p).unwrap();
    f.context = 31;
    assert_eq!(prefix, reference::output_f64(&f, p).unwrap());
    // Future invalid positive metadata must not veto the restored prefix.
    f.table[8] = i32::MAX;
    assert_eq!(prefix, reference::output_f64(&f, p).unwrap());
    f.table[1] = -17;
    assert_ne!(prefix, reference::output_f64(&f, p).unwrap());
    f.table.fill(-1);
    assert!(reference::output_f64(&f, p)
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    f.table[0] = i32::MAX;
    assert!(reference::output_f64(&f, p).is_err());
    f.table[0] = -1;
    f.position = f.context;
    assert!(reference::output_f64(&f, p).is_err());
}
#[test]
fn short_uses_existing_checked_buffer_and_alias_contract() {
    let f = Fixture::for_tile(256, 32, SHORT_R4T128);
    let p = DecodePlan::new(SHORT_R4T128, f.shape, DecodeOutput::Bf16).unwrap();
    let mut b = DecodeBuffers {
        q: 64,
        k: 16448,
        v: 16448 + p.cache_bytes,
        block_tables: 16448 + 2 * p.cache_bytes,
        context_lens: 16480 + 2 * p.cache_bytes,
        positions: 16484 + 2 * p.cache_bytes,
        output: 16512 + 2 * p.cache_bytes,
    };
    let size = b.output + 16384 + 64;
    assert!(p.buffers_fit(b, size));
    assert!(!p.buffers_fit(b, b.output + 16383));
    let good = b;
    for source in [b.q, b.k, b.v, b.block_tables, b.context_lens, b.positions] {
        b = good;
        b.output = source;
        assert!(!p.buffers_fit(b, size));
    }
    b = good;
    b.k += 1;
    assert!(!p.buffers_fit(b, size));
    b = good;
    b.block_tables += 2;
    assert!(!p.buffers_fit(b, size));
    b = good;
    b.v = b.k;
    assert!(!p.buffers_fit(b, size));
}
