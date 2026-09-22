"""Independent CPU design models. Not execution of Rust, Metal or the ANE compiler.

Tolerances here concern float32 vs float64 CPU models only. They neither replace
nor modify the repository's device/tensor acceptance tolerances.
"""
from __future__ import annotations
import unittest
import numpy as np


def half(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def softcap(x):
    return half(half(np.tanh(half(np.asarray(x, dtype=np.float32) / 30.0))) * 30.0)


def pruned_top5(values):
    """Streaming witness design, compared with an all-transform stable sort."""
    scores = [-float('inf')] * 5
    witnesses = [-float('inf')] * 5
    ids = [-1] * 5
    transformed = 0
    for token, raw in enumerate(values):
        raw = float(raw)
        if not np.isfinite(raw):
            raise ValueError('nonfinite, including skipped rows')
        if ids[4] >= 0 and raw <= witnesses[4]:
            continue
        transformed += 1
        score = float(softcap(raw))
        for rank in range(5):
            if score > scores[rank]:
                scores.insert(rank, score); witnesses.insert(rank, raw); ids.insert(rank, token)
                scores.pop(); witnesses.pop(); ids.pop()
                break
    return np.array(ids), np.array(scores, dtype=np.float32), transformed


def quantized_matrices(seed, h=128, i=256):
    rng = np.random.default_rng(seed)
    return [half(rng.integers(-7, 8, size=shape, dtype=np.int8).astype(np.float32)
                 * np.full((shape[0], 1), 1 / 128, dtype=np.float16))
            for shape in [(i, h), (i, h), (h, i)]]


def gelu_reference(g, u):
    cube = half(half(g * g) * g)
    summed = half(g + half(cube * half(0.044715)))
    argument = half(summed * half(0.7978845608))
    factor = half(half(np.tanh(argument)) + 1.0)
    return half(half(half(g * 0.5) * factor) * u)


class HostRankingTests(unittest.TestCase):
    def test_exhaustive_finite_half_monotonicity_and_adversarial_orders(self):
        bits = np.arange(65536, dtype=np.uint16)
        values = bits.view(np.float16).astype(np.float32)
        values = values[np.isfinite(values)]
        ordered = np.sort(values, kind='stable')
        self.assertEqual(len(values), 63488)
        self.assertTrue(np.all(np.diff(softcap(ordered)) >= 0))
        perm = ((np.arange(65536, dtype=np.uint32) * 40503 + 17) & 65535).astype(np.uint16)
        perm = perm.view(np.float16).astype(np.float32)
        perm = perm[np.isfinite(perm)]
        for stream in (ordered, ordered[::-1], perm, perm[::-1]):
            actual, scores, count = pruned_top5(stream)
            all_scores = softcap(stream)
            expected = np.argsort(-all_scores, kind='stable')[:5]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(scores.view(np.uint32), all_scores[expected].view(np.uint32))
            self.assertLessEqual(count, len(stream))

    def test_raw_top5_mutant_is_rejected_by_rounding_plateaus(self):
        values = np.array([256.] * 5 + list(range(512, 576)), dtype=np.float16).astype(np.float32)
        expected = np.argsort(-softcap(values), kind='stable')[:5]
        raw_mutant = np.argsort(-values, kind='stable')[:5]
        np.testing.assert_array_equal(expected, np.arange(5))
        self.assertFalse(np.array_equal(raw_mutant, expected))
        np.testing.assert_array_equal(pruned_top5(values)[0], expected)

    def test_signed_zero_and_nonfinite_cannot_hide_in_pruned_rows(self):
        values = np.array([-0., 0., -0., 0., 0., -1., -0.], dtype=np.float32)
        ids, scores, _ = pruned_top5(values)
        expected = np.argsort(-softcap(values), kind='stable')[:5]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_array_equal(scores.view(np.uint32), softcap(values)[expected].view(np.uint32))
        for bad in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                pruned_top5(np.array([1., 2., 3., 4., 5., bad], dtype=np.float32))


class ProjectionTests(unittest.TestCase):
    def test_register_staging_is_bijective_and_fragments_cover_outputs(self):
        indices = [tid + 128 * i for tid in range(128) for i in range(8)]
        self.assertEqual(sorted(indices), list(range(1024)))
        hits = np.zeros((32, 32), dtype=np.int32)
        for sg in range(4):
            sm, sn = sg // 2 * 16, sg % 2 * 16
            for dr, dc in [(0, 0), (0, 8), (8, 0), (8, 8)]:
                hits[sm + dr:sm + dr + 8, sn + dc:sn + dc + 8] += 1
        np.testing.assert_array_equal(hits, 1)

    def test_next_k_and_prompt_tails_against_integer_matmul_with_duplicate_tile_mutant(self):
        # Integer-valued inputs/products make all sums exactly representable.
        for k in [3840, 4096, 8192, 15360]:
            a = ((np.arange(32 * k).reshape(32, k) * 3 + 1) % 7 - 3).astype(np.float32)
            b = ((np.arange(32 * k).reshape(32, k) * 5 + 2) % 11 - 5).astype(np.float32)
            expected = a.astype(np.int64) @ b.astype(np.int64).T
            staged_a, staged_b = a[:, :32], b[:, :32]
            out = np.zeros((32, 32), dtype=np.float32)
            addresses = list(range(32))
            for kb in range(0, k, 32):
                if kb + 32 < k:
                    next_a, next_b = a[:, kb + 32:kb + 64], b[:, kb + 32:kb + 64]
                    addresses.extend(range(kb + 32, kb + 64))
                for kk in range(0, 32, 8):
                    out += staged_a[:, kk:kk + 8] @ staged_b[:, kk:kk + 8].T
                if kb + 32 < k:
                    staged_a, staged_b = next_a, next_b
            self.assertEqual(addresses, list(range(k)))
            np.testing.assert_array_equal(out, expected)
            duplicate_mutant = a[:, :32] @ b[:, :32].T * (k // 32)
            self.assertFalse(np.array_equal(duplicate_mutant, expected))
        for m in [6, 21, 33, 64, 84, 1024]:
            writes = [32 * block + row for block in range((m + 31) // 32)
                      for row in range(32) if 32 * block + row < m]
            self.assertEqual(writes, list(range(m)))

    def test_down_output_partition_not_split_reduction(self):
        for seed in range(7):
            g, u, d = quantized_matrices(seed)
            x = half(np.sin(np.arange(128) + seed) / 8)
            activated = gelu_reference(half(g @ x), half(u @ x))
            expected = half(d @ activated)
            pieces = [half(rows @ activated) for rows in np.split(d, 4)]
            actual = np.concatenate(pieces)
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.array_equal(np.concatenate(pieces[::-1]), expected))
            # All raw bytes and per-output-row scales are partitioned, not requantized.
            raw = np.arange(128 * 256, dtype=np.uint8).reshape(128, 256)
            self.assertEqual(b''.join(p.tobytes() for p in np.split(raw, 4)), raw.tobytes())

    def test_post_projection_rms_real_gamma_and_fp32_statistic(self):
        x = (np.sin(np.arange(3840) / 7) * 13).astype(np.float32)
        gamma = (0.3 + np.cos(np.arange(3840) / 13)).astype(np.float32)
        per_lane = np.zeros(32, dtype=np.float32)
        for first in range(0, 3840, 32):
            per_lane += x[first:first + 32] * x[first:first + 32]
        for width in [16, 8, 4, 2, 1]:
            per_lane[:width] += per_lane[width:2 * width]
        actual = x / np.sqrt(np.float32(per_lane[0] / np.float32(3840) + np.float32(1e-6))) * gamma
        expected = x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6) * gamma
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        gamma_mutant = x / np.sqrt(np.mean(x * x) + 1e-6) * (1 + gamma)
        self.assertGreater(float(np.max(np.abs(gamma_mutant - expected))), 1.)
        # This does not assert bit parity with the incumbent 256-thread reduction.


class LayoutTests(unittest.TestCase):
    def test_packed32_declared_codec_preserves_every_half_bit_pattern(self):
        bits = np.arange(65536, dtype=np.uint16)
        for first in range(0, len(bits), 3840):
            source = np.resize(bits[first:first + 3840], 3840).astype('<u2')
            encoded = source.reshape(1, 120, 1, 32).tobytes()
            self.assertEqual(len(encoded), 7680)
            np.testing.assert_array_equal(np.frombuffer(encoded, dtype='<u2'), source)
        # These are declared bytes. The driver stride/size contract remains unknown.

    def test_attention_transpose_flags_match_explicit_operands(self):
        rng = np.random.default_rng(811)
        for heads, d, capacity in [(2, 7, 11), (16, 256, 64), (16, 512, 64)]:
            q = rng.normal(size=(1, heads, d, 1)).astype(np.float32)
            k = rng.normal(size=(1, heads, d, capacity)).astype(np.float32)
            v = rng.normal(size=(1, heads, d, capacity)).astype(np.float32)
            qt, vt = np.swapaxes(q, -1, -2).copy(), np.swapaxes(v, -1, -2).copy()
            explicit_scores = half(np.matmul(qt, k))
            flag_scores = half(np.einsum('bhdi,bhdk->bhik', q, k))
            np.testing.assert_allclose(flag_scores, explicit_scores, rtol=2e-3, atol=0.08)
            p = np.exp(explicit_scores - np.max(explicit_scores, axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            p = half(half(p) * 32.)
            explicit = half(half(np.matmul(p, vt)) * 0.03125)
            flags = half(half(np.einsum('bhik,bhdk->bhid', p, v)) * 0.03125)
            np.testing.assert_allclose(flags, explicit, rtol=2e-3, atol=0.01)
            # Scale moves/extra materializations are deliberately not tested as equivalent.


def paged_fixture(m, d, kv_heads, seed):
    rng = np.random.default_rng(seed)
    block_size = 16
    pages = (m + block_size - 1) // block_size
    blocks = rng.permutation(pages).astype(np.int32)
    kc = half(rng.normal(size=(pages, block_size, kv_heads, d)) / 8)
    vc = half(rng.normal(size=kc.shape) / 8)
    q = half(rng.normal(size=(m, 16, d)) / 8)
    return q, kc, vc, blocks, block_size


def dense_attention(q, kc, vc, blocks, bs, positions, context, window):
    m, heads, d = q.shape
    kv_heads = kc.shape[2]
    outputs = np.empty_like(q)
    for row in range(m):
        end = int(positions[row]) + 1
        start = max(0, end - window) if window else 0
        ts = np.arange(start, end)
        page = blocks[ts // bs]
        valid = page >= 0
        if np.any(page >= len(kc)) or not np.isfinite(q[row]).all():
            outputs[row] = np.nan; continue
        ts, page = ts[valid], page[valid]
        for head in range(heads):
            group = head // (heads // kv_heads)
            k = kc[page, ts % bs, group].astype(np.float64)
            v = vc[page, ts % bs, group].astype(np.float64)
            if len(k) == 0:
                outputs[row, head] = 0; continue
            scores = k @ q[row, head].astype(np.float64)
            weights = np.exp(scores - scores.max()); weights /= weights.sum()
            outputs[row, head] = weights @ v
    return outputs


def temporal_attention(q, kc, vc, blocks, bs, positions, context, window):
    m, heads, d = q.shape
    kv_heads = kc.shape[2]
    rows = 16 if d == 256 else 8
    out = np.zeros_like(q)
    for first_query in range(0, m, 4):
        query_ids = np.arange(first_query, min(first_query + 4, m))
        ends = positions[query_ids] + 1
        starts = np.maximum(ends - window, 0) if window else np.zeros_like(ends)
        for head in range(heads):
            group = head // (heads // kv_heads)
            queries = q[query_ids, head]
            result = np.zeros((len(query_ids), d), dtype=np.float32)
            maximum = np.full(len(query_ids), -np.inf, dtype=np.float32)
            denominator = np.zeros(len(query_ids), dtype=np.float32)
            bad = ~np.isfinite(queries).all(axis=1)
            for first in range(int(starts.min()), int(ends.max()), rows):
                for t in range(first, min(first + rows, int(ends.max()))):
                    visible = (t >= starts) & (t < ends)
                    page = int(blocks[t // bs])
                    if page >= len(kc):
                        bad |= visible; continue
                    if page < 0:
                        continue
                    k = kc[page, t % bs, group]
                    v = vc[page, t % bs, group]
                    # The lane sums visit i*32+lane; reduction tree modeled explicitly.
                    terms = (queries * k).reshape(len(query_ids), d // 32, 32)
                    lanes = np.zeros((len(query_ids), 32), dtype=np.float32)
                    for i in range(d // 32): lanes += terms[:, i]
                    for width in [16, 8, 4, 2, 1]: lanes[:, :width] += lanes[:, width:width * 2]
                    score = lanes[:, 0]
                    active = np.flatnonzero(visible)
                    bad[active] |= ~np.isfinite(score[active])
                    new = np.maximum(maximum[active], score[active])
                    prev_weight = np.where(np.isfinite(maximum[active]), np.exp(maximum[active] - new), 0.)
                    weight = np.exp(score[active] - new)
                    result[active] = result[active] * prev_weight[:, None] + weight[:, None] * v
                    denominator[active] = denominator[active] * prev_weight + weight
                    maximum[active] = new
            nonempty = denominator > 0
            result[nonempty] /= denominator[nonempty, None]
            result[bad] = np.nan
            out[query_ids, head] = result
    return out


class TemporalAttentionTests(unittest.TestCase):
    def test_grouped_staging_against_independent_dense_gather(self):
        for m, d, kv, window in [(64, 256, 8, 1024), (65, 256, 8, 1024),
                                  (84, 512, 1, 0), (65, 512, 1, 0)]:
            q, k, v, b, bs = paged_fixture(m, d, kv, m + d)
            positions = np.arange(m, dtype=np.int32)
            # Include page holes, tails and nonidentity physical allocation.
            b[1] = -1
            actual = temporal_attention(q, k, v, b, bs, positions, m, window)
            expected = dense_attention(q, k, v, b, bs, positions, m, window)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            future_mutant = dense_attention(q, k, v, b, bs, np.full(m, m - 1), m, window)
            self.assertGreater(float(np.max(np.abs(future_mutant - expected))), 0.01)

    def test_invalid_pages_and_hole_only_nonfinite_query_propagate(self):
        q, k, v, b, bs = paged_fixture(64, 256, 8, 3)
        positions = np.arange(64, dtype=np.int32)
        b[0] = len(k)
        result = temporal_attention(q, k, v, b, bs, positions, 64, 1024)
        self.assertTrue(np.isnan(result).all())
        b[:] = -1
        q[0, 0, 0] = np.nan
        result = temporal_attention(q, k, v, b, bs, positions, 64, 1024)
        self.assertTrue(np.isnan(result[0, 0]).all())
        self.assertTrue(np.all(result[1:] == 0))

    def test_sliding_window_boundary_uses_position_not_local_query_index(self):
        # Small independent gather isolates the mask identity at/after 1024.
        positions = np.array([1023, 1024, 1025, 1026])
        starts = positions + 1 - np.minimum(positions + 1, 1024)
        np.testing.assert_array_equal(starts, [0, 1, 2, 3])
        self.assertFalse(np.array_equal(starts, np.zeros(4)))
        self.assertEqual((int(starts.min()), int((positions + 1).max())), (0, 1027))

if __name__ == '__main__':
    unittest.main()
