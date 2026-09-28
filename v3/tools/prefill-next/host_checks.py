#!/usr/bin/env python3
"""Independent HOST algebra and source-contract checks. Never Metal qualification."""
import hashlib
import json
from pathlib import Path
import re
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'crates/rvllm-apple-metal/src'
SHADERS = SRC / 'research_shaders'


def coord(lane):
    q = lane // 4
    return (q & 4) + ((lane // 2) % 4), (q & 2) * 2 + (lane % 2) * 2


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    u = u + np.uint32(0x7fff) + ((u >> 16) & 1)
    return (u & np.uint32(0xffff0000)).view(np.float32).astype(np.float64)


def dense_reference(q, k, v, positions, window, present):
    result = np.zeros_like(q)
    for r, pos in enumerate(positions):
        start = max(0, pos + 1 - window) if window else 0
        ids = [t for t in range(start, pos + 1) if present[t]]
        if not ids:
            continue
        s = np.array([sum(q[r, d] * k[t, d] for d in range(q.shape[1])) for t in ids])
        p = np.exp(s - np.max(s)); p /= sum(p)
        result[r] = sum((p[i] * v[t] for i, t in enumerate(ids)), start=np.zeros(q.shape[1]))
    return result


def block_attention(q, k, v, positions, window, present):
    """PR27 mathematical schedule, not a simulation of hardware rounding."""
    m, d = q.shape
    result = np.zeros_like(q)
    for first in range(0, m, 8):
        qs = q[first:first + 8]
        ps = positions[first:first + 8]
        lo = min(max(0, p + 1 - window) if window else 0 for p in ps)
        hi = max(ps) + 1
        mx = np.full(len(ps), -np.inf); denom = np.zeros(len(ps))
        acc = np.zeros_like(qs); poison = np.zeros(len(ps), dtype=bool)
        for kb in range(lo // 32 * 32, hi, 32):
            end = min(kb + 32, len(k))
            keys = np.arange(kb, end)
            # Nonpresent lanes are zero-filled before either matrix product.
            kp = np.where(present[kb:end, None], k[kb:end], 0)
            vp = np.where(present[kb:end, None], v[kb:end], 0)
            scores = qs @ kp.T
            visible = keys[None, :] <= np.array(ps)[:, None]
            if window:
                visible &= keys[None, :] >= np.maximum(0, np.array(ps)[:, None] + 1 - window)
            visible &= present[kb:end]
            scores = np.where(visible, scores, -np.inf)
            new_max = np.maximum(mx, np.max(scores, axis=1))
            corr = np.zeros(len(ps)); live = denom > 0
            corr[live] = np.exp(mx[live] - new_max[live])
            weights = np.zeros_like(scores)
            for r in range(len(ps)):
                take = np.isfinite(scores[r])
                weights[r, take] = np.exp(scores[r, take] - new_max[r])
            bad_keys = np.any(~np.isfinite(vp), axis=1)
            poison |= np.any((weights != 0) & bad_keys, axis=1)
            vp = np.where(np.isfinite(vp), vp, 0)
            acc = acc * corr[:, None] + weights @ vp
            denom = denom * corr + np.sum(weights, axis=1)
            mx = new_max
        good = denom > 0
        acc[good] /= denom[good, None]
        acc[poison] = np.nan
        result[first:first + len(ps)] = acc
    return result


class HostAlgebra(unittest.TestCase):
    def test_fragment_map_and_inverse(self):
        cells = [(r, c + e) for lane in range(32) for r, c in [coord(lane)] for e in range(2)]
        self.assertEqual(len(set(cells)), 64)
        self.assertEqual(set(cells), {(r, c) for r in range(8) for c in range(8)})
        matrix = np.arange(64).reshape(8, 8)
        rebuilt = np.zeros((8, 8), dtype=int)
        for lane in range(32):
            r, c = coord(lane); rebuilt[r, c:c+2] = matrix[r, c:c+2]
        np.testing.assert_array_equal(rebuilt, matrix)

    def test_device_probe_has_exact_dyadic_reference(self):
        a = np.arange(1, 65).reshape(8, 8)
        b = np.outer(np.arange(1, 9), np.arange(2, 10)) / 16
        ref = a @ b
        self.assertEqual(ref[0, 0], 25.5)
        np.testing.assert_array_equal(ref, ref.astype(np.float32))
        wrong = np.concatenate([a.T.ravel(), ref.ravel()])
        self.assertFalse(np.array_equal(wrong, np.concatenate([a.ravel(), ref.ravel()])))

    def test_both_projection_tilings_and_padded_load_coverage(self):
        for bm, bn, gm, gn in [(64, 64, 2, 2), (32, 128, 1, 4)]:
            loads = np.zeros((bm + bn, 40), dtype=int)
            output = np.zeros((bm, bn), dtype=int)
            for tid in range(128):
                for j in range((bm + bn) // 16):
                    v = tid + j * 128; loads[v // 8, v % 8 * 4:v % 8 * 4 + 4] += 1
            self.assertTrue(np.all(loads[:, :32] == 1)); self.assertTrue(np.all(loads[:, 32:] == 0))
            for sg in range(4):
                for lane in range(32):
                    r, c = coord(lane)
                    for i in range(4):
                        for j in range(4):
                            rr = sg // gn * 32 + i * 8 + r; cc = sg % gn * 32 + j * 8 + c
                            output[rr, cc:cc+2] += 1
            self.assertTrue(np.all(output == 1)); self.assertEqual(gm * gn, 4)

    def test_projection_tail_guards_and_role_divisibility(self):
        for bm, bn in [(64, 64), (32, 128)]:
            for m in [6, 17, 33, 63, 65, 101, 256, 304, 512, 1024, 2048]:
                rows = [tile * bm + r for tile in range((m+bm-1)//bm) for r in range(bm) if tile*bm+r < m]
                self.assertEqual(rows, list(range(m)))
            for n in [3840, 8192, 9216, 30720]: self.assertEqual(n % bn, 0)

    def test_projection_full_bf16_range_is_not_fp16_narrowing(self):
        for k in [3840, 4096, 8192, 15360]:
            ex = np.resize(np.array([-40, -20, 0, 20, 40]), k)
            a = bf16(np.exp2(ex) / 8); b = bf16(np.exp2(-ex) * 3 / 16)
            self.assertEqual(np.dot(a, b), k * 3 / 128)
            with np.errstate(over='ignore', invalid='ignore'):
                narrowed = a.astype(np.float16).astype(float) * b.astype(np.float16).astype(float)
            self.assertFalse(np.all(np.isfinite(narrowed)))

    def test_attention_qk_fragment_coordinate_orientation(self):
        q = np.arange(8*16).reshape(8, 16) / 64
        k = np.arange(32*16).reshape(32, 16) / 32
        scores = np.zeros((8, 32))
        for sg in range(4):
            acc = np.zeros((8, 8))
            for dk in range(0, 16, 8):
                a = np.zeros((8, 8)); b = np.zeros((8, 8))
                for lane in range(32):
                    r, c = coord(lane)
                    for e in range(2):
                        a[r, c+e] = q[r, dk+c+e]
                        b[r, c+e] = k[sg*8+c+e, dk+r]
                acc += a @ b
            scores[:, sg*8:sg*8+8] = acc
        np.testing.assert_array_equal(scores, q @ k.T)

    def test_attention_pv_fragment_splits_cover_all_d_columns(self):
        for d in [256, 512]:
            count = np.zeros((8, d), dtype=int)
            for sg in range(4):
                for lane in range(32):
                    r, c = coord(lane)
                    for j in range(d//32): count[r, sg*(d//4)+j*8+c:sg*(d//4)+j*8+c+2] += 1
            self.assertTrue(np.all(count == 1))

    def test_block_softmax_matches_full_reference_with_holes_and_tails(self):
        rng = np.random.default_rng(2709)
        for d in [256, 512]:
            for m, window in [(6, 0), (17, 0), (33, 1024)]:
                length = 119 + m
                q = bf16(rng.normal(0, .08, (m, d)))
                k = bf16(rng.normal(0, .08, (length, d)))
                v = bf16(rng.normal(0, .3, (length, d)))
                positions = np.arange(97, 97+m)
                present = np.arange(length)//32 != 1
                actual = block_attention(q, k, v, positions, window, present)
                expected = dense_reference(q, k, v, positions, window, present)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_absolute_sliding_window_excludes_old_and_speculative_rows(self):
        rng = np.random.default_rng(4); length = 2100; d = 256; m = 6
        q = bf16(rng.normal(0, .03, (m, d))); k = bf16(rng.normal(0, .03, (length, d)))
        v = bf16(rng.normal(0, .3, (length, d))); ps = np.arange(2042, 2048)
        present = np.ones(length, bool)
        a = block_attention(q, k, v, ps, 1024, present)
        changed = v.copy(); changed[:1019] = 1e20; changed[2048:] = -1e20
        b = block_attention(q, k, changed, ps, 1024, present)
        np.testing.assert_array_equal(a, b)

    def test_all_holes_output_zero(self):
        z = np.zeros((6, 256)); kv = np.full((64, 256), np.nan)
        self.assertTrue(np.all(block_attention(z, kv, kv, np.arange(32, 38), 0, np.zeros(64, bool)) == 0))

    def test_masked_nan_value_cannot_poison_unrelated_query_rows(self):
        q = np.zeros((6, 256)); k = np.ones((64, 256)); v = np.ones_like(k)
        v[35, 7] = np.nan
        actual = block_attention(q, k, v, np.arange(32, 38), 0, np.ones(64, bool))
        self.assertTrue(np.all(actual[:3] == 1)); self.assertTrue(np.all(np.isnan(actual[3:])))

    def test_raw_norm_boundary_is_not_bf16_materialization(self):
        x = np.array([1.003, .203, .717], dtype=np.float32)
        before = bf16(x / np.sqrt(np.mean(x*x) + 1e-6))
        b = bf16(x)
        after = bf16(b / np.sqrt(np.mean(b*b) + 1e-6))
        self.assertFalse(np.array_equal(before, after))

    def test_catalog_and_registry_prefix_and_source_budgets(self):
        catalog = json.loads((ROOT/'tools/gemma4_metal_catalog.json').read_text())
        old = json.dumps(catalog['candidates'][:52], sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(hashlib.sha256(old).hexdigest(), 'e7704d1ed0bc7bb3a91f8242be101f041d6919b2faeb95e86a2ab5ef167a4d57')
        self.assertEqual(catalog['default'], 'off'); self.assertEqual(len(catalog['candidates']), 56)
        self.assertEqual(catalog['dispatch_schema'], 'rvllm.metal.research-dispatch.v7')
        names = re.search(r'RESEARCH_KERNEL_NAMES:.*?= \[(.*?)\];', (SRC/'research_evidence.rs').read_text(), re.S).group(1)
        names = re.findall(r'"([^"]+)"', names)
        self.assertEqual(len(names), 118)
        registry = [k for c in catalog['candidates'] for k in c['kernels']]
        self.assertEqual(set(names), set(registry)); self.assertEqual(len(set(names)), 118)
        self.assertEqual(hashlib.sha256('\n'.join(names[:102]).encode()).hexdigest(), '7db532a73a44f428883a88ff0de8c6c1b8bb078b6624596459b18a669f882b4b')
        for c in catalog['candidates'][52:]:
            leaf = (ROOT/c['source_file']).read_text()
            found = re.findall(r'kernel void (research_\w+)\(', leaf)
            self.assertEqual(found, c['kernels'])
            for b in c['budgets']:
                self.assertLessEqual(b['source_shared_bytes'], 12800)
        self.assertEqual([8*d*2+1024+128+96+32+16 for d in (256,512)], [5392,9488])

    def test_new_rust_and_same_process_probe_are_explicit(self):
        paths = [SRC/'prefill_next.rs', SRC/'prefill_next_tests.rs', SRC/'bin/rvllm-prefill-next.rs',
                 *list((SRC/'bin/prefill_next').glob('*.rs'))]
        for path in paths:
            text = path.read_text(); self.assertIn('#![forbid(unsafe_code)]', text)
            self.assertNotRegex(text, r'\bunsafe\s+(?:fn|\{)')
        swift = (ROOT/'tools/prefill-next/MetalArm.swift').read_text()
        self.assertLess(swift.index('try verifyLayout(bytes)'), swift.index('let first = try arm.execute(job.passes)'))
        self.assertIn('job.preflight.count == (needsLayout(job) ? 2 : 1)', swift)
        attn = (SHADERS/'prefill27_attention.metal').read_text()
        self.assertIn('isfinite(value) ? value : 0.0f', attn)
        self.assertNotIn('half probability', attn)
        self.assertIn('simdgroup_float8x8 p;', attn)

if __name__ == '__main__':
    unittest.main(verbosity=2)
