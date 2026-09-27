#!/usr/bin/env python3
"""Supplemental Python geometry/source checks, NOT Rust or Metal execution.

Uses only the standard library. The safe-Rust tests and Apple oracles are
separate required gates; no result here is evidence of on-device correctness.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import unittest

V3 = Path(__file__).resolve().parents[2]
SRC = V3 / "crates/rvllm-apple-metal/src"
CATALOG = json.loads((V3 / "tools/gemma4_metal_catalog.json").read_text())
# SHA256 of canonical JSON for the first 48 exact-fb5f169c catalog records.
PREFIX_SHA256 = "e9fa847fa33c190ac9286038cf3e144a0b1975703f9b62cd31171d90a5c06ac6"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if bits & 0x7fffffff > 0x7f800000:
        word = bits >> 16 | 0x40
    else:
        word = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]

def visible(position, context, capacity, window):
    if context <= 0 or context > capacity or position < 0 or position >= context:
        raise ValueError("metadata")
    return range(max(0, position + 1 - window) if window else 0, position + 1)

class SourceAndGeometry(unittest.TestCase):
    def test_old_catalog_prefix_is_byte_equivalent_as_data(self):
        self.assertEqual(hashlib.sha256(canonical(CATALOG["candidates"][:48])).hexdigest(), PREFIX_SHA256)
        self.assertEqual(len(CATALOG["candidates"]), 52)
        self.assertEqual(CATALOG["dispatch_schema"], "rvllm.metal.research-dispatch.v6")
        self.assertEqual(sum(len(c["kernels"]) for c in CATALOG["candidates"]), 102)
        self.assertFalse(CATALOG["device_qualified"])
        self.assertEqual(CATALOG["default"], "off")

    def test_all_new_source_symbols_and_bf16_rewrite(self):
        spec = importlib.util.spec_from_file_location("catalog", V3 / "tools/gemma4_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = (SRC / "research_catalog.rs").read_text()
        for candidate in CATALOG["candidates"][48:]:
            block = text[text.index('name: "' + candidate["name"] + '"'):]
            block = block[:block.index("\n            },")]
            includes = re.findall(r'include_str!\("([^"]+)"\)', block)
            self.assertTrue(includes)
            source = "".join((SRC / path).read_text() for path in includes)
            module.check_source(CATALOG, candidate["name"], source)
            # Mirrors only the exact generator's *candidate* rewrite. This is
            # not execution of the Rust exporter or compilation of the source.
            generated = re.sub(r"\bhalf\b", "bfloat", source).replace("f16_sat", "bf16_sat")
            self.assertNotRegex(generated, r"\bhalf\b")
            if "pipeline" in candidate["name"] or "control" in candidate["name"]:
                self.assertIn("device float *C [[buffer(2)]]", generated)
                self.assertIn("device const float *X", generated)
                self.assertIn("bfloat *O", generated)

    def test_vector_staging_output_fragment_coverage(self):
        a = [0] * (32 * 32)
        b = [0] * (64 * 32)
        out = [0] * (32 * 64)
        for tid in range(128):
            for j in range(2):
                for lane in range(4): a[(tid + j * 128) * 4 + lane] += 1
            for j in range(4):
                for lane in range(4): b[(tid + j * 128) * 4 + lane] += 1
        for sg in range(4):
            for r in range(16):
                for c in range(32): out[(sg // 2 * 16 + r) * 64 + sg % 2 * 32 + c] += 1
        self.assertEqual(set(a + b + out), {1})
        for k in (3840, 4096, 8192, 15360):
            reads = [0] + [kb + 32 for kb in range(0, k, 32) if kb + 32 < k]
            self.assertEqual(reads, list(range(0, k, 32)))

    def test_target_grids_and_source_memory_budgets(self):
        for m in (256, 512, 1024, 2048):
            self.assertEqual(math.ceil(m / 32) * (3840 // 64), m * 60 // 32)
            self.assertEqual(math.ceil(m / 4) * 16, m * 4)
        self.assertEqual((32 + 64) * 32 * 2, 6144)
        self.assertEqual(32 * 64 * 4, 8192)
        self.assertEqual(16 * 256 * 2 + 64 * 4 + 16 * 4, 8512)
        self.assertEqual(16 * 512 * 2 + 64 * 4 + 16 * 4, 16704)

    def test_absolute_sliding_bounds_and_page_permutation(self):
        for m in (6, 17, 64, 256, 512, 1024, 2048):
            prefix = 32 if m < 64 else 97
            context = prefix + m + 3
            count = math.ceil(context / 32)
            physical = 1 << (count + 1).bit_length()
            pages = [(i * 5 + 3) % physical for i in range(count)]
            self.assertEqual(len(set(pages)), count)
            for r in (0, min(31, m - 1), m - 1):
                v = visible(prefix + r, context, count * 32, 1024)
                self.assertEqual(v.stop, prefix + r + 1)
                self.assertLessEqual(len(v), 1024)
                self.assertNotIn(context - 1, v)
        for args in ((-1, 40, 64, 0), (40, 40, 64, 0), (0, 65, 64, 0)):
            with self.assertRaises(ValueError): visible(*args)

    def test_tiled_softmax_algebra_with_holes_and_newest(self):
        # Double-precision algebra at boundary rows, not GPU/FP32 validation.
        for m in (6, 17, 64, 256, 512, 1024, 2048):
            prefix = 32 if m < 64 else 97
            for window in (0, 1024):
                for row in sorted({0, min(31, m - 1), m - 1}):
                    span = visible(prefix + row, prefix + m + 3, 4096, window)
                    entries = {t: (((t * 13 + row) % 31 - 15) / 32,
                                   32.0 if t == prefix + m - 1 else ((t * 7) % 23 - 11) / 32)
                               for t in span if t // 32 % 5 != 1}
                    ref = 0.0
                    if entries:
                        mx = max(s for s, _ in entries.values())
                        weights = [(math.exp(s - mx), v) for s, v in entries.values()]
                        ref = sum(w * v for w, v in weights) / sum(w for w, _ in weights)
                    running_max, total, acc = -math.inf, 0.0, 0.0
                    for kb in range(span.start // 16 * 16, span.stop, 16):
                        tile = [entries[t] for t in range(kb, kb + 16) if t in entries]
                        maximum = max([running_max] + [s for s, _ in tile])
                        correction = math.exp(running_max - maximum) if total else 0.0
                        ws = [(math.exp(s - maximum), v) for s, v in tile]
                        total = total * correction + sum(w for w, _ in ws)
                        acc = acc * correction + sum(w * v for w, v in ws)
                        running_max = maximum
                    self.assertAlmostEqual(acc / total if total else 0.0, ref, places=12)

    def test_structured_dot_closed_form_and_rounding_witness(self):
        for k in (3840, 4096, 8192, 15360):
            for r, c in ((0, 0), (1, 16), (7, 63), (16, 30719)):
                a, b = (r * 5 % 7) - 3, (r * 11 % 5) - 2
                x, y = (c * 7 % 31) - 15, (c * 11 % 17) - 8
                dot = sum((a * (1 if i % 2 == 0 else -1) + b * (1 if i % 8 < 4 else -1)) / 32
                          * (x * (1 if i % 2 == 0 else -1) + y * (1 if i % 8 < 4 else -1)) / 64
                          for i in range(k))
                self.assertEqual(dot, k * (a * x + b * y) / 2048)
        x = [1.003, .203, .717]
        rounded = list(map(bf16, x))
        normalize = lambda values: [bf16(v / math.sqrt(sum(z*z for z in values) / 3 + 1e-6)) for v in values]
        self.assertNotEqual(normalize(x), normalize(rounded))

    def test_referee_and_queue_do_not_add_unsafe_rust(self):
        files = [SRC / "prefill_round.rs", SRC / "prefill_round_tests.rs",
                 SRC / "bin/rvllm-prefill-round.rs", *list((SRC / "bin/prefill_round").glob("*.rs"))]
        for path in files:
            source = path.read_text()
            self.assertIn("#![forbid(unsafe_code)]", source, str(path))
            code = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
            self.assertNotRegex(code, r"\bunsafe\s*(?:\{|fn|impl|extern|trait)")
        driver = (V3 / "tools/prefill-round/MetalArm.swift").read_text()
        self.assertLess(driver.index('try need(observed == job.sourceBodySha256'),
                        driver.index('let first = try arm.execute(job.passes)'))
        queue = (SRC / "bin/prefill_round/queue.rs").read_text()
        self.assertIn('"stable_seconds": 0', queue)
        self.assertIn('[a, b, b, a, b, a, a, b]', queue)
        self.assertIn('"production_promotion": false', queue)

if __name__ == "__main__":
    unittest.main(verbosity=2)
