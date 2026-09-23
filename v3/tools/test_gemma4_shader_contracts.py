"""Host address/reduction models and source guards, never shader execution."""
from pathlib import Path
import re,unittest
ROOT=Path(__file__).resolve().parents[1]/'crates/rvllm-apple-metal/src'
class ShaderContracts(unittest.TestCase):
    def test_new_matrix_entrypoints_refuse_wrong_uniform_threadgroups(self):
        for stem in ['long_mma32x64','mma32_f32','mma32_load4']:
            source=(ROOT/f'research_shaders/{stem}.metal').read_text()
            entries=re.split(r'kernel void ',source)[1:]
            self.assertEqual(len(entries),2)
            for entry in entries:
                self.assertIn('uint3 threads [[threads_per_threadgroup]]',entry)
                self.assertIn('uint3 group [[threadgroup_position_in_grid]]',entry)
                self.assertIn('threads.x != 128u || threads.y != 1u || threads.z != 1u',entry)
                self.assertLess(entry.index('threads.x !='),entry.index('threadgroup ',entry.index(']]) {')))
                self.assertIn('group.xy, tid, sg',entry)
    def test_vector_transaction_alignment_and_storage_coverage(self):
        # Four-element loads cover every A/B tile scalar once; no fp16 conversion
        # or assumed lane-to-matrix layout is used in this independent address model.
        indices=[]
        for tid in range(128):
            for vector in range(tid,256,128):
                row,col=divmod(vector,8)
                indices.extend(row*32+4*col+j for j in range(4))
        self.assertEqual(sorted(indices),list(range(1024)))
        for k in [3840,4096,8192,15360]:
            for row in [0,1,31,63]:
                for col in range(0,32,4):self.assertEqual((row*k+col)*2%8,0)
    def test_matrix_fragment_rectangles_write_every_output_once(self):
        for bm,bn,groups in [(16,64,4),(32,32,4),(32,64,4)]:
            out=[]
            for sg in range(groups):
                if bm==16:sm,sn=0,sg*16
                else:sm,sn=(sg//2)*16,(sg%2)*(bn//2)
                for row in range(16):
                    for col in range(bn//(4 if bm==16 else 2)):
                        out.append((sm+row)*bn+sn+col)
            self.assertEqual(sorted(out),list(range(bm*bn)))
    def test_projection_tail_grid_matches_independent_cartesian_domain(self):
        for bm,bn in [(16,64),(32,32),(32,64)]:
            for m,n in [(6,32),(15,67),(16,65),(17,64),(63,67),(64,64),(65,35),(84,128)]:
                actual=[(i*bm+r,j*bn+c) for i in range((m+bm-1)//bm) for j in range((n+bn-1)//bn)
                        for r in range(bm) for c in range(bn) if i*bm+r<m and j*bn+c<n]
                self.assertEqual(len(actual),m*n);self.assertEqual(set(actual),{(r,c) for r in range(m) for c in range(n)})
    def test_fp32_and_vector_candidates_keep_intended_operand_distinction(self):
        fp=(ROOT/'research_shaders/mma32_f32.metal').read_text()
        vec=(ROOT/'research_shaders/mma32_load4.metal').read_text()
        self.assertIn('simdgroup_matrix<float, 8, 8>',fp)
        self.assertIn('simdgroup_matrix<half, 8, 8>',vec)
        self.assertIn('vec<half, 4>',vec)
        self.assertNotIn('simdgroup_matrix<float, 8, 8>',vec)
        self.assertIn('K % 32u != 0u || N % 32u != 0u',vec)
    def test_prefetch_probe_preserves_exact_accumulator_gate_and_short_shapes(self):
        source=(ROOT/'prefill_mma_tile_tests.rs').read_text()
        self.assertIn('candidate == Some(crate::MetalResearchCandidate::Mma32Prefetch)',source)
        self.assertIn('let small_m = if short { 6 } else { 84 };',source)
        self.assertIn('let large_m = if short { 63 } else { 652 };',source)
        self.assertIn('if short { 64 } else { 1024 }',source)
        self.assertNotIn('fn tile_source(',source)
        self.assertIn('layout-only variant must preserve every FP32 accumulator result',source)
    def test_rms256_uses_full_precision_statistics_and_real_gamma(self):
        s=(ROOT/'research_shaders/rmsnorm_simd256.metal').read_text()
        self.assertIn('float partial = 0.0f',s)
        self.assertIn('float total = simd_sum',s)
        self.assertIn('f16_sat(v * rms * float(gamma[i]))',s)
        self.assertNotIn('1.0f + float(gamma',s)
        coverage=[i for tid in range(256) for i in range(tid,3840,256)]
        self.assertEqual(sorted(coverage),list(range(3840)))
if __name__=='__main__':unittest.main()
