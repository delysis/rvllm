// Gemma 4 12B donor-schedule adaptation, 2026-09-26.
// Donor copyright (c) 2026, Daisuke Majima. BSD-3-Clause.
// Full notice: v3/third_party/coreai-model-zoo-LICENSE.
// Scheduling reference: john-rocky/coreai-model-zoo @ a2a664e84ee4807cf4f6441944bd0ac6d047224d,
// apps/CoreAIChat/Resources/g4msl/{gemma4_matvec,gemma4_prefill}.metal.txt.
// This implementation uses rvLLM's SIGNED group-32/FP16-scale ABI, NOT the
// donor's unsigned affine group-64 ABI. No activation quantization or repacking.
// All address products use size_t; the checked host owns allocation lifetimes.
#include <metal_stdlib>
#pragma METAL fp math_mode(safe)
using namespace metal;

struct Donor12bParams {
    uint m, n, k, stride, column, output_f32, reserved0, reserved1;
};
struct Donor12bAttentionParams {
    uint block_size, max_blocks, num_blocks, window;
};
inline float d12_load(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort d12_round(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7fffffffU) > 0x7f800000U) return ushort((b >> 16) | 0x40U);
    return ushort((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
inline float d12_gelu(float x) {
    if (x >= 5.0f) return x;
    if (x <= -5.0f) return 0.0f;
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f *
                                             (x + 0.044715f * x * x * x)));
}
inline void d12_store(device uchar *out, size_t index, float value, uint fp32) {
    if (fp32) reinterpret_cast<device float *>(out)[index] = value;
    else reinterpret_cast<device ushort *>(out)[index] = d12_round(value);
}
// Each eight-code word is entirely within one 32-element quantization group.
// W8 uses two aligned packed uint loads and explicit sign extension; it does
// not depend on implementation-defined char signedness or aligned char8 loads.
template<bool W4>
inline void d12_weights(device const uchar *w, device const half *sc,
                        uint row, uint k, uint k0, thread float (&v)[8]) {
    const size_t words = size_t(k) / (W4 ? 8 : 4);
    const size_t index = size_t(row) * words + k0 / (W4 ? 8 : 4);
    const device uint *packed = reinterpret_cast<device const uint *>(w);
    const uint lo = packed[index];
    const uint hi = W4 ? 0U : packed[index + 1];
    const float scale = float(sc[size_t(row) * (k / 32) + k0 / 32]);
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint code = W4 ? ((lo >> (4*j)) & 15U)
                            : (((j < 4 ? lo : hi) >> (8*(j & 3))) & 255U);
        const int q = W4 ? int(code ^ 8U) - 8 : int(code ^ 128U) - 128;
        v[j] = float(q) * scale;
    }
}

// Donor R4 schedule: eight consecutive activations/lane, reused across rows.
// Do not move group-varying scales into the final row epilogue. The eight-term
// partial followed by block accumulation is intentional and tested separately
// from the incumbent's different dot/reduction order.
template<bool W4, uint SG>
inline void d12_qmv(device const ushort *x, device const uchar *w,
                    device const half *sc, device uchar *out,
                    constant Donor12bParams &p, uint3 group, uint sg, uint lane) {
    const uint row0 = (group.x * SG + sg) * 4;
    if (p.m != 1 || p.k % 256 != 0 || p.n % 4 != 0 ||
        p.stride < p.n || p.column > p.stride - p.n || p.output_f32 > 1 || row0 >= p.n) return;
    float acc[4] = {0, 0, 0, 0};
    for (uint kb = 0; kb < p.k; kb += 256) {
        const uint k0 = kb + lane * 8;
        float a[8];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) a[j] = d12_load(x[k0+j]);
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            float v[8]; d12_weights<W4>(w, sc, row0+r, p.k, k0, v);
            float partial = 0.0f;
            #pragma unroll
            for (uint j = 0; j < 8; ++j) partial += a[j] * v[j];
            acc[r] += partial;
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        const float sum = simd_sum(acc[r]);
        if (lane == 0) d12_store(out, size_t(p.column) + row0 + r, sum, p.output_f32);
    }
}

// M=8/RP=2 wide lane. A weight word is loaded/dequantized once, then consumed
// by EIGHT token accumulators, not eight independent GEMVs. Remainders are
// masked before loads and stores; no changes to token order or positions.
template<bool W4, uint SG>
inline void d12_batch8(device const ushort *x, device const uchar *w,
                       device const half *sc, device uchar *out,
                       constant Donor12bParams &p, uint3 group, uint sg, uint lane) {
    const uint row0 = (group.x * SG + sg) * 2;
    const uint token0 = group.y * 8;
    if (p.m == 0 || p.m > 128 || p.k % 256 != 0 || p.n % 2 != 0 ||
        p.stride < p.n || p.column > p.stride - p.n || p.output_f32 > 1 || row0 >= p.n) return;
    float acc[8][2];
    #pragma unroll
    for (uint t=0;t<8;++t) for (uint r=0;r<2;++r) acc[t][r]=0.0f;
    for (uint kb=0;kb<p.k;kb+=256) {
        const uint k0=kb+lane*8;
        float weight[2][8];
        #pragma unroll
        for (uint r=0;r<2;++r) d12_weights<W4>(w,sc,row0+r,p.k,k0,weight[r]);
        #pragma unroll
        for (uint t=0;t<8;++t) {
            if (token0+t < p.m) {
                float partial[2]={0.0f,0.0f};
                #pragma unroll
                for (uint j=0;j<8;++j) {
                    const float a=d12_load(x[size_t(token0+t)*p.k+k0+j]);
                    #pragma unroll
                    for (uint r=0;r<2;++r) partial[r]+=a*weight[r][j];
                }
                #pragma unroll
                for (uint r=0;r<2;++r) acc[t][r]+=partial[r];
            }
        }
    }
    #pragma unroll
    for (uint t=0;t<8;++t) for (uint r=0;r<2;++r) {
        const float sum=simd_sum(acc[t][r]);
        if (lane==0 && token0+t<p.m)
            d12_store(out,size_t(token0+t)*p.stride+p.column+row0+r,sum,p.output_f32);
    }
}

// Two projections share X loads. Gate and up each round to BF16 IN REGISTERS
// before the incumbent GELU and multiply. Trace requests use the unfused path.
template<bool W4, uint SG>
inline void d12_gate(device const ushort *x,
                     device const uchar *wg, device const half *sgate,
                     device const uchar *wu, device const half *sup,
                     device ushort *out, uint3 group, uint sg, uint lane) {
    const uint row0=(group.x*SG+sg)*4;
    if (row0>=15360) return;
    float g[4]={0,0,0,0}, u[4]={0,0,0,0};
    for (uint kb=0;kb<3840;kb+=256) {
        const uint k0=kb+lane*8;
        float a[8];
        #pragma unroll
        for(uint j=0;j<8;++j) a[j]=d12_load(x[k0+j]);
        #pragma unroll
        for(uint r=0;r<4;++r) {
            float vg[8],vu[8];
            d12_weights<W4>(wg,sgate,row0+r,3840,k0,vg);
            d12_weights<W4>(wu,sup,row0+r,3840,k0,vu);
            float pg=0.0f,pu=0.0f;
            #pragma unroll
            for(uint j=0;j<8;++j) { pg+=a[j]*vg[j]; pu+=a[j]*vu[j]; }
            g[r]+=pg; u[r]+=pu;
        }
    }
    #pragma unroll
    for(uint r=0;r<4;++r) {
        const float gr=d12_load(d12_round(simd_sum(g[r])));
        const float ur=d12_load(d12_round(simd_sum(u[r])));
        if(lane==0) out[row0+r]=d12_round(d12_gelu(gr)*ur);
    }
}

// The host admits global K/V reuse only when both authenticated descriptors
// reference exactly the same packed-values AND scales storage. It does not
// infer equality from matching dimensions. Otherwise use three real segments.
template<bool W4,uint SG>
inline void d12_qkv(device const ushort *x,
                    device const uchar *wq,device const half *sq,
                    device const uchar *wk,device const half *sk,
                    device const uchar *wv,device const half *sv,
                    device ushort *out,constant Donor12bParams &p,
                    uint3 group,uint sg,uint lane) {
    // p.n=Q rows, p.k=KV rows, p.reserved0=raw-K reuse.
    const uint qn=p.n, kn=p.k;
    const bool reuse=p.reserved0!=0;
    const uint row0=(group.x*SG+sg)*4;
    const uint rows=qn+(reuse?kn:2*kn);
    if(p.m!=1 || !((qn==4096 && kn==2048 && !reuse) ||
                   (qn==8192 && kn==512)) || row0>=rows) return;
    const uint seg=row0<qn?0:(row0<qn+kn?1:2);
    const uint local=row0-(seg==0?0:(seg==1?qn:qn+kn));
    device const uchar *w=seg==0?wq:(seg==1?wk:wv);
    device const half *sc=seg==0?sq:(seg==1?sk:sv);
    float acc[4]={0,0,0,0};
    for(uint kb=0;kb<3840;kb+=256) {
        const uint k0=kb+lane*8;
        float a[8];
        #pragma unroll
        for(uint j=0;j<8;++j) a[j]=d12_load(x[k0+j]);
        #pragma unroll
        for(uint r=0;r<4;++r) {
            float v[8]; d12_weights<W4>(w,sc,local+r,3840,k0,v);
            float partial=0.0f;
            #pragma unroll
            for(uint j=0;j<8;++j) partial+=a[j]*v[j];
            acc[r]+=partial;
        }
    }
    #pragma unroll
    for(uint r=0;r<4;++r) {
        const float value=simd_sum(acc[r]);
        if(lane==0) {
            const ushort rounded=d12_round(value);
            out[row0+r]=rounded;
            if(reuse && seg==1) out[qn+kn+local+r]=rounded;
        }
    }
}

template<uint SG>
inline void d12_native_gate(device const ushort *x,device const ushort *w,
                            device ushort *out,uint3 group,uint sg,uint lane) {
    const uint row0=(group.x*SG+sg)*4;
    if(row0>=15360) return;
    float g[4]={0,0,0,0},u[4]={0,0,0,0};
    for(uint kb=0;kb<3840;kb+=256) {
        const uint k0=kb+lane*8;
        float a[8];
        #pragma unroll
        for(uint j=0;j<8;++j) a[j]=d12_load(x[k0+j]);
        #pragma unroll
        for(uint r=0;r<4;++r) {
            float pg=0.0f,pu=0.0f;
            #pragma unroll
            for(uint j=0;j<8;++j) {
                pg+=a[j]*d12_load(w[size_t(row0+r)*3840+k0+j]);
                pu+=a[j]*d12_load(w[size_t(15360+row0+r)*3840+k0+j]);
            }
            g[r]+=pg;u[r]+=pu;
        }
    }
    #pragma unroll
    for(uint r=0;r<4;++r) {
        const float gr=d12_load(d12_round(simd_sum(g[r])));
        const float ur=d12_load(d12_round(simd_sum(u[r])));
        if(lane==0) out[row0+r]=d12_round(d12_gelu(gr)*ur);
    }
}

// Native BF16 projection: R4/M1 or R2/M8, requested BF16 or genuinely FP32
// output. FP32 mode has no BF16 round/widen in the epilogue.
template<uint SG,uint B,uint R>
inline void d12_native_projection(device const ushort *x,device const ushort *w,
                                  device uchar *out,constant Donor12bParams &p,
                                  uint3 group,uint sg,uint lane) {
    const uint row0=(group.x*SG+sg)*R,token0=group.y*B;
    if(p.m==0 || p.m>128 || p.k%256 || p.n%R || row0>=p.n ||
       p.stride<p.n || p.column>p.stride-p.n || p.output_f32>1) return;
    float acc[B][R];
    #pragma unroll
    for(uint t=0;t<B;++t) for(uint r=0;r<R;++r) acc[t][r]=0;
    for(uint kb=0;kb<p.k;kb+=256) {
        const uint k0=kb+lane*8;
        float weight[R][8];
        #pragma unroll
        for(uint r=0;r<R;++r) for(uint j=0;j<8;++j)
            weight[r][j]=d12_load(w[size_t(row0+r)*p.k+k0+j]);
        #pragma unroll
        for(uint t=0;t<B;++t) if(token0+t<p.m) {
            float partial[R];
            #pragma unroll
            for(uint r=0;r<R;++r) partial[r]=0;
            #pragma unroll
            for(uint j=0;j<8;++j) {
                const float a=d12_load(x[size_t(token0+t)*p.k+k0+j]);
                #pragma unroll
                for(uint r=0;r<R;++r) partial[r]+=a*weight[r][j];
            }
            #pragma unroll
            for(uint r=0;r<R;++r) acc[t][r]+=partial[r];
        }
    }
    #pragma unroll
    for(uint t=0;t<B;++t) for(uint r=0;r<R;++r) {
        const float sum=simd_sum(acc[t][r]);
        if(lane==0 && token0+t<p.m)
            d12_store(out,size_t(token0+t)*p.stride+p.column+row0+r,sum,p.output_f32);
    }
}

// Paged decode occupancy kernel. One threadgroup/query head, G strided
// independent SIMD scans, followed by one FP32 sufficient-statistics merge.
// Norm/RoPE/cache writes precede this encoder. No cross-threadgroup write/read
// dependency, scratch allocation, capacity-dependent logits buffer or new KV ABI.
template<uint D,uint KV,uint G>
inline void d12_attention(device const ushort *q,device const ushort *kc,
                          device const ushort *vc,device ushort *out,
                          device const int *table,device const int *length,
                          device const int *positions,
                          constant Donor12bAttentionParams &p,
                          threadgroup float *scratch,
                          uint3 group,uint sg,uint lane) {
    const uint head=group.x;
    const int len=length[0],position=positions[0];
    if(head>=16 || p.block_size==0 || p.max_blocks==0 || p.num_blocks==0 ||
       len<=0 || position<0 || position>=len ||
       ulong(len)>ulong(p.block_size)*p.max_blocks) return;
    const uint end=uint(position)+1;
    const uint begin=p.window==0?0:(end>p.window?end-p.window:0);
    // Validate the entire VISIBLE block range before stores. Negative holes
    // are legal; positive out-of-arena pages are not. Uniform across all SGs.
    for(uint block=begin/p.block_size;block<=(end-1)/p.block_size;++block)
        if(table[block]>=0 && uint(table[block])>=p.num_blocks) return;
    const uint kvhead=head/(16/KV);
    float query[D/32],pv[D/32];
    #pragma unroll
    for(uint j=0;j<D/32;++j) {
        query[j]=d12_load(q[size_t(head)*D+lane+32*j]);pv[j]=0.0f;
    }
    float maximum=-INFINITY,denom=0.0f;
    // Assignment follows absolute token t (including holes), not compacted
    // valid-key indices. This makes resumed/sliding prefixes deterministic.
    uint t=begin+((sg+G-(begin%G))%G);
    for(;t<end;t+=G) {
        const int page=table[t/p.block_size];
        if(page<0) continue;
        const size_t base=((size_t(page)*p.block_size+t%p.block_size)*KV+kvhead)*D;
        float dot=0.0f;
        #pragma unroll
        for(uint j=0;j<D/32;++j) dot+=query[j]*d12_load(kc[base+lane+32*j]);
        const float score=simd_sum(dot); // Gemma attention scale is exactly 1.
        const float next=max(maximum,score);
        const float a=denom==0.0f?0.0f:precise::exp(maximum-next);
        const float b=precise::exp(score-next);
        #pragma unroll
        for(uint j=0;j<D/32;++j) pv[j]=a*pv[j]+b*d12_load(vc[base+lane+32*j]);
        denom=a*denom+b;maximum=next;
    }
    #pragma unroll
    for(uint j=0;j<D/32;++j) scratch[sg*(D+2)+lane+32*j]=pv[j];
    if(lane==0) {scratch[sg*(D+2)+D]=maximum;scratch[sg*(D+2)+D+1]=denom;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float m=-INFINITY;
        for(uint s=0;s<G;++s) if(scratch[s*(D+2)+D+1]>0.0f)
            m=max(m,scratch[s*(D+2)+D]);
        float z=0.0f,value[D/32];
        #pragma unroll
        for(uint j=0;j<D/32;++j) value[j]=0.0f;
        for(uint s=0;s<G;++s) {
            const float ds=scratch[s*(D+2)+D+1];
            // Never evaluate exp(-inf - -inf) for an empty subgroup.
            if(ds>0.0f) {
                const float factor=precise::exp(scratch[s*(D+2)+D]-m);
                z+=factor*ds;
                #pragma unroll
                for(uint j=0;j<D/32;++j) value[j]+=factor*scratch[s*(D+2)+lane+32*j];
            }
        }
        #pragma unroll
        for(uint j=0;j<D/32;++j)
            out[size_t(head)*D+lane+32*j]=d12_round(z>0.0f?value[j]/z:0.0f);
    }
}
