// FP32 SIMD-matrix family. No explicit operand/P downcast or
// softmax over D panels. SIMD-matrix arithmetic has its own numerical order.
// Scratch lifetimes: qt is Q during QK and a PV result panel during PV;
// stage is K during QK and V during PV. Barriers separate all lifetimes.
template<uint D, uint R, uint BK, uint P, uint T>
inline void atlas_matrix_body(device const ushort *q, device const ushort *k,
    device const ushort *v, device const int *table, device const int *positions,
    device uchar *out, device float *partial, device const uint *status,
    device const float *factor, device const ushort *gamma, device const float *cs,
    device const float *sn, constant AtlasParams &p, uint3 group, uint tid,
    ushort sg, ushort lane, uint3 threads, threadgroup float *qt,
    threadgroup float *stage, threadgroup float *scores, threadgroup float *weights,
    threadgroup float *ml, threadgroup int *pages) {
    (void)partial;
    if (status[0] != 0u || !atlas_shape_ok(p) || p.dim != D || p.rows != R
        || p.keys != BK || p.panel != P || p.threads != T || p.splits != 1u
        || threads.x != T || threads.y != 1u || threads.z != 1u
        || group.y >= p.kv_heads || group.z != 0u) return;
    constexpr uint NSG = T/32u;
    constexpr uint OWN = (R+NSG-1u)/NSG;
    constexpr uint RM = (R+NSG*8u-1u)/(NSG*8u);
    uint gqa = 16u/p.kv_heads, packed0 = group.x*R;
    if (packed0 >= p.queries*gqa) return;
    float u[OWN][D/32u];
    for (uint r=0u;r<OWN;++r) for (uint d=0u;d<D/32u;++d) u[r][d]=0.0f;
    for (uint r=tid;r<R;r+=T) { ml[r]=-INFINITY; ml[R+r]=0.0f; ml[2u*R+r]=1.0f; }
    uint start=atlas_start(uint(positions[packed0/gqa]),p.window);
    uint last=(min(packed0+R,p.queries*gqa)-1u)/gqa;
    uint stop=uint(positions[last])+1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first=start;first<stop;first+=BK) {
        for (uint j=tid;j<BK;j+=T) pages[j]=first+j<stop?table[(first+j)/p.page_size]:-1;
        for (uint i=tid;i<R*BK;i+=T) { scores[i]=0.0f; weights[i]=0.0f; }
        for (uint r=tid;r<R;r+=T) ml[2u*R+r]=1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool any_page=false;
        for (uint j=0u;j<BK;++j) any_page|=pages[j]>=0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!any_page) continue;
        simdgroup_float8x8 accum[RM][BK/8u];
        for (uint m=0u;m<RM;++m) for (uint n=0u;n<BK/8u;++n)
            accum[m][n]=simdgroup_float8x8(0.0f);
        for (uint panel=0u;panel<D;panel+=P) {
            for (uint i=tid;i<R*P;i+=T) {
                uint packed=packed0+i/P, token=packed/gqa;
                uint head=group.y*gqa+packed%gqa;
                qt[i]=token<p.queries?atlas_widen(q[(size_t(token)*16u+head)*D+panel+i%P]):0.0f;
            }
            for (uint i=tid;i<BK*P;i+=T) {
                uint j=i/P;
                size_t pt=pages[j]>=0?size_t(pages[j])*p.page_size+(first+j)%p.page_size:0ul;
                stage[i]=pages[j]>=0?atlas_widen(atlas_k(k,factor,gamma,cs,sn,pt,first+j,group.y,panel+i%P,p)):0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m=0u;m<RM;++m) {
                uint row=(uint(sg)+m*NSG)*8u;
                if (row>=R) continue; // SIMD-uniform; no barrier inside
                for (uint d=0u;d<P;d+=8u) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a,qt+row*P+d,P);
                    for (uint n=0u;n<BK/8u;++n) {
                        simdgroup_float8x8 b;
                        simdgroup_load(b,stage+n*8u*P+d,P,ulong2(0),true);
                        simdgroup_multiply_accumulate(accum[m][n],a,b,accum[m][n]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint m=0u;m<RM;++m) {
            uint row=(uint(sg)+m*NSG)*8u;
            if (row>=R) continue;
            for (uint n=0u;n<BK/8u;++n) simdgroup_store(accum[m][n],scores+row*BK+n*8u,BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane==0) for (uint r=0u;r<OWN;++r) {
            uint row=uint(sg)+r*NSG, token=(packed0+row)/gqa;
            if (row>=R || token>=p.queries) continue;
            uint pos=uint(positions[token]);
            float next=ml[row]; bool valid=false;
            for (uint j=0u;j<BK;++j) if (pages[j]>=0 && atlas_visible(pos,first+j,p.window)) {
                next=max(next,scores[row*BK+j]); valid=true;
            }
            if (!valid) continue;
            float l=ml[R+row], m=ml[row];
            float a=l==0.0f?0.0f:(m==next?1.0f:precise::exp(m-next));
            float sum=0.0f;
            for (uint j=0u;j<BK;++j) if (pages[j]>=0 && atlas_visible(pos,first+j,p.window)) {
                float s=scores[row*BK+j], w=s==next?1.0f:precise::exp(s-next);
                weights[row*BK+j]=w; sum+=w;
            }
            ml[row]=next; ml[R+row]=fma(l,a,sum); ml[2u*R+row]=a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint panel=0u;panel<D;panel+=P) {
            for (uint i=tid;i<BK*P;i+=T) {
                uint j=i/P;
                size_t pt=pages[j]>=0?size_t(pages[j])*p.page_size+(first+j)%p.page_size:0ul;
                stage[i]=pages[j]>=0?atlas_widen(atlas_v(k,v,factor,pt,group.y,panel+i%P,p)):0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m=0u;m<RM;++m) {
                uint row=(uint(sg)+m*NSG)*8u;
                if (row>=R) continue;
                for (uint d=0u;d<P;d+=8u) {
                    simdgroup_float8x8 pv(0.0f);
                    for (uint j=0u;j<BK;j+=8u) {
                        simdgroup_float8x8 a,b;
                        simdgroup_load(a,weights+row*BK+j,BK);
                        simdgroup_load(b,stage+j*P+d,P);
                        simdgroup_multiply_accumulate(pv,a,b,pv);
                    }
                    simdgroup_store(pv,qt+row*P+d,P);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r=0u;r<OWN;++r) {
                uint row=uint(sg)+r*NSG;
                if (row>=R) continue;
                for (uint d=uint(lane);d<P;d+=32u) {
                    uint slot=(panel+d)/32u;
                    u[r][slot]=fma(u[r][slot],ml[2u*R+row],qt[row*P+d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint r=0u;r<OWN;++r) {
        uint row=uint(sg)+r*NSG, packed=packed0+row, token=packed/gqa;
        if (row>=R || token>=p.queries) continue;
        uint head=group.y*gqa+packed%gqa; float l=ml[R+row];
        float inv=l>0.0f?1.0f/l:0.0f;
        for (uint d=0u;d<D/32u;++d)
            atlas_store(out,(size_t(token)*16u+head)*D+uint(lane)+d*32u,
                u[r][d]*inv,p.output_kind);
    }
}
#define ATLAS_MATRIX_ENTRY(NAME,D,R,BK,P,T) \
    kernel void NAME(ATLAS_ARGUMENTS) { \
        threadgroup float qt[R*P],stage[BK*P],scores[R*BK],weights[R*BK],ml[3*R]; \
        threadgroup int pages[BK]; \
        atlas_matrix_body<D,R,BK,P,T>(ATLAS_PASS,qt,stage,scores,weights,ml,pages); \
    }
