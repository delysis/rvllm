// Host-only concurrency emulation of the ACTUAL common MSL body. Not a Metal
// compiler, SIMD instruction, resource, safety or performance qualification.
// This file contains no implementation used by the production Rust backend.
#include <algorithm>
#include <array>
#include <functional>
#include <ucontext.h>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using uint = uint32_t; using ushort = uint16_t; using uchar = uint8_t;
struct uint3 { uint x,y,z; };
using std::max; using std::size_t;
inline float fma(float a,float b,float c) { return std::fma(a,b,c); }
namespace precise { inline float exp(float x) { return std::exp(x); } }
template<class T,class U> T as_type(U x) { return std::bit_cast<T>(x); }
namespace mem_flags { constexpr int mem_threadgroup=0; }
// Cooperative stackful fibers avoid hundreds of oversubscribed OS threads.
// A barrier releases only its actual participants; deadlock is an assertion,
// not a timeout silently treated as a pass. These are host emulation primitives.
struct Barrier {
    uint expected; std::vector<uint> waiting;
    explicit Barrier(uint n):expected(n) {}
};
struct Simd { Barrier sync{32}; std::array<float,32> values{}; };
struct Scheduler {
    struct Fiber { ucontext_t context{}; std::unique_ptr<char[]> stack;
        bool ready=true,done=false; };
    ucontext_t main_context{};
    std::vector<Fiber> fibers;
    std::function<void(uint)> work;
    uint current=0;
    std::vector<std::unique_ptr<Simd>> simds;
    Barrier group;
    explicit Scheduler(uint n):fibers(n),group(n) {
        for(uint i=0;i<n/32;i++) simds.push_back(std::make_unique<Simd>());
    }
    void wait(Barrier& barrier) {
        uint owner=current;
        fibers[owner].ready=false;
        barrier.waiting.push_back(owner);
        if(barrier.waiting.size()==barrier.expected) {
            for(uint id:barrier.waiting) fibers[id].ready=true;
            barrier.waiting.clear();
        }
        if(swapcontext(&fibers[owner].context,&main_context)) throw std::runtime_error("swapcontext");
    }
    void run(std::function<void(uint)> function);
};
Scheduler* active=nullptr;
void fiber_entry() {
    uint owner=active->current;
    active->work(owner);
    active->fibers[owner].done=true;
}
void Scheduler::run(std::function<void(uint)> function) {
    work=std::move(function); active=this;
    for(auto& fiber:fibers) {
        fiber.stack=std::make_unique<char[]>(65536);
        if(getcontext(&fiber.context)) throw std::runtime_error("getcontext");
        fiber.context.uc_stack.ss_sp=fiber.stack.get();
        fiber.context.uc_stack.ss_size=65536;
        fiber.context.uc_link=&main_context;
        makecontext(&fiber.context,fiber_entry,0);
    }
    while(true) {
        uint done=0,runnable=0;
        for(uint i=0;i<fibers.size();i++) {
            if(fibers[i].done) { done++; continue; }
            if(!fibers[i].ready) continue;
            runnable++; current=i;
            if(swapcontext(&main_context,&fibers[i].context)) throw std::runtime_error("dispatch context");
        }
        if(done==fibers.size()) break;
        if(!runnable) throw std::runtime_error("divergent/deadlocked barrier");
    }
}
inline float simd_shuffle_down(float value,uint delta) {
    uint lane=active->current%32;
    auto& simd=*active->simds[active->current/32];
    simd.values[lane]=value; active->wait(simd.sync);
    float out=simd.values[lane+delta<32?lane+delta:lane];
    active->wait(simd.sync); return out;
}
inline float simd_broadcast(float value,uint source) {
    auto& simd=*active->simds[active->current/32];
    simd.values[active->current%32]=value; active->wait(simd.sync);
    float out=simd.values[source]; active->wait(simd.sync); return out;
}
inline void threadgroup_barrier(int) { active->wait(active->group); }
#define device
#define constant const
#define threadgroup
#include "../../crates/rvllm-apple-metal/src/research_shaders/global_decode_common.metal"
#undef device
#undef constant
#undef threadgroup

float widen(ushort x) { return std::bit_cast<float>(uint(x)<<16); }
// Independently expressed nearest-even conversion, not the shader helper.
ushort rounded(float x) {
    uint bits=std::bit_cast<uint>(x),upper=bits>>16,lower=bits&65535;
    if((bits&0x7fffffff)>0x7f800000) return ushort(upper|0x40);
    if(lower>32768 || (lower==32768 && (upper&1))) upper++;
    return ushort(upper);
}
struct Fixture {
    GlobalDecodeParams p{1,16,1,512,7,8,11,0,1,1};
    std::vector<ushort> q,k,v;
    std::vector<int> table;
    int context,position;
    explicit Fixture(uint n) : context(n),position(n-1) {
        p.max_blocks=(n-1)/7+3; p.num_blocks=p.max_blocks+3;
        q.resize(8192); k.assign(size_t(p.num_blocks)*7*512,0x7fc1); v=k;
        table.resize(p.max_blocks);
        for(uint i=0;i<p.max_blocks;i++) table[i]=p.num_blocks-1-i;
        auto noise=[](uint x) { x^=x<<13; x^=x>>17; x^=x<<5;
            return float(x&65535)/32768.0f-1.0f; };
        std::array<float,4> scales{0.0001f,0.1f,1.0f,4.0f};
        for(uint i=0;i<8192;i++) q[i]=rounded(noise(i*179+1009)*scales[(i/512)%4]);
        for(uint t=0;t<n;t++) for(uint d=0;d<512;d++) {
            uint seed=t*12347+d*139+913;
            k[base(t)+d]=rounded(noise(seed)*0.5f);
            v[base(t)+d]=rounded(noise(seed+7919));
        }
    }
    size_t base(uint token) const { return (size_t(table[token/7])*7+token%7)*512; }
};
void check(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
float scheduled_dot(const ushort* q,const ushort* k,uint panel) {
    check(panel==64 || panel==128,"staging panel admission");
    float score=0;
    for(uint first=0;first<512;first+=64) {
        std::array<float,32> partials{};
        for(uint l=0;l<32;l++) for(uint d=l;d<64;d+=32)
            partials[l]=std::fma(widen(q[first+d]),widen(k[first+d]),partials[l]);
        for(uint delta=16;delta;delta/=2) for(uint l=0;l<delta;l++) partials[l]+=partials[l+delta];
        score+=partials[0];
    }
    return score;
}
std::vector<float> serial(const Fixture& f,uint panel) {
    std::vector<float> out(8192);
    for(uint h=0;h<16;h++) {
        float m=-INFINITY,l=0; std::array<float,512> u{};
        for(int t=0;t<=f.position;t++) {
            if(f.table[t/7]<0) continue;
            size_t b=f.base(t); float s=scheduled_dot(&f.q[h*512],&f.k[b],panel);
            float next=std::max(m,s);
            float a=l==0?0:(m==next?1:std::exp(m-next));
            float w=s==next?1:std::exp(s-next);
            l=std::fma(l,a,w);
            for(uint d=0;d<512;d++) u[d]=std::fma(w,widen(f.v[b+d]),u[d]*a);
            m=next;
        }
        float inv=l>0?1/l:0;
        for(uint d=0;d<512;d++) out[h*512+d]=u[d]*inv;
    }
    return out;
}
double dense_error(const Fixture& f,const std::vector<float>& result) {
    double error=0;
    for(uint h=0;h<16;h++) {
        std::vector<double> scores; std::vector<size_t> bases;
        for(int t=0;t<=f.position;t++) {
            if(f.table[t/7]<0) continue;
            size_t b=f.base(t); double dot=0,absolute=0;
            for(uint d=0;d<512;d++) {
                double x=double(widen(f.q[h*512+d]))*widen(f.k[b+d]); dot+=x; absolute+=std::abs(x);
            }
            for(uint panel:{64u,128u}) check(std::abs(double(scheduled_dot(&f.q[h*512],&f.k[b],panel))-dot)
                <=32*double(std::numeric_limits<float>::epsilon())*absolute+1e-30,"FP64 dot bound");
            scores.push_back(dot); bases.push_back(b);
        }
        if(scores.empty()) { for(uint d=0;d<512;d++) check(result[h*512+d]==0,"empty identity"); continue; }
        double m=*std::max_element(scores.begin(),scores.end()),denom=0;
        std::array<double,512> u{};
        for(size_t i=0;i<scores.size();i++) {
            double w=std::exp(scores[i]-m); denom+=w;
            for(uint d=0;d<512;d++) u[d]+=w*double(widen(f.v[bases[i]+d]));
        }
        for(uint d=0;d<512;d++) error=std::max(error,std::abs(result[h*512+d]-u[d]/denom));
    }
    check(error<=2e-5,"independent FP64 output tolerance"); return error;
}
template<uint R,uint P,uint T>
std::vector<uint> run(const Fixture& f,bool bf16=false,uint actual_threads=T) {
    const size_t elements=bf16?4096:8192;
    std::vector<uint> raw(elements+16,0xa5a5a5a5u);
    std::fill(raw.begin()+8,raw.end()-8,0xffffffffu);
    GlobalDecodeParams params=f.p; params.output_kind=bf16?0:1;
    for(uint group=0;group<16/R;group++) {
        std::array<ushort,R*512> qt{}; std::array<ushort,8*P> stage{};
        std::array<float,R*8> scores{},alpha{},weight{}; std::array<int,8> pages{};
        Scheduler scheduler(T);
        scheduler.run([&](uint tid) {
            global_decode_body<R,P,T>(f.q.data(),f.k.data(),f.v.data(),
                reinterpret_cast<uchar*>(raw.data()+8),f.table.data(),&f.context,&f.position,params,
                {group,0,0},tid,tid/32,tid%32,{actual_threads,1,1},qt.data(),stage.data(),
                scores.data(),alpha.data(),weight.data(),pages.data());
        });
    }
    for(size_t i=0;i<8;i++) check(raw[i]==0xa5a5a5a5u && raw[raw.size()-1-i]==0xa5a5a5a5u,"output guards");
    return std::vector<uint>(raw.begin()+8,raw.end()-8);
}
template<uint R,uint P,uint T> void family(uint& positive,uint& rejected,double& max_error) {
    for(uint mode=0;mode<7;mode++) {
        Fixture f(mode==0?1:mode==1?9:33);
        if(mode==2) f.table[1]=-17;
        if(mode==3) std::fill(f.table.begin(),f.table.end(),-1);
        if(mode==4 || mode==5) {
            f.position=8;
            for(uint t=9;t<33;t++) {
                size_t b=f.base(t); std::fill_n(f.k.begin()+b,512,0x7fc1); std::fill_n(f.v.begin()+b,512,0x7fc1);
            }
            if(mode==5) f.context=9;
        }
        if(mode==6) std::fill(f.q.begin(),f.q.end(),0);
        auto expected=serial(f,P); auto actual=run<R,P,T>(f);
        max_error=std::max(max_error,dense_error(f,expected));
        for(size_t i=0;i<8192;i++) check(actual[i]==std::bit_cast<uint>(expected[i]),"exact FP32 cooperative body");
        auto bf=run<R,P,T>(f,true);
        for(size_t i=0;i<4096;i++) check(bf[i]==(uint(rounded(expected[2*i])) | (uint(rounded(expected[2*i+1]))<<16)),"once-rounded BF16");
        check(actual==run<R,P,T>(f),"repeated execution stability"); positive++;
    }
    for(uint mode=0;mode<5;mode++) {
        Fixture bad(9);
        if(mode==0) bad.context=0;
        if(mode==1) bad.position=bad.context;
        if(mode==2) bad.table[0]=bad.p.num_blocks;
        if(mode==3) bad.p.scale=0.5f;
        auto result=run<R,P,T>(bad,false,mode==4?T-1:T);
        check(std::all_of(result.begin(),result.end(),[](uint x){return x==0xffffffffu;}),"rejected output touched"); rejected++;
    }
    std::cout<<"r"<<R<<"p"<<P<<"t"<<T<<" passed host source-body emulation"<<std::endl;
}
int main() {
    try {
        static_assert(sizeof(GlobalDecodeParams)==40);
        uint positive=0,rejected=0; double error=0;
        family<8,64,64>(positive,rejected,error); family<8,64,128>(positive,rejected,error);
        family<8,128,64>(positive,rejected,error); family<8,128,128>(positive,rejected,error);
        family<16,64,64>(positive,rejected,error); family<16,64,128>(positive,rejected,error);
        family<16,128,64>(positive,rejected,error); family<16,128,128>(positive,rejected,error);
        std::cout<<"positive_cases="<<positive<<" rejected_cases="<<rejected<<" max_fp64_abs_error="<<error
            <<"\nHOST_ONLY: Metal compiler/device, Rust, full-model and performance gates NOT run\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
