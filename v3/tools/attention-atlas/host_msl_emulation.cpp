// Host-only execution of transformed atlas MSL bodies. Not a Metal
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
using std::max; using std::min; using std::size_t;
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

inline void simdgroup_barrier(int) { active->wait(active->simds[active->current/32]->sync); }
struct ulong2 { explicit ulong2(unsigned long) {} };
// Every emulated lane holds a complete fragment. This validates matrix indexing
// and algebra ONLY, not native fragment layout, instruction order or rounding.
struct simdgroup_float8x8 {
    std::array<float,64> values{};
    explicit simdgroup_float8x8(float v=0.0f) {values.fill(v);}
};
void simdgroup_load(simdgroup_float8x8& x,const float* p,uint stride) {
    for(uint r=0;r<8;r++)for(uint c=0;c<8;c++)x.values[r*8+c]=p[r*stride+c];
}
void simdgroup_load(simdgroup_float8x8& x,const float* p,uint stride,ulong2,bool transposed) {
    for(uint r=0;r<8;r++)for(uint c=0;c<8;c++)x.values[r*8+c]=transposed?p[c*stride+r]:p[r*stride+c];
}
void simdgroup_store(const simdgroup_float8x8& x,float* p,uint stride) {
    for(uint r=0;r<8;r++)for(uint c=0;c<8;c++)p[r*stride+c]=x.values[r*8+c];
}
void simdgroup_multiply_accumulate(simdgroup_float8x8& out,const simdgroup_float8x8& a,
    const simdgroup_float8x8& b,const simdgroup_float8x8& add) {
    auto x=add;
    for(uint r=0;r<8;r++)for(uint c=0;c<8;c++)for(uint k=0;k<8;k++)
        x.values[r*8+c]=std::fma(a.values[r*8+k],b.values[k*8+c],x.values[r*8+c]);
    out=x;
}
#define device
#define constant const
#define threadgroup
#define kernel
#include "atlas_under_test.hpp"
#undef device
#undef constant
#undef threadgroup
#undef kernel
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
ushort bf(float x){uint b=std::bit_cast<uint>(x);uint high=b>>16,low=b&65535;
    if(low>32768 || (low==32768 && (high&1)))high++;return ushort(high);}
float fp(ushort x){return std::bit_cast<float>(uint(x)<<16);}
struct Fixture {
    AtlasParams p{};
    std::vector<ushort> q,k,v,gamma;
    std::vector<float> factor,cs,sn;
    std::vector<int> pages,positions;
    Fixture(uint dim,uint queries,uint keys,uint first,uint pattern,bool raw){
        p={0x41540001u,queries,keys,dim==256?8u:1u,dim,dim==256?1024u:0u,7,keys/7+3,keys/7+3,1,1,raw?1u:0u,1,1,64,32};
        pages.resize(p.max_blocks);positions.resize(queries);
        for(uint i=0;i<queries;i++)positions[i]=first+i;
        for(uint b=0;b<p.max_blocks;b++)pages[b]=p.physical_blocks-b-1;
        q.resize(queries*16*dim);k.resize(size_t(p.physical_blocks)*7*p.kv_heads*dim);v.resize(k.size());
        auto noise=[](uint x){x^=x<<13;x^=x>>17;x^=x<<5;return float(x&65535)/32768.0f-1.0f;};
        for(uint i=0;i<q.size();i++)q[i]=bf(pattern==2?0.0f:noise(i*317+1)*0.3f);
        for(uint i=0;i<k.size();i++){k[i]=bf(noise(i*131+17)*0.3f);v[i]=bf(noise(i*257+31));}
        if(pattern==1)pages[0]=-1;
        if(pattern==3)std::fill(pages.begin(),pages.end(),-1);
        // Sparse boundary fixture retains the oldest and newest visible pages.
        // Absolute positions still cross 1024; emulation avoids millions of
        // uninformative software-shuffle context switches in interior pages.
        if(pattern==5)for(uint b=1;b<p.max_blocks;b++)if(b!=(first+queries-1)/7)pages[b]=-1;
        if(pattern==4)for(uint b=(first+queries-1)/7+1;b<p.max_blocks;b++)pages[b]=2147483647;
        gamma.resize(dim);factor.resize(p.physical_blocks*7,1.25f);cs.resize(keys*64);sn.resize(keys*64);
        for(uint d=0;d<dim;d++)gamma[d]=bf(0.5f+(d%23)/23.0f);
        for(uint t=0;t<keys;t++)for(uint d=0;d<64;d++){
            double angle=t*std::pow(1e6,-2.0*d/512.0);cs[t*64+d]=std::cos(angle);sn[t*64+d]=std::sin(angle);
        }
    }
    size_t physical(uint token)const{return size_t(pages[token/7])*7+token%7;}
    std::pair<float,float> kv(uint token,uint head,uint d)const{
        size_t pt=physical(token),base=(pt*p.kv_heads+head)*p.dim;
        if(!p.cache_kind)return {fp(k[base+d]),fp(v[base+d])};
        auto scaled=[&](uint c){return fp(bf((fp(k[base+c])*factor[pt])*fp(gamma[c])));};
        float value=fp(bf(fp(k[base+d])*factor[pt]));float key=scaled(d);
        if(d<64 || (d>=256 && d<320)){
            uint i=d<64?d:d-256;float x0=scaled(i),x1=scaled(i+256),c=cs[token*64+i],s=sn[token*64+i];
            key=fp(bf(d<64?std::fma(-x1,s,x0*c):std::fma(x0,s,x1*c)));
        }
        return {key,value};
    }
};
std::vector<double> oracle(const Fixture& f){
    uint D=f.p.dim;std::vector<double> out(f.q.size());
    for(uint t=0;t<f.p.queries;t++)for(uint h=0;h<16;h++){
        uint pos=f.positions[t],lo=f.p.window?(pos+1>f.p.window?pos+1-f.p.window:0):0;
        std::vector<double> scores;std::vector<uint> keys;
        for(uint k=lo;k<=pos;k++)if(f.pages[k/7]>=0){
            double dot=0;for(uint d=0;d<D;d++)dot+=double(fp(f.q[(t*16+h)*D+d]))*f.kv(k,h/(16/f.p.kv_heads),d).first;
            scores.push_back(dot);keys.push_back(k);
        }
        if(keys.empty())continue;
        double m=*std::max_element(scores.begin(),scores.end()),l=0;
        for(uint j=0;j<keys.size();j++){
            double w=std::exp(scores[j]-m);l+=w;
            for(uint d=0;d<D;d++)out[(t*16+h)*D+d]+=w*f.kv(keys[j],h/(16/f.p.kv_heads),d).second;
        }
        for(uint d=0;d<D;d++)out[(t*16+h)*D+d]/=l;
    }return out;
}
uint positive=0,negative=0;double maximum=0;
template<uint D,uint R,uint BK,uint P,uint T,bool Key,bool Matrix=false>
void run_case(uint Q,uint S,uint first,uint pattern,uint splits=1,bool raw=false){
    Fixture f(D,Q,S,first,pattern,raw);auto p=f.p;p.rows=R;p.keys=BK;p.panel=P;p.threads=T;p.splits=splits;
    uint status=999;atlas_validate(f.pages.data(),f.positions.data(),&status,p,0);check(status==0,"valid metadata refused");
    std::vector<uint> rawout(f.q.size()+16,0xa5a5a5a5);auto* out=reinterpret_cast<uchar*>(rawout.data()+8);
    std::vector<float> partial(size_t(Q)*16*splits*(D+2),-12345.0f);
    for(uint h=0;h<p.kv_heads;h++)for(uint group=0;group<(Q*16/p.kv_heads+R-1)/R;group++)for(uint split=0;split<splits;split++){
        std::array<ushort,R*P> qt{};std::array<ushort,BK*P> stage{};
        std::array<float,R*P> qm{};std::array<float,BK*P> sm{};
        std::array<float,R*BK> scores{},alpha{},weights{};std::array<float,3*R> ml{};std::array<int,BK> pages{};
        Scheduler scheduler(T);
        scheduler.run([&](uint tid){
            if constexpr(Matrix)atlas_matrix_body<D,R,BK,P,T>(f.q.data(),f.k.data(),f.v.data(),f.pages.data(),f.positions.data(),out,
                partial.data(),&status,f.factor.data(),f.gamma.data(),f.cs.data(),f.sn.data(),p,{group,h,split},tid,tid/32,tid%32,{T,1,1},
                qm.data(),sm.data(),scores.data(),weights.data(),ml.data(),pages.data());
            else atlas_coop_body<D,R,BK,P,T,Key>(f.q.data(),f.k.data(),f.v.data(),f.pages.data(),f.positions.data(),out,
                partial.data(),&status,f.factor.data(),f.gamma.data(),f.cs.data(),f.sn.data(),p,{group,h,split},tid,tid/32,tid%32,{T,1,1},
                qt.data(),stage.data(),scores.data(),alpha.data(),weights.data(),ml.data(),pages.data());
        });
    }
    if(splits>1)for(uint row=0;row<Q*16;row++){
        Scheduler scheduler(32);scheduler.run([&](uint tid){atlas_merge(out,partial.data(),&status,p,{row,0,0},tid,{32,1,1});});
    }
    auto expected=oracle(f);
    for(uint i=0;i<f.q.size();i++){
        double actual=std::bit_cast<float>(rawout[i+8]);check(std::isfinite(actual),"nonfinite output");
        double e=std::abs(actual-expected[i]);maximum=std::max(maximum,e);check(e<2e-5,"FP64 output error");
    }
    for(uint i=0;i<8;i++)check(rawout[i]==0xa5a5a5a5 && rawout[rawout.size()-1-i]==0xa5a5a5a5,"output guard");
    auto original_positions=f.positions;auto original_pages=f.pages;
    for(uint invalid=0;invalid<4;invalid++) {
        auto bad=p;f.positions=original_positions;f.pages=original_pages;
        if(invalid==0)f.positions[0]=p.live_keys;
        if(invalid==1)bad.abi=0;
        if(invalid==2)f.positions[0]=-1;
        if(invalid==3){uint pos=uint(f.positions[0]);uint lo=p.window?(pos+1>p.window?pos+1-p.window:0):0;f.pages[lo/7]=p.physical_blocks;}
        atlas_validate(f.pages.data(),f.positions.data(),&status,bad,0);
        check(status==(invalid==1?1u:invalid==3?3u:2u),"negative metadata admitted");
        auto saved=rawout;auto saved_partial=partial;
        std::array<ushort,R*P> qt{};std::array<ushort,BK*P> stage{};
        std::array<float,R*P> qm{};std::array<float,BK*P> sm{};
        std::array<float,R*BK> scores{},alpha{},weights{};std::array<float,3*R> ml{};std::array<int,BK> pages{};
        Scheduler scheduler(T);scheduler.run([&](uint tid){
            if constexpr(Matrix)atlas_matrix_body<D,R,BK,P,T>(f.q.data(),f.k.data(),f.v.data(),f.pages.data(),f.positions.data(),out,
                partial.data(),&status,f.factor.data(),f.gamma.data(),f.cs.data(),f.sn.data(),bad,{0,0,0},tid,tid/32,tid%32,{T,1,1},
                qm.data(),sm.data(),scores.data(),weights.data(),ml.data(),pages.data());
            else atlas_coop_body<D,R,BK,P,T,Key>(f.q.data(),f.k.data(),f.v.data(),f.pages.data(),f.positions.data(),out,
                partial.data(),&status,f.factor.data(),f.gamma.data(),f.cs.data(),f.sn.data(),bad,{0,0,0},tid,tid/32,tid%32,{T,1,1},
                qt.data(),stage.data(),scores.data(),alpha.data(),weights.data(),ml.data(),pages.data());
        });
        check(saved==rawout && saved_partial==partial,"refused body wrote output or partial states");negative++;
    }
    positive++;
    std::cout<<"passed D"<<D<<" R"<<R<<" K"<<BK<<" P"<<P<<" T"<<T<<" q"<<Q<<" pattern"<<pattern<<" splits"<<splits<<" mma"<<Matrix<<" raw"<<raw<<std::endl;
}
int main(int argc,char** argv){try{
    if(argc==2){std::string mode=argv[1];
        if(mode=="core"){run_case<512,16,8,64,128,true>(1,17,16,0);run_case<256,8,16,128,64,false>(3,17,14,0);}
        else if(mode=="split")run_case<512,8,8,128,64,true>(1,17,16,0,4);
        else if(mode=="window")run_case<256,16,8,64,128,true>(1,1031,1024,5);
        else if(mode=="raw")run_case<512,16,8,64,128,true>(1,9,8,0,4,true);
        else throw std::runtime_error("unknown host test mode");
        return 0;
    }
    static_assert(sizeof(AtlasParams)==64);
    for(uint mode=0;mode<5;mode++){
        run_case<512,16,8,64,128,true>(1,17,mode==4?7:16,mode);
        run_case<256,8,16,128,64,false>(3,17,mode==4?5:14,mode);
        run_case<512,8,16,64,64,false,true>(1,17,mode==4?7:16,mode);
    }
    for(uint splits:{2u,4u,8u,16u,32u})run_case<512,8,8,128,64,true>(1,9,8,1,splits);
    run_case<256,16,8,64,128,true>(3,1031,1024,5,4);
    run_case<256,8,16,64,128,false,true>(3,19,16,0);
    run_case<512,16,32,128,128,false,true>(2,17,15,0);
    run_case<512,16,8,64,128,true>(1,9,8,0,4,true);
    run_case<512,8,16,64,64,false,true>(1,9,8,0,1,true);
    run_case<256,32,16,64,128,false>(7,17,10,0);
    run_case<512,32,32,64,128,false,true>(3,17,14,0);
    run_case<256,1,1,64,32,true>(1,9,8,0);
    run_case<256,2,8,64,64,true>(1,9,8,0,4);
    run_case<256,2,8,128,128,true>(1,9,8,0);
    run_case<256,4,8,64,64,true>(3,17,14,0,4);
    run_case<512,1,1,64,32,true>(1,9,8,0,4);
    for(uint high:{0x3f80u,0x3f81u,0xbf80u,0xbf81u,0x0080u})for(uint low:{0u,32767u,32768u,32769u,65535u}){
        float x=std::bit_cast<float>((high<<16)|low);check(atlas_round(x)==bf(x),"BF16 nearest-even helper");
    }
    std::cout<<"positive="<<positive<<" negative="<<negative<<" max_abs_fp64="<<maximum
        <<"\nHOST EMULATION ONLY; native compiler, Rust, device, model and performance gates not run\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}}
