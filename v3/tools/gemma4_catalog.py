#!/usr/bin/env python3
"""Read/validate the reviewed Metal catalog. Data only; never run a command."""
from __future__ import annotations
import argparse,json,re,stat,sys
from pathlib import Path
MAX_JSON=128*1024
PATH=Path(__file__).resolve().with_name('gemma4_metal_catalog.json')
NAMES=('off','metal-short-mma16x64','metal-rounded-gate32','metal-gqa-kv8',
       'metal-mma32-prefetch','metal-attn-q4','metal-rms-simd32','metal-mma32-f32',
       'metal-long-mma32x64','metal-mma32-load4','metal-rmsnorm-simd256',
       'metal-load4-m16n32k64','metal-load4-m16n64k64','metal-load4-m32n32k64',
       'metal-load4-m32n64k32','metal-load4-m32n64k64','metal-load4-m32n64k128',
       'metal-load4-m64n64k64',
       'metal-global-d512-r8p64t64','metal-global-d512-r8p64t128',
       'metal-global-d512-r8p128t64','metal-global-d512-r8p128t128',
       'metal-global-d512-r16p64t64','metal-global-d512-r16p64t128',
       'metal-global-d512-r16p128t64','metal-global-d512-r16p128t128',
       'metal-global-d512-r1p128t32',
       'metal-global-d512-split-r8s256t128',
       'metal-global-d512-split-mma_r8k32s256t128',
       'metal-global-d512-split-coopkey_r8k8p64t128s32',
       'metal-global-d512-atlas_r16k16p64t128',
       'metal-global-d512-atlas_r16k32p64t128',
       'metal-global-d512-atlas_tile_r16k16p64t128',
       'metal-global-d512-atlas_tile_r16k32p64t128',
       'metal-global-d512-atlas_mma_r16k16p64t128',
       'metal-global-d512-atlas_mma_r16k32p64t128',
       'metal-global-d512-atlas_mma_r16k16p128t128',
       'metal-global-d512-atlas_mma_r8k32p64t128',
       'metal-global-d512-atlas_mma_r16k16p64t64',
       'metal-global-d512-atlas_mma_r16k64p64t128',
       'metal-ffn-bf16-r4-sg2','metal-qmv-w4-g32-r8-sg2',
       'metal-qmv-w8-g32-r8-sg2','metal-global-d512-short-r4t128',
       'metal-qmv-w4-g32-r4-sg8-k8','metal-qmv-w8-g32-r4-sg8-k8',
       'metal-donor12b-sg8','metal-donor12b-sg4')
SUPPORT_SOURCES=(
    'crates/rvllm-apple-metal/src/research_shaders/load4_tiled_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/global_decode_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/global_decode_split_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/global_decode_split_matrix_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/global_decode_matrix_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/donor12b_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/decode_round_common.metal',
    'crates/rvllm-apple-metal/src/research_shaders/qmv_g32_r8_sg2_common.metal',
)
CONTRACTS={'baseline','storage-boundaries-preserved','fp32-online-softmax',
           'same-contraction-order','reduction-order-change','operand-lowering-change',
           'layout-only-bitwise-fp32-gate',
           'bf16-fp32-fixed64-tree-online-once-rounded',
           'bf16-fp32-fixed64-tree-online-k16-once-rounded',
           'bf16-fp32-fixed64-tree-online-k32-once-rounded',
           'bf16-fp32-fixed64-tree-per-tile-k16-once-rounded',
           'bf16-fp32-fixed64-tree-per-tile-k32-once-rounded',
           'bf16-fp32-simd-matrix-per-tile-once-rounded',
           'bf16-fp32-simd-matrix-split-sufficient-stat-once-rounded',
           'bf16-fp32-split-sufficient-stat-once-rounded',
           'bf16-paged-fp32-online-unsplit-rne',
           'bf16-projection-boundary-gelu-fp32-rne',
           'authenticated-g32-fp16-scales-bf16-qmv-fp32-rne',
           'signed-g32-fp16-scales-bf16-boundaries-fp32-donor-order-not-bitwise-incumbent'}

def decode(text: str) -> dict:
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate JSON key: '+k)
            result[k]=v
        return result
    def nonfinite(value):raise ValueError('nonfinite JSON number: '+value)
    try:return json.loads(text,object_pairs_hook=unique,parse_constant=nonfinite)
    except RecursionError as e:raise ValueError('JSON nesting too deep') from e

def validate(value: dict) -> dict:
    if (type(value) is not dict or set(value)!={'schema','dispatch_schema','default','device_qualified','candidates'}
        or value['schema']!='rvllm.metal.research-catalog.v1'
        or value['dispatch_schema']!='rvllm.metal.research-dispatch.v5'
        or value['default']!='off' or value['device_qualified'] is not False):
        raise ValueError('unrecognized catalog contract')
    candidates=value['candidates']
    if type(candidates) is not list or len(candidates)!=len(NAMES):raise ValueError('all reviewed candidates required')
    all_kernels=set();files=set()
    for expected,row in zip(NAMES,candidates):
        if type(row) is not dict or set(row)!={'name','kernels','source_file','min_tokens','max_tokens','window_independent','numerical_contract','budgets'}:
            raise ValueError('invalid candidate fields')
        if row['name']!=expected:raise ValueError('candidate order, identity or duplication mismatch')
        if type(row['window_independent']) is not bool or row['numerical_contract'] not in CONTRACTS:
            raise ValueError('unknown numerical/window contract')
        lo,hi=row['min_tokens'],row['max_tokens']
        if type(lo) is not int or type(hi) is not int or not 0<=lo<=hi<=1024:raise ValueError('invalid token bounds')
        ks=row['kernels'];bs=row['budgets']
        if type(ks) is not list or type(bs) is not list or len(ks)!=len(bs) or len(ks)>12:
            raise ValueError('kernel budget count')
        if expected=='off':
            if ks or row['source_file'] is not None or lo!=0 or hi!=0 or row['numerical_contract']!='baseline':
                raise ValueError('baseline must not bind candidate source')
        else:
            path=row['source_file']
            if (type(path) is not str or not re.fullmatch(r'crates/rvllm-apple-metal/src/research_shaders/[a-z0-9_]+\.metal',path)
                or path in files or not ks or lo<1):raise ValueError('candidate source or shape invalid')
            files.add(path)
        for kernel,budget in zip(ks,bs):
            if type(kernel) is not str or not re.fullmatch(r'(research|wave2)_[A-Za-z0-9_]+',kernel) or kernel in all_kernels:
                raise ValueError('unknown, malformed or duplicate kernel')
            all_kernels.add(kernel)
            if (type(budget) is not dict or set(budget)!={'kernel','threads','source_shared_bytes'}
                or budget['kernel']!=kernel or type(budget['threads']) is not int or budget['threads'] not in (32,64,128,256,512)
                or type(budget['source_shared_bytes']) is not int or not 0<=budget['source_shared_bytes']<=32768
                or budget['source_shared_bytes']%4):raise ValueError('invalid source resource budget')
    if len(all_kernels)!=86:raise ValueError('incomplete append-only entry registry')
    return value

def load(path: Path=PATH) -> dict:
    path=Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('catalog must be a regular file, not a symlink')
    with path.open('rb') as f:
        if not stat.S_ISREG(__import__('os').fstat(f.fileno()).st_mode):raise ValueError('catalog changed type')
        raw=f.read(MAX_JSON+1)
    if len(raw)>MAX_JSON:raise ValueError('catalog exceeds byte bound')
    return validate(decode(raw.decode('utf-8')))

def exports(value: dict) -> dict[str,tuple[str,...]]:
    return {r['name']:tuple(r['kernels']) for r in value['candidates']}

def sources(value: dict) -> list[str]:
    return [r['source_file'] for r in value['candidates'] if r['source_file'] is not None]+list(SUPPORT_SOURCES)

def verify_exported(reviewed: dict,actual: dict) -> None:
    if validate(actual)!=validate(reviewed):raise ValueError('runtime catalog differs from reviewed golden')

def check_source(value: dict,candidate: str,source: str) -> None:
    expected=exports(value).get(candidate)
    # Exclude comments: documentary examples must not count as compiled entry points.
    code=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
    names=re.findall(r'\bkernel\s+void\s+((?:research|wave2)_\w+)\s*\(',code)
    if expected is None or not source.strip() or len(names)!=len(expected) or set(names)!=set(expected):
        raise ValueError('wrong research entry points in '+candidate)

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--names',action='store_true');group.add_argument('--sources',action='store_true')
    group.add_argument('--verify-exported',type=Path)
    group.add_argument('--check-source',nargs=2,metavar=('CANDIDATE','SOURCE'))
    args=parser.parse_args()
    try:
        value=load()
        if args.names:print('\n'.join(exports(value)))
        elif args.sources:
            workspace=PATH.parents[1]
            for path in sources(value):
                p=workspace/path
                # The reviewed source list cannot escape through a symlinked directory.
                if any(x.is_symlink() for x in [p,*p.parents] if x!=workspace.parent) or not p.is_file():
                    raise ValueError('missing or symlinked shader source: '+path)
            print('\n'.join(sources(value)))
        elif args.verify_exported:verify_exported(value,load(args.verify_exported))
        elif args.check_source:
            candidate,path=args.check_source
            p=Path(path)
            if not p.is_file() or p.is_symlink() or p.stat().st_size>8*1024*1024:raise ValueError('bounded regular source required')
            check_source(value,candidate,p.read_text())
        else:print(json.dumps(value,indent=2))
    except (OSError,UnicodeError,ValueError,TypeError) as e:
        print(str(e),file=sys.stderr);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
