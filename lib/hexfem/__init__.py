from __future__ import annotations
# ---------------- fem_rt_cuda ----------------
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of hexfem.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hexfem.cu'), encoding='utf-8', newline='').read()
        p = 0
        while p < len(t):
            e = t.index('\n', p)
            _, name, n = t[p:e].split(' ')
            _CU[name] = t[e + 1:e + 1 + int(n)]
            p = e + 1 + int(n) + 1
    return _CU


def cu(name):
    return _cu_sources()[name]


_CU = None

import numpy as np
import torch
_SRC = cu('_SRC')
rt_cuda__MOD = [None]
RAWPTR = [False]
_XS = {}

def _cp(t):
    if RAWPTR[0]:
        return np.uint64(t.data_ptr())
    import cupy as cp
    return cp.asarray(t)

class _Stream:
    """torch 지금 스트림에 CuPy 커널을 띄운다."""

    def __enter__(self):
        import cupy as cp
        sp = torch.cuda.current_stream().cuda_stream
        if RAWPTR[0]:
            if sp not in _XS:
                _XS[sp] = cp.cuda.ExternalStream(sp)
            self.s = _XS[sp]
        else:
            self.s = cp.cuda.ExternalStream(sp)
        self.s.__enter__()
        return self

    def __exit__(self, *e):
        self.s.__exit__(*e)

class CudaFem:

    def __init__(self, fr):
        raise NotImplementedError('not used by the paper settings')

    def energy(self, U):
        raise NotImplementedError('not used by the paper settings')

class CudaPcg:

    def __init__(self, Mmax, dev, ck=10):
        raise NotImplementedError('not used by the paper settings')

    def solve(self, M, b, rtol, ncg, use_graph=True):
        raise NotImplementedError('not used by the paper settings')

class CudaTetq:

    def __init__(self, fr):
        raise NotImplementedError('not used by the paper settings')

    def energy(self, U):
        raise NotImplementedError('not used by the paper settings')

class GridK:

    def __init__(self, x, node_h):
        raise NotImplementedError('not used by the paper settings')

def slots_k(G):
    raise NotImplementedError('not used by the paper settings')

def asm_k(G, PC, sym, Ms):
    raise NotImplementedError('not used by the paper settings')

class CudaTetqK(CudaTetq):

    def __init__(self, G, x, v, F, m, vol, cfg, h, U, eps_sp=0.01, eps_st=0.01, stick_xn=True, fixd=None, floor=True):
        raise NotImplementedError('not used by the paper settings')

    def g2p(self, U, F):
        raise NotImplementedError('not used by the paper settings')

def _vbd_setup(self, G):
    raise NotImplementedError('not used by the paper settings')

def vbd_grad_all(self, U):
    raise NotImplementedError('not used by the paper settings')

def vbd(self, U, Dinv, iters, rho=0.0, step=1.0, maxstep=0.0, refresh=0, dinv_fn=None, ls=0, floor=None):
    raise NotImplementedError('not used by the paper settings')
CudaTetq.vbd_setup = _vbd_setup
CudaTetq.vbd_grad_all = vbd_grad_all
CudaTetq.vbd = vbd
CudaFem.vbd_setup = _vbd_setup
CudaFem.vbd_grad_all = vbd_grad_all
CudaFem.vbd = vbd

# ---------------- fem_ex_cuda ----------------
import sys
RTC = sys.modules[__name__]
_EX = cu('_EX')
ex_cuda__MOD = [None]

def ex_cuda_mod():
    raise NotImplementedError('not used by the paper settings')

def ex_cuda_P(t):
    raise NotImplementedError('not used by the paper settings')

def ex_cuda_launch(name, n, args, b=128):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_an_cuda ----------------
import cupy as cp
RTC = sys.modules[__name__]
_K = cu('_K')
an_cuda__MOD = [None]
an_cuda_NAMES = ('an_elem', 'an_mv', 'an_bd', 'an_rsp', 'an_rsf', 'an_mv32', 'an_bd32', 'an_b3', 'an_p2g', 'an_g2p')

def an_cuda_P(t):
    raise NotImplementedError('not used by the paper settings')

def an_cuda_mod():
    raise NotImplementedError('not used by the paper settings')
PROF = dict(on=False, ev=[])

def an_cuda_launch(name, n, args, b=128):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_c_cuda ----------------
RTC = sys.modules[__name__]
EX = sys.modules[__name__]
_C = cu('_C')
c_cuda__MOD = [None]
NBW = 336

def c_cuda_mod():
    raise NotImplementedError('not used by the paper settings')

def c_cuda_launch(name, n, args, b=128, grid=None):
    raise NotImplementedError('not used by the paper settings')
c_cuda_P = EX.ex_cuda_P
CVS_DT = np.dtype([('mn', np.uint64), ('f', np.uint64), ('gn', np.uint64), ('vn', np.uint64), ('g0', np.float32), ('g1', np.float32), ('g2', np.float32), ('hs', np.float32), ('damp', np.float32), ('fixd', np.uint64), ('vbc', np.uint64), ('acc', np.uint64), ('mode', np.int32), ('np_', np.int32), ('n0', np.float32), ('n1', np.float32), ('n2', np.float32)], align=True)

# ---------------- fem_pl_cuda ----------------
import math
RTC = sys.modules[__name__]
EX = sys.modules[__name__]
CCU = sys.modules[__name__]
MAT = {'jelly': 0, 'metal': 1, 'sand': 2, 'foam': 3, 'plasticine': 5, 'watermelon': 7}
_PL = cu('_PL')
pl_cuda__MOD = [None]

def pl_cuda_mod():
    raise NotImplementedError('not used by the paper settings')

def pl_cuda_launch(name, n, args, b=128, grid=None):
    raise NotImplementedError('not used by the paper settings')
pl_cuda_P = EX.ex_cuda_P

def params(cfg, mu, lam, dev='cuda'):
    mname = cfg.get('material', 'jelly')
    if mname not in MAT:
        raise SystemExit(f"--plast: 재질 '{mname}' 은 아직 없다 ({sorted(MAT)})")
    mat = MAT[mname]
    gf = mat == 7
    phi = float(cfg.get('friction_angle', 45.0 if gf else 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    Mn = alpha * 3.0 / math.sqrt(2.0 / (6.0 - 3.0))
    kap = 2.0 * mu / 3.0 + lam
    pr = dict(mu=mu, lam=lam, kappa=kap, alpha=alpha, M=Mn, beta=float(cfg.get('beta', 1.0)), xi=float(cfg.get('xi', 0.0)), hardening=float(cfg.get('hardening', 0.0)), softening=float(cfg.get('softening', 0.1)), eta=float(cfg.get('plastic_viscosity', 0.0)), yield_stress=float(cfg.get('yield_stress', 0.0)), logJp0=float(cfg.get('alpha_0', -0.04)), friction_angle=phi)
    v = [pr['mu'], pr['lam'], pr['kappa'], pr['alpha'], pr['M'], pr['beta'], pr['xi'], pr['hardening'], pr['softening'], pr['eta']]
    t = torch.zeros(16, device=dev, dtype=torch.float32)
    t[:len(v)] = torch.tensor(v, dtype=torch.float32)
    return (mat, t, pr)

# ---------------- fem_ao_cuda ----------------
AN = sys.modules[__name__]
RTC = sys.modules[__name__]
_SAND = cu('_SAND')
ao_cuda__MOD = [None]
ao_cuda_NAMES = ('ao_elem', 'ao_rm')

def ao_cuda_P(t):
    raise NotImplementedError('not used by the paper settings')

def ao_cuda_mod():
    raise NotImplementedError('not used by the paper settings')

def ao_cuda_launch(name, n, args, b=128):
    raise NotImplementedError('not used by the paper settings')

def cdp_of(cfg, mu, lam):
    import math
    phi = float(cfg.get('friction_angle', 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    return ((3.0 * lam + 2.0 * mu) / (2.0 * mu) * alpha, alpha, phi)

# ---------------- fem_common1 ----------------
UL_SRC = cu('UL_SRC')
UL_BAR_SRC = cu('UL_BAR_SRC')

def ulify(src, uel, ufw, bar):
    if not (uel or ufw):
        return src
    i = src.rstrip().rfind('}')
    return src[:i] + UL_SRC + (UL_BAR_SRC if bar else '') + '\n}\n'

def uelify(src):
    i = src.index('__global__ void rt_ga_end_uf(')
    j = src.index('\n}\n', i) + 3
    k = src[i:j].replace('rt_ga_end_uf(', 'rt_ga_end_ufl(', 1)
    for (o_, n_) in (('float damp, const int* cjl, int um) {', 'float damp, const int* cjl, int um, const int* ulm) {'), ('    if (n >= IP[9]) return;\n', '    if (n >= IP[9]) return;\n    uf_ = uf_ && ulm[n];                                                        // 절 CM31 --ue_loc: 노드 집합만\n')):
        assert k.count(o_) == 1, o_
        k = k.replace(o_, n_, 1)
    return src[:j] + k + src[j:]
LS0_OLD = '    float st = full ? 1.f : (float)SC[14];\n'
LS0_NEW = '    float st = full ? 1.f : (float)SC[14];\n    if (!(st > 0.f)) { if (n == 0) atomicAdd(&AF_L0N[0], 1); if (!(isfinite(dl[n*3]) && isfinite(dl[n*3+1]) && isfinite(dl[n*3+2]))) atomicAdd(&AF_L0N[1], 1); return; }   // 절 CM31 --ls0_keep\n'

def ls0ify(src, on):
    if not on:
        return src
    i = src.index('__global__ void rt_lsup(')
    j = src.index('\n}\n', i) + 3
    k = src[i:j]
    assert k.count(LS0_OLD) == 1
    src = src[:i] + k.replace(LS0_OLD, LS0_NEW, 1) + src[j:]
    return src.replace('extern "C" {', 'extern "C" {\n__device__ int AF_L0N[2];', 1)

# ---------------- fem_common2 ----------------
C2_SRC = cu('C2_SRC')
C2_GATE_SRC = cu('C2_GATE_SRC')

def c2ify(src, dil, gate):
    if not (dil or gate):
        return src
    i = src.rstrip().rfind('}')
    return src[:i] + (C2_SRC if dil else '') + (C2_GATE_SRC if gate else '') + '\n}\n'

def dilate(s, capN, IP, R):
    raise NotImplementedError('not used by the paper settings')

def common2__P(t):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_common3 ----------------
def nfail(s, fl, g2, K):
    T = float(getattr(s, 'ksbig', 0.0) or 0.0)
    if T <= 0.0:
        return int(fl[:K].sum())
    return int(((fl[:K] > 0) & ~(g2[:K] <= T)).sum())

def mark(s, kind):
    c = getattr(s, '_c3cap', None)
    if c is None:
        return
    c['cur'].capture_end()
    c['segs'].append((c['cond'], c['cur']))
    if c['pool'] is None:
        c['pool'] = c['segs'][0][1].pool()
    g = torch.cuda.CUDAGraph()
    g.capture_begin(pool=c['pool'])
    (c['cur'], c['cond']) = (g, kind in ('q', 'g'))

class SegGraph:

    def __init__(self, s, segs):
        (self.s, self.segs) = (s, segs)

    def replay(self):
        s = self.s
        mode = int(s.xg)
        st = s.c3st
        skip = False
        for (cond, g) in self.segs:
            if not cond:
                skip = False
                if mode == 2:
                    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    e0.record()
                    g.replay()
                    e1.record()
                    st['ev'].append((0, e0, e1))
                else:
                    g.replay()
                continue
            if not skip:
                skip = bool(int(s.CT[0]) != 0)
            st['sk' if skip else 'run'] += 1
            if mode == 1 and skip:
                continue
            if mode == 2:
                (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                e0.record()
                g.replay()
                e1.record()
                st['ev'].append((2 if skip else 1, e0, e1))
            else:
                g.replay()

def stats_reset(s):
    s.c3st = dict(sk=0, run=0, ev=[])

def frame_stats(s):
    st = s.c3st
    o = dict(sk=st['sk'], run=st['run'])
    if st['ev']:
        torch.cuda.synchronize()
        m = [0.0, 0.0, 0.0]
        for (k_, e0, e1) in st['ev']:
            m[k_] += e0.elapsed_time(e1)
        o['ms'] = [round(q, 4) for q in m]
    stats_reset(s)
    return o

def capture(s):
    assert not getattr(s, 'pk', False), '--xg 는 그래프 판만'
    s.mod.get_function('rt_reset')
    K = s.Kcur
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        s.record(0.0, K)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    st2 = torch.cuda.Stream()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st2):
        g0.capture_begin()
        c = s._c3cap = dict(cur=g0, cond=False, segs=[], pool=None)
        try:
            s.record(0.0, K)
        finally:
            s._c3cap = None
        c['cur'].capture_end()
        c['segs'].append((c['cond'], c['cur']))
    torch.cuda.synchronize()
    g = SegGraph(s, c['segs'])
    s.graph = g
    s.graphs[K] = g

def common3_install(s, mode):
    import types
    s.xg = int(mode)
    stats_reset(s)
    s.capture = types.MethodType(capture, s)
    s.graph = None
    if hasattr(s, 'graphs'):
        s.graphs.clear()

# ---------------- fem_fpi ----------------
FPI_SRC = cu('FPI_SRC')

def fpify(src):
    s = src.rstrip()
    i = s.rfind('}')
    s = s[:i] + FPI_SRC + '\n}\n'
    return s.replace('extern "C" {', 'extern "C" {\n#define AF_FPI 1', 1)

# ---------------- fem_wolfprop_diag ----------------
wolfprop_diag_ON = [False]
NM_N = ('y', 'eta', 'g0buf', 'f', 'bGn')
NM_S = ('U', 'vk', 'ak', 'z0', 'Ut', 'mn', 'sk', 'pm', 'an', 'kb', 'meff')
NM_E = ('vk', 'ak', 'U')

def wolfprop_diag_alloc(s):
    raise NotImplementedError('not used by the paper settings')

def _on(s, kstep):
    return getattr(s, 'prec', False) and kstep == 0

def start(s, kstep):
    if not _on(s, kstep):
        return
    for (nm, b) in s.PRs.items():
        if nm != 'meff':
            s.dcopy(b, getattr(s, nm))

def fel(s, kstep, it):
    if _on(s, kstep):
        s.dcopy(s.PRn['f'][it], s.f)

def bar(s, kstep, it):
    if _on(s, kstep) and 'bGn' in s.PRn:
        s.dcopy(s.PRn['bGn'][it], s.bGn)

def grad(s, kstep, it):
    if not _on(s, kstep):
        return
    for nm in ('y', 'eta', 'g0buf'):
        s.dcopy(s.PRn[nm][it], getattr(s, nm))
    if it == 0:
        s.dcopy(s.PRs['meff'], s.meff)

def wolfprop_diag_end(s, kstep):
    if not _on(s, kstep):
        return
    for (nm, b) in s.PRe.items():
        s.dcopy(b, getattr(s, nm))

def _T(sk, pm, n, v):
    raise NotImplementedError('not used by the paper settings')

class Ctx:

    def __init__(s, S, FL, g, TREL, X0, h):
        raise NotImplementedError('not used by the paper settings')

    def frame(s, fi, gt_orig):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_wolfwhy_diag ----------------
wolfwhy_diag_ON = [False]

def wolfwhy_diag_alloc(s):
    if not wolfwhy_diag_ON[0]:
        return
    s.nrec = True
    dev = s.dev
    KK = 4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0))
    s.NR_Y0F = torch.zeros(KK, 4, device=dev, dtype=torch.int32)
    s.NR_CT = torch.zeros(KK * s.nt * 2, 16, device=dev, dtype=torch.int32)
    s.NR_CD = torch.zeros(KK, 4, device=dev, dtype=torch.float64)
    s.NR_SC = torch.zeros(KK, 64, device=dev, dtype=torch.float64)
    s.NR_GB = torch.zeros(KK, s.capN, device=dev, dtype=torch.float32)
    s.NR_bk = {}
    for nm in ('f', 'B', 'H', 'H30', 'SC', 'r', 'z', 'p', 'dl', 'Tp', 'g0buf', 'Dinv'):
        t = getattr(s, nm, None)
        if t is not None:
            s.NR_bk[nm] = torch.empty_like(t)
    for nm in ('bGn', 'bKn', 'bcf', 'bpg', 'bph'):
        t = getattr(s, nm, None)
        if isinstance(t, torch.Tensor):
            s.NR_bk[nm] = torch.empty_like(t)
    PRD = sys.modules[__name__]
    PRD.wolfprop_diag_alloc(s)

def y0f(s, kstep):
    if getattr(s, 'nrec', False):
        s.dcopy(s.NR_Y0F[kstep], s.kk)

def ct(s, kstep, it, j):
    if getattr(s, 'nrec', False):
        s.dcopy(s.NR_CT[2 * (kstep * s.nt + it) + j], s.CT)

def wolfwhy_diag_end(s, kstep, sg, capQ, capN, it2, BA, nr, barE):
    if not getattr(s, 'nrec', False):
        return
    P = an_rt_P
    PRD = sys.modules[__name__]
    (i32, f) = (np.int32, np.float32)
    IP = s.IP
    s.dcopy(s.NR_CD[kstep], s.CD)
    for (nm, b) in s.NR_bk.items():
        s.dcopy(b, getattr(s, nm))
    s.zf(s.f, 9, 3)
    s.zf(s.B, 9, 9)
    s.k('rt_sc_clear', 1, (P(s.SC), i32(0), i32(60)), b=1)
    if s.matk:
        s.mat_elem(kstep, -1, 10 ** 6, capQ)
        PRD.fel(s, kstep, s.nt)
    else:
        s.k('rt_elem', capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8), P(s.SC[10:]), P(s.f), P(s.H), i32(capQ), P(s.B)))
    if barE:
        assert not s.bnode, '--nrec: 노드 장벽 판 미지원'
        s.zf(s.bGn)
        s.zf(s.bKn)
        s.k('rt_bar_q', capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg['thinv']), *BA, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])))
        if getattr(s, 'det', False):
            s.k('ao_gath_q', capN, (P(IP), P(s.cidx), P(s.BW), P(s.bpg), P(s.bGn)), b=256)
            s.k('ao_gath_q', capN, (P(IP), P(s.cidx), P(s.BW), P(s.bph), P(s.bKn)), b=256)
        s.dcopy(s.NR_GB[kstep], s.bGn)
        PRD.bar(s, kstep, s.nt)
        s.k('rt_bar_n', capN, (P(IP), P(s.bGn), P(s.bKn), *[f(q) for q in nr], P(s.f), P(s.B)), b=256)
    s.k('rt_grad', capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
    s.dcopy(s.NR_SC[kstep], s.SC)
    PRD.grad(s, kstep, s.nt)
    for (nm, b) in s.NR_bk.items():
        s.dcopy(getattr(s, nm), b)

def frame(s, tolN):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_an_mg2 ----------------
MG2_SRC = cu('MG2_SRC')

def mg2_src(src):
    s = src.rstrip()
    i = s.rfind('}')
    return s[:i] + MG2_SRC + '\n}\n'

def det_mg2(s):
    _sub_in = an_det__sub_in
    s = _sub_in(s, 'rt_mv2', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }')
    s = _sub_in(s, 'rt_mv2s', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }')
    s = _sub_in(s, 'rt_mg_mv2', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }', 'if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_MGV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }')
    return s

def occ_counts(cells, nl, cap_mul):
    raise NotImplementedError('not used by the paper settings')

class MG2Mixin:

    def mg2_alloc(s, cells):
        dev = s.dev
        s.mg_cnt = torch.zeros(max(s.nl, 1), device=dev, dtype=torch.int32)
        if s.nl <= 1:
            return
        caps = occ_counts(cells, s.nl, s.cap_mul) if s.mg_occ or s.mg_fuse else [0] * s.nl
        for l in range(1, s.nl):
            L = s.ML[l - 1]
            dense = L['capq'] // 8
            L['lcap'] = min(dense, caps[l]) if s.mg_occ or s.mg_fuse else dense
            L['list'] = torch.zeros(L['lcap'], device=dev, dtype=torch.int32)
            L['pos'] = torch.zeros(dense, device=dev, dtype=torch.int32)
            if s.mg_fuse and l >= s.mg_fuse:
                L['cbuf'] = torch.zeros(L['lcap'] * 81, device=dev, dtype=torch.float32)
        if s.mg_fuse:
            MP = np.zeros((s.nl, 16), dtype=np.uint64)
            MI = np.zeros((s.nl, 4), dtype=np.int32)
            for l in range(s.mg_fuse, s.nl):
                L = s.ML[l - 1]
                for (k_, t_) in enumerate((L['Fq'], L['Vq'], L['H'], L['mc'], L['D'], L['r'], L['z'], L['t'], L['KT'], L['list'], L['pos'], L['cbuf'])):
                    MP[l, k_] = np.uint64(t_.data_ptr())
                MI[l, 0] = L['capq']
            s.mg_MP = torch.from_numpy(MP.view(np.int64).reshape(-1).copy()).to(dev)
            s.mg_MI = torch.from_numpy(MI.reshape(-1).copy()).to(dev)

    def mg2_occ(s, l):
        if not (s.mg_occ or (s.mg_fuse and l >= s.mg_fuse)):
            return
        L = s.ML[l - 1]
        s.k('rt_mg_occ', 1024, (np.int32(l), an_mg2_P(s.LM), an_mg2_P(L['Vq']), an_mg2_P(L['list']), an_mg2_P(L['pos']), an_mg2_P(s.mg_cnt), np.int32(L['lcap']), an_mg2_P(s.IP)), b=1024)

    def mg2_mv(s, l, z, KT):
        (i32, f) = (np.int32, np.float32)
        if l == 0:
            s.k(s.fmv_name(), s.capQc * 8, (an_mg2_P(s.IP), an_mg2_P(s.cb), an_mg2_P(s.BG), an_mg2_P(s.Fq), an_mg2_P(s.Vq), an_mg2_P(s.fmv_H()), an_mg2_P(z), an_mg2_P(KT), i32(s.capQc * 8)))
            return
        L = s.ML[l - 1]
        use = s.mg_occ
        n = (L['lcap'] if use else L['capq'] // 8) * 8
        s.k('rt_mg_mv2', n, (i32(l), an_mg2_P(s.LM), an_mg2_P(s.BG), f(0.5 ** l), an_mg2_P(L['Fq']), an_mg2_P(L['Vq']), an_mg2_P(L['H']), i32(L['capq']), an_mg2_P(z), an_mg2_P(KT), an_mg2_P(L['list'] if use else None), an_mg2_P(s.mg_cnt if use else None)))

    def mg2_cvc(s, r, z):
        raise NotImplementedError('not used by the paper settings')

    def kbench(s, reps=20, iters=20):
        raise NotImplementedError('not used by the paper settings')

def an_mg2_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

# ---------------- fem_an_rt ----------------
import os
RTC = sys.modules[__name__]
MG2 = sys.modules[__name__]
WW = sys.modules[__name__]
PRD = sys.modules[__name__]
C3 = sys.modules[__name__]
an_rt_SRC = cu('an_rt_SRC')
GQ_SRC = cu('GQ_SRC')
an_rt_SRC = an_rt_SRC.rstrip()[:-1].rstrip() + '\n' + GQ_SRC + '}\n'
SKIP_EXEMPT = ('rt_reset', 'rt_ctl')

def skipify(src):
    import re
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ int AF_SKIP;\n#define AF_ADAPT 1', 1)
    (out, pos) = ([], 0)
    for m in re.finditer('__global__\\s+void\\s+(?:__launch_bounds__\\(\\d+\\)\\s+)?(\\w+)\\s*\\(', src):
        b = src.index('{', m.end())
        out.append(src[pos:b + 1])
        if m.group(1) not in SKIP_EXEMPT:
            out.append(' if (AF_SKIP) return;')
        pos = b + 1
    out.append(src[pos:])
    return ''.join(out)

def octify(src):
    return src.replace('extern "C" {', 'extern "C" {\n#define AF_OCT 1\n__device__ float* AF_MO;\n__device__ float AF_GOCT;', 1)
BAR_SRC = cu('BAR_SRC')

def flptify(src, kex):
    raise NotImplementedError('not used by the paper settings')

def barify(src):
    src = src.replace('extern "C" {', 'extern "C" {\n#define AF_BAR 1\n__device__ float* AF_BX;\n__device__ float* AF_BPG;\n__device__ float* AF_BPH;\n__device__ int* AF_BACT;\n__device__ int AF_BDET;\n__device__ const float* AF_XP;\n__device__ float AF_BNRM[3];', 1)
    i = src.rstrip().rfind('}')
    return src[:i] + BAR_SRC + '\n}\n'

def bstify(src, mode, dh, full):
    add = ''
    if mode:
        add += '\n#define AF_BST %d\n#define AF_BDHC %.9ef' % (int(mode), float(dh))
    if full:
        add += '\n#define AF_BFULL 1'
    return src.replace('extern "C" {', 'extern "C" {' + add, 1) if add else src
BARN_SRC = cu('BARN_SRC')

def bnodeify(src, mode):
    if not mode:
        return src
    add = '\n#define AF_BNODE 1\n__device__ float* AF_BNV;\n__device__ int* AF_BNACT;\n__device__ int* AF_BNF;'
    if '__device__ const float* AF_BAM;' not in src:
        add += '\n__device__ const float* AF_BAM;'
    src = src.replace('extern "C" {', 'extern "C" {' + add, 1)
    if mode == 'only':
        old = 'if (mode == 0) AF_BACT[qq] = (d > 0.f) ? 1 : 0;'
        assert src.count(old) == 1
        src = src.replace(old, 'if (mode == 0) AF_BACT[qq] = 0;                                             // 절 CM14 (나) 노드 장벽만: 표본점 장벽 끔', 1)
    i = src.rstrip().rfind('}')
    return src[:i] + BARN_SRC + '\n}\n'
BARE_SRC = cu('BARE_SRC')

def becellify(src, on):
    if not on:
        return src
    i = src.rstrip().rfind('}')
    return src[:i] + BARE_SRC + '\n}\n'
VRES_SRC = cu('VRES_SRC')
Y0L_SRC = cu('Y0L_SRC')

def y0locify(src, on):
    if not on:
        return src
    i = src.rstrip().rfind('}')
    return src[:i] + Y0L_SRC + '\n}\n'

def vresify(src, k):
    if not k:
        return src
    i = src.rstrip().rfind('}')
    return src[:i] + VRES_SRC + '\n}\n'
NCL_SRC = cu('NCL_SRC')

def nclify(src, on):
    raise NotImplementedError('not used by the paper settings')
FUP_SRC = cu('FUP_SRC')
SNH_SRC = cu('SNH_SRC')

def snhify(src, on):
    return src.replace('extern "C" {', 'extern "C" {' + SNH_SRC, 1) if on else src
QJD_SRC = cu('QJD_SRC')

def fupify(src, ver):
    raise NotImplementedError('not used by the paper settings')

def fstkify(src, on):
    return src.replace('extern "C" {', 'extern "C" {\n#define AF_FSTK 1\n__device__ float* AF_ANA;', 1) if on else src

def bcrossify(src, cross, cnt):
    add = ''
    if cross:
        add += '\n#define AF_BCROSS 1'
    if cnt:
        add += '\n#define AF_BCNT 1\n__device__ int* AF_BCF;'
    if cross or cnt:
        add += '\n__device__ const float* AF_BAM;'
    return src.replace('extern "C" {', 'extern "C" {' + add, 1) if add else src
HB_ = '    if (blockIdx.x*blockDim.x >= IP[9]) return;\n    int n = blockIdx.x*blockDim.x + threadIdx.x;\n'
HS_ = '    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= IP[9]) return;\n'
CN_SUB = (('rt_ap', HB_, '    if (blockIdx.x*blockDim.x >= ACT[0]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < ACT[0]) ? ACT[1+ii_] : IP[9];\n'), ('rt_dotz', HB_, '    if (blockIdx.x*blockDim.x >= ACT[0]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < ACT[0]) ? ACT[1+ii_] : IP[9];\n'), ('rt_cg1m', HS_, '    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n'), ('rt_cg2', HS_, '    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n'), ('rt_pinit', HS_, '    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n'), ('rt_mg_node', '    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= LM[l*8+3]) return;\n', '    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n'), ('rt_mg_pw1', '    int nn = LM[l*8+3];\n    if (blockIdx.x*blockDim.x >= nn) return;\n    if (n < nn) {\n', '    int nn = ACT[0];\n    if (blockIdx.x*blockDim.x >= nn) return;\n    if (n < nn) { n = ACT[1+n];\n'), ('rt_mg_pw2', '    if (n >= LM[l*8+3]) return;\n', '    if (n >= ACT[0]) return;\n    n = ACT[1+n];\n'), ('rt_mg_prolong', '    if (I >= A[3]) return;\n', '    if (I >= ACT[0]) return;\n    I = ACT[1+I];\n'))
CN_ALL = frozenset((q[0] for q in CN_SUB))
CN_LM = frozenset(('rt_mg_node', 'rt_mg_pw1', 'rt_mg_pw2', 'rt_mg_prolong'))

def compactify(src):
    out = []
    for (nm, old, new) in CN_SUB:
        i = src.index('__global__ void ' + nm + '(')
        j = src.index('\n}\n', i) + 3
        body = src[i:j].replace('void ' + nm + '(', 'void ' + nm + '_c(', 1)
        k = body.index(') {')
        body = body[:k] + ', const int* ACT' + body[k:]
        assert body.count(old) == 1, nm
        out.append(body.replace(old, new, 1))
    s = src.rstrip()
    i = s.rfind('}')
    return s[:i] + '\n'.join(out) + '\n}\n'
G2P_GA_FLINE = '    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }\n'
CJ_SRC = cu('CJ_SRC')
CJ_SIG_OLD = 'float damp) {\n'
CJ_SIG_NEW = 'float damp, const int* cjl, int um) {\n    int uf_ = (um == 3) ? 1 : cjl[0];                                         // 절 CM24: um 1 덜 풀린 걸음 후퇴 오일러, 2 가속도 기억 0, 3 모든 걸음 후퇴 오일러 갱신\n'
CJ_U_OLD = '        U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*ac + be*a1);\n'
CJ_U_NEW = '        if (uf_ && um != 2 && !fx_) { v1 = (y[n*3+a] - u)/dt; U[n*3+a] = y[n*3+a]; } else U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*ac + be*a1);   // 후퇴 오일러: U1 = y, v1 = (y − U)/h\n        if (uf_) a1 = 0.f;\n'

def cjueify(src):
    i = src.index('__global__ void rt_ga_end(')
    j = src.index('\n}\n', i) + 3
    k = src[i:j].replace('rt_ga_end(', 'rt_ga_end_uf(', 1)
    for (o_, n_) in ((CJ_SIG_OLD, CJ_SIG_NEW), (CJ_U_OLD, CJ_U_NEW)):
        assert k.count(o_) == 1, o_
        k = k.replace(o_, n_, 1)
    return src[:j] + k + src[j:]

def build_src(src, adapt, goct, floor=None, fk=False, hspec=False, ncomp=False, gqm=False, snh=False, fpi=False, cj=False, cue=False, uel=0, ufw=False, ls0=False, c2=None):
    if cj:
        src = src.replace('extern "C" {', 'extern "C" {\n' + CJ_SRC, 1)
    src = snhify(src, snh)
    if fpi:
        pass
        i_ = src.index('__global__ void rt_g2p_ga(')
        j_ = src.index('\n}\n', i_) + 1
        assert src[i_:j_].count(G2P_GA_FLINE) == 1
        src = src[:i_] + src[i_:j_].replace(G2P_GA_FLINE, '', 1) + src[j_:]
        src = fpify(src)
    if gqm:
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_GQM 1', 1)
    if ncomp:
        src = compactify(src)
    if os.environ.get('AF_MGDIMS_OLD', '0') == '1':
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_MGDIMS_OLD 1', 1)
    if hspec:
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_HSPEC 1', 1)
    if fk:
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_FK 1', 1)
    if goct is not None and goct >= 0:
        src = octify(src)
    if floor and floor.get('pt'):
        src = flptify(src, floor['KEX'])
    if floor and floor.get('bar'):
        src = barify(src)
        src = bstify(src, floor.get('bst', 0), floor['DH'], floor.get('bfull', False))
        src = bcrossify(src, floor.get('bcross', False), floor.get('bcnt', False))
        src = fstkify(src, floor.get('stk', False))
        src = bnodeify(src, floor.get('bnode', ''))
        src = becellify(src, floor.get('becell', False))
        src = vresify(src, floor.get('vres', 0.0))
        src = y0locify(src, floor.get('y0loc', 0))
    if floor and floor.get('nclamp'):
        src = nclify(src, True)
    if floor and floor.get('fup'):
        src = fupify(src, floor['fup'])
    if cue:
        src = cjueify(src)
    if uel or ufw:
        C1 = sys.modules[__name__]
        if uel:
            src = C1.uelify(src)
        src = C1.ulify(src, uel, ufw, bool(floor and floor.get('bar')))
    if ls0:
        C1 = sys.modules[__name__]
        src = C1.ls0ify(src, True)
    if c2:
        C2 = sys.modules[__name__]
        src = C2.c2ify(src, 'dil' in c2, bool(c2.get('xf')))
    return skipify(src) if adapt else src

def _force_redo_frames():
    v = os.environ.get('AF_FORCE_REDO', '')
    return {int(q) for q in v.split(',') if q.strip()}

def an_rt_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

class an_rt_RTSolver(MG2.MG2Mixin):

    def __init__(s, x, v, F, m, vol, A, Ai, mu, lam, g, frame_dt, K, integ, newton, cg, ls, floor=None, damp=1.0, bc=None, cap_mul=2.0, dev='cuda', fpi=False, mg_levels=1, mg_nc=8, mg_pow=6, mg_omega=1.0, descent=False, ls_n=0, scheme=None, theta=0.5, mfloor=0.0, K_contact=None, gfin=False, fuse2=True, mg_wreuse=False, mg_lreuse=False, mg_csolve=False, ga=None, rho=0.7, areset=True, tolN=-1.0, tolP=-1.0, goct=-1.0, pk=False, pk_loc=2, pkb=256, pkminb=1, mv2=False, mg_occ=False, mg_fuse=0, y0ls=False, lsfast=False, mgbar=False, cg_list=None, fk=False, y0ut=False, hspec=False, ncomp=False, y0eb=False, gqm=False, wd=False, snh=False, wdcap=2000000, cj=None):
        s.dev = dev
        s.cjj = bool(cj and cj.get('judge'))
        s.cjr = bool(cj and cj.get('res'))
        s.cue = {'': 0, 'be': 1, 'a0': 2, 'beall': 3}[(cj or {}).get('ue', '')]
        s.uel = int((cj or {}).get('uel', 0))
        s.ufw = bool((cj or {}).get('ufw', False))
        assert not s.uel or s.cue in (1, 2), '--ue_loc 는 --ue be·a0 위'
        assert not s.ufw or s.cjr, '--uf_w 는 --uf_res 위'
        s.ls0 = bool((cj or {}).get('ls0', False)) if cj else False
        s.c2 = {k_: (cj or {})[k_] for k_ in ('xf', 'dil') if (cj or {}).get(k_)}
        if s.uel & 48:
            s.c2.setdefault('dil', 0)
            s.c2.setdefault('bigT', float((cj or {}).get('bigT', 1000000.0)))
        s.c2 = s.c2 or None
        assert not (s.c2 or {}).get('dil') or s.uel, '--ue_dil 은 --ue_loc 위'
        s.fpi = bool(fpi)
        assert not (s.fpi and pk), '--fp_interp 는 그래프 판만'
        s.gqm = bool(gqm)
        assert not (s.gqm and pk), '--gq_mass 는 그래프 판만'
        s.hspec = bool(hspec)
        s.ncomp = bool(ncomp)
        assert not s.hspec or bool(mv2) or bool(mg_occ), '--hspec 은 --mv2 판만'
        assert not (s.ncomp and pk), '--ncomp 는 그래프 판만'
        assert not (s.hspec and (mg_fuse or mg_csolve)), '--hspec 은 거친 층 한 블록 판 (mg_fuse·mg_csolve) 과 함께 못 씀 (81 성분 읽음)'
        s.y0eb = bool(y0eb)
        s.y0ut = bool(y0ut)
        s.matk = None
        s.lsfast = bool(lsfast)
        s.cg_list = [int(q) for q in cg_list.split(',')] if cg_list else None
        assert s.cg_list is None or len(s.cg_list) == int(newton), '--cg_list 길이는 --newton'
        s.fk = bool(fk)
        s.mgbar = int(mgbar)
        s.y0ls = bool(y0ls)
        (s.mg_occ, s.mg_fuse) = (bool(mg_occ), int(mg_fuse or 0))
        s.mv2 = bool(mv2) or s.mg_occ
        s.mg2 = s.mv2 or s.mg_fuse > 0
        assert not (s.mg2 and pk), '절 AN.20 판은 그래프 판만'
        assert s.mg_fuse == 0 or 1 <= s.mg_fuse <= int(mg_levels) - 1, '--mg_fuse 는 1 .. 층 수 − 1'
        (s.pk, s.pk_loc_l, s.pkb, s.pkminb) = (bool(pk), int(pk_loc), int(pkb), int(pkminb))
        (s._pkc, s._pk_loc, s.pkmod) = (None, 0, None)
        s._src0 = MG2.mg2_src(an_rt_SRC) if s.mg2 else an_rt_SRC
        (s.tolN, s.tolP) = (float(tolN), float(tolP))
        s.adapt = s.tolN >= 0 or s.tolP >= 0
        s.goct = float(goct)
        f32 = torch.float32
        s.N = int(x.shape[0])
        yy = x.double() @ Ai.double().T
        bb = torch.floor(yy - 0.5).long()
        bb = bb - bb.min(0).values
        mx = bb.max(0).values + 1
        s.perm = torch.argsort((bb[:, 0] * mx[1] + bb[:, 1]) * mx[2] + bb[:, 2])
        pr = s.perm
        s.x = x[pr].float().contiguous().clone()
        s.v = v[pr].float().contiguous().clone()
        s.F = F.reshape(-1, 9)[pr].float().contiguous().clone()
        s.m = m[pr].float().contiguous()
        s.vol = vol[pr].float().contiguous()
        s.A = A.float().contiguous()
        s.Ai = Ai.float().contiguous()
        (s.mu, s.lam, s.g) = (float(mu), float(lam), [float(q) for q in g])
        s.snh = bool(snh)
        s.h = float(frame_dt)
        s.K = int(K)
        s.mid = integ == 'mid'
        s.dt = s.h / s.K
        s.tau = s.dt / 2 if s.mid else s.dt
        s.al = 2.0 / s.tau if s.mid else 1.0 / s.tau
        s.beta = 0.5 if s.mid else 0.0
        (s.nt, s.cg, s.ls) = (int(newton), int(cg), int(ls))
        s.floor = floor
        s.damp = float(damp)
        (s.nl, s.mg_nc, s.mg_pow, s.mg_omega) = (int(mg_levels), int(mg_nc), int(mg_pow), float(mg_omega))
        s.descent = bool(descent)
        s.ls_n = int(ls_n)
        s.scheme = scheme or 'theta'
        s.theta = float(theta) if scheme else 0.5 if s.mid else 1.0
        s.mfloor = float(mfloor)
        s.gfin = bool(gfin)
        s.wdcap = int(wdcap)
        s.wd = bool(wd)
        s.fuse2 = bool(fuse2)
        s.mg_wreuse = bool(mg_wreuse)
        s.mg_lreuse = bool(mg_lreuse)
        s.mg_csolve = bool(mg_csolve)
        s.ga = ga
        if ga:
            r = float(rho)
            (am, af) = (0.0, (1.0 - r) / (1.0 + r)) if ga == 'hht' else ((2 * r - 1) / (r + 1), r / (r + 1))
            s.ga_c = dict(am=am, af=af, gm=0.5 - am + af, be=0.25 * (1 - am + af) ** 2)
            if ga == 'be':
                assert s.cue == 3, '--ga be 는 --ue beall 과 같이 (매 걸음 가속도 기억 0 이어야 정확한 후퇴 오일러)'
                s.ga_c = dict(am=0.0, af=0.0, gm=1.0, be=1.0)
            s.areset = bool(areset)
            s.ga_first = True
        s.Kc = int(K_contact) if K_contact else None
        s.Kcur = s.K
        s.graphs = {}
        s.bc = bc
        s.mod = cp.RawModule(code=build_src(s._src0, s.adapt, s.goct, s.floor, fk=s.fk, hspec=s.hspec, ncomp=s.ncomp, gqm=s.gqm, snh=s.snh, fpi=s.fpi, cj=s.cjj, cue=bool(s.cue), uel=s.uel, ufw=s.ufw, ls0=s.ls0, c2=s.c2), options=('--std=c++14',))
        s.CT = torch.zeros(16, device=dev, dtype=torch.int32)
        s.CD = torch.zeros(4, device=dev, dtype=torch.float64)
        s.fn = {}
        GG = 0.5 / math.sqrt(3.0)
        BW = np.zeros((8, 27))
        BG = np.zeros((8, 27, 3))
        Ai64 = Ai.double().cpu().numpy()
        for oc in range(8):
            fxv = [1.0 + (GG if oc >> 2 - a & 1 else -GG) for a in range(3)]
            w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
            dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
            for o in range(27):
                (i, j, k) = (o // 9, o // 3 % 3, o % 3)
                BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
                gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
                BG[oc, o] = gy @ Ai64
        s.BW = torch.tensor(BW, device=dev, dtype=f32).reshape(-1).contiguous()
        s.BG = torch.tensor(BG, device=dev, dtype=f32).reshape(-1).contiguous()
        s.IP = torch.zeros(16, device=dev, dtype=torch.int32)
        s.SC = torch.zeros(64, device=dev, dtype=torch.float64)
        s.Np = torch.zeros(s.N, 3, device=dev, dtype=torch.int32)
        s.fx = torch.zeros(s.N, 3, device=dev, dtype=f32)
        s.pcd = torch.zeros(s.N, device=dev, dtype=torch.int32)
        s.nbad = torch.zeros(1, device=dev, dtype=torch.int32)
        s.BP = torch.zeros(max(1, len(bc or [])) * 10, device=dev, dtype=f32)
        s.cap_mul = cap_mul
        s.graph = None
        s.alloc(None)

    def k(s, name, n, args, b=128):
        if s._pkc is not None:
            i32 = np.int32
            if name == 'rt_scan':
                s._pkc.append(('pk_scan_a', s.capC, 256, (args[0], args[1], an_rt_P(s.pk_bs)), s._pk_loc > 0))
                s._pkc.append(('pk_scan_b', 256, 256, (args[0], an_rt_P(s.pk_bs), args[4], i32(s.pk_capB)), True))
                s._pkc.append(('pk_scan_c', s.capC, 256, (args[0], args[1], an_rt_P(s.pk_bs), args[2], args[3], args[4], i32(s.pk_capB)), s._pk_loc > 0))
                return
            assert name not in ('rt_mg_csolve', 'ao_act_scan'), name
            s._pkc.append((name, int(n), int(b), tuple(args), s._pk_loc > 0))
            return
        if s.ncomp and name in CN_ALL and (name not in CN_LM or int(args[0]) == 0):
            name = name + '_c'
            args = tuple(args) + (an_rt_P(s.actl),)
        if name not in s.fn:
            s.fn[name] = s.mod.get_function(name)
        if getattr(s, 'prof', None) is not None:
            (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            e0.record()
        with RTC._Stream():
            s.fn[name](((int(n) + b - 1) // b,), (b,), args)
        if getattr(s, 'prof', None) is not None:
            e1.record()
            s.prof.append((name, e0, e1))

    def profile(s, t):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s, need):
        (dev, f32) = (s.dev, torch.float32)
        if need is None:
            y = s.x.double() @ s.Ai.double().T
            b = torch.floor(y - 0.5)
            nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
            nn = nb[0] * nb[1] * nb[2]
            ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
            uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=dev)).sum(1)).numel()
            need = (nn, ncb, uc)
            s.nb0 = nb
        if not hasattr(s, 'nb0'):
            s.nb0 = [round(need[0] ** (1 / 3))] * 3
        keepW = s.mg_lreuse and getattr(s, '_wdone', False) and (getattr(s, 'W', None) is not None) and (os.environ.get('AF_REDO_KEEPW', '1') == '1')
        oldW = s.W.clone() if keepW else None
        (s.capN, s.capC, s.capQc) = [int(q * s.cap_mul) + 1024 for q in need]
        capQ = s.capQc * 8
        Z = lambda *sh: torch.zeros(*sh, device=dev, dtype=f32)
        s.occ = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cidx = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cb = torch.zeros(s.capQc, 3, device=dev, dtype=torch.int32)
        s.Aacc = Z(capQ, 28)
        s.Cacc = Z(s.capQc, 30)
        s.badq = torch.zeros(capQ, device=dev, dtype=torch.int32)
        s.badc = torch.zeros(s.capQc, device=dev, dtype=torch.int32)
        s.Mo = Z(capQ) if s.goct >= 0 else None
        s.gqP = Z(capQ * 6) if getattr(s, 'gqm', False) else None
        s.gqC = Z(s.capQc * 6) if getattr(s, 'gqm', False) else None
        s.Fe = Z(capQ, 28) if getattr(s, 'fpi', False) else None
        s.bar = bool(s.floor and s.floor.get('bar'))
        s.vres = float(s.floor.get('vres', 0.0)) if s.bar and s.floor else 0.0
        if s.bar:
            s.bx = Z(capQ * 4)
            s.bpg = Z(capQ)
            s.bph = Z(capQ)
            s.bact = torch.zeros(capQ, device=dev, dtype=torch.int32)
            s.bGn = Z(s.capN)
            s.bKn = Z(s.capN)
            s.bfull = bool(s.floor.get('bfull', False))
            s.bcross = bool(s.floor.get('bcross', False))
            s.bcnt = bool(s.floor.get('bcnt', False))
            if s.bcnt:
                s.bcf = torch.zeros(capQ, device=dev, dtype=torch.int32)
            s.fstk = bool(s.floor.get('stk', False))
            if s.fstk:
                s.ana = Z(s.capN)
            if s.bfull:
                s.bS = Z(s.capN)
                s.btq = Z(capQ)
            s.bnode = s.floor.get('bnode', '')
            if s.bnode:
                assert not s.bfull, '--bar_node 는 덩어리 장벽 헤시안 판만 (노드 몫은 bKn 에 더함)'
                s.bnv = Z(s.capN)
                s.bnact = torch.zeros(s.capN, device=dev, dtype=torch.int32)
                s.bnf = torch.zeros(s.capN, device=dev, dtype=torch.int32)
            s.y0loc = int(s.floor.get('y0loc', 0))
            if s.y0loc:
                s.ykn = torch.zeros(s.capN, device=dev, dtype=torch.int32)
                s.ykn2 = torch.zeros(s.capN, device=dev, dtype=torch.int32)
                s.ylU = Z(s.capN, 3)
                s.ylE = Z(s.capN, 3)
                s.yfl = torch.zeros(16, device=dev, dtype=torch.int32)
                s.YLS = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)), 24, device=dev, dtype=torch.int32)
            s.becell = bool(s.floor.get('becell', False))
            if s.becell:
                s.beC = Z(s.capN)
                s.beL = Z(s.capN)
                s.beM = Z(s.capN)
                s.beD = Z(s.capN, 3)
                s.beT = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)) * 4, device=dev, dtype=torch.float64)
        s.fup = int(s.floor.get('fup', 0)) if s.floor else 0
        if s.fup:
            assert not s.bar and (not s.y0ls) and s.ga, '--fl_up 은 (A) 가우스 점 고착 판 (장벽 없음, HHT/GA, --y0ls 없음) 만'
            s.ufl = torch.zeros(s.capN, device=dev, dtype=torch.int32)
            s.fsk = Z(s.capN)
            s.fvo = Z(s.capN, 3)
            s.fao = Z(s.capN, 3)
            s.fucnt = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)) * 4, device=dev, dtype=torch.int32)
        s.ncl = bool(s.floor and s.floor.get('nclamp'))
        if s.ncl:
            s.ncf = torch.zeros(s.capN, device=dev, dtype=torch.int32)
            s.yrn = Z(s.capN)
            s.nccnt = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)) * 4, device=dev, dtype=torch.int32)
        s.Fq = Z(capQ, 9)
        s.Vq = Z(capQ)
        s.mq = Z(capQ)
        s.xq0 = Z(capQ, 3)
        s.H = Z(capQ, 81)
        if s.ncomp:
            nbk = (s.capN + 255) // 256 + 1
            s.actl = torch.zeros(s.capN + 1, device=dev, dtype=torch.int32)
            s.actb = torch.zeros(nbk, device=dev, dtype=torch.int32)
            s.acto = torch.zeros(nbk, device=dev, dtype=torch.int32)
            nbc = (s.capC + 255) // 256 + 1
            s.scb = torch.zeros(nbc, device=dev, dtype=torch.int32)
            s.sco = torch.zeros(nbc, device=dev, dtype=torch.int32)
        s.mn = Z(s.capN)
        s.mv = Z(s.capN, 3)
        s.mqn = Z(s.capN)
        s.an = Z(s.capN)
        s.U = Z(s.capN, 3)
        s.vk = Z(s.capN, 3)
        s.Ut = Z(s.capN, 3)
        s.z0 = Z(s.capN, 3)
        s.y = Z(s.capN, 3)
        s.yh = Z(s.capN, 3)
        s.eta = Z(s.capN, 3)
        s.sk = Z(s.capN)
        s.pm = torch.zeros(s.capN, device=dev, dtype=torch.int32)
        s.f = torch.zeros(s.capN, 3, device=dev, dtype=torch.float64)
        s.B = Z(s.capN, 9)
        s.Dinv = Z(s.capN, 9)
        s.r = Z(s.capN, 3)
        s.z = Z(s.capN, 3)
        s.p = Z(s.capN, 3)
        s.dl = Z(s.capN, 3)
        s.Tp = Z(s.capN, 3)
        s.KTp = Z(s.capN, 3)
        s.Ap = Z(s.capN, 3)
        s.yc = Z(s.capN, 3)
        s.fixd = torch.zeros(s.capN, device=dev, dtype=torch.uint8) if s.bc else None
        s.vbc = Z(s.capN, 3) if s.bc else None
        s.g0buf = Z(s.capN, 3)
        s.graphs = {}
        s._wdone = False
        s.an2 = Z(s.capN)
        s.p0 = Z(s.capN, 3)
        s.ma = Z(s.capN, 3)
        s.ak = Z(s.capN, 3)
        s.eS = Z(s.capN, 3)
        s.ySv = Z(s.capN, 3)
        s.kk = torch.zeros(4, device=dev, dtype=torch.int32)
        s.bloc = int(os.environ.get('AF_BLOC', '0')) >= 1
        s.bkn = torch.zeros(s.capN, device=dev, dtype=torch.int32) if s.bloc else None
        if not hasattr(s, 'pa'):
            s.pa = torch.zeros(s.N, 3, device=dev, dtype=torch.float32)
        s.meff = Z(s.capN)
        s.U0 = Z(s.capN, 3)
        s.v0 = Z(s.capN, 3)
        s.wb = Z(s.capN, 3)
        s.wvb = Z(s.capN, 3)
        s.DG = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)) * s.nt, 64, device=dev, dtype=torch.float64)
        s.Y0K = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)), 4, device=dev, dtype=torch.int32)
        s.BSK = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)) * s.nt, 4, device=dev, dtype=torch.int32)
        s.Y0S = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)), 64, device=dev, dtype=torch.float64)
        s.DGF = torch.zeros(4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0)), 64, device=dev, dtype=torch.float64)
        if getattr(s, 'cjj', False):
            _kc = max(s.K, s.Kc or 0, getattr(s, 'kcap', 0))
            s.cjs = torch.zeros(_kc, device=dev, dtype=torch.float32)
            s.cjg = torch.zeros(_kc, device=dev, dtype=torch.float32)
            s.cjl = torch.zeros(_kc, device=dev, dtype=torch.int32)
            if getattr(s, 'uel', 0):
                s.ulm = torch.zeros(s.capN, device=dev, dtype=torch.int32)
                s.ULS = torch.zeros(_kc, 4, device=dev, dtype=torch.int32)
                if (getattr(s, 'c2', None) or {}).get('dil'):
                    s.ulm2 = torch.zeros(s.capN, device=dev, dtype=torch.int32)
            if getattr(s, 'ufw', False):
                s.cjw = torch.zeros(_kc, device=dev, dtype=torch.float64)
        s.wd_on = bool(getattr(s, 'wd', False)) and s.capN <= getattr(s, 'wdcap', 2000000)
        for _nm in ('WY', 'WY0', 'WU', 'WFE', 'WBE', 'WME'):
            if hasattr(s, _nm):
                delattr(s, _nm)
        if s.wd_on:
            KK = max(s.K, s.Kc or 0, getattr(s, 'kcap', 0))
            s.WY = Z(KK * s.nt, s.capN, 3)
            s.WY0 = Z(KK, s.capN, 3)
            s.WU = Z(KK, s.capN, 3)
            s.WCT = torch.zeros(KK * s.nt * 2 + KK, 16, device=dev, dtype=torch.int32)
            s.WCD = torch.zeros(KK * s.nt * 2, 4, device=dev, dtype=torch.float64)
            s.WFE = torch.zeros(KK * s.nt, s.capN, 3, device=dev, dtype=torch.float64)
            s.WBE = Z(KK * s.nt, s.capN, 9)
            s.WME = Z(KK, s.capN)
        s.wd_tau = []
        s.ML = []
        if s.nl > 1:
            nbh = list(s.nb0)
            (capn, capq) = ([0], [0])
            for l in range(1, s.nl):
                nbh = [q // 2 + 3 for q in nbh]
                cn = int(nbh[0] * nbh[1] * nbh[2] * s.cap_mul) + 1024
                cq = int((nbh[0] - 2) * (nbh[1] - 2) * (nbh[2] - 2) * 8 * s.cap_mul) + 1024
                cq = (cq + 7) // 8 * 8
                capn.append(cn)
                capq.append(cq)
                L = dict(capn=cn, capq=cq, Fq=Z(cq, 9), Vq=Z(cq), H=Z(81, cq), B=Z(cn, 9), D=Z(cn, 9), mc=Z(cn), A=Z(cq, 28), C=Z(cq // 8, 30), badq=torch.zeros(cq, device=dev, dtype=torch.int32), badc=torch.zeros(cq // 8, device=dev, dtype=torch.int32), r=Z(cn, 3), z=Z(cn, 3), t=Z(cn, 3), KT=Z(cn, 3), y=Z(cn, 3), Kb=Z(cn), Db=Z(cn, 9))
                s.ML.append(L)
            s.capn_d = torch.tensor(capn, device=dev, dtype=torch.int32)
            s.capq_d = torch.tensor(capq, device=dev, dtype=torch.int32)
            s.LM = torch.zeros(s.nl * 8, device=dev, dtype=torch.int32)
            s.W = torch.zeros(s.nl, device=dev, dtype=torch.float64)
            s.SCM = torch.zeros(s.nl * 4, device=dev, dtype=torch.float64)
            if oldW is not None and oldW.numel() == s.W.numel():
                s.W.copy_(oldW)
                s._wdone = True
            s.m0t = Z(s.capN, 3)
            s.m0u = Z(s.capN, 3)
            s.m0KT = Z(s.capN, 3)
            s.zo = Z(s.capN, 3)
            s.mnd = s.mn
            if s.mg2:
                yb = torch.floor(s.x.double() @ s.Ai.double().T - 0.5)
                s.mg2_alloc(torch.unique((yb - yb.min(0).values).long(), dim=0))
        if s.pk:
            s.pk_capB = (s.capC + 255) // 256
            s.pk_bs = torch.zeros(s.pk_capB + 1, device=dev, dtype=torch.int32)
            s.pk_cnt = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32)
            s.pk_st = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32)
            s.pk_cur = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32)
            s.pk_idx = torch.zeros(s.N, device=dev, dtype=torch.int32)
            s.pk_scr = torch.zeros(s.N * 9, device=dev, dtype=torch.float32)
        s.graph = None
        s.oct_bind()
        WW.wolfwhy_diag_alloc(s)

    def _mods(s):
        return [s.mod] + ([s.pkmod] if s.pkmod is not None else [])

    def oct_bind(s):
        if getattr(s, 'bar', False):
            for mod in s._mods():
                for (nm, t) in (('AF_BX', s.bx), ('AF_BPG', s.bpg), ('AF_BPH', s.bph), ('AF_BACT', s.bact), ('AF_XP', s.x)):
                    g = mod.get_global(nm)
                    cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(t.data_ptr())
                g = mod.get_global('AF_BDET')
                cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = 1 if getattr(s, 'det', False) else 0
                g = mod.get_global('AF_BNRM')
                cp.ndarray((3,), dtype=cp.float32, memptr=g)[:] = cp.asarray(np.array(s.floor['NR'], dtype=np.float32))
                if s.bcross or s.bcnt:
                    g = mod.get_global('AF_BAM')
                    cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.A.data_ptr())
                if s.bcnt:
                    g = mod.get_global('AF_BCF')
                    cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.bcf.data_ptr())
                if getattr(s, 'fstk', False):
                    g = mod.get_global('AF_ANA')
                    cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.ana.data_ptr())
                if getattr(s, 'bnode', ''):
                    for (nm, t) in (('AF_BNV', s.bnv), ('AF_BNACT', s.bnact), ('AF_BNF', s.bnf), ('AF_BAM', s.A)):
                        g = mod.get_global(nm)
                        cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(t.data_ptr())
        if s.goct < 0:
            return
        for mod in s._mods():
            g = mod.get_global('AF_MO')
            cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.Mo.data_ptr())
            g = mod.get_global('AF_GOCT')
            cp.ndarray((1,), dtype=cp.float32, memptr=g)[0] = s.goct

    def mg_build(s):
        (i32, f) = (np.int32, np.float32)
        it2 = f(1.0 / getattr(s, 'mg_tau', s.tau) ** 2)
        s.k('rt_mg_dims', 1, (an_rt_P(s.IP), an_rt_P(s.LM), i32(s.nl), an_rt_P(s.capn_d), an_rt_P(s.capq_d)), b=1)
        for l in range(1, s.nl):
            L = s.ML[l - 1]
            for (t_, wh, mul) in ((L['A'], 1, 28), (L['C'], 2, 30), (L['B'], 0, 9), (L['mc'], 0, 1)):
                s.k('rt_zero_lm', t_.numel(), (i32(l), an_rt_P(s.LM), i32(wh), i32(mul), an_rt_P(t_), an_rt_P(None)), b=256)
            s.k('rt_zero_lm', L['badq'].numel(), (i32(l), an_rt_P(s.LM), i32(1), i32(1), an_rt_P(None), an_rt_P(L['badq'])), b=256)
            s.k('rt_zero_lm', L['badc'].numel(), (i32(l), an_rt_P(s.LM), i32(2), i32(1), an_rt_P(None), an_rt_P(L['badc'])), b=256)
            src_cap = s.capQc * 8 if l == 1 else s.ML[l - 2]['capq']
            Fs = s.Fq if l == 1 else s.ML[l - 2]['Fq']
            Vs = s.Vq if l == 1 else s.ML[l - 2]['Vq']
            s.k('rt_mg_acc', src_cap, (i32(l), an_rt_P(s.IP), an_rt_P(s.LM), an_rt_P(s.cb), an_rt_P(Fs), an_rt_P(Vs), an_rt_P(L['A']), an_rt_P(L['C']), an_rt_P(L['badq']), an_rt_P(L['badc'])))
            s.k('rt_mg_fin', L['capq'], (i32(l), an_rt_P(s.LM), an_rt_P(L['A']), an_rt_P(L['C']), an_rt_P(L['badq']), an_rt_P(L['badc']), an_rt_P(L['Fq']), an_rt_P(L['Vq'])))
            if s.mg2:
                s.mg2_occ(l)
            mprev = s.mn if l == 1 else s.ML[l - 2]['mc']
            s.k('rt_mg_restrict', L['capn'], (i32(l - 1), an_rt_P(s.LM), an_rt_P(mprev), an_rt_P(L['mc']), i32(1)), b=256)
            s.k('rt_mg_elem', L['capq'], (i32(l), an_rt_P(s.LM), an_rt_P(s.BG), f(0.5 ** l), an_rt_P(L['Fq']), an_rt_P(L['Vq']), f(s.mu), f(s.lam), an_rt_P(L['H']), i32(L['capq']), an_rt_P(L['B'])))
            s.k('rt_mg_D', L['capn'], (i32(l), an_rt_P(s.LM), an_rt_P(L['mc']), it2, an_rt_P(L['B']), an_rt_P(L['D'])), b=256)
        for l in range(1, s.nl):
            if not (s.mg_lreuse and getattr(s, '_wdone', False)):
                s.mg_power(l)
        s._wdone = True

    def mg_Az(s, l, z, KT, tz_ready=False, bar=False):
        (i32, f) = (np.int32, np.float32)
        nr = s.floor['NR'] if s.floor else [0.0, 0.0, 1.0]
        if l == 0:
            if not tz_ready:
                s.k('rt_mg_node', s.capN, (i32(0), an_rt_P(s.LM), i32(3), an_rt_P(s.mn), f(0), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(None), an_rt_P(s.W), an_rt_P(None), an_rt_P(None), an_rt_P(z), an_rt_P(s.m0u), an_rt_P(None)), b=256)
            if s.mv2:
                s.mg2_mv(0, s.m0u, KT)
            else:
                s.k('rt_mv', s.capQc * 8, (an_rt_P(s.IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.H), an_rt_P(s.m0u), an_rt_P(KT), i32(s.capQc * 8)))
            if bar and s.mgbar and getattr(s, 'bar', False):
                s.k('rt_bar_mv', s.capN, (an_rt_P(s.IP), an_rt_P(s.bKn), *[f(q) for q in nr], an_rt_P(s.m0u), an_rt_P(KT)), b=256)
            return
        else:
            if s.mv2:
                s.mg2_mv(l, z, KT)
                if bar and getattr(s, '_mgb_on', False):
                    s.k('rt_bar_mvl', s.ML[l - 1]['capn'], (i32(l), an_rt_P(s.LM), an_rt_P(s.ML[l - 1]['Kb']), *[f(q) for q in nr], an_rt_P(z), an_rt_P(KT)), b=256)
                return
            L = s.ML[l - 1]
            s.k('rt_mg_mv', L['capq'], (i32(l), an_rt_P(s.LM), an_rt_P(s.BG), f(0.5 ** l), an_rt_P(L['Fq']), an_rt_P(L['Vq']), an_rt_P(L['H']), i32(L['capq']), an_rt_P(z), an_rt_P(KT)))

    def fmv_name(s):
        return 'rt_mv2s' if s.hspec else 'rt_mv2' if s.mv2 else 'rt_mv'

    def fmv_H(s):
        h = getattr(s, 'Hs', None)
        return s.H if h is None else h

    def mg_lv(s, l):
        if l == 0:
            return dict(mn=s.meff, D=s.Dinv, cap=s.capN)
        L = s.ML[l - 1]
        return dict(mn=L['mc'], D=L['Db'] if getattr(s, '_mgb_on', False) else L['D'], cap=L['capn'])

    def mg_power(s, l):
        (i32, f) = (np.int32, np.float32)
        it2 = f(1.0 / getattr(s, 'mg_tau', s.tau) ** 2)
        nr = s.floor['NR'] if s.floor else [0.0, 0.0, 1.0]
        V = s.mg_lv(l)
        z = s.m0t if l == 0 else s.ML[l - 1]['z']
        y = s.zo if l == 0 else s.ML[l - 1]['y']
        KT = s.m0KT if l == 0 else s.ML[l - 1]['KT']
        s.k('rt_fill_rand', V['cap'], (i32(l), an_rt_P(s.LM), an_rt_P(z)), b=256)
        for it in range(s.mg_pow):
            s.k('rt_mg_pw0', 1, (i32(l), an_rt_P(s.SCM)), b=1)
            s.mg_Az(l, z, KT, bar=True)
            s.k('rt_mg_pw1', V['cap'], (i32(l), an_rt_P(s.LM), an_rt_P(V['mn']), it2, an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(V['D']), an_rt_P(z), an_rt_P(KT), an_rt_P(y), an_rt_P(s.SCM)), b=256)
            s.k('rt_mg_pw2', V['cap'], (i32(l), an_rt_P(s.LM), an_rt_P(y), an_rt_P(z), an_rt_P(s.SCM), an_rt_P(s.W), f(s.mg_omega), i32(1 if it == s.mg_pow - 1 else 0)), b=256)

    def _pkl(s, on):
        import contextlib

        @contextlib.contextmanager
        def cm():
            if on:
                s._pk_loc += 1
            try:
                yield
            finally:
                if on:
                    s._pk_loc -= 1
        return cm()

    def mg_vc(s, l, r, z):
        with s._pkl(s._pkc is not None and l >= 1 and (l >= s.pk_loc_l)):
            s._mg_vc(l, r, z)

    def _mg_vc(s, l, r, z):
        (i32, f) = (np.int32, np.float32)
        it2 = f(1.0 / getattr(s, 'mg_tau', s.tau) ** 2)
        nr = s.floor['NR'] if s.floor else [0.0, 0.0, 1.0]
        V = s.mg_lv(l)
        KT = s.m0KT if l == 0 else s.ML[l - 1]['KT']
        res = s.m0t if l == 0 else s.ML[l - 1]['t']
        KB = None
        if getattr(s, 'bar', False) and s.mgbar and (not getattr(s, '_pkc', None)):
            KB = s.bKn if l == 0 else s.ML[l - 1]['Kb'] if getattr(s, '_mgb_on', False) else None
        node = lambda mode, rr, kt, zz, rs: s.k('rt_mg_node', V['cap'], (i32(l), an_rt_P(s.LM), i32(mode), an_rt_P(V['mn']), it2, an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(V['D']), an_rt_P(s.W), an_rt_P(rr), an_rt_P(kt), an_rt_P(zz), an_rt_P(rs), an_rt_P(KB)), b=256)
        csolve = l == s.nl - 1 and s.mg_csolve and (l > 0) and (not getattr(s, 'det', False))
        if l == 0:
            node(6, r, None, z, s.m0u)
        elif not csolve:
            node(0, r, None, z, None)
        if l == s.nl - 1:
            if csolve:
                L = s.ML[l - 1]
                s.k('rt_mg_csolve', 1024, (i32(l), an_rt_P(s.LM), an_rt_P(s.BG), f(0.5 ** l), an_rt_P(L['Fq']), an_rt_P(L['Vq']), an_rt_P(L['H']), i32(L['capq']), an_rt_P(V['mn']), it2, an_rt_P(V['D']), an_rt_P(s.W), an_rt_P(r), an_rt_P(z), an_rt_P(L['KT']), i32(s.mg_nc)), b=1024)
                return
            for _ in range(s.mg_nc - 1):
                s.mg_Az(l, z, KT)
                node(1, r, KT, z, None)
            return
        Lc = s.ML[l]
        if l == 0:
            s.mg_Az(0, z, KT, tz_ready=True)
            node(7, r, KT, z, s.m0u)
            with s._pkl(s._pkc is not None and 1 >= s.pk_loc_l):
                s.k(*(('rt_mg_restrict3', Lc['capn'] * 16) if s.mv2 else ('rt_mg_restrict', Lc['capn'])), (i32(0), an_rt_P(s.LM), an_rt_P(s.m0u), an_rt_P(Lc['r']), i32(3)), b=256)
            if not 1 >= s.pk_loc_l and (not (s.nl - 1 == 1 and s.mg_csolve)):
                s.pk_chain('rt_mg_restrict', 'rt_mg_node')
        else:
            s.mg_Az(l, z, KT)
            node(2, r, KT, z, res)
            with s._pkl(s._pkc is not None and l + 1 >= s.pk_loc_l):
                s.k(*(('rt_mg_restrict3', Lc['capn'] * 16) if s.mv2 else ('rt_mg_restrict', Lc['capn'])), (i32(l), an_rt_P(s.LM), an_rt_P(res), an_rt_P(Lc['r']), i32(3)), b=256)
            if not l + 1 >= s.pk_loc_l and (not (s.nl - 1 == l + 1 and s.mg_csolve)):
                s.pk_chain('rt_mg_restrict', 'rt_mg_node')
        if s.mg_fuse and l + 1 == s.mg_fuse:
            s.mg2_cvc(Lc['r'], Lc['z'])
        else:
            s.mg_vc(l + 1, Lc['r'], Lc['z'])
        if l == 0:
            s.k('rt_mg_prolong', V['cap'], (i32(0), an_rt_P(s.LM), an_rt_P(Lc['z']), an_rt_P(s.m0u), i32(0)), b=256)
            s.pk_chain('rt_mg_prolong', 'rt_mg_node')
            if s.fuse2:
                node(9, s.m0u, None, z, s.m0u)
            else:
                node(5, s.m0u, None, z, None)
        else:
            s.k('rt_mg_prolong', V['cap'], (i32(l), an_rt_P(s.LM), an_rt_P(Lc['z']), an_rt_P(z), i32(1)), b=256)
        s.mg_Az(l, z, KT, tz_ready=l == 0 and s.fuse2)
        node(1, r, KT, z, None)

    def zf(s, t, cnt_idx=None, mul=1):
        n = t.numel()
        if t.dtype == torch.float64:
            s.k('rt_zero_d', n, (np.int32(n), an_rt_P(s.IP if cnt_idx is not None else None), np.int32(cnt_idx or 0), np.int32(mul), an_rt_P(t)), b=256)
        elif t.dtype == torch.int32:
            s.k('rt_zero_i', n, (np.int32(n), an_rt_P(t)), b=256)
        else:
            s.k('rt_zero_f', n, (np.int32(n), an_rt_P(s.IP if cnt_idx is not None else None), np.int32(cnt_idx or 0), np.int32(mul), an_rt_P(t)), b=256)

    def stages(s, K):
        dt = s.h / K
        out = []
        for _ in range(K):
            if s.ga:
                c = s.ga_c
                tau = math.sqrt((1 - c['af']) * c['be'] * dt * dt / (1 - c['am']))
                al = c['gm'] * dt / ((1 - c['af']) * c['be'] * dt * dt)
                out.append(dict(kind='ga', tau=tau, al=al, beta=0.0, thinv=1.0 / (1 - c['af']), dtf=(1 - c['af']) * dt, frac=1.0, last=True, dt=dt, **c))
                continue
            if s.scheme == 'trbdf2':
                ga = 2.0 - math.sqrt(2.0)
                t1 = 0.5 * ga * dt
                out.append(dict(kind='theta', tau=t1, al=1.0 / (0.25 * ga * dt), beta=0.5, thinv=2.0, dtf=ga * dt, save0=True, frac=ga, last=False))
                c2 = (1.0 - ga) / (2.0 - ga)
                c1 = 1.0 / (ga * (2.0 - ga))
                c0 = (1.0 - ga) ** 2 / (ga * (2.0 - ga))
                out.append(dict(kind='bdf2', tau=c2 * dt, al=1.0 / (c2 * dt), beta=0.0, thinv=1.0, dtf=(1.0 - ga) * dt, c1=c1, c0=c0, frac=1.0 - ga, last=True))
            else:
                th = s.theta
                out.append(dict(kind='theta', tau=th * dt, al=1.0 / (th * th * dt), beta=1.0 - th, thinv=1.0 / th, dtf=th * dt, frac=1.0, last=True))
        return out

    def record(s, t_frame, K=None):
        (N, IP) = (s.N, s.IP)
        (i32, f) = (np.int32, np.float32)
        (capN, capQ) = (s.capN, s.capQc * 8)
        s._mgb_on = False
        s.k('rt_reset', 1, (an_rt_P(IP),), b=1)
        s.ctl(0)
        s.k('rt_coord', N, (i32(N), an_rt_P(s.x), an_rt_P(s.Ai), an_rt_P(s.Np), an_rt_P(s.fx), an_rt_P(IP)))
        s.k('rt_box', 1, (an_rt_P(IP), i32(capN), i32(s.capC)), b=1)
        s.zf(s.occ)
        s.k('rt_mark', N, (i32(N), an_rt_P(s.Np), an_rt_P(s.fx), an_rt_P(IP), an_rt_P(s.pcd), an_rt_P(s.occ)))
        if s.ncomp:
            s.k('rt_scan_a', s.capC, (an_rt_P(IP), an_rt_P(s.occ), an_rt_P(s.scb)), b=256)
            s.k('rt_scan_b', 1024, (an_rt_P(IP), an_rt_P(s.scb), an_rt_P(s.sco), i32(s.capQc)), b=1024)
            s.k('rt_scan_c', s.capC, (an_rt_P(IP), an_rt_P(s.occ), an_rt_P(s.sco), an_rt_P(s.cidx), an_rt_P(s.cb), i32(s.capQc)), b=256)
        else:
            s.k('rt_scan', 1024, (an_rt_P(IP), an_rt_P(s.occ), an_rt_P(s.cidx), an_rt_P(s.cb), i32(s.capQc)), b=1024)
        if s.ncomp:
            s.k('rt_act_cnt', capN, (an_rt_P(IP), an_rt_P(s.occ), an_rt_P(s.actb)), b=256)
            s.k('rt_act_scan', 1024, (an_rt_P(IP), an_rt_P(s.actb), an_rt_P(s.acto), an_rt_P(s.actl)), b=1024)
            s.k('rt_act_wr', capN, (an_rt_P(IP), an_rt_P(s.occ), an_rt_P(s.acto), an_rt_P(s.actl)), b=256)
            if s.nl > 1:
                s.zf(s.m0u, 9, 3)
        for (t_, ci, mul) in ((s.Aacc, 12, 28), (s.Cacc, 11, 30), (s.mn, 9, 1), (s.mv, 9, 3), (s.mqn, 9, 1)) + (((s.ma, 9, 3),) if s.ga else ()) + (((s.Mo, 12, 1),) if s.goct >= 0 else ()) + (((s.gqP, 12, 6), (s.gqC, 11, 6)) if s.gqm else ()):
            s.zf(t_, ci, mul)
        s.zf(s.badq)
        s.zf(s.badc)
        s.zf(s.nbad)
        if getattr(s, 'bar', False):
            s.zf(s.bx)
            if s.bcnt:
                s.zf(s.bcf)
            if s.bnode:
                s.zf(s.bnf)
            if s.becell:
                s.zf(s.beM, 9, 1)
                s.zf(s.beT)
        if getattr(s, 'ncl', False):
            s.zf(s.nccnt)
        if getattr(s, 'fup', 0):
            s.zf(s.fucnt)
        if s.matk:
            s.mat_pre_p2g()
        s.k('rt_p2g', N, (i32(N), an_rt_P(s.x), an_rt_P(s.v), an_rt_P(s.F), an_rt_P(s.m), an_rt_P(s.vol), an_rt_P(s.Np), an_rt_P(s.fx), an_rt_P(s.pcd), an_rt_P(s.cidx), an_rt_P(IP), an_rt_P(s.Aacc), an_rt_P(s.Cacc), an_rt_P(s.badq), an_rt_P(s.badc), an_rt_P(s.mn), an_rt_P(s.mv), an_rt_P(s.pa if s.ga else None), an_rt_P(s.ma if s.ga else None)))
        if s.gqm:
            s.k('gq_p2q', N, (i32(N), an_rt_P(s.v), an_rt_P(s.m), an_rt_P(s.pa if s.ga else None), an_rt_P(s.fx), an_rt_P(s.pcd), an_rt_P(s.cidx), an_rt_P(IP), an_rt_P(s.gqP), an_rt_P(s.gqC)))
        s.k('rt_gauss', capQ, (an_rt_P(IP), an_rt_P(s.Aacc), an_rt_P(s.Cacc), an_rt_P(s.badq), an_rt_P(s.badc), an_rt_P(s.cb), an_rt_P(s.A), an_rt_P(s.BW), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.mq), an_rt_P(s.xq0), an_rt_P(s.mqn), an_rt_P(s.nbad)))
        if s.gqm:
            s.k('gq_q2n', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.mq), an_rt_P(s.gqP), an_rt_P(s.gqC), f(max(s.goct, 0.0)), i32(1 if s.goct >= 0 else 0), an_rt_P(s.mn), an_rt_P(s.mv), an_rt_P(s.ma if s.ga else None)))
        s.k('rt_vn', capN, (an_rt_P(IP), an_rt_P(s.mn), an_rt_P(s.mv), an_rt_P(s.vk), an_rt_P(s.U)), b=256)
        if s.matk:
            s.mat_post_vn()
        if s.nl > 1:
            s.mg_build()
        if s.bc:
            s.bzero(s.fixd)
            for ib in range(len(s.bc)):
                s.k('rt_bc_box', capN, (an_rt_P(IP), an_rt_P(s.A), an_rt_P(s.BP), i32(ib), an_rt_P(s.fixd), an_rt_P(s.vbc)), b=256)
        FL = s.floor
        nr = FL['NR'] if FL else [0.0, 0.0, 1.0]
        VR = getattr(s, 'vres', 0.0)
        barE = s.bar and (not VR)
        ST = s.stages(K or s.K)
        s.nstage = len(ST)
        for (kstep, sg) in enumerate(ST):
            it2 = f(1.0 / (sg['tau'] * sg['tau']))
            if sg.get('save0'):
                s.dcopy(s.U0, s.U)
                s.dcopy(s.v0, s.vk)
            if sg['kind'] == 'ga':
                if kstep == 0:
                    if s.ga_first and (not s.matk) and (s.ga != 'be'):
                        s.zf(s.f, 9, 3)
                        s.zf(s.y, 9, 3)
                        s.k('rt_elem', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.y), f(s.mu), f(s.lam), i32(1), an_rt_P(s.SC[10:]), an_rt_P(s.f), an_rt_P(s.H), i32(capQ), an_rt_P(s.B)))
                        mm = s.mn[:int(s.IP[9])]
                        mlow = 0.001 * float(mm[mm > 0].median())
                        s.k('rt_ga_a0', capN, (an_rt_P(IP), an_rt_P(s.mn), an_rt_P(s.f), f(mlow), *[f(q) for q in s.g], an_rt_P(s.fixd), an_rt_P(s.ak)), b=256)
                    else:
                        s.k('rt_ga_anode', capN, (an_rt_P(IP), an_rt_P(s.mn), an_rt_P(s.ma), an_rt_P(s.fixd), an_rt_P(s.ak)), b=256)
                    if s.matk:
                        s.mat_post_anode()
                s.k('rt_ga_step0', capN, (an_rt_P(IP), an_rt_P(s.U), an_rt_P(s.vk), an_rt_P(s.ak), f(sg['dt']), f(sg['am']), f(sg['af']), f(sg['gm']), f(sg['be']), f(sg['tau']), f(sg['al']), *[f(q) for q in s.g], an_rt_P(s.fixd), an_rt_P(s.vbc), an_rt_P(s.Ut), an_rt_P(s.z0), an_rt_P(s.y)), b=256)
                Ub = vb = None
            elif sg['kind'] == 'bdf2':
                s.k('rt_bdf_base', capN, (an_rt_P(IP), f(sg['c1']), f(sg['c0']), an_rt_P(s.U), an_rt_P(s.vk), an_rt_P(s.U0), an_rt_P(s.v0), an_rt_P(s.wb), an_rt_P(s.wvb)), b=256)
                (Ub, vb) = (s.wb, s.wvb)
            else:
                (Ub, vb) = (s.U, s.vk)
            if sg['kind'] != 'ga':
                s.k('rt_step0g', capN, (an_rt_P(IP), an_rt_P(Ub), an_rt_P(vb), f(sg['tau']), f(sg['beta']), *[f(q) for q in s.g], an_rt_P(s.fixd), an_rt_P(s.vbc), an_rt_P(s.U), f(sg['dtf']), an_rt_P(s.Ut), an_rt_P(s.z0), an_rt_P(s.y)), b=256)
            if FL:
                s.zf(s.an, 9, 1)
            if FL and FL['KEX'] > 0:
                s.k('rt_floor_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.xq0), an_rt_P(s.mq), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in FL['PT']], *[f(q) for q in nr], f(FL['PN']), f(FL['DXG']), an_rt_P(s.an)))
                if getattr(s, 'fstk', False):
                    s.zf(s.ana, 9, 1)
                    s.k('rt_floor_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.xq0), an_rt_P(s.mq), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in FL['PT']], *[f(q) for q in nr], f(FL['PN']), f(-FL['DXG']), an_rt_P(s.ana)))
            elif getattr(s, 'fstk', False):
                s.zf(s.ana, 9, 1)
            FA = (an_rt_P(IP), an_rt_P(s.an), an_rt_P(s.mqn), f(FL['KEX'] * (s.K / (K or s.K)) * sg['frac'] if FL else 1.0), i32(1 if FL else 0), i32(1 if FL and FL['vout'] else 0), f(sg['al']), *[f(q) for q in nr], an_rt_P(s.fixd), an_rt_P(s.vbc), an_rt_P(s.U), f(sg['dtf']), an_rt_P(s.Ut), an_rt_P(s.z0), an_rt_P(s.y), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.yh), an_rt_P(s.eta))
            if s.matk:
                s.mat_floor_n(FA, kstep, capN)
            else:
                s.k('rt_floor_n', capN, FA, b=256)
            y0s = s.y0ls and (sg['kind'] == 'ga' or s.bar)
            if y0s:
                s.dcopy(s.dl, s.eta)
            y0u_ = (sg['kind'] == 'ga' or barE) and (not (s.y0ut and (not barE)))
            if y0u_ and s.y0ut:
                s.dcopy(s.ySv, s.y)
                s.dcopy(s.eS, s.eta)
            if y0u_:
                s.k('rt_ga_y0', capN, (an_rt_P(IP), an_rt_P(s.U), an_rt_P(s.yh), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.eta), an_rt_P(s.y)), b=256)
            if s.bar:
                BA = (*[f(q) for q in nr], f(FL['PN']), f(FL['DH']), f(FL['KS']))
                if VR:
                    VA = (*[f(q) for q in nr], f(FL['PN']), f(VR / (sg['tau'] ** 2 * sg['thinv'] ** 2)))
                    s.k('rt_vres_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *VA, i32(0), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])))
                else:
                    s.k('rt_bar_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BA, i32(0), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])))
                if s.bnode:
                    BN = (*[f(q) for q in nr], f(FL['PN']), f(FL['DHN']), f(FL['KSN']))
                    s.zf(s.bnv, 9, 1)
                    if getattr(s, 'det', False):
                        s.k('ao_gath_q', capN, (an_rt_P(IP), an_rt_P(s.cidx), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.bnv)), b=256)
                    else:
                        s.k('rt_barn_vq', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.bnv)))
                    s.k('rt_barn_q', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BN, i32(0), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])), b=256)
                if s.y0ut and (not VR):
                    s.zf(s.kk)
                    s.k('rt_bar_kfe', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.ySv), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.kk)))
                    if s.bnode:
                        s.k('rt_barn_kfe', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.ySv), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.kk)), b=256)
                    WW.y0f(s, kstep)
                    if s.y0eb:
                        s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(0), i32(60)), b=1)
                        s.k('rt_bar_y0e', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.y), an_rt_P(s.ySv), an_rt_P(s.U), f(sg['thinv']), *BA, an_rt_P(s.kk), an_rt_P(s.SC), i32(59)))
                        if s.bnode:
                            s.k('rt_barn_y0e', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.ySv), an_rt_P(s.U), f(sg['thinv']), *BN, an_rt_P(s.kk), an_rt_P(s.SC), i32(59)), b=256)
                        s.k('rt_y0epick', 1, (an_rt_P(s.SC), an_rt_P(s.kk), i32(59)), b=1)
                    s.dcopy(s.Y0K[kstep], s.kk)
                    if getattr(s, 'y0loc', 0):
                        assert not s.bnode, '--y0loc 는 표본점 장벽 판만'
                        R_ = 0 if int(os.environ.get('AF_BLOC', '0')) >= 2 else s.y0loc
                        YM = (an_rt_P(IP), an_rt_P(s.ykn), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.ylE), an_rt_P(s.eS), an_rt_P(s.eta), an_rt_P(s.y))
                        YC = lambda r_: (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.ykn), an_rt_P(s.ykn2), an_rt_P(s.yfl[r_:]))
                        s.dcopy(s.ylU, s.y)
                        s.dcopy(s.ylE, s.eta)
                        s.zf(s.ykn)
                        s.zf(s.ykn2)
                        s.zf(s.yfl)
                        s.k('yl_kq', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.ylU), an_rt_P(s.ySv), an_rt_P(s.U), f(sg['thinv']), *BA, i32(1 if s.y0eb else 0), an_rt_P(s.ykn)))
                        for r_ in range(R_):
                            s.k('yl_mix', capN, YM, b=256)
                            s.k('yl_chk', capQ, YC(r_))
                            s.k('yl_merge', capN, (an_rt_P(IP), an_rt_P(s.ykn), an_rt_P(s.ykn2)), b=256)
                        s.k('yl_mix', capN, YM, b=256)
                        s.k('yl_chk', capQ, YC(R_))
                        if int(os.environ.get('AF_BLOC', '0')) < 2:
                            s.k('yl_fb', capN, (an_rt_P(IP), an_rt_P(s.yfl[R_:]), an_rt_P(s.kk), an_rt_P(s.ykn)), b=256)
                        s.k('yl_mix', capN, YM, b=256)
                        s.k('rt_zero_i', 4, (np.int32(4), an_rt_P(s.YLS[kstep])), b=256)
                        s.k('yl_st', capN, (an_rt_P(IP), an_rt_P(s.ykn), an_rt_P(s.mn), an_rt_P(s.YLS[kstep])), b=256)
                        s.dcopy(s.YLS[kstep, 4:4 + R_ + 1], s.yfl[:R_ + 1])
                    else:
                        s.k('rt_y0mix', capN, (an_rt_P(IP), an_rt_P(s.kk), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.eS), an_rt_P(s.eta), an_rt_P(s.y)), b=256)
            if y0s:
                NC0 = i32(s.ls_n)
                s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(0), i32(60)), b=1)
                s.k('rt_y0dir', capN, (an_rt_P(IP), an_rt_P(s.eta), an_rt_P(s.dl)), b=256)
                s.k('rt_lsyN', capN, (an_rt_P(IP), an_rt_P(s.mn), it2, an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.g0buf), an_rt_P(s.SC), NC0, an_rt_P(None)), b=256)
                s.k('rt_elem_lsN', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), an_rt_P(s.SC), NC0, i32(0), an_rt_P(None)))
                if s.bar and VR:
                    s.k('rt_vres_lsN', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *VA, an_rt_P(s.SC), NC0, i32(0), an_rt_P(None)))
                elif s.bar:
                    s.k('rt_bar_lsN', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *BA[:4], BA[4], BA[5], an_rt_P(s.SC), NC0, i32(0), an_rt_P(None)))
                    if s.bnode:
                        s.k('rt_barn_lsN', capN, (an_rt_P(IP), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *BN, an_rt_P(s.SC), NC0, i32(0), an_rt_P(None)), b=256)
                s.k('rt_y0pick', 1, (an_rt_P(s.SC), NC0), b=1)
                s.dcopy(s.Y0S[kstep], s.SC)
                s.k('rt_lsup', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.y), i32(0)), b=256)
            if getattr(s, 'ncl', False):
                NCA = lambda mode_, it2_=it2: (an_rt_P(IP), i32(mode_), an_rt_P(s.A), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.U), an_rt_P(s.mn), an_rt_P(s.meff), it2_, an_rt_P(s.f), an_rt_P(s.sk), an_rt_P(s.y), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.pm), an_rt_P(s.ncf), an_rt_P(s.yrn), an_rt_P(s.nccnt[4 * kstep:]))
                s.k('rt_ncl', capN, NCA(0), b=256)
            if getattr(s, 'fup', 0):
                FUA = lambda mode_, it2_=it2: (an_rt_P(IP), i32(mode_), *[f(q) for q in nr], an_rt_P(s.U), an_rt_P(s.Ut), an_rt_P(s.z0), an_rt_P(s.an), an_rt_P(s.mn), an_rt_P(s.meff), it2_, an_rt_P(s.f), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.y), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.ufl), an_rt_P(s.fsk), an_rt_P(s.fucnt[4 * kstep:]), f(sg['dt']), f(sg['gm']), f(sg['al']), an_rt_P(s.fvo), an_rt_P(s.fao), an_rt_P(s.ak))
                s.k('rt_fup', capN, FUA(0), b=256)
            PRD.start(s, kstep)
            s.ctl(1)
            if s.cjj:
                s.k('cj_init', 1, (an_rt_P(s.cjs), an_rt_P(s.cjg), i32(kstep)), b=1)
                if getattr(s, 'ufw', False):
                    s.k('ul_wsum', 256, (an_rt_P(IP), an_rt_P(s.meff), f(math.sqrt(sum((q * q for q in s.g)))), an_rt_P(s.cjw), i32(kstep)), b=256)
            WD = getattr(s, 'wd_on', False) and s.adapt
            if WD and len(s.wd_tau) < len(ST):
                s.wd_tau.append(float(sg['tau']))
            if WD:
                s.dcopy(s.WY0[kstep], s.y)
            _xg = int(getattr(s, 'xg', 0)) and int((getattr(s, 'c2', None) or {}).get('xf') or 0)
            for it in range(s.nt):
                if _xg and it == _xg:
                    C3.mark(s, 'q')
                s.ctl(2, it)
                s.zf(s.f, 9, 3)
                s.zf(s.B, 9, 9)
                s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(0), i32(60)), b=1)
                if s.matk:
                    s.mat_elem(kstep, it, len(ST), capQ)
                else:
                    s.k('rt_elem', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8 | (4 if s.ls else 0)), an_rt_P(s.SC[10:]), an_rt_P(s.f), an_rt_P(s.H), i32(capQ), an_rt_P(s.B)))
                PRD.fel(s, kstep, it)
                if WD and (not s.matk):
                    s.dcopy(s.WFE[kstep * s.nt + it], s.f)
                    s.dcopy(s.WBE[kstep * s.nt + it], s.B)
                if s.bar:
                    s.zf(s.bGn)
                    s.zf(s.bKn)
                    if VR:
                        s.k('rt_vres_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *VA, i32(1), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])))
                    else:
                        s.k('rt_bar_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BA, i32(1), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])))
                    if getattr(s, 'det', False):
                        s.k('ao_gath_q', capN, (an_rt_P(IP), an_rt_P(s.cidx), an_rt_P(s.BW), an_rt_P(s.bpg), an_rt_P(s.bGn)), b=256)
                        s.k('ao_gath_q', capN, (an_rt_P(IP), an_rt_P(s.cidx), an_rt_P(s.BW), an_rt_P(s.bph), an_rt_P(s.bKn)), b=256)
                    if s.bnode:
                        s.k('rt_barn_q', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BN, i32(1), an_rt_P(s.bGn), an_rt_P(s.bKn), an_rt_P(s.SC[10:])), b=256)
                    s.k('rt_bar_n', capN, (an_rt_P(IP), an_rt_P(s.bGn), an_rt_P(s.bKn), *[f(q) for q in nr], an_rt_P(s.f), an_rt_P(s.B)), b=256)
                    if s.mgbar >= 2 and s.nl > 1 and s.mv2:
                        for l_ in range(1, s.nl):
                            Lc_ = s.ML[l_ - 1]
                            s.k('rt_mg_restrict3', Lc_['capn'] * 16, (i32(l_ - 1), an_rt_P(s.LM), an_rt_P(s.bKn if l_ == 1 else s.ML[l_ - 2]['Kb']), an_rt_P(Lc_['Kb']), i32(1)), b=256)
                            s.k('rt_mg_dbar', Lc_['capn'], (i32(l_), an_rt_P(s.LM), an_rt_P(Lc_['D']), an_rt_P(Lc_['Kb']), *[f(q) for q in nr], an_rt_P(Lc_['Db'])), b=256)
                        s._mgb_on = True
                PRD.bar(s, kstep, it)
                if it == 0:
                    s.k('rt_meff', capN, (an_rt_P(IP), an_rt_P(s.mn), an_rt_P(s.B), f(s.mfloor * sg['tau'] ** 2), an_rt_P(s.meff)), b=256)
                    if WD:
                        s.dcopy(s.WME[kstep], s.meff)
                if getattr(s, 'ncl', False):
                    s.k('rt_ncl', capN, NCA(1), b=256)
                if getattr(s, 'fup', 0):
                    s.k('rt_fup', capN, FUA(1), b=256)
                s.k('rt_grad', capN, (an_rt_P(IP), an_rt_P(s.meff), it2, an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.eta), an_rt_P(s.f), an_rt_P(s.B), an_rt_P(s.Dinv), an_rt_P(s.r), an_rt_P(s.z), an_rt_P(s.p), an_rt_P(s.dl), an_rt_P(s.Tp), an_rt_P(s.SC), an_rt_P(s.g0buf)), b=256)
                s.ctl(3, it)
                WW.ct(s, kstep, it, 0)
                PRD.grad(s, kstep, it)
                if s.cjj and s.cjr and s.adapt:
                    if getattr(s, 'ufw', False):
                        s.k('ul_gw', 1, (an_rt_P(s.SC), an_rt_P(s.CD), an_rt_P(s.cjw), an_rt_P(s.cjg), i32(kstep)), b=1)
                    else:
                        s.k('cj_g', 1, (an_rt_P(s.SC), an_rt_P(s.CD), an_rt_P(s.cjg), i32(kstep)), b=1)
                if s.cjj and (getattr(s, 'c2', None) or {}).get('xf') and (it == int(s.c2['xf'])):
                    s.k('c2_gate', 1, (an_rt_P(s.CT), an_rt_P(s.cjs), an_rt_P(s.cjg), i32(kstep), i32(1 if s.cjr else 0)), b=1)
                    if _xg:
                        C3.mark(s, 'g')
                if WD:
                    s.dcopy(s.WCT[2 * (kstep * s.nt + it)], s.CT)
                    s.dcopy(s.WCD[2 * (kstep * s.nt + it)], s.CD)
                if s.nl > 1:
                    if it == 0 and (not (s.mg_wreuse and kstep > 0 and (abs(getattr(s, 'mg_tau', -1.0) - sg['tau']) == 0.0))):
                        s.mg_tau = sg['tau']
                        s.mg_power(0)
                    s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(0), i32(1)), b=1)
                    s.mg_vc(0, s.r, s.z)
                    s.zf(s.zo, 9, 3)
                    s.k('rt_dotz', capN, (an_rt_P(IP), an_rt_P(s.r), an_rt_P(s.z), an_rt_P(s.zo), an_rt_P(s.SC), i32(1)), b=256)
                    s.k('rt_pinit', capN, (an_rt_P(IP), an_rt_P(s.z), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.p), an_rt_P(s.Tp)), b=256)
                s.dcopy(s.DG[2 * (kstep * s.nt + it)], s.SC)
                s.ctl(4, it)
                if s.descent:
                    s.dcopy(s.p0, s.p)
                    s.k('rt_sc_copy', 1, (an_rt_P(s.SC), i32(60), i32(0)), b=1)
                for ci in range(s.cg_list[it] if s.cg_list else s.cg):
                    s.ctl(5, ci)
                    if ci == 0 or not s.fuse2:
                        s.zf(s.KTp, 9, 3)
                    s.k(s.fmv_name(), capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.fmv_H()), an_rt_P(s.Tp), an_rt_P(s.KTp), i32(capQ)))
                    if s.bar and s.bfull:
                        s.zf(s.bS, 9, 1)
                        s.k('rt_bar_mvq', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), *[f(q) for q in nr], an_rt_P(s.Tp), an_rt_P(s.bS), an_rt_P(s.btq)))
                        if getattr(s, 'det', False):
                            s.k('ao_gath_q', capN, (an_rt_P(IP), an_rt_P(s.cidx), an_rt_P(s.BW), an_rt_P(s.btq), an_rt_P(s.bS)), b=256)
                        s.k('rt_bar_mvn', capN, (an_rt_P(IP), an_rt_P(s.bS), *[f(q) for q in nr], an_rt_P(s.KTp)), b=256)
                    elif s.bar and (not s.mgbar):
                        s.k('rt_bar_mv', capN, (an_rt_P(IP), an_rt_P(s.bKn), *[f(q) for q in nr], an_rt_P(s.Tp), an_rt_P(s.KTp)), b=256)
                    s.k('rt_ap', capN, (an_rt_P(IP), an_rt_P(s.meff), it2, an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.p), an_rt_P(s.KTp), an_rt_P(s.Ap), an_rt_P(s.SC), an_rt_P(s.bKn if s.bar and s.mgbar and (not s.bfull) else None)), b=256)
                    if s.descent and ci == 0:
                        s.k('rt_sc_copy', 1, (an_rt_P(s.SC), i32(61), i32(2)), b=1)
                    if s.nl > 1:
                        s.k('rt_cg1m', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.p), an_rt_P(s.Ap), an_rt_P(s.dl), an_rt_P(s.r)), b=256)
                        s.pk_chain('rt_cg1m', 'rt_mg_node')
                        s.mg_vc(0, s.r, s.z)
                        s.pk_chain('rt_mg_node', 'rt_dotz')
                        s.k('rt_dotz', capN, (an_rt_P(IP), an_rt_P(s.r), an_rt_P(s.z), an_rt_P(s.zo), an_rt_P(s.SC), i32(0)), b=256)
                    else:
                        s.k('rt_cg1', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.Dinv), an_rt_P(s.p), an_rt_P(s.Ap), an_rt_P(s.dl), an_rt_P(s.r), an_rt_P(s.z), an_rt_P(s.SC)), b=256)
                    s.k('rt_cg2', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.z), an_rt_P(s.p), an_rt_P(s.Tp), an_rt_P(s.KTp if s.fuse2 else None)), b=256)
                    s.k('rt_sc_roll', 1, (an_rt_P(s.SC),), b=1)
                    s.ctl(6, ci)
                s.ctl(7, it)
                WW.ct(s, kstep, it, 1)
                if WD:
                    s.dcopy(s.WCT[2 * (kstep * s.nt + it) + 1], s.CT)
                    s.dcopy(s.WCD[2 * (kstep * s.nt + it) + 1], s.CD)
                if s.ls and s.ls_n:
                    NC = i32(s.ls_n)
                    s.lsy(capN, it2, NC, None)
                    if s.descent:
                        s.k('rt_fixdir', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.p0), an_rt_P(s.dl)), b=256)
                        if s.lsfast:
                            G59 = an_rt_P(s.SC[59:])
                            s.k('rt_sc_clear_g', 1, (an_rt_P(s.SC), i32(9), i32(10), G59), b=1)
                            s.k('rt_sc_clear_g', 1, (an_rt_P(s.SC), i32(20), i32(40), G59), b=1)
                            s.lsy(capN, it2, NC, s.SC[59:])
                        else:
                            s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(9), i32(10)), b=1)
                            s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(20), i32(40)), b=1)
                            s.lsy(capN, it2, NC, None)
                    EA = (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), an_rt_P(s.SC), NC)
                    BB = (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.Vq), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *BA[:4], BA[4], BA[5], an_rt_P(s.SC), NC) if s.bar else None
                    BLS = 'rt_bar_lsN'
                    if s.bar and VR:
                        BB = (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *VA, an_rt_P(s.SC), NC)
                        BLS = 'rt_vres_lsN'
                    BBN = (an_rt_P(IP), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), an_rt_P(s.U), f(sg['thinv']), *BN, an_rt_P(s.SC), NC) if s.bar and s.bnode else None
                    if s.lsfast:
                        G63 = an_rt_P(s.SC[63:])
                        s.k('rt_elem_lsN', capQ, EA[:-1] + (i32(1), i32(0), an_rt_P(None)))
                        if s.bar:
                            s.k(BLS, capQ, BB[:-1] + (i32(1), i32(0), an_rt_P(None)))
                        if BBN:
                            s.k('rt_barn_lsN', capN, BBN[:-1] + (i32(1), i32(0), an_rt_P(None)), b=256)
                        s.k('rt_lsgate', 1, (an_rt_P(s.SC),), b=1)
                        s.k('rt_elem_lsN', capQ, EA + (i32(1), G63))
                        if s.bar:
                            s.k(BLS, capQ, BB + (i32(1), G63))
                        if BBN:
                            s.k('rt_barn_lsN', capN, BBN + (i32(1), G63), b=256)
                    else:
                        s.k('rt_elem_lsN', capQ, EA + (i32(0), an_rt_P(None)))
                        if s.bar:
                            s.k(BLS, capQ, BB + (i32(0), an_rt_P(None)))
                        if BBN:
                            s.k('rt_barn_lsN', capN, BBN + (i32(0), an_rt_P(None)), b=256)
                    s.k('rt_lspickN', 1, (an_rt_P(s.SC), NC), b=1)
                elif s.ls:
                    assert not s.bar, '장벽 판은 --ls_n 선탐색만'
                    s.k('rt_lsy4', capN, (an_rt_P(IP), an_rt_P(s.meff), it2, an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.g0buf), an_rt_P(s.SC)), b=256)
                    if s.descent:
                        s.k('rt_fixdir', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.p0), an_rt_P(s.dl)), b=256)
                        s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(5), i32(10)), b=1)
                        s.k('rt_lsy4', capN, (an_rt_P(IP), an_rt_P(s.meff), it2, an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.g0buf), an_rt_P(s.SC)), b=256)
                    s.k('rt_elem_ls', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), an_rt_P(s.SC)))
                    s.k('rt_lspick', 1, (an_rt_P(s.SC),), b=1)
                s.dcopy(s.DG[2 * (kstep * s.nt + it) + 1], s.SC)
                fbar = not s.ls and barE
                if fbar:
                    s.k('rt_ytry', capN, (an_rt_P(IP), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.yc)), b=256)
                    s.zf(s.kk)
                    if getattr(s, 'bloc', False):
                        s.zf(s.bkn)
                        s.k('rt_bar_kfe', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.yc), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(-1e+30), an_rt_P(s.kk)))
                        s.k('bl_kq', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.yc), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.bkn)))
                    elif not getattr(s, 'nobk', False):
                        s.k('rt_bar_kfe', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.yc), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.kk)))
                    if s.bnode:
                        s.k('rt_barn_kfe', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.yc), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), an_rt_P(s.kk)), b=256)
                    s.k('rt_kstep', 1, (an_rt_P(s.kk), an_rt_P(s.SC)), b=1)
                    s.dcopy(s.BSK[kstep * s.nt + it], s.kk)
                if fbar and getattr(s, 'bloc', False):
                    s.k('bl_lsup', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.y), an_rt_P(s.bkn)), b=256)
                else:
                    s.k('rt_lsup', capN, (an_rt_P(IP), an_rt_P(s.SC), an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.yh), an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.y), i32(0 if s.ls or fbar else 1)), b=256)
                if s.cjj and (s.ls or fbar):
                    s.k('cj_min', 1, (an_rt_P(s.SC), an_rt_P(s.cjs), i32(kstep)), b=1)
                if getattr(s, 'ncl', False):
                    s.k('rt_ncl', capN, NCA(2), b=256)
                if getattr(s, 'fup', 0):
                    s.k('rt_fup', capN, FUA(2), b=256)
                if WD:
                    s.dcopy(s.WY[kstep * s.nt + it], s.y)
            if _xg:
                C3.mark(s, 'e')
            s.ctl(8)
            WW.wolfwhy_diag_end(s, kstep, sg, capQ, capN, it2, BA if s.bar else None, nr, barE)
            if s.cjj:
                s.k('cj_end', 1, (an_rt_P(s.CT), an_rt_P(s.cjs), an_rt_P(s.cjg), an_rt_P(s.cjl), i32(kstep), i32(1 if s.adapt else 0), i32(1 if s.cjr else 0)), b=1)
            if WD:
                s.dcopy(s.WCT[2 * s.nt * len(ST) + kstep], s.CT)
            if getattr(s, 'ncl', False):
                s.k('rt_ncl', capN, NCA(3), b=256)
            if getattr(s, 'fup', 0):
                s.k('rt_fup', capN, FUA(3), b=256)
            if getattr(s, 'becell', False):
                s.k('rt_bare_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BA, an_rt_P(s.Vq), an_rt_P(s.beC), an_rt_P(s.beT[4 * kstep:])))
                if s.bnode:
                    s.k('rt_bare_n', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *BN, an_rt_P(s.beC), an_rt_P(s.beT[4 * kstep:])), b=256)
                s.k('rt_bare_acc', capN, (an_rt_P(IP), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), an_rt_P(s.beC), an_rt_P(s.beL), an_rt_P(s.beM), an_rt_P(s.beD)), b=256)
            if s.gfin:
                s.zf(s.f, 9, 3)
                s.zf(s.B, 9, 9)
                s.k('rt_sc_clear', 1, (an_rt_P(s.SC), i32(0), i32(60)), b=1)
                s.k('rt_elem', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.Vq), an_rt_P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8), an_rt_P(s.SC[10:]), an_rt_P(s.f), an_rt_P(s.H), i32(capQ), an_rt_P(s.B)))
                s.k('rt_grad', capN, (an_rt_P(IP), an_rt_P(s.meff), it2, an_rt_P(s.sk), an_rt_P(s.pm), *[f(q) for q in nr], an_rt_P(s.eta), an_rt_P(s.f), an_rt_P(s.B), an_rt_P(s.Dinv), an_rt_P(s.r), an_rt_P(s.z), an_rt_P(s.p), an_rt_P(s.dl), an_rt_P(s.Tp), an_rt_P(s.SC), an_rt_P(s.g0buf)), b=256)
                s.dcopy(s.DGF[kstep], s.SC)
            if FL:
                s.zf(s.an2, 9, 1)
                s.k('rt_floor_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.xq0), an_rt_P(s.mq), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in FL['PT']], *[f(q) for q in nr], f(FL['PN']), f(FL['DXG']), an_rt_P(s.an2)))
            if FL and kstep == 0:
                s.k('rt_count_stuck', capN, (an_rt_P(IP), an_rt_P(s.an), an_rt_P(IP)), b=256)
            dmp = f(s.damp ** (s.h / (K or s.K) / (s.h / s.K)) if s.damp != 1.0 and sg['last'] else 1.0)
            if sg['kind'] == 'ga' and getattr(s, 'fup', 0) and s.areset:
                s.dcopy(s.fvo, s.vk)
                s.dcopy(s.fao, s.ak)
            if sg['kind'] == 'ga':
                GE = (an_rt_P(IP), f(sg['dt']), f(sg['gm']), f(sg['be']), f(sg['al']), an_rt_P(s.y), an_rt_P(s.z0), an_rt_P(s.U), an_rt_P(s.vk), an_rt_P(s.ak), an_rt_P(s.an if FL else None), i32(1 if s.areset else 0), an_rt_P(s.fixd), an_rt_P(s.vbc), dmp)
                if getattr(s, 'uel', 0):
                    s.k('ul_zero', capN, (an_rt_P(IP), an_rt_P(s.ulm)), b=256)
                    if s.uel & 1 and s.bar:
                        s.k('ul_bar', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BW), an_rt_P(s.y), an_rt_P(s.U), f(sg['thinv']), *[f(q) for q in nr], f(FL['PN']), f(FL['DH']), an_rt_P(s.ulm)))
                    if s.uel & 2 and getattr(s, 'y0loc', 0):
                        s.k('ul_kn', capN, (an_rt_P(IP), an_rt_P(s.ykn), an_rt_P(s.ulm)), b=256)
                    if s.uel & 4:
                        s.k('ul_res', capN, (an_rt_P(IP), an_rt_P(s.g0buf), an_rt_P(s.meff), f(math.sqrt(sum((q * q for q in s.g)))), an_rt_P(s.ulm)), b=256)
                    if s.uel & 32:
                        s.k('ul_big', capN, (an_rt_P(IP), an_rt_P(s.cjg), i32(kstep), f(s.c2.get('bigT', 1000000.0)), an_rt_P(s.ulm)), b=256)
                    if s.uel & 16:
                        s.k('ul_flip', capN, (an_rt_P(IP), f(sg['al']), an_rt_P(s.y), an_rt_P(s.z0), an_rt_P(s.vk), an_rt_P(s.ulm)), b=256)
                    if (getattr(s, 'c2', None) or {}).get('dil'):
                        C2 = sys.modules[__name__]
                        C2.dilate(s, capN, IP, s.c2['dil'])
                    s.k('rt_zero_i', 4, (np.int32(4), an_rt_P(s.ULS[kstep])), b=256)
                    s.k('ul_st', capN, (an_rt_P(IP), an_rt_P(s.ulm), an_rt_P(s.mn), an_rt_P(s.ULS[kstep])), b=256)
                    s.k('rt_ga_end_ufl', capN, GE + (an_rt_P(s.cjl[kstep:]), i32(s.cue), an_rt_P(s.ulm)), b=256)
                elif getattr(s, 'cue', 0):
                    s.k('rt_ga_end_uf', capN, GE + (an_rt_P(s.cjl[kstep:]), i32(s.cue)), b=256)
                else:
                    s.k('rt_ga_end', capN, GE, b=256)
                if getattr(s, 'fup', 0) and s.areset:
                    s.k('rt_fup', capN, FUA(4), b=256)
            elif kstep < len(ST) - 1:
                assert not s.matk, '재질 판은 ga 만'
                s.k('rt_stepend_g', capN, (an_rt_P(IP), f(sg['thinv']), f(sg['al']), an_rt_P(s.y), an_rt_P(s.z0), an_rt_P(s.U), an_rt_P(s.vk), an_rt_P(s.fixd), an_rt_P(s.vbc), dmp), b=256)
            if s.matk:
                s.mat_step_end(kstep, len(ST), capN, capQ)
            PRD.wolfprop_diag_end(s, kstep)
            if getattr(s, 'wd_on', False) and s.adapt:
                s.dcopy(s.WU[kstep], s.U)
        sl = ST[-1]
        if s.matk:
            s.mat_g2p()
            return
        if sl['kind'] == 'ga':
            s.k('rt_g2p_ga', N, (i32(N), an_rt_P(IP), an_rt_P(s.Ai), an_rt_P(s.Np), an_rt_P(s.fx), an_rt_P(s.U), an_rt_P(s.vk), an_rt_P(s.ak), an_rt_P(s.x), an_rt_P(s.v), an_rt_P(s.pa), an_rt_P(s.F)))
            if s.fpi:
                s.k('fpi_q', capQ, (an_rt_P(IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.Fq), an_rt_P(s.U), an_rt_P(s.Fe)))
                s.k('fpi_p', N, (i32(N), an_rt_P(IP), an_rt_P(s.fx), an_rt_P(s.pcd), an_rt_P(s.cidx), an_rt_P(s.Fe), i32(0), i32(0), an_rt_P(s.F), an_rt_P(None), an_rt_P(None), an_rt_P(None), an_rt_P(None)))
            return
        assert not s.fpi, '--fp_interp 는 generalized-α (--ga) 판만'
        s.k('rt_g2p_endg', N, (i32(N), an_rt_P(IP), an_rt_P(s.Ai), an_rt_P(s.Np), an_rt_P(s.fx), an_rt_P(s.y), an_rt_P(s.z0), an_rt_P(s.U), f(sl['thinv']), f(sl['al']), an_rt_P(s.fixd), an_rt_P(s.vbc), dmp, an_rt_P(s.x), an_rt_P(s.v), an_rt_P(s.F)))

    def lsy(s, capN, it2, NC, gate):
        G = an_rt_P(gate) if gate is not None else an_rt_P(None)
        if s.fk:
            s.k('rt_lsyQ', capN, (an_rt_P(s.IP), an_rt_P(s.meff), it2, an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.g0buf), an_rt_P(s.SC), G), b=256)
            s.k('rt_lsyfin', 1, (an_rt_P(s.SC), NC, G), b=1)
        else:
            s.k('rt_lsyN', capN, (an_rt_P(s.IP), an_rt_P(s.meff), it2, an_rt_P(s.eta), an_rt_P(s.dl), an_rt_P(s.g0buf), an_rt_P(s.SC), NC, G), b=256)

    def dcopy(s, dst, src):
        if s._pkc is None:
            dst.copy_(src)
            return
        nw = dst.numel() * dst.element_size() // 4
        assert src.numel() * src.element_size() // 4 == nw and dst.is_contiguous() and src.is_contiguous()
        s.k('pk_copy32', nw, (np.int32(nw), an_rt_P(dst), an_rt_P(src)), b=256)

    def bzero(s, t):
        if s._pkc is None:
            t.zero_()
            return
        s.k('pk_zero_b', t.numel(), (np.int32(t.numel()), an_rt_P(t)), b=256)

    def pk_chain(s, prod, cons):
        if s._pkc is None or os.environ.get('AF_PK_CHAIN', '1') != '1':
            return
        assert s._pkc[-1][0] == prod, (s._pkc[-1][0], prod)
        s._pk_chain_set.add(len(s._pkc) - 1)
        s._pk_chain_next = cons

    def pk_profile(s, t):
        raise NotImplementedError('not used by the paper settings')

    def ctl(s, op, it=0):
        if not s.adapt:
            return
        eN2 = s.tolN * s.tolN if s.tolN >= 0 else -1.0
        eP2 = s.tolP * s.tolP if s.tolP >= 0 else -1.0
        s.k('rt_ctl', 1, (np.int32(op), np.int32(it), an_rt_P(s.SC), an_rt_P(s.CT), an_rt_P(s.CD), np.float64(eN2), np.float64(eP2)), b=1)

    def _bk(s):
        return tuple((getattr(s, nm).clone() for nm in ('x', 'v', 'F', 'm', 'vol', 'perm', 'pa')))

    def _rs(s, bk):
        for (nm, t) in zip(('x', 'v', 'F', 'm', 'vol', 'perm', 'pa'), bk):
            getattr(s, nm).copy_(t)

    def set_bc(s, t):
        if not s.bc:
            return
        v = []
        for (c_, sz_, vel_, t0, t1) in s.bc:
            on = 1.0 if t0 <= t < t1 else 0.0
            v += [c_[a] + vel_[a] * (t - t0) for a in range(3)] + list(sz_) + list(vel_) + [on]
        s.BP.copy_(torch.tensor(v, dtype=torch.float32))

    def capture(s):
        raise NotImplementedError('not used by the paper settings')

    def step(s, t):
        bk = s._bk()
        if s.ga and s.ga_first:
            for tries in range(4):
                s.set_bc(t)
                (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                e0.record()
                s.record(t, s.Kcur)
                e1.record()
                torch.cuda.synchronize()
                if int(s.IP[13]) == 0:
                    break
                s._rs(bk)
                s.cap_mul = 1.5 + 0.5 * (tries + 1)
                s.alloc(None)
                s._rs(bk)
            s.ga_first = False
            s.last_ms = e0.elapsed_time(e1)
            s.contact_prev = int(s.IP[14]) > 0
            s.last_K = s.Kcur
            return False
        Kw = s.Kc if s.Kc and getattr(s, 'contact_prev', False) else s.K
        if Kw != s.Kcur:
            s.Kcur = Kw
            s.graph = s.graphs.get(Kw)
        if s.graph is None:
            s.capture()
            s._rs(bk)
        s.set_bc(t)
        (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        e0.record()
        s.graph.replay()
        e1.record()
        redo = False
        tries = 0
        fr = int(round(t / s.h))
        force = fr in _force_redo_frames()
        while int(s.IP[13]) != 0 or force:
            force = False
            if not bool(torch.isfinite(bk[0]).all()) or tries >= 3:
                s.x.fill_(float('nan'))
                s.last_ms = float('nan')
                return False
            tries += 1
            s._rs(bk)
            snap = s._cap_snap() if os.environ.get('AF_REDO_COPY', '0') == '1' else None
            s.cap_mul = 1.5 + 0.5 * tries
            s.alloc(None)
            s.capture()
            s._rs(bk)
            if snap is not None:
                s._cap_put(snap)
                del snap
            s.set_bc(t)
            e0.record()
            s.graph.replay()
            e1.record()
            redo = True
        torch.cuda.synchronize()
        s.last_ms = e0.elapsed_time(e1)
        s.contact_prev = int(s.IP[14]) > 0
        s.last_K = s.Kcur
        return redo
    _CAP_SKIP = ('capn_d', 'capq_d', 'mg_MP', 'mg_MI', 'IP', 'SC', 'CT', 'CD', 'BW', 'BG', 'BP', 'A', 'Ai')

    def _cap_snap(s):
        raise NotImplementedError('not used by the paper settings')

    def _cap_put(s, snap):
        raise NotImplementedError('not used by the paper settings')

    def counts(s):
        ip = s.IP.cpu().tolist()
        o = dict(nn=ip[9], nc=ip[11], nq=ip[12], nb=ip[6:9], nbad=int(s.nbad))
        if s.adapt:
            ct = s.CT.cpu().tolist()
            o['it'] = dict(nN=ct[2], nP=ct[3], capN=ct[4], capP=ct[5], steps=ct[6], solves=ct[7])
        return o

    def bar_counts(s):
        nq = int(s.IP[12])
        fl = s.bcf[:nq]
        occ = s.bx[:nq * 4].view(-1, 4)[:, 3] > 0
        eng = fl & 1 > 0
        cr = (fl & 2 > 0) & occ
        cid = torch.arange(nq, device=fl.device) >> 3
        return dict(neng=int(eng.sum()), nocc=int(occ.sum()), ceng=int(torch.unique(cid[eng]).numel()), nc=nq // 8, ccr=int(torch.unique(cid[cr]).numel()), ncr=int(cr.sum()), engcr=int((eng & cr).sum()))

    def qjd(s, nr, pn):
        if getattr(s, '_qjd_fn', None) is None:
            s._qjd_fn = cp.RawModule(code=QJD_SRC, options=('--std=c++14',)).get_function('rt_qjd')
        torch.cuda.synchronize()
        nq = int(s.IP[12])
        capQ = s.Fq.shape[0]
        Jq = torch.full((capQ,), float('nan'), device=s.Fq.device, dtype=torch.float64)
        Xq = torch.zeros(capQ, 3, device=s.Fq.device, dtype=torch.float32)
        s._qjd_fn(((nq + 127) // 128,), (128,), (an_rt_P(s.IP), an_rt_P(s.cb), an_rt_P(s.BG), an_rt_P(s.BW), an_rt_P(s.Fq), an_rt_P(s.xq0), an_rt_P(s.U), an_rt_P(Jq), an_rt_P(Xq)))
        cp.cuda.runtime.deviceSynchronize()
        v = s.Vq[:nq] > 0
        J = Jq[:nq][v]
        X = Xq[:nq][v].double()
        o = dict(nv=int(v.sum()), nneg=int((J <= 0).sum()), nnf=int((~torch.isfinite(J)).sum()))
        if J.numel():
            Jf = torch.nan_to_num(J, nan=-1e+30)
            w = int(torch.argmin(Jf))
            o.update(Jmin=float(Jf[w]), x=X[w].tolist(), h=float(X[w] @ torch.tensor(nr, device=X.device, dtype=torch.float64) - pn))
            if o['nneg']:
                ng = J <= 0
                hh = X[ng] @ torch.tensor(nr, device=X.device, dtype=torch.float64) - pn
                o.update(neg_h=[float(hh.min()), float(hh.median()), float(hh.max())], neg_ctr=X[ng].mean(0).tolist())
        return o

    def bar_ecell(s):
        raise NotImplementedError('not used by the paper settings')

    def x_orig(s):
        o = torch.empty_like(s.x)
        o[s.perm] = s.x
        return o

    def diag(s):
        raise NotImplementedError('not used by the paper settings')

def set_K_an(s, K, h):
    if not hasattr(s, '_k0'):
        s._k0 = (s.K, (s.floor or {}).get('KEX'), s.damp)
    (K0, kex0, dmp0) = s._k0
    s.K = int(K)
    s.dt = h / s.K
    if s.floor and kex0 is not None:
        s.floor['KEX'] = kex0 * K0 / s.K
    s.damp = dmp0 ** (K0 / s.K) if dmp0 != 1.0 else 1.0

def an_rt_split_step(s, t, h, depth):
    if s.ga and getattr(s, 'ga_first', False):
        set_K_an(s, 1, h)
        redo = s.step(t)
        return (redo, 1, [(1, int(s.cjl[:1].sum()), float(s.last_ms))])
    bk = s._bk()
    (tries, tot, redo) = ([], 0.0, False)
    for dp in range(depth + 1):
        K = 1 << dp
        if dp > 0:
            s._rs(bk)
        set_K_an(s, K, h)
        redo = s.step(t) or redo
        ms = float(s.last_ms)
        nf = C3.nfail(s, s.cjl, s.cjg, K)
        tries.append((K, nf, ms))
        tot += ms
        if nf == 0 or not math.isfinite(ms):
            break
    s.last_ms = tot
    return (redo, K, tries)

# ---------------- fem_an_det ----------------
ART = sys.modules[__name__]
an_det_P = ART.an_rt_P
an_det_MAXB = 65536
an_det_NSL = 96
an_det_DET_DECL = cu('an_det_DET_DECL')
an_det_DET_SRC = cu('an_det_DET_SRC')

def an_det__kslice(src, name):
    import re
    mt = re.search('__global__ void (?:__launch_bounds__\\(\\d+\\) )?' + name + '\\(', src)
    i = mt.start()
    j = src.index('\n}\n', i) + 3
    return (i, j)

def an_det__sub_in(src, name, old, new, count=1):
    (i, j) = an_det__kslice(src, name)
    body = src[i:j]
    assert old in body, (name, old[:40])
    return src[:i] + body.replace(old, new, count) + src[j:]

def an_det_det_src(head):
    s = head.replace('extern "C" {', 'extern "C" {\n' + an_det_DET_DECL, 1)
    s = s.replace('if (l == 0) atomicAdd(out, v); }', 'if (l == 0) ao_bs_out(out, v); }', 1)
    s = an_det__sub_in(s, 'rt_elem', 'if (on && head) atomicAdd(&f[ids[o]*3+a], s);', 'if (on && head) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = s; else atomicAdd(&f[ids[o]*3+a], s); }')
    s = an_det__sub_in(s, 'rt_elem', 'if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]);', 'if (on && head) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = m[k]; else atomicAdd(&B[ids[o]*9+k], m[k]); }')
    s = an_det__sub_in(s, 'rt_mv', 'if (on && head) atomicAdd(&out[ids[o]*3+a], s);', 'if (on && head) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = s; else atomicAdd(&out[ids[o]*3+a], s); }')
    s = an_det__sub_in(s, 'rt_gauss', 'atomicAdd(&mqn[nid], BW[oc*27+o]*mm);', 'if (!AO_DET) atomicAdd(&mqn[nid], BW[oc*27+o]*mm);')
    s = an_det__sub_in(s, 'rt_floor_q', 'if (!(s < 0.f)) return;', 'if (!(s < 0.f)) { if (AO_DET) AO_MW[q] = 0.f; return; }')
    s = an_det__sub_in(s, 'rt_floor_q', 'if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);', 'if (AO_DET) AO_MW[q] = mw; else if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);')
    s = an_det__sub_in(s, 'rt_mg_elem', 'if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }', 'if (on && head) for (int k=0;k<9;k++) { if (AO_DET) AO_MGB[((size_t)(qq>>3)*27+o)*9+k] = m[k]; else atomicAdd(&B[ids[o]*9+k], m[k]); } }')
    s = an_det__sub_in(s, 'rt_mg_mv', 'if (on && head) atomicAdd(&out[ids[o]*3+a], s); }', 'if (on && head) { if (AO_DET) AO_MGV[((size_t)(qq>>3)*27+o)*3+a] = s; else atomicAdd(&out[ids[o]*3+a], s); } }')
    return s

def an_det_det_ao(ao_src):
    s = an_det__sub_in(ao_src, 'ao_elem', 'if (on && head) atomicAdd(&f[ids[o]*3+a], sv); } }', 'if (on && head) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = sv; else atomicAdd(&f[ids[o]*3+a], sv); } } }')
    s = an_det__sub_in(s, 'ao_elem', 'if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], mm[k]); }', 'if (on && head) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); } }')
    return s
an_det_BS_KERNELS = {'rt_elem', 'rt_grad', 'rt_grad_c', 'rt_ap', 'rt_ap_c', 'rt_cg1', 'rt_cg1_c', 'rt_lsy', 'rt_lsy4', 'rt_elem_ls', 'ao_warm', 'rt_fixdir', 'rt_lsyN', 'rt_elem_lsN', 'rt_dotz', 'rt_mg_pw1', 'rt_bar_q', 'rt_bar_lsN', 'rt_lsyQ', 'rt_bar_y0e', 'rt_barn_q', 'rt_barn_lsN', 'rt_barn_y0e', 'rt_vres_q', 'rt_vres_lsN'}

class an_det_DetMixin(ART.an_rt_RTSolver):

    def det_alloc(s):
        Z = lambda *sh, dt=torch.float32: torch.zeros(*sh, device=s.dev, dtype=dt)
        s.d_part = Z(an_det_NSL * an_det_MAXB, dt=torch.float64)
        s.d_bmax = Z(an_det_NSL, dt=torch.int32)
        s.d_cf = Z(s.capQc * 27 * 3, dt=torch.float64)
        s.d_cb = Z(s.capQc * 27 * 9)
        s.d_cv = Z(s.capQc * 27 * 3)
        s.d_mw = Z(s.capQc * 8)
        s.d_st = torch.zeros(s.capQc + 1, device=s.dev, dtype=torch.int32)
        mgc = max([L['capq'] // 8 for L in getattr(s, 'ML', [])] or [1])
        s.d_mgb = Z(mgc * 27 * 9)
        s.d_mgv = Z(mgc * 27 * 3)
        s.d_mscr = Z(s.capN)

    def det_bind(s):
        for mod in s._mods():
            s._det_bind1(mod)

    def _det_bind1(s, mod):

        def setp(nm, t, dt):
            g = mod.get_global(nm)
            ART.cp.ndarray((1,), dtype=dt, memptr=g)[0] = int(t.data_ptr()) if t is not None else 0
        setp('AO_SCB', s.SC, ART.cp.uint64)
        setp('AO_PART', s.d_part, ART.cp.uint64)
        setp('AO_BMAX', s.d_bmax, ART.cp.uint64)
        setp('AO_SCM', getattr(s, 'SCM', None), ART.cp.uint64)
        setp('AO_MGB', s.d_mgb, ART.cp.uint64)
        setp('AO_MGV', s.d_mgv, ART.cp.uint64)
        setp('AO_CF', s.d_cf, ART.cp.uint64)
        setp('AO_CB', s.d_cb, ART.cp.uint64)
        setp('AO_CV', s.d_cv, ART.cp.uint64)
        setp('AO_MW', s.d_mw, ART.cp.uint64)
        import os
        setp('AO_ACTL', s.act if getattr(s, 'compact', False) and os.environ.get('AO_DET_ACTG', '1') == '1' else None, ART.cp.uint64)
        g = mod.get_global('AO_DET')
        ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=g)[0] = 1 if s.det else 0

    def det_sort(s):
        raise NotImplementedError('not used by the paper settings')

    def k(s, name, n, args, b=128):
        if not getattr(s, 'det', False):
            return super().k(name, n, args, b)
        i32 = np.int32
        IP = s.IP
        if name == 'rt_p2g':
            s.det_sort()
            s.zf(s.d_st)
            super().k('ao_seg', s.N + 1, (i32(s.N), an_det_P(IP), an_det_P(s.pcd), an_det_P(s.cidx), an_det_P(s.d_st)), 256)
            super().k('ao_p2g_q', s.capQc, (i32(s.N), an_det_P(s.F), an_det_P(s.m), an_det_P(s.vol), an_det_P(s.fx), an_det_P(s.d_st), an_det_P(IP), an_det_P(s.Aacc), an_det_P(s.Cacc), an_det_P(s.badq), an_det_P(s.badc)), 128)
            import os
            if getattr(s, 'gqm', False):
                return
            if os.environ.get('AO_DET_P2GC', '0') == '1':
                super().k('ao_p2g_c', s.capQc, (an_det_P(IP), an_det_P(s.d_st), an_det_P(s.fx), an_det_P(s.m), an_det_P(s.v), an_det_P(s.d_cb)), 64)
                super().k('ao_gath_mv', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.d_cb), an_det_P(s.mn), an_det_P(s.mv)), 256)
            else:
                super().k('ao_p2g_n', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.d_st), an_det_P(s.fx), an_det_P(s.m), an_det_P(s.v), an_det_P(s.mn), an_det_P(s.mv), an_det_P(None), an_det_P(None), i32(1), i32(1), np.float32(0), an_det_P(None)), 128)
            if getattr(s, 'ga', None):
                s.zf(s.d_mscr)
                super().k('ao_p2g_n', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.d_st), an_det_P(s.fx), an_det_P(s.m), an_det_P(s.pa), an_det_P(s.d_mscr), an_det_P(s.ma), an_det_P(None), an_det_P(None), i32(1), i32(1), np.float32(0), an_det_P(None)), 128)
            return
        if name == 'ao_pbc_acc':
            super().k('ao_p2g_n', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.d_st), an_det_P(s.fx), an_det_P(s.m), an_det_P(None), an_det_P(None), an_det_P(None), an_det_P(s.trel), an_det_P(s.tbuf), args[7], args[8], args[9], args[10]), 128)
            return
        if name == 'gq_p2q':
            super().k('gq_p2q_d', s.capQc, (an_det_P(IP), an_det_P(s.d_st), args[1], args[2], args[3], args[4], args[8], args[9]), 128)
            return
        if name == 'gq_q2n':
            super().k('gq_gath', s.capN, (an_det_P(IP), an_det_P(s.cidx), args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]), 256)
            return
        if name == 'ao_act':
            super().k('ao_act_scan', 1024, (an_det_P(IP), an_det_P(s.mn), an_det_P(s.act)), 1024)
            return
        if name == 'ao_act_reset':
            return
        if name == 'rt_mg_acc':
            l = int(args[0])
            L = s.ML[l - 1]
            super().k('ao_mg_acc_det', L['capq'], (args[0], an_det_P(IP), an_det_P(s.LM), an_det_P(s.cidx), args[4], args[5], args[6], args[8]), 128)
            super().k('ao_mg_cell', L['capq'] // 8, (args[0], an_det_P(s.LM), args[6], args[8], args[7], args[9]), 128)
            return
        super().k(name, n, args, b)
        if name in an_det_BS_KERNELS:
            super().k('ao_bsfin', an_det_NSL * 256, (), 256)
        if name in ('rt_elem', 'ao_elem'):
            mode = int(args[8]) if name == 'rt_elem' else int(args[9])
            fptr = args[10] if name == 'rt_elem' else args[10]
            bptr = args[13] if name == 'rt_elem' else args[13]
            if mode & 1:
                super().k('ao_gath_d3', s.capN, (an_det_P(IP), an_det_P(s.cidx), fptr), 256)
            if mode & 2 and mode & 8:
                super().k('ao_gath_f', s.capN, (an_det_P(IP), an_det_P(s.cidx), i32(9), an_det_P(s.d_cb), bptr), 256)
        elif name in ('rt_mg_mv', 'rt_mg_mv2'):
            l = int(args[0])
            super().k('ao_mg_gath', s.ML[l - 1]['capn'], (args[0], an_det_P(s.LM), i32(3), an_det_P(s.d_mgv), args[5], args[9]), 256)
        elif name == 'rt_mg_elem':
            l = int(args[0])
            super().k('ao_mg_gath', s.ML[l - 1]['capn'], (args[0], an_det_P(s.LM), i32(9), an_det_P(s.d_mgb), args[5], args[10]), 256)
        elif name in ('rt_mv', 'rt_mv2', 'rt_mv2s'):
            super().k('ao_gath_f', s.capN, (an_det_P(IP), an_det_P(s.cidx), i32(3), an_det_P(s.d_cv), args[7]), 256)
        elif name == 'rt_gauss':
            super().k('ao_gath_q', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.BW), an_det_P(s.mq), an_det_P(s.mqn)), 256)
        elif name == 'rt_floor_q':
            super().k('ao_gath_q', s.capN, (an_det_P(IP), an_det_P(s.cidx), an_det_P(s.BW), an_det_P(s.d_mw), args[-1]), 256)

def det_module_src(mg2=False):
    src = (ART.MG2.mg2_src(ART.an_rt_SRC) if mg2 else ART.an_rt_SRC).rstrip()
    i = src.rfind('}')
    s = an_det_det_src(src[:i])
    if mg2:
        s = ART.MG2.det_mg2(s)
    return s + an_det_DET_SRC + '\n}\n'

class DetRT(an_det_DetMixin):

    def __init__(s, *args, det=True, **kw):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s, need):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_an_rt_v0 ----------------
RTC = sys.modules[__name__]
an_rt_v0_SRC = cu('an_rt_v0_SRC')

def an_rt_v0_P(t):
    raise NotImplementedError('not used by the paper settings')

class an_rt_v0_RTSolver:

    def __init__(s, x, v, F, m, vol, A, Ai, mu, lam, g, frame_dt, K, integ, newton, cg, ls, floor=None, damp=1.0, bc=None, cap_mul=2.0, dev='cuda', mg_levels=1, mg_nc=8, mg_pow=6, mg_omega=1.0, descent=False, ls_n=0, scheme=None, theta=0.5, mfloor=0.0, K_contact=None):
        raise NotImplementedError('not used by the paper settings')

    def k(s, name, n, args, b=128):
        raise NotImplementedError('not used by the paper settings')

    def profile(s, t):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s, need):
        raise NotImplementedError('not used by the paper settings')

    def mg_build(s):
        raise NotImplementedError('not used by the paper settings')

    def mg_Az(s, l, z, KT, tz_ready=False):
        raise NotImplementedError('not used by the paper settings')

    def mg_lv(s, l):
        raise NotImplementedError('not used by the paper settings')

    def mg_power(s, l):
        raise NotImplementedError('not used by the paper settings')

    def mg_vc(s, l, r, z):
        raise NotImplementedError('not used by the paper settings')

    def zf(s, t, cnt_idx=None, mul=1):
        raise NotImplementedError('not used by the paper settings')

    def stages(s, K):
        raise NotImplementedError('not used by the paper settings')

    def record(s, t_frame, K=None):
        raise NotImplementedError('not used by the paper settings')

    def _bk(s):
        raise NotImplementedError('not used by the paper settings')

    def _rs(s, bk):
        raise NotImplementedError('not used by the paper settings')

    def set_bc(s, t):
        raise NotImplementedError('not used by the paper settings')

    def capture(s):
        raise NotImplementedError('not used by the paper settings')

    def step(s, t):
        raise NotImplementedError('not used by the paper settings')

    def counts(s):
        raise NotImplementedError('not used by the paper settings')

    def x_orig(s):
        raise NotImplementedError('not used by the paper settings')

    def diag(s):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_det_rt ----------------
ART = sys.modules[__name__]
det_rt_P = ART.an_rt_v0_P
det_rt_MAXB = 16384
det_rt_NSL = 64
det_rt_DET_DECL = cu('det_rt_DET_DECL')
det_rt_DET_SRC = cu('det_rt_DET_SRC')

def det_rt__kslice(src, name):
    raise NotImplementedError('not used by the paper settings')

def det_rt__sub_in(src, name, old, new, count=1):
    raise NotImplementedError('not used by the paper settings')

def det_rt_det_src(head):
    raise NotImplementedError('not used by the paper settings')

def det_rt_det_ao(ao_src):
    raise NotImplementedError('not used by the paper settings')
det_rt_BS_KERNELS = {'rt_elem', 'rt_grad', 'rt_grad_c', 'rt_ap', 'rt_ap_c', 'rt_cg1', 'rt_cg1_c', 'rt_lsy', 'rt_lsy4', 'rt_elem_ls', 'ao_warm', 'rt_fixdir'}

class det_rt_DetMixin(ART.an_rt_v0_RTSolver):

    def det_alloc(s):
        raise NotImplementedError('not used by the paper settings')

    def det_bind(s):
        raise NotImplementedError('not used by the paper settings')

    def det_sort(s):
        raise NotImplementedError('not used by the paper settings')

    def _bk(s):
        raise NotImplementedError('not used by the paper settings')

    def _rs(s, bk):
        raise NotImplementedError('not used by the paper settings')

    def k(s, name, n, args, b=128):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_nsd_diag ----------------
ART = sys.modules[__name__]
nsd_diag_P = ART.an_rt_v0_P
UNSET = 2147483647
NSD_SRC = cu('NSD_SRC')

def code(k, ph, sub=0):
    raise NotImplementedError('not used by the paper settings')

class NSD:

    def __init__(s_, S, src):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s_):
        raise NotImplementedError('not used by the paper settings')

    def k(s_, name, n, args, b=256):
        raise NotImplementedError('not used by the paper settings')

    def reset(s_):
        raise NotImplementedError('not used by the paper settings')

    def node(s_, t, mk, c):
        raise NotImplementedError('not used by the paper settings')

    def q(s_, y, c):
        raise NotImplementedError('not used by the paper settings')

    def sc(s_, c):
        raise NotImplementedError('not used by the paper settings')

    def jump(s_, kstep):
        raise NotImplementedError('not used by the paper settings')

    def dump(s_, path, x_prev_orig, extra=None):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_ao_rt ----------------
ART = sys.modules[__name__]
DET = sys.modules[__name__]
C3 = sys.modules[__name__]
ao_rt_P = ART.an_rt_v0_P
NSDM = sys.modules[__name__]
AO_SRC = cu('AO_SRC')
AO_SRC = '#define AF_CTAN_MODE %d\n' % int(os.environ.get('AF_CTAN', '0')) + AO_SRC
AO_SRC = '#define AF_BLOC %d\n' % int(os.environ.get('AF_BLOC', '0')) + AO_SRC
AO_SRC = '#define AF_PTAN %d\n' % int(os.environ.get('AF_PTAN', '0')) + AO_SRC
AO_SRC = '#define AF_AO_RMPOS %d\n' % int(os.environ.get('AF_AO_RMPOS', '0')) + AO_SRC
HB = '    if (blockIdx.x*blockDim.x >= IP[9]) return;\n    int n = blockIdx.x*blockDim.x + threadIdx.x;\n'
HB2 = '    if (blockIdx.x*blockDim.x >= IP[14]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < IP[14]) ? AO_ACT[ii_] : IP[9];\n'
HS = '    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= IP[9]) return;\n'
HS2 = '    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= IP[14]) return;\n    int n = AO_ACT[ii_];\n'

def _pl_funcs():
    PLC = sys.modules[__name__]
    t = PLC._PL
    i = t.index('__device__ __forceinline__ void pl_usv(')
    j = t.index('// GF kirchoff_stress_neoHookeanBoarden 를 F_e 에서 바로')
    return t[i:j]
TAP_OLD = '    if (pm == 0) { o[0] = sk*v[0]; o[1] = sk*v[1]; o[2] = sk*v[2]; return; }'
TAP_NEW = TAP_OLD + '\n    if (pm == 3) { float vn3 = v[0]*n0 + v[1]*n1 + v[2]*n2; o[0] = v[0] - (1.f-sk)*vn3*n0; o[1] = v[1] - (1.f-sk)*vn3*n1; o[2] = v[2] - (1.f-sk)*vn3*n2; return; }   // 미끄럼 바닥: 법선만 √k'
TOL_GRAD_OLD = '    blocksum_add(rz, &SC[0]); blocksum_add(gg, &SC[1]);\n}'
TOL_GRAD_NEW = '    blocksum_add(rz, &SC[0]); blocksum_add(gg, &SC[1]);\n#ifdef AF_ADAPT\n    blocksum_add(ff, &SC[56]);\n#endif\n}'
TOL_G1_OLD = '        for (int a=0;a<3;a++) g[a] = (double)Mi*eta[n*3+a] + (double)T[a*3]*f[n*3] + (double)T[a*3+1]*f[n*3+1] + (double)T[a*3+2]*f[n*3+2];\n'
TOL_G1_NEW = TOL_G1_OLD + '#ifdef AF_ADAPT\n        for (int a=0;a<3;a++) { double tf = g[a] - (double)Mi*eta[n*3+a]; ff += tf*tf; }\n#endif\n'
TOL_CG_OLD = 's3 += (double)rr[a]*zn; s4 += (double)rr[a]*(zn - zo); }\n    }\n    blocksum_add(s3, &SC2[3]); blocksum_add(s4, &SC2[4]);\n}'
TOL_CG_NEW = 's3 += (double)rr[a]*zn; s4 += (double)rr[a]*(zn - zo); s5 += (double)rr[a]*rr[a]; }\n    }\n    blocksum_add(s3, &SC2[3]); blocksum_add(s4, &SC2[4]);\n#ifdef AF_ADAPT\n    blocksum_add(s5, &SC2[57]);\n#endif\n}'
TOL_CTL = cu('TOL_CTL')
AW_SRC = cu('AW_SRC')
TMAT_OLD = 'if (pm == 2) T[a*3+b] = 0.f; else if (pm == 0) T[a*3+b] = sk*I; else {'
TMAT_NEW = 'if (pm == 2) T[a*3+b] = 0.f; else if (pm == 0) T[a*3+b] = sk*I; else if (pm == 3) T[a*3+b] = I - (1.f - sk)*nn[a]*nn[b]; else {'
GQ_OLD = 'if (bad || badq[q]) { for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; atomicAdd(nbad, 1); }'
GQ_NEW = 'if (bad || badq[q]) { float eu[9]; for (int k=0;k<9;k++) eu[k] = aq[18+k]/aq[27]; if (bad || !(aw_det9(eu) > 0.f)) ; else for (int k=0;k<9;k++) out[k] = eu[k]; if (bad) for (int k=0;k<9;k++) out[k] = eu[k]; atomicAdd(nbad, 1); }'
GC_OLD = 'if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27];'
GC_NEW = 'if (bad || badc[c]) { float eu[9]; for (int k=0;k<9;k++) eu[k] = ac[18+k]/ac[27]; if (bad || aw_det9(eu) > 0.f) for (int k=0;k<9;k++) out[k] = eu[k]; }'
AW_DET9 = cu('AW_DET9')
GQ_P2G_OLD = '    if (ng) { atomicOr(&badq[q], 1); atomicOr(&badc[c], 1); }\n    float wx[3][3], dw[3][3];'
GQ_P2G_NEW = '    if (ng) { atomicOr(&badq[q], 1); atomicOr(&badc[c], 1); }\n    return;                                                                 // 절 CM9 --gq_mass: 노드 몫은 gq_q2n\n    float wx[3][3], dw[3][3];'
WB_SRC = cu('WB_SRC')
LJI_SRC = cu('LJI_SRC')
WB_FLN_OLD = '        if (slip) { if (vv > 0.f) { k = 1.f; p = 0; } else p = 3; }        // 미끄럼 바닥 (GF bounding_box): 들어가는 법선 성분만 고착 몫\n'
WB_FLN_NEW = WB_FLN_OLD + '        p = 1;                                                                 // 절 CM21 장벽 바닥: 법선은 장벽, 고착은 접선에만 (절 AN.19)\n'
WB_SE_OLD = '    bool st = (anf && mqn[n] > 0.f && anf[n] > 0.f) || (kb && kb[n] < 1.f);\n'
WB_SE_NEW = '    bool stf_ = anf && mqn[n] > 0.f && anf[n] > 0.f; bool st = kb && kb[n] < 1.f;     // 절 CM21 장벽 바닥: 고착 노드 가속도 0 은 접선 성분만 (새 실행기 rt_ga_end AF_BAR 와 같음)\n'
WB_SE2_OLD = '        ga[n*3+a] = st ? 0.f : a1;\n    }\n}\n'
WB_SE2_NEW = '        ga[n*3+a] = st ? 0.f : a1;\n    }\n    if (stf_ && !st) { float an_ = ga[n*3]*WB_N0 + ga[n*3+1]*WB_N1 + ga[n*3+2]*WB_N2; ga[n*3] = an_*WB_N0; ga[n*3+1] = an_*WB_N1; ga[n*3+2] = an_*WB_N2; }\n}\n'
UF_SRC = cu('UF_SRC')
UF_SIG_OLD = 'float damp) {\n'
UF_SIG_NEW = 'float damp, const int* ufl, int um) {\n    int uf_ = (um == 3) ? 1 : ufl[0];                                        // um 1 = 덜 풀린 걸음 후퇴 오일러, 2 = 덜 풀린 걸음 가속도 기억 0, 3 = 모든 걸음 후퇴 오일러 갱신\n'
UF_U_OLD = '        U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*an + be*a1);\n'
UF_U_NEW = '        if (uf_ && um != 2) { v1 = (y[n*3+a] - u)/dt; U[n*3+a] = y[n*3+a]; } else U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*an + be*a1);   // 후퇴 오일러: U1 = y, v1 = (y − U)/h\n        if (uf_) a1 = 0.f;                                                     // 가속도 기억 0 (후퇴 오일러·(3))\n'

def _src(tol=False, fqrl=False, gqm=False, elx='', dmg='', fpi=False, bar=None, lji=False, uf=False, wf=None, c1=None):
    raise NotImplementedError('not used by the paper settings')

class AORTSolver(DET.det_rt_DetMixin):

    def __init__(s, *args, cdp=1.0, trel=None, pbc_ns=4, dt_gt=2e-05, mthr=0.0, compact=True, warm=True, det=False, hpack=False, bdsh=False, mvc=False, h16=False, theta=None, hspec=False, ga=None, mat=2, prm=None, eta=0.0, pvb=None, ys0=0.0, lj0=0.0, slip=False, tolN=-1.0, tolP=-1.0, vfill=True, aqk=False, lsinv=False, y0z=False, y0u=False, y0l=False, fqrl=False, lsend=False, gqm=False, elx='', dmg='', pf_it=30, fpi=False, wbar=None, lji=False, bcnt=False, uf=None, wf=None, **kw):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s, need):
        raise NotImplementedError('not used by the paper settings')

    def bind_act(s):
        raise NotImplementedError('not used by the paper settings')

    def set_bc(s, t):
        raise NotImplementedError('not used by the paper settings')

    def record(s, t_frame, K=None):
        raise NotImplementedError('not used by the paper settings')

    def check_aq(s, seed=0, eps=(0.01, 0.001, 0.0001)):
        raise NotImplementedError('not used by the paper settings')

    def k(s, name, n, args, b=128):
        raise NotImplementedError('not used by the paper settings')

    def ctl(s, op, it=0):
        raise NotImplementedError('not used by the paper settings')

    def counts(s):
        raise NotImplementedError('not used by the paper settings')

def set_K_ga(s, K, h, sub_dt):
    raise NotImplementedError('not used by the paper settings')

def ao_rt_split_step(s, t, h, sub_dt, depth):
    raise NotImplementedError('not used by the paper settings')

def adapt_step(s, t, h, cell, sub_dt, kmax, c=1.0):
    raise NotImplementedError('not used by the paper settings')

class RTAdapt(DET.det_rt_DetMixin):
    CN = ('rt_grad', 'rt_ap', 'rt_cg1', 'rt_cg2', 'rt_lsup')

    def __init__(s, *args, compact=False, warm=False, det=False, **kw):
        raise NotImplementedError('not used by the paper settings')

    def alloc(s, need):
        raise NotImplementedError('not used by the paper settings')

    def bind_act(s):
        raise NotImplementedError('not used by the paper settings')

    def k(s, name, n, args, b=128):
        raise NotImplementedError('not used by the paper settings')

# ---------------- fem_mat_rt ----------------
ART = sys.modules[__name__]
ADET = sys.modules[__name__]
AORT = sys.modules[__name__]
mat_rt_P = ART.an_rt_P
MT_SRC = cu('MT_SRC')
MT_PHI_SRC = cu('MT_PHI_SRC')

def mat_module_src(mg2):
    src = ADET.det_module_src(mg2).rstrip()
    i = src.rfind('}')
    body = src[:i]
    decl = '\n__device__ int* AO_ACT;\n__device__ float* AQ_DBG;\n__device__ int AQ_DBGQ;'
    body = body.replace('extern "C" {', 'extern "C" {' + decl, 1)
    return '#include <cuda_fp16.h>\n' + body + AORT._pl_funcs() + ADET.an_det_det_ao(AORT.AO_SRC) + MT_SRC + '\n}\n'

class MatRT(ADET.an_det_DetMixin):

    def __init__(s, *args, mat=2, cdp=1.0, prm=None, visc=0.0, trel=None, pvb=None, pbc_ns=4, dt_gt=2e-05, mthr=0.0, ys0=0.0, lj0=0.0, vfill=True, ar=True, det=False, pbc_phi=False, **kw):
        s.pbc_phi = bool(pbc_phi)
        s.det = bool(det)
        s.ao_ready = False
        (s._mat, s.cdp, s.prm, s.visc) = (int(mat), float(cdp), prm, float(visc))
        s.aq = s._mat != 2
        s.has_pbc = trel is not None
        (s.pbc_ns, s.dt_gt, s.mthr) = (int(pbc_ns), float(dt_gt), float(mthr))
        (s.vfill, s.ar) = (bool(vfill), bool(ar))
        s._pvb_on = pvb is not None
        super().__init__(*args, **kw)
        assert s.ga, '재질 판은 --ga (HHT) 만'
        assert not s.bc or getattr(s, '_rbc_allow', False), '재질 판은 고정 구속 상자 없는 씬만 (exe/rotbc_wrap.py 의 회전 구동 노드는 예외: 같은 fixd·vbc 를 씀)'
        s.matk = s._mat
        s.ga_first = False
        s.pa.copy_(torch.tensor(s.g, device=s.dev, dtype=torch.float32).expand(s.N, 3))
        s.trel = trel[s.perm].float().contiguous() if s.has_pbc else None
        s.pvb = None if pvb is None else pvb[s.perm].float().contiguous()
        if s.pvb is not None:
            s.pvc = torch.zeros(s.N, 3, device=s.dev, dtype=torch.float32)
            s.pmk = torch.zeros(s.N, 3, device=s.dev, dtype=torch.float32)
        if s.aq:
            s.ysp = torch.full((s.N,), float(ys0), device=s.dev, dtype=torch.float32)
            s.ljp = torch.full((s.N,), float(lj0), device=s.dev, dtype=torch.float32)
            s.alp = torch.ones(s.N, device=s.dev, dtype=torch.float32)
        s.extra_p = (('ysp', 'ljp', 'alp') if s.aq else ()) + (('pvb', 'pvc', 'pmk') if s.pvb is not None else ())
        s.tbuf = torch.zeros(1, device=s.dev, dtype=torch.float32)
        src = mat_module_src(getattr(s, 'mg2', False))
        if s.pbc_phi:
            assert s.has_pbc, '--pbc_phi 는 순차 방출 구속 씬만'
            i_ = src.rstrip().rfind('}')
            src = src[:i_] + MT_PHI_SRC + '\n}\n'
        s.mod = ART.cp.RawModule(code=ART.build_src(src, s.adapt, s.goct, s.floor, fk=getattr(s, 'fk', False), hspec=getattr(s, 'hspec', False), ncomp=getattr(s, 'ncomp', False), gqm=getattr(s, 'gqm', False), fpi=getattr(s, 'fpi', False), cj=getattr(s, 'cjj', False), cue=bool(getattr(s, 'cue', 0)), uel=getattr(s, 'uel', 0), ufw=getattr(s, 'ufw', False), ls0=getattr(s, 'ls0', False), c2=getattr(s, 'c2', None)), options=('--std=c++14',))
        s._src0 = src
        s.fn = {}
        s.ao_ready = True
        s.det_bind()
        s.oct_bind()
        s.mat_bind()
        s.graph = None
        s.graphs = {}

    def alloc(s, need):
        super().alloc(need)
        s.det_alloc()
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        I = lambda n_: torch.zeros(n_, device=s.dev, dtype=torch.int32)
        capQ = s.capQc * 8
        s.Ust = Z(max(s.K, getattr(s, 'kcap', 0)), s.capN, 3)
        s.H30 = Z(capQ * 30) if s.aq else None
        s.Hs = s.H30 if s.aq and s.hspec else None
        s.cnt = I(4)
        s.qb = I(4)
        s.act = I(s.capN)
        s.Sa = Z(s.capQc * 32)
        s.Sc = Z(s.capQc * 4)
        s.Sq = Z(s.capQc * 32)
        s.gaM = Z(s.capN)
        if s.has_pbc:
            s.pacc = Z(s.capN * max(s.K, getattr(s, 'kcap', 0)) * s.pbc_ns)
            s.kb = Z(s.capN)
        if s._pvb_on:
            s.vm = Z(s.capN, 3)
            s.cm = Z(s.capN, 3)
        if getattr(s, 'ao_ready', False):
            s.det_bind()
            s.mat_bind()
        ART.WW.wolfwhy_diag_alloc(s)

    def mat_bind(s):
        for (nm, t) in (('AO_ACT', s.act), ('AQ_DBG', None)):
            g = s.mod.get_global(nm)
            ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(t.data_ptr()) if t is not None else 0

    def det_sort(s):
        raise NotImplementedError('not used by the paper settings')

    def _bkn(s):
        return ('x', 'v', 'F', 'm', 'vol', 'perm', 'pa') + (('trel',) if getattr(s, 'trel', None) is not None else ()) + tuple(getattr(s, 'extra_p', ()))

    def _bk(s):
        return tuple((getattr(s, nm).clone() for nm in s._bkn()))

    def _rs(s, bk):
        for (nm, t) in zip(s._bkn(), bk):
            getattr(s, nm).copy_(t)

    def set_bc(s, t):
        super().set_bc(t)
        s.tbuf.fill_(float(t))

    def k(s, name, n, args, b=128):
        super().k(name, n, args, b)
        if getattr(s, 'det', False) and name in ('aq_elem', 'ao_elem_s'):
            ART.an_rt_RTSolver.k(s, 'ao_gath_d3', s.capN, (mat_rt_P(s.IP), mat_rt_P(s.cidx), args[10]), 256)
            ART.an_rt_RTSolver.k(s, 'ao_gath_f', s.capN, (mat_rt_P(s.IP), mat_rt_P(s.cidx), np.int32(9), mat_rt_P(s.d_cb), args[13]), 256)

    def mat_pre_p2g(s):
        (N, i32) = (s.N, np.int32)
        s.zf(s.cnt)
        if s.aq:
            s.zf(s.qb)
        if s.has_pbc:
            if s.pvb is not None:
                s.k('aq_vset', N, (i32(N), mat_rt_P(s.trel), mat_rt_P(s.tbuf), mat_rt_P(s.pvb), mat_rt_P(s.v), mat_rt_P(s.pvc), mat_rt_P(s.pmk)))
            else:
                s.k('ao_vzero', N, (i32(N), mat_rt_P(s.IP), mat_rt_P(s.trel), mat_rt_P(s.tbuf), mat_rt_P(s.v)))
            s.k('ao_azero', N, (i32(N), mat_rt_P(s.trel), mat_rt_P(s.tbuf), mat_rt_P(s.pa)))

    def mat_post_vn(s):
        (N, IP, i32, f) = (s.N, s.IP, np.int32, np.float32)
        (capN, capQ) = (s.capN, s.capQc * 8)
        if s.aq:
            s.zf(s.Sa)
            s.zf(s.Sc)
            if s.det:
                s.k('aq_p2g_sd', s.capQc, (mat_rt_P(IP), mat_rt_P(s.d_st), mat_rt_P(s.fx), mat_rt_P(s.vol), mat_rt_P(s.ysp), mat_rt_P(s.ljp), mat_rt_P(s.alp), mat_rt_P(s.Sa), mat_rt_P(s.Sc)), b=128)
            else:
                s.k('aq_p2g_s', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.fx), mat_rt_P(s.pcd), mat_rt_P(s.cidx), mat_rt_P(s.vol), mat_rt_P(s.ysp), mat_rt_P(s.ljp), mat_rt_P(s.alp), mat_rt_P(s.Sa), mat_rt_P(s.Sc)))
            s.k('aq_gauss_s', capQ, (mat_rt_P(IP), mat_rt_P(s.Sa), mat_rt_P(s.Sc), mat_rt_P(s.Sq)))
        if s.pvb is not None:
            s.zf(s.vm, 9, 3)
            s.zf(s.cm, 9, 3)
            if s.det:
                s.zf(s.gaM, 9, 1)
                s.k('ao_p2g_n', capN, (mat_rt_P(IP), mat_rt_P(s.cidx), mat_rt_P(s.d_st), mat_rt_P(s.fx), mat_rt_P(s.m), mat_rt_P(s.pvc), mat_rt_P(s.gaM), mat_rt_P(s.vm), mat_rt_P(None), mat_rt_P(None), i32(1), i32(1), f(0.0), mat_rt_P(None)), b=128)
                s.zf(s.gaM, 9, 1)
                s.k('ao_p2g_n', capN, (mat_rt_P(IP), mat_rt_P(s.cidx), mat_rt_P(s.d_st), mat_rt_P(s.fx), mat_rt_P(s.m), mat_rt_P(s.pmk), mat_rt_P(s.gaM), mat_rt_P(s.cm), mat_rt_P(None), mat_rt_P(None), i32(1), i32(1), f(0.0), mat_rt_P(None)), b=128)
            else:
                s.k('aq_p2g_v', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.Np), mat_rt_P(s.fx), mat_rt_P(s.m), mat_rt_P(s.pvc), mat_rt_P(s.pmk), mat_rt_P(s.vm), mat_rt_P(s.cm)))
            s.k('aq_vbar', capN, (mat_rt_P(IP), mat_rt_P(s.cm), mat_rt_P(s.vm)), b=256)
        if s.has_pbc:
            s.zf(s.pacc, 9, s.K * s.pbc_ns)
            s.k('ao_pbc_acc', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.Np), mat_rt_P(s.fx), mat_rt_P(s.m), mat_rt_P(s.trel), mat_rt_P(s.tbuf), i32(s.K), i32(s.pbc_ns), f(s.dt), mat_rt_P(s.pacc)))

    def mat_post_anode(s):
        if s.aq and s.vfill:
            s.k('aq_vfill', s.capN, (mat_rt_P(s.IP), mat_rt_P(s.mn), mat_rt_P(s.vk), mat_rt_P(s.ak)), b=256)

    def mat_floor_n(s, FA, kstep, capN):
        (i32, f) = (np.int32, np.float32)
        if s.has_pbc:
            s.k('mt_pbc_kphi' if s.pbc_phi else 'ao_pbc_k', capN, (mat_rt_P(s.IP), mat_rt_P(s.mn), mat_rt_P(s.pacc), i32(s.K), i32(s.pbc_ns), i32(kstep), f(s.dt / s.pbc_ns / s.dt_gt), mat_rt_P(s.kb)), b=256)
        s.k('mt_floor_n', capN, tuple(FA) + (mat_rt_P(s.kb if s.has_pbc else None), mat_rt_P(s.mn), f(s.mthr), mat_rt_P(s.vm if s.pvb is not None else None)), b=256)

    def mat_elem(s, kstep, it, nst, capQ):
        (i32, f) = (np.int32, np.float32)
        IP = s.IP
        last = kstep == nst - 1 and it == s.nt - 1
        if s.aq:
            al = s.ga_c['gm'] * s.dt / ((1 - s.ga_c['af']) * s.ga_c['be'] * s.dt * s.dt)
            s.k('aq_elem', capQ, (mat_rt_P(IP), mat_rt_P(s.cb), mat_rt_P(s.BG), mat_rt_P(s.Fq), mat_rt_P(s.Vq), mat_rt_P(s.y), i32(s._mat), mat_rt_P(s.prm), mat_rt_P(s.Sq), f(s.dt), mat_rt_P(s.f), mat_rt_P(s.H30), i32(capQ), mat_rt_P(s.B), mat_rt_P(s.cnt if last else None), mat_rt_P(s.z0), f(al), f(s.visc), mat_rt_P(s.qb)))
            if not s.hspec:
                s.k('mt_h81', capQ, (mat_rt_P(IP), mat_rt_P(s.Fq), mat_rt_P(s.Vq), mat_rt_P(s.H30), mat_rt_P(s.H), i32(capQ)))
        else:
            s.k('ao_elem_s' if s.hspec else 'ao_elem', capQ, (mat_rt_P(IP), mat_rt_P(s.cb), mat_rt_P(s.BG), mat_rt_P(s.Fq), mat_rt_P(s.Vq), mat_rt_P(s.y), f(s.mu), f(s.lam), f(s.cdp), i32(1 | 2 | 8), mat_rt_P(s.f), mat_rt_P(s.H), i32(capQ), mat_rt_P(s.B), mat_rt_P(s.cnt if last else None)))

    def mat_step_end(s, kstep, nst, capN, capQ):
        (i32, f) = (np.int32, np.float32)
        IP = s.IP
        if s.has_pbc and s.ar:
            s.k('mt_akb', capN, (mat_rt_P(IP), mat_rt_P(s.kb), mat_rt_P(s.ak)), b=256)
        s.dcopy(s.Ust[kstep], s.U)
        if kstep < nst - 1 or getattr(s, 'fpi', False):
            if s.aq:
                s.k('aq_commit', capQ, (mat_rt_P(IP), mat_rt_P(s.cb), mat_rt_P(s.BG), mat_rt_P(s.Fq), mat_rt_P(s.U), i32(s._mat), mat_rt_P(s.prm), mat_rt_P(s.Sq), f(s.dt)))
            else:
                s.k('ao_commit', capQ, (mat_rt_P(IP), mat_rt_P(s.cb), mat_rt_P(s.BG), mat_rt_P(s.Fq), mat_rt_P(s.U), f(s.cdp)))

    def mat_g2p(s):
        (N, IP, i32, f) = (s.N, s.IP, np.int32, np.float32)
        s.k('ao_g2p_a', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.Np), mat_rt_P(s.fx), mat_rt_P(s.ak), mat_rt_P(s.pa)))
        if s.aq:
            s.k('aq_g2p', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.Ai), mat_rt_P(s.Np), mat_rt_P(s.fx), mat_rt_P(s.Ust), i32(s.K), i32(s.capN), mat_rt_P(s.vk), i32(s._mat), mat_rt_P(s.prm), f(s.dt), mat_rt_P(s.x), mat_rt_P(s.v), mat_rt_P(s.F), mat_rt_P(s.ysp), mat_rt_P(s.ljp), mat_rt_P(s.alp), mat_rt_P(s.cnt)))
        else:
            s.k('ao_g2p', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.Ai), mat_rt_P(s.Np), mat_rt_P(s.fx), mat_rt_P(s.Ust), i32(s.K), i32(s.capN), mat_rt_P(s.vk), f(s.cdp), mat_rt_P(s.x), mat_rt_P(s.v), mat_rt_P(s.F), mat_rt_P(s.cnt)))
        if getattr(s, 'fpi', False):
            capQ = s.capQc * 8
            s.k('fpi_q', capQ, (mat_rt_P(IP), mat_rt_P(s.cb), mat_rt_P(s.BG), mat_rt_P(s.Fq), mat_rt_P(s.U), mat_rt_P(s.Fe)))
            s.k('fpi_p', N, (i32(N), mat_rt_P(IP), mat_rt_P(s.fx), mat_rt_P(s.pcd), mat_rt_P(s.cidx), mat_rt_P(s.Fe), i32(0), i32(0), mat_rt_P(s.F), mat_rt_P(s.Sq if s.aq else None), mat_rt_P(s.ysp if s.aq else None), mat_rt_P(s.ljp if s.aq else None), mat_rt_P(s.alp if s.aq else None)))

    def counts(s):
        o = super().counts()
        c = s.cnt.cpu().tolist()
        o.update(q_tail=c[0], q_apex=c[1], p_rm=c[2])
        if s.mthr > 0 and o['nn'] > 0:
            mn = s.mn[:o['nn']]
            o.update(nlow=int(((mn > 0) & (mn < s.mthr)).sum()), nact=int((mn > 0).sum()))
        return o

# ---------------- fem_wmrx ----------------
HEAD = '__device__ int AF_WRXON = 0;\n__device__ int AF_WRXB = 0;\n__device__ float AF_SKC = 0.f;\n'
A1 = '    float s = sqrtf(k);\n    float d[3];\n'
A1N = '    float s = sqrtf(k);\n    if (AF_WRXON && (AF_WRXB & 1) && s < AF_SKC) s = 0.f;                                  // 절 CM38 후보 1: 고착 상한\n    float d[3];\n'
A2 = '    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];\n'
A2N = A2 + '    if (AF_WRXON && (AF_WRXB & 2) && floor_) {                                              // 절 CM38 후보 2: 속도 수준 접촉 (들어가는 법선 이동을 띠 몫만큼 지움)\n        float r_ = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;\n        float dn_ = (h[0]-U[n*3])*n0 + (h[1]-U[n*3+1])*n1 + (h[2]-U[n*3+2])*n2;\n        if (r_ > 0.f && dn_ < 0.f) { h[0] -= r_*dn_*n0; h[1] -= r_*dn_*n1; h[2] -= r_*dn_*n2; }\n    }\n'
KNAMES = ('ao_floor_n', 'rt_floor_n', 'mt_floor_n')
A1G = '    float s = sqrtf(k);\n'
A1GN = A1G + '    if (AF_WRXON && (AF_WRXB & 1) && s < AF_SKC) s = 0.f;                                  // 절 CM38 후보 1: 고착 상한\n'

def wmrx_apply(src):
    (out, pos, head) = ([], 0, False)
    for nm in KNAMES:
        key = '__global__ void ' + nm + '('
        if key not in src:
            continue
        i = src.index(key)
        j = src.index('__global__', i + 10)
        body = src[i:j]
        assert body.count(A1G) == 1 and body.count(A2) == 1, f'--wrx: {nm} 글자가 바뀜'
        src = src[:i] + body.replace(A1G, A1GN, 1).replace(A2, A2N, 1) + src[j:]
        head = True
    assert head, '--wrx: 바닥 노드 커널 없음'
    i0 = min((src.index('__global__ void ' + nm + '(') for nm in KNAMES if '__global__ void ' + nm + '(' in src))
    i0 = src.rfind('\n', 0, i0) + 1
    return src[:i0] + HEAD + src[i0:]

def wmrx_install(new=False):
    if new:
        ART = sys.modules[__name__]
        if getattr(ART, 'an_rt__wrx_on', False):
            return
        ART.an_rt__wrx_on = True
        o_b = ART.build_src

        def build_src(*a, **kw):
            return wmrx_apply(o_b(*a, **kw))
        ART.build_src = build_src
        return
    AORT = sys.modules[__name__]
    if getattr(AORT, 'ao_rt__wrx_on', False):
        return
    AORT.ao_rt__wrx_on = True
    o_src = AORT._src

    def _src(*a, **kw):
        return wmrx_apply(o_src(*a, **kw))
    AORT._src = _src

def _set(s, name, val, dt):
    ART = sys.modules[__name__]
    g = s.mod.get_global(name)
    ART.cp.ndarray((1,), dtype=dt, memptr=g)[0] = val

def setup(s, bits, skc):
    ART = sys.modules[__name__]
    s.wrx = dict(bits=int(bits), skc=float(skc))
    _set(s, 'AF_WRXB', int(bits), ART.cp.int32)
    _set(s, 'AF_SKC', float(skc), ART.cp.float32)

def on(s, v):
    ART = sys.modules[__name__]
    _set(s, 'AF_WRXON', int(v), ART.cp.int32)
    s._wrxv = int(v)

def split_col(s, t, h, sub_dt, depth, fn=None):
    if fn is None:
        AORT = sys.modules[__name__]
        fn = lambda d: AORT.ao_rt_split_step(s, t, h, sub_dt, d)
    bk = s._bk()
    on(s, 0)
    (r0, K0, tr0) = fn(0)
    tries = [(a, b, c, 0) for (a, b, c) in tr0]
    if tr0[-1][1] == 0 or not math.isfinite(tr0[-1][2]):
        return (r0, K0, tries)
    tot0 = sum((q[2] for q in tr0))
    s._rs(bk)
    on(s, 1)
    (r1, K1, tr1) = fn(depth)
    on(s, 0)
    tries += [(a, b, c, 1) for (a, b, c) in tr1]
    s.last_ms = tot0 + float(s.last_ms)
    return (r0 or r1, K1, tries)

# ---------------- fem_vcap ----------------
vcap_SRC = cu('vcap_SRC')
_M = {}

def vcap__P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def vmax_energy(X0, V0, g, FL, T, extra_v=None):
    g = np.asarray(g, dtype=np.float64)
    gn = float(np.linalg.norm(g))
    v2 = float((V0.double() ** 2).sum(1).max()) if V0 is not None and V0.numel() else 0.0
    if extra_v is not None:
        v2 = max(v2, float((extra_v.double() ** 2).sum(1).max()))
    if FL is None or gn == 0.0:
        return (float(np.sqrt(v2) + gn * T), dict(kind='time', v0=float(np.sqrt(v2)), T=T))
    up = torch.tensor(-g / gn, device=X0.device, dtype=torch.float64)
    zt = float((X0.double() @ up).max())
    nr = np.asarray(FL['NR'], dtype=np.float64)
    pt = np.asarray(FL['PT'], dtype=np.float64)
    zf = float(pt @ (-g / gn))
    dz = max(zt - zf, 0.0)
    return (float(np.sqrt(v2 + 2.0 * gn * dz)), dict(kind='energy', v0=float(np.sqrt(v2)), dz=dz, g=gn, nr=nr.tolist()))

def vcap_apply(s, vmax, dt):
    raise NotImplementedError('not used by the paper settings')

def apply_mmed(s, mm, cs=0.0):
    RTC = sys.modules[__name__]
    if 'm' not in _M:
        _M['m'] = cp.RawModule(code=vcap_SRC)
        _M['f'] = _M['m'].get_function('vc_cap')
    if 'g' not in _M:
        _M['g'] = _M['m'].get_function('vc_mmed')
    g = ((int(s.capN) + 255) // 256,)
    with RTC._Stream():
        _M['g'](g, (256,), (vcap__P(s.IP), vcap__P(s.mn), np.float32(mm), vcap__P(s.B), np.float32(cs), vcap__P(s.meff)))

def setf(s, name, v):
    raise NotImplementedError('not used by the paper settings')
