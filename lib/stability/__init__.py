# ---------------- fem_ghost1 ----------------
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of stability.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stability.cu'), encoding='utf-8', newline='').read()
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

import math
import numpy as np
import torch
ghost1_GAMMA_A = 0.0001
ghost1_ST = dict(on=False, said=False)
ghost1_ELEM = ('rt_elem', 'aq_elem', 'ao_elem', 'ao_elem_s')
ghost1_MV = ('rt_mv', 'rt_mv2', 'rt_mv2s', 'ao_mv_q', 'ao_mv_h', 'ao_mv_c', 'ao_mv_s')
ghost1_GP_SRC = cu('ghost1_GP_SRC')

def ghost1__append(src):
    raise NotImplementedError('not used by the paper settings')

def ghost1__k_of(s):
    raise NotImplementedError('not used by the paper settings')

def ghost1__alloc(s):
    raise NotImplementedError('not used by the paper settings')

def ghost1_install():
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_ghost2 ----------------
import json
import os
ghost2_GP_SRC = ghost1_GP_SRC
ghost2__append = ghost1__append
ghost2_GAMMA_A = float(os.environ.get('AF_G2_GAMMA', '1e-4'))
ghost2_ST = dict(on=False, said=False, nk={})
ghost2_ELEM = ('rt_elem', 'aq_elem', 'ao_elem', 'ao_elem_s')
ghost2_MV = ('rt_mv', 'rt_mv2', 'rt_mv2s', 'ao_mv_q', 'ao_mv_h', 'ao_mv_c', 'ao_mv_s')
ghost2_REC = ghost2_ELEM + ghost2_MV + ('rt_elem_lsN', 'en_phi_q')
MM9 = np.array([[6, 13, 1], [13, 54, 13], [1, 13, 6]], dtype=np.float64) / 120.0
CL = (-1.0, 3.0, -3.0, 1.0)

def ghost2_P(t):
    raise NotImplementedError('not used by the paper settings')

def ghost2__k_of(s):
    raise NotImplementedError('not used by the paper settings')

def ghost2__alloc(s):
    raise NotImplementedError('not used by the paper settings')

def ghost2__cnt(name):
    raise NotImplementedError('not used by the paper settings')

def ghost2_install():
    raise NotImplementedError('not used by the paper settings')

def ghost2__rel(a, b):
    raise NotImplementedError('not used by the paper settings')

def ghost2__T(v, sk, pm, nrm):
    raise NotImplementedError('not used by the paper settings')

def ghost2_check(s, fi, out):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_ghost6 ----------------
import sys
ghost6_GP_SRC = ghost1_GP_SRC
G2 = sys.modules[__name__]
_ghost6_env_done = [False]

def ghost6__env_init(first=True):
    global GAMMA, MASS, FIXX, WMODE, WMAX, CONS, THIN
    if first and _ghost6_env_done[0]:
        return
    GAMMA = float(os.environ.get('AF_G4_GAMMA', '1e-1'))
    MASS = int(os.environ.get('AF_G4_MASS', '1'))
    FIXX = int(os.environ.get('AF_G4_FIXX', '0'))
    WMODE = int(os.environ.get('AF_G5_W', '0'))
    WMAX = float(os.environ.get('AF_G5_WMAX', '3'))
    CONS = int(os.environ.get('AF_G6_CONS', '0'))
    THIN = int(os.environ.get('AF_G4_THIN', '1'))
    if first:
        _ghost6_env_done[0] = True
ghost6__env_init(first=False)
ghost6_ST = dict(on=False, said=False, nk={}, nmass=0, nstep=0)
ghost6_ELEM = ('rt_elem', 'aq_elem', 'ao_elem', 'ao_elem_s')
ghost6_MV = ('rt_mv', 'rt_mv2', 'rt_mv2s', 'ao_mv_q', 'ao_mv_h', 'ao_mv_c', 'ao_mv_s')
STEP0 = {'rt_ga_step0': 9, 'rt_step0g': 3}
ghost6_REC = ghost6_ELEM + ghost6_MV + ('rt_elem_lsN', 'en_phi_n')
GQ_SRC = cu('GQ_SRC')

def _phi_src():
    s = GQ_SRC
    i = s.index('__global__ void __launch_bounds__(128) gq_force(')
    j = s.index('// ---- 선탐색 후보')
    body = s[i:j]
    body = body.replace('gq_force(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, double* f, float* B, double* E) {', 'gq_phi(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, const int* MO, double* PH) {\n    if (!MO[0]) return;\n    double* f = 0; float* B = 0; double* E = PH + 1;', 1)
    k0 = s.index('// ---- 옛 실행기 --en_mo Φ')
    return s[:k0] + '// ---- 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ) (en_phi_n 과 같은 MO 문; gq_force 본문 복제, f·B 없음)\n' + body
ghost6_SRC = None

def ghost6__append(src):
    global ghost6_SRC
    if 'gq_force(' in src:
        return src
    assert 'blocksum_add(' in src and 'Tapply(' in src, 'ghost3: 바탕 소스에 blocksum_add·Tapply 없음'
    if ghost6_SRC is None:
        ghost6_SRC = _phi_src()
    i = src.rstrip().rfind('}')
    add = (ghost6_GP_SRC if 'gp_force(' not in src else '') + ghost6_SRC
    return src[:i] + add + '\n}\n'

def ghost6_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def FWP(s):
    return ghost6_P(s.g5w) if WMODE and getattr(s, 'g5w', None) is not None else ghost6_P(None)

def ghost6__k_of(s):
    if getattr(s, '_gpk', None) is None:
        A = s.A.double().reshape(3, 3).cpu()
        h = abs(float(torch.linalg.det(A))) ** (1.0 / 3.0)
        kap = 2.0 * float(s.mu) + float(s.lam)
        (m_, v_) = (float(s.m.double().sum()), float(s.vol.double().sum()))
        rho = m_ / v_ if v_ > 0 else 0.0
        s._gpk = GAMMA * kap * h
        s._g3kM = GAMMA * rho * h ** 3 if MASS else 0.0
        (s._gph, s._gpkap, s._g3rho) = (h, kap, rho)
        print(f'[절 CM53 ghost4] γ̄ {GAMMA:g} (강성 = 질량), 질량 {MASS} · 면 한정 방식 {THIN}; μ {float(s.mu):.6g}, λ {float(s.lam):.6g}, 2μ+λ {kap:.6g}, h {h:.6g} m, ρ̄ {rho:.6g} kg/m³ -> k = γ̄ (2μ+λ) h = {s._gpk:.6g}, k_M = γ̄ ρ̄ h³ = {s._g3kM:.6g} kg', flush=True)
    return s._gpk

def ghost6__alloc(s):
    cap = int(s.capQc)
    if getattr(s, 'g3cut', None) is None or s.g3cut.numel() < cap:
        s.g3cut = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.g3thk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.g3fm = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
    if getattr(s, 'g5w', None) is None or s.g5w.numel() < cap * 3:
        s.g5phi = torch.zeros(cap, device=s.dev, dtype=torch.float32)
        s.g5rho = torch.zeros(cap, device=s.dev, dtype=torch.float32)
        s.g5w = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
    if CONS and (getattr(s, 'g6u', None) is None or s.g6u.numel() < int(s.capN) * 3):
        capN = int(s.capN)
        s.g6num = torch.zeros(capN * 3, device=s.dev, dtype=torch.float32)
        s.g6den = torch.zeros(capN, device=s.dev, dtype=torch.float32)
        s.g6u = torch.zeros(capN * 3, device=s.dev, dtype=torch.float32)
        s.g6bad = torch.zeros(1, device=s.dev, dtype=torch.int32)
    if getattr(s, 'gpcnt', None) is None:
        s.gpcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.gpGQ = torch.zeros(8, device=s.dev, dtype=torch.float64)

def ghost6__cnt(name):
    ghost6_ST['nk'][name] = ghost6_ST['nk'].get(name, 0) + 1

def cm_of(s):
    if not getattr(s, '_g3in', False) or not MASS:
        return 0.0
    return float(s._g3kM) * float(getattr(s, '_g3it2', 0.0))

def ghost6_install():
    import hexfem as ART
    import hexfem as AORT
    import hexfem as V0
    ghost6_ST['on'] = True
    (_obs, _oas) = (ART.build_src, AORT._src)
    ART.build_src = lambda *a, **kw: ghost6__append(_obs(*a, **kw))
    AORT._src = lambda *a, **kw: ghost6__append(_oas(*a, **kw))
    f32 = np.float32

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name in ghost6_REC and (not getattr(s, '_g3chk', False)):
                if getattr(s, '_g3args', None) is None:
                    s._g3args = {}
                s._g3args[name] = tuple(args)
            r = _ok(s, name, n, args, b)
            if getattr(s, '_gpin', False) or name.startswith('gp_') or name.startswith('gq_'):
                return r
            s._gpin = True
            try:
                _hook(s, _ok, name, args)
            finally:
                s._gpin = False
            return r
        cls.k = k

    def _go(s, _ok, nm, n, args, b):
        ghost6__cnt(nm)
        _ok(s, nm, n, args, b)

    def _hook(s, _ok, name, args):
        if name == 'rt_gauss':
            ghost6__k_of(s)
            if getattr(s, 'det', False):
                assert s._gpk == 0.0 and s._g3kM == 0.0, 'ghost3: 결정적 판 (det) 은 γ̄ = 0 대조만 (원자 합 순서)'
            assert getattr(s, 'pkmod', None) is None, 'ghost3: 지속형 판은 안 고침'
            ghost6__alloc(s)
            s._g3in = False
            cq = int(s.capQc)
            _go(s, _ok, 'gq_zero', 1, (ghost6_P(s.gpcnt),), 1)
            _go(s, _ok, 'gq_cut', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3cut), ghost6_P(s.gpcnt)), 128)
            _go(s, _ok, 'gq_thk', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3cut), ghost6_P(s.g3thk), np.int32(THIN), ghost6_P(s.gpcnt)), 128)
            if THIN == 2:
                _go(s, _ok, 'gq_thk2', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3thk), ghost6_P(s.gpcnt)), 128)
            if THIN == 4:
                _go(s, _ok, 'gq_dil', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3thk)), 128)
            _go(s, _ok, 'gq_face', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3cut), ghost6_P(s.g3thk), ghost6_P(s.g3fm), np.int32(THIN), ghost6_P(s.gpcnt)), 128)
            (s._g5C, s._g5A) = (args[2], args[6])
            _go(s, _ok, 'gq5_cell', cq, (ghost6_P(s.IP), args[2], args[6], ghost6_P(s.g5phi), ghost6_P(s.g5rho)), 128)
            if CONS:
                assert getattr(s, 'g6X0', None) is not None, 'ghost6: X0 (첫 프레임 입자 위치) 없음 -- 감싸개가 첫 step 앞에 둠'
                capN = int(s.capN)
                _go(s, _ok, 'g6_zero', capN, (ghost6_P(s.IP), ghost6_P(s.g6num), ghost6_P(s.g6den), ghost6_P(s.g6bad)), 256)
                _go(s, _ok, 'g6_p2g', int(s.N), (np.int32(int(s.N)), ghost6_P(s.IP), ghost6_P(s.A), ghost6_P(s.Np), ghost6_P(s.fx), ghost6_P(s.x), ghost6_P(s.g6X0), ghost6_P(s.F), ghost6_P(s.m), ghost6_P(s.g6num), ghost6_P(s.g6den), ghost6_P(s.g6bad)), 128)
                _go(s, _ok, 'g6_fin', capN, (ghost6_P(s.IP), ghost6_P(s.g6num), ghost6_P(s.g6den), ghost6_P(s.g6u)), 256)
            if WMODE:
                _go(s, _ok, 'gq5_w', cq, (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.cidx), ghost6_P(s.g3fm), ghost6_P(s.g5phi), np.int32(WMODE), np.float32(WMAX), ghost6_P(s.g5w)), 128)
            if not ghost6_ST['said']:
                ghost6_ST['said'] = True
                print(f'[절 CM55 ghost5] 면 가중 방식 {WMODE} (w_max {WMAX:g}): 칸 점유 φ·가중 (rt_gauss 뒤, 면 표지 다음)', flush=True)
                print('[절 CM53 ghost4] 켬: 경계 칸·두꺼운 칸·칸별 면 표지 (rt_gauss 뒤), 요소·곱·에너지 커널 뒤 칸별 G 몫 (강성 + 걸음 안 질량)', flush=True)
            return
        if name == 'rt_bc_box':
            if FIXX and getattr(s, 'g3fm', None) is not None:
                _go(s, _ok, 'gq_fix', int(s.capQc), (ghost6_P(s.IP), ghost6_P(s.cb), args[4], ghost6_P(s.g3fm), ghost6_P(s.gpcnt)), 128)
            return
        if name in STEP0:
            tau = float(args[STEP0[name]])
            s._g3it2 = 1.0 / (tau * tau) if tau > 0 else 0.0
            s._g3in = True
            ghost6_ST['nstep'] += 1
            return
        if name == 'ao_floor_n':
            s._g3it2 = 1.0 / float(s.tau) ** 2
            s._g3in = True
            ghost6_ST['nstep'] += 1
            return
        if getattr(s, 'gpcnt', None) is None:
            return
        k = float(s._gpk)
        cm = cm_of(s)
        if cm != 0.0:
            ghost6_ST['nmass'] += 1
        nq = int(s.capQc)
        G = (ghost6_P(s.IP), ghost6_P(s.cb), ghost6_P(s.g3fm), FWP(s))
        YH = ghost6_P(s.yh) if cm != 0.0 else ghost6_P(None)
        Y0 = ghost6_P(s.g6u) if CONS else ghost6_P(None)
        DT = bool(getattr(s, 'det', False))
        if name in ghost6_ELEM:
            if DT:
                args = list(args)
                if name == 'rt_elem':
                    args[8] = np.int32(int(args[8]) & ~4)
                args = tuple(args)
            if name == 'rt_elem':
                mode = int(args[8])
                (y, E, f_, B) = (args[5], args[9], args[10], args[13])
                _go(s, _ok, 'gq_force', nq, G + (f32(k), f32(cm), y, YH, Y0, f_ if mode & 1 else ghost6_P(None), B if mode & 8 else ghost6_P(None), E if mode & 4 else ghost6_P(None)), 128)
            elif name == 'aq_elem':
                _go(s, _ok, 'gq_force', nq, G + (f32(k), f32(cm), args[5], YH, Y0, args[10], args[13], ghost6_P(None)), 128)
            else:
                mode = int(args[9])
                _go(s, _ok, 'gq_force', nq, G + (f32(k), f32(cm), args[5], YH, Y0, args[10] if mode & 1 else ghost6_P(None), args[13] if mode & 8 else ghost6_P(None), ghost6_P(None)), 128)
        elif name in ghost6_MV:
            _go(s, _ok, 'gq_mv', nq, G + (f32(k + cm), args[6], args[7]), 128)
        elif name == 'rt_bd':
            _go(s, _ok, 'gq_force', nq, G + (f32(k), f32(cm), ghost6_P(s.y), YH, Y0, ghost6_P(None), args[6], ghost6_P(None)), 128)
        elif name in ('rt_elem_lsN', 'en_phi_n') and DT:
            return
        elif name == 'rt_elem_lsN':
            assert len(args) >= 19, 'ghost3: rt_elem_lsN (i0·gate 없는 옛 판) 은 안 고침'
            (yh, eta, dl, sk, pm, n0, n1, n2, SC, NC, i0, gate) = (args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], args[16], args[17], args[18])
            _go(s, _ok, 'gq_ls', nq, G + (yh, Y0, eta, dl, sk, pm, n0, n1, n2, ghost6_P(s.gpGQ), gate), 128)
            _go(s, _ok, 'gq_lsfin', 1, (SC, NC, i0, gate, ghost6_P(s.gpGQ), f32(k), f32(cm)), 1)
        elif name == 'en_phi_n':
            cmp = float(s._g3kM) * float(args[2]) if MASS else 0.0
            _go(s, _ok, 'gq_phi', nq, G + (f32(k), f32(cmp), args[3], args[4] if cmp != 0.0 else ghost6_P(None), Y0, args[5], args[6]), 128)
        elif name == 'rt_elem_ls':
            raise RuntimeError(f'ghost3: {name} 경로는 안 고침')
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_k(cls)

def ghost6_check(s, fi, out, reps=200):
    raise NotImplementedError('not used by the paper settings')

def ediag(s, fi, out):
    raise NotImplementedError('not used by the paper settings')

def _bw(f):
    raise NotImplementedError('not used by the paper settings')

def check6(s, fi, out):
    raise NotImplementedError('not used by the paper settings')

def endiag(s, fi, out, kref):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_escape3 ----------------
import re
G2PN = ('rt_g2p_ga', 'aq_g2p', 'ao_g2p', 'ao_g2p_a')
BARK = ('rt_floor_q', 'wb_floor_q', 'rt_bar_q', 'rt_bar_mvq', 'rt_bar_lsN', 'rt_bar_kfe', 'rt_bar_y0e', 'rt_bare_q', 'rt_vres_q', 'rt_vres_lsN', 'yl_kq', 'yl_chk', 'wb_cnt', 'ul_bar', 'ws_bk', 'en_phi_q')
escape3_ST = dict(w=0, g=0, patched=set(), gpatched=set(), skipped=set())
E3_SRC = cu('E3_SRC')

def _bodies(src):
    for m in re.finditer('__global__\\s+void\\s+(?:__launch_bounds__\\(\\d+\\)\\s+)?(\\w+)\\s*\\(', src):
        b = src.find('{', m.end())
        if b < 0:
            continue
        (d, i) = (0, b)
        while i < len(src):
            if src[i] == '{':
                d += 1
            elif src[i] == '}':
                d -= 1
                if d == 0:
                    break
            i += 1
        yield (m.group(1), m.start(), i + 1)

def w_apply(src):
    if '__device__ float* AF_BX;' not in src:
        return src
    (out, pos, hit) = ([], 0, [])
    for (nm, i, j) in list(_bodies(src)):
        b = src[i:j]
        if 'BW[oc*27+o]' not in b or not any((k_ in b for k_ in ('AF_BX', 'AF_BPH', 'AF_BACT'))) or nm.startswith('e3_'):
            continue
        if nm not in BARK:
            escape3_ST['skipped'].add(nm)
            continue
        if nm in ('rt_floor_q', 'wb_floor_q'):
            old = 'float w = BW[oc*27+o];'
            assert b.count(old) == 1 and b.count('BW[oc*27+o]*mw') == 1, ('--e3w', nm)
            b2 = b.replace(old, 'float w = E3_FLW(bm_, q, oc, o);' if nm == 'rt_floor_q' else 'float w = E3W(q, o);', 1)
        else:
            qv = 'qq' if re.search('\\bint\\s+qq\\b', b) else 'q'
            assert re.search('\\bint\\s+%s\\b' % qv, b) or re.search('\\b%s\\s*=' % qv, b), ('--e3w 표본점 번호', nm)
            b2 = b.replace('BW[oc*27+o]', 'E3W(%s, o)' % qv)
        out.append(src[pos:i] + b2)
        pos = j
        hit.append(nm)
    out.append(src[pos:])
    src = ''.join(out)
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ float* AF_WQ;\n__device__ int AF_WQN;\n#define E3W(q, o) AF_WQ[(size_t)(o)*AF_WQN + (size_t)(q)]   // W̄ 는 o 별로 이어 둠 (같은 o 를 이웃 q 가 이어 읽음)', 1)
    if 'rt_floor_q' in hit:
        i = src.index('__global__ void rt_floor_q(')
        src = src[:i] + '#ifdef AF_BAR\n#define E3_FLW(bm_, q, oc, o) (((bm_) > 0.f) ? E3W(q, o) : BW[(oc)*27+(o)])\n#else\n#define E3_FLW(bm_, q, oc, o) BW[(oc)*27+(o)]\n#endif\n' + src[i:]
    i = src.rstrip().rfind('}')
    src = src[:i] + E3_SRC + '\n}\n'
    assert 'rt_bar_q' in hit, ('--e3w: rt_bar_q 못 고침', hit)
    escape3_ST['patched'] |= set(hit)
    return src

def g_apply(src):
    raise NotImplementedError('not used by the paper settings')

def _bind(s):
    if getattr(s, 'e3wq', None) is None:
        return
    import hexfem as ART
    assert getattr(s, 'pkmod', None) is None, '--e3w: 지속형 판 모듈은 안 고침'
    try:
        g = s.mod.get_global('AF_WQ')
    except Exception:
        return
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.e3wq.data_ptr())
    ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=s.mod.get_global('AF_WQN'))[0] = int(s.e3wq.numel() // int(os.environ.get('AF_E3K', '27')))
    s._e3bound = (id(s.mod), int(s.e3wq.data_ptr()))

def escape3_install(w=0, g=0, newg=False):
    import hexfem as ART
    import hexfem as AORT
    P = ART.an_rt_P
    (escape3_ST['w'], escape3_ST['g']) = (int(w), bool(newg))
    _obs = ART.build_src
    _oas = AORT._src

    def bs(*a, **kw):
        r = _obs(*a, **kw)
        if escape3_ST['g']:
            r = g_apply(r)
        return w_apply(r) if escape3_ST['w'] else r
    ART.build_src = bs
    if escape3_ST['w']:

        def as_(*a, **kw):
            return w_apply(_oas(*a, **kw))
        AORT._src = as_
    import hexfem as V0

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if escape3_ST['g'] and name in G2PN and (name in escape3_ST['gpatched']):
                args = tuple(args) + (P(s.mn),)
            elif escape3_ST['g'] and name in ('rt_g2p_endg',):
                raise RuntimeError('--e3g: rt_g2p_endg (generalized-α 아닌 판) 는 안 고침')
            if escape3_ST['w'] and (not getattr(s, '_e3in', False)):
                if name == 'rt_reset':
                    hasbar = bool(getattr(s, 'bar', False)) or getattr(s, 'wbar', None) is not None
                    assert not hasbar or 'rt_bar_q' in escape3_ST['patched'], '--e3w: 장벽 판인데 장벽 커널을 못 고침'
                    if not hasbar and (not escape3_ST.get('nobar_said')):
                        escape3_ST['nobar_said'] = True
                        print('[절 CM45] 장벽 바닥 없는 씬 -- W̄ 해당 없음 (표본점 없음, 소스 그대로)', flush=True)
                    s._e3done = not hasbar
                elif name in escape3_ST['patched'] and (not getattr(s, '_e3done', True)):
                    if not escape3_ST.get('said'):
                        escape3_ST['said'] = True
                        print(f"[절 CM45] W̄ 고친 커널 {sorted(escape3_ST['patched'])}, 안 고친 AF_BX 커널 {sorted(escape3_ST['skipped'])}, W̄ 모으는 자리 = {name} 앞", flush=True)
                    assert getattr(s, '_e3bound', None) == (id(s.mod), int(s.e3wq.data_ptr())), '--e3w: AF_WQ 가 이 모듈·버퍼에 안 묶임'
                    assert not getattr(s, 'det', False), '--e3w: 결정적 판 (AF_BDET 노드 모으기 커널) 은 안 고침'
                    s._e3in = True
                    try:
                        i32 = np.int32
                        _ok(s, 'e3_wqz', s.capQc * 8 * int(os.environ.get('AF_E3K', '27')), (P(s.IP),), 256)
                        _ok(s, 'e3_wq', s.N, (i32(s.N), P(s.IP), P(s.Np), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.mn), i32(escape3_ST['w'])))
                        _ok(s, 'e3_wqn', s.capQc * 8 * int(os.environ.get('AF_E3K', '27')), (P(s.IP),), 256)
                    finally:
                        s._e3in = False
                    s._e3done = True
                    s._e3trig = (name, int(getattr(s, '_e3trig', (None, 0))[1]) + 1)
            return _ok(s, name, n, args, b)
        cls.k = k

    def wrap_alloc(cls):
        _oa = cls.alloc

        def alloc(s, need):
            r = _oa(s, need)
            s.e3wq = torch.zeros(s.capQc * 8 * int(os.environ.get('AF_E3K', '27')), device=s.dev, dtype=torch.float32)
            _bind(s)
            return r
        cls.alloc = alloc
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_k(cls)
    if not escape3_ST['w']:
        return
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_alloc(cls)
    _ob = ART.an_rt_RTSolver.oct_bind

    def oct_bind(s):
        r = _ob(s)
        _bind(s)
        return r
    ART.an_rt_RTSolver.oct_bind = oct_bind
    _oba = AORT.AORTSolver.bind_act

    def bind_act(s):
        r = _oba(s)
        _bind(s)
        return r
    AORT.AORTSolver.bind_act = bind_act

# ---------------- fem_escape4 ----------------
escape4_ST = dict(r=0, a=0)
G2PK = ('aq_g2p', 'ao_g2p', 'rt_g2p_ga', 'rt_g2p_endg')
E5_SRC = cu('E5_SRC')
A0_OLD = '        if (mode == 0) AF_BACT[qq] = (d > 0.f) ? 1 : 0;'
A0_NEW = '        if (mode == 0 && !(d > 0.f) && E5_DQ) { float de_ = dh - d; AF_BX[qq*4] += bm*de_*n0; AF_BX[qq*4+1] += bm*de_*n1; AF_BX[qq*4+2] += bm*de_*n2; E5_DQ[qq] += de_; d = dh; atomicAdd(&E5_CNT[2], 1); }   // 절 CM49 A2: 걸음 시작 판정 자리 (고정 노드는 관성 예측) 에서 d ≤ 0 -> 빼지 않고 그 자리를 원점으로 한 상대 장벽 (기준 d̂)\n' + A0_OLD

def _apply(src):
    if '__device__ float* AF_BX;' not in src:
        return src
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ float* E5_DQ;\n__device__ int* E5_CNT;', 1)
    if escape4_ST['a'] == 2:
        assert src.count(A0_OLD) == 1, '--e5a 2: rt_bar_q 걸음 시작 판정 줄'
        src = src.replace(A0_OLD, A0_NEW, 1)
    i = src.rstrip().rfind('}')
    return src[:i] + E5_SRC + '\n}\n'

def _bind5(s):
    if getattr(s, 'e5DQ', None) is None or getattr(s, 'mod', None) is None:
        return
    import hexfem as ART
    try:
        g = s.mod.get_global('E5_DQ')
    except Exception:
        return
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.e5DQ.data_ptr())
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=s.mod.get_global('E5_CNT'))[0] = int(s.e5cnt.data_ptr())
    s._e5bound = (id(s.mod), int(s.e5DQ.data_ptr()))

class escape4__BK(tuple):
    h = None

def plane(s):
    W = getattr(s, 'wbar', None) or s.floor
    nr = (s.floor or {}).get('NR', [0.0, 0.0, 1.0])
    return ([float(q) for q in nr], float(W['PN']), float(W['DH']))

def hasbar(s):
    return bool(getattr(s, 'bar', False)) or getattr(s, 'wbar', None) is not None

def init_hr(s):
    if getattr(s, 'e5hr', None) is not None or not hasbar(s):
        return
    (nr, pn, _) = plane(s)
    N = int(getattr(s, 'N', s.x.shape[0]))
    s.e5hr = (s.x[:N].double() @ torch.tensor(nr, device=s.x.device, dtype=torch.float64) - pn).float().contiguous()

def escape4_install(r=0, a=0):
    import hexfem as ART
    import hexfem as AORT
    import hexfem as V0
    import hexfem as DET
    import hexfem as MAT
    E3 = sys.modules[__name__]
    (escape4_ST['r'], escape4_ST['a']) = (int(r), int(a))
    P = ART.an_rt_P
    (_obs, _oas) = (ART.build_src, AORT._src)
    ART.build_src = lambda *a_, **k_: _apply(_obs(*a_, **k_))
    AORT._src = lambda *a_, **k_: _apply(_oas(*a_, **k_))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if getattr(s, '_e5in', False):
                return _ok(s, name, n, args, b)
            if name == 'rt_reset':
                s._e5st = not hasbar(s)
            elif name in E3.BARK and (not getattr(s, '_e5st', True)):
                s._e5st = True
                s._e5in = True
                try:
                    if escape4_ST['a'] == 2:
                        assert getattr(s, '_e5bound', None) == (id(s.mod), int(s.e5DQ.data_ptr())), '--e5a 2: E5_DQ 가 이 모듈·버퍼에 안 묶임'
                    init_hr(s)
                    (nr, pn, dh) = plane(s)
                    (i32, f) = (np.int32, np.float32)
                    N = int(s.N)
                    nq = s.capQc * 8
                    _ok(s, 'e5_z', nq, (P(s.IP), P(s.e5R), P(s.e5cnt)), 256)
                    if escape4_ST['r']:
                        _ok(s, 'e5_rp', N, (i32(N), P(s.IP), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.e5hr), P(s.e5R)), 256)
                    _ok(s, 'e5_sh', nq, (P(s.IP), P(s.e5R), P(s.e5DQ), *[f(q) for q in nr], f(pn), f(dh), i32(escape4_ST['r']), i32(escape4_ST['a']), P(s.e5cnt)), 256)
                finally:
                    s._e5in = False
                if not escape4_ST.get('said'):
                    escape4_ST['said'] = True
                    print(f"[절 CM49] 표본점 기준 높이 이어 받기 R {escape4_ST['r']} · 상대 장벽 A {escape4_ST['a']} (d̂ {dh:g}), 기준 옮김 자리 = {name} 앞", flush=True)
            r_ = _ok(s, name, n, args, b)
            if name in G2PK and escape4_ST['r'] and hasbar(s) and (getattr(s, 'e5hr', None) is not None):
                s._e5in = True
                try:
                    (nr, pn, dh) = plane(s)
                    (i32, f) = (np.int32, np.float32)
                    N = int(s.N)
                    nq = s.capQc * 8
                    _ok(s, 'e5_z', nq, (P(s.IP), P(s.e5S), P(s.e5cnt2)), 256)
                    _ok(s, 'e5_ep', N, (i32(N), P(s.IP), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.x), *[f(q) for q in nr], f(pn), P(s.e5S)), 256)
                    _ok(s, 'e5_eh', N, (i32(N), P(s.IP), P(s.fx), P(s.pcd), P(s.cidx), P(s.e5S), P(s.e5DQ), P(s.e5hr), P(s.x), *[f(q) for q in nr], f(pn), f(dh), i32(escape4_ST['r'])), 256)
                finally:
                    s._e5in = False
            return r_
        cls.k = k

    def wrap_alloc(cls):
        _oa = cls.alloc

        def alloc(s, need):
            r_ = _oa(s, need)
            nq = s.capQc * 8
            Z = lambda : torch.zeros(nq, device=s.dev, dtype=torch.float32)
            (s.e5R, s.e5S, s.e5DQ) = (Z(), Z(), Z())
            s.e5cnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
            s.e5cnt2 = torch.zeros(4, device=s.dev, dtype=torch.int32)
            _bind5(s)
            return r_
        cls.alloc = alloc

    def wrap_bk(cls):
        (_ob, _or) = (cls.__dict__['_bk'], cls.__dict__['_rs'])

        def _bk(s):
            init_hr(s)
            b_ = escape4__BK(_ob(s))
            h = getattr(s, 'e5hr', None)
            b_.h = h.clone() if h is not None else None
            return b_

        def _rs(s, bk):
            _or(s, bk)
            h = getattr(s, 'e5hr', None)
            if h is not None and isinstance(bk, escape4__BK) and (bk.h is not None):
                h.copy_(bk.h)
        cls._bk = _bk
        cls._rs = _rs
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_k(cls)
        wrap_alloc(cls)
    _ob = ART.an_rt_RTSolver.oct_bind

    def oct_bind(s):
        r_ = _ob(s)
        _bind5(s)
        return r_
    ART.an_rt_RTSolver.oct_bind = oct_bind
    _oba = AORT.AORTSolver.bind_act

    def bind_act(s):
        r_ = _oba(s)
        _bind5(s)
        return r_
    AORT.AORTSolver.bind_act = bind_act
    for cls in {ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver, DET.det_rt_DetMixin, MAT.MatRT, AORT.AORTSolver}:
        if '_bk' in cls.__dict__ and '_rs' in cls.__dict__:
            wrap_bk(cls)

# ---------------- fem_wolfh5 ----------------
import atexit
wolfh5_ST = dict(dg=0, s=0, c=0, g=0, out='.', marks=[], sol=None, said=False, info={})
(NB, NH, PF, QF) = (9, 2, 14, 10)
CAPR = 4096
EDGES = [-6, -3, 0, 3, 6, 11, 21, 40]
PFIELDS = ['n', 'M', 'Mk', 'Ma', 'Mpin', 'Mv0', 'Mv1', 'Mtau', 'Fnet', 'Fel', 'Ftot', 'Fnet_free', 'Fel_free', 'Mvp']
QFIELDS = ['n', 'V', 'V_cone', 'V_apex', 'V_p', 'V_sdev', 'V_dep', 'V_k', 'V_a', 'V_tr']
W5_SRC = cu('W5_SRC')
FLOOR_OLD = '    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy;'
FLOOR_NEW = '    if (mn[n] < mthr) { float kc_ = (kb && W5_CF) ? kb[n] : 1.f;   // 절 CM70 C: 구속 노드의 저질량 고정 자리 = 구속 예측\n        for (int a=0;a<3;a++) { float yy = Ut[n*3+a] + ((kc_ < 1.f) ? (1.f - kc_)*(z0[n*3+a] + vt[a] - Ut[n*3+a]) : 0.f); yh[n*3+a] = yy;'
ELEM_OLD = 'else atomicAdd(&f[ids[o]*3+a], fs[a]); }'
ELEM_NEW = 'else atomicAdd(&f[ids[o]*3+a], fs[a]); if (W5_FE) atomicAdd(&W5_FE[ids[o]*3+a], fs[a]); }'

def patch_src(src):
    if 'w5_pdg(' in src or 'ao_elem_s(' not in src:
        return src
    i = src.index('__global__ void __launch_bounds__(128) ao_elem_s(')
    j = src.find('__global__', i + 10)
    body = src[i:j]
    assert body.count(ELEM_OLD) == 1, ('wolfh5: ao_elem_s 힘 줄', body.count(ELEM_OLD))
    src = src[:i] + body.replace(ELEM_OLD, ELEM_NEW) + src[j:]
    i = src.index('__global__ void __launch_bounds__(128) ao_elem_s(')
    i = src.rfind('\n', 0, src.rfind('__device__ __forceinline__ void ao_pairs', 0, i))
    src = src[:i] + '\n__device__ double* W5_FE;\n' + src[i:]
    i = src.index('__global__ void mt_floor_n(')
    j = src.find('__global__', i + 10)
    body = src[i:j]
    assert body.count(FLOOR_OLD) == 1, ('wolfh5: mt_floor_n 저질량 줄', body.count(FLOOR_OLD))
    src = src[:i] + '__device__ int W5_CF;\n' + body.replace(FLOOR_OLD, FLOOR_NEW) + src[j:]
    body = W5_SRC
    k = src.rstrip().rfind('}')
    return src[:k] + body + '\n}\n'

def wolfh5_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def wolfh5__rel(s_):
    return bool(getattr(s_, 'has_pbc', False)) and getattr(s_, 'pvb', None) is None and (getattr(s_, 'trel', None) is not None)

def wolfh5__alloc(s_):
    import torch
    dev = s_.dev
    capN = int(s_.capN)
    if getattr(s_, '_w5capN', -1) != capN:
        s_._w5capN = capN
        Z = lambda n, dt=torch.float32: torch.zeros(n, device=dev, dtype=dt)
        s_._w5fe = Z(capN * 3, torch.float64)
        s_._w5ft = Z(capN * 3, torch.float64)
        s_._w5v0 = Z(capN * 3)
        s_._w5al = Z(capN)
        s_._w5num = Z(capN)
        s_._w5nom = Z(capN)
        s_._w5den = Z(capN)
    if getattr(s_, '_w5ctr', None) is None:
        s_._w5ctr = torch.zeros(1, device=dev, dtype=torch.int32)
        s_._w5D = torch.zeros(CAPR * NB * NH * PF, device=dev, dtype=torch.float64)
        s_._w5Q = torch.zeros(CAPR * NB * NH * QF, device=dev, dtype=torch.float64)
        s_._w5meta = torch.zeros(CAPR * 4, device=dev, dtype=torch.float32)
        s_._w5gc = torch.zeros(2, device=dev, dtype=torch.int32)

def _gtbox(s_):
    import torch
    if getattr(s_, '_w5g', None) is not None:
        return s_._w5g
    dx = float(wolfh5_ST['dxg'])
    x = s_.x.reshape(-1, 3)[:s_.N].double()
    lo = torch.floor(x.min(0).values / dx).long() - 3
    hi = torch.floor(x.max(0).values / dx).long() + 4
    G = (hi - lo + 1).tolist()
    K = max(int(s_.K), int(getattr(s_, 'kcap', 0) or 0))
    S = int(s_.pbc_ns)
    ng = G[0] * G[1] * G[2]
    s_._w5g = dict(lo=[int(q) for q in lo.tolist()], G=[int(q) for q in G], ng=ng, KS=K * S, gm=torch.zeros(ng, device=s_.dev, dtype=torch.float32), gacc=torch.zeros(ng * K * S, device=s_.dev, dtype=torch.float32), omg=torch.zeros(ng, device=s_.dev, dtype=torch.float32))
    wolfh5_ST['info'].update(gt_box_lo=s_._w5g['lo'], gt_box_G=s_._w5g['G'], gt_nodes=ng, gt_KS=K * S, gt_dx=dx)
    return s_._w5g

def _bindfe(s_):
    mod = getattr(s_, 'mod', None)
    if mod is None:
        return
    import hexfem as ART
    try:
        g = mod.get_global('W5_FE')
    except Exception:
        return
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s_._w5fe.data_ptr()) if wolfh5_ST['dg'] else 0
    ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=mod.get_global('W5_CF'))[0] = 1 if wolfh5_ST['c'] and wolfh5__rel(s_) else 0

def wolfh5_install(dg=0, s=0, out='.', rel=None, c=0, g=0, gk0=0, gw=0):
    import hexfem as ART
    import hexfem as MRT
    wolfh5_ST.update(dg=int(dg), s=int(s), c=int(c), g=int(g), gk0=int(gk0), gw=int(gw), out=out, rel=rel or {}, dxg=float((rel or {}).get('dxg', 0.01)))
    _ob = ART.build_src
    ART.build_src = lambda *a, **kw: patch_src(_ob(*a, **kw))
    (i32, f32) = (np.int32, np.float32)
    _mb = MRT.MatRT.mat_bind

    def mat_bind(s_):
        r = _mb(s_)
        wolfh5__alloc(s_)
        wolfh5_ST['sol'] = s_
        _bindfe(s_)
        if wolfh5_ST['s'] and wolfh5__rel(s_):
            _gtbox(s_)
        return r
    MRT.MatRT.mat_bind = mat_bind
    _k = ART.an_rt_RTSolver.k

    def k(s_, name, nn, args, b=128):
        if getattr(s_, '_w5in', False) or name.startswith('w5') or (not hasattr(s_, 'matk')) or (not s_.matk):
            return _k(s_, name, nn, args, b)
        s_._w5in = True
        try:
            wolfh5__alloc(s_)
            on_rel = wolfh5__rel(s_)
            if wolfh5_ST['gw'] and on_rel and (name == 'rt_gauss'):
                G6 = sys.modules[__name__]
                G6.ghost6__env_init()
                if not G6.WMODE:
                    G6.WMODE = 99
            if name == 'rt_reset':
                (s_._w5ks, s_._w5toff) = (0, 0.0)
            if wolfh5_ST['dg']:
                if name == 'rt_ga_step0':
                    _k(s_, 'w5_copy_f', int(s_.capN) * 3, (wolfh5_P(s_.IP), wolfh5_P(s_.vk), wolfh5_P(s_._w5v0)), 256)
                    s_._w5dt = float(args[4])
                    s_._w5toff_c = getattr(s_, '_w5toff', 0.0) + 0.5 * float(args[4])
                elif name == 'ao_elem_s':
                    _k(s_, 'w5_zero_d', int(s_.capN) * 3, (i32(int(s_.capN) * 3), wolfh5_P(s_._w5fe)), 256)
            if wolfh5_ST['s'] and on_rel and (name == 'mt_akb'):
                return _k(s_, 'w5s_akb', nn, (args[0], args[1], args[2]), 256)
            r = _k(s_, name, nn, args, b)
            if wolfh5_ST['g'] and on_rel and (name == 'mt_floor_n') and (getattr(s_, 'g3fm', None) is not None):
                _k(s_, 'w5_gfx', int(s_.capQc), (args[0], wolfh5_P(s_.cb), wolfh5_P(s_.kb), wolfh5_P(s_.g3fm), wolfh5_P(s_._w5gc)), 128)
            if wolfh5_ST['gw'] and on_rel and (name == 'mt_floor_n') and (getattr(s_, 'g5w', None) is not None):
                _k(s_, 'w5_gwk', int(s_.capQc), (args[0], wolfh5_P(s_.cb), wolfh5_P(s_.kb), wolfh5_P(s_.g3fm), wolfh5_P(s_.g5w)), 128)
            if wolfh5_ST['dg'] and name == 'ao_elem_s':
                _k(s_, 'w5_copy_d', int(s_.capN) * 3, (wolfh5_P(s_.IP), wolfh5_P(s_.f), wolfh5_P(s_._w5ft)), 256)
            if name == 'ao_pbc_acc' and wolfh5_ST['s'] and on_rel:
                G = _gtbox(s_)
                _k(s_, 'w5_zero_f', G['ng'], (i32(G['ng']), wolfh5_P(G['gm'])), 256)
                _k(s_, 'w5_zero_f', G['ng'] * G['KS'], (i32(G['ng'] * G['KS']), wolfh5_P(G['gacc'])), 256)
                _k(s_, 'w5s_acc', s_.N, (i32(s_.N), wolfh5_P(s_.x), wolfh5_P(s_.m), wolfh5_P(s_.trel), wolfh5_P(s_.tbuf), args[7], args[8], args[9], f32(1.0 / float(wolfh5_ST['dxg'])), *[i32(q) for q in G['lo']], *[i32(q) for q in G['G']], wolfh5_P(G['gm']), wolfh5_P(G['gacc'])), 256)
            if name in ('ao_pbc_k', 'mt_pbc_kphi'):
                if wolfh5_ST['s'] and on_rel:
                    G = _gtbox(s_)
                    _k(s_, 'w5s_kg', G['ng'], (i32(G['ng']), wolfh5_P(G['gm']), wolfh5_P(G['gacc']), args[3], args[4], args[5], args[6], i32(1 if name == 'mt_pbc_kphi' else 0), wolfh5_P(G['omg'])), 256)
                    for t_ in (s_._w5num, s_._w5nom, s_._w5den):
                        _k(s_, 'w5_zero_f', int(s_.capN), (i32(int(s_.capN)), wolfh5_P(t_)), 256)
                    _k(s_, 'w5s_kn', s_.N, (i32(s_.N), wolfh5_P(s_.IP), wolfh5_P(s_.Np), wolfh5_P(s_.fx), wolfh5_P(s_.x), wolfh5_P(s_.m), f32(1.0 / float(wolfh5_ST['dxg'])), *[i32(q) for q in G['lo']], *[i32(q) for q in G['G']], wolfh5_P(G['omg']), wolfh5_P(s_._w5num), wolfh5_P(s_._w5nom), wolfh5_P(s_._w5den)), 256)
                    _k(s_, 'w5s_kb', int(s_.capN), (wolfh5_P(s_.IP), wolfh5_P(s_._w5num), wolfh5_P(s_._w5nom), wolfh5_P(s_._w5den), args[7]), 256)
                if wolfh5_ST['dg']:
                    _k(s_, 'w5_alpha', int(s_.capN), (args[0], args[1], args[2], args[3], args[4], args[5], wolfh5_P(s_._w5al)), 256)
            if wolfh5_ST['dg'] and name == 'ao_commit':
                FL = s_.floor or dict(NR=[0.0, 0.0, 1.0], PN=-1e+30)
                nr = FL['NR']
                toff = getattr(s_, '_w5toff_c', 0.5 * float(s_.dt))
                dt = getattr(s_, '_w5dt', float(s_.dt))
                rl = wolfh5_ST['rel']
                kbp = wolfh5_P(s_.kb) if getattr(s_, 'has_pbc', False) else wolfh5_P(None)
                alp = wolfh5_P(s_._w5al) if getattr(s_, 'has_pbc', False) else wolfh5_P(None)
                trel = s_.trel if getattr(s_, 'trel', None) is not None else None
                if trel is not None:
                    _k(s_, 'w5_pdg', s_.N, (i32(s_.N), wolfh5_P(s_.IP), wolfh5_P(s_.Np), wolfh5_P(s_.fx), wolfh5_P(s_.x), wolfh5_P(s_.m), wolfh5_P(s_.v), wolfh5_P(trel), wolfh5_P(s_.tbuf), f32(toff), f32(rl.get('vf', 0.0)), kbp, alp, wolfh5_P(s_.mn), wolfh5_P(s_._w5v0), wolfh5_P(s_.vk), wolfh5_P(s_._w5fe), wolfh5_P(s_._w5ft), f32(dt), *[f32(q) for q in s_.g], *[f32(q) for q in nr], f32(FL['PN']), wolfh5_P(s_._w5D), wolfh5_P(s_._w5ctr), i32(CAPR)), 256)
                if getattr(s_, 'bar', False):
                    _k(s_, 'w5_qdg', nn, (args[0], args[1], args[2], wolfh5_P(s_.BW), args[3], wolfh5_P(s_.Vq), args[4], f32(s_.mu), f32(s_.lam), args[5], kbp, alp, i32(rl.get('ax', 0)), f32(rl.get('c0', 0.0)), f32(rl.get('H0', 0.0)), f32(rl.get('vf', 0.0)), wolfh5_P(s_.tbuf), f32(toff), *[f32(q) for q in nr], f32(FL['PN']), wolfh5_P(s_._w5Q), wolfh5_P(s_._w5ctr), i32(CAPR)), 128)
                _k(s_, 'w5_inc', 1, (wolfh5_P(s_._w5ctr), wolfh5_P(s_._w5meta), wolfh5_P(s_.tbuf), f32(toff), f32(dt), i32(getattr(s_, '_w5ks', 0)), i32(CAPR)), 1)
                s_._w5ks = getattr(s_, '_w5ks', 0) + 1
                s_._w5toff = getattr(s_, '_w5toff', 0.0) + dt
            elif name == 'ao_commit':
                s_._w5ks = getattr(s_, '_w5ks', 0) + 1
            return r
        finally:
            s_._w5in = False
    ART.an_rt_RTSolver.k = k
    _os = ART.an_rt_RTSolver.step

    def step(s_, t, *a, **kw):
        d = getattr(s_, '_w5dep', 0)
        s_._w5dep = d + 1
        try:
            r = _os(s_, t, *a, **kw)
        finally:
            s_._w5dep = d
        if d == 0 and wolfh5_ST['dg']:
            c = getattr(s_, '_w5ctr', None)
            wolfh5_ST['marks'].append((int(round(t / s_.h)), int(c.item()) if c is not None else -1))
        if d == 0 and wolfh5_ST['s'] and (not wolfh5_ST['said']) and (getattr(s_, '_w5g', None) is not None):
            wolfh5_ST['said'] = True
            print(f"[절 CM70] S_구속 정답 격자 상자 {s_._w5g['G']} 노드 {s_._w5g['ng']} 표본 {s_._w5g['KS']}", flush=True)
        return r
    ART.an_rt_RTSolver.step = step
    if wolfh5_ST.get('gk0'):
        G6 = sys.modules[__name__]
        G6.ghost6__env_init()
        _ok0 = G6.ghost6__k_of

        def _k_of(s_):
            first = getattr(s_, '_w5k0', None) is None
            _ok0(s_)
            if first:
                s_._w5k0 = float(s_._gpk)
                print(f'[절 CM70] gk0: ghost penalty 강성 k {s_._gpk:.6g} -> 0 (고스트 질량 그대로)', flush=True)
            s_._gpk = 0.0
            return s_._gpk
        G6.ghost6__k_of = _k_of
    print(f"[절 CM70] wolfh5 켬: 진단 {wolfh5_ST['dg']} · S_구속 {wolfh5_ST['s']} · C {wolfh5_ST['c']} · B_g {wolfh5_ST['g']} · 강성 0 {wolfh5_ST['gk0']} · B_gk {wolfh5_ST['gw']} · 방출 {wolfh5_ST['rel']}", flush=True)

    def _fin():
        info = dict(dg=wolfh5_ST['dg'], s=wolfh5_ST['s'], c=wolfh5_ST['c'], g=wolfh5_ST['g'], gw=wolfh5_ST['gw'], rel=wolfh5_ST['rel'], **wolfh5_ST['info'])
        sol = wolfh5_ST.get('sol')
        if sol is not None:
            try:
                import torch
                torch.cuda.synchronize()
                info['rel_scene'] = wolfh5__rel(sol)
                if wolfh5_ST['gw'] and getattr(sol, 'g5w', None) is not None:
                    G6 = sys.modules[__name__]
                    G6.ghost6__env_init()
                    info['gw_mode'] = int(G6.WMODE)
                    fw = sol.g5w[:int(sol.IP[11].item()) * 3]
                    info.update(gw_lt1=int((fw < 1).sum()), gw_eq0=int((fw == 0).sum()))
                if wolfh5_ST['g'] and getattr(sol, '_w5gc', None) is not None:
                    info['g_removed_faces_total'] = int(sol._w5gc[0].item())
                if wolfh5_ST['s'] and getattr(sol, 'has_pbc', False):
                    kb = sol.kb[:int(sol.IP[9].item())]
                    info.update(kb_lt1=int((kb < 1).sum()), kb_lt05=int((kb < 0.5).sum()), kb_min=float(kb.min()) if kb.numel() else None)
                if wolfh5_ST['dg']:
                    nr = int(sol._w5ctr.item())
                    D = sol._w5D.view(CAPR, NB, NH, PF)[:min(nr, CAPR)].cpu().numpy()
                    Q = sol._w5Q.view(CAPR, NB, NH, QF)[:min(nr, CAPR)].cpu().numpy()
                    M = sol._w5meta.view(CAPR, 4)[:min(nr, CAPR)].cpu().numpy()
                    np.savez_compressed(os.path.join(out, 'w5dg.npz'), D=D, Q=Q, meta=M, marks=np.array(wolfh5_ST['marks']))
                    (fr, lo) = ([], 0)
                    for (f_, hi) in wolfh5_ST['marks']:
                        hi = min(hi, D.shape[0])
                        if hi > lo:
                            rows = list(range(lo, hi))
                            k0 = [r_ for r_ in rows if int(M[r_, 2]) == 0]
                            use = rows[rows.index(k0[-1]):] if k0 else rows
                            fr.append(dict(f=f_, rows=[lo, hi], use=use, dt=[float(M[r_, 1]) for r_ in use], P=np.sum([D[r_] * M[r_, 1] for r_ in use], 0).tolist(), Q=np.sum([Q[r_] * M[r_, 1] for r_ in use], 0).tolist()))
                        lo = hi
                    json.dump(dict(edges_cm=EDGES, pfields=PFIELDS, qfields=QFIELDS, nrows=nr, frames=fr, note='P·Q = 그 프레임 걸음들의 Δt 가중 합 (나누기는 표에서 Σ Δt)'), open(os.path.join(out, 'w5dg.json'), 'w'))
                    info['dg_rows'] = nr
            except Exception as e:
                info['err'] = repr(e)
        json.dump(info, open(os.path.join(out, 'w5info.json'), 'w'))
    atexit.register(_fin)

# ---------------- fem_wolfh6 ----------------
wolfh6_ST = dict(X=0, nfl=0, K=0, L=0, out='.', nsrc=0, nhl=0, kap=None, k0=None, k1=None, mbar=None, Lon=None, said=False, nlaunch=0)
KERS = (('rt_ga_end_hc', True), ('rt_ga_end_ufl', True), ('ao_stepend_hc', False), ('ao_stepend_ufl', False))
VK_LINE = '        vk[n*3+a] = v1*damp;\n'
LOOP = '    for (int a=0;a<3;a++) {\n'

def hlify(src):
    raise NotImplementedError('not used by the paper settings')
LX_DEV = cu('LX_DEV')
FL_KEYS = ('mt_floor_n', 'ao_floor_n')

def lxify(src):
    raise NotImplementedError('not used by the paper settings')

def wolfh6_P(t):
    raise NotImplementedError('not used by the paper settings')

class _Launch:

    def __init__(self, s, nh):
        raise NotImplementedError('not used by the paper settings')

def _mbar(s):
    raise NotImplementedError('not used by the paper settings')

def wolfh6_install(out, K=0, L=0, X=0):
    wolfh6_ST.update(K=int(K), L=int(L), X=int(X), out=out, mbar_of=_mbar)
    import hexfem as ART
    import hexfem as AORT
    if wolfh6_ST['K']:
        G6 = sys.modules[__name__]
        G6.ghost6__env_init()
        L4 = sys.modules[__name__]
        _ok = G6.ghost6__k_of

        def _k_of(s):
            first = not getattr(s, '_w6kd', False)
            _ok(s)
            if first:
                (kap, info) = L4.kappa_of(s)
                s._w6kd = True
                k0 = float(s._gpk)
                s._gpk = s._gpk * float(kap)
                wolfh6_ST.update(kap=float(kap), k0=k0, k1=float(s._gpk), kinfo=info)
                print(f"[절 CM77 wolfh6] (K) 재질 {info.get('name')} κ {kap:g} ({info.get('rule')}): P1X 강성 k {k0:.6g} -> {float(s._gpk):.6g} (고스트 질량 k_M {float(getattr(s, '_g3kM', 0.0)):.6g} 그대로)", flush=True)
                wolfh6__dump()
            return s._gpk
        G6.ghost6__k_of = _k_of
    if wolfh6_ST['X']:
        (o_bx, o_sx) = (ART.build_src, AORT._src)
        ART.build_src = lambda *a, **kw: lxify(o_bx(*a, **kw))
        AORT._src = lambda *a, **kw: lxify(o_sx(*a, **kw))
    if wolfh6_ST['L']:
        (o_b, o_s) = (ART.build_src, AORT._src)
        ART.build_src = lambda *a, **kw: hlify(o_b(*a, **kw))
        AORT._src = lambda *a, **kw: hlify(o_s(*a, **kw))
        for cls in (ART.an_rt_RTSolver, AORT.AORTSolver):
            wolfh6__wrap_k(cls)
    print(f"[절 CM77] wolfh6 켬 -- (K) P1X 강성 × κ {wolfh6_ST['K']} · (L) 가벼운 자유 노드 HHT·후퇴 오일러 섞기 {wolfh6_ST['L']} · (L') 저질량 고정 노드 자리 이웃 외삽 {wolfh6_ST['X']}", flush=True)
    import atexit
    atexit.register(wolfh6__dump)

def wolfh6__wrap_k(cls):
    raise NotImplementedError('not used by the paper settings')

def wolfh6__dump():
    try:
        d = {k: v for (k, v) in wolfh6_ST.items() if k not in ('mbar_of',)}
        with open(os.path.join(wolfh6_ST['out'], 'w6info.json'), 'w') as f_:
            json.dump(d, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

# ---------------- fem_wolfh8 ----------------
wolfh8_ST = dict(D=2, T=0, X=0, out='.', nsrc=0, nclass=0, nfold=0, nexp=0, ngg=0, nalloc=0, nstep=0, said=False, err=None)
wolfh8_MARK = '// ---- 절 CM81 wolfh8'
W8_SRC = cu('W8_SRC')

def add_src(src):
    if wolfh8_MARK in src:
        return src
    assert 'Tapply(' in src and 'blocksum_add(' in src, 'wolfh8: Tapply·blocksum_add 없는 모듈'
    wolfh8_ST['nsrc'] += 1
    s = src.rstrip()
    i = s.rfind('}')
    return s[:i] + W8_SRC + '\n}\n'

def wolfh8_P(t):
    raise NotImplementedError('not used by the paper settings')

def wolfh8__on(s):
    return getattr(s, '_w8on', None) is not None and getattr(s, 'w8b', None) is not None

def wolfh8_install(out, D=2, T=0, X=0):
    import hexfem as ART
    wolfh8_ST.update(D=int(D), T=int(T), X=int(X), out=out)
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.an_rt_RTSolver
    (oa, ok, oc, os_) = (R.alloc, R.k, R.ctl, R.step)

    def alloc(s, need):
        r = oa(s, need)
        dev = s.dev
        s.w8ic = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.w8b = torch.full((s.capN,), -1, device=dev, dtype=torch.int32)
        s.w8w = torch.zeros(s.capN * 9, device=dev, dtype=torch.float32)
        s.w8acc = torch.zeros(s.capN * 3, device=dev, dtype=torch.float32)
        s.w8cnt = torch.zeros(16, device=dev, dtype=torch.int32)
        s.w8cf = torch.zeros(4, device=dev, dtype=torch.float32)
        s._w8on = True
        wolfh8_ST['nalloc'] += 1
        return r

    def fold(s, v, sk, pm, nr):
        i32 = np.int32
        ok(s, 'w8_fold', s.capN, (wolfh8_P(s.IP), wolfh8_P(s.w8b), wolfh8_P(s.w8w), v, wolfh8_P(s.w8acc)), 256)
        ok(s, 'w8_fold2', s.capN, (wolfh8_P(s.IP), sk, pm, *nr, wolfh8_P(s.w8acc), v), 256)
        wolfh8_ST['nfold'] += 1

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if not wolfh8__on(s):
            return r
        i32 = np.int32
        if name == 'rt_grad':
            fold(s, args[12], args[3], args[4], tuple(args[5:8]))
            ok(s, 'rt_sc_clear', 1, (args[17], i32(1), i32(2)), 1)
            ok(s, 'w8_gg', s.capN, (wolfh8_P(s.IP), args[12], args[17]), 256)
            wolfh8_ST['ngg'] += 1
        elif name == 'rt_ap':
            fold(s, args[10], args[3], args[4], tuple(args[5:8]))
        elif name == 'rt_pinit':
            ok(s, 'w8_expand', s.capN, (wolfh8_P(s.IP), wolfh8_P(s.w8b), wolfh8_P(s.w8w), args[7], args[8]), 256)
            wolfh8_ST['nexp'] += 1
        elif name == 'rt_cg2':
            ok(s, 'w8_expand', s.capN, (wolfh8_P(s.IP), wolfh8_P(s.w8b), wolfh8_P(s.w8w), args[8], args[9]), 256)
            wolfh8_ST['nexp'] += 1
        return r

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 1 and wolfh8__on(s):
            assert getattr(s, 'nl', 1) > 1, 'wolfh8: MG 판 (--mg_levels > 1) 만 (rt_grad 의 z·p 를 MG 가 다시 씀)'
            i32 = np.int32
            ok(s, 'rt_zero_i', s.capC, (i32(s.capC), wolfh8_P(s.w8ic)), 256)
            if wolfh8_ST['T'] in (1, 2):
                ok(s, 'w8_icell2', s.capQc, (wolfh8_P(s.IP), wolfh8_P(s.cb), wolfh8_P(s.occ), wolfh8_P(s.w8ic), wolfh8_P(s.w8cnt), i32(1 if wolfh8_ST['T'] == 2 else 0)), 128)
            else:
                ok(s, 'w8_icell', s.capQc, (wolfh8_P(s.IP), wolfh8_P(s.cb), wolfh8_P(s.Aacc), wolfh8_P(s.w8ic), wolfh8_P(s.w8cnt)), 128)
            ok(s, 'w8_class', s.capN, (wolfh8_P(s.IP), wolfh8_P(s.mn), wolfh8_P(s.w8ic), i32(wolfh8_ST['D']), wolfh8_P(s.w8b), wolfh8_P(s.w8w), wolfh8_P(s.w8cnt), wolfh8_P(s.w8cf), i32(wolfh8_ST['X'])), 256)
            ok(s, 'w8_free', s.capN, (wolfh8_P(s.IP), wolfh8_P(s.w8b), wolfh8_P(s.sk), wolfh8_P(s.pm), wolfh8_P(s.yh), wolfh8_P(s.y), wolfh8_P(s.eta)), 256)
            wolfh8_ST['nclass'] += 1
            if not wolfh8_ST['said']:
                wolfh8_ST['said'] = True
                print(f"[절 CM81 wolfh8] 확장 훅 -- 차수 {wolfh8_ST['D']}, 걸음 시작 분류·변수 바꿈 항등, rt_grad 접기 + |g|² 다시, rt_pinit·rt_cg2 펼치기, rt_ap 접기 (MG 층 {s.nl})", flush=True)
                wolfh8__dump()
        return r

    def step(s, t):
        on = wolfh8__on(s)
        if on:
            s.w8cnt.zero_()
            s.w8cf.zero_()
        r = os_(s, t)
        if on:
            try:
                c = s.w8cnt.tolist()
                m = s.w8cf.tolist()
                mt = sum(m[:3])
                with open(os.path.join(wolfh8_ST['out'], 'w8.jsonl'), 'a') as f_:
                    f_.write(json.dumps(dict(f=wolfh8_ST['nstep'], icell=c[3], ncell=int(s.IP[11]), inner=c[0], outer=c[1], iso=c[2], ax1=c[5], ax2=c[6], ax3=c[7], xiso=c[8], m_xiso=m[3], m_inner=m[0], m_outer=m[1], m_iso=m[2], mf_outer=m[1] / mt if mt > 0 else None, mf_iso=m[2] / mt if mt > 0 else None, redo=bool(r))) + '\n')
            except Exception as e_:
                wolfh8_ST['err'] = f'{type(e_).__name__}: {e_}'
        wolfh8_ST['nstep'] += 1
        return r
    (R.alloc, R.k, R.ctl, R.step) = (alloc, k, ctl, step)
    print(f"[절 CM81] wolfh8 켬 -- 확장 B-스플라인 차수 {wolfh8_ST['D']} (내부 칸 정의 {wolfh8_ST['T']}: 0 = 팔분 8 개 점유, 1 = 26 이웃 칸 모두 점유, 2 = 면 이웃 6 칸 모두 점유; 외부 노드 = 받침에 내부 칸 없는 활성 노드, 증분 아핀 확장)", flush=True)
    import atexit
    atexit.register(wolfh8__dump)

def wolfh8__dump():
    try:
        with open(os.path.join(wolfh8_ST['out'], 'w8info.json'), 'w') as f_:
            json.dump(wolfh8_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

# ---------------- fem_wolfh12 ----------------
wolfh12_ST = dict(mode=0, dg=0, out='.', nsrc=0, nalloc=0, ncall=0, nframe=0, err=None)
wolfh12_MARK = '// ---- 절 CM93 wolfh12'
wolfh12_LOG = [None]

def wolfh12_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())
W12_SRC = cu('W12_SRC')
wolfh12_HEAD = '__global__ void ao_commit('

def wolfh12_patch(code):
    if wolfh12_MARK in code or wolfh12_HEAD not in code:
        return code
    i = code.index(wolfh12_HEAD)
    wolfh12_ST['nsrc'] += 1
    return code[:i] + wolfh12_MARK + '\n' + W12_SRC + code[i:]

def wolfh12__dump():
    try:
        with open(os.path.join(wolfh12_ST['out'], 'w12info.json'), 'w') as f_:
            json.dump(wolfh12_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

def _grp(B, m):
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    b = B[m].double()
    V = b[:, 7]
    sv = float(V.sum())
    wm = (lambda c: float((b[:, c] * V).sum() / sv)) if sv > 0 else lambda c: float(b[:, c].mean())
    return dict(n=n, V=sv, tr=wm(1), p=wm(3), hr=wm(6), apex=float(((b[:, 2] == 2).double() * V).sum() / sv) if sv > 0 else None, cone=float(((b[:, 2] == 1).double() * V).sum() / sv) if sv > 0 else None, Ev=float((b[:, 4] * V).sum()), Ed=float((b[:, 5] * V).sum()), Jmin=float(b[:, 0].min()))

def _hook(R):
    (oa, ok, os_) = (R.alloc, R.k, R.step)

    def alloc(s, need):
        r = oa(s, need)
        if wolfh12_ST['dg'] and getattr(s, '_mat', None) == 2 and (getattr(s, 'Fq', None) is not None):
            capQ = s.Fq.numel() // 9
            if getattr(s, 'w12b', None) is None or s.w12b.numel() != capQ * 8:
                s.w12b = torch.zeros(capQ * 8, device=s.dev, dtype=torch.float32)
                wolfh12_ST['nalloc'] += 1
        return r

    def k(s, name, n, args, b=128):
        if name == 'ao_commit' and getattr(s, 'w12b', None) is not None:
            ok(s, 'w12_q', n, tuple(args[:4]) + (wolfh12_P(s.Vq), args[4], np.float32(s.mu), np.float32(s.lam), args[5], wolfh12_P(s.w12b)), 128)
            wolfh12_ST['ncall'] += 1
        return ok(s, name, n, args, b)

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if getattr(s, 'w12b', None) is not None:
            wolfh12_ST['nframe'] += 1
            if wolfh12_LOG[0] is not None:
                try:
                    nq = int(s.IP[12].item())
                    B = s.w12b[:nq * 8].view(nq, 8)
                    B = B[B[:, 7] > 0]
                    J = B[:, 0]
                    o = dict(f=wolfh12_ST['nframe'], nq=nq, mu=float(s.mu), lam=float(s.lam), cdp=float(s.cdp), all=_grp(B, torch.ones_like(J, dtype=torch.bool)), c1=_grp(B, J < 1), c09=_grp(B, J < 0.9), c05=_grp(B, J < 0.5), inv=_grp(B, J <= 0))
                    wolfh12_LOG[0].write(json.dumps(o) + '\n')
                    wolfh12_LOG[0].flush()
                except Exception as e:
                    wolfh12_ST['err'] = repr(e)
            if wolfh12_ST['nframe'] in (1, 2, 5) or wolfh12_ST['nframe'] % 25 == 0:
                wolfh12__dump()
        return r
    (R.alloc, R.k, R.step) = (alloc, k, step)

def wolfh12_install(out, mode=0, dg=0):
    import hexfem as ART
    wolfh12_ST.update(mode=int(mode), dg=int(dg), out=out)
    if wolfh12_ST['dg']:
        wolfh12_LOG[0] = open(os.path.join(out, 'w12.jsonl'), 'w')
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if 'code' in kw:
            kw['code'] = wolfh12_patch(kw['code'])
        elif a:
            a = (wolfh12_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    _hook(ART.an_rt_RTSolver)
    print(f"[절 CM93] wolfh12 켬 -- 모드 {wolfh12_ST['mode']}, 진단 {wolfh12_ST['dg']}", flush=True)
    wolfh12__dump()

# ---------------- fem_ext1 ----------------
import weakref
ext1_ST = dict(D=2, T=1, X=0, ck=0, out='.', nsrc=0, nlog_src=0, nalloc=0, nclass=0, nfold=0, nexp=0, ngg=0, njz=0, nwarm=0, nwscand=0, nkprop=0, nwslsup=0, nlog=0, nck=0, ninv=0, said=False, err=None, ck_err=None)
ext1_MARK = '// ---- 절 CM83 ext1'
MARKL = '// ---- 절 CM83 ext1 기록'
HCAP = 8192
_SOL = []
X1_LOG = cu('X1_LOG')
X1_SRC = cu('X1_SRC')

def ext1_P(t):
    raise NotImplementedError('not used by the paper settings')

def _ins(src, body, mark):
    if mark in src:
        return src
    assert 'Tapply(' in src and 'blocksum_add(' in src, 'ext1: Tapply·blocksum_add 없는 모듈'
    s = src.rstrip()
    i = s.rfind('}')
    return s[:i] + body + '\n}\n'

def _add_ao(src):
    raise NotImplementedError('not used by the paper settings')

def _add_log(src):
    if MARKL in src:
        return src
    ext1_ST['nlog_src'] += 1
    return _ins(src, X1_LOG, MARKL)

def _log(s, ok):
    raise NotImplementedError('not used by the paper settings')

def _install_an(out, D, T, X):
    import hexfem as ART
    W8 = sys.modules[__name__]
    R = ART.an_rt_RTSolver
    st0 = R.step
    W8.wolfh8_install(out, D=D, T=T, X=X)
    R.step = st0
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: _add_log(ob(*a, **kw))
    (oc, ok) = (R.ctl, R.k)

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 0:
            s._x13ks = 0
        elif op == 1 and W8.wolfh8__on(s):
            _log(s, ok)
        return r
    R.ctl = ctl

def _install_ao():
    import hexfem as AORT
    oas = AORT._src
    AORT._src = lambda *a, **kw: _add_ao(oas(*a, **kw))
    C = AORT.AORTSolver
    (oa, ok, oc) = (C.alloc, C.k, C.ctl)
    i32 = np.int32

    def on(s):
        return getattr(s, 'w8b', None) is not None and getattr(s, '_x13on', False)

    def alloc(s, need):
        r = oa(s, need)
        dev = s.dev
        s.w8ic = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.w8b = torch.full((s.capN,), -1, device=dev, dtype=torch.int32)
        s.w8w = torch.zeros(s.capN * 9, device=dev, dtype=torch.float32)
        s.w8acc = torch.zeros(s.capN * 3, device=dev, dtype=torch.float32)
        s.w8cnt = torch.zeros(16, device=dev, dtype=torch.int32)
        s.w8cf = torch.zeros(4, device=dev, dtype=torch.float32)
        s._x13on = True
        ext1_ST['nalloc'] += 1
        return r

    def cmp(s):
        return i32(1 if getattr(s, 'compact', False) else 0)

    def fold(s, v, sk, pm, nr):
        ok(s, 'w8_fold', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.w8w), v, ext1_P(s.w8acc)), 256)
        ok(s, 'w8_fold2', s.capN, (ext1_P(s.IP), sk, pm, *nr, ext1_P(s.w8acc), v), 256)
        ext1_ST['nfold'] += 1

    def expand(s, p, Tp):
        ok(s, 'w8_expand', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.w8w), p, Tp), 256)
        ext1_ST['nexp'] += 1

    def jz(s, Dinv, sk, pm, nr, r, z, p, Tp, SC):
        ok(s, 'rt_sc_clear', 1, (SC, i32(0), i32(1)), 1)
        ok(s, 'x1_jz', s.capN, (ext1_P(s.IP), Dinv, sk, pm, *nr, r, z, p, Tp, SC, cmp(s)), 256)
        ext1_ST['njz'] += 1
        expand(s, p, Tp)

    def k(s, name, n, args, b=128):
        if on(s) and ext1_ST['ck'] and (not getattr(s, '_x13in', False)) and (name in ('aq_elem', 'ao_elem_s', 'ao_elem', 'rt_bar_q', 'rt_bar_n')):
            s._x13rec = getattr(s, '_x13rec', {})
            s._x13rec[name] = tuple(args)
        r = ok(s, name, n, args, b)
        if not on(s):
            return r
        if name in ('rt_grad', 'rt_grad_c'):
            nr = tuple(args[5:8])
            fold(s, args[12], args[3], args[4], nr)
            ok(s, 'rt_sc_clear', 1, (args[17], i32(1), i32(2)), 1)
            ok(s, 'x1_gg', s.capN, (ext1_P(s.IP), args[12], args[17], cmp(s)), 256)
            ext1_ST['ngg'] += 1
            if getattr(s, 'nl', 1) <= 1:
                jz(s, args[11], args[3], args[4], nr, args[12], args[13], args[14], args[16], args[17])
            if ext1_ST['ck'] and (not getattr(s, '_x13in', False)):
                s._x13g = dict(name=name, args=tuple(args))
                _check_iter(s, ok)
        elif name in ('rt_ap', 'rt_ap_c'):
            fold(s, args[10], args[3], args[4], tuple(args[5:8]))
        elif name == 'rt_pinit':
            expand(s, args[7], args[8])
            if ext1_ST['ck'] and (not getattr(s, '_x13in', False)):
                _check_mg(s)
        elif name in ('rt_cg2', 'rt_cg2_c'):
            expand(s, args[8], args[9])
        elif name == 'ao_tvec':
            if int(args[6]) == int(s.d0.data_ptr()):
                expand(s, args[6], args[7])
        elif name == 'ao_warm':
            nr = tuple(args[5:8])
            fold(s, args[12], args[3], args[4], nr)
            jz(s, args[10], args[3], args[4], nr, args[12], args[13], args[14], args[15], args[16])
            ext1_ST['nwarm'] += 1
        elif name == 'ws_cand':
            ok(s, 'x1_wscand', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.w8w), args[1], args[10], ext1_P(s.y), args[11], args[12]), 256)
            ext1_ST['nwscand'] += 1
        elif name == 'ws_merge':
            ok(s, 'x1_kprop', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.w8w), args[1]), 256)
            ext1_ST['nkprop'] += 1
        elif name == 'ws_lsup':
            ok(s, 'x1_wslsup', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.w8w), args[1], ext1_P(s.awtd), args[9], args[7], args[8], args[10]), 256)
            ext1_ST['nwslsup'] += 1
            if ext1_ST['ck'] and (not getattr(s, '_x13in', False)):
                _check_inv(s, 'ws_lsup')
        elif name in ('rt_lsup', 'rt_lsup_c', 'en_bt_fin'):
            if ext1_ST['ck'] and (not getattr(s, '_x13in', False)):
                _check_inv(s, name)
        return r

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 0:
            s._x13ks = 0
        if op == 1 and on(s):
            ok(s, 'rt_zero_i', s.capC, (i32(s.capC), ext1_P(s.w8ic)), 256)
            if ext1_ST['T'] in (1, 2):
                ok(s, 'w8_icell2', s.capQc, (ext1_P(s.IP), ext1_P(s.cb), ext1_P(s.occ), ext1_P(s.w8ic), ext1_P(s.w8cnt), i32(1 if ext1_ST['T'] == 2 else 0)), 128)
            else:
                ok(s, 'w8_icell', s.capQc, (ext1_P(s.IP), ext1_P(s.cb), ext1_P(s.Aacc), ext1_P(s.w8ic), ext1_P(s.w8cnt)), 128)
            ok(s, 'w8_class', s.capN, (ext1_P(s.IP), ext1_P(s.mn), ext1_P(s.w8ic), i32(ext1_ST['D']), ext1_P(s.w8b), ext1_P(s.w8w), ext1_P(s.w8cnt), ext1_P(s.w8cf), i32(ext1_ST['X'])), 256)
            ok(s, 'w8_free', s.capN, (ext1_P(s.IP), ext1_P(s.w8b), ext1_P(s.sk), ext1_P(s.pm), ext1_P(s.yh), ext1_P(s.y), ext1_P(s.eta)), 256)
            ext1_ST['nclass'] += 1
            _log(s, ok)
            if ext1_ST['ck'] and (not torch.cuda.is_current_stream_capturing()):
                s._x13y0 = s.y.clone()
            if not ext1_ST['said']:
                ext1_ST['said'] = True
                print(f"[절 CM83 ext1] 옛 실행기 확장 훅 -- 차수 {ext1_ST['D']} 내부 칸 정의 {ext1_ST['T']} 고립 {ext1_ST['X']}, 걸음 시작 분류·변수 바꿈 항등, rt_grad 접기 + |g|² 다시{(' + 야코비 z·p 다시' if getattr(s, 'nl', 1) <= 1 else '')}, rt_pinit·rt_cg2 펼치기, rt_ap 접기, 첫 추정 d0 펼치기·ao_warm 접기, 노드별 보폭 (ws_cand·ws_merge·ws_lsup) 외부 노드 = 주인 노드 실제 증분 (MG 층 {getattr(s, 'nl', 1)}, 압축 {bool(getattr(s, 'compact', False))}, 검사 {ext1_ST['ck']})", flush=True)
                ext1__dump()
        return r
    (C.alloc, C.k, C.ctl) = (alloc, k, ctl)

def ext1_install(out, D=2, T=1, X=0, ck=0):
    ext1_ST.update(D=int(D), T=int(T), X=int(X), ck=int(ck), out=out)
    _install_an(out, ext1_ST['D'], ext1_ST['T'], ext1_ST['X'])
    _install_ao()
    print(f"[절 CM83] ext1 켬 -- 확장 B-스플라인 차수 {ext1_ST['D']} 내부 칸 정의 {ext1_ST['T']} (새 실행기 = fem_wolfh8 훅, 옛 실행기 = fem_ext1 훅), 검사 {ext1_ST['ck']}", flush=True)
    atexit.register(_fin)

def _fin():
    rows = []
    for wr in _SOL:
        s = wr()
        if s is None or getattr(s, 'x13H', None) is None:
            continue
        try:
            torch.cuda.synchronize()
            n = min(int(s.x13hp.item()), HCAP)
            H = s.x13H[:n].cpu().tolist()
            HF = s.x13HF[:n].cpu().tolist()
            for i in range(n):
                (c, m) = (H[i], HF[i])
                mt = sum(m[:3])
                rows.append(dict(seq=i, K=c[13], ks=c[12], icell=c[3], ncell=c[14], nn=c[15], inner=c[0], outer=c[1], iso=c[2], ax1=c[5], ax2=c[6], ax3=c[7], xiso=c[8], m_inner=m[0], m_outer=m[1], m_iso=m[2], m_xiso=m[3], mf_outer=m[1] / mt if mt > 0 else None, mf_iso=m[2] / mt if mt > 0 else None))
        except Exception as e_:
            ext1_ST['err'] = f'fin {type(e_).__name__}: {e_}'
    try:
        with open(os.path.join(ext1_ST['out'], 'w8.jsonl'), 'w') as f_:
            for q in rows:
                f_.write(json.dumps(q) + '\n')
    except Exception as e_:
        ext1_ST['err'] = f'w8.jsonl {type(e_).__name__}: {e_}'
    ext1_ST['nrows'] = len(rows)
    ext1__dump()

def ext1__dump():
    try:
        with open(os.path.join(ext1_ST['out'], 'x1info.json'), 'w') as f_:
            json.dump({k: v for (k, v) in ext1_ST.items()}, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
(NCK, NINV) = (24, 400)

def ext1__skip(s):
    raise NotImplementedError('not used by the paper settings')

def ext1__T(v, sk, pm, nr):
    raise NotImplementedError('not used by the paper settings')

def ext1__rel(a, b):
    raise NotImplementedError('not used by the paper settings')

def _check_iter(s, ok):
    raise NotImplementedError('not used by the paper settings')
ext1__BK = ('y', 'eta', 'f', 'B', 'H', 'r', 'z', 'p', 'dl', 'Tp', 'Dinv', 'g0buf', 'SC', 'bGn', 'bKn', 'bE', 'qb', 'cnt')

def _check_inv(s, where):
    raise NotImplementedError('not used by the paper settings')

def _check_mg(s):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_lightg3 ----------------
_lightg3_env_done = [False]

def lightg3__env_init(first=True):
    global GL, SC, lightg3_FR, R0, DG, CK, FD
    if first and _lightg3_env_done[0]:
        return
    GL = float(os.environ.get('AF_L3_G', '0'))
    SC = int(os.environ.get('AF_L3_SC', '0'))
    lightg3_FR = int(os.environ.get('AF_L3_FR', '1'))
    R0 = int(os.environ.get('AF_L3_R0', '0'))
    DG = int(os.environ.get('AF_L3_DG', '0'))
    CK = int(os.environ.get('AF_L3_CK', '0'))
    FD = {int(q) for q in os.environ.get('AF_L3_FD', '').split(',') if q.strip()}
    if first:
        _lightg3_env_done[0] = True
lightg3__env_init(first=False)
lightg3_FLOORN = ('ao_floor_n', 'rt_floor_n', 'mt_floor_n')
PEN = ('gq_force', 'gq_mv', 'gq_ls', 'gq_phi')
lightg3_ST = dict(on=None, log=None, out='.', nw=0, nw0=0, said=False, pre=False, npre=0, npre_tot=0, nfl=0, ljprev=None)
FRN = {0: 'λ_free (절 CM71)', 1: '(A) 원래 고정 판정에서 자유', 2: '(B) (A) + 장벽 받침 표시 낀 면 뺌'}
LG_SRC = cu('LG_SRC')

def lightg3__append(src):
    if 'gq_l3w(' in src:
        return src
    assert 'gq_base(' in src and 'gq_en(' in src, 'lightg3: ghost6 소스 (gq_base·gq_en) 가 먼저 붙어야 함'
    i = src.rstrip().rfind('}')
    return src[:i] + LG_SRC + '\n}\n'

def lightg3_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def lightg3__on(s):
    return float(getattr(s, 'mthr', 0.0) or 0.0) > 0.0 and GL > 0.0

def lightg3__alloc(s):
    cap = int(s.capQc)
    if getattr(s, 'l3thk', None) is None or s.l3thk.numel() < cap:
        s.l3thk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.l3l0 = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3lA = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3bt = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3la = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3mi = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3id = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        return True
    return False

def ratio():
    G6 = sys.modules[__name__]
    G6.ghost6__env_init()
    return GL / float(G6.GAMMA)

def reff():
    return 0.0 if R0 else ratio()

def thick(s, ok):
    lightg3__alloc(s)
    ok(s, 'gq_l3thk', int(s.capQc), (lightg3_P(s.IP), lightg3_P(s.cb), lightg3_P(s.cidx), lightg3_P(s.g3cut), lightg3_P(s.l3thk)), 128)

def weights(s, ok, pm):
    if lightg3__alloc(s):
        thick(s, ok)
    cq = int(s.capQc)
    bf = getattr(s, 'mfbf', None)
    assert lightg3_FR != 2 or bf is not None, 'lightg3 (B): 절 CM67 (2) 표시 버퍼 (s.mfbf) 없음 -- @mf2 1 이 켜져 있어야 함'
    ok(s, 'gq_l3w', cq, (lightg3_P(s.IP), lightg3_P(s.cb), lightg3_P(s.cidx), lightg3_P(s.g3fm), lightg3_P(s.l3thk), lightg3_P(s.mn), lightg3_P(pm), lightg3_P(bf), np.float32(s.mthr), np.float32(reff()), np.int32(SC), np.int32(lightg3_FR), lightg3_P(s.g5w), lightg3_P(s.l3l0), lightg3_P(s.l3lA), lightg3_P(s.l3bt), lightg3_P(s.l3la), lightg3_P(s.l3mi), lightg3_P(s.l3id)), 128)
    if pm is None:
        lightg3_ST['nw0'] += 1
    else:
        lightg3_ST['nw'] += 1
        if (DG or FD) and bf is not None:
            if getattr(s, 'l3bf', None) is None or s.l3bf.numel() < bf.numel():
                s.l3bf = torch.zeros_like(bf)
            s.l3bf[:bf.numel()].copy_(bf)

def _npen(G6):
    return sum((int(G6.ghost6_ST['nk'].get(q, 0)) for q in PEN))

def lightg3_install(out):
    import hexfem as ART
    import hexfem as V0
    import hexfem as AORT
    import hexfem as WRX
    G6 = sys.modules[__name__]
    G6.ghost6__env_init()
    assert G6.ghost6_ST['on'], 'lightg3: ghost6 (P1X) 가 켜져 있어야 함'
    assert G6.THIN == 1 and G6.WMODE == 0 and (not G6.CONS), ('lightg3: 바탕은 P1X (면 방식 1, 가중 없음, 증분 판)', G6.THIN, G6.WMODE, G6.CONS)
    lightg3_ST.update(out=out)
    if DG or CK:
        lightg3_ST['log'] = open(os.path.join(out, 'lg3.jsonl'), 'w')
    (o_b, o_s) = (ART.build_src, AORT._src)
    ART.build_src = lambda *a, **kw: lightg3__append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: lightg3__append(o_s(*a, **kw))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name.startswith('gq_') or getattr(s, '_l3in', False):
                return _ok(s, name, n, args, b)
            if name == 'rt_gauss' and lightg3_ST['on'] is None:
                lightg3_ST['on'] = lightg3__on(s)
                if lightg3_ST['on']:
                    G6.THIN = 0
                    G6.WMODE = 99
                    assert getattr(s, 'pm', None) is not None, 'lightg3: s.pm 없음'
                print(f"[절 CM73 lightg3] 씬 결정: m̄ {float(getattr(s, 'mthr', 0.0) or 0.0):.4g}, γ̄_L {GL:g} (γ̄ {G6.GAMMA:g}, r {ratio():g}, 쓰는 r {reff():g}), 범위 {('P1X 밖만' if SC else '전부')}, 정의 {FRN[lightg3_FR]} -> {('후보 면 전부 + 가벼움 가중' if lightg3_ST['on'] else '아무것도 안 함 (P1X 그대로)')}", flush=True)
            r = _ok(s, name, n, args, b)
            if lightg3_ST['on'] and (name == 'rt_vn' or name in lightg3_FLOORN):
                s._l3in = True
                try:
                    if name == 'rt_vn':
                        thick(s, _ok)
                        weights(s, _ok, None)
                        lightg3_ST['pre'] = True
                        lightg3_ST['pk0'] = _npen(G6)
                    else:
                        if lightg3_ST['pre']:
                            lightg3_ST['npre'] += _npen(G6) - lightg3_ST['pk0']
                        weights(s, _ok, s.pm)
                        lightg3_ST['pre'] = False
                        lightg3_ST['nfl'] += 1
                finally:
                    s._l3in = False
            return r
        cls.k = k
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_k(cls)
    dep = dict(d=0)

    def outer(fn):

        def w(s, t, h, *a, **kw):
            dep['d'] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep['d'] -= 1
            if dep['d'] == 0 and lightg3_ST['on']:
                fi = int(round(t / h)) + 1
                try:
                    if DG or CK:
                        diag(s, fi)
                    if fi in FD:
                        fdcheck(s, fi, out)
                except Exception:
                    import traceback
                    with open(os.path.join(out, 'lg3_err.txt'), 'a') as f_:
                        f_.write(f'f{fi}\n' + traceback.format_exc()[-3000:] + '\n')
                lightg3_ST['npre_tot'] += lightg3_ST['npre']
                lightg3_ST['npre'] = 0
            return res
        return w
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, 'an_rt_split_step'):
        ART.an_rt_split_step = outer(ART.an_rt_split_step)
    print(f"[절 CM73] lightg3 켬 -- 면 가중 1[P1X] + (γ̄_L/γ̄) λ, 정의 {lightg3_FR} {FRN[lightg3_FR]}, γ̄_L {GL:g}, 범위 {SC} ({('P1X 밖만' if SC else '전부')}), 진단 전용 {R0}, 진단 {DG}, 금 면 진단 {CK}, 검사 프레임 {sorted(FD)}", flush=True)

def _q(v, qs=(0.1, 0.5, 0.9)):
    raise NotImplementedError('not used by the paper settings')

@torch.no_grad()
def diag(s, fi):
    raise NotImplementedError('not used by the paper settings')

def fdcheck(s, fi, out):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_lightg4 ----------------
lightg4_ST = dict(kap=None, info=None, out='.')
MATN = {None: '탄성 (재질 판 아님)', 0: 'jelly (PG 탄성)', 1: 'metal (PG von Mises)', 2: 'sand (PG Drucker-Prager)', 3: 'foam', 5: 'plasticine (PG von Mises + 손상)', 7: 'watermelon (GF NACC)'}

def _matno(s):
    m = getattr(s, '_mat', None)
    if m is None:
        mm = getattr(s, 'mat', None)
        m = mm if isinstance(mm, int) else None
    return m

def kappa_of(s):
    m = _matno(s)
    inf = float('inf')
    if m is None or m == 0:
        return (1.0, dict(mat=m, name=MATN.get(m), rule='항복 없음 -> σ ↦ −σ 대칭', st='∞', sc='∞'))
    if m in (1, 3, 5):
        ys = None
        if getattr(s, 'ysp', None) is not None and s.ysp.numel():
            ys = float(s.ysp[0])
        return (1.0, dict(mat=m, name=MATN.get(m), rule='von Mises: 압력 무관 항복 -> 대칭', st='∞ (정수압)', sc='∞ (정수압)', yield_stress=ys))
    if m == 2:
        return (0.0, dict(mat=m, name=MATN.get(m), rule='DP 응집력 c = 0 (PG 반환 사상에 응집 항 없음): σ_t = c cot φ = 0, σ_c = ∞', st=0.0, sc='∞', cdp=float(getattr(s, 'cdp', float('nan')))))
    if m == 7:
        wf = getattr(s, 'wf', None) or {}
        if wf.get('seedb') or '--seedb' in sys.argv:
            raise NotImplementedError('lightg4: 씨 영역 β (칸마다 다른 κ) 는 아직 없음')
        prm = s.prm.detach().double().cpu()
        (beta, kb, xi) = (float(prm[5]), float(prm[2]), float(prm[6]))
        lj0 = float(s.ljp[0]) if getattr(s, 'ljp', None) is not None and s.ljp.numel() else -0.04
        p0 = kb * (1e-05 + math.sinh(xi * max(-lj0, 0.0)))
        return (2.0 * beta / (1.0 + beta), dict(mat=m, name=MATN.get(m), rule='NACC 꼭짓점: σ_c = p0, σ_t = β p0 -> κ = 2β/(1+β)', beta=beta, p0=p0, st=beta * p0, sc=p0, logJp0=lj0, xi=xi, bulk=kb))
    raise NotImplementedError(f'lightg4: 재질 {m} 의 κ 정의 없음')

def lightg4_install(out):
    L3 = sys.modules[__name__]
    L3.lightg3__env_init()
    assert L3.GL > 0.0, 'lightg4: lightg3 (--l3 γ̄_L) 가 켜져 있어야 함'
    lightg4_ST['out'] = out
    (o_on, o_reff) = (L3.lightg3__on, L3.reff)

    def _on(s):
        (k, info) = kappa_of(s)
        lightg4_ST.update(kap=float(k), info=info)
        base = bool(o_on(s))
        on = base and k > 0.0
        info.update(kappa=float(k), mbar=float(getattr(s, 'mthr', 0.0) or 0.0), lightg3_on=base, on=on, r=L3.ratio(), r_used=o_reff() * k)
        with open(os.path.join(out, 'l4info.json'), 'w') as f_:
            json.dump(info, f_, ensure_ascii=False, default=str)
        print(f"[절 CM75 lightg4] 재질 {info.get('name')} -> κ {k:g} ({info.get('rule')}); m̄ > 0 {base} -> {('가중 r κ λ (r ' + format(o_reff(), 'g') + ' x κ ' + format(k, 'g') + ')' if on else '아무것도 안 함 (P1X 그대로)')}", flush=True)
        return on

    def reff():
        r = o_reff()
        return r * lightg4_ST['kap'] if lightg4_ST['kap'] is not None else r
    L3.lightg3__on = _on
    L3.reff = reff
    print(f'[절 CM75] lightg4 켬 -- 가벼움 몫에 재질 응집 몫 κ = 2σ_t/(σ_t + σ_c) 곱 (정의 {L3.lightg3_FR} {L3.FRN[L3.lightg3_FR]}, γ̄_L {L3.GL:g}, 범위 {L3.SC})', flush=True)

# ---------------- fem_ext2 ----------------
import types
ext2_ST = dict(out='.', dec=[], nalloc_ext=0, nalloc_base=0, D=2, T=1, X=0)

def _kap(s):
    L4 = sys.modules[__name__]
    k = s.__dict__.get('_x2k')
    if k is not None:
        return k
    src = s
    if not hasattr(s, '_mat') and 'mat' not in s.__dict__ and ('_x2kw' in s.__dict__):
        src = types.SimpleNamespace(**s.__dict__['_x2kw'])
    (k, info) = L4.kappa_of(src)
    k = float(k)
    s._x2k = k
    info = dict(info, kappa=k, solver=type(s).__name__, ext=k == 0.0, from_init_args=src is not s)
    ext2_ST['dec'].append(info)
    ext2__dump()
    print(f"[절 CM84 ext2] {type(s).__name__} 재질 {info.get('name')} -> κ {k:g} ({info.get('rule')}) -> {('확장 T1E2 켬 + P1X 강성 × κ' if k == 0.0 else '확장 안 함 (H5 그대로, P1X 강성 × κ)')}", flush=True)
    return k

def ext2_install(out, D=2, T=1, X=0, ck=0):
    import hexfem as ART
    import hexfem as AORT
    X1 = sys.modules[__name__]
    ext2_ST.update(out=out, D=int(D), T=int(T), X=int(X))
    (R, C) = (ART.an_rt_RTSolver, AORT.AORTSolver)
    (a0R, a0C) = (R.alloc, C.alloc)
    X1.ext1_install(out, D=D, T=T, X=X, ck=ck)
    (a1R, a1C) = (R.alloc, C.alloc)
    oiC = C.__init__

    def initC(s, *a, **kw):
        s._x2kw = dict(mat=int(kw.get('mat', 2)), prm=kw.get('prm'), wf=dict(kw.get('wf') or {}))
        return oiC(s, *a, **kw)

    def allocR(s, need):
        if _kap(s) == 0.0:
            ext2_ST['nalloc_ext'] += 1
            return a1R(s, need)
        ext2_ST['nalloc_base'] += 1
        return a0R(s, need)

    def allocC(s, need):
        return (a1C if _kap(s) == 0.0 else a0C)(s, need)
    (R.alloc, C.alloc, C.__init__) = (allocR, allocC, initC)
    print(f'[절 CM84] ext2 켬 -- H7: 노드 κ (받침 칸 κ 최솟값) 0 이면 확장 T1E2 (차수 {D} 내부 칸 정의 {T}) + P1X 강성 × κ, 아니면 H5 그대로', flush=True)
    import atexit
    atexit.register(ext2__dump)

def ext2__dump():
    try:
        with open(os.path.join(ext2_ST['out'], 'x2info.json'), 'w') as f_:
            json.dump(ext2_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

# ---------------- fem_hhtc2 ----------------
MODE = 0
hhtc2_ST = dict(on=False, said=False, out='.', log=None, nsrc=0, nhc=0, nbar=0)
(NV, MR) = (20, 32)
HC_SRC = cu('HC_SRC')
SIG_OLD = 'int um, const int* ulm) {'
SIG_NEW = 'int um0, const int* ulm, const int* hcm, int hk) {\n    int um = um0;                                                              // 절 CM58: 노드마다 갱신 방식'
UL_LINE = '    uf_ = uf_ && ulm[n];'
HC_LINE = '    if (hcm[n] && !uf_) { uf_ = 1; um = hk; }                                    // 절 CM58: 표시 노드 = 후퇴 오일러 (hk 1) 또는 가속도 기억 0 (hk 2)\n'
PAIRS = (('rt_ga_end_ufl', 'rt_ga_end_hc'), ('ao_stepend_ufl', 'ao_stepend_hc'))

def hcify(src):
    add = []
    for (nm, nh) in PAIRS:
        key = '__global__ void ' + nm + '('
        if key not in src:
            continue
        i = src.index(key)
        j = src.index('\n}\n', i) + 3
        k = src[i:j].replace(nm + '(', nh + '(', 1)
        assert k.count(SIG_OLD) == 1 and k.count(UL_LINE) == 1, ('절 CM58 hcify', nm)
        k = k.replace(SIG_OLD, SIG_NEW, 1)
        e = k.index('\n', k.index(UL_LINE)) + 1
        k = k[:e] + HC_LINE + k[e:]
        add.append(k)
        hhtc2_ST['nhc'] += 1
    if not add:
        return src
    if '__device__ float* AF_BX;' in src:
        add.append(HC_SRC)
        hhtc2_ST['nbar'] += 1
    hhtc2_ST['nsrc'] += 1
    s = src.rstrip()
    i = s.rfind('}')
    return s[:i] + '\n'.join(add) + '\n}\n'
hhtc2_P = lambda t: np.uint64(0 if t is None else t.data_ptr())

def _buf(s, n):
    if getattr(s, '_hcst', None) is None:
        assert not torch.cuda.is_current_stream_capturing(), '절 CM58 버퍼는 데움 record 에서'
        s._hcst = torch.zeros(12, device=s.dev, dtype=torch.int32)
        s._hcfa = torch.zeros(2, device=s.dev, dtype=torch.float64)
        s._hcmo = torch.zeros(MR * NV, device=s.dev, dtype=torch.float64)
    if getattr(s, '_hcm', None) is None or s._hcm.numel() < n:
        assert not torch.cuda.is_current_stream_capturing(), '절 CM58 노드 버퍼는 데움 record 에서'
        if getattr(s, '_hcm', None) is not None:
            s._hckeep = getattr(s, '_hckeep', []) + [s._hcm, s._hcms, s._hcme]
        m = int(n * 1.25) + 1024
        s._hcm = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcms = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcme = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcdum = torch.zeros(m, device=s.dev, dtype=torch.int32)

def hhtc2__wrap_k(cls, nm, nh):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name == 'rt_bar_q' and len(args) >= 12:
            s._hcbq = (int(n), tuple(args))
        if name == nm and getattr(s, '_hcbq', None) is not None:
            capN = int(n)
            _buf(s, capN)
            (nq, ba) = s._hcbq
            IP = args[0]
            (y, U) = (args[5], args[7])
            i32 = np.int32
            ok(s, 'hc_zero', capN, (IP, hhtc2_P(s._hcms), hhtc2_P(s._hcme), hhtc2_P(s._hcst)), 256)
            ok(s, 'hc_bar', nq, (IP, ba[1], ba[2], y, U, ba[6], ba[7], ba[8], ba[9], ba[10], ba[11], hhtc2_P(s._hcms), hhtc2_P(s._hcme)), 128)
            ok(s, 'hc_fin', capN, (IP, hhtc2_P(s._hcms), hhtc2_P(s._hcme), i32(MODE), hhtc2_P(s._hcm), hhtc2_P(s._hcst)), 256)
            if MODE == 1:
                ok(s, 'hc_all', capN, (IP, hhtc2_P(s._hcst), hhtc2_P(s._hcm)), 256)
            if MODE in (0, 4, 5, 6):
                ok(s, 'hc_fz', 1, (hhtc2_P(s._hcfa),), 1)
                ok(s, 'hc_flip', capN, (IP, hhtc2_P(s._hcms), hhtc2_P(s._hcme), y, args[6], args[8], args[4], ba[7], ba[8], ba[9], hhtc2_P(s._hcfa)), 256)
            ok(s, 'hc_mz', 1, (hhtc2_P(s._hcst), hhtc2_P(s._hcmo)), 32)
            ok(s, 'hc_mom', capN, (IP, hhtc2_P(s._hcms), hhtc2_P(s._hcme), hhtc2_P(s.mn), y, args[6], U, args[8], args[9], args[1], args[2], args[3], args[4], ba[7], ba[8], ba[9], hhtc2_P(s._hcst), hhtc2_P(s._hcmo)), 256)
            if MODE in (0, 4):
                ok(s, 'hc_all4', capN, (IP, hhtc2_P(s._hcst), hhtc2_P(s._hcfa), hhtc2_P(s._hcm) if MODE == 4 else hhtc2_P(s._hcdum)), 256)
            if MODE in (0, 4, 5, 6):
                ok(s, 'hc_all5', capN, (IP, hhtc2_P(s._hcst), hhtc2_P(s._hcfa), hhtc2_P(s._hcmo), i32(MODE), hhtc2_P(s._hcm) if MODE in (5, 6) else hhtc2_P(s._hcdum)), 256)
            ok(s, 'hc_cnt', 1, (hhtc2_P(s._hcst),), 1)
            s._hcn = getattr(s, '_hcn', 0) + 1
            if MODE in (1, 2, 3, 4, 5, 6):
                return ok(s, nh, n, tuple(args) + (hhtc2_P(s._hcm), i32(2 if MODE == 3 else 1)), b)
        return ok(s, name, n, args, b)
    cls.k = k

def _wrap_step(cls):
    o_s = cls.step

    def step(s, t, *a, **kw):
        d = getattr(s, '_hcdep', 0)
        s._hcdep = d + 1
        try:
            r = o_s(s, t, *a, **kw)
        finally:
            s._hcdep = d
        if d == 0 and getattr(s, '_hcst', None) is not None and (hhtc2_ST['log'] is not None):
            try:
                c = [int(q) for q in s._hcst.cpu().tolist()]
                mo = s._hcmo.view(MR, NV)[:min(c[4], MR)].cpu().tolist()
                s._hcst.zero_()
                hhtc2_ST['log'].write(json.dumps(dict(t=float(t), h=float(getattr(s, 'h', 0.0) or 0.0), K=int(getattr(s, 'Kcur', getattr(s, 'K', 1)) or 1), new=c[0], eng=c[1], ncol=c[3], nstep=c[4], napp=c[5], nflip=c[6], nf5=c[7], nf6=c[8], mo=mo, nlaunch=int(getattr(s, '_hcn', 0)), ms=float(getattr(s, 'last_ms', float('nan')) or float('nan')))) + '\n')
                hhtc2_ST['log'].flush()
            except Exception as e_:
                with open(os.path.join(hhtc2_ST['out'], 'hc_err.txt'), 'a') as f_:
                    f_.write(f'step 기록: {e_!r}\n')
        return r
    cls.step = step

def hhtc2_install(mode, out):
    global MODE
    MODE = int(mode)
    assert MODE in (0, 1, 2, 3, 4, 5, 6), MODE
    import hexfem as ART
    import hexfem as V0
    import hexfem as AORT
    hhtc2_ST.update(on=True, out=out)
    hhtc2_ST['log'] = open(os.path.join(out, 'hc.jsonl'), 'w')
    (o_b, o_s) = (ART.build_src, AORT._src)
    ART.build_src = lambda *a, **kw: hcify(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: hcify(o_s(*a, **kw))
    hhtc2__wrap_k(ART.an_rt_RTSolver, 'rt_ga_end_ufl', 'rt_ga_end_hc')
    hhtc2__wrap_k(AORT.AORTSolver, 'ao_stepend_ufl', 'ao_stepend_hc')
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        _wrap_step(cls)
    name = {0: '진단만 (결과 불변)', 1: 'H1 충돌 걸음 몸 전체 후퇴 오일러', 2: 'H2 새 접촉 노드 후퇴 오일러', 3: 'H3 걸음 끝 접촉 노드 가속도 기억 0', 4: 'H4 HHT 가 새 접촉 노드 법선 속도를 뒤집는 충돌 걸음만 몸 전체 후퇴 오일러', 5: 'H5 H4 걸음 중 HHT 끝 속도가 몸 질량중심 법선 운동량을 뒤집고 걸음 평균 운동량은 아직 들어가는 쪽인 걸음만 몸 전체 후퇴 오일러', 6: 'H5a H4 걸음 중 몸 질량중심 법선 운동량 뒤집힘 걸음만'}[MODE]
    print(f'[절 CM61] hhtc2 켬 -- 모드 {MODE}: {name}', flush=True)
    import atexit

    @atexit.register
    def _info():
        try:
            json.dump(dict(mode=MODE, nsrc=hhtc2_ST['nsrc'], nhc=hhtc2_ST['nhc'], nbar=hhtc2_ST['nbar']), open(os.path.join(out, 'hcinfo.json'), 'w'))
        except Exception:
            pass

# ---------------- fem_mfix ----------------
_mfix_env_done = [False]

def mfix__env_init(first=True):
    global C1, C2
    if first and _mfix_env_done[0]:
        return
    C1 = int(os.environ.get('AF_MF1', '0'))
    C2 = int(os.environ.get('AF_MF2', '0'))
    if first:
        _mfix_env_done[0] = True
mfix__env_init(first=False)
mfix_ST = dict(on=False, log=None, nset=0, nmark=0, said=False)
mfix_FLOORN = ('ao_floor_n', 'mt_floor_n')
PBCK = ('ao_pbc_k', 'mt_pbc_kphi')
MF_SRC = cu('MF_SRC')
MF_BAR_SRC = cu('MF_BAR_SRC')

def mfix__append(src):
    if 'mf_save(' in src:
        return src
    add = MF_SRC + (MF_BAR_SRC if '__device__ float* AF_BX;' in src else '')
    i = src.rstrip().rfind('}')
    return src[:i] + add + src[i:]

def mfix_install(out):
    import hexfem as ART
    import hexfem as V0
    import hexfem as AORT
    import hexfem as WRX
    mfix_ST.update(on=True)
    mfix_ST['log'] = open(os.path.join(out, 'mf.jsonl'), 'w')
    (o_b, o_s) = (ART.build_src, AORT._src)
    ART.build_src = lambda *a, **kw: mfix__append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: mfix__append(o_s(*a, **kw))
    P = ART.an_rt_P
    (f32, i32) = (np.float32, np.int32)

    def on(s):
        return float(getattr(s, 'mthr', 0.0) or 0.0) > 0.0 and (C1 or C2)

    def _alloc(s):
        cap = int(s.capN)
        if getattr(s, 'mfb', None) is None or s.mfb.numel() < cap:
            s.mfb = torch.zeros(cap, device=s.dev, dtype=torch.float32)
            s.mfx = torch.zeros(cap, device=s.dev, dtype=torch.float32)
            s.mfbf = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        if getattr(s, 'mfcnt', None) is None:
            s.mfcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)

    def mark(s, ok, y, U, thv, nr, pn):
        dh = None
        wb = getattr(s, 'wbar', None)
        if isinstance(wb, dict) and 'DH' in wb:
            dh = float(wb['DH'])
        elif getattr(s, 'bar', False) and s.floor:
            dh = float(s.floor['DH'])
        ok(s, 'mf_zero', int(s.capN), (P(s.IP), P(s.mfbf)), 256)
        if dh is None or not C2:
            return
        capQ = int(s.capQc) * 8
        ok(s, 'mf_mark', capQ, (P(s.IP), P(s.cb), P(s.BW), y, U, thv, *nr, pn, f32(dh), P(s.mfbf), P(s.mfcnt)), 128)
        mfix_ST['nmark'] += 1

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if getattr(s, '_mfin', False) or name.startswith('mf_') or (not on(s)):
                return _ok(s, name, n, args, b)
            s._mfin = True
            try:
                _alloc(s)
                if name in mfix_FLOORN:
                    ok_ = _ok
                    ok_(s, 'mf_set', int(s.capN), (P(s.IP), P(s.mfb), P(s.mn), P(s.mfx), P(s.mfbf), f32(s.mthr), f32(s.mthr), i32(C1), i32(C2)), 256)
                    mfix_ST['nset'] += 1
                    a = list(args)
                    pm_ = P(s.mn)
                    ix = [i for i in range(len(a) - 1) if isinstance(a[i], np.uint64) and a[i] == pm_ and isinstance(a[i + 1], np.float32)]
                    assert len(ix) == 1, ('mfix: floor_n 질량 인자', name, ix)
                    a[ix[0]] = P(s.mfx)
                    args = tuple(a)
                elif name in PBCK and C1:
                    a = list(args)
                    assert isinstance(a[1], np.uint64) and a[1] == P(s.mn), ('mfix: pbc 질량 인자', name)
                    a[1] = P(s.mfb)
                    args = tuple(a)
            finally:
                s._mfin = False
            r = _ok(s, name, n, args, b)
            s._mfin = True
            try:
                if name == 'rt_vn':
                    _ok(s, 'mf_save', int(s.capN), (P(s.IP), P(s.mn), P(s.mfb)), 256)
                elif name == 'wb_floor_q':
                    mark(s, _ok, args[4], args[5], args[6], args[7:10], args[10])
                elif name == 'rt_floor_q' and float(args[15]) > 0:
                    mark(s, _ok, args[5], args[6], args[7], args[11:14], args[14])
            finally:
                s._mfin = False
            return r
        cls.k = k
    for cls in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
        wrap_k(cls)
    dep = dict(d=0)

    def outer(fn):

        def w(s, t, h, *a, **kw):
            dep['d'] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep['d'] -= 1
            if dep['d'] == 0:
                try:
                    post(s, int(round(t / h)) + 1)
                except Exception:
                    import traceback
                    with open(os.path.join(out, 'mf_err.txt'), 'a') as f_:
                        f_.write(traceback.format_exc()[-3000:] + '\n')
            return res
        return w
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, 'an_rt_split_step'):
        ART.an_rt_split_step = outer(ART.an_rt_split_step)
    print(f'[절 CM67] mfix 켬 -- (1) 연속 고정 (관성 덧질량 m̄ = mthr) {C1} · (2) 장벽 표본점 27 노드 고정 빼기 {C2}', flush=True)

@torch.no_grad()
def post(s, fi):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_startpt ----------------
from concurrent.futures import ThreadPoolExecutor as _TPE
import atexit as _ax
_SPW = _TPE(max_workers=1)
_ax.register(lambda : _SPW.shutdown(wait=True))
INFO = dict(n_snap=0, n_snapdrop=0, meta={}, n_inst=0, n_alloc=0, n_wrapfn=0, n_win=0, n_delta=0, n_fin=0, n_rec=0, n_line=0, err=None, names=[], over=[], skipptr=None, cls=None)
OUT = ['.']
KS = 4
EV = 4
NS = 48
KSG = 8
startpt_FR = [3, 12]
startpt_LOG = [None]
NBK = 8
startpt_ELEM = ('rt_elem', 'aq_elem', 'ao_elem', 'ao_elem_s')
startpt_SRC = cu('startpt_SRC')

def startpt_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def startpt__dump():
    try:
        with open(os.path.join(OUT[0], 'spinfo.json'), 'w') as f_:
            json.dump(INFO, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

def _bucket(name):
    nm = INFO['names']
    if name in nm:
        return nm.index(name)
    if len(nm) < NBK - 1:
        nm.append(name)
        return len(nm) - 1
    if len(nm) == NBK - 1:
        nm.append('기타')
    if name not in INFO['over']:
        INFO['over'].append(name)
    return NBK - 1

class Buf:

    def __init__(s):
        import cupy as cp
        s.mod = cp.RawModule(code=startpt_SRC)
        s.fn = {n: s.mod.get_function(n) for n in ('cf_reset', 'cf_delta', 'cf_fin', 'cf_g0', 'cf_node', 'cf_cp', 'cf_frame', 'sp_snap', 'sp_snap1', 'sp_clr', 'sp_fin', 'sp_g0z', 'sp_g2', 'sp_gclr', 'sp_gcnt')}
        s.capN = 0
        s.kstep = -1
        s.evi = 0
        s.win = False
        s.skip = None
        s.ns = 0
        s.phase = 0
        s.lsit = 0

    def alloc(s, sv):
        capN = int(sv.capN)
        if capN == s.capN:
            return
        dev = sv.dev
        s.capN = capN
        s.prev = torch.zeros(capN, 3, device=dev, dtype=torch.float64)
        s.acc = torch.zeros(KS * EV + 1, NBK, capN, device=dev, dtype=torch.float64)
        s.en = torch.zeros(KS * EV, capN, device=dev, dtype=torch.float32)
        s.gn = torch.zeros(KS * EV, capN, device=dev, dtype=torch.float32)
        s.sk = torch.zeros(KS, capN, device=dev, dtype=torch.float32)
        s.pm = torch.zeros(KS, capN, device=dev, dtype=torch.int32)
        s.me = torch.zeros(KS, capN, device=dev, dtype=torch.float32)
        s.w8 = torch.full((KS, capN), -1, device=dev, dtype=torch.int32)
        s.itb = torch.full((KS * EV,), -1, device=dev, dtype=torch.int32)
        s.itf = torch.zeros(KS * EV, device=dev, dtype=torch.float32)
        s.sn = torch.zeros(KS, NS, capN, device=dev, dtype=torch.float32)
        s.flg = torch.zeros(KS * NS, device=dev, dtype=torch.int32)
        s.gs = torch.full((KSG * 2,), -1.0, device=dev, dtype=torch.float64)
        s.gc = torch.zeros(KSG * 2, device=dev, dtype=torch.int32)
        INFO['n_alloc'] += 1

    def go(s, name, n, args, b=256):
        import hexfem as RTC
        with RTC._Stream():
            s.fn[name](((int(n) + b - 1) // b,), (b,), tuple(args))

def _fl(sv):
    FL = sv.floor or {}
    nr = FL.get('NR', [0.0, 0.0, 1.0])
    return ([float(q) for q in nr], float(FL.get('PN', 0.0)))

def startpt__skip(sv, W):
    if W.skip is None:
        try:
            W.skip = np.uint64(int(sv.mod.get_global('AF_SKIP')))
        except Exception:
            W.skip = np.uint64(0)
        INFO['skipptr'] = bool(int(W.skip))
    return W.skip

def _has_ptr(args, p):
    for a in args:
        if isinstance(a, (np.integer, int)) and (not isinstance(a, bool)) and (int(a) == p):
            return True
    return False

class _Call:

    def __init__(s_, sv, name, f):
        (s_.sv, s_.name, s_.f) = (sv, name, f)

    def __call__(s_, grid, block, args, **kw):
        (sv, name, f) = (s_.sv, s_.name, s_.f)
        W = getattr(sv, '_cf', None)
        if W is None or not W.capN:
            return f(grid, block, args, **kw)
        post = None
        try:
            post = _pre(sv, W, name, args)
        except Exception as e:
            INFO['err'] = f'pre {name}: {e!r}'[:400]
        r = f(grid, block, args, **kw)
        if post is not None:
            try:
                post()
            except Exception as e:
                INFO['err'] = f'post {name}: {e!r}'[:400]
        return r

def _snap(sv, W, S, NR, ptr, tag):
    k = W.kstep
    if k < 0 or k >= KS:
        return
    if W.ns >= NS:
        INFO['n_snapdrop'] += 1
        return
    j = W.ns
    W.ns += 1
    W.go('sp_snap', W.capN, (startpt_P(sv.IP), ptr, *NR, startpt_P(W.sn[k, j]), startpt_P(W.flg), np.int32(k * NS + j), S))
    INFO['meta'][f'{k}:{j}'] = tag
    INFO['n_snap'] += 1

def _snap1(sv, W, S, ptr, tag):
    k = W.kstep
    if k < 0 or k >= KS or W.ns >= NS:
        return
    j = W.ns
    W.ns += 1
    W.go('sp_snap1', W.capN, (startpt_P(sv.IP), ptr, startpt_P(W.sn[k, j]), startpt_P(W.flg), np.int32(k * NS + j), S))
    INFO['meta'][f'{k}:{j}'] = tag

def _pre(sv, W, name, args):
    (i32, f32) = (np.int32, np.float32)
    capN = W.capN
    if name == 'rt_reset':
        (W.kstep, W.evi, W.win, W.ns, W.phase) = (-1, 0, False, 0, 0)

        def _p():
            W.go('sp_clr', KS * NS, (startpt_P(W.flg), i32(KS * NS), startpt_P(W.itb), i32(KS * EV)), 256)
            W.go('sp_gclr', KSG * 2, (startpt_P(W.gs), i32(KSG * 2), startpt_P(W.gc)), 64)
        return _p
    S = startpt__skip(sv, W)
    (nr, pn) = _fl(sv)
    NR = [f32(q) for q in nr]
    if name == 'rt_ga_step0':
        if W.kstep + 1 < KS:
            W.kstep += 1
            W.ns = 0
            _snap(sv, W, S, NR, startpt_P(sv.U), ('U_이전끝', name, -1))
            _snap(sv, W, S, NR, args[2], ('v_시작', name, -1))
            W.kstep -= 1

        def _p():
            W.kstep += 1
            W.evi = 0
            W.win = False
            if W.kstep >= KS:
                W.ns = NS
            W.phase = 1
            W.lsit = 0
            _snap(sv, W, S, NR, startpt_P(sv.Ut), ('Ut_관성예측', name, -1))
            _snap(sv, W, S, NR, startpt_P(sv.y), ('y', name, -1))
        return _p
    if W.kstep < 0:
        return None
    fp = sv.f.data_ptr()
    yp = sv.y.data_ptr()
    if name == 'rt_zero_d' and len(args) >= 5 and (int(args[4]) == fp):

        def _p():
            W.go('cf_reset', capN, (startpt_P(sv.IP), i32(capN), i32(NBK), startpt_P(W.prev), startpt_P(W.acc[KS * EV]), S))
            W.win = True
            INFO['n_win'] += 1
        return _p
    if name.startswith('rt_grad') and W.win:
        IPp = args[0]
        if W.phase == 1:
            _snap(sv, W, S, NR, startpt_P(sv.y), ('y0_시작점', name, -1))
            _snap(sv, W, S, NR, startpt_P(sv.yh), ('yh_기준', name, -1))
            W.phase = 2
        W.go('cf_delta', capN, (IPp, startpt_P(sv.f), startpt_P(W.prev), startpt_P(W.acc[KS * EV, _bucket('~rt_grad')]), *NR, S))
        (k, e) = (W.kstep, W.evi)
        if k < KS and e < EV:
            sl = k * EV + e
            W.go('sp_fin', capN, (IPp, i32(capN), i32(NBK), startpt_P(W.acc[KS * EV]), startpt_P(W.acc[sl]), args[8], *NR, startpt_P(W.en[sl]), startpt_P(W.itb), i32(sl), i32(k * 100 + e), startpt_P(W.itf), f32(float(args[2])), S))
            if e == 0:
                w8b = getattr(sv, 'w8b', None)
                W.go('cf_node', capN, (IPp, args[3], args[4], args[1], startpt_P(w8b), startpt_P(W.sk[k]), startpt_P(W.pm[k]), startpt_P(W.me[k]), startpt_P(W.w8[k]), S))
        INFO['n_fin'] += 1

        def _p():
            if k < KS and e < EV:
                W.go('cf_g0', capN, (IPp, args[-1], *NR, startpt_P(W.gn[k * EV + e]), S))
            if 0 <= k < KSG:
                for idx in (2 * k, 2 * k + 1) if e == 0 else (2 * k + 1,):
                    W.go('sp_g0z', 1, (startpt_P(W.gs), i32(idx), S), 1)
                    W.go('sp_g2', capN, (IPp, args[-1], startpt_P(W.gs), i32(idx), S), 256)
                W.go('sp_gcnt', 1, (startpt_P(W.gc), i32(2 * k + 1), S), 1)
            W.evi += 1
            W.win = False
        return _p
    if W.win and _has_ptr(args, fp):
        IPp = startpt_P(sv.IP)
        bn = '탄성' if name in startpt_ELEM else name
        INFO['n_delta'] += 1
        return lambda : W.go('cf_delta', capN, (IPp, startpt_P(sv.f), startpt_P(W.prev), startpt_P(W.acc[KS * EV, _bucket(bn)]), *NR, S))
    if W.phase == 1 and name in ('ao_pbc_k', 'mt_pbc_kphi'):
        return lambda : _snap1(sv, W, S, args[7], ('kb_기본', name, -1))
    if W.phase == 1 and name == 'w5s_kb':
        return lambda : _snap1(sv, W, S, args[4], ('kb_S', name, -1))
    if W.phase == 1 and _has_ptr(args, yp):
        return lambda : _snap(sv, W, S, NR, startpt_P(sv.y), ('y', name, -1))
    if W.phase == 2 and name == 'rt_lsup':
        it = W.lsit
        W.lsit += 1
        return lambda : _snap(sv, W, S, NR, startpt_P(sv.y), ('y_갱신', name, it))
    return None

class _CFn(dict):

    def __init__(s_, inner, owner):
        super().__init__()
        (s_.inner, s_.owner) = (inner, owner)

    def __getitem__(s_, name):
        return _Call(s_.owner, name, s_.inner[name])

    def __setitem__(s_, k, v):
        s_.inner[k] = v

    def __contains__(s_, k):
        return k in s_.inner

    def get(s_, k, d=None):
        return s_.inner.get(k, d)

def _ensure_fn(s):
    if not isinstance(getattr(s, 'fn', None), _CFn):
        s.fn = _CFn(s.fn, s)
        INFO['n_wrapfn'] += 1

def _wrap(R):
    (ok, os_, oa) = (R.k, R.step, R.alloc)

    def alloc(s, need):
        r = oa(s, need)
        if getattr(s, '_cf', None) is None:
            s._cf = Buf()
            INFO['n_inst'] += 1
            INFO['cls'] = type(s).__name__
        s._cf.alloc(s)
        return r

    def k(s, name, n, args, b=128):
        if getattr(s, '_cf', None) is not None and hasattr(s, 'fn'):
            _ensure_fn(s)
        return ok(s, name, n, args, b)

    def step(s, t, *a, **kw):
        pend = getattr(s, '_cf_pend', None)
        if pend is not None and pend != t:
            _flush(s, pend)
            s._cf_pend = None
        r = os_(s, t, *a, **kw)
        if getattr(s, '_cf', None) is not None and s._cf.capN:
            s._cf_pend = t
            if not getattr(s, '_cf_atexit', False):
                import atexit
                s._cf_atexit = True
                atexit.register(lambda s_=s: (_flush(s_, getattr(s_, '_cf_pend', None)), startpt__dump()))
        return r
    (R.alloc, R.k, R.step) = (alloc, k, step)

def _flush(s, t):
    W = getattr(s, '_cf', None)
    if t is None or W is None or (not W.capN) or (startpt_LOG[0] is None):
        return
    try:
        torch.cuda.synchronize()
        if bool(torch.isfinite(s.x[:1]).all()):
            o = _analyze(s, W, t)
            startpt_LOG[0].write(json.dumps(o, ensure_ascii=False) + '\n')
            startpt_LOG[0].flush()
            INFO['n_line'] += 1
    except Exception as e:
        INFO['err'] = 'analyze: ' + repr(e)[:400]
    INFO['n_rec'] += 1
    s._cf_pend = None
    if INFO['n_rec'] in (1, 2, 5) or INFO['n_rec'] % 10 == 0:
        startpt__dump()

def _analyze(s, W, t):
    fr = int(round(float(t) / float(s.h))) + 1
    o = dict(fr=fr, err=INFO['err'])
    if not startpt_FR[0] <= fr <= startpt_FR[1]:
        o['skip'] = '프레임 밖'
        return o
    IP = s.IP[:16].cpu().numpy().astype(np.int64)
    nn = int(IP[9])
    nb = IP[6:9].astype(np.int64)
    bmin = IP[0:3].astype(np.int64)
    (nb1, nb2) = (int(nb[1]), int(nb[2]))
    dev = s.U.device
    (nr, pn) = _fl(s)
    n_ = torch.tensor(nr, device=dev, dtype=torch.float64)
    n_ = n_ / n_.norm()
    A = s.A.double().reshape(3, 3)
    dx = float(A[:, 0].norm())
    ids = torch.arange(nn, device=dev, dtype=torch.int64)
    ijk = torch.stack([ids // (nb1 * nb2), ids // nb2 % nb1, ids % nb2], 1) + torch.tensor(bmin, device=dev)
    h0 = ijk.double() @ A.T @ n_ - pn
    N = int(s.N)
    c = s.Np[:N].long() + (s.fx[:N] >= 1.0).long()
    OFF = torch.tensor([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.int64)
    nid = ((c[:, None, 0] + OFF[None, :, 0]) * nb1 + (c[:, None, 1] + OFF[None, :, 1])) * nb2 + (c[:, None, 2] + OFF[None, :, 2])
    okp = (nid >= 0).all(1) & (nid < nn).all(1)
    key = torch.where(okp, nid[:, 0], torch.full_like(nid[:, 0], -1))
    (uk, inv) = torch.unique(key, return_inverse=True)
    first = torch.full((len(uk),), N, device=dev, dtype=torch.int64).scatter_reduce(0, inv, torch.arange(N, device=dev), reduce='amin')
    nidc = nid[first[uk >= 0]]
    cmap = torch.cumsum((uk >= 0).long(), 0) - 1
    pcell = torch.where(uk[inv] >= 0, cmap[inv], torch.full_like(inv, -1))
    trel = getattr(s, 'trel', None)
    held = (trel[:N] > float(t)).float() if trel is not None else torch.zeros(N, device=dev)
    nc_ = int(nidc.shape[0])
    cnt = torch.zeros(nc_, device=dev).index_add_(0, pcell.clamp(min=0), (pcell >= 0).float())
    hf = torch.zeros(nc_, device=dev).index_add_(0, pcell.clamp(min=0), held * (pcell >= 0).float()) / cnt.clamp_min(1)
    a0 = h0[nidc]
    hs = a0.max(1).values - a0.min(1).values
    hb = a0.min(1).values
    lay = torch.clamp(torch.floor(hb / dx + 1e-06), min=0).long()
    order = torch.argsort(a0, dim=1)
    wsg = torch.empty_like(a0)
    wsg.scatter_(1, order[:, :4], -1.0 / math.sqrt(8.0))
    wsg.scatter_(1, order[:, 4:], 1.0 / math.sqrt(8.0))
    mn = s.mn[:nn].double()
    gz = float(sum((float(q) * float(p) for (q, p) in zip(s.g, n_.tolist()))))
    flg = W.flg.cpu().numpy().reshape(KS, NS)
    itb = W.itb.cpu().tolist()
    itf = W.itf.cpu().tolist()
    names = list(INFO['names'])
    nbu = len(names)

    def rr(v):
        a = a0 + v[nidc]
        return (a.max(1).values - a.min(1).values) / hs.clamp_min(1e-12)
    out = dict(pcell=pcell.int().cpu().numpy(), held=hf.half().cpu().numpy(), lay=lay.short().cpu().numpy(), hb=hb.float().cpu().numpy(), m=mn[nidc].sum(1).float().cpu().numpy(), dx=dx)
    steps = []
    for k in range(KS):
        valid = [j for j in range(NS) if flg[k, j]]
        if not valid:
            continue
        sk = W.sk[k, :nn].double()
        pm = W.pm[k, :nn]
        tn = torch.where(pm == 0, sk, torch.where(pm == 2, torch.zeros_like(sk), torch.ones_like(sk)))
        cons = tn.abs() < 0.001
        ccn = cons[nidc]
        out[f'k{k}_ncons'] = ccn.sum(1).short().cpu().numpy()
        out[f'k{k}_next'] = (W.w8[k, :nn][nidc] >= 0).sum(1).short().cpu().numpy()
        tags = []
        R = []
        RC = []
        RF = []
        prev = None
        snapv = {}
        kbs = {}
        for j in valid:
            tg = INFO['meta'].get(f'{k}:{j}')
            v = W.sn[k, j, :nn].double()
            if tg:
                snapv.setdefault(tg[0] + ':' + tg[1], v)
                snapv[tg[0] + ':' + tg[1] + ':last'] = v
            if tg and tg[0].startswith('kb_'):
                kbs[tg[0]] = W.sn[k, j, :nn].clone()
                continue
            if tg and tg[0].startswith('v_'):
                vt = (v[nidc] * (wsg > 0)).sum(1) / 4.0
                vb = (v[nidc] * (wsg < 0)).sum(1) / 4.0
                out[f'k{k}_vrate'] = ((vt - vb) / hs.clamp_min(1e-12)).float().cpu().numpy()
                continue
            tags.append([str(q) for q in tg] if tg else ['?', '?', -1])
            R.append(rr(v))
            if prev is None or tg[0].startswith('Ut') or tg[0].startswith('yh'):
                RC.append(torch.full_like(hs, float('nan')))
                RF.append(torch.full_like(hs, float('nan')))
            else:
                vC = torch.where(cons, v, prev)
                vF = torch.where(cons, prev, v)
                RC.append(rr(vC))
                RF.append(rr(vF))
            if not (tg and (tg[0].startswith('Ut') or tg[0].startswith('yh'))):
                prev = v
        tags = [t for t in tags if not t[0].startswith('v_')]
        act = mn > 0
        w8k = W.w8[k, :nn] >= 0
        kfin = getattr(s, 'kb', None)
        if kfin is not None:
            kbs['kb_최종'] = kfin[:nn].clone()
        Np_ = s.Np[:N].long()
        o27 = torch.tensor([(i, j, l) for i in range(3) for j in range(3) for l in range(3)], device=dev, dtype=torch.int64)
        nid27 = ((Np_[:, None, 0] + o27[None, :, 0]) * nb1 + (Np_[:, None, 1] + o27[None, :, 1])) * nb2 + (Np_[:, None, 2] + o27[None, :, 2])
        ok27 = (nid27 >= 0) & (nid27 < nn)
        cntn = torch.zeros(nn, device=dev).index_add_(0, nid27[ok27], torch.ones(int(ok27.sum()), device=dev))
        heldp = (trel[:N] > float(t)).float() if trel is not None else torch.zeros(N, device=dev)
        hn = torch.zeros(nn, device=dev).index_add_(0, nid27[ok27], heldp[:, None].expand(-1, 27)[ok27])
        alph = hn / cntn.clamp_min(1)
        front = act & (alph > 0) & (alph < 1)
        KBS = {}
        for (nmk, kv) in kbs.items():
            kv = kv.double()
            part = act & (kv > 0.001) & (kv <= 0.999)
            held_ = act & (kv <= 0.001)
            KBS[nmk] = dict(partial=int(part.sum()), held=int(held_.sum()), free=int((act & (kv > 0.999)).sum()), partial_ext=int((part & w8k).sum()), partial_front=int((part & front).sum()), corner_partial=int(part[nidc].sum()), partial_cnt_med=float(cntn[part].median()) if bool(part.any()) else None)
            out[f'k{k}_{nmk}'] = kv[nidc].half().cpu().numpy()
        out[f'k{k}_kbstat'] = np.array(json.dumps(dict(KBS=KBS, nact=int(act.sum()), front=int(front.sum()), front_corner=int(front[nidc].sum()), ext=int((act & w8k).sum()), cnt_med=float(cntn[act].median()), cnt_front_med=float(cntn[front].median()) if bool(front.any()) else None, held_part=float(heldp.mean())), ensure_ascii=False))
        kb = getattr(s, 'kb', None)
        if kb is not None:
            out[f'k{k}_kb'] = kb[:nn][nidc].half().cpu().numpy()
        out[f'k{k}_stk'] = (s.an[:nn] / s.mqn[:nn].clamp_min(1e-30)).clamp(0, 1)[nidc].half().cpu().numpy()
        out[f'k{k}_sk'] = sk[nidc].half().cpu().numpy()
        out[f'k{k}_mnc'] = mn[nidc].float().cpu().numpy()
        Ut_ = snapv.get('Ut_관성예측:rt_ga_step0')
        ymt = snapv.get('y:mt_floor_n')
        yh_ = snapv.get('yh_기준:rt_grad')
        if Ut_ is not None and ymt is not None:
            dm_ = (ymt - Ut_)[nidc]
            out[f'k{k}_dmt'] = dm_.float().cpu().numpy()
            out[f'k{k}_dmt_tb'] = ((dm_ * (wsg > 0)).sum(1) / 4.0 - (dm_ * (wsg < 0)).sum(1) / 4.0).float().cpu().numpy()
        if Ut_ is not None and yh_ is not None:
            out[f'k{k}_dyh'] = (yh_ - Ut_)[nidc].float().cpu().numpy()
        out[f'k{k}_r'] = torch.stack(R).float().cpu().numpy()
        out[f'k{k}_rC'] = torch.stack(RC).float().cpu().numpy()
        out[f'k{k}_rF'] = torch.stack(RF).float().cpu().numpy()
        me = W.me[k, :nn].double()
        FE = []
        ev_ok = []
        for e in range(EV):
            sl = k * EV + e
            if itb[sl] < 0:
                continue
            it2 = float(itf[sl])
            en = W.en[sl, :nn].double()
            gn = W.gn[sl, :nn].double()
            acc = W.acc[sl, :nbu, :nn]
            T = [mn * it2 * en + tn * mn * gz, -tn * mn * gz, (me - mn) * it2 * en] + [tn * acc[b] for b in range(nbu)] + [me * it2 * en + tn * acc.sum(0), gn]
            FE.append(torch.stack([(g_[nidc] * wsg).sum(1) for g_ in T], 0).float().cpu().numpy())
            ev_ok.append(e)
        if FE:
            out[f'k{k}_F'] = np.stack(FE, 0)
        steps.append(dict(k=k, tags=tags, ev=ev_ok, ncons=int(cons[mn > 0].sum()), nact=int((mn > 0).sum())))
    out['rEnd'] = rr(s.U[:nn].double() @ n_).float().cpu().numpy()
    out['terms'] = np.array(['관성', '중력', '질량하한'] + names + ['합', '풀이기잔차'])
    os.makedirs(os.path.join(OUT[0], 'sp'), exist_ok=True)
    _SPW.submit(np.savez_compressed, os.path.join(OUT[0], 'sp', f'f{fr:03d}.npz'), steps=np.array(json.dumps(steps, ensure_ascii=False)), **out)
    gs = W.gs.cpu().tolist()
    gc = W.gc.cpu().tolist()
    o.update(ncell=int(len(hs)), steps=steps, snapdrop=INFO['n_snapdrop'], g2=[[gs[2 * k], gs[2 * k + 1], gc[2 * k + 1]] for k in range(KSG) if gs[2 * k] >= 0])
    return o

def startpt_install(out, fr0=3, fr1=12):
    (startpt_FR[0], startpt_FR[1]) = (int(fr0), int(fr1))
    if os.environ.get('AF_SP_FR'):
        (startpt_FR[0], startpt_FR[1]) = (int(q) for q in os.environ['AF_SP_FR'].split(','))
    import hexfem as ART
    OUT[0] = out
    startpt_LOG[0] = open(os.path.join(out, 'sp.jsonl'), 'w')
    _wrap(ART.an_rt_RTSolver)
    print(f'[절 CM113] startpt 켬 -- 걸음 시작 단계·뉴턴 갱신별 칸 r, 평가별 항 (결과 불변), 프레임 {startpt_FR}', flush=True)
    startpt__dump()

# ---------------- fem_convwhy ----------------
import glob
convwhy_ST = dict(save=[], load='', ks=1.0, nopl=0, out='.', nsave=0, nload=0, nframe=0, nctl3=0, nctl7=0, nroll=0, nalloc=0, nopl_src=0, err=None, states=[])
convwhy_LOG = [None]
NOPL_OLD = '    if (!(dg > 0.f)) return 0;\n    if (tr > 0.f) { s = 0.f; return 2; }'
NOPL_NEW = '    if (1) return 0;   // 절 CM96 소성 끔 (진단)\n    if (tr > 0.f) { s = 0.f; return 2; }'

def convwhy__dump():
    try:
        with open(os.path.join(convwhy_ST['out'], 'cwinfo.json'), 'w') as f_:
            json.dump(convwhy_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

def _kmax(s):
    return 4 * max(s.K, s.Kc or 0, getattr(s, 'kcap', 0))

def _cgmax(s):
    return max(s.cg_list) if getattr(s, 'cg_list', None) else int(s.cg)

def _alloc_bufs(s):
    (KK, nt, cg) = (_kmax(s), int(s.nt), _cgmax(s))
    dev = s.dev
    s.cwS3 = torch.zeros(KK * nt, 64, device=dev, dtype=torch.float64)
    s.cwD3 = torch.zeros(KK * nt, 4, device=dev, dtype=torch.float64)
    s.cwT3 = torch.zeros(KK * nt, 16, device=dev, dtype=torch.int32)
    s.cwD7 = torch.zeros(KK * nt, 4, device=dev, dtype=torch.float64)
    s.cwT7 = torch.zeros(KK * nt, 16, device=dev, dtype=torch.int32)
    s.cwG = torch.zeros(KK * nt, cg, 6, device=dev, dtype=torch.float64)
    s.cwGT = torch.zeros(KK * nt, cg, 2, device=dev, dtype=torch.int32)
    (s._cw_k, s._cw_it, s._cw_ci) = (0, 0, 0)
    convwhy_ST['nalloc'] += 1

def _clear(s):
    for nm in ('cwS3', 'cwD3', 'cwT3', 'cwD7', 'cwT7', 'cwG', 'cwGT'):
        getattr(s, nm).zero_()
    s.cwGT.fill_(-1)
    s.cwT3[:, 0] = -1

def _lanczos(al, be):
    m = len(al)
    if m == 0:
        return None
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0.0)
        if j + 1 < m:
            T[j, j + 1] = T[j + 1, j] = math.sqrt(max(be[j], 0.0)) / al[j]
    ev = np.linalg.eigvalsh(T)
    return (float(ev[0]), float(ev[-1]))

def _read(s, label, K):
    nt = int(s.nt)
    (S3, D3, T3) = (s.cwS3.cpu().numpy(), s.cwD3.cpu().numpy(), s.cwT3.cpu().numpy())
    (D7, T7) = (s.cwD7.cpu().numpy(), s.cwT7.cpu().numpy())
    (G, GT) = (s.cwG.cpu().numpy(), s.cwGT.cpu().numpy())
    BSK = s.BSK.cpu().numpy() if getattr(s, 'BSK', None) is not None else None
    Y0K = s.Y0K.cpu().numpy() if getattr(s, 'Y0K', None) is not None else None
    cjg = s.cjg.cpu().numpy().tolist() if getattr(s, 'cjg', None) is not None else None
    steps = []
    for k in range(K):
        its = []
        F2 = None
        for it in range(nt):
            i = k * nt + it
            if T3[i, 0] < 0:
                break
            if it == 0:
                F2 = float(D3[i, 2])
            g2 = float(S3[i, 1])
            stop = int(T3[i, 0])
            o = dict(it=it, r=math.sqrt(g2 / F2) if F2 and F2 > 0 and (g2 >= 0) else None, stop=stop)
            if stop:
                its.append(o)
                break
            nP = int(T7[i, 3]) - int(T3[i, 3])
            o.update(nP=nP, Preach=int(T7[i, 1]), Ptgt=float(D7[i, 1]))
            (al, be, rr) = ([], [], [])
            for ci in range(G.shape[1]):
                if GT[i, ci, 0] != 0 or GT[i, ci, 1] != 0:
                    break
                (rz, pAp, rz1, r2) = (G[i, ci, 0], G[i, ci, 2], G[i, ci, 3], G[i, ci, 5])
                if not (pAp > 0 and rz > 0):
                    break
                al.append(rz / pAp)
                be.append(rz1 / rz)
                rr.append(r2)
            o['nL'] = len(al)
            if al:
                lam = _lanczos(al, be)
                if lam:
                    o.update(lmin=lam[0], lmax=lam[1], kap=lam[1] / lam[0] if lam[0] > 0 else None)
                o['Prel'] = math.sqrt(rr[-1] / g2) if g2 > 0 else None
            if BSK is not None:
                o['bk'] = int(BSK[i, 0])
            its.append(o)
        rs = [q['r'] for q in its if q.get('r') is not None]
        steps.append(dict(k=k, its=its, F2=F2, y0k=int(Y0K[k, 0]) if Y0K is not None else None, g2end=cjg[k] if cjg and k < len(cjg) else None, rs=rs))
    return steps

def convwhy_install(out, save='', load='', ks=1.0, nopl=0):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_nlwhy ----------------
nlwhy_ST = dict(out='.', nsrc=0, nalloc=0, nframe=0, npl=0, nls=0, err=None)
nlwhy_LOG = [None]
nlwhy_MARK = '// ---- 절 CM119 nlwhy'
nlwhy_HEAD = '__global__ void ao_commit('
NL_SRC = cu('NL_SRC')

def nlwhy_P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def nlwhy_patch(code):
    if nlwhy_MARK in code or nlwhy_HEAD not in code or 'ao_dpf(' not in code:
        return code
    i = code.index(nlwhy_HEAD)
    nlwhy_ST['nsrc'] += 1
    return code[:i] + nlwhy_MARK + '\n' + NL_SRC + code[i:]

def nlwhy__dump():
    try:
        with open(os.path.join(nlwhy_ST['out'], 'nlinfo.json'), 'w') as f_:
            json.dump(nlwhy_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

def nlwhy_install(out):
    import hexfem as ART
    CW = sys.modules[__name__]
    nlwhy_ST.update(out=out)
    nlwhy_LOG[0] = open(os.path.join(out, 'nl.jsonl'), 'w')
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if 'code' in kw:
            kw['code'] = nlwhy_patch(kw['code'])
        elif a:
            a = (nlwhy_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    R = ART.an_rt_RTSolver
    (oa, octl, ok) = (R.alloc, R.ctl, R.k)

    def alloc(s, need):
        r = oa(s, need)
        try:
            CW._alloc_bufs(s)
            (KK, nt) = (CW._kmax(s), int(s.nt))
            s.nlLS = torch.zeros(KK * nt, 2, device=s.dev, dtype=torch.float64)
            s.nlPL = torch.zeros(KK * nt, 8, device=s.dev, dtype=torch.int32)
            capQ = s.Fq.numel() // 9
            if getattr(s, 'nlST', None) is None or s.nlST.numel() != capQ:
                s.nlST = torch.full((capQ,), -1, device=s.dev, dtype=torch.int32)
            nlwhy_ST['nalloc'] += 1
        except Exception as e:
            nlwhy_ST['err'] = 'alloc ' + repr(e)
        return r

    def row(s):
        return max(s._cw_k, 0) * int(s.nt) + s._cw_it

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if getattr(s, 'cwS3', None) is None:
            return r
        if op == 0:
            s._cw_k = -1
        elif op == 1:
            s._cw_k += 1
        elif op == 2:
            s._cw_it = it
        elif op == 5:
            s._cw_ci = it
        elif op in (3, 7):
            i = max(s._cw_k, 0) * int(s.nt) + it
            if op == 3:
                s.dcopy(s.cwS3[i], s.SC)
                s.dcopy(s.cwD3[i], s.CD)
                s.dcopy(s.cwT3[i], s.CT)
            else:
                s.dcopy(s.cwD7[i], s.CD)
                s.dcopy(s.cwT7[i], s.CT)
        return r

    def k(s, name, n, args, b=128):
        if getattr(s, 'cwS3', None) is not None:
            if name == 'rt_sc_roll':
                i = row(s)
                ci = s._cw_ci
                if ci < s.cwG.shape[1]:
                    s.dcopy(s.cwG[i, ci, 0:5], s.SC[0:5])
                    s.dcopy(s.cwG[i, ci, 5:6], s.SC[57:58])
                    s.dcopy(s.cwGT[i, ci], s.CT[0:2])
            elif name == 'rt_lsup':
                i = row(s)
                s.dcopy(s.nlLS[i, 0:1], s.SC[14:15])
                if int(args[-1]):
                    s._nl_full = getattr(s, '_nl_full', set()) | {i}
                nlwhy_ST['nls'] += 1
            elif name in ('ao_elem_s', 'ao_elem') and getattr(s, 'nlST', None) is not None:
                i = row(s)
                ok(s, 'nl_pst', n, (args[0], args[1], args[2], args[3], args[5], args[8], nlwhy_P(s.nlST), nlwhy_P(s.nlPL[i])), 128)
                nlwhy_ST['npl'] += 1
        return ok(s, name, n, args, b)
    (R.alloc, R.ctl, R.k) = (alloc, ctl, k)

    def frame(fn, s, fi, *a):
        if getattr(s, '_nl_in', False):
            return fn(s, *a)
        s._nl_in = True
        try:
            return frame1(fn, s, fi, *a)
        finally:
            s._nl_in = False

    def frame1(fn, s, fi, *a):
        if getattr(s, 'cwS3', None) is not None:
            CW._clear(s)
            s.nlLS.zero_()
            s.nlPL.zero_()
        if not hasattr(s, '_nl_full'):
            s._nl_full = set()
        r = fn(s, *a)
        try:
            if getattr(s, 'cwS3', None) is not None:
                torch.cuda.synchronize()
                K = int(s.Kcur)
                steps = CW._read(s, None, K)
                (LS, PL) = (s.nlLS.cpu().numpy(), s.nlPL.cpu().numpy())
                nt = int(s.nt)
                for st_ in steps:
                    for o in st_['its']:
                        i = st_['k'] * nt + o['it']
                        o['ls'] = float(LS[i, 0])
                        o['full'] = int(i in s._nl_full)
                        o['pl'] = PL[i].tolist()
                c = {}
                try:
                    c = s.counts()
                except Exception:
                    pass
                nlwhy_LOG[0].write(json.dumps(dict(f=fi + 1, K=K, steps=steps, ms=float(getattr(s, 'last_ms', 0.0)), nq=c.get('nq'), q_tail=c.get('q_tail'), q_apex=c.get('q_apex')), ensure_ascii=False, default=float) + '\n')
                nlwhy_LOG[0].flush()
        except Exception as e:
            nlwhy_ST['err'] = 'read ' + repr(e)
        nlwhy_ST['nframe'] += 1
        if nlwhy_ST['nframe'] in (1, 3) or nlwhy_ST['nframe'] % 10 == 0:
            nlwhy__dump()
        return r
    osplit = ART.an_rt_split_step

    def split_step(s, t, h, depth):
        if depth > 0:
            return osplit(s, t, h, depth)
        return frame(lambda s_, *a: osplit(s_, *a), s, int(round(t / h)), t, h, depth)
    ART.an_rt_split_step = split_step
    try:
        import hexfem as WRX
        ocol = WRX.split_col

        def split_col(s, t, h, sub_dt, depth, fn=None):
            if depth > 0:
                return ocol(s, t, h, sub_dt, depth, fn)
            return frame(lambda s_, *a: ocol(s_, *a), s, int(round(t / h)), t, h, sub_dt, depth, fn)
        WRX.split_col = split_col
    except Exception as e:
        nlwhy_ST['err'] = 'wrx ' + repr(e)
    import atexit
    atexit.register(nlwhy__dump)
    print('[절 CM119] nlwhy 켬 -- 뉴턴 반복별 |g|·PCG·선탐색·장벽 보폭·소성 상태 바뀜 기록 (결과 불변)', flush=True)
    nlwhy__dump()

# ---------------- stabstat ----------------
NSUB = 32768

def subset(n, k=NSUB, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))
KEYS = ('ke', 'pe', 'nan', 'out', 'vmax', 'detneg', 'detmin', 'ee', 'jdev', 'jdev10', 'penr', 'pend', 'px', 'py', 'pz')
SAFE_KEYS = ('safe_n', 'safe_split', 'safe_inert')

def elastic_fcr(F, vol, E, nu):
    (mu, lam) = (E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)))
    (U, S, Vt) = np.linalg.svd(F)
    R = U @ Vt
    J = np.prod(S, axis=1)
    psi = mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2
    return float((vol * psi).sum())

def frame_stats(x, v, F, mass, g, floor_z, box=(0.0, 2.0), vol=None, E=None, nu=None):
    x = np.asarray(x, np.float64)
    v = np.asarray(v, np.float64)
    fin = np.isfinite(x).all(1) & np.isfinite(v).all(1)
    out = fin & ((x < box[0]) | (x > box[1])).any(1)
    (xm, vm, mm) = (x[fin], v[fin], mass[fin])
    gz = -float(g[2])
    ke = 0.5 * float((mm * (vm * vm).sum(1)).sum())
    pe = gz * float((mm * (xm[:, 2] - floor_z)).sum())
    st = dict(ke=ke, pe=pe, nan=float(1 - fin.mean()), out=float(out.mean()), vmax=float(np.sqrt((vm * vm).sum(1)).max()) if len(vm) else float('nan'))
    if F is not None:
        F = np.asarray(F, np.float64).reshape(-1, 3, 3)
        d = np.linalg.det(F[fin]) if fin.any() else np.array([np.nan])
        st['detneg'] = float((d <= 0).mean())
        st['detmin'] = float(np.nanmin(d))
        st['jdev'] = float(np.nanmean(np.abs(d - 1)))
        st['jdev10'] = float((np.abs(d - 1) > 0.1).mean())
        if vol is not None and E is not None and fin.any():
            st['ee'] = elastic_fcr(F[fin], np.asarray(vol, np.float64)[fin], float(E), float(nu))
    if len(xm):
        dep = float(floor_z) - xm[:, 2]
        st['penr'] = float((dep > 0).mean())
        st['pend'] = float(max(dep.max(), 0.0))
        p = (mm[:, None] * vm).sum(0)
        (st['px'], st['py'], st['pz']) = (float(q) for q in p)
    return st
SIG_HI = 100.0
