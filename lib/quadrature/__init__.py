# ---------------- fem_gq1 ----------------
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of quadrature.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quadrature.cu'), encoding='utf-8', newline='').read()
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

import json
import math
import os
import numpy as np
CFG = dict(q1=0, hg=0.0, hgm=0, vs=1.0, ck=0)
INFO = dict(cfg=None, mods=0, gauss_patch=0, sgauss_patch=0, hooks={}, names={}, solvers=[], err=None, check=None, coarse_swap=0)
OUT = ['.']
ELEM = ('rt_elem', 'ao_elem', 'ao_elem_s', 'aq_elem')
MV = ('rt_mv', 'rt_mv2', 'rt_mv2s')
gq1_COARSE_K = ('rt_mg_', 'ao_mg_', 'cvc')
UNSUP = ('rt_elem_ls', 'en_phi_n', 'ao_mv_q', 'ao_mv_h', 'ao_mv_c', 'ao_mv_s', 'rt_mg_mv2_f')
SRC = cu('SRC')
HM_SRC = cu('HM_SRC')

def gq1__dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], 'gq1info.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass

def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())

def gq1__sub(src, old, new, name):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"[절 CM194 gq1] {name}: '{old[:70]}' 수 {n} (기대 1)")
    return src.replace(old, new)

def _fn_span(src, name):
    i = src.find('void ' + name + '(')
    if i < 0:
        return None
    j = src.find('\n__global__', i + 10)
    j2 = src.find('\n__device__', i + 10)
    ends = [q for q in (j, j2) if q > 0]
    return (i, min(ends) if ends else len(src))

def gq1_transform(src):
    did = False
    if CFG['q1']:
        sp = _fn_span(src, 'rt_gauss')
        if sp is not None:
            (a, b) = sp
            body = src[a:b]
            body = gq1__sub(body, 'if (aq[27] > 0.f) {', 'if (false) {   // 절 CM194 1 점: 칸 평균 F', 'rt_gauss')
            body = gq1__sub(body, 'const float GG = 0.28867513459481287f;', 'const float GG = 0.f;   // 절 CM194 1 점: 칸 가운데', 'rt_gauss')
            if 'AF_GOCT' in body:
                body = gq1__sub(body, 'Vq[q] = (1.f - AF_GOCT)*aq[27] + AF_GOCT*Vq[q]; mm = (1.f - AF_GOCT)*AF_MO[q] + AF_GOCT*mm;', '   // 절 CM194 1 점: V = 칸 부피 / 8 (κ 무관)', 'rt_gauss')
            src = src[:a] + body + src[b:]
            INFO['gauss_patch'] += 1
            did = True
        sp = _fn_span(src, 'aq_gauss_s')
        if sp is not None:
            (a, b) = sp
            body = gq1__sub(src[a:b], 'const float* a = (Sa[(size_t)q*4+3] > 0.f) ? Sa + (size_t)q*4 : Sc + (size_t)c*4;', 'const float* a = Sc + (size_t)c*4;   // 절 CM194 1 점: 칸 평균 상태', 'aq_gauss_s')
            src = src[:a] + body + src[b:]
            INFO['sgauss_patch'] += 1
            did = True
    if CFG['hg'] > 0 and 'extern "C" {' in src and ('void Tapply(' in src) and ('void blocksum_add(' in src) and ('void qnodes(' in src) and ('hg_force(' not in src):
        j = src.rindex('}')
        src = src[:j] + SRC + '\n' + src[j:]
        did = True
    if did:
        INFO['mods'] += 1
        gq1__dump()
    return src

def gq1_tables(Ai, offs):
    Ai64 = Ai.double().cpu().numpy().reshape(3, 3)
    n = len(offs)
    BW = np.zeros((n, 27))
    BG = np.zeros((n, 27, 3))
    for (oc, fxv) in enumerate(offs):
        w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for o in range(27):
            (i, j, k) = (o // 9, o // 3 % 3, o % 3)
            BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
            gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
            BG[oc, o] = gy @ Ai64
    return (BW, BG)
GG = 0.5 / math.sqrt(3.0)
OFF8 = [[1.0 + (GG if oc >> 2 - a & 1 else -GG) for a in range(3)] for oc in range(8)]

class St:

    def __init__(h, s):
        import torch
        (h.s, h.torch) = (s, torch)
        (mu, lam) = (getattr(s, 'mu', None), getattr(s, 'lam', None))
        prm = getattr(s, 'prm', None)
        try:
            (mu, lam) = (float(mu), float(lam))
        except Exception:
            (mu, lam) = (float(prm[0]), float(prm[1]))
        (h.mu, h.lam) = (mu, lam)
        dev = s.BG.device
        (BW8, BG8) = gq1_tables(s.Ai, OFF8)
        if s.BG.numel() == BG8.size:
            h.err8 = float(np.abs(BG8.reshape(-1) - s.BG.double().cpu().numpy()).max() / max(np.abs(BG8).max(), 1e-30))
            h.errw = float(np.abs(BW8.reshape(-1) - s.BW.double().cpu().numpy()).max())
        else:
            h.err8 = h.errw = None
        assert not CFG['q1'] or (h.err8 < 1e-05 and h.errw < 1e-06), f'[절 CM194] 실행기 2x2x2 표와 다름 {h.err8} {h.errw}'
        (BWc, BGc) = gq1_tables(s.Ai, [[1.0, 1.0, 1.0]])
        f32 = torch.float32
        h.BG8 = torch.tensor(BG8, device=dev, dtype=f32).reshape(-1).contiguous()
        h.BW8 = torch.tensor(BW8, device=dev, dtype=f32).reshape(-1).contiguous()
        h.BGc = torch.tensor(BGc, device=dev, dtype=f32).reshape(-1).contiguous()
        h.kg = float(CFG['hg']) * h.mu
        if CFG['q1']:
            (s.BW8_orig, s.BG8_orig) = (s.BW, s.BG)
            s.BW = torch.tensor(np.repeat(BWc, 8, 0), device=dev, dtype=f32).reshape(-1).contiguous()
            s.BG = torch.tensor(np.repeat(BGc, 8, 0), device=dev, dtype=f32).reshape(-1).contiguous()
        h.rows = []
        h.hm = None
        if CFG['hgm']:
            import cupy as cp
            h.hm = cp.RawModule(code=HM_SRC, options=('--std=c++14',)).get_function('hm_cell')
            h.hmout = torch.zeros(4, device=dev, dtype=torch.float64)
            nu = h.lam / (2.0 * (h.lam + h.mu))
            h.r = nu / (1.0 - 2.0 * nu)
        INFO['solvers'].append(dict(cls=type(s).__name__, mu=h.mu, lam=h.lam, kg=h.kg, q1=CFG['q1'], hg=CFG['hg'], tab_err8=h.err8, tab_errw=h.errw, BWc_sum=float(BWc.sum()), BGc_rowsum=float(np.abs(BGc.sum(1)).max())))
        print(f"[절 CM194 gq1] {type(s).__name__}: 1 점 {CFG['q1']} 벌점 γ {CFG['hg']:g} (kg = γμ = {h.kg:.6g}) 에너지 비 {CFG['hgm']} 표 검사 {h.err8}", flush=True)
        gq1__dump()

    def frame(h, t):
        if h.hm is None:
            return
        (s, torch) = (h.s, h.torch)
        import cupy as cp
        torch.cuda.synchronize()
        ip = s.IP.cpu().numpy().tolist()
        nc = int(ip[11])
        h.hmout.zero_()
        if nc > 0 and int(ip[13]) == 0:
            n = nc * 8
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                h.hm(((n + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.U), _P(h.BG8), np.float64(h.r), _P(h.hmout)))
        torch.cuda.synchronize()
        o = h.hmout.cpu().numpy().tolist()
        h.rows.append(dict(t=float(t), ehg=o[0], etot=o[1], nc=int(o[2]), nbad=int(o[3]), ip=ip[6:14]))
        with open(os.path.join(OUT[0], 'hgm.jsonl'), 'a') as f_:
            f_.write(json.dumps(h.rows[-1]) + '\n')

def gq1__hook(s, _ok, name, args):
    h = getattr(s, '_gq1', None)
    if h is None or not CFG['hg'] > 0:
        return
    if name in UNSUP:
        raise RuntimeError(f'[절 CM194] 벌점 판에서 안 고친 커널 {name}')
    (f32, i32) = (np.float32, np.int32)
    G = (_P(s.IP), _P(s.cb), _P(s.Fq), _P(s.Vq), _P(h.BG8), _P(h.BGc), f32(h.kg))
    capQ = int(s.Fq.shape[0])
    if name in ELEM:
        if name == 'rt_elem':
            mode = int(args[8])
            (y, E, f_, B) = (args[5], args[9], args[10], args[13])
        elif name == 'aq_elem':
            mode = 1 | 8
            (y, E, f_, B) = (args[5], None, args[10], args[13])
        else:
            mode = int(args[9])
            (y, E, f_, B) = (args[5], None, args[10], args[13])
        if mode & 13:
            _ok(s, 'hg_force', capQ, G + (y, f_ if mode & 1 else _P(None), B if mode & 8 else _P(None), E if E is not None and mode & 4 else _P(None)), 128)
        INFO['hooks']['%s:%d' % (name, mode)] = INFO['hooks'].get('%s:%d' % (name, mode), 0) + 1
    elif name in MV:
        _ok(s, 'hg_mv', capQ, G + (args[6], args[7]), 128)
        INFO['hooks'][name] = INFO['hooks'].get(name, 0) + 1
    elif name == 'rt_bd':
        _ok(s, 'hg_force', capQ, G + (_P(s.y), _P(None), args[6], _P(None)), 128)
        INFO['hooks'][name] = INFO['hooks'].get(name, 0) + 1
    elif name == 'rt_elem_lsN':
        if len(args) >= 19:
            (NC, i0, gate) = (int(args[16]), int(args[17]), args[18])
        else:
            (NC, i0, gate) = (int(args[16]), 0, _P(None))
        _ok(s, 'hg_ls', capQ, G + (args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], i32(NC), i32(i0), gate), 128)
        INFO['hooks'][name] = INFO['hooks'].get(name, 0) + 1

def selfcheck(h):
    raise NotImplementedError('not used by the paper settings')

def gq1_install(out, q1=0, hg=0.0, hgm=0, vs=1.0, ck=0):
    import cupy as cp
    import torch
    OUT[0] = out
    os.makedirs(out, exist_ok=True)
    CFG.update(q1=int(q1), hg=float(hg), hgm=int(hgm), vs=float(vs), ck=int(ck))
    assert CFG['hg'] == 0 or CFG['q1'], '--gq_hg 는 --gq_q1 1 위에서만'
    INFO['cfg'] = dict(CFG)
    if CFG['vs'] != 1.0:
        tl = torch.load

        def load(*a, **kw):
            d = tl(*a, **kw)
            if isinstance(d, dict) and isinstance(d.get('cfg'), dict) and ('x' in d) and (not d.get('_gq_vs')):
                c = dict(d['cfg'])
                sc = CFG['vs']
                c['g'] = [float(q) * sc * sc for q in c['g']]
                bcs = []
                for b in c.get('boundary_conditions') or []:
                    b = dict(b)
                    if 'velocity' in b:
                        b['velocity'] = [float(q) * sc for q in b['velocity']]
                    for kk in ('rotation_scale', 'translation_scale'):
                        if kk in b:
                            b[kk] = float(b[kk]) * sc
                    bcs.append(b)
                c['boundary_conditions'] = bcs
                d['cfg'] = c
                d['_gq_vs'] = True
                INFO['vs_cfg'] = dict(g=c['g'], bc=bcs)
                gq1__dump()
                print(f"[절 CM194 gq1] 충격 x{sc:g}: 중력 {c['g']}, 경계 속도 x{sc:g}", flush=True)
            return d
        torch.load = load
    if CFG['q1'] or CFG['hg'] > 0:
        base = cp.RawModule
        if not getattr(base, '_gq1', False):

            def RawModule(*a, **kw):
                a = list(a)
                code = kw.get('code', a[0] if a else None)
                if isinstance(code, str):
                    code2 = gq1_transform(code)
                    if code2 is not code:
                        if 'code' in kw:
                            kw['code'] = code2
                        else:
                            a[0] = code2
                return base(*a, **kw)
            RawModule._gq1 = True
            cp.RawModule = RawModule
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import hexfem as ART
    for cls in (ART.an_rt_RTSolver,):
        if getattr(cls.k, '_gq1', False):
            continue
        _ok = cls.k

        def k(s, name, n, args, b=128, _ok=_ok):
            INFO['names'][name] = INFO['names'].get(name, 0) + 1
            if CFG['q1'] and name.startswith(gq1_COARSE_K) and (getattr(s, 'BG8_orig', None) is not None):
                (pb, pw) = (_P(s.BG), _P(s.BW))
                args = tuple((_P(s.BG8_orig) if isinstance(x, np.uint64) and x == pb else _P(s.BW8_orig) if isinstance(x, np.uint64) and x == pw else x for x in args))
                INFO['coarse_swap'] += 1
            r = _ok(s, name, n, args, b)
            if getattr(s, '_gqin', False) or name.startswith('hg_'):
                return r
            s._gqin = True
            try:
                gq1__hook(s, _ok, name, args)
            except Exception as e_:
                INFO['err'] = f'{name}: {e_}'[:600]
                gq1__dump()
                raise
            finally:
                s._gqin = False
            return r
        k._gq1 = True
        cls.k = k
    C = ART.an_rt_RTSolver
    if not getattr(C.__init__, '_gq1', False):
        f0 = C.__init__

        def init(s, *a, _f0=f0, **kw):
            _f0(s, *a, **kw)
            if getattr(s, '_gq1', None) is not None:
                return
            s._gq1 = St(s)
            s._gq1_ok = _ok_base
            st0 = s.step
            nstep = [0]

            def step(t, *a2, **kw2):
                r = st0(t, *a2, **kw2)
                nstep[0] += 1
                h = s._gq1
                try:
                    h.frame(t)
                except Exception as e_:
                    INFO['hm_err'] = repr(e_)[:300]
                    gq1__dump()
                if nstep[0] == 2:
                    INFO['first_names'] = dict(INFO['names'])
                    gq1__dump()
                if CFG['ck'] and CFG['hg'] > 0 and (not getattr(h, '_ckd', False)) and (nstep[0] >= 3):
                    h._ckd = True
                    try:
                        o = selfcheck(h)
                    except Exception as e_:
                        o = dict(err=repr(e_)[:600])
                    INFO['check'] = o
                    json.dump(o, open(os.path.join(OUT[0], 'gq1ck.json'), 'w'), indent=1, default=str)
                    print(f"[절 CM194 gq1] 검사 {('통과' if o.get('ok') else '실패')}: {json.dumps(o, default=str)}", flush=True)
                    gq1__dump()
                if CFG['q1'] and nstep[0] == 1:
                    INFO['bg_is_center'] = bool(torch.equal(s.BG, s._gq1_center_BG)) if hasattr(s, '_gq1_center_BG') else None
                return r
            s.step = step
            if CFG['q1']:
                s._gq1_center_BG = s.BG
        init._gq1 = True
        C.__init__ = init
    _ok_base = ART.an_rt_RTSolver.k
    print(f'[절 CM194] gq1 켬 -- {json.dumps(CFG)}', flush=True)
    gq1__dump()

# ---------------- fem_q27 ----------------
import hashlib
import inspect
import sys
import traceback
q27_ST = dict(mode=0, cap=0, dg=0, exe=None, out='.', nsrc=0, nmod=0, ntr=0, kern={}, err=None, solver=None, nalloc=0)
q27_SQ35 = 0.5 * 0.6 ** 0.5
q27_XI3 = [0.5 - q27_SQ35, 0.5, 0.5 + q27_SQ35]
q27_W3 = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]

def q27_log(msg):
    raise NotImplementedError('not used by the paper settings')

def q27__info():
    try:
        with open(os.path.join(q27_ST['out'], 'q27info.json'), 'w') as f_:
            json.dump({k: v for (k, v) in q27_ST.items()}, f_, ensure_ascii=False, default=str, indent=1)
    except Exception:
        pass

def q27__capture(code):
    raise NotImplementedError('not used by the paper settings')

def q27__hook_rawmodule():
    import cupy as cp
    orig = cp.RawModule

    def RawModule(*a, **kw):
        code = kw.get('code') if 'code' in kw else a[0] if a else None
        if isinstance(code, str):
            q27_ST['nsrc'] += 1
            if q27_ST['cap']:
                q27__capture(code)
            if q27_ST['mode']:
                code2 = q27_transform(code)
                if 'code' in kw:
                    kw['code'] = code2
                else:
                    a = (code2,) + tuple(a[1:])
            q27__info()
        return orig(*a, **kw)
    cp.RawModule = RawModule

def q27__hook_solvers():
    import torch
    clss = []
    try:
        import hexfem as ART
        clss.append(ART.an_rt_RTSolver)
    except Exception:
        pass
    try:
        import hexfem as AORT
        clss += [AORT.AORTSolver, AORT.RTAdapt]
    except Exception:
        pass
    for C in clss:
        if getattr(C.__init__, '_q27', False):
            continue
        oi = C.__init__

        def init(s, *a, _oi=oi, _C=C, **kw):
            if q27_ST['solver'] is None:
                try:
                    names = ['x', 'v', 'F', 'm', 'vol', 'A', 'Ai', 'mu', 'lam', 'g', 'frame_dt']
                    d = {n: a[i] for (i, n) in enumerate(names) if i < len(a)}
                    sv = dict(cls=_C.__name__, goct=kw.get('goct', None), frame_dt=float(d['frame_dt']) if 'frame_dt' in d else None)
                    q27_JT['h'] = sv['frame_dt']
                    for n in ('m', 'vol', 'A', 'Ai', 'mu', 'lam'):
                        v = d.get(n)
                        sv[n] = v.detach().cpu() if torch.is_tensor(v) else v
                    torch.save(sv, os.path.join(q27_ST['out'], 'q27solver.pt'))
                    q27_ST['solver'] = dict(cls=_C.__name__, goct=kw.get('goct', None), N=int(d['x'].shape[0]) if 'x' in d else None, A=[float(q) for q in d['A'].reshape(-1).tolist()] if 'A' in d and torch.is_tensor(d['A']) else None, mu=float(d['mu']) if not torch.is_tensor(d.get('mu')) else 'tensor', lam=float(d['lam']) if not torch.is_tensor(d.get('lam')) else 'tensor')
                    q27__info()
                except Exception as e:
                    q27_ST['err'] = 'solver_info: ' + repr(e)
                    q27__info()
            return _oi(s, *a, **kw)
        init._q27 = True
        C.__init__ = init

def q27_install(out, mode, cap, dg, exe, jit=0, phase='all'):
    q27_ST.update(mode=int(mode), cap=int(cap), dg=int(dg), exe=exe, out=out, jit=int(jit))
    print('[절 CM101] q27 켬 mode %d cap %d dg %d 실행기 %s 단계 %s' % (q27_ST['mode'], q27_ST['cap'], q27_ST['dg'], exe, phase), flush=True)
    try:
        if phase in ('all', 'src'):
            q27__hook_rawmodule()
        if phase in ('all', 'cls'):
            q27__hook_solvers()
            if q27_ST['mode']:
                q27__install_q27()
            if q27_ST['jit']:
                q27__install_jit()
    except Exception as e:
        q27_ST['err'] = repr(e) + ' ' + traceback.format_exc()[-1500:]
        q27__info()
        raise
    q27__info()
import re
q27_Q27_HELP = cu('q27_Q27_HELP')

def q27__w32():
    w = []
    for o in range(32):
        if q27_ST['mode'] == 2:
            w.append(1.0 if o == 13 else 0.0)
        elif o < 27:
            w.append(q27_W3[o // 9] * q27_W3[o // 3 % 3] * q27_W3[o % 3])
        else:
            w.append(0.0)
    return w
q27_Q1_AX = cu('q27_Q1_AX')
q27_Q1_AX2 = cu('q27_Q1_AX2')

def q27__help():
    h = q27_Q27_HELP % ', '.join(('%.17ef' % v for v in q27__w32()))
    if q27_ST['mode'] == 2:
        a = h.index('__device__ __forceinline__ int q27_i')
        b = h.index('\n', a)
        h = h[:a] + q27_Q1_AX + h[b:]
        a = h.index('__device__ __forceinline__ void q27_ax')
        h = h[:a] + q27_Q1_AX2 + '\n'
    return h
q27_COARSE = set('seg8f seg8d rs_fin rt_vn rt_step0 rt_g2p_end rt_mg_dims rt_mg_fin qnodes_d rt_mg_elem rt_mg_mv rt_mg_restrict rt_mg_prolong rt_zero_lm\nrt_mg_D rt_mg_node rt_mg_pw1 rt_mg_pw2 rt_fill_rand rt_mg_csolve rt_mg_mv2 rt_mg_restrict2 rt_mg_restrict3 rt_mg_occ rt_mg_cvc cvc_Kz rt_mg_prolong_c\nrt_bar_mvl rt_mg_dbar ao_mg_gath ao_mg_acc_det ao_mg_cell mv2_core'.split())
q27_DETONLY = set('ao_mg_gath ao_mg_acc_det ao_mg_cell'.split())
q27_PAT = re.compile('(>>\\s*3\\b|&\\s*7\\b|\\*\\s*8\\b|\\b8\\s*\\*|0\\.125|\\bGG\\b|>= 1\\.f \\?|oc\\s*<\\s*8|o\\s*<\\s*8;)')
GEN = [(re.compile('\\b(qq|q)\\s*>>\\s*3\\b'), '\\1 >> 5'), (re.compile('\\b(qq|q)\\s*&\\s*7\\b'), '\\1 & 31'), (re.compile('\\bc\\s*\\*\\s*8\\s*\\+\\s*oc\\b'), 'c*32 + oc'), (re.compile('\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*4\\s*:\\s*0\\s*\\)\\s*\\+\\s*\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*2\\s*:\\s*0\\s*\\)\\s*\\+\\s*\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*1\\s*:\\s*0\\s*\\)'), 'q27_oc(\\1, \\2, \\3)'), (re.compile('\\(\\(oc & 4\\) \\? GG : -GG\\)'), 'q27_d(oc, 0)'), (re.compile('\\(\\(oc & 2\\) \\? GG : -GG\\)'), 'q27_d(oc, 1)'), (re.compile('\\(\\(oc & 1\\) \\? GG : -GG\\)'), 'q27_d(oc, 2)'), (re.compile('BG \\+ \\(j\\*27\\+o\\)\\*3'), 'BG + ((((qq) & 24) + j)*27+o)*3'), (re.compile('IP\\[12\\] = carry\\*8; if \\(carry > capQc\\)'), 'IP[12] = carry*32; if (carry*4 > capQc)')]
q27_OKLEFT = [re.compile('threadIdx\\.x\\s*&\\s*7'), re.compile('\\btl\\s*&\\s*7'), re.compile('tl\\s*&\\s*~7'), re.compile('\\btl\\s*>>\\s*3'), re.compile('threadIdx\\.x\\s*>>\\s*3'), re.compile('blockDim\\.x\\s*>>\\s*3'), re.compile('\\bcn\\s*>>'), re.compile('cn\\s*<\\s*8'), re.compile('\\(cn>>2\\)&1'), re.compile('j\\s*<\\s*8;'), re.compile('r\\s*\\*\\s*8'), re.compile('8\\s*\\*\\s*r\\b'), re.compile('oc8|o8_'), re.compile('const float GG = '), re.compile('oc<8; oc\\+\\+\\) \\{ const float\\* Qo = sw'), re.compile('LM\\s*\\+\\s*l\\*8|l\\*8\\s*\\+|\\(l-1\\)\\*8|\\(lf\\+1\\)\\*8|lf\\*8')]

def q27__split(code):
    idx = []
    for m in re.finditer('^(?:__global__|__device__)[^\\n]*', code, re.M):
        h = re.sub('__launch_bounds__\\(\\s*\\d+(\\s*,\\s*\\d+)?\\s*\\)', '', m.group(0))
        mm = re.search('(\\w+)\\s*\\(', h)
        idx.append((m.start(), mm.group(1) if mm else None))
    if not idx:
        return (code, [])
    head = code[:idx[0][0]]
    parts = []
    for (i, (st, n)) in enumerate(idx):
        en = idx[i + 1][0] if i + 1 < len(idx) else len(code)
        parts.append((n, code[st:en]))
    return (head, parts)

def q27__sub(b, old, new, n=1, name=''):
    c = b.count(old)
    if c != n:
        raise RuntimeError("q27 고침 '%s' 수 %d (기대 %d) in %s" % (old[:80], c, n, name))
    return b.replace(old, new)

def q27__special(n, b):
    if n == 'rt_gauss':
        b = q27__sub(b, 'ac[28]*0.125f', 'ac[28]*Q27W[oc]', 1, n)
        b = q27__sub(b, 'ac[29]*0.125f', 'ac[29]*Q27W[oc]', 1, n)
        b = q27__sub(b, 'for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];', 'if (oc >= 27) for (int k=0;k<9;k++) out[k] = (k % 4 == 0) ? 1.f : 0.f;   // 절 CM101 빈 자리\n    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];', 1, n)
    elif n == 'gq_pq':
        b = q27__sub(b, 'PC[(size_t)c*6+a]*0.125f', 'PC[(size_t)c*6+a]*Q27W[q & 31]', 1, n)
    elif n == 'rt_mg_acc':
        b = q27__sub(b, 'int c = q >> 3, oc = q & 7;', 'int c = (l == 1) ? (q >> 5) : (q >> 3), oc = (l == 1) ? (q & 31) : (q & 7);', 1, n)
        b = q27__sub(b, 'float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};', 'float tq[3] = {cx + 1.f + ((l == 1) ? q27_d(oc, 0) : ((oc & 4) ? GG : -GG)), cy + 1.f + ((l == 1) ? q27_d(oc, 1) : ((oc & 2) ? GG : -GG)), cz + 1.f + ((l == 1) ? q27_d(oc, 2) : ((oc & 1) ? GG : -GG))};', 1, n)
        return (b, False)
    elif n == 'rt_mv2':
        b = q27__sub(b, 'mv2_core(j, on,', 'mv2_core_q27((int)(qq & 24), j, on,', 1, n)
    elif n == 'rt_mv2s':
        b = q27__sub(b, 'const float* bq = BG + j*81;', 'const float* bq = BG + ((qq & 24) + j)*81;', 1, n)
        b = q27__sub(b, 'const float* bg = BG + oc*81 + o*3;', 'const float* bg = BG + ((qq & 24) + oc)*81 + o*3;', 1, n)
    elif n in ('fpi_p', 'lji_p'):
        b = q27__sub(b, 'const float GG = 0.28867513459481287f;\n    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);', 'int lo_[3]; float t[3]; for (int a=0;a<3;a++) q27_ax(fx[p*3+a], &lo_[a], &t[a]);   // 절 CM101 27 점 보간', 1, n)
        b = q27__sub(b, 'for (int oc=0; oc<8; oc++) {\n        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);', 'for (int o8_=0; o8_<8; o8_++) {\n        float w = ((o8_ & 4) ? t[0] : 1.f - t[0]) * ((o8_ & 2) ? t[1] : 1.f - t[1]) * ((o8_ & 1) ? t[2] : 1.f - t[2]);\n        int oc = (lo_[0] + ((o8_ >> 2) & 1))*9 + (lo_[1] + ((o8_ >> 1) & 1))*3 + (lo_[2] + (o8_ & 1));', 1, n)
    elif n == 'aq_p2g_sd':
        b = q27__sub(b, 'float a[8][4]; for (int o=0;o<8;o++)', 'float a[32][4]; for (int o=0;o<32;o++)', 1, n)
        b = q27__sub(b, 'for (int o=0;o<8;o++) for (int k=0;k<4;k++) Sa[((size_t)c*8+o)*4+k] = a[o][k];', 'for (int o=0;o<32;o++) for (int k=0;k<4;k++) Sa[((size_t)c*32+o)*4+k] = a[o][k];', 1, n)
    elif n in ('ao_gath_q', 'gq_gath'):
        b = re.sub('for \\(int oc=0;oc<8;oc\\+\\+\\)', 'for (int oc=0;oc<32;oc++)', b)
        b = b.replace('qv[(size_t)c*8+oc]', 'qv[(size_t)c*32+oc]')
    elif n in ('sp_csum', 'sp_q2n'):
        b = re.sub('for \\(int oc=0; oc<8; oc\\+\\+\\)', 'for (int oc=0; oc<32; oc++)', b)
    return (b, True)

def q27__mv2_core_q27(b):
    b = q27__sub(b, '__device__ __forceinline__ void mv2_core(int j,', '__device__ __forceinline__ void mv2_core_q27(int gofs, int j,', 1, 'mv2_core')
    b = q27__sub(b, 'const float* bq = BG + j*81;', 'const float* bq = BG + (gofs + j)*81;', 1, 'mv2_core')
    b = q27__sub(b, 'const float* bg = BG + oc*81 + o*3;', 'const float* bg = BG + (gofs + oc)*81 + o*3;', 1, 'mv2_core')
    return b

def q27_transform(code):
    (head, parts) = q27__split(code)
    if not parts or 'IP[12]' not in code:
        return code
    (out, rep, unk) = ([], {}, [])
    for (n, b) in parts:
        if n is None or not q27_PAT.search(b) or n in q27_COARSE or (n.startswith(('rt_mg_', 'ao_mg_', 'cvc')) and n != 'rt_mg_acc'):
            out.append(b)
            if n == 'mv2_core' and 'mv2_core(j, on,' in code:
                out.append(q27__mv2_core_q27(b))
            continue
        (b2, gen) = q27__special(n, b)
        if gen:
            for (rx, rp) in GEN:
                b2 = rx.sub(rp, b2)
        left = []
        for ln in b2.split('\n'):
            core = ln.split('//')[0]
            for o_ in q27_OKLEFT:
                core = o_.sub(' ', core)
            if q27_PAT.search(core) and n != 'rt_mg_acc':
                left.append(ln.strip()[:160])
        if left:
            unk.append((n, left))
        rep[n] = int(b2 != b)
        out.append(b2)
    q27_ST['kern'].update(rep)
    if unk:
        q27_ST['left'] = q27_ST.get('left', []) + [(n, l[:4]) for (n, l) in unk]
    new = head + ''.join(out)
    i = new.find('extern "C" {')
    new = new[:i + len('extern "C" {')] + q27__help() + new[i + len('extern "C" {'):] if i >= 0 else q27__help() + new
    q27_ST['ntr'] += 1
    return new

def q27_tables(Ai):
    Ai64 = Ai.double().cpu().numpy().reshape(3, 3)
    BW = np.zeros((32, 27))
    BG = np.zeros((32, 27, 3))
    for oc in range(27):
        ii = (oc // 9, oc // 3 % 3, oc % 3)
        fxv = [1.0 + (q27_XI3[ii[a]] - 0.5) for a in range(3)]
        w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for o in range(27):
            (i, j, k) = (o // 9, o // 3 % 3, o % 3)
            BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
            gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
            BG[oc, o] = gy @ Ai64
    return (BW, BG)
q27_COARSE_K = ('rt_mg_', 'ao_mg_', 'cvc')

def q27__install_q27():
    import torch
    import hexfem as ART
    bases = [ART.an_rt_RTSolver]
    try:
        import hexfem as V0
        bases.append(V0.an_rt_v0_RTSolver)
    except Exception:
        pass
    P = ART.an_rt_P
    for C in bases:
        if C.__dict__.get('_q27', False):
            continue
        C._q27 = True
        (oa, oi, ok) = (C.alloc, C.__init__, C.k)

        def alloc(s, need, _oa=oa):
            if need is None:
                y = s.x.double() @ s.Ai.double().T
                b = torch.floor(y - 0.5)
                nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
                nn = nb[0] * nb[1] * nb[2]
                ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
                uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=s.dev)).sum(1)).numel()
                need = (nn, ncb, 4 * uc)
                s.nb0 = nb
                q27_ST['nalloc'] += 1
                q27_ST['alloc_last'] = dict(nn=nn, ncb=ncb, uc=uc)
            else:
                raise RuntimeError('q27: alloc(need) 경로는 지원 안 함')
            _oa(s, need)
            q27_ST['capQc'] = int(s.capQc)
            q27__info()

        def init(s, *a, _oi=oi, **kw):
            _oi(s, *a, **kw)
            if getattr(s, '_q27tab', False):
                return
            s._q27tab = True
            (BW, BG) = q27_tables(s.Ai)
            (s.BW8, s.BG8) = (s.BW, s.BG)
            s.BW = torch.tensor(BW, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            s.BG = torch.tensor(BG, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            q27_ST['tab'] = dict(bw_rowsum_min=float(BW[:27].sum(1).min()), bw_rowsum_max=float(BW[:27].sum(1).max()), bg_rowsum_absmax=float(np.abs(BG[:27].sum(1)).max()), w_sum=float(sum(q27__w32())), cls=type(s).__name__)
            q27__info()

        def k(s, name, n, args, b=128, _ok=ok):
            if name.startswith(q27_COARSE_K) and getattr(s, 'BG8', None) is not None:
                (pb, pw) = (P(s.BG), P(s.BW))
                args = tuple((P(s.BG8) if isinstance(x, np.uint64) and x == pb else P(s.BW8) if isinstance(x, np.uint64) and x == pw else x for x in args))
                q27_ST['ncoarse'] = q27_ST.get('ncoarse', 0) + 1
            return _ok(s, name, n, args, b)
        (C.alloc, C.__init__, C.k) = (alloc, init, k)
    mods = [ART]
    try:
        import hexfem as AORT
        mods.append(AORT)
    except Exception:
        pass
    for M in mods:
        for fn in ('split_step', 'adapt_step'):
            f0 = getattr(M, fn, None)
            if f0 is None or getattr(f0, '_q27', False):
                continue

            def w(s, *a, _f0=f0, **kw):
                try:
                    q27__check(s)
                except Exception as e:
                    q27_ST['chk_err'] = repr(e)
                return _f0(s, *a, **kw)
            w._q27 = True
            setattr(M, fn, w)
    try:
        import hexfem as V0
        ox = V0.an_rt_v0_RTSolver.x_orig
        if not getattr(ox, '_q27', False):

            def x_orig(s, _ox=ox):
                try:
                    q27__check(s)
                except Exception as e:
                    q27_ST['chk_err'] = repr(e)
                return _ox(s)
            x_orig._q27 = True
            V0.an_rt_v0_RTSolver.x_orig = x_orig
    except Exception as e:
        q27_ST['err'] = 'x_orig hook: ' + repr(e)
    q27_ST['installed'] = True
    q27__info()
q27__NCHK = [0]

def q27__check(s):
    import torch
    q27__NCHK[0] += 1
    i = q27__NCHK[0]
    if not (i <= 6 or i % 10 == 0) or not q27_ST['dg']:
        return
    with torch.no_grad():
        torch.cuda.synchronize()
        (nq, nc) = (int(s.IP[12]), int(s.IP[11]))
        if nq <= 0:
            return
        V = s.Vq[:nq].double()
        m = s.mq[:nq].double()
        sl = torch.arange(nq, device=V.device) & 31
        dmy = sl >= 27
        r = dict(call=i, nq=nq, nc=nc, nq_ok=int(nq == 32 * nc), sumV=float(V.sum()), sumvol=float(s.vol.double().sum()), summ=float(m.sum()), summp=float(s.m.double().sum()), dummyV=float(V[dmy].abs().sum()), dummym=float(m[dmy].abs().sum()), nV0=int((V[~dmy] <= 0).sum()), nreal=int((~dmy).sum()), Fdet_min=float(torch.linalg.det(s.Fq[:nq][~dmy].double().reshape(-1, 3, 3)).min()))
        with open(os.path.join(q27_ST['out'], 'q27.jsonl'), 'a') as f_:
            f_.write(json.dumps(r) + '\n')
q27_JT = dict(h=None, prev=None, ck=None, k=0, rows=[], f0=29, f1=99, err=None)

def q27__install_jit():
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_q27b ----------------
q27b_ST = dict(np=3, mode=0, cap=0, dg=0, exe=None, out='.', nsrc=0, nmod=0, ntr=0, kern={}, err=None, solver=None, nalloc=0)
q27b_SQ35 = 0.5 * 0.6 ** 0.5
q27b_XI3 = [0.5 - q27b_SQ35, 0.5, 0.5 + q27b_SQ35]
q27b_W3 = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]
_G4A = (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5
_G4B = (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5
XI4 = [0.5 - 0.5 * _G4B, 0.5 - 0.5 * _G4A, 0.5 + 0.5 * _G4A, 0.5 + 0.5 * _G4B]
_W4A = (18.0 + 30.0 ** 0.5) / 36.0
_W4B = (18.0 - 30.0 ** 0.5) / 36.0
W4 = [0.5 * _W4B, 0.5 * _W4A, 0.5 * _W4A, 0.5 * _W4B]

def XI():
    return XI4 if q27b_ST['np'] == 4 else q27b_XI3

def WAX():
    return W4 if q27b_ST['np'] == 4 else q27b_W3

def NPT():
    return q27b_ST['np'] ** 3

def SLOTS():
    return 64 if q27b_ST['np'] == 4 else 32

def LSL():
    return 6 if q27b_ST['np'] == 4 else 5

def q27b_log(msg):
    raise NotImplementedError('not used by the paper settings')

def q27b__info():
    try:
        with open(os.path.join(q27b_ST['out'], 'q27info.json'), 'w') as f_:
            json.dump({k: v for (k, v) in q27b_ST.items()}, f_, ensure_ascii=False, default=str, indent=1)
    except Exception:
        pass

def q27b__capture(code):
    raise NotImplementedError('not used by the paper settings')

def q27b__hook_rawmodule():
    import cupy as cp
    orig = cp.RawModule

    def RawModule(*a, **kw):
        code = kw.get('code') if 'code' in kw else a[0] if a else None
        if isinstance(code, str):
            q27b_ST['nsrc'] += 1
            if q27b_ST['cap']:
                q27b__capture(code)
            if q27b_ST['mode']:
                code2 = q27b_transform(code)
                if 'code' in kw:
                    kw['code'] = code2
                else:
                    a = (code2,) + tuple(a[1:])
            q27b__info()
        return orig(*a, **kw)
    cp.RawModule = RawModule

def q27b__hook_solvers():
    import torch
    clss = []
    try:
        import hexfem as ART
        clss.append(ART.an_rt_RTSolver)
    except Exception:
        pass
    try:
        import hexfem as AORT
        clss += [AORT.AORTSolver, AORT.RTAdapt]
    except Exception:
        pass
    for C in clss:
        if getattr(C.__init__, '_q27', False):
            continue
        oi = C.__init__

        def init(s, *a, _oi=oi, _C=C, **kw):
            if q27b_ST['solver'] is None:
                try:
                    names = ['x', 'v', 'F', 'm', 'vol', 'A', 'Ai', 'mu', 'lam', 'g', 'frame_dt']
                    d = {n: a[i] for (i, n) in enumerate(names) if i < len(a)}
                    sv = dict(cls=_C.__name__, goct=kw.get('goct', None), frame_dt=float(d['frame_dt']) if 'frame_dt' in d else None)
                    q27b_JT['h'] = sv['frame_dt']
                    for n in ('m', 'vol', 'A', 'Ai', 'mu', 'lam'):
                        v = d.get(n)
                        sv[n] = v.detach().cpu() if torch.is_tensor(v) else v
                    torch.save(sv, os.path.join(q27b_ST['out'], 'q27solver.pt'))
                    q27b_ST['solver'] = dict(cls=_C.__name__, goct=kw.get('goct', None), N=int(d['x'].shape[0]) if 'x' in d else None, A=[float(q) for q in d['A'].reshape(-1).tolist()] if 'A' in d and torch.is_tensor(d['A']) else None, mu=float(d['mu']) if not torch.is_tensor(d.get('mu')) else 'tensor', lam=float(d['lam']) if not torch.is_tensor(d.get('lam')) else 'tensor')
                    q27b__info()
                except Exception as e:
                    q27b_ST['err'] = 'solver_info: ' + repr(e)
                    q27b__info()
            return _oi(s, *a, **kw)
        init._q27 = True
        C.__init__ = init

def q27b_install(out, mode, cap, dg, exe, jit=0, phase='all', npts=3):
    q27b_ST.update(np=int(npts), mode=int(mode), cap=int(cap), dg=int(dg), exe=exe, out=out, jit=int(jit))
    print('[절 CM101] q27 켬 mode %d cap %d dg %d 실행기 %s 단계 %s' % (q27b_ST['mode'], q27b_ST['cap'], q27b_ST['dg'], exe, phase), flush=True)
    try:
        if phase in ('all', 'src'):
            q27b__hook_rawmodule()
        if phase in ('all', 'cls'):
            q27b__hook_solvers()
            if q27b_ST['mode']:
                q27b__install_q27()
            if q27b_ST['jit']:
                q27b__install_jit()
    except Exception as e:
        q27b_ST['err'] = repr(e) + ' ' + traceback.format_exc()[-1500:]
        q27b__info()
        raise
    q27b__info()
q27b_Q27_HELP = cu('q27b_Q27_HELP')
_QI3 = cu('_QI3')
_QX3 = cu('_QX3')

def _help_parts():
    if q27b_ST['np'] == 3:
        return (_QI3, _QX3)
    cw = np.cumsum(W4)
    QI = '__device__ __forceinline__ int q27_i(float f) { float u = f - 0.5f; return (u < %.17ef) ? 0 : ((u < %.17ef) ? 1 : ((u < %.17ef) ? 2 : 3)); }' % (cw[0], cw[1], cw[2])
    QX = '__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { float u = f - 0.5f;\n    const float X[4] = {%.17ef, %.17ef, %.17ef, %.17ef};\n    int l = (u >= X[1] ? 1 : 0) + (u >= X[2] ? 1 : 0); *lo = l; *t = fminf(fmaxf((u - X[l])/(X[l+1] - X[l]), 0.f), 1.f); }' % tuple(XI4)
    return (QI, QX)

def _wS():
    w = []
    (S, N) = (SLOTS(), NPT())
    A = WAX()
    P = q27b_ST['np']
    for o in range(S):
        if q27b_ST['mode'] == 2:
            w.append(1.0 if o == 13 else 0.0)
        elif o < N:
            w.append(A[o // (P * P)] * A[o // P % P] * A[o % P])
        else:
            w.append(0.0)
    return w
q27b__w32 = _wS
q27b_Q1_AX = cu('q27b_Q1_AX')
q27b_Q1_AX2 = cu('q27b_Q1_AX2')

def q27b__help():
    P = q27b_ST['np']
    (QI, QX) = _help_parts()
    D = [x - 0.5 for x in XI()]
    h = q27b_Q27_HELP % dict(NP=P, NP2=P * P, NPT=NPT(), S=SLOTS(), D=', '.join(('%.17ef' % v for v in D)), W=', '.join(('%.17ef' % v for v in _wS())), QI=QI, QX=QX)
    if q27b_ST['mode'] == 2:
        a = h.index('__device__ __forceinline__ int q27_i')
        b = h.index('\n', a)
        h = h[:a] + q27b_Q1_AX + h[b:]
        a = h.index('__device__ __forceinline__ void q27_ax')
        h = h[:a] + q27b_Q1_AX2 + '\n'
    return h
q27b_COARSE = set('seg8f seg8d rs_fin rt_vn rt_step0 rt_g2p_end rt_mg_dims rt_mg_fin qnodes_d rt_mg_elem rt_mg_mv rt_mg_restrict rt_mg_prolong rt_zero_lm\nrt_mg_D rt_mg_node rt_mg_pw1 rt_mg_pw2 rt_fill_rand rt_mg_csolve rt_mg_mv2 rt_mg_restrict2 rt_mg_restrict3 rt_mg_occ rt_mg_cvc cvc_Kz rt_mg_prolong_c\nrt_bar_mvl rt_mg_dbar ao_mg_gath ao_mg_acc_det ao_mg_cell mv2_core'.split())
q27b_DETONLY = set('ao_mg_gath ao_mg_acc_det ao_mg_cell'.split())
q27b_PAT = re.compile('(>>\\s*3\\b|&\\s*7\\b|\\*\\s*8\\b|\\b8\\s*\\*|0\\.125|\\bGG\\b|>= 1\\.f \\?|oc\\s*<\\s*8|o\\s*<\\s*8;)')

def _gen():
    (S, L) = (SLOTS(), LSL())
    P = q27b_ST['np']
    return [(re.compile('\\b(qq|q)\\s*>>\\s*3\\b'), '\\1 >> %d' % L), (re.compile('\\b(qq|q)\\s*&\\s*7\\b'), '\\1 & %d' % (S - 1)), (re.compile('\\bc\\s*\\*\\s*8\\s*\\+\\s*oc\\b'), 'c*%d + oc' % S), (re.compile('\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*4\\s*:\\s*0\\s*\\)\\s*\\+\\s*\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*2\\s*:\\s*0\\s*\\)\\s*\\+\\s*\\(\\s*([^()?]+?)\\s*>=\\s*1\\.f\\s*\\?\\s*1\\s*:\\s*0\\s*\\)'), 'q27_oc(\\1, \\2, \\3)'), (re.compile('\\(\\(oc & 4\\) \\? GG : -GG\\)'), 'q27_d(oc, 0)'), (re.compile('\\(\\(oc & 2\\) \\? GG : -GG\\)'), 'q27_d(oc, 1)'), (re.compile('\\(\\(oc & 1\\) \\? GG : -GG\\)'), 'q27_d(oc, 2)'), (re.compile('BG \\+ \\(j\\*27\\+o\\)\\*3'), 'BG + ((((qq) & %d) + j)*27+o)*3' % (S - 8)), (re.compile('IP\\[12\\] = carry\\*8; if \\(carry > capQc\\)'), 'IP[12] = carry*%d; if (carry*%d > capQc)' % (S, S // 8))]
q27b_OKLEFT = [re.compile('carry\\*\\d+ > capQc'), re.compile('threadIdx\\.x\\s*&\\s*7'), re.compile('\\btl\\s*&\\s*7'), re.compile('tl\\s*&\\s*~7'), re.compile('\\btl\\s*>>\\s*3'), re.compile('threadIdx\\.x\\s*>>\\s*3'), re.compile('blockDim\\.x\\s*>>\\s*3'), re.compile('\\bcn\\s*>>'), re.compile('cn\\s*<\\s*8'), re.compile('\\(cn>>2\\)&1'), re.compile('j\\s*<\\s*8;'), re.compile('r\\s*\\*\\s*8'), re.compile('8\\s*\\*\\s*r\\b'), re.compile('oc8|o8_'), re.compile('const float GG = '), re.compile('oc<8; oc\\+\\+\\) \\{ const float\\* Qo = sw'), re.compile('LM\\s*\\+\\s*l\\*8|l\\*8\\s*\\+|\\(l-1\\)\\*8|\\(lf\\+1\\)\\*8|lf\\*8')]

def q27b__split(code):
    idx = []
    for m in re.finditer('^(?:__global__|__device__)[^\\n]*', code, re.M):
        h = re.sub('__launch_bounds__\\(\\s*\\d+(\\s*,\\s*\\d+)?\\s*\\)', '', m.group(0))
        mm = re.search('(\\w+)\\s*\\(', h)
        idx.append((m.start(), mm.group(1) if mm else None))
    if not idx:
        return (code, [])
    head = code[:idx[0][0]]
    parts = []
    for (i, (st, n)) in enumerate(idx):
        en = idx[i + 1][0] if i + 1 < len(idx) else len(code)
        parts.append((n, code[st:en]))
    return (head, parts)

def q27b__sub(b, old, new, n=1, name=''):
    c = b.count(old)
    if c != n:
        raise RuntimeError("q27 고침 '%s' 수 %d (기대 %d) in %s" % (old[:80], c, n, name))
    return b.replace(old, new)

def q27b__special(n, b):
    (S, L, NT) = (SLOTS(), LSL(), NPT())
    if n == 'rt_gauss':
        b = q27b__sub(b, 'ac[28]*0.125f', 'ac[28]*Q27W[oc]', 1, n)
        b = q27b__sub(b, 'ac[29]*0.125f', 'ac[29]*Q27W[oc]', 1, n)
        b = q27b__sub(b, 'for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];', 'if (oc >= %d) for (int k=0;k<9;k++) out[k] = (k %% 4 == 0) ? 1.f : 0.f;   // 절 CM101 빈 자리\n    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];' % NT, 1, n)
    elif n == 'gq_pq':
        b = q27b__sub(b, 'PC[(size_t)c*6+a]*0.125f', 'PC[(size_t)c*6+a]*Q27W[q & %d]' % (S - 1), 1, n)
    elif n == 'rt_mg_acc':
        b = q27b__sub(b, 'int c = q >> 3, oc = q & 7;', 'int c = (l == 1) ? (q >> %d) : (q >> 3), oc = (l == 1) ? (q & %d) : (q & 7);' % (L, S - 1), 1, n)
        b = q27b__sub(b, 'float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};', 'float tq[3] = {cx + 1.f + ((l == 1) ? q27_d(oc, 0) : ((oc & 4) ? GG : -GG)), cy + 1.f + ((l == 1) ? q27_d(oc, 1) : ((oc & 2) ? GG : -GG)), cz + 1.f + ((l == 1) ? q27_d(oc, 2) : ((oc & 1) ? GG : -GG))};', 1, n)
        return (b, False)
    elif n == 'rt_mv2':
        b = q27b__sub(b, 'mv2_core(j, on,', 'mv2_core_q27((int)(qq & %d), j, on,' % (S - 8), 1, n)
    elif n == 'rt_mv2s':
        b = q27b__sub(b, 'const float* bq = BG + j*81;', 'const float* bq = BG + ((qq & %d) + j)*81;' % (S - 8), 1, n)
        b = q27b__sub(b, 'const float* bg = BG + oc*81 + o*3;', 'const float* bg = BG + ((qq & %d) + oc)*81 + o*3;' % (S - 8), 1, n)
    elif n in ('fpi_p', 'lji_p'):
        b = q27b__sub(b, 'const float GG = 0.28867513459481287f;\n    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);', 'int lo_[3]; float t[3]; for (int a=0;a<3;a++) q27_ax(fx[p*3+a], &lo_[a], &t[a]);   // 절 CM183 보간', 1, n)
        b = q27b__sub(b, 'for (int oc=0; oc<8; oc++) {\n        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);', 'for (int o8_=0; o8_<8; o8_++) {\n        float w = ((o8_ & 4) ? t[0] : 1.f - t[0]) * ((o8_ & 2) ? t[1] : 1.f - t[1]) * ((o8_ & 1) ? t[2] : 1.f - t[2]);\n        int oc = (lo_[0] + ((o8_ >> 2) & 1))*%d + (lo_[1] + ((o8_ >> 1) & 1))*%d + (lo_[2] + (o8_ & 1));' % (q27b_ST['np'] ** 2, q27b_ST['np']), 1, n)
    elif n == 'aq_p2g_sd':
        b = q27b__sub(b, 'float a[8][4]; for (int o=0;o<8;o++)', 'float a[%d][4]; for (int o=0;o<%d;o++)' % (S, S), 1, n)
        b = q27b__sub(b, 'for (int o=0;o<8;o++) for (int k=0;k<4;k++) Sa[((size_t)c*8+o)*4+k] = a[o][k];', 'for (int o=0;o<%d;o++) for (int k=0;k<4;k++) Sa[((size_t)c*%d+o)*4+k] = a[o][k];' % (S, S), 1, n)
    elif n in ('ao_gath_q', 'gq_gath'):
        b = re.sub('for \\(int oc=0;oc<8;oc\\+\\+\\)', 'for (int oc=0;oc<%d;oc++)' % S, b)
        b = b.replace('qv[(size_t)c*8+oc]', 'qv[(size_t)c*%d+oc]' % S)
    elif n in ('sp_csum', 'sp_q2n'):
        b = re.sub('for \\(int oc=0; oc<8; oc\\+\\+\\)', 'for (int oc=0; oc<%d; oc++)' % S, b)
    return (b, True)

def q27b__mv2_core_q27(b):
    b = q27b__sub(b, '__device__ __forceinline__ void mv2_core(int j,', '__device__ __forceinline__ void mv2_core_q27(int gofs, int j,', 1, 'mv2_core')
    b = q27b__sub(b, 'const float* bq = BG + j*81;', 'const float* bq = BG + (gofs + j)*81;', 1, 'mv2_core')
    b = q27b__sub(b, 'const float* bg = BG + oc*81 + o*3;', 'const float* bg = BG + (gofs + oc)*81 + o*3;', 1, 'mv2_core')
    return b

def q27b_transform(code):
    (head, parts) = q27b__split(code)
    if not parts or 'IP[12]' not in code:
        return code
    (out, rep, unk) = ([], {}, [])
    for (n, b) in parts:
        if n is None or not q27b_PAT.search(b) or n in q27b_COARSE or (n.startswith(('rt_mg_', 'ao_mg_', 'cvc')) and n != 'rt_mg_acc'):
            out.append(b)
            if n == 'mv2_core' and 'mv2_core(j, on,' in code:
                out.append(q27b__mv2_core_q27(b))
            continue
        (b2, gen) = q27b__special(n, b)
        if gen:
            for (rx, rp) in _gen():
                b2 = rx.sub(rp, b2)
        left = []
        for ln in b2.split('\n'):
            core = ln.split('//')[0]
            for o_ in q27b_OKLEFT:
                core = o_.sub(' ', core)
            if q27b_PAT.search(core) and n != 'rt_mg_acc':
                left.append(ln.strip()[:160])
        if left:
            unk.append((n, left))
        rep[n] = int(b2 != b)
        out.append(b2)
    q27b_ST['kern'].update(rep)
    if unk:
        q27b_ST['left'] = q27b_ST.get('left', []) + [(n, l[:4]) for (n, l) in unk]
    new = head + ''.join(out)
    i = new.find('extern "C" {')
    new = new[:i + len('extern "C" {')] + q27b__help() + new[i + len('extern "C" {'):] if i >= 0 else q27b__help() + new
    q27b_ST['ntr'] += 1
    return new

def q27b_tables(Ai):
    Ai64 = Ai.double().cpu().numpy().reshape(3, 3)
    P = q27b_ST['np']
    BW = np.zeros((SLOTS(), 27))
    BG = np.zeros((SLOTS(), 27, 3))
    for oc in range(NPT()):
        ii = (oc // (P * P), oc // P % P, oc % P)
        fxv = [1.0 + (XI()[ii[a]] - 0.5) for a in range(3)]
        w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for o in range(27):
            (i, j, k) = (o // 9, o // 3 % 3, o % 3)
            BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
            gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
            BG[oc, o] = gy @ Ai64
    return (BW, BG)
q27b_COARSE_K = ('rt_mg_', 'ao_mg_', 'cvc')

def q27b__install_q27():
    import torch
    import hexfem as ART
    bases = [ART.an_rt_RTSolver]
    try:
        import hexfem as V0
        bases.append(V0.an_rt_v0_RTSolver)
    except Exception:
        pass
    P = ART.an_rt_P
    for C in bases:
        if C.__dict__.get('_q27', False):
            continue
        C._q27 = True
        (oa, oi, ok) = (C.alloc, C.__init__, C.k)

        def alloc(s, need, _oa=oa):
            if need is None:
                y = s.x.double() @ s.Ai.double().T
                b = torch.floor(y - 0.5)
                nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
                nn = nb[0] * nb[1] * nb[2]
                ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
                uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=s.dev)).sum(1)).numel()
                need = (nn, ncb, SLOTS() // 8 * uc)
                s.nb0 = nb
                q27b_ST['nalloc'] += 1
                q27b_ST['alloc_last'] = dict(nn=nn, ncb=ncb, uc=uc)
            else:
                raise RuntimeError('q27: alloc(need) 경로는 지원 안 함')
            _oa(s, need)
            q27b_ST['capQc'] = int(s.capQc)
            q27b__info()

        def init(s, *a, _oi=oi, **kw):
            _oi(s, *a, **kw)
            if getattr(s, '_q27tab', False):
                return
            s._q27tab = True
            (BW, BG) = q27b_tables(s.Ai)
            (s.BW8, s.BG8) = (s.BW, s.BG)
            s.BW = torch.tensor(BW, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            s.BG = torch.tensor(BG, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            q27b_ST['tab'] = dict(bw_rowsum_min=float(BW[:NPT()].sum(1).min()), bw_rowsum_max=float(BW[:NPT()].sum(1).max()), bg_rowsum_absmax=float(np.abs(BG[:NPT()].sum(1)).max()), w_sum=float(sum(q27b__w32())), cls=type(s).__name__)
            q27b__info()

        def k(s, name, n, args, b=128, _ok=ok):
            if name.startswith(q27b_COARSE_K) and getattr(s, 'BG8', None) is not None:
                (pb, pw) = (P(s.BG), P(s.BW))
                args = tuple((P(s.BG8) if isinstance(x, np.uint64) and x == pb else P(s.BW8) if isinstance(x, np.uint64) and x == pw else x for x in args))
                q27b_ST['ncoarse'] = q27b_ST.get('ncoarse', 0) + 1
            return _ok(s, name, n, args, b)
        (C.alloc, C.__init__, C.k) = (alloc, init, k)
    mods = [ART]
    try:
        import hexfem as AORT
        mods.append(AORT)
    except Exception:
        pass
    for M in mods:
        for fn in ('split_step', 'adapt_step'):
            f0 = getattr(M, fn, None)
            if f0 is None or getattr(f0, '_q27', False):
                continue

            def w(s, *a, _f0=f0, **kw):
                try:
                    q27b__check(s)
                except Exception as e:
                    q27b_ST['chk_err'] = repr(e)
                return _f0(s, *a, **kw)
            w._q27 = True
            setattr(M, fn, w)
    try:
        import hexfem as V0
        ox = V0.an_rt_v0_RTSolver.x_orig
        if not getattr(ox, '_q27', False):

            def x_orig(s, _ox=ox):
                try:
                    q27b__check(s)
                except Exception as e:
                    q27b_ST['chk_err'] = repr(e)
                return _ox(s)
            x_orig._q27 = True
            V0.an_rt_v0_RTSolver.x_orig = x_orig
    except Exception as e:
        q27b_ST['err'] = 'x_orig hook: ' + repr(e)
    q27b_ST['installed'] = True
    q27b__info()
q27b__NCHK = [0]

def q27b__check(s):
    import torch
    q27b__NCHK[0] += 1
    i = q27b__NCHK[0]
    if not (i <= 6 or i % 10 == 0) or not q27b_ST['dg']:
        return
    with torch.no_grad():
        torch.cuda.synchronize()
        (nq, nc) = (int(s.IP[12]), int(s.IP[11]))
        if nq <= 0:
            return
        V = s.Vq[:nq].double()
        m = s.mq[:nq].double()
        sl = torch.arange(nq, device=V.device) & SLOTS() - 1
        dmy = sl >= NPT()
        r = dict(call=i, nq=nq, nc=nc, nq_ok=int(nq == SLOTS() * nc), sumV=float(V.sum()), sumvol=float(s.vol.double().sum()), summ=float(m.sum()), summp=float(s.m.double().sum()), dummyV=float(V[dmy].abs().sum()), dummym=float(m[dmy].abs().sum()), nV0=int((V[~dmy] <= 0).sum()), nreal=int((~dmy).sum()), Fdet_min=float(torch.linalg.det(s.Fq[:nq][~dmy].double().reshape(-1, 3, 3)).min()))
        with open(os.path.join(q27b_ST['out'], 'q27.jsonl'), 'a') as f_:
            f_.write(json.dumps(r) + '\n')
q27b_JT = dict(h=None, prev=None, ck=None, k=0, rows=[], f0=29, f1=99, err=None)

def q27b__install_jit():
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_finterp ----------------
import torch
finterp_ST = dict(mode=0, dg=0, out='.', nsrc=0, nalloc=0, ngauss=0, ncommit=0, nfpi=0, nframe=0, err=None)
MARK = '// ---- 절 CM92 finterp'
LOG = [None]

def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())
FI_SRC = cu('FI_SRC')
SIG_ADD = ', int fi_mode, const float* fi_Jq0, const float* fi_Jp0, float* fi_dg'
C_OLD = '    if (c < 0) return;\n'
C_NEW = '    if (c < 0) return;\n    float fi_sJ = 0.f, fi_sJr = 0.f, fi_ws = 0.f; bool fi_j0b = false; float fi_jpG = fi_det(F + (size_t)p*9);   // 절 CM92\n'
E_OLD = '        const float* e = Fe + (size_t)(c*8 + oc)*28;\n'
E_NEW = '        const float* e = Fe + (size_t)(c*8 + oc)*28;\n        { float jq_ = fi_det(e + 18); fi_ws += w; fi_sJ += w*jq_; if (fi_Jq0) { float j0_ = fi_Jq0[c*8 + oc]; if (j0_ > 0.f) fi_sJr += w*jq_/j0_; else fi_j0b = true; } }   // 절 CM92\n'
OK_OLD = '    bool ok = true;\n'
OK_NEW = '    {   // 절 CM92: 부피 J 만 모드별 (회전·로그 경로이고 뒤집힌 점이 안 낀 입자만; 유클리드 대체·뒤집힘은 지금 정의 그대로)\n        if (fi_ws > 0.f) { fi_sJ /= fi_ws; fi_sJr /= fi_ws; }\n        float jb_ = fi_det(out), jp0_ = fi_Jp0 ? fi_Jp0[p] : 0.f;\n        bool rsp_ = !anyb && !bad;\n        if (rsp_) {\n            float trL_ = sL[0] + sL[4] + sL[8];\n            if (fi_mode == 1) { float sc_ = expf((logf(fi_sJ) - trL_) / 3.f); for (int k=0;k<9;k++) out[k] *= sc_; }\n            else if (fi_mode == 2 && !fi_j0b && jp0_ > 0.f) { float sc_ = expf((logf(jp0_ * fi_sJr) - trL_) / 3.f); for (int k=0;k<9;k++) out[k] *= sc_; }\n        }\n        if (fi_dg) { float* d_ = fi_dg + (size_t)p*8; d_[0] = jp0_; d_[1] = fi_jpG; d_[2] = fi_det(out); d_[3] = fi_sJ; d_[4] = jb_; d_[5] = rsp_ ? 0.f : 1.f; d_[6] = anyb ? 1.f : 0.f; d_[7] = fi_j0b ? 1.f : 0.f; }\n    }\n    bool ok = true;\n'
HEAD = '__global__ void fpi_p('

def patch(code):
    if MARK in code or HEAD not in code:
        return code
    assert code.count(HEAD) == 1
    i = code.index(HEAD)
    j = code.find('__global__', i + len(HEAD))
    j = len(code) if j < 0 else j
    b = code[i:j]
    k = b.index(') {')
    b = b[:k] + SIG_ADD + b[k:]
    for (o_, n_) in ((C_OLD, C_NEW), (E_OLD, E_NEW), (OK_OLD, OK_NEW)):
        assert b.count(o_) == 1, ('절 CM92 fpi_p 고칠 글자', o_, b.count(o_))
        b = b.replace(o_, n_, 1)
    finterp_ST['nsrc'] += 1
    return code[:i] + MARK + '\n' + FI_SRC + b + code[j:]

def finterp__dump():
    try:
        with open(os.path.join(finterp_ST['out'], 'fiinfo.json'), 'w') as f_:
            json.dump(finterp_ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass

def _wm(v, w):
    ok = torch.isfinite(v) & (w > 0)
    if not bool(ok.any()):
        return None
    v = v[ok].double()
    w = w[ok].double()
    return float((v * w).sum() / w.sum())

def finterp__hook(R, fa):
    (oa, ok, os_) = (R.alloc, R.k, R.step)

    def alloc(s, need):
        r = oa(s, need)
        fa(s)
        return r

    def on(s):
        return getattr(s, 'fiJ0', None) is not None and getattr(s, 'Fe', None) is not None

    def k(s, name, n, args, b=128):
        if not on(s):
            return ok(s, name, n, args, b)
        i32 = np.int32
        if name == 'rt_gauss' and (finterp_ST['mode'] == 2 or finterp_ST['dg']):
            r = ok(s, name, n, args, b)
            ok(s, 'fi_jg', n, (args[0], P(s.Fq), P(s.fiJ0)), 128)
            ok(s, 'fi_jp', s.N, (i32(s.N), P(s.F), P(s.fiJp0)), 256)
            finterp_ST['ngauss'] += 1
            return r
        if name in ('ao_commit', 'aq_commit') and finterp_ST['dg']:
            ok(s, 'fi_jx', n, tuple(args[:5]) + (P(s.fiJX),), 128)
            finterp_ST['ncommit'] += 1
            return ok(s, name, n, args, b)
        if name == 'fpi_p':
            finterp_ST['nfpi'] += 1
            return ok(s, name, n, tuple(args) + (i32(finterp_ST['mode']), P(s.fiJ0), P(s.fiJp0), P(s.fiD)), b)
        return ok(s, name, n, args, b)

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if on(s):
            finterp_ST['nframe'] += 1
            if LOG[0] is not None and s.fiD is not None:
                try:
                    nq = int(s.IP[12].item())
                    N = s.N
                    mq = s.mq[:nq]
                    m = s.m[:N]
                    D = s.fiD[:N * 8].view(N, 8)
                    JE = torch.linalg.det(s.Fe[:nq, 18:27].reshape(nq, 3, 3).double()).float()
                    jn = D[:, 2]
                    fin = torch.isfinite(jn)
                    o = dict(f=finterp_ST['nframe'], t=float(a[0]) if a else None, nq=nq, N=N, Jq0=_wm(s.fiJ0[:nq], mq), JqX=_wm(s.fiJX[:nq], mq), JqE=_wm(JE, mq), Jp0=_wm(D[:, 0], m), JpG=_wm(D[:, 1], m), JpA=_wm(jn, m), Jar=_wm(D[:, 3], m), Jbase=_wm(D[:, 4], m), euc=_wm(D[:, 5], m), invq=_wm(D[:, 6], m), j0b=_wm(D[:, 7], m), medA=float(jn[fin].median()) if bool(fin.any()) else None, med0=float(D[:, 0].median()), nneg=int((jn <= 0).sum()), qneg=int(((JE <= 0) & (mq > 0)).sum()))
                    LOG[0].write(json.dumps(o) + '\n')
                    LOG[0].flush()
                except Exception as e:
                    finterp_ST['err'] = repr(e)
            if finterp_ST['nframe'] in (1, 2, 5) or finterp_ST['nframe'] % 25 == 0:
                finterp__dump()
        return r
    (R.alloc, R.k, R.step) = (alloc, k, step)

def finterp_install(out, mode=0, dg=0):
    import hexfem as ART
    import hexfem as AORT
    finterp_ST.update(mode=int(mode), dg=int(dg), out=out)
    if finterp_ST['dg']:
        LOG[0] = open(os.path.join(out, 'fi.jsonl'), 'w')
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if 'code' in kw:
            kw['code'] = patch(kw['code'])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule

    def fa(s):
        if getattr(s, 'Fe', None) is None:
            return
        capQ = s.Fe.shape[0]
        NP = s.F.numel() // 9
        if getattr(s, 'fiJ0', None) is not None and s.fiJ0.numel() == capQ and (s.fiJp0.numel() == NP):
            return
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.fiJ0 = Z(capQ)
        s.fiJX = Z(capQ)
        s.fiJp0 = Z(NP)
        s.fiD = Z(NP * 8) if finterp_ST['dg'] else None
        finterp_ST['nalloc'] += 1
    import hexfem as ART0
    for R in (ART.an_rt_RTSolver, ART0.an_rt_v0_RTSolver):
        finterp__hook(R, fa)
    oao = AORT.AORTSolver.alloc

    def alloc_ao(s, need):
        r = oao(s, need)
        fa(s)
        return r
    AORT.AORTSolver.alloc = alloc_ao
    print(f"[절 CM92] finterp 켬 -- 모드 {finterp_ST['mode']} (0 지금 정의, 1 J 산술, 2 J 증분), 진단 {finterp_ST['dg']}", flush=True)
    finterp__dump()
