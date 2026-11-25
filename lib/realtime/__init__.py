# ---------------- fem_speed3 ----------------
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of realtime.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realtime.cu'), encoding='utf-8', newline='').read()
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
import os
import numpy as np
import torch

def _bw(fx):
    raise NotImplementedError('not used by the paper settings')

class ULDiag:

    def __init__(s, out):
        raise NotImplementedError('not used by the paper settings')

    def frame(s, S, fi):
        raise NotImplementedError('not used by the paper settings')
TB_SRC = cu('TB_SRC')

class TB:
    mod = None
    fn = {}
    where = 1
    B = 16.7
    m = 0.5
    log = None

    @classmethod
    def k(cls, name, args):
        import cupy as cp
        import hexfem as RTC
        fn = cls.fn.get(name)
        if fn is None:
            fn = cls.fn[name] = cls.mod.get_function(name)
        with RTC._Stream():
            fn((1,), (1,), args)

    @classmethod
    def g(cls, name, dt, n=1):
        import cupy as cp
        return cp.ndarray((n,), dtype=dt, memptr=cls.mod.get_global(name))

def _xf(s):
    c2 = getattr(s, 'c2', None) or {}
    if c2.get('xf'):
        return int(c2['xf'])
    en = (getattr(s, 'wf', None) or {}).get('en') or {}
    return int(en.get('xf') or 0)

def _skip_ptr(s):
    if not hasattr(s, '_tb_skip'):
        s._tb_skip = np.uint64(int(s.mod.get_global('AF_SKIP')))
    return s._tb_skip

def tb_install(B, m, out, where=1):
    import cupy as cp
    import hexfem as ART
    import hexfem as AORT
    import hexfem as C3
    import hexfem as WRX
    TB.mod = cp.RawModule(code=TB_SRC)
    for nm in ('tb_t0', 'tb_tick', 'tb_gate', 'tb_end', 'tb_unmark'):
        TB.fn[nm] = TB.mod.get_function(nm)
    (TB.B, TB.m, TB.where) = (float(B), float(m), int(where))
    Bns = np.uint64(int(round((TB.B - TB.m) * 1000000.0)))
    P = lambda t: np.uint64(t.data_ptr())

    def wrap_record(cls):
        o = cls.record

        def record(s, t_frame, K=None):
            s._tbK = int(K if K is not None else getattr(s, 'Kcur', getattr(s, 'K', 1)))
            s._tbcap = bool(torch.cuda.is_current_stream_capturing())
            if s._tbcap:
                TB.k('tb_t0', ())
            r = o(s, t_frame, K)
            if s._tbcap:
                TB.k('tb_end', ())
            return r
        cls.record = record

    def gate(s):
        if getattr(s, '_tbK', 1) == 1 and getattr(s, '_tbcap', False):
            TB.k('tb_gate', (P(s.CT), _skip_ptr(s), Bns))

    def wrap_ctl(cls, ready):
        o = cls.ctl

        def ctl(s, op, it=0):
            on = getattr(s, 'adapt', False) and ready(s) and getattr(s, '_tbcap', False)
            xf = _xf(s)
            if on and op == 2 and (xf > 0) and (it >= xf if TB.where == 1 else it > xf):
                gate(s)
            if on and op == 8:
                TB.k('tb_unmark', (P(s.CT),))
            o(s, op, it)
            if on and op in (2, 8):
                TB.k('tb_tick', (P(s.CT), np.int32(op)))
        cls.ctl = ctl
    wrap_record(ART.an_rt_RTSolver)
    wrap_record(AORT.AORTSolver)
    wrap_ctl(ART.an_rt_RTSolver, lambda s: True)
    wrap_ctl(AORT.AORTSolver, lambda s: getattr(s, 'ao_ready', False))
    o_mark = C3.mark

    def mark(s, kind):
        if kind == ('q' if TB.where == 1 else 'g'):
            gate(s)
        o_mark(s, kind)
    C3.mark = mark
W1_OLD = ('    uf_ = uf_ && ulm[n];', '        if (uf_) a1 = 0.f;')
W1_NEW = ('    int ufa_ = uf_; uf_ = uf_ && ulm[n];                                       // 절 CM39 W1: 덜 풀린 걸음 (S 밖 포함)', '        if (ufa_) a1 = 0.f;                                                    // 절 CM39 W1: 가속도 기억 0 은 몸 전체')
X3E_SRC = cu('X3E_SRC')

class X3E:
    mod = None
    T = 1000000.0

class SegGraph3:

    def __init__(s, sv, segs, kinds):
        raise NotImplementedError('not used by the paper settings')

    def replay(self):
        raise NotImplementedError('not used by the paper settings')

def x3e_frame_n():
    if X3E.mod is None:
        return None
    import cupy as cp
    a = cp.ndarray((1,), dtype=cp.int32, memptr=X3E.mod.get_global('X3E_N'))
    n = int(a[0])
    a[0] = 0
    return n

def frame_hook_install(out):
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX
    log = open(os.path.join(out, 's3.jsonl'), 'w')
    dep = [0]

    def outer(fn):

        def w(s, *a, **kw):
            if dep[0] == 0 and TB.mod is not None:
                TB.g('TB_NEWF', np.int32)[0] = 1
            dep[0] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                dep[0] -= 1
            if dep[0] == 0:
                rec = {}
                if TB.mod is not None:
                    fr = int(TB.g('TB_FR', np.int32)[0])
                    (mit, mtl) = (TB.g('TB_MIT', np.uint64, 8).get(), TB.g('TB_MTL', np.uint64, 8).get())
                    (t0, te) = (int(TB.g('TB_T0', np.uint64)[0]), int(TB.g('TB_TE', np.uint64)[0]))
                    rec.update(cut=int(TB.g('TB_CUT', np.int32)[0]), gates=int(TB.g('TB_GATES', np.int32)[0]), cit=round(float(mit.max()) * 1e-06, 3), ctail=round(float(mtl.max()) * 1e-06, 3), it_f=round(float(mit[fr]) * 1e-06, 3), ela=round((te - t0) * 1e-06, 3))
                n = x3e_frame_n()
                if n is not None:
                    rec['x3e'] = n
                ic = inv_counts(s)
                if ic is not None:
                    rec['inv'] = ic
                log.write(json.dumps(rec) + '\n')
                log.flush()
            return r
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)

class INV:
    mode = 0

def inv_counts(s):
    if INV.mode != 2:
        return None
    import cupy as cp
    a = cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global('S3_INVOK'))
    b = cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global('S3_INVBAD'))
    o = [int(a[0]), int(b[0])]
    a[0] = 0
    b[0] = 0
    return o
W3_SRC = cu('W3_SRC')
W3_UL = '    uf_ = uf_ && ulm[n];'

class W3:
    mod = None
    fn = None

# ---------------- fem_speed4 ----------------
import sys
import re
import time
speed4_SP_SRC = cu('speed4_SP_SRC')
speed4_SPN = 65536
speed4_CATS = ['P2G', '걸음 시작', 'W̄', 'R2·A2', '요소', '장벽 기울기', '기울기·판정', 'PCG', '장벽·뒤집힘 보폭', '선탐색 에너지', '걸음 끝', 'RM', 'G2P', '구간 사이', '시도 사이', '기타']
speed4_CI = {c: i for (i, c) in enumerate(speed4_CATS)}
speed4_NEWTON_C = {speed4_CI[c] for c in ('요소', '장벽 기울기', '기울기·판정', 'PCG', '장벽·뒤집힘 보폭', '선탐색 에너지')}
speed4_BEGIN = 1 << 14
speed4_INHERIT = {'rt_zero_i', 'rt_zero_f', 'rt_zero_d', 'rt_ctl', 'rt_sc_clear', 'rt_reset'}
speed4_N_ELEM = {'aq_elem', 'ao_elem', 'ao_elem_s'}
speed4_N_BAR = {'rt_bar_q', 'rt_bar_n', 'wb_cnt'}
speed4_N_GRAD = {'rt_grad', 'rt_grad_c', 'uf_g', 'en_gate', 'c2_gate'}
N_PCG = {'rt_bar_mv', 'rt_ap', 'rt_ap_c', 'rt_cg1', 'rt_cg1_c', 'rt_cg2', 'rt_cg2_c', 'rt_sc_roll', 'rt_cg1m', 'rt_dotz', 'rt_pinit', 'ao_warm'}
N_STEP = {'rt_ytry', 'rt_bar_kfe', 'wb_kmin', 'rt_lsup', 'rt_lsup_c', 'uf_min', 'ao_tvec', 'aw_tdir', 'ws4_chk'}
T_RM = {'aq_commit', 'ao_commit'}
T_G2P = {'ao_g2p_a', 'aq_g2p', 'ao_g2p', 'fpi_q', 'fpi_p', 'lji_p', 'pf_g2p'}

class speed4_SP:
    mod = None
    fn = None
    on = False
    unk = set()

def sp_g(name, dt, n=1):
    raise NotImplementedError('not used by the paper settings')

def speed4_sp_frame():
    raise NotImplementedError('not used by the paper settings')
WS4_HEAD = '__device__ int WS4_DN; __device__ unsigned long long WS4_BAD, WS4_NCLEAN, WS4_NROUND;\n'
WS4_K = cu('WS4_K')
WS4_SKIPK = ('ws_cand', 'ws_inv_q', 'ws_nk', 'ws_bk', 'ws_merge')

class WS4:
    mode = 0

def ws4_src(src):
    raise NotImplementedError('not used by the paper settings')

def ws4_install(mode):
    import hexfem as AORT
    from hexfem import ao_rt_P as P
    WS4.mode = int(mode)
    cls = AORT.AORTSolver
    (o_init, o_k) = (cls.__init__, cls.k)

    def __init__(s, *a, **kw):
        if not getattr(AORT, 'ao_rt__ws4_src', False):
            AORT.ao_rt__ws4_src = True
            o_s = AORT._src
            AORT._src = lambda *a_, **k_: ws4_src(o_s(*a_, **k_))
        return o_init(s, *a, **kw)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == 'ws_bk' and int(args[-1]) == 1:
            o_k(s, 'ws4_chk', 1, (args[-2],), 1)
        return r
    (cls.__init__, cls.k) = (__init__, k)

def ws4_counts(s):
    raise NotImplementedError('not used by the paper settings')

class X1:
    on = False
    level = 0
    T = 0.0
    hits = 0
    x2 = 0
    x3e = 0
XB_SRC = cu('XB_SRC')

class XB:
    mod = None
    fn = {}
    hb = []
    hmax = 0.0

def _xb(name, grid, args):
    raise NotImplementedError('not used by the paper settings')

def _hb(tsy):
    raise NotImplementedError('not used by the paper settings')

def _xb_buf(s):
    raise NotImplementedError('not used by the paper settings')

def _d2h(s):
    raise NotImplementedError('not used by the paper settings')

def _cpn(src, dst, IP, ci, mul):
    raise NotImplementedError('not used by the paper settings')

def _x_read(s):
    raise NotImplementedError('not used by the paper settings')

def _restore(s):
    raise NotImplementedError('not used by the paper settings')
X2C_F = ('U', 'vk', 'gaA', 'mn', 'mv', 'mqn', 'Fq', 'Vq', 'mq', 'xq0', 'Sq', 'bx')
X2C_I = ('cidx', 'cb', 'act')

def _x2c_snap(s, first):
    raise NotImplementedError('not used by the paper settings')

def _x2c_cmp(a, b):
    raise NotImplementedError('not used by the paper settings')

class X2C:
    rec = []

def x1_install(level=1, x3e=0.0):
    import hexfem as AORT
    import hexfem as C3
    import hexfem as WRX
    (X1.on, X1.level, X1.T) = (True, int(level), float(x3e))
    assert X1.T <= 0 or X1.level == 2, '--s4e 는 --s4x 2 위'
    cls = AORT.AORTSolver
    (o_rec, o_k, o_ctl) = (cls.record, cls.k, cls.ctl)

    def record(s, t_frame, K=None):
        s._x1K = int(K if K is not None else getattr(s, 'Kcur', getattr(s, 'K', 1)))
        s._x1ks = -1
        cap = getattr(s, '_c3cap', None) is not None
        if cap:
            s._k4b = ['start']
        elif X1.level >= 2 and getattr(s, 'wbar', None) is not None:
            if getattr(s, '_x2bx', None) is None or s._x2bx.numel() != s.bx.numel():
                s._x2bx = torch.empty_like(s.bx)
            if getattr(s, '_x2y', None) is None or s._x2y.numel() != s.y.numel():
                s._x2y = torch.empty_like(s.y)
            dq = getattr(s, 'e5DQ', None)
            if dq is not None and (getattr(s, '_x2dq', None) is None or s._x2dq.numel() != dq.numel()):
                s._x2dq = torch.empty_like(dq)
        if not cap:
            _xb_buf(s)
        return o_rec(s, t_frame, K)

    def k(s, name, n, args, b=128):
        if X1.level >= 2 and name == 'ao_floor_n':
            s._x1ks = int(getattr(s, '_x1ks', -1)) + 1
            if s._x1ks == 0 and getattr(s, '_x1K', 0) == 1 and (getattr(s, 'wbar', None) is not None):
                _cpn(s.bx, s._x2bx, s.IP, 12, 4)
                _cpn(s.y, s._x2y, s.IP, 9, 3)
                if getattr(s, '_x2dq', None) is not None:
                    _cpn(s.e5DQ, s._x2dq, s.IP, 12, 1)
                if getattr(s, '_c3cap', None) is not None:
                    C3.mark(s, 's')
        r = o_k(s, name, n, args, b)
        if name == 'uf_end' and getattr(s, '_c3cap', None) is not None and (getattr(s, '_x1K', 0) == 1):
            P_ = lambda t: np.uint64(t.data_ptr())
            _xb('xb_x', 1, (P_(s.IP), P_(s.ufl), P_(s.ufg), np.float32(float(getattr(s, 'ksbig', 0.0) or 0.0)), P_(s._xbd)))
            _d2h(s)
            C3.mark(s, 'x')
        return r

    def ctl(s, op, it=0):
        r = o_ctl(s, op, it)
        if X1.T > 0 and op == 3 and (it == 0) and (getattr(s, '_c3cap', None) is not None) and (getattr(s, '_x1K', 0) == 1):
            P_ = lambda t: np.uint64(t.data_ptr())
            _xb('xb_r', 1, (P_(s.IP), P_(s.SC), P_(s.CD), np.float64(X1.T), P_(s._xbd)))
            _d2h(s)
            C3.mark(s, 'r')
        return r
    (cls.record, cls.k, cls.ctl) = (record, k, ctl)
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, '_c3cap', None) is not None and getattr(s, '_k4b', None) is not None:
            s._k4b.append(kind)
        return o_mark(s, kind)
    C3.mark = mark
    (o_init, o_rep) = (C3.SegGraph.__init__, C3.SegGraph.replay)

    def __init__(self, s, segs):
        o_init(self, s, segs)
        kb = getattr(s, '_k4b', None)
        self.k4 = list(kb) if kb is not None and len(kb) == len(segs) else None
        assert self.k4 is None or (self.k4.count('x') <= 1 and self.k4.count('s') <= 1 and (self.k4.count('r') <= 1))
        s._k4b = None

    def replay(self):
        s = self.s
        st = getattr(s, '_s4st', None)
        if self.k4 is None or not ('x' in self.k4 and (getattr(s, '_x1ok', False) or st is not None)):
            return o_rep(self)
        mode = int(s.xg)
        cst = s.c3st
        skip = False
        i = 0
        si = self.k4.index('s') if 's' in self.k4 else None
        if st is not None:
            st['e0'] = torch.cuda.Event(enable_timing=True)
            st['e0'].record()
        tsy = None
        while i < len(self.segs):
            (cond, g) = self.segs[i]
            kd = self.k4[i]
            if tsy is not None and tsy[2] != i:
                tsy = None
            if kd == 'r' and st is not None and (X1.T > 0) and (not st['jump']) and (st['won'] == 0) and (si is not None):
                torch.cuda.current_stream().synchronize()
                tsy = (time.perf_counter(), 'r', i)
                if int(s._xbn[2]) == 0 and int(s._xbn[3]) == 1:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    st['ev'] = ev
                    _restore(s)
                    S3 = sys.modules[__name__]
                    if S3.TB.mod is not None:
                        S3.TB.g('TB_RUN', np.int32)[0] = 0
                        torch.cuda.synchronize()
                    WRX.on(s, 1)
                    st['won'] = 1
                    st['jump'] = 'x3e'
                    X1.x3e += 1
                    (i, skip) = (si, False)
                    continue
            if kd == 's' and X1.level == 3 and (int(s.IP[13]) == 0):
                if getattr(s, '_x1won', 0) == 0:
                    s._x2s = _x2c_snap(s, False)
                elif getattr(s, '_x2c', None) is not None:
                    X2C.rec.append(dict(kind='다시', **_x2c_cmp(s._x2c, _x2c_snap(s, False))))
                    s._x2c = None
            xr = _x_read(s) if kd == 'x' else None
            if xr is not None:
                tsy = (time.perf_counter(), 'x', i)
            if kd == 'x' and xr[0] == 0:
                coll = s._x4coll
                if X1.level == 3 and getattr(s, '_x1won', 1) == 0 and (getattr(s, '_x2s', None) is not None):
                    xs = _x2c_snap(s, False)
                    r_ = _x2c_cmp(s._x2s, xs)
                    r_['bx_snap'] = _x2c_cmp(dict(IP=s._x2s['IP'], bx=s._x2s['bx']), dict(IP=s._x2s['IP'], bx=s._x2bx.reshape(-1)[:s._x2s['bx'].numel()].clone())).get('bx')
                    r_['y_snap'] = _x2c_cmp(dict(IP=s._x2s['IP'], U=s._x2s['_y']), dict(IP=s._x2s['IP'], U=s._x2y.reshape(-1)[:s._x2s['_y'].numel()].clone())).get('U')
                    if '_dq' in s._x2s:
                        r_['dq_snap'] = _x2c_cmp(dict(IP=s._x2s['IP'], U=s._x2s['_dq']), dict(IP=s._x2s['IP'], U=s._x2dq.reshape(-1)[:s._x2s['_dq'].numel()].clone())).get('U')
                    X2C.rec.append(dict(kind='같은 시도', coll=bool(coll), **r_))
                    s._x2s = None
                    if coll:
                        s._x2c = _x2c_snap(s, True)
                if coll and st is not None and (not st['jump']) and (st['won'] == 0) and (si is not None):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    st['ev'] = ev
                    st['nf0'] = 1
                    _restore(s)
                    WRX.on(s, 1)
                    st['won'] = 1
                    st['jump'] = 'x2'
                    X1.x2 += 1
                    (i, skip) = (si, False)
                    continue
                discard = getattr(s, '_x1ok', False) if st is None else coll and (st['depth'] > 0 or st['won'] == 0)
                if coll and discard:
                    X1.hits += 1
                    return
            if not cond:
                skip = False
                if mode == 2:
                    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    e0.record()
                    g.replay()
                    e1.record()
                    cst['ev'].append((0, e0, e1))
                else:
                    g.replay()
                _hb(tsy)
                tsy = None
                i += 1
                continue
            if not skip:
                skip = bool(int(s.CT[0]) != 0)
                tsy = (time.perf_counter(), kd, i)
            cst['sk' if skip else 'run'] += 1
            if mode == 1 and skip:
                i += 1
                continue
            if mode == 2:
                (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                e0.record()
                g.replay()
                e1.record()
                cst['ev'].append((2 if skip else 1, e0, e1))
            else:
                g.replay()
            _hb(tsy)
            tsy = None
            i += 1
    (C3.SegGraph.__init__, C3.SegGraph.replay) = (__init__, replay)
    o_on = WRX.on

    def on(s, v):
        s._x1won = int(v)
        return o_on(s, v)
    WRX.on = on
    o_split = AORT.ao_rt_split_step

    def split_step(s, t, h, sub_dt, depth):
        col = bool(getattr(s, 'wrx', None)) and getattr(s, '_x1won', None) == 0
        prev = getattr(s, '_x1ok', False)
        s._x1ok = int(depth) > 0 or col
        try:
            return o_split(s, t, h, sub_dt, depth)
        finally:
            s._x1ok = prev
    AORT.ao_rt_split_step = split_step
    if X1.level != 2:
        return
    o_col = WRX.split_col

    def split_col(s, t, h, sub_dt, depth, fn=None):
        if fn is not None or not isinstance(s, cls):
            return o_col(s, t, h, sub_dt, depth, fn)
        import math
        bk = s._bk()
        WRX.on(s, 0)
        base = (s.ga_c, s.nt, s.cg)
        AORT.set_K_ga(s, 1, h, sub_dt)
        st = s._s4st = dict(jump=None, won=0, depth=int(depth), ev=None, nf0=None, e0=None)
        try:
            redo = s.step(t)
        finally:
            s._s4st = None
        ms = float(s.last_ms)
        nf = C3.nfail(s, s.ufl, s.ufg, 1)
        if st['jump'] is None:
            tries = [(1, nf, ms, 0)]
            if nf == 0 or not math.isfinite(ms):
                return (redo, 1, tries)
            s._rs(bk)
            WRX.on(s, 1)
            (r1, K1, tr1) = AORT.ao_rt_split_step(s, t, h, sub_dt, depth)
            WRX.on(s, 0)
            s.last_ms = ms + float(s.last_ms)
            return (redo or r1, K1, tries + [(a, b, c, 1) for (a, b, c) in tr1])
        torch.cuda.synchronize()
        m1 = st['e0'].elapsed_time(st['ev']) if st['ev'] is not None and st['e0'] is not None else 0.0
        m1 = min(max(m1, 0.0), ms)
        tries = [(1, st['nf0'] if st['jump'] == 'x2' else -1, m1, 0), (1, nf, ms - m1, 1)]
        tot = ms
        K = 1
        if nf > 0 and math.isfinite(ms) and (depth > 0):
            wf = getattr(s, 'wf', {})
            for dp in range(1, depth + 1):
                K = 1 << dp
                s._rs(bk)
                if wf.get('ks_ga') is not None:
                    s.ga_c = wf['ks_ga']
                if wf.get('ks_nt'):
                    (s.nt, s.cg) = (int(wf['ks_nt']), int(wf['ks_cg']))
                AORT.set_K_ga(s, K, h, sub_dt)
                redo = s.step(t) or redo
                m_ = float(s.last_ms)
                nf_ = C3.nfail(s, s.ufl, s.ufg, K)
                tries.append((K, nf_, m_, 1))
                tot += m_
                if nf_ == 0 or not math.isfinite(m_):
                    break
            (s.ga_c, s.nt, s.cg) = base
        WRX.on(s, 0)
        s.last_ms = tot
        return (redo, K, tries)
    WRX.split_col = split_col
YL4_HEAD = '__device__ int YL4_DN; __device__ unsigned long long YL4_BAD, YL4_NCLEAN, YL4_NCHK;\n'
YL4_K = cu('YL4_K')

class YL4:
    mode = 0

def yl4_src(src):
    raise NotImplementedError('not used by the paper settings')

def yl4_install(mode):
    import hexfem as AORT
    YL4.mode = int(mode)
    cls = AORT.AORTSolver
    (o_init, o_k) = (cls.__init__, cls.k)

    def __init__(s, *a, **kw):
        if not getattr(AORT, 'ao_rt__yl4_src', False):
            AORT.ao_rt__yl4_src = True
            o_s = AORT._src
            AORT._src = lambda *a_, **k_: yl4_src(o_s(*a_, **k_))
        return o_init(s, *a, **kw)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == 'yl_chk':
            o_k(s, 'yl4_chk', 1, (args[-1],), 1)
        return r
    (cls.__init__, cls.k) = (__init__, k)

def yl4_counts(s):
    raise NotImplementedError('not used by the paper settings')

class GC4:
    mode = 0
    ev = []
    t0 = None
    win = False
    frozen = False

def gc_enter():
    raise NotImplementedError('not used by the paper settings')

def gc_exit():
    raise NotImplementedError('not used by the paper settings')

def speed4_frame_install(out):
    import hexfem as AORT
    import hexfem as WRX
    log = open(os.path.join(out, 's4.jsonl'), 'w')
    (dep, fr) = ([0], [0])

    def outer(fn):

        def w(s, *a, **kw):
            if dep[0] == 0 and speed4_SP.on:
                torch.cuda.synchronize()
                sp_g('SP_N', np.int32)[0] = 0
                torch.cuda.synchronize()
            if dep[0] == 0 and GC4.mode:
                gc_enter()
            dep[0] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                gce = gc_exit() if dep[0] == 0 and GC4.mode else None
            if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                rec = dict(t=fr[0])
                if gce is not None:
                    rec['gc'] = gce
                fr[0] += 1
                if speed4_SP.on:
                    rec['sp'] = speed4_sp_frame()
                    if speed4_SP.unk:
                        rec['unk'] = sorted((n_ for (_, n_) in speed4_SP.unk))
                        speed4_SP.unk.clear()
                if WS4.mode:
                    rec['ws4'] = ws4_counts(s)
                if YL4.mode:
                    rec['yl4'] = yl4_counts(s)
                if X1.on:
                    rec['hb'] = [round(XB.hmax, 3)] + XB.hb
                    (XB.hb, XB.hmax) = ([], 0.0)
                    rec['x1'] = [X1.hits, X1.x2, X1.x3e]
                    X1.hits = X1.x2 = X1.x3e = 0
                    if X2C.rec:
                        rec['x2c'] = X2C.rec
                        X2C.rec = []
                log.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log.flush()
            return r
        return w
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)

# ---------------- fem_speed5 ----------------
speed5_SP_SRC = cu('speed5_SP_SRC')
speed5_SPN = 65536
speed5_CATS = ['P2G', '걸음 시작', 'W̄', 'R2·A2', '벌점', 'H4 판정', '요소', '장벽 기울기', '기울기·판정', 'PCG', '보폭·선탐색', '걸음 끝', 'G2P', '구간 사이', '시도 사이', '기타']
speed5_CI = {c: i for (i, c) in enumerate(speed5_CATS)}
speed5_NEWTON_C = {speed5_CI[c] for c in ('요소', '장벽 기울기', '기울기·판정', 'PCG', '보폭·선탐색', '벌점')}
speed5_BEGIN = 1 << 14
speed5_INHERIT = {'rt_zero_i', 'rt_zero_f', 'rt_zero_d', 'rt_ctl', 'rt_sc_clear', 'rt_reset', 'sp5_st', 'sp_st'}
STEP0 = {'ao_step0_ga', 'rt_step0', 'ao_step0_th', 'rt_ga_step0', 'rt_step0g', 'rt_bdf_base'}
speed5_N_ELEM = {'aq_elem', 'ao_elem', 'ao_elem_s', 'rt_elem', 'rt_elem_c'}
speed5_N_BAR = {'rt_bar_q', 'rt_bar_n', 'rt_barn_q', 'wb_cnt', 'rt_bar_q_c', 'rt_barn_q_c'}
speed5_N_GRAD = {'rt_grad', 'rt_grad_c', 'uf_g', 'en_gate', 'c2_gate'}
N_PCG_P = ('rt_ap', 'rt_cg', 'rt_mg_', 'ao_mv', 'rt_bar_mv', 'rt_barn_mv', 'rt_dotz', 'rt_pinit', 'ao_warm', 'rt_sc_roll', 'mg_', 'rt_lm', 'rt_jac')
T_G2P_P = ('ao_g2p', 'aq_g2p', 'rt_g2p', 'fpi_', 'lji_', 'pf_g2p', 'g2p')

class speed5_SP:
    mod = None
    fn = None
    on = False

class O5:
    on = False
    S = None
    prev_end = None
    pend = []

class _FnDict(dict):

    def __init__(s_, src, owner):
        raise NotImplementedError('not used by the paper settings')

    def __getitem__(s_, name):
        raise NotImplementedError('not used by the paper settings')

def speed5_sp_frame():
    raise NotImplementedError('not used by the paper settings')

def ov_install():
    import cupy as cp
    import gsrender as RR
    O5.on = True
    O5.S = torch.cuda.Stream()
    torch.cuda.set_stream(O5.S)
    O5.cps = cp.cuda.ExternalStream(O5.S.cuda_stream)
    O5.cps.use()
    D = torch.cuda.default_stream()

    def wu(fn):

        def u(*a, **kw):
            if O5.prev_end is not None:
                torch.cuda.current_stream().wait_event(O5.prev_end)
            return fn(*a, **kw)
        return u

    def wr(fn):

        def r(G, U, t):
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            D.wait_event(ev)
            (es, ee) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            with torch.cuda.stream(D):
                es.record(D)
                img = fn(G, U, t)
                ee.record(D)
            O5.prev_end = ee
            O5.pend.append((es, ee))
            return img
        return r
    for nm in ('upd_pg0', 'upd_pg1', 'upd_gf0', 'upd_gf1'):
        setattr(RR, nm, wu(getattr(RR, nm)))
    for nm in ('ras_pg', 'ras_gf'):
        setattr(RR, nm, wr(getattr(RR, nm)))

def speed5_frame_install(out, mode):
    raise NotImplementedError('not used by the paper settings')

def speed5_install(out, p=0, o=0, m=1):
    raise NotImplementedError('not used by the paper settings')

# ---------------- fem_speed6 ----------------
import gzip

class speed6_C:
    out = '.'
    on = False
    o = 0
    lean = 0
    prof = None
    b = 0
    r = 0
    rw = 0

class Defer:

    def __init__(s, S, L, ljp=False, vmax=False):
        raise NotImplementedError('not used by the paper settings')

    def keep(s, fi, S, redo, K, tries):
        raise NotImplementedError('not used by the paper settings')

class W:
    q = []
    recs = []
    ref = None
    tref = None
    prev_P = None
    prev_ev1 = None
    prev_h1 = None
    fr = 0
    dep = 0
    ng = 0
    nsync = 0
    prof = None

def _wall(ev):
    return W.tref + W.ref.elapsed_time(ev)

def _resolve(force=False):
    while W.q:
        (rec, ev0, ev1, ras) = W.q[0]
        hold = None
        if isinstance(ras, dict):
            if ras.get('ev') is None:
                if not force:
                    break
                r_flush()
            hold = ras
            ras = ras.get('ev')
        last = ras if ras is not None else ev1
        if not force and (not last.query()):
            break
        if not force and rec.get('tb') is not None and (not rec['tb'][0].query()):
            break
        W.q.pop(0)
        if force:
            last.synchronize()
            ev1.synchronize()
        rec['per'] = round(ev0.elapsed_time(ev1), 4)
        if W.prev_ev1 is not None:
            rec['Ws'] = round(W.prev_ev1.elapsed_time(ev1), 4)
        W.prev_ev1 = ev1
        if rec.get('tb') is not None:
            (ok_, tb_) = rec.pop('tb')
            if force:
                ok_.synchronize()
            if ok_.query():
                rec['T'] = round((int(tb_[1]) - int(tb_[0])) * 1e-06, 4)
        P = _wall(last)
        rec['P'] = round(P, 4)
        rec['lat'] = round(P - rec['h0'], 4)
        rec['lag'] = round(_wall(ev0) - rec['h0'], 4)
        if ras is not None:
            rec['ras_end_after_ev1'] = round(ev1.elapsed_time(ras), 4)
        if hold is not None and hold.get('craw') is not None:
            rec['craw'] = round(hold['craw'][0].elapsed_time(hold['craw'][1]), 4)
        if W.prev_P is not None:
            rec['W'] = round(P - W.prev_P, 4)
        W.prev_P = P
        W.recs.append(rec)

def _flush_file():
    r_flush()
    _resolve(True)
    with open(os.path.join(speed6_C.out, 's6.jsonl'), 'w') as f:
        for r in W.recs:
            f.write(json.dumps(r) + '\n')

def _counters():
    og = torch.cuda.CUDAGraph.replay

    def replay(self, *a, **kw):
        W.ng += 1
        BK.mut += 1
        r_ = og(self, *a, **kw)
        if R.pend is not None:
            r_flush()
        return r_
    torch.cuda.CUDAGraph.replay = replay
    osy = torch.cuda.synchronize

    def synchronize(*a, **kw):
        W.nsync += 1
        return osy(*a, **kw)
    torch.cuda.synchronize = synchronize

def speed6_frame_install(out):
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX
    S5 = None
    if speed6_C.o:
        S5 = sys.modules[__name__]

    def outer(fn):

        def w(s, *a, **kw):
            top = W.dep == 0
            if top:
                if W.ref is None:
                    torch.cuda.synchronize()
                    W.ref = torch.cuda.Event(enable_timing=True)
                    W.ref.record()
                    W.ref.synchronize()
                    W.tref = 1000.0 * time.perf_counter()
                if speed6_C.prof and W.fr == speed6_C.prof[0]:
                    _prof_start()
                h0 = 1000.0 * time.perf_counter()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                (ng0, ns0) = (W.ng, W.nsync)
                BK.frame += 1
                if speed6_C.b:
                    bk_install_on(type(s))
                rw_ = rw_set() if speed6_C.rw else None
                R.cur = None
                if S5 is not None:
                    S5.O5.pend = []
                if W.prof is not None:
                    W.rf = torch.profiler.record_function(f's6f{W.fr}')
                    W.rf.__enter__()
            W.dep += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                W.dep -= 1
            if top:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                h1 = 1000.0 * time.perf_counter()
                if W.prof is not None:
                    W.rf.__exit__(None, None, None)
                rec = dict(t=W.fr, o=int(speed6_C.o), a=float(getattr(s, 'last_ms', float('nan'))), h0=round(h0, 4), host=round(h1 - h0, 4), ng=W.ng - ng0, nsync=W.nsync - ns0)
                if W.prev_h1 is not None:
                    rec['gap'] = round(h0 - W.prev_h1, 4)
                W.prev_h1 = h1
                if rw_ is not None:
                    rec['rw'] = rw_
                if speed6_C.b:
                    rec['bkhit'] = BK.hit
                    BK.hit = 0
                tbc = tb_copy()
                if tbc is not None:
                    rec['tb'] = tbc
                if speed6_C.r and R.cur is not None:
                    ras = R.cur
                else:
                    ras = S5.O5.pend[-1][1] if S5 is not None and S5.O5.pend else None
                W.q.append((rec, ev0, ev1, ras))
                W.fr += 1
                _resolve()
                if speed6_C.prof and W.fr == speed6_C.prof[1]:
                    _prof_stop()
            return r
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)
    import atexit
    atexit.register(_flush_file)

class BK:
    mut = 0
    frame = 0
    hit = 0
    done = set()

def bk_install_on(cls):
    if cls in BK.done:
        return
    BK.done.add(cls)
    (ob, ors, orec, oal) = (cls._bk, cls._rs, cls.record, cls.alloc)

    def _bk(s):
        c = getattr(s, '_s6c', None)
        if c is not None and c[1] == BK.mut and (c[2] == BK.frame):
            BK.hit += 1
            return c[0]
        b = ob(s)
        s._s6c = [b, BK.mut, BK.frame]
        return b

    def _rs(s, bk):
        r = ors(s, bk)
        BK.mut += 1
        c = getattr(s, '_s6c', None)
        if c is not None and bk is c[0]:
            c[1] = BK.mut
        return r

    def record(s, *a, **kw):
        BK.mut += 1
        return orec(s, *a, **kw)

    def alloc(s, *a, **kw):
        BK.mut += 1
        return oal(s, *a, **kw)
    (cls._bk, cls._rs, cls.record, cls.alloc) = (_bk, _rs, record, alloc)
    print(f'[절 CM65] 백업 한 번 (B) 설치 {cls.__module__}.{cls.__name__}', flush=True)

class R:
    pend = None
    cur = None
    XS = None
    FS = None
    ar = None
    eu = [None, None]
    k = 0
    n = 0

class _Prox:

    def __init__(s, x, F):
        raise NotImplementedError('not used by the paper settings')

def r_flush():
    if R.pend is None:
        return
    (G, U, t, ras, holder) = R.pend
    R.pend = None
    S5 = sys.modules[__name__]
    D = torch.cuda.default_stream()
    with torch.no_grad(), torch.cuda.stream(D):
        ras(G, U, t)
    holder['ev'] = S5.O5.pend[-1][1] if S5.O5.pend else None
    holder['craw'] = S5.O5.pend[-1] if S5.O5.pend else None

def r_install():
    raise NotImplementedError('not used by the paper settings')

class RW:
    nmed = 5
    tv = None
    ring = None
    k = 0

def _tb_views():
    if RW.tv is not None:
        return RW.tv
    S3 = sys.modules[__name__]
    if S3.TB.mod is None:
        return None
    import cupy as cp
    a = torch.as_tensor(cp.ndarray((1,), dtype=cp.int64, memptr=S3.TB.mod.get_global('TB_T0')), device='cuda')
    b = torch.as_tensor(cp.ndarray((1,), dtype=cp.int64, memptr=S3.TB.mod.get_global('TB_TE')), device='cuda')
    RW.tv = (a, b)
    RW.ring = torch.zeros(256, 2, dtype=torch.int64, pin_memory=True)
    return RW.tv

def tb_copy():
    tv = _tb_views()
    if tv is None:
        return None
    k = RW.k
    RW.k = (RW.k + 1) % 256
    RW.ring[k, 0:1].copy_(tv[0], non_blocking=True)
    RW.ring[k, 1:2].copy_(tv[1], non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return (ev, RW.ring[k])

def rw_set():
    _resolve()
    S3 = sys.modules[__name__]
    if S3.TB.mod is None:
        return None
    vs = []
    for r in reversed(W.recs):
        if r['t'] >= 3 and r.get('Ws') is not None and (r.get('T') is not None):
            vs.append((r['t'], r['Ws'] - r['T']))
            if len(vs) == RW.nmed:
                break
    if not vs:
        return None
    sv = sorted((q[1] for q in vs))
    m_ = len(sv) // 2
    v = sv[m_] if len(sv) % 2 else 0.5 * (sv[m_ - 1] + sv[m_])
    import cupy as cp
    cp.ndarray((1,), dtype=cp.int64, memptr=S3.TB.mod.get_global('TC_RMS'))[0] = int(round(v * 1000000.0))
    return [vs[0][0], round(v, 4), len(vs)]

def _prof_start():
    raise NotImplementedError('not used by the paper settings')

def _prof_stop():
    raise NotImplementedError('not used by the paper settings')

class _Gaps:

    def __init__(s, gaps):
        raise NotImplementedError('not used by the paper settings')

    def F(s, t):
        raise NotImplementedError('not used by the paper settings')

def analyze(fn, top=40):
    raise NotImplementedError('not used by the paper settings')

def speed6_install(out, o=0, lean=0, prof=None, b=0, r=0, rw=0):
    (speed6_C.out, speed6_C.o, speed6_C.lean, speed6_C.on, speed6_C.b, speed6_C.r, speed6_C.rw) = (out, int(o), int(lean), True, int(b), int(r), int(rw))
    if prof:
        speed6_C.prof = tuple((int(q) for q in str(prof).split(',')))
    if speed6_C.r:
        assert speed6_C.o, '--s6r 은 겹치기 O (--s6o 1) 위에서만'
    if speed6_C.o:
        S5 = sys.modules[__name__]
        S5.ov_install()
    if speed6_C.r:
        r_install()
    if speed6_C.rw:
        import cupy as cp
        z_ = cp.zeros(1, dtype=cp.int64)
        z_[0] = 1
        cp.cuda.get_current_stream().synchronize()
    _counters()
    speed6_frame_install(out)
    print(f'[절 CM65] speed6 켬 겹치기 {speed6_C.o} 가벼운 판 {speed6_C.lean} 백업 한 번 {speed6_C.b} 렌더 밖으로 {speed6_C.r} 벽시계 렌더 몫 {speed6_C.rw} 프로파일 {speed6_C.prof} 시뮬 스트림 {torch.cuda.current_stream()}', flush=True)

# ---------------- fem_speed7 ----------------
class speed7_C:
    out = '.'
    kp = None
    nm = 0
    z = 0
    p = 0

class LZ:
    mode = 0
    ob = None
    orr = None
    cur = None
    st = dict(lazy=0, mat=0, rs_noop=0, rs_real=0, mis=0, mis_mat=0, cmp=0)
    log = None

def _cmp_state(s, ref):
    raise NotImplementedError('not used by the paper settings')

class _Lazy:

    def __init__(s_, s, ref=None):
        raise NotImplementedError('not used by the paper settings')

    def mat(s_):
        raise NotImplementedError('not used by the paper settings')

    def __getitem__(s_, i):
        raise NotImplementedError('not used by the paper settings')

def lz_install(out):
    import hexfem as AORT
    cls = AORT.AORTSolver
    (LZ.ob, LZ.orr) = (cls._bk, cls._rs)
    LZ.log = open(os.path.join(out, 's7z.jsonl'), 'w')

    def _bk(s):
        ref = LZ.ob(s) if LZ.mode == 2 else None
        b = _Lazy(s, ref)
        LZ.cur = b
        LZ.st['lazy'] += 1
        return b

    def _rs(s, bk):
        if isinstance(bk, _Lazy):
            if bk.real is None:
                LZ.st['rs_noop'] += 1
                if bk.ref is not None:
                    bad = _cmp_state(s, bk.ref)
                    if bad:
                        LZ.st['mis'] += 1
                        LZ.st.setdefault('mis_names', []).append(bad)
                        return LZ.orr(s, bk.ref)
                return None
            LZ.st['rs_real'] += 1
            return LZ.orr(s, bk.real)
        return LZ.orr(s, bk)
    (cls._bk, cls._rs) = (_bk, _rs)
    ocap = cls.capture

    def capture(s, *a, **kw):
        if LZ.cur is not None and LZ.cur.s is s:
            LZ.cur.mat()
        return ocap(s, *a, **kw)
    cls.capture = capture
    orec = cls.record

    def record(s, *a, **kw):
        if LZ.cur is not None and LZ.cur.s is s and (not torch.cuda.is_current_stream_capturing()):
            LZ.cur.mat()
        return orec(s, *a, **kw)
    cls.record = record
    oK = AORT.set_K_ga

    def set_K_ga(s, K, *a, **kw):
        if int(K) > 1 and isinstance(s, cls) and (LZ.cur is not None) and (LZ.cur.s is s):
            LZ.cur.mat()
        return oK(s, K, *a, **kw)
    AORT.set_K_ga = set_K_ga
    print(f'[절 CM72] 지연 백업 Z 켬 (모드 {LZ.mode}) {cls.__module__}.{cls.__name__}', flush=True)

def lz_frame_end(fr):
    if LZ.log is None:
        return
    st = dict(LZ.st)
    if any((st[k] for k in ('lazy', 'mat', 'rs_noop', 'rs_real', 'mis', 'mis_mat'))) or 'mis_names' in st:
        LZ.log.write(json.dumps(dict(f=fr, **st), ensure_ascii=False) + '\n')
        LZ.log.flush()
    for k in LZ.st:
        LZ.st[k] = 0 if not isinstance(LZ.st[k], list) else LZ.st[k]
    LZ.st.pop('mis_names', None)
    LZ.cur = None
P_SRC = cu('P_SRC')
PK = {'rt_bar_y0e': (6, 4, 5, 7, 8), 'rt_bar_kfe': (5, 3, 4, 6, 7), 'ws_bk': (5, 3, 4, 6, 7), 'yl_kq': (6, 4, 5, 7, 8), 'rt_bar_q': (5, 4, None, 6, 7), 'wb_floor_q': (5, 4, None, 6, 7), 'yl_chk': (4, 3, None, 5, 6), 'en_phi_q': (7, 6, None, 8, 13), 'mf_mark': (4, 4, 3, 5, 6), 'hc_bar': (4, 4, 3, 5, 6)}
PSK2 = {'ws_bk'}

def _db(Q):
    return f'(AF_BX[{Q}*4]/bm)*n0 + (AF_BX[{Q}*4+1]/bm)*n1 + (AF_BX[{Q}*4+2]/bm)*n2 - pn'

def _pre(Q, T, loop):
    ind = loop[:len(loop) - len(loop.lstrip())]
    return f'{ind}bool s7f_ = s7far({Q}, IP, cb, {_db(Q)}, {T});\n{ind}if (!(s7f_ && S7ON == 1))\n' + loop
L2 = 'for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o];\n'
D2 = '    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n'
speed7_PATCH = {'rt_bar_y0e': [('        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n', _pre('qq', 'dh', '        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n')), ('        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n', '        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n        S7_CNT(s7f_, !(fminf(d0, d1) >= dh)); if (s7f_ && S7ON == 1) { d0 = 1e30f; d1 = 1e30f; }\n')], 'rt_bar_kfe': [('    ' + L2, _pre('q', '0.f', '    ' + L2)), (D2, D2 + '    S7_CNT(s7f_, !(d1 > 0.f)); if (s7f_ && S7ON == 1) return;\n')], 'ws_bk': [('    ' + L2, _pre('q', '0.f', '    ' + L2)), (D2, D2 + '    S7_CNT(s7f_, !(d1 > 0.f)); if (s7f_ && S7ON == 1) return;\n')], 'yl_kq': [('    ' + L2, _pre('q', 'dh', '    ' + L2)), (D2, D2 + '    S7_CNT(s7f_, !(fminf(d0, d1) >= dh)); if (s7f_ && S7ON == 1) { d0 = 1e30f; d1 = 1e30f; }\n')], 'rt_bar_q': [('        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n', _pre('qq', 'dh', '        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n')), ('        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n', '        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n        S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) d = 1e30f;\n')], 'wb_floor_q': [('    ' + L2, _pre('q', 'dh', '    ' + L2)), ('    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n    if (!(d < dh)) return;\n', '    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n    S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) return;\n    if (!(d < dh)) return;\n')], 'yl_chk': [('    for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xa[a] +=', _pre('q', '0.f', '    for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xa[a] +=')), ('    if (xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn > 0.f) return;\n', '    { float d_ = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; S7_CNT(s7f_, !(d_ > 0.f)); if (s7f_ && S7ON == 1) return; if (d_ > 0.f) return; }\n')], 'en_phi_q': [('            for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xq[a] +=', _pre('q', 'dh', '            for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xq[a] +=')), ('            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n            if (d < dh) {\n', '            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n            S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) d = 1e30f;\n            if (d < dh) {\n')], 'mf_mark': [('    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];\n', _pre('q', 'dh', '    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];\n')), ('    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;\n', '    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;\n    S7_CNT(s7f_, !(fminf(ds, dp) >= dh)); if (s7f_ && S7ON == 1) return;\n')], 'hc_bar': [('    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];\n', _pre('q', 'dh', '    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];\n')), ('    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;\n', '    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;\n    S7_CNT(s7f_, !(fminf(ds, de) >= dh)); if (s7f_ && S7ON == 1) return;\n')]}
PSEL = ('rt_bar_kfe', 'rt_bar_q', 'wb_floor_q', 'mf_mark', 'hc_bar', 'rt_bar_y0e', 'yl_kq')
YPATCH = [('{ if (AF_SKIP) return;\n', '{ if (AF_SKIP) return; if (S7YON && S7YGT) return;\n'), ('    blocksum_add(e, &SC[0]);\n    float c = ldexpf(1.f, -kk[0]);\n    for (int i=0;i<NE;i++) {\n', '    if (S7YI0 == 0) blocksum_add(e, &SC[0]);\n    float c = ldexpf(1.f, -kk[0] - S7YI0);\n    for (int i=S7YI0;i<min(NE, S7YI1);i++) {\n')]

class PP:
    y = 0
    ym = 4
    sel = PSEL
    mode = 0
    marg = 0.0001
    patched = set()
    log = None
    bad = set()
    said = set()

def _p_src(src):
    raise NotImplementedError('not used by the paper settings')

class _PFn(dict):

    def __init__(s_, src, owner):
        raise NotImplementedError('not used by the paper settings')

    def __getitem__(s_, name):
        raise NotImplementedError('not used by the paper settings')

class _PCall:

    def __init__(s_, s, name, f):
        raise NotImplementedError('not used by the paper settings')

    def pre(s_, args):
        raise NotImplementedError('not used by the paper settings')

    def post(s_, args):
        raise NotImplementedError('not used by the paper settings')

class speed7__YCall:

    def __init__(s_, s, name, f):
        raise NotImplementedError('not used by the paper settings')

class _YPick:

    def __init__(s_, s, f):
        raise NotImplementedError('not used by the paper settings')

def _pfn_get_raw(self, name):
    raise NotImplementedError('not used by the paper settings')
_PFn.get_raw = _pfn_get_raw

def p_bind(s):
    raise NotImplementedError('not used by the paper settings')

def p_install(out, mode, marg=0.0001, y=0, ym=4, sel=None):
    import hexfem as AORT
    (PP.mode, PP.marg, PP.y, PP.ym) = (int(mode), float(marg), int(y), int(ym))
    if sel:
        PP.sel = tuple((q for q in str(sel).split(',') if q))
    PP.log = open(os.path.join(out, 's7p.jsonl'), 'w')
    o_s = AORT._src
    AORT._src = lambda *a, **kw: _p_src(o_s(*a, **kw))
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        r = ob(s, *a, **kw)
        if 's7_bmin(' in getattr(s, '_s7code', 's7_bmin(') and getattr(s, 'mod', None) is not None:
            try:
                p_bind(s)
            except Exception as e_:
                PP.bad.add(('bind', str(e_)[:200], 0))
        return r
    cls.bind_act = bind_act
    print(f'[절 CM72] 장벽 거르기 P 켬 (모드 {PP.mode}, 여유 {PP.marg} m, 커널 {(PP.sel if PP.mode else ())}) · 시작점 후보 나눠 셈 Y {PP.y} (앞 {PP.ym})', flush=True)

def p_frame_end(s, fr):
    if PP.log is None or getattr(s, 's7v', None) is None:
        return
    rec = dict(f=fr)
    if fr <= 3 or PP.mode == 2:
        rec['patched'] = sorted(PP.patched) if fr <= 3 else None
        rec['bad'] = [list(b) for b in sorted(PP.bad)] if fr <= 3 else None
        rec['fnwrap'] = isinstance(s.fn, _PFn) if fr <= 3 else None
    if PP.mode == 2 or PP.y == 2:
        torch.cuda.synchronize()
        v = s.s7v.tolist()
        s.s7v.zero_()
        if PP.mode == 2:
            rec.update(viol=v[0], far=v[1], act=v[2])
        if PP.y == 2:
            rec.update(ychk=v[3], ygate=v[4], yviol=v[5])
    if len(rec) > 1:
        PP.log.write(json.dumps({k: v for (k, v) in rec.items() if v is not None}, ensure_ascii=False) + '\n')
        PP.log.flush()

class KP:
    prof = None
    mod = None
    fn = None
    dep = 0
    nfr = 0
    t0 = None
FM_SRC = '\nextern "C" __global__ void s7_fm(int f) { }\n'

def kp_install():
    raise NotImplementedError('not used by the paper settings')

def nm_install():
    import stability as MF
    MF.mfix__env_init()

    def post(s, fi):
        return None
    MF.post = post
    print('[절 CM72] mfix 프레임 진단 끔 (장치 일은 그대로)', flush=True)

def speed7_frame_install():
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX
    S6 = sys.modules[__name__]
    dep = [0]

    def outer(fn):

        def w(s, *a, **kw):
            dep[0] += 1
            try:
                return fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                if dep[0] == 0:
                    lz_frame_end(S6.W.fr)
                    if speed7_C.p:
                        p_frame_end(s, S6.W.fr)
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)

def src_install(out):
    raise NotImplementedError('not used by the paper settings')

def ovf_install(out):
    raise NotImplementedError('not used by the paper settings')

def speed7_install(out, kp=None, nm=0, z=0, src=0, ovf=0, p=0, pm=0.0001, y=0, ym=4, pk=None):
    (speed7_C.out, speed7_C.nm, speed7_C.z, speed7_C.p) = (out, int(nm), int(z), int(p) or int(y))
    if speed7_C.p:
        p_install(out, int(p), pm, y=int(y), ym=int(ym), sel=pk)
    if int(src):
        src_install(out)
    if int(ovf):
        ovf_install(out)
    if speed7_C.z:
        LZ.mode = speed7_C.z
        lz_install(out)
    if speed7_C.z or speed7_C.p:
        speed7_frame_install()
    if kp:
        speed7_C.kp = tuple((int(q) for q in str(kp).split(',')))
    if speed7_C.nm:
        nm_install()
    if speed7_C.kp:
        kp_install()
    print(f'[절 CM72] speed7 켬 프로파일 {speed7_C.kp} mfix 진단 끔 {speed7_C.nm} 지연 백업 {speed7_C.z}', flush=True)

# ---------------- fem_speed8 ----------------
class speed8_C:
    out = '.'
    w = 0
    g = 0.0
    log = None
    patched = set()
    bad = []
    said = False
    gcnt = 0
    gtot = 0
S8_HDR = cu('S8_HDR')
WPATCH = {'rt_p2g': [('    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]); }\n    atomicAdd(&aq[27], w); atomicAdd(&ac[29], m[p]);', '    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q), gc = __match_any_sync(s8m, c); bool lq = s8_lead(gq), lc = s8_lead(gc);   // 절 CM74 W\n      for (int k=0;k<9;k++) { float a0 = s8_gsum(gq, w*R[k]), a1 = s8_gsum(gq, w*L[k]), a2 = s8_gsum(gq, w*Fp[k]);\n                              if (lq) { atomicAdd(&aq[k], a0); atomicAdd(&aq[9+k], a1); atomicAdd(&aq[18+k], a2); }\n                              s8_sh(0, (size_t)q*28+k, w*R[k]); s8_sh(0, (size_t)q*28+9+k, w*L[k]); s8_sh(0, (size_t)q*28+18+k, w*Fp[k]); }\n      float a3 = s8_gsum(gq, w), a4 = s8_gsum(gc, m[p]); if (lq) atomicAdd(&aq[27], a3); if (lc) atomicAdd(&ac[29], a4);\n      s8_sh(0, (size_t)q*28+27, w); s8_sh(1, (size_t)c*30+29, m[p]); s8_ld(0, lq); }')], 'aq_p2g_s': [('    for (int k=0;k<4;k++) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }', '    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q), gc = __match_any_sync(s8m, c); bool lq = s8_lead(gq), lc = s8_lead(gc);   // 절 CM74 W\n      for (int k=0;k<4;k++) { float a0 = s8_gsum(gq, v4[k]), a1 = s8_gsum(gc, v4[k]); if (lq) atomicAdd(&Sa[(size_t)q*4+k], a0); if (lc) atomicAdd(&Sc[(size_t)c*4+k], a1);\n                              s8_sh(2, (size_t)q*4+k, v4[k]); s8_sh(3, (size_t)c*4+k, v4[k]); }\n      s8_ld(1, lq); }')], 'gq_p2q': [('    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); }\n    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); }', '    unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q); bool lq = s8_lead(gq);   // 절 CM74 W\n    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&PQ[(size_t)q*6+a], a0); s8_sh(4, (size_t)q*6+a, t); }\n    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&PQ[(size_t)q*6+3+a], a0); s8_sh(4, (size_t)q*6+3+a, t); }\n    s8_ld(2, lq);')], 'wb_p2q': [('    for (int a=0;a<3;a++) atomicAdd(&AF_BX[q*4+a], m[p]*x[p*3+a]); atomicAdd(&AF_BX[q*4+3], m[p]);', '    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q); bool lq = s8_lead(gq);   // 절 CM74 W\n      for (int a=0;a<3;a++) { float t = m[p]*x[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&AF_BX[q*4+a], a0); s8_sh(5, (size_t)q*4+a, t); }\n      float a3 = s8_gsum(gq, m[p]); if (lq) atomicAdd(&AF_BX[q*4+3], a3); s8_sh(5, (size_t)q*4+3, m[p]); s8_ld(3, lq); }')]}
SLOT = {0: 'Aacc', 1: 'Cacc', 2: 'Sa', 3: 'Sc', 4: 'gqP', 5: 'bx'}
KSLOT = {'rt_p2g': (0, 1), 'aq_p2g_s': (2, 3), 'gq_p2q': (4,), 'wb_p2q': (5,)}
KNAME = {0: 'rt_p2g', 1: 'aq_p2g_s', 2: 'gq_p2q', 3: 'wb_p2q'}

def _w_patch(src):
    import re
    for (nm, reps) in WPATCH.items():
        m = re.search('__global__\\s+void\\s+(?:__launch_bounds__\\(\\d+\\)\\s+)?' + nm + '\\s*\\(', src)
        if not m:
            continue
        i = m.start()
        j = src.find('\n__global__', m.end())
        j = len(src) if j < 0 else j
        body = src[i:j]
        ok = True
        for (old, new) in reps:
            if body.count(old) != 1:
                ok = False
                speed8_C.bad.append([nm, old[:70], body.count(old)])
                break
            body = body.replace(old, new, 1)
        if ok:
            src = src[:i] + body + src[j:]
            speed8_C.patched.add(nm)
    return src

def _w_src(src):
    if 's8_gsum(' in src or 'void rt_p2g(' not in src:
        return src
    src = _w_patch(src)
    k = src.find('__global__')
    k = src.rfind('\n', 0, k) + 1
    src = src[:k] + S8_HDR + src[k:]
    if not speed8_C.said:
        speed8_C.said = True
        print(f'[절 CM74] 워프 모으기 W 켬 (모드 {speed8_C.w}) 고친 커널 {sorted(speed8_C.patched)} 못 고침 {speed8_C.bad}', flush=True)
    return src

def speed8__P(t):
    raise NotImplementedError('not used by the paper settings')

def w_bind(s):
    raise NotImplementedError('not used by the paper settings')

def w_install(out, mode):
    import hexfem as AORT
    speed8_C.w = int(mode)
    import cupy as cp
    o_rm = cp.RawModule

    def RawModule(*a, **kw):
        if 'code' in kw and isinstance(kw['code'], str):
            kw['code'] = _w_src(kw['code'])
        elif a and isinstance(a[0], str):
            a = (_w_src(a[0]),) + tuple(a[1:])
        return o_rm(*a, **kw)
    cp.RawModule = RawModule
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        r = ob(s, *a, **kw)
        try:
            w_bind(s)
        except Exception as e_:
            speed8_C.bad.append(['bind', str(e_)[:200]])
        return r
    cls.bind_act = bind_act
    if speed8_C.w != 2:
        return
    o_k = cls.k

    def k(s, name, n, args, b=128):
        sl = KSLOT.get(name) if name in speed8_C.patched and getattr(s, 's8sh', None) is not None and getattr(s, 'ao_ready', False) else None
        if not sl:
            return o_k(s, name, n, args, b)
        L = []
        for kk in sl:
            t = getattr(s, SLOT[kk], None)
            if not torch.is_tensor(t) or kk not in s.s8sh:
                continue
            (a, h, c_) = s.s8sh[kk]
            L.append((kk, t, a, h, c_))
            o_k(s, 's8_pre', min(t.numel(), 1 << 22), (speed8__P(t), speed8__P(a), speed8__P(h), speed8__P(c_), np.int64(t.numel())), 256)
        r = o_k(s, name, n, args, b)
        for (kk, t, a, h, c_) in L:
            o_k(s, 's8_cmp', min(t.numel(), 1 << 22), (speed8__P(t), speed8__P(a), speed8__P(h), speed8__P(c_), np.int64(t.numel()), speed8__P(s.s8st), np.int32(kk)), 256)
        return r
    cls.k = k

def w_frame_end(s, fr):
    raise NotImplementedError('not used by the paper settings')

def g_install(T):
    import hexfem as AORT
    import hexfem as C3
    S4 = sys.modules[__name__]
    assert S4.X1.on and S4.X1.level == 2, '--s8g 는 X2 (@s4x 2) 위'
    speed8_C.g = float(T)
    S4.X1.T = speed8_C.g
    cls = AORT.AORTSolver
    o_ctl = cls.ctl

    def ctl(s, op, it=0):
        cap = getattr(s, '_c3cap', None) is not None
        if op == 3 and it == 0:
            if cap and getattr(s, '_x1K', 0) == 1 and (getattr(s, '_xbd', None) is not None):
                s._xbd[2:4].zero_()
            T0 = S4.X1.T
            S4.X1.T = 0.0
            try:
                return o_ctl(s, op, it)
            finally:
                S4.X1.T = T0
        r = o_ctl(s, op, it)
        if op == 3 and it == 1 and cap and (getattr(s, '_x1K', 0) == 1):
            assert C3.mark is not None
            S4._xb('xb_r', 1, (speed8__P(s.IP), speed8__P(s.SC), speed8__P(s.CD), np.float64(speed8_C.g), speed8__P(s._xbd)))
            S4._d2h(s)
            C3.mark(s, 'r')
        return r
    cls.ctl = ctl
    o_rep = C3.SegGraph.replay

    def replay(self):
        s = self.s
        xn = getattr(s, '_xbn', None)
        if xn is not None:
            xn[2] = 0
            xn[3] = 0
        e0 = S4.X1.x3e
        r = o_rep(self)
        if S4.X1.x3e > e0:
            speed8_C.gcnt += S4.X1.x3e - e0
        return r
    C3.SegGraph.replay = replay

def speed8_frame_install():
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX
    S6 = sys.modules[__name__]
    dep = [0]

    def outer(fn):

        def w(s, *a, **kw):
            dep[0] += 1
            try:
                return fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                    w_frame_end(s, S6.W.fr)
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)

def speed8_install(out, w=0, g=0.0):
    speed8_C.out = out
    speed8_C.log = open(os.path.join(out, 's8w.jsonl'), 'w')
    if int(w):
        w_install(out, int(w))
    if float(g) > 0:
        g_install(float(g))
    speed8_frame_install()
    print(f'[절 CM74] speed8 켬 워프 모으기 {speed8_C.w} X3g {speed8_C.g}', flush=True)

# ---------------- fem_speed9 ----------------
class speed9_C:
    out = '.'
    dg = 0
    cm = 0
    yl = 0
    e3 = 0
    r = 0
    ew = 0.0
    log = None
    patched = set()
    bad = []
    said = False
    rsaid = False
    rc = dict(cont=0, fb=0)
HDR = cu('HDR')
TAIL = cu('TAIL')
Y0E_OLD = '    if (!__syncthreads_or(act && fminf(d0, d1) < dh + 1e-5f*(fabsf(d0) + fabsf(d1)))) return;'
Y0E_NEW = '    bool s9b_ = act && fminf(d0, d1) < dh + 1e-5f*(fabsf(d0) + fabsf(d1));   // 절 CM76 (D)·(C)\n    if (S9DG) s9_cnt(act, s9b_);\n    if (S9CM) s9_push(qq, d0, d1, s9b_);\n    if (S9CM == 1) return;\n    if (!__syncthreads_or(s9b_)) return;\n    if (S9DG && threadIdx.x == 0) atomicAdd(&S9C[2], 1ULL);'
YL_OLD = '        float c = ldexpf(1.f, -k); int i = 0;\n        for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }'
YL_NEW = '        float c = ldexpf(1.f, -k); int i = 0;\n        if (S9YL && isfinite(d0) && isfinite(d1)) { i = s9_ylf(d0, d1, dh, c);   // 절 CM76 (Y)\n            if (S9YL == 2) { int j = 0; float c2 = c; for (; j<59; j++) { double E = yl_e(d0 + c2*(d1 - d0), dh, kap); if (E <= E0) break; c2 *= 0.5f; }\n                             atomicAdd(&S9C[5], 1ULL); if (j != i) atomicAdd(&S9C[6], 1ULL); i = j; } }\n        else for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }'
CTL3_OLD = '        double g2 = SC[1];\n'
CTL3_NEW = '        double g2 = SC[1];\n        if (S9DG) s9_lg(100 + it, it == 0 ? g2/SC[56] : g2/CD[2]);   // 절 CM76 (D)\n'
CTL4_OLD = '    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); CD[1] = (eP2 >= 0.0) ? eP2*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }'
CTL4_NEW = '    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); double e2_ = s9_ew(it, eP2, SC[1]); S9RF = rf;   // 절 CM76 (P)\n        if (S9DG && S9EW) s9_lg(300 + it, sqrt(e2_));\n        CD[1] = (eP2 >= 0.0) ? e2_*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }'
CTL6_OLD = 'CT[3]++; if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }'
CTL6_NEW = 'CT[3]++; if (S9DG) s9_lg(200 + it, S9RF > 0.0 ? r2/S9RF : -1.0); if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }'
speed9_PATCH = {'rt_bar_y0e': [(Y0E_OLD, Y0E_NEW)], 'yl_kq': [(YL_OLD, YL_NEW)], 'rt_ctl': [(CTL3_OLD, CTL3_NEW), (CTL4_OLD, CTL4_NEW), (CTL6_OLD, CTL6_NEW)]}
E3_OLD = '    for (int o=0;o<27;o++) atomicAdd(&E3W(q, o), ww[o]*sc);'
E3_NEW = '    { unsigned s8m = s8_act(p, N); int s9q = (int)q; unsigned gq = __match_any_sync(s8m, s9q); bool lq = s8_lead(gq);   // 절 CM76 (E)\n      for (int o=0;o<27;o++) { float t = ww[o]*sc; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&E3W(q, o), a0); s8_sh(6, (size_t)o*AF_WQN + q, t); }\n      s8_ld(4, lq); }'

def _patch(src):
    for (nm, reps) in speed9_PATCH.items():
        m = re.search('__global__\\s+void\\s+(?:__launch_bounds__\\(\\d+\\)\\s+)?' + nm + '\\s*\\(', src)
        if not m:
            continue
        i = m.start()
        j = src.find('\n__global__', m.end())
        j = len(src) if j < 0 else j
        body = src[i:j]
        ok = True
        for (old, new) in reps:
            if body.count(old) != 1:
                ok = False
                speed9_C.bad.append([nm, old[:70], body.count(old)])
                break
            body = body.replace(old, new, 1)
        if ok:
            src = src[:i] + body + src[j:]
            speed9_C.patched.add(nm)
    return src

def _src(src):
    if 's9_y0b(' in src or not re.search('__global__\\s+void\\s+__launch_bounds__\\(128\\)\\s+rt_bar_y0e\\s*\\(', src):
        return src
    src = _patch(src)
    k = src.find('__global__')
    k = src.rfind('\n', 0, k) + 1
    src = src[:k] + HDR + src[k:]
    e = src.rstrip().rfind('\n}')
    src = src[:e] + '\n' + TAIL + src[e:]
    if not speed9_C.said:
        speed9_C.said = True
        print(f'[절 CM76] 소스 고침 {sorted(speed9_C.patched)} 못 고침 {speed9_C.bad}', flush=True)
    return src

def speed9__P(t):
    raise NotImplementedError('not used by the paper settings')

def speed9__g(s, nm, dt):
    raise NotImplementedError('not used by the paper settings')

def bind(s):
    raise NotImplementedError('not used by the paper settings')

class speed9__YCall:

    def __init__(s_, s, f):
        raise NotImplementedError('not used by the paper settings')

class _ECall:

    def __init__(s_, s, f):
        raise NotImplementedError('not used by the paper settings')

def _install_src():
    import cupy as cp
    o_rm = cp.RawModule

    def RawModule(*a, **kw):
        if 'code' in kw and isinstance(kw['code'], str):
            kw['code'] = _src(kw['code'])
        elif a and isinstance(a[0], str):
            a = (_src(a[0]),) + tuple(a[1:])
        return o_rm(*a, **kw)
    cp.RawModule = RawModule

def _install_fn():
    S7 = sys.modules[__name__]
    o_gi = S7._PFn.__getitem__

    def gi(s_, name):
        f = o_gi(s_, name)
        if name == 'rt_bar_y0e' and speed9_C.cm and ('rt_bar_y0e' in speed9_C.patched):
            return speed9__YCall(s_.owner, f)
        if name == 'e3_wq' and speed9_C.e3:
            S8 = sys.modules[__name__]
            if S8.speed8_C.w == 2 and 'e3_wq' in S8.speed8_C.patched:
                return _ECall(s_.owner, f)
        return f
    S7._PFn.__getitem__ = gi

def _install_e3():
    raise NotImplementedError('not used by the paper settings')

class _TProxy:

    def __init__(s_, s, g):
        raise NotImplementedError('not used by the paper settings')

    def replay(s_):
        raise NotImplementedError('not used by the paper settings')

def _install_r():
    raise NotImplementedError('not used by the paper settings')

def frame_end(s, fr):
    raise NotImplementedError('not used by the paper settings')

def speed9_frame_install():
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX
    S6 = sys.modules[__name__]
    dep = [0]

    def outer(fn):

        def w(s, *a, **kw):
            dep[0] += 1
            try:
                return fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                    frame_end(s, S6.W.fr)
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)

def speed9_install(out, dg=0, cm=0, yl=0, e3=0, r=0, ew=0.0):
    import hexfem as AORT
    (speed9_C.out, speed9_C.dg, speed9_C.cm, speed9_C.yl, speed9_C.e3, speed9_C.r, speed9_C.ew) = (out, int(dg), int(cm), int(yl), int(e3), int(r), float(ew))
    speed9_C.log = open(os.path.join(out, 's9.jsonl'), 'w')
    if speed9_C.e3:
        _install_e3()
    _install_src()
    _install_fn()
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        rr = ob(s, *a, **kw)
        try:
            bind(s)
        except Exception as e_:
            speed9_C.bad.append(['bind', str(e_)[:200]])
        return rr
    cls.bind_act = bind_act
    if speed9_C.r:
        _install_r()
    speed9_frame_install()
    print(f'[절 CM76] speed9 켬 진단 {speed9_C.dg} 압축 {speed9_C.cm} 시작점 조기 끝 {speed9_C.yl} W̄ 워프 모으기 {speed9_C.e3} 이어 하기 {speed9_C.r} EW γ {speed9_C.ew}', flush=True)

# ---------------- fem_t2chain ----------------
S3 = sys.modules[__name__]
MODE = 0
NIT = 16
_T0_OLD = '    if (TB_NEWF) { TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }\n    TB_RUN = 0; TB_TP = n; TB_CS = 0;'
_T0_NEW = '    if (TB_NEWF) {\n        if (TC_PEND) { long long d_ = TC_BF - (long long)(TB_TE - TB_T0) - ((TC_MODE & 8) ? TC_RMS : 0ll); TC_LAST = d_; if (TC_MODE & 4) TC_BANK = d_; else TC_BANK += d_; }   // 절 CM56: 앞 (그래프 안) 프레임 몫 이월\n        TC_PEND = 1; TC_NALL = 0; TC_NCK = 0;\n        TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }\n    TB_RUN = 0; TB_TP = n; TB_CS = 0; TC_CK = 0; for (int i_ = 0; i_ < 3*TC_NIT; i_++) TC_IT[i_] = -1.0;   // 진단은 시도마다 (마지막 시도가 남음)'
_GATE_OLD = '    if (tb_now() - TB_T0 + ci + ct > B) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; }'
_GATE_NEW = '    long long lim_ = (long long)B + ((TC_MODE & 5) ? TC_BANK : 0ll) - ((TC_MODE & 8) ? TC_RMS : 0ll);   // 절 CM56 비트 1·4: 이월 잔고만큼 한도를 옮김 (음수면 줄임), 비트 8: 앞 프레임 렌더 시간을 뺌\n    TC_LIM = lim_;\n    if ((long long)(tb_now() - TB_T0 + ci + ct) > lim_) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; if (TC_MODE & 2) { TC_CK = 1; TC_NCK++; } }'
_GLOB_OLD = '__device__ int TB_NEWF, TB_RUN, TB_CUT, TB_FR, TB_GATES, TB_CS;'
_GLOB_NEW = _GLOB_OLD + '\n#define TC_NIT %d\n__device__ long long TC_BANK, TC_BF, TC_LAST, TC_LIM, TC_RMS;\n__device__ int TC_PEND, TC_CK, TC_MODE, TC_NALL, TC_NCK;\n__device__ double TC_IT[3*TC_NIT];' % NIT
_TAIL = cu('_TAIL')

def patch_src(src):
    for (o_, n_) in ((_GLOB_OLD, _GLOB_NEW), (_T0_OLD, _T0_NEW), (_GATE_OLD, _GATE_NEW)):
        assert src.count(o_) == 1, o_[:60]
        src = src.replace(o_, n_, 1)
    i = src.rstrip().rindex('}')
    return src[:i] + _TAIL.lstrip('\n') + src[i:]

class TC:
    rms = None
    fn = {}
    log = None
    out = '.'
    nall_tot = 0

def t2chain__g(name, dt, n=1):
    import cupy as cp
    return cp.ndarray((n,), dtype=dt, memptr=S3.TB.mod.get_global(name))

def _launch(name, grid, block, args):
    import hexfem as RTC
    with RTC._Stream():
        TC.fn[name]((grid,), (block,), args)

def t2chain_install(mode, out):
    global MODE
    MODE = int(mode)
    assert not (MODE & 1 and MODE & 4), '비트 1 (누적) 과 4 (한 프레임) 는 함께 안 씀'
    TC.out = out
    S3.TB_SRC = patch_src(S3.TB_SRC)
    o_inst = S3.tb_install

    def tb_install(B, m, out_, where=1):
        o_inst(B, m, out_, where)
        for nm in ('tc_all', 'tc_rec'):
            TC.fn[nm] = S3.TB.mod.get_function(nm)
        t2chain__g('TC_MODE', np.int32)[0] = MODE
        t2chain__g('TC_BF', np.int64)[0] = int(round(float(B) * 1000000.0))
        t2chain__g('TC_BANK', np.int64)[0] = 0
        t2chain__g('TC_PEND', np.int32)[0] = 0
        t2chain__g('TC_RMS', np.int64)[0] = 0
        TC.log = open(os.path.join(out_, 't2c.jsonl'), 'w')
        print(f'[절 CM56 t2chain] 모드 {MODE} (1 누적 이월 · 4 한 프레임 이월 · 2 자른 걸음 몸 전체 후퇴 오일러 · 8 앞 프레임 렌더 시간 빼기; 0 = 진단만), B {B} ms 여유 {m} ms 자리 {where}', flush=True)
    S3.tb_install = tb_install
    import hexfem as ART
    import hexfem as AORT
    for cls in (ART.an_rt_RTSolver, AORT.AORTSolver):
        _wrap_k(cls)
        _wrap_ctl(cls)
    o_ic = S3.inv_counts

    def inv_counts(s):
        r = o_ic(s)
        try:
            frame_rec(s)
        except Exception:
            import traceback
            with open(os.path.join(TC.out, 't2c_err.txt'), 'a') as f_:
                f_.write(traceback.format_exc()[-2000:] + '\n')
        return r
    S3.inv_counts = inv_counts
    _hook_render()

class _RLog:

    def __init__(s, f):
        s.f = f

    def write(s, line):
        try:
            q = json.loads(line)
            r = float(q['b']) + float(q['c'])
            TC.rms = r
            if S3.TB.mod is not None:
                t2chain__g('TC_RMS', np.int64)[0] = int(round(r * 1000000.0))
        except Exception:
            pass
        return s.f.write(line)

    def flush(s):
        return s.f.flush()

def _hook_render():
    try:
        import gsrender as RR
    except Exception:
        return
    o_f = RR._frame

    def _frame(s, upd, ras):
        r = o_f(s, upd, ras)
        lg = RR.ST.get('log')
        if lg is not None and (not isinstance(lg, _RLog)):
            RR.ST['log'] = _RLog(lg)
        return r
    RR._frame = _frame

def _wrap_k(cls):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name == 'ul_st' and MODE & 2 and (S3.TB.mod is not None) and getattr(s, '_tbcap', False):
            _launch('tc_all', (int(n) + 255) // 256, 256, (args[0], args[1]))
            s._tc_nins = getattr(s, '_tc_nins', 0) + 1
        return ok(s, name, n, args, b)
    cls.k = k

def _wrap_ctl(cls):
    oc = cls.ctl

    def ctl(s, op, it=0):
        oc(s, op, it)
        if op == 3 and S3.TB.mod is not None and getattr(s, 'adapt', False) and getattr(s, '_tbcap', False):
            P = lambda t: np.uint64(t.data_ptr())
            _launch('tc_rec', 1, 1, (P(s.SC), P(s.CD), P(s.CT), np.int32(it)))
    cls.ctl = ctl

def frame_rec(s):
    if TC.log is None or S3.TB.mod is None:
        return
    it = t2chain__g('TC_IT', np.float64, 3 * NIT).get().reshape(NIT, 3)
    rows = [[float(a), float(b), int(c)] for (a, b, c) in it if a >= 0]
    nall = int(t2chain__g('TC_NALL', np.int32)[0])
    rec = dict(bank=round(int(t2chain__g('TC_BANK', np.int64)[0]) * 1e-06, 3), last=round(int(t2chain__g('TC_LAST', np.int64)[0]) * 1e-06, 3), lim=round(int(t2chain__g('TC_LIM', np.int64)[0]) * 1e-06, 3), cut=int(S3.TB.g('TB_CUT', np.int32)[0]), nck=int(t2chain__g('TC_NCK', np.int32)[0]), nall=nall, ins=int(getattr(s, '_tc_nins', 0)), rms=int(t2chain__g('TC_RMS', np.int64)[0]) * 1e-06, it=rows)
    TC.log.write(json.dumps(rec) + '\n')
    TC.log.flush()
    _ = torch
