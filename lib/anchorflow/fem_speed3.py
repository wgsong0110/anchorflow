"""절 CM39 묶음 speed3 (조정자 위임 11-08): 공통 후보 D 60 fps -- 진단·가속·후보 모듈 (두 실행기 훅 없이 감싸개 exe/speed3_wrap.py 가 부름).

(진단 1, 결과 불변, 호스트 읽기만) wolf 입자 이탈 -- --ue_loc 국소 갱신 경계: 덜 풀린 걸음이 있는 프레임마다 (K 1 마지막 시도 = 그 프레임 결과)
  입자마다 B-스플라인 27 노드 가중으로 S 몫 sfrac = Σ w ulm (S = 후퇴 오일러 노드 집합), 둘레 노드 속도 퍼짐 vspr = Σ w |v_n − v_p|,
  둘레 노드 속도 최대 vnmax (w > 1e-3), 입자 속도 |v_p| 를 원래 입자 순서로 ul_f<프레임>.npz 에, 노드 묶음 (S · S 밖 S 이웃 · 나머지) 속도·질량 통계를 ul.jsonl 에.
"""
import json
import os

import numpy as np
import torch


def _bw(fx):
    """이차 B-스플라인 가중 (fx ∈ [0.5, 1.5), 노드 0·1·2) -- rt_coord 의 fx = y − floor(y − 0.5)."""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return torch.stack([w0, w1, w2], -1)                                              # [N, 3(축), 3(노드)]


class ULDiag:
    def __init__(s, out):
        s.out = out
        s.f = open(os.path.join(out, "ul.jsonl"), "w")

    def frame(s, S, fi):
        cjl = getattr(S, "cjl", None)
        K = int(getattr(S, "Kcur", 1))
        fail = [int(q) for q in cjl[:K].cpu()] if cjl is not None else []
        rec = dict(f=fi + 1, K=K, fail=fail)
        if not any(fail) or getattr(S, "ulm", None) is None or K != 1:
            s.f.write(json.dumps(rec) + "\n"); s.f.flush()
            return
        IP = S.IP.cpu()
        nb0, nb1, nb2, nn = int(IP[6]), int(IP[7]), int(IP[8]), int(IP[9])
        ulm = S.ulm[:nn].float()
        mn = S.mn[:nn].float()
        vk = S.vk.reshape(-1, 3)[:nn].float()
        sp = vk.norm(dim=-1)
        N = S.x.shape[0]
        pb = S.Np.reshape(-1, 3)[:N].long()
        fx = S.fx.reshape(-1, 3)[:N]
        w = _bw(fx.float())
        vp = S.v[:N].float()
        sfrac = torch.zeros(N, device=vp.device); vspr = torch.zeros(N, device=vp.device); vnmax = torch.zeros(N, device=vp.device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    n = ((pb[:, 0] + i) * nb1 + (pb[:, 1] + j)) * nb2 + (pb[:, 2] + k)
                    ok = (n >= 0) & (n < nn)
                    n = n.clamp(0, nn - 1)
                    ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k] * ok
                    sfrac += ww * ulm[n]
                    vspr += ww * (vk[n] - vp).norm(dim=-1)
                    vnmax = torch.where(ww > 1e-3, torch.maximum(vnmax, sp[n]), vnmax)
        # 노드 묶음: S, S 밖이면서 27 이웃에 S 가 있는 노드 (경계 띠), 나머지 -- 질량 > 0 노드만
        m3 = ulm.view(nb0, nb1, nb2)
        dil = torch.nn.functional.max_pool3d(m3[None, None], 3, stride=1, padding=1)[0, 0].reshape(-1)
        has = mn > 0
        grp = dict(S=has & (ulm > 0), B=has & (ulm == 0) & (dil > 0), O=has & (dil == 0))
        med = float(mn[has].median()) if bool(has.any()) else 1.0
        for g, R in grp.items():
            if not bool(R.any()):
                rec[g] = dict(n=0)
                continue
            m_, s_ = mn[R], sp[R]
            i_ = int(torch.argmax(s_))
            rec[g] = dict(n=int(R.sum()), m=float(m_.sum()), vmean=float((m_ * s_).sum() / m_.sum()), vmax=float(s_.max()), m_at_vmax=float(m_[i_]) / med,
                          n_v3=int((s_ > 3.0).sum()), n_v10=int((s_ > 10.0).sum()))
        perm = S.perm[:N].long()
        o = lambda t: (lambda a: (a.__setitem__(perm, t), a)[1])(torch.empty_like(t))
        np.savez_compressed(os.path.join(s.out, f"ul_f{fi + 1:03d}.npz"), sfrac=o(sfrac).cpu().numpy().astype(np.float16), vspr=o(vspr).cpu().numpy().astype(np.float16),
                            vnmax=o(vnmax).cpu().numpy().astype(np.float16), vp=o(vp.norm(dim=-1)).cpu().numpy().astype(np.float16))
        rec["p"] = dict(mixed=int(((sfrac > 1e-3) & (sfrac < 1 - 1e-3)).sum()), inS=int((sfrac >= 1 - 1e-3).sum()), vp_max=float(vp.norm(dim=-1).max()), vspr_max=float(vspr.max()))
        s.f.write(json.dumps(rec) + "\n"); s.f.flush()


# ======================= (후보 T) 시간 예산 뉴턴 상한 (결과 바뀜, 정의 변경) =======================
# 정의: 프레임 GPU 시계 (globaltimer, 프레임 첫 시도의 첫 커널부터) 로, 뉴턴 반복 it ≥ xf (--xit_from; 덜 풀린 걸음의 '추가' 반복과 그 판정 잔차) 를 시작하기 전에
#   경과 + C_it + C_tail ≤ B − m  이 아니면 그 걸음 나머지 반복을 건너뜀 (CT[0] = 1·AF_SKIP = 1). 자른 걸음의 덜 풀림 판정은 '뉴턴 상한에 닿은 걸음' 과 같게
#   (뉴턴 고리 끝에서 CT[0] 을 0 으로 되돌려 cj_end·uf_end 가 마지막으로 돈 반복 시작 잔차·보폭으로 판정 -- 허용치 멈춤으로 보지 않음).
#   C_it = 최근 8 프레임에서 실제로 돈 뉴턴 반복 (반복 시작 -> 다음 반복 시작 또는 뉴턴 고리 끝) 시간 최대, C_tail = 최근 8 프레임 뉴턴 고리 끝 -> 프레임 끝 시간 최대.
#   반복 0 .. xf−1 은 늘 돎 (D 와 같음). 걸음 수 K > 1 시도 (쪼갠 걸음) 에는 안 검 (D 와 같음). 쪼개기·X3c 다시 하기도 같은 프레임 시계 (첫 시도 시작부터).
# 결과가 GPU 속도에 따라 달라짐 (결정적 판도 시계로 자르므로 비트 대조 대상 아님).
TB_SRC = r"""
extern "C" {
__device__ unsigned long long TB_T0, TB_TP, TB_TL, TB_TE;
__device__ unsigned long long TB_MIT[8], TB_MTL[8];
__device__ int TB_NEWF, TB_RUN, TB_CUT, TB_FR, TB_GATES, TB_CS;
__device__ __forceinline__ unsigned long long tb_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void tb_t0() {
    unsigned long long n = tb_now();
    if (TB_NEWF) { TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }
    TB_RUN = 0; TB_TP = n; TB_CS = 0;
}
__global__ void tb_tick(const int* CT, int op) {
    unsigned long long n = tb_now();
    if (TB_RUN) { unsigned long long d = n - TB_TP; if (d > TB_MIT[TB_FR]) TB_MIT[TB_FR] = d; }
    TB_TP = n;
    if (op == 8) { TB_TL = n; TB_RUN = 0; } else TB_RUN = (CT[0] == 0);
}
__global__ void tb_gate(int* CT, int* SKIP, unsigned long long B) {
    if (CT[0]) return;
    unsigned long long ci = 0ull, ct = 0ull;
    for (int i = 0; i < 8; i++) { if (TB_MIT[i] > ci) ci = TB_MIT[i]; if (TB_MTL[i] > ct) ct = TB_MTL[i]; }
    TB_GATES++;
    if (tb_now() - TB_T0 + ci + ct > B) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; }
}
// 예산으로 자른 걸음은 '허용치로 멈춤' 이 아니라 '반복 상한에 닿음' 과 같게 판정되도록 뉴턴 고리 끝 (ctl 8, 판정 cj_end·uf_end 앞) 에서 CT[0] 을 0 으로 되돌림
__global__ void tb_unmark(int* CT) { if (TB_CS) { CT[0] = 0; TB_CS = 0; } }
__global__ void tb_end() { unsigned long long n = tb_now(); unsigned long long d = n - TB_TL; if (TB_TL > TB_T0 && d > TB_MTL[TB_FR]) TB_MTL[TB_FR] = d; TB_TE = n; }
}
"""


class TB:
    """시간 예산 (후보 T). install(B ms, m ms) 뒤 풀이기 클래스의 record·ctl 과 쪼개기 함수를 감쌈 (감싸개 exe/speed3_wrap.py)."""
    mod = None
    fn = {}
    B = 16.7
    m = 0.5
    log = None

    @classmethod
    def k(cls, name, args):
        import cupy as cp  # noqa: F401
        from . import fem_rt_cuda as RTC
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
    c2 = getattr(s, "c2", None) or {}
    if c2.get("xf"):
        return int(c2["xf"])
    en = (getattr(s, "wf", None) or {}).get("en") or {}
    return int(en.get("xf") or 0)


def _skip_ptr(s):
    if not hasattr(s, "_tb_skip"):
        s._tb_skip = np.uint64(int(s.mod.get_global("AF_SKIP")))
    return s._tb_skip


def tb_install(B, m, out):
    import cupy as cp
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_wmrx as WRX
    TB.mod = cp.RawModule(code=TB_SRC)
    for nm in ("tb_t0", "tb_tick", "tb_gate", "tb_end", "tb_unmark"):                               # 캡처 밖에서 미리 적재
        TB.fn[nm] = TB.mod.get_function(nm)
    TB.B, TB.m = float(B), float(m)
    Bns = np.uint64(int(round((TB.B - TB.m) * 1e6)))
    P = lambda t: np.uint64(t.data_ptr())

    def wrap_record(cls):
        o = cls.record

        def record(s, t_frame, K=None):
            s._tbK = int(K if K is not None else getattr(s, "Kcur", getattr(s, "K", 1)))
            s._tbcap = bool(torch.cuda.is_current_stream_capturing())                  # 그래프 캡처 때만 넣음 (데움·첫 프레임·프로파일 같은 그래프 밖 실행은 시계·기록 안 건드림)
            if s._tbcap:
                TB.k("tb_t0", ())
            r = o(s, t_frame, K)
            if s._tbcap:
                TB.k("tb_end", ())
            return r
        cls.record = record

    def gate(s):
        if getattr(s, "_tbK", 1) == 1 and getattr(s, "_tbcap", False):
            TB.k("tb_gate", (P(s.CT), _skip_ptr(s), Bns))

    def wrap_ctl(cls, ready):
        o = cls.ctl

        def ctl(s, op, it=0):
            on = getattr(s, "adapt", False) and ready(s) and getattr(s, "_tbcap", False)
            xf = _xf(s)
            if on and op == 2 and xf > 0 and it >= xf:
                gate(s)
            if on and op == 8:
                TB.k("tb_unmark", (P(s.CT),))
            o(s, op, it)
            if on and op in (2, 8):
                TB.k("tb_tick", (P(s.CT), np.int32(op)))
        cls.ctl = ctl
    wrap_record(ART.RTSolver); wrap_record(AORT.AORTSolver)
    wrap_ctl(ART.RTSolver, lambda s: True); wrap_ctl(AORT.AORTSolver, lambda s: getattr(s, "ao_ready", False))
    o_mark = C3.mark

    def mark(s, kind):
        if kind == "q":                                                                # 나눈 그래프: 반복 xf 구간 앞에서 예산을 먼저 봄 (호스트가 CT[0] 를 읽어 구간째 건너뜀)
            gate(s)
        o_mark(s, kind)
    C3.mark = mark


# ======================= (후보 W1) 덜 풀린 걸음의 가속도 기억 0 을 몸 전체로 (결과 바뀜, 정의 변경) =======================
# 진단 (s1 uW*): wolf 이탈은 끝 덜 풀린 걸음 f47~50 에서 질량 작은 노드 속도가 26~80 m/s 로 튀며 시작 -- S (후퇴 오일러) 와 S 이웃 (HHT) 에서 먼저,
#   다음 걸음들에 S 밖 (HHT) 노드로 번짐 (f47 S 밖 >3 m/s 51 개 -> f49 2202 개). --ue_loc 없는 판 (몸 전체 후퇴 오일러: 모든 노드 가속도 기억 0) 은 이탈 없음.
# 정의: --ue_loc 국소 갱신 커널 (rt_ga_end_ufl·ao_stepend_ufl) 에서 덜 풀린 걸음이면 S 밖 노드도 가속도 기억 a = 0 (속도·변위 갱신은 그대로: S 후퇴 오일러, S 밖 HHT).
W1_OLD = ("    uf_ = uf_ && ulm[n];", "        if (uf_) a1 = 0.f;")
W1_NEW = ("    int ufa_ = uf_; uf_ = uf_ && ulm[n];                                       // 절 CM39 W1: 덜 풀린 걸음 (S 밖 포함)", "        if (ufa_) a1 = 0.f;                                                    // 절 CM39 W1: 가속도 기억 0 은 몸 전체")


def w1ify(src):
    out = src
    hit = 0
    for nm in ("rt_ga_end_ufl", "ao_stepend_ufl"):
        key = "__global__ void " + nm + "("
        if key not in out:
            continue
        i = out.index(key)
        j = out.index("\n}\n", i) + 3
        k = out[i:j]
        for o_, n_ in zip(W1_OLD, W1_NEW):
            assert k.count(o_) == 1, (nm, o_)
            k = k.replace(o_, n_, 1)
        out = out[:i] + k + out[j:]
        hit += 1
    return out


def w1_install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    o_b = ART.build_src
    ART.build_src = lambda *a, **kw: w1ify(o_b(*a, **kw))
    o_s = AORT._src
    AORT._src = lambda *a, **kw: w1ify(o_s(*a, **kw))


# ======================= (후보 X3e) X3c 의 '무너진 프레임' 판정을 반복 0 시작 잔차로 앞당기고 걸음 시작부터만 다시 (옛 실행기 = 수박; 결과 바뀜, 정의 변경) =======================
# X3c (절 CM38): 끈 채 K 1 프레임을 끝까지 푼 뒤 마지막 시도에 ‖g‖²/F² > 1e6 인 덜 풀린 걸음이 있으면 입자 상태를 되돌리고 켠 채 프레임을 처음부터 다시 (P2G 부터, 시간 2 배).
# X3e: K 1 걸음의 반복 0 시작 잔차 (rt_grad 뒤, 판정 2 와 같은 식 ‖g‖²/F², F² = 이 반복 ‖T f‖²) 가 --ue_bigT (1e6) 넘거나 비유한이면 바로 켜고 (AF_WRXON = 1)
#   그 걸음 시작 커널 (ao_step0_ga ~ 반복 0 잔차) 만 다시 (P2G·MG 만들기 등 걸음 앞은 켬·끔과 무관하고 걸음 안 커널이 덮어쓰지 않음). 놓친 무너짐은 X3c 경로 그대로 (프레임 다시).
#   나눈 그래프 (--xg) 의 경계 's' (걸음 시작 앞)·'r' (반복 0 잔차 뒤) 에서 호스트가 X3E_REDO 를 읽어 's'~'r' 구간을 한 번 더 재생.
X3E_SRC = r"""
extern "C" {
__device__ int X3E_REDO, X3E_N;
__global__ void x3e_chk(const double* SC, const double* CD, int* WON, double T) {
    if (*WON) return;
    double g2 = SC[1], F2 = CD[2];
    if (!(g2 <= T*F2)) { *WON = 1; X3E_REDO = 1; X3E_N++; }
}
}
"""


class X3E:
    mod = None
    T = 1e6


class SegGraph3:
    """fem_common3.SegGraph + 's'~'r' 다시 재생 (X3e)."""

    def __init__(s, sv, segs, kinds):
        s.s, s.segs, s.kinds = sv, segs, kinds

    def replay(self):
        s = self.s
        mode = int(s.xg)
        st = s.c3st
        skip, i, sidx, redone = False, 0, None, False
        red = None
        while i < len(self.segs):
            cond, g = self.segs[i]
            kd = self.kinds[i]
            if kd == "s":
                sidx = i
            if kd == "r" and not redone and sidx is not None:
                if red is None:
                    import cupy as cp
                    red = cp.ndarray((1,), dtype=cp.int32, memptr=X3E.mod.get_global("X3E_REDO"))
                if int(red[0]) != 0:
                    red[0] = 0
                    redone = True
                    i = sidx
                    continue
            if not cond:
                skip = False
                g.replay()
                i += 1
                continue
            if not skip:
                skip = bool(int(s.CT[0]) != 0)
            st["sk" if skip else "run"] += 1
            if not (mode == 1 and skip):
                g.replay()
            i += 1


def x3e_install(T):
    import types
    import cupy as cp
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_rt_cuda as RTC
    X3E.mod = cp.RawModule(code=X3E_SRC)
    X3E.fn = X3E.mod.get_function("x3e_chk")
    X3E.T = float(T)
    o_mark = C3.mark                                                                   # (시간 예산이 먼저 감쌌으면 그 감싼 것)

    def mark(s, kind):
        c = getattr(s, "_c3cap", None)
        if c is not None and "kinds" in c:
            c["kinds"].append(kind)
        o_mark(s, kind)
    C3.mark = mark
    o_k = AORT.AORTSolver.k

    def k(s, name, n, args, b=128):
        if name == "ao_step0_ga" and getattr(s, "_c3cap", None) is not None and getattr(s, "_x3eK", 0) == 1 and getattr(s, "wrx", None):
            mark(s, "s")
        return o_k(s, name, n, args, b)
    AORT.AORTSolver.k = k
    o_ctl = AORT.AORTSolver.ctl

    def ctl(s, op, it=0):
        o_ctl(s, op, it)
        if op == 3 and it == 0 and getattr(s, "_c3cap", None) is not None and getattr(s, "_x3eK", 0) == 1 and getattr(s, "wrx", None) and getattr(s, "adapt", False) and getattr(s, "ao_ready", False):
            with RTC._Stream():
                X3E.fn((1,), (1,), (np.uint64(s.SC.data_ptr()), np.uint64(s.CD.data_ptr()), np.uint64(int(s.mod.get_global("AF_WRXON"))), np.float64(X3E.T)))
            mark(s, "r")
    AORT.AORTSolver.ctl = ctl
    o_rec = AORT.AORTSolver.record

    def record(s, t_frame, K=None):
        s._x3eK = int(K if K is not None else getattr(s, "Kcur", getattr(s, "K", 1)))
        return o_rec(s, t_frame, K)
    AORT.AORTSolver.record = record

    def capture(s):
        """fem_common3.capture 와 같은 데움 + 나눈 캡처, 구간 종류 기록 (X3e 's'·'r')."""
        assert not getattr(s, "pk", False)
        s.mod.get_function("rt_reset")
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
            c = s._c3cap = dict(cur=g0, cond=False, segs=[], pool=None, kinds=["start"])
            try:
                s.record(0.0, K)
            finally:
                s._c3cap = None
            c["cur"].capture_end()
            c["segs"].append((c["cond"], c["cur"]))
        torch.cuda.synchronize()
        g = SegGraph3(s, c["segs"], c["kinds"])
        s.graph = g
        s.graphs[K] = g
    C3.capture = capture                                                               # C3.install 이 types.MethodType(capture, s) 로 붙임 (전역 이름을 그때 찾음)
    _ = types


def x3e_frame_n():
    if X3E.mod is None:
        return None
    import cupy as cp
    a = cp.ndarray((1,), dtype=cp.int32, memptr=X3E.mod.get_global("X3E_N"))
    n = int(a[0]); a[0] = 0
    return n


def frame_hook_install(out):
    """쪼개기 함수 (프레임 진입) 를 감싸 프레임마다 시간 예산 시계 시작 표시 (TB_NEWF) 와 기록 (s3.jsonl: 예산 자른 반복·C_it·C_tail·GPU 시계 프레임 시간, X3e 켠 수)."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    log = open(os.path.join(out, "s3.jsonl"), "w")
    dep = [0]

    def outer(fn):
        def w(s, *a, **kw):
            if dep[0] == 0 and TB.mod is not None:
                TB.g("TB_NEWF", np.int32)[0] = 1
            dep[0] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                dep[0] -= 1
            if dep[0] == 0:
                rec = {}
                if TB.mod is not None:
                    fr = int(TB.g("TB_FR", np.int32)[0])
                    mit, mtl = TB.g("TB_MIT", np.uint64, 8).get(), TB.g("TB_MTL", np.uint64, 8).get()
                    t0, te = int(TB.g("TB_T0", np.uint64)[0]), int(TB.g("TB_TE", np.uint64)[0])
                    rec.update(cut=int(TB.g("TB_CUT", np.int32)[0]), gates=int(TB.g("TB_GATES", np.int32)[0]), cit=round(float(mit.max()) * 1e-6, 3),
                               ctail=round(float(mtl.max()) * 1e-6, 3), it_f=round(float(mit[fr]) * 1e-6, 3), ela=round((te - t0) * 1e-6, 3))
                n = x3e_frame_n()
                if n is not None:
                    rec["x3e"] = n
                ic = inv_counts(s)
                if ic is not None:
                    rec["inv"] = ic
                log.write(json.dumps(rec) + "\n"); log.flush()
            return r
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


# ======================= (가속 I, 결과 불변) 수박 국소 장벽 보폭 ws_inv_q 에 절 CM36 --sp2 비트 16 과 같은 f32 증명 빠른 길 =======================
# ws_inv_q = aw_inv_q 글자 + 점별 kq 출력 (fem_wmsparse._inv_q). 증명 (det(X + tD) 하한 > 1e-3 m³) 이면 f64 경로도 k = 0 이라 kq[q] = 0 (부르기 전 0 으로 채움)·
# atomicMax 없음 -- 바로 돌아가도 출력 같음. mode 2 = 확인: 빠른 길 없이 f64 경로를 다 돌고, 증명 참인데 k > 0 인 점 수를 S3_INVBAD 에 (0 이어야), 증명 참 점 수 S3_INVOK.
def inv_install(mode):
    from . import fem_ao_rt as AORT
    from . import fem_speed2 as SP2
    o_s = AORT._src

    def _src(*a, **kw):
        r = o_s(*a, **kw)
        if "ws_inv_q(" not in r:
            return r
        i, j = SP2._kernel_span(r, "ws_inv_q")
        r = r[:i] + SP2.CERT_SRC.lstrip("\n").replace("sp_cert(", "s3_cert(") + r[i:]               # 같은 식, 이름만 (--sp2 16 이 aw_inv_q 앞에 넣는 sp_cert 와 겹치지 않게)
        inew = SP2.INV_NEW.replace("sp_cert(", "s3_cert(")
        if int(mode) == 1:
            return SP2._sub_in(r, "ws_inv_q", SP2.INV_OLD, inew)
        new = inew.replace("        if (ok_) return;\n    }\n", "        cert_ = ok_;\n    }\n").replace("    if (s3_cert(X0f, Dxf)) {", "    bool cert_ = false;\n    if (s3_cert(X0f, Dxf)) {", 1)
        r = SP2._sub_in(r, "ws_inv_q", SP2.INV_OLD, new)
        r = SP2._sub_in(r, "ws_inv_q", "    kq[q] = k;\n", "    kq[q] = k;\n    if (cert_) { atomicAdd(&S3_INVOK, 1); if (k > 0) atomicAdd(&S3_INVBAD, 1); }\n")
        return r.replace('extern "C" {', 'extern "C" {\n__device__ unsigned long long S3_INVOK, S3_INVBAD;', 1)
    AORT._src = _src
    INV.mode = int(mode)


class INV:
    mode = 0


def inv_counts(s):
    if INV.mode != 2:
        return None
    import cupy as cp
    a = cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global("S3_INVOK")); b = cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global("S3_INVBAD"))
    o = [int(a[0]), int(b[0])]
    a[0] = 0; b[0] = 0
    return o
