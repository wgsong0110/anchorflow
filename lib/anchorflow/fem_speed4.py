"""절 CM50 묶음 speed4 (조정자 위임 11-08, 문제 4 수박 60 fps): 공통 후보 F 수박 -- 단계별 시간 기록 (결과 불변) + 결과 불변 가속.
감싸개 exe/speed4_wrap.py 가 부름 (인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로). 옛 실행기 (fem_ao_rt.AORTSolver, 수박) 만 건드림 -- 새 실행기 (레고·wolf·ficus·bread) 는 안 건드림.

(P) 단계 시간 기록 (--s4p 1, 결과 불변): 그래프 캡처 때만, 단계가 바뀌는 커널 앞에 한 스레드 커널 sp_st (globaltimer 를 따로 둔 버퍼 SP_T·SP_C 에 씀) 를 넣음.
    재생 뒤 (프레임마다) 이웃 표식 사이 시간을 앞 표식의 단계에 더함 -- 실제 그래프 재생 (나눈 그래프·예산·다시 하기 포함) 의 단계별 GPU 시간.
    단계: P2G (프레임 시작 ~ 걸음 시작 앞: 칸·상자·P2G·가우스 점·노드 속도), 걸음 시작 (관성 예측·띠 고착·시작점·장벽 시작 판정·y0loc·시작 뒤집힘 필터),
    W̄ (e3_*), R2·A2 (e5_*), 뉴턴 반복 안 [요소 (aq_elem) · 장벽 기울기 (rt_bar_q·rt_bar_n) · 기울기·판정 (rt_grad·uf_g·en_gate) · PCG (헤시안 곱·장벽 곱·CG 벡터)
    · 장벽·뒤집힘 보폭 (ws_*·aw_*·rt_ytry·rt_bar_kfe·rt_lsup) · 선탐색 에너지 (en_*)], 걸음 끝 (uf_end·ul_*·ao_stepend), RM (aq_commit), G2P (ao_g2p_a·aq_g2p·fpi_*),
    구간 사이 (나눈 그래프 경계 -> 다음 구간 첫 커널: 호스트 읽기·그래프 띄우기), 시도 사이 (쪼개기·X3c 다시 하기 시도 경계, GPU 시간 아님).
    0 채우기·제어 커널 (rt_zero_*·rt_ctl·rt_sc_clear) 은 앞 단계에 붙음. 다른 모듈이 s.k 를 거치지 않고 띄우는 작은 커널 (시간 예산 tb_*) 은 앞 단계에 붙음.
    표식 커널 하나 ~1-3 µs 라 기록 판의 프레임 시간은 조금 늘어남 (기록 없는 단독 판과 따로 잼).

(S1) 장벽·뒤집힘 국소 보폭 (절 CM30 (b) fem_wmsparse.newton) 의 확인 라운드 조기 끝 (--s4w 1, 결과 불변):
    라운드 r (sc 1) 이 어긴 점 0 (fl[0] + fl[1] = 0) 이면 kn2 에 아무것도 안 흩고 ws_merge 도 kn 을 안 바꿈 -> 다음 라운드·최종 확인 (sc 0) 은 같은 kn·같은 입력이라
    후보 y1·방향 dd·kq·kk·어긴 수 모두 같고 (원자 연산은 정수 최대·합이라 순서 무관) fl 은 0 그대로 -> 그 뒤 ws_cand·ws_inv_q·ws_nk·ws_bk·ws_merge 를 건너뛰어도
    kn·wsy1·wsdd·wskq·wskk·wsfl (= ws_fb·ws_lsup 입력) 이 비트 같음. 장치 표시 WS4_DN (ws_init 이 0 으로, 라운드 끝 ws4_chk 가 어긴 점 0 이면 1), 다섯 커널 첫머리 'WS4_DN 이면 돌아감'.
    --s4w 2 = 확인 판: 건너뛰지 않고, WS4_DN 이 선 뒤 라운드에서 어긴 점이 나오면 WS4_BAD 를 셈 (0 이어야 위 논리가 맞음).
"""
import json
import os
import re

import numpy as np
import torch

# ======================= (P) 단계 시간 기록 =======================
SP_SRC = r"""
extern "C" {
#define SPN 65536
__device__ unsigned long long SP_T[SPN];
__device__ int SP_C[SPN];
__device__ int SP_N;
__global__ void sp_st(int c) {
    int i = SP_N;
    if (i < SPN) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); SP_T[i] = t; SP_C[i] = c; }
    SP_N = i + 1;
}
}
"""
SPN = 65536
CATS = ["P2G", "걸음 시작", "W̄", "R2·A2", "요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지", "걸음 끝", "RM", "G2P", "구간 사이", "시도 사이", "기타"]
CI = {c: i for i, c in enumerate(CATS)}
NEWTON_C = {CI[c] for c in ("요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지")}
BEGIN = 1 << 14
INHERIT = {"rt_zero_i", "rt_zero_f", "rt_zero_d", "rt_ctl", "rt_sc_clear", "rt_reset"}
N_ELEM = {"aq_elem", "ao_elem", "ao_elem_s"}
N_BAR = {"rt_bar_q", "rt_bar_n", "wb_cnt"}
N_GRAD = {"rt_grad", "rt_grad_c", "uf_g", "en_gate", "c2_gate"}
N_PCG = {"rt_bar_mv", "rt_ap", "rt_ap_c", "rt_cg1", "rt_cg1_c", "rt_cg2", "rt_cg2_c", "rt_sc_roll", "rt_cg1m", "rt_dotz", "rt_pinit", "ao_warm"}
N_STEP = {"rt_ytry", "rt_bar_kfe", "wb_kmin", "rt_lsup", "rt_lsup_c", "uf_min", "ao_tvec", "aw_tdir", "ws4_chk"}
T_RM = {"aq_commit", "ao_commit"}
T_G2P = {"ao_g2p_a", "aq_g2p", "ao_g2p", "fpi_q", "fpi_p", "lji_p", "pf_g2p"}


class SP:
    mod = None
    fn = None
    on = False
    unk = set()


def _cat(s, name):
    """단계 번호 (None = 앞 단계에 붙음)."""
    if name in INHERIT:
        return None
    if name.startswith("e3_"):
        return CI["W̄"]
    if name.startswith("e5_"):
        return CI["R2·A2"]
    ph = getattr(s, "_spph", "pre")
    if ph == "pre":
        return CI["P2G"]
    if ph == "start":
        return CI["걸음 시작"]
    if ph == "newton":
        if name in N_ELEM:
            return CI["요소"]
        if name in N_BAR:
            return CI["장벽 기울기"]
        if name in N_GRAD:
            return CI["기울기·판정"]
        if name in N_PCG or name.startswith("ao_mv") or name.startswith("rt_mg_"):
            return CI["PCG"]
        if name in N_STEP or name.startswith("ws_") or name.startswith("aw_") or name.startswith("s3_"):
            return CI["장벽·뒤집힘 보폭"]
        if name.startswith("en_"):
            return CI["선탐색 에너지"]
        SP.unk.add(("newton", name))
        return CI["기타"]
    if name in T_RM:
        return CI["RM"]
    if name in T_G2P:
        return CI["G2P"]
    return CI["걸음 끝"]


def _stamp(code):
    from . import fem_rt_cuda as RTC
    with RTC._Stream():
        SP.fn((1,), (1,), (np.int32(code),))


def _cap():
    return bool(torch.cuda.is_current_stream_capturing())


def _emit(s, cid, begin=False):
    it = int(getattr(s, "_spit", 0)) if cid in NEWTON_C else 0
    ks = max(0, int(getattr(s, "_spks", 0)))
    code = cid + 32 * min(it, 15) + 512 * min(ks, 7) + (BEGIN if begin else 0)
    if code != getattr(s, "_splast", None) or begin:
        _stamp(code)
        s._splast = code


def sp_g(name, dt, n=1):
    import cupy as cp
    return cp.ndarray((n,), dtype=dt, memptr=SP.mod.get_global(name))


def sp_frame():
    """이 프레임 (모든 시도) 표식 -> 시도별·단계별 GPU ms."""
    torch.cuda.synchronize()
    n = int(sp_g("SP_N", np.int32)[0])
    m = min(n, SPN)
    if m == 0:
        return dict(n=0)
    T = sp_g("SP_T", np.uint64, SPN)[:m].get().astype(np.int64)
    C = sp_g("SP_C", np.int32, SPN)[:m].get()
    tries, cur = [], None
    for i in range(m):
        c = int(C[i])
        if c & BEGIN:
            cur = dict(cat={}, it={}, t0=int(T[i]), t1=None)
            tries.append(cur)
        if cur is None:
            continue
        cid = c & 31
        if cid == CI["시도 사이"]:
            cur["t1"] = int(T[i])
        if i + 1 < m:
            d = (int(T[i + 1]) - int(T[i])) * 1e-6
            nm = CATS[cid]
            cur["cat"][nm] = cur["cat"].get(nm, 0.0) + d
            if cid in NEWTON_C:
                it = (c >> 5) & 15
                cur["it"][it] = cur["it"].get(it, 0.0) + d
    out = []
    for q in tries:
        gpu = ((q["t1"] - q["t0"]) * 1e-6) if q["t1"] else None
        out.append(dict(ms=round(gpu, 4) if gpu is not None else None, cat={k: round(v, 4) for k, v in q["cat"].items() if k != "시도 사이"},
                        gap=round(q["cat"].get("시도 사이", 0.0), 4), it={str(k): round(v, 4) for k, v in sorted(q["it"].items())}))
    sp_g("SP_N", np.int32)[0] = 0
    torch.cuda.synchronize()
    return dict(n=n, tries=out)


def sp_install():
    import cupy as cp
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    SP.mod = cp.RawModule(code=SP_SRC)
    SP.fn = SP.mod.get_function("sp_st")
    SP.on = True
    cls = AORT.AORTSolver
    o_rec, o_k, o_ctl = cls.record, cls.k, cls.ctl

    def record(s, t_frame, K=None):
        s._spph, s._spit, s._spks, s._splast = "pre", 0, -1, None
        cap = _cap()
        if cap:
            _emit(s, CI["P2G"], begin=True)
        r = o_rec(s, t_frame, K)
        if cap:
            _stamp(CI["시도 사이"])
            s._splast = None
        return r

    def k(s, name, n, args, b=128):
        if name in ("ao_step0_ga", "rt_step0", "ao_step0_th"):
            s._spph, s._spit = "start", 0
            s._spks = int(getattr(s, "_spks", -1)) + 1
        if _cap():
            c = _cat(s, name)
            if c is not None:
                _emit(s, c)
        return o_k(s, name, n, args, b)

    def ctl(s, op, it=0):
        if op == 2:
            s._spph, s._spit = "newton", int(it)
        elif op == 8:
            s._spph = "tail"
        return o_ctl(s, op, it)
    cls.record, cls.k, cls.ctl = record, k, ctl
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, "_c3cap", None) is not None:
            _stamp(CI["구간 사이"])
            s._splast = None
        return o_mark(s, kind)
    C3.mark = mark


# ======================= (S1) 국소 보폭 확인 라운드 조기 끝 =======================
WS4_HEAD = "__device__ int WS4_DN; __device__ unsigned long long WS4_BAD, WS4_NCLEAN, WS4_NROUND;\n"
WS4_K = r"""
__global__ void ws4_chk(const int* fl) {
    if (AF_SKIP) return;
    WS4_NROUND++;
    if (!WS4_DN && fl[0] + fl[1] == 0) { WS4_DN = 1; WS4_NCLEAN++; }
}
"""
WS4_SKIPK = ("ws_cand", "ws_inv_q", "ws_nk", "ws_bk", "ws_merge")


class WS4:
    mode = 0


def _ins_body(src, name, text):
    m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", src)
    assert m is not None, f"--s4w: {name} 없음"
    b = src.index("{", m.end())
    assert len(re.findall(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", src)) == 1, name
    return src[:b + 1] + text + src[b + 1:]


def ws4_src(src):
    if "ws_init(" not in src:
        return src
    assert "__device__ int AF_SKIP;" in src, "--s4w 는 허용치 판 (AF_SKIP) 위"
    src = src.replace('extern "C" {', 'extern "C" {\n' + WS4_HEAD, 1)
    src = _ins_body(src, "ws_init", "\n    if (blockIdx.x == 0 && threadIdx.x == 0) WS4_DN = 0;                               // 절 CM50 S1\n")
    if WS4.mode == 1:
        for nm in WS4_SKIPK:
            src = _ins_body(src, nm, " if (WS4_DN) return;")
    else:
        for nm, o_ in (("ws_bk", "atomicAdd(&fl[0], 1);"), ("ws_nk", "atomicAdd(&fl[1], 1);")):
            m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + nm + r"\s*\(", src)
            j = src.index("\n}\n", m.end())
            body = src[m.start():j]
            assert body.count(o_) == 1, (nm, o_)
            src = src[:m.start()] + body.replace(o_, o_ + " if (WS4_DN) atomicAdd(&WS4_BAD, 1ull);", 1) + src[j:]
    i = src.rfind("}")
    return src[:i] + WS4_K + src[i:]


def ws4_install(mode):
    from . import fem_ao_rt as AORT
    from .fem_ao_rt import P
    WS4.mode = int(mode)
    cls = AORT.AORTSolver
    o_init, o_k = cls.__init__, cls.k

    def __init__(s, *a, **kw):
        if not getattr(AORT, "_ws4_src", False):                                      # 처음 풀이기 만들 때 바깥에서 감쌈 (다른 모듈의 소스 바꾸기가 모두 끝난 뒤)
            AORT._ws4_src = True
            o_s = AORT._src
            AORT._src = lambda *a_, **k_: ws4_src(o_s(*a_, **k_))
        return o_init(s, *a, **kw)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == "ws_bk" and int(args[-1]) == 1:                                      # 라운드 (sc 1) 끝: 어긴 점 0 이면 표시
            o_k(s, "ws4_chk", 1, (args[-2],), 1)
        return r
    cls.__init__, cls.k = __init__, k


def ws4_counts(s):
    import cupy as cp
    g = lambda nm: cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global(nm))
    o = {}
    for nm, key in (("WS4_NROUND", "round"), ("WS4_NCLEAN", "clean"), ("WS4_BAD", "bad")):
        a = g(nm)
        o[key] = int(a[0])
        a[0] = 0
    return o


# ======================= (X1) 버릴 시도의 꼬리 건너뛰기 (결과 불변) =======================
# 수박 F 의 K 1 시도는 무너진 걸음 (덜 풀림 ∧ !(‖g‖²/F² ≤ --ue_bigT), C3.nfail) 이 있으면 버려짐: X3c 첫 시도 (AF_WRXON 0, --wrx_col 1) 는 입자 상태를 되돌리고
#   켠 채 다시, 쪼개기 (split_step depth > 0) 는 되돌리고 K 2 로 다시. 판정 값 ufl·ufg 는 걸음 끝 uf_end 가 쓰고 그 뒤 커널 (걸음 끝 갱신·RM 커밋·G2P·R2 끝 기준 높이)
#   은 입자·노드 상태만 바꿈 -> 되돌리는 시도에서는 버려짐. 여기서는 K 1 그래프를 uf_end 뒤에서 한 번 더 나누고 ('x' 구간), 재생 때 '버려질 수 있는 시도' 이면
#   그 경계에서 호스트가 넘침 (IP[13]) 0 · nfail > 0 을 읽어 나머지 구간 (꼬리) 을 안 띄움. 판정·되돌림·다음 시도는 그대로 (같은 ufl·ufg 를 split_step 이 다시 읽음).
#   시간 예산 (T2) 기록 tb_end 도 꼬리 구간이라 그 시도는 C_tail 표본을 안 냄 (다음 시도가 냄).
class X1:
    on = False
    hits = 0


def x1_install():
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_wmrx as WRX
    X1.on = True
    cls = AORT.AORTSolver
    o_rec, o_k = cls.record, cls.k

    def record(s, t_frame, K=None):
        s._x1K = int(K if K is not None else getattr(s, "Kcur", getattr(s, "K", 1)))
        if getattr(s, "_c3cap", None) is not None:
            s._k4b = ["start"]
        return o_rec(s, t_frame, K)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == "uf_end" and getattr(s, "_c3cap", None) is not None and getattr(s, "_x1K", 0) == 1:
            C3.mark(s, "x")
        return r
    cls.record, cls.k = record, k
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, "_c3cap", None) is not None and getattr(s, "_k4b", None) is not None:
            s._k4b.append(kind)
        return o_mark(s, kind)
    C3.mark = mark
    o_init, o_rep = C3.SegGraph.__init__, C3.SegGraph.replay

    def __init__(self, s, segs):
        o_init(self, s, segs)
        kb = getattr(s, "_k4b", None)
        self.k4 = list(kb) if (kb is not None and len(kb) == len(segs)) else None
        assert self.k4 is None or self.k4.count("x") <= 1
        s._k4b = None

    def replay(self):
        s = self.s
        if self.k4 is None or "x" not in self.k4 or not getattr(s, "_x1ok", False):
            return o_rep(self)
        mode = int(s.xg)
        st = s.c3st
        skip = False
        for i, (cond, g) in enumerate(self.segs):
            if self.k4[i] == "x" and int(s.IP[13]) == 0 and C3.nfail(s, s.ufl, s.ufg, 1) > 0:
                X1.hits += 1
                return
            if not cond:                                                                 # 아래는 fem_common3.SegGraph.replay 글자 그대로
                skip = False
                if mode == 2:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); st["ev"].append((0, e0, e1))
                else:
                    g.replay()
                continue
            if not skip:
                skip = bool(int(s.CT[0]) != 0)
            st["sk" if skip else "run"] += 1
            if mode == 1 and skip:
                continue
            if mode == 2:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); st["ev"].append((2 if skip else 1, e0, e1))
            else:
                g.replay()
    C3.SegGraph.__init__, C3.SegGraph.replay = __init__, replay
    o_on = WRX.on

    def on(s, v):
        s._x1won = int(v)
        return o_on(s, v)
    WRX.on = on
    o_split = AORT.split_step

    def split_step(s, t, h, sub_dt, depth):
        col = bool(getattr(s, "wrx", None)) and getattr(s, "_x1won", None) == 0      # X3c 첫 시도 (끈 채): 무너지면 되돌려 켠 채 다시
        prev = getattr(s, "_x1ok", False)
        s._x1ok = (int(depth) > 0) or col
        try:
            return o_split(s, t, h, sub_dt, depth)
        finally:
            s._x1ok = prev
    AORT.split_step = split_step


# ======================= 프레임 기록 =======================
def frame_install(out):
    """쪼개기 함수 (프레임 진입) 를 감싸 프레임마다 s4.jsonl 한 줄 (단계 시간 · S1 셈)."""
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    log = open(os.path.join(out, "s4.jsonl"), "w")
    dep, fr = [0], [0]

    def outer(fn):
        def w(s, *a, **kw):
            if dep[0] == 0 and SP.on:
                torch.cuda.synchronize()
                sp_g("SP_N", np.int32)[0] = 0
                torch.cuda.synchronize()
            dep[0] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                dep[0] -= 1
            if dep[0] == 0:
                rec = dict(t=fr[0])
                fr[0] += 1
                if SP.on:
                    rec["sp"] = sp_frame()
                    if SP.unk:
                        rec["unk"] = sorted(n_ for _, n_ in SP.unk)
                        SP.unk.clear()
                if WS4.mode:
                    rec["ws4"] = ws4_counts(s)
                if X1.on:
                    rec["x1"] = X1.hits
                    X1.hits = 0
                log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            return r
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
