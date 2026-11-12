"""절 CM113 (조정자 11-12 후속, 사용자 지적 "끝이 아니라 시작에서 누가 누르는지"): H9 wolf 걸음 시작점에서 칸이 처음 눌리는 원인 -- 결과 불변 기록.
fem_cellforce (절 CM111) 의 가장 안쪽 띄우기 감싸개 + 아래 더함:
  - 노드 위치 사본 (법선 성분 n·b, b = y·Ut·yh·U): 걸음 (kstep < KS) 시작 커널 rt_ga_step0 앞 n·U (이전 걸음 끝), 뒤 n·Ut (관성 예측) · n·y,
    걸음 시작 단계 (rt_ga_step0 뒤 ~ 첫 rt_grad) 동안 인자에 y 가 있는 커널마다 뒤에 n·y (이름 기록), 첫 rt_grad 앞 n·y (= 뉴턴 시작점 y0) · n·yh,
    뉴턴 반복 rt_lsup 뒤 n·y (반복 끝). 걸음 끝 = 다음 걸음 rt_ga_step0 앞 n·U, 마지막 걸음은 프레임 끝 s.U (호스트).
    사본마다 '돌았음' 표시 (풀이기 AF_SKIP 이면 안 씀).
  - 항별 기울기 (절 CM111 과 같은 정의) 를 걸음 (kstep < KS) x 뉴턴 평가 (< EV) 마다 칸에.
호스트 (다음 프레임 첫 s.step 때 앞 프레임 것, 프레임 FR0..FR1 만): 칸 (cellh 정의) 마다 사본별 r = 두께/시작 격자 두께, 연속 사본 사이 변화의
  구속 노드 몫 (T ≈ 0 꼭짓점만 옮긴 r) · 자유 노드 몫 -> <out>/sp/f<프레임>.npz, <out>/sp.jsonl (사본 이름·차례).
"""
import json
import math
import os

import numpy as np
import torch

INFO = dict(n_snap=0, n_snapdrop=0, meta={}, n_inst=0, n_alloc=0, n_wrapfn=0, n_win=0, n_delta=0, n_fin=0, n_rec=0, n_line=0, err=None, names=[], over=[], skipptr=None, cls=None)
OUT = ["."]
KS = 4
EV = 4
NS = 48
KSG = 8
FR = [3, 12]
LOG = [None]
NBK = 8
ELEM = ("rt_elem", "aq_elem", "ao_elem", "ao_elem_s")

SRC = r'''
extern "C" {
__device__ __forceinline__ bool cfs(const int* S) { return S != 0 && *S != 0; }
__global__ void cf_reset(const int* IP, int capN, int nbk, double* prev, double* acc, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= capN || n >= IP[9]) return;
    prev[n*3] = 0.0; prev[n*3+1] = 0.0; prev[n*3+2] = 0.0;
    for (int b=0;b<nbk;b++) acc[(size_t)b*capN + n] = 0.0;
}
__global__ void cf_delta(const int* IP, const double* f, double* prev, double* acc, float n0, float n1, float n2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    double a = f[n*3], b = f[n*3+1], c = f[n*3+2];
    acc[n] += (a - prev[n*3])*(double)n0 + (b - prev[n*3+1])*(double)n1 + (c - prev[n*3+2])*(double)n2;
    prev[n*3] = a; prev[n*3+1] = b; prev[n*3+2] = c;
}
// 창 끝: 작업 통 (acc2) -> 끝 칸 (acc1), n·η, 평가 번호·1/τ²
__global__ void cf_fin(const int* IP, int capN, int nbk, const double* acc2, double* acc1, const float* eta, float n0, float n1, float n2, float* en, int* itb, int code, float* itf, float it2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[1] = code; itf[1] = it2; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) acc1[(size_t)b*capN + n] = acc2[(size_t)b*capN + n];
    en[n] = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
}
__global__ void cf_g0(const int* IP, const float* r0, float n0, float n1, float n2, float* gn, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    gn[n] = -(r0[n*3]*n0 + r0[n*3+1]*n1 + r0[n*3+2]*n2);
}
__global__ void cf_node(const int* IP, const float* sk, const int* pm, const float* me, const int* w8b, float* osk, int* opm, float* ome, int* ow8, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    osk[n] = sk[n]; opm[n] = pm[n]; ome[n] = me[n]; ow8[n] = w8b ? w8b[n] : -1;
}
// 끝 칸 -> 시작 칸 (프레임 첫 걸음 첫 평가 뒤 한 번): acc·en·gn·node
__global__ void cf_cp(const int* IP, int capN, int nbk, const double* acc1, double* acc0, const float* en1, float* en0, const float* gn1, float* gn0,
                      const float* sk1, float* sk0, const int* pm1, int* pm0, const float* me1, float* me0, const int* w81, int* w80,
                      int* itb, float* itf, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[0] = itb[1]; itf[0] = itf[1]; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) acc0[(size_t)b*capN + n] = acc1[(size_t)b*capN + n];
    en0[n] = en1[n]; gn0[n] = gn1[n]; sk0[n] = sk1[n]; pm0[n] = pm1[n]; me0[n] = me1[n]; w80[n] = w81[n];
}
__global__ void cf_frame(int* itb) { itb[0] = -1; itb[1] = -1; }
__global__ void sp_snap(const int* IP, const float* b, float n0, float n1, float n2, float* o, int* flg, int slot, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) flg[slot] = 1;
    if (n >= IP[9]) return;
    o[n] = b[n*3]*n0 + b[n*3+1]*n1 + b[n*3+2]*n2;
}
__global__ void sp_snap1(const int* IP, const float* b, float* o, int* flg, int slot, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) flg[slot] = 1;
    if (n >= IP[9]) return;
    o[n] = b[n];
}
// 절 CM117: 평가별 |r0|² (풀이기 잔차, 안 접은 것) -> gs[idx] (먼저 0 으로; 둘 다 AF_SKIP 지킴)
__global__ void sp_g0z(double* gs, int idx, const int* S) { if (cfs(S)) return; gs[idx] = 0.0; }
__global__ void sp_g2(const int* IP, const float* r0, double* gs, int idx, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v = 0.0;
    if (n < IP[9]) { float a = r0[n*3], b = r0[n*3+1], c = r0[n*3+2]; v = (double)a*a + (double)b*b + (double)c*c; }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0 && v != 0.0) atomicAdd(&gs[idx], v);
}
__global__ void sp_gclr(double* gs, int n, int* gc) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) { gs[i] = -1.0; gc[i] = 0; } }
__global__ void sp_gcnt(int* gc, int idx, const int* S) { if (cfs(S)) return; gc[idx] += 1; }
__global__ void sp_clr(int* flg, int n, int* itb, int m) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) flg[i] = 0; if (i < m) itb[i] = -1; }
// 걸음 x 평가 칸으로: 작업 통 -> acc[slot], n·η, 번호
__global__ void sp_fin(const int* IP, int capN, int nbk, const double* acc2, double* accs, const float* eta, float n0, float n1, float n2, float* en, int* itb, int slot, int code, float* itf, float it2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[slot] = code; itf[slot] = it2; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) accs[(size_t)b*capN + n] = acc2[(size_t)b*capN + n];
    en[n] = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
}
}
'''


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _dump():
    try:
        with open(os.path.join(OUT[0], "spinfo.json"), "w") as f_:
            json.dump(INFO, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _bucket(name):
    nm = INFO["names"]
    if name in nm:
        return nm.index(name)
    if len(nm) < NBK - 1:
        nm.append(name)
        return len(nm) - 1
    if len(nm) == NBK - 1:
        nm.append("기타")
    if name not in INFO["over"]:
        INFO["over"].append(name)
    return NBK - 1


class Buf:
    def __init__(s):
        import cupy as cp
        s.mod = cp.RawModule(code=SRC)
        s.fn = {n: s.mod.get_function(n) for n in ("cf_reset", "cf_delta", "cf_fin", "cf_g0", "cf_node", "cf_cp", "cf_frame", "sp_snap", "sp_snap1", "sp_clr", "sp_fin", "sp_g0z", "sp_g2", "sp_gclr", "sp_gcnt")}
        s.capN = 0
        s.kstep = -1; s.evi = 0; s.win = False; s.skip = None; s.ns = 0; s.phase = 0; s.lsit = 0

    def alloc(s, sv):
        capN = int(sv.capN)
        if capN == s.capN:
            return
        dev = sv.dev
        s.capN = capN
        s.prev = torch.zeros(capN, 3, device=dev, dtype=torch.float64)
        s.acc = torch.zeros(KS * EV + 1, NBK, capN, device=dev, dtype=torch.float64)       # [걸음·평가 ..., 창 작업 통]
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
        s.gs = torch.full((KSG * 2,), -1.0, device=dev, dtype=torch.float64)                 # 절 CM117: 걸음 x (첫 평가, 마지막 평가) |r0|²
        s.gc = torch.zeros(KSG * 2, device=dev, dtype=torch.int32)                           #           걸음별 평가 수
        INFO["n_alloc"] += 1

    def go(s, name, n, args, b=256):
        from . import fem_rt_cuda as RTC
        with RTC._Stream():
            s.fn[name](((int(n) + b - 1) // b,), (b,), tuple(args))


def _fl(sv):
    FL = sv.floor or {}
    nr = FL.get("NR", [0.0, 0.0, 1.0])
    return [float(q) for q in nr], float(FL.get("PN", 0.0))


def _skip(sv, W):
    if W.skip is None:
        try:
            W.skip = np.uint64(int(sv.mod.get_global("AF_SKIP")))
        except Exception:
            W.skip = np.uint64(0)
        INFO["skipptr"] = bool(int(W.skip))
    return W.skip


def _has_ptr(args, p):
    for a in args:
        if isinstance(a, (np.integer, int)) and not isinstance(a, bool) and int(a) == p:
            return True
    return False


class _Call:
    """가장 안쪽 띄우기 하나 (s.fn[name](grid, block, args)) 를 감쌈. 원 커널은 꼭 한 번만 띄움; 기록 쪽 실패는 INFO err 에만."""

    def __init__(s_, sv, name, f):
        s_.sv, s_.name, s_.f = sv, name, f

    def __call__(s_, grid, block, args, **kw):
        sv, name, f = s_.sv, s_.name, s_.f
        W = getattr(sv, "_cf", None)
        if W is None or not W.capN:
            return f(grid, block, args, **kw)
        post = None
        try:
            post = _pre(sv, W, name, args)
        except Exception as e:
            INFO["err"] = f"pre {name}: {e!r}"[:400]
        r = f(grid, block, args, **kw)
        if post is not None:
            try:
                post()
            except Exception as e:
                INFO["err"] = f"post {name}: {e!r}"[:400]
        return r


def _snap(sv, W, S, NR, ptr, tag):
    """걸음 W.kstep 의 다음 사본 자리에 n·(ptr 버퍼)."""
    k = W.kstep
    if k < 0 or k >= KS:
        return
    if W.ns >= NS:
        INFO["n_snapdrop"] += 1
        return
    j = W.ns; W.ns += 1
    W.go("sp_snap", W.capN, (P(sv.IP), ptr, *NR, P(W.sn[k, j]), P(W.flg), np.int32(k * NS + j), S))
    INFO["meta"][f"{k}:{j}"] = tag
    INFO["n_snap"] += 1


def _snap1(sv, W, S, ptr, tag):
    k = W.kstep
    if k < 0 or k >= KS or W.ns >= NS:
        return
    j = W.ns; W.ns += 1
    W.go("sp_snap1", W.capN, (P(sv.IP), ptr, P(W.sn[k, j]), P(W.flg), np.int32(k * NS + j), S))
    INFO["meta"][f"{k}:{j}"] = tag


def _pre(sv, W, name, args):
    """원 커널 앞 일을 하고, 뒤에 할 일 (함수 또는 None) 을 돌려줌."""
    i32, f32 = np.int32, np.float32
    capN = W.capN
    if name == "rt_reset":                                                                     # 프레임 시작
        W.kstep, W.evi, W.win, W.ns, W.phase = -1, 0, False, 0, 0
        def _p():
            W.go("sp_clr", KS * NS, (P(W.flg), i32(KS * NS), P(W.itb), i32(KS * EV)), 256)
            W.go("sp_gclr", KSG * 2, (P(W.gs), i32(KSG * 2), P(W.gc)), 64)
        return _p
    S = _skip(sv, W)
    nr, pn = _fl(sv)
    NR = [f32(q) for q in nr]
    if name == "rt_ga_step0":                                                                  # 걸음 시작: 앞 = 이전 걸음 끝 U
        if W.kstep + 1 < KS:
            W.kstep += 1; W.ns = 0
            _snap(sv, W, S, NR, P(sv.U), ("U_이전끝", name, -1))
            _snap(sv, W, S, NR, args[2], ("v_시작", name, -1))                              # 절 CM114: 걸음 시작 노드 속도 vk
            W.kstep -= 1

        def _p():
            W.kstep += 1; W.evi = 0; W.win = False
            if W.kstep >= KS:
                W.ns = NS
            W.phase = 1; W.lsit = 0
            _snap(sv, W, S, NR, P(sv.Ut), ("Ut_관성예측", name, -1))
            _snap(sv, W, S, NR, P(sv.y), ("y", name, -1))
        return _p
    if W.kstep < 0:
        return None
    fp = sv.f.data_ptr(); yp = sv.y.data_ptr()
    if name == "rt_zero_d" and len(args) >= 5 and int(args[4]) == fp:                        # 창 시작
        def _p():
            W.go("cf_reset", capN, (P(sv.IP), i32(capN), i32(NBK), P(W.prev), P(W.acc[KS * EV]), S))
            W.win = True; INFO["n_win"] += 1
        return _p
    if name.startswith("rt_grad") and W.win:                                                   # 창 끝 (평가)
        IPp = args[0]
        if W.phase == 1:                                                                       # 첫 평가 = 뉴턴 시작점
            _snap(sv, W, S, NR, P(sv.y), ("y0_시작점", name, -1))
            _snap(sv, W, S, NR, P(sv.yh), ("yh_기준", name, -1))
            W.phase = 2
        W.go("cf_delta", capN, (IPp, P(sv.f), P(W.prev), P(W.acc[KS * EV, _bucket("~rt_grad")]), *NR, S))
        k, e = W.kstep, W.evi
        if k < KS and e < EV:
            sl = k * EV + e
            W.go("sp_fin", capN, (IPp, i32(capN), i32(NBK), P(W.acc[KS * EV]), P(W.acc[sl]), args[8], *NR, P(W.en[sl]), P(W.itb), i32(sl), i32(k * 100 + e), P(W.itf), f32(float(args[2])), S))
            if e == 0:
                w8b = getattr(sv, "w8b", None)
                W.go("cf_node", capN, (IPp, args[3], args[4], args[1], P(w8b), P(W.sk[k]), P(W.pm[k]), P(W.me[k]), P(W.w8[k]), S))
        INFO["n_fin"] += 1

        def _p():
            if k < KS and e < EV:
                W.go("cf_g0", capN, (IPp, args[-1], *NR, P(W.gn[k * EV + e]), S))
            if 0 <= k < KSG:                                                                   # 절 CM117: |r0|² 첫·마지막 평가
                for idx in ((2 * k, 2 * k + 1) if e == 0 else (2 * k + 1,)):
                    W.go("sp_g0z", 1, (P(W.gs), i32(idx), S), 1)
                    W.go("sp_g2", capN, (IPp, args[-1], P(W.gs), i32(idx), S), 256)
                W.go("sp_gcnt", 1, (P(W.gc), i32(2 * k + 1), S), 1)
            W.evi += 1; W.win = False
        return _p
    if W.win and _has_ptr(args, fp):                                                           # 창 안 f 쓰는 커널
        IPp = P(sv.IP)
        bn = "탄성" if name in ELEM else name
        INFO["n_delta"] += 1
        return lambda: W.go("cf_delta", capN, (IPp, P(sv.f), P(W.prev), P(W.acc[KS * EV, _bucket(bn)]), *NR, S))
    if W.phase == 1 and name in ("ao_pbc_k", "mt_pbc_kphi"):                                 # 절 CM115: 기본 구속 몫 (e^-x)
        return lambda: _snap1(sv, W, S, args[7], ("kb_기본", name, -1))
    if W.phase == 1 and name == "w5s_kb":                                                      # 절 CM115: S_구속 이 덮어쓴 구속 몫
        return lambda: _snap1(sv, W, S, args[4], ("kb_S", name, -1))
    if W.phase == 1 and _has_ptr(args, yp):                                                    # 시작 단계: y 를 받는 커널 뒤 사본
        return lambda: _snap(sv, W, S, NR, P(sv.y), ("y", name, -1))
    if W.phase == 2 and name == "rt_lsup":                                                     # 뉴턴 갱신 뒤
        it = W.lsit; W.lsit += 1
        return lambda: _snap(sv, W, S, NR, P(sv.y), ("y_갱신", name, it))
    return None


class _CFn(dict):
    """풀이기 s.fn 감쌈: 내용은 안쪽 사전 (inner) 에 두고, 꺼낼 때 _Call 로 돌려줌."""

    def __init__(s_, inner, owner):
        super().__init__()
        s_.inner, s_.owner = inner, owner

    def __getitem__(s_, name):
        return _Call(s_.owner, name, s_.inner[name])

    def __setitem__(s_, k, v):
        s_.inner[k] = v

    def __contains__(s_, k):
        return k in s_.inner

    def get(s_, k, d=None):
        return s_.inner.get(k, d)

    def __getattr__(s_, a):
        if a in ("inner", "owner"):
            raise AttributeError(a)
        return getattr(s_.inner, a)


def _ensure_fn(s):
    if not isinstance(getattr(s, "fn", None), _CFn):
        s.fn = _CFn(s.fn, s)
        INFO["n_wrapfn"] += 1


def _wrap(R):
    ok, os_, oa = R.k, R.step, R.alloc

    def alloc(s, need):
        r = oa(s, need)
        if getattr(s, "_cf", None) is None:
            s._cf = Buf(); INFO["n_inst"] += 1; INFO["cls"] = type(s).__name__
        s._cf.alloc(s)
        return r

    def k(s, name, n, args, b=128):
        if getattr(s, "_cf", None) is not None and hasattr(s, "fn"):
            _ensure_fn(s)
        return ok(s, name, n, args, b)

    def step(s, t, *a, **kw):
        # 칸 계산은 다음 프레임 첫 s.step 들어갈 때 (또는 끝날 때) 앞 프레임 것을 한 번 -- 같은 프레임 다시 하기 (쪼개기) 사이에 호스트 시간을 안 넣음
        # (시간 예산 시계는 프레임 첫 시도 첫 커널부터 벽시계라 시도 사이 호스트 시간이 결과를 바꿀 수 있음). 그 사이 U·Np·fx·기록 버퍼는 받아들인 마지막 시도 그대로.
        pend = getattr(s, "_cf_pend", None)
        if pend is not None and pend != t:
            _flush(s, pend)
            s._cf_pend = None
        r = os_(s, t, *a, **kw)
        if getattr(s, "_cf", None) is not None and s._cf.capN:
            s._cf_pend = t
            if not getattr(s, "_cf_atexit", False):
                import atexit
                s._cf_atexit = True
                atexit.register(lambda s_=s: (_flush(s_, getattr(s_, "_cf_pend", None)), _dump()))
        return r

    R.alloc, R.k, R.step = alloc, k, step


def _flush(s, t):
    W = getattr(s, "_cf", None)
    if t is None or W is None or not W.capN or LOG[0] is None:
        return
    try:
        torch.cuda.synchronize()
        if bool(torch.isfinite(s.x[:1]).all()):
            o = _analyze(s, W, t)
            LOG[0].write(json.dumps(o, ensure_ascii=False) + "\n"); LOG[0].flush(); INFO["n_line"] += 1
    except Exception as e:
        INFO["err"] = "analyze: " + repr(e)[:400]
    INFO["n_rec"] += 1
    s._cf_pend = None
    if INFO["n_rec"] in (1, 2, 5) or INFO["n_rec"] % 10 == 0:
        _dump()


def _analyze(s, W, t):
    fr = int(round(float(t) / float(s.h))) + 1
    o = dict(fr=fr, err=INFO["err"])
    if not (FR[0] <= fr <= FR[1]):
        o["skip"] = "프레임 밖"
        return o
    IP = s.IP[:16].cpu().numpy().astype(np.int64)
    nn = int(IP[9]); nb = IP[6:9].astype(np.int64); bmin = IP[0:3].astype(np.int64)
    nb1, nb2 = int(nb[1]), int(nb[2])
    dev = s.U.device
    nr, pn = _fl(s)
    n_ = torch.tensor(nr, device=dev, dtype=torch.float64); n_ = n_ / n_.norm()
    A = s.A.double().reshape(3, 3)
    dx = float(A[:, 0].norm())
    ids = torch.arange(nn, device=dev, dtype=torch.int64)
    ijk = torch.stack([ids // (nb1 * nb2), (ids // nb2) % nb1, ids % nb2], 1) + torch.tensor(bmin, device=dev)
    h0 = (ijk.double() @ A.T) @ n_ - pn
    N = int(s.N)
    c = s.Np[:N].long() + (s.fx[:N] >= 1.0).long()
    OFF = torch.tensor([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.int64)
    nid = ((c[:, None, 0] + OFF[None, :, 0]) * nb1 + (c[:, None, 1] + OFF[None, :, 1])) * nb2 + (c[:, None, 2] + OFF[None, :, 2])
    okp = (nid >= 0).all(1) & (nid < nn).all(1)
    key = torch.where(okp, nid[:, 0], torch.full_like(nid[:, 0], -1))
    uk, inv = torch.unique(key, return_inverse=True)
    first = torch.full((len(uk),), N, device=dev, dtype=torch.int64).scatter_reduce(0, inv, torch.arange(N, device=dev), reduce="amin")
    nidc = nid[first[uk >= 0]]
    cmap = torch.cumsum((uk >= 0).long(), 0) - 1
    pcell = torch.where(uk[inv] >= 0, cmap[inv], torch.full_like(inv, -1))                  # 절 CM114: 입자 -> 칸 번호 (입자 차례는 프레임 사이 고정)
    trel = getattr(s, "trel", None)
    held = (trel[:N] > float(t)).float() if trel is not None else torch.zeros(N, device=dev)
    nc_ = int(nidc.shape[0])
    cnt = torch.zeros(nc_, device=dev).index_add_(0, pcell.clamp(min=0), (pcell >= 0).float())
    hf = torch.zeros(nc_, device=dev).index_add_(0, pcell.clamp(min=0), held * (pcell >= 0).float()) / cnt.clamp_min(1)
    a0 = h0[nidc]
    hs = a0.max(1).values - a0.min(1).values
    hb = a0.min(1).values
    lay = torch.clamp(torch.floor(hb / dx + 1e-6), min=0).long()
    order = torch.argsort(a0, dim=1)
    wsg = torch.empty_like(a0)
    wsg.scatter_(1, order[:, :4], -1.0 / math.sqrt(8.0)); wsg.scatter_(1, order[:, 4:], 1.0 / math.sqrt(8.0))
    mn = s.mn[:nn].double()
    gz = float(sum(float(q) * float(p) for q, p in zip(s.g, n_.tolist())))
    flg = W.flg.cpu().numpy().reshape(KS, NS)
    itb = W.itb.cpu().tolist(); itf = W.itf.cpu().tolist()
    names = list(INFO["names"]); nbu = len(names)

    def rr(v):
        a = a0 + v[nidc]
        return (a.max(1).values - a.min(1).values) / hs.clamp_min(1e-12)
    out = dict(pcell=pcell.int().cpu().numpy(), held=hf.half().cpu().numpy(), lay=lay.short().cpu().numpy(), hb=hb.float().cpu().numpy(), m=mn[nidc].sum(1).float().cpu().numpy(), dx=dx)
    steps = []
    for k in range(KS):
        valid = [j for j in range(NS) if flg[k, j]]
        if not valid:
            continue
        sk = W.sk[k, :nn].double(); pm = W.pm[k, :nn]
        tn = torch.where(pm == 0, sk, torch.where(pm == 2, torch.zeros_like(sk), torch.ones_like(sk)))
        cons = (tn.abs() < 1e-3)                                                              # 구속 (T ≈ 0) 노드
        ccn = cons[nidc]
        out[f"k{k}_ncons"] = ccn.sum(1).short().cpu().numpy()
        out[f"k{k}_next"] = (W.w8[k, :nn][nidc] >= 0).sum(1).short().cpu().numpy()
        tags = []; R = []; RC = []; RF = []
        prev = None
        snapv = {}; kbs = {}
        for j in valid:
            tg = INFO["meta"].get(f"{k}:{j}")
            v = W.sn[k, j, :nn].double()
            if tg:
                snapv.setdefault(tg[0] + ":" + tg[1], v)                                     # 첫 사본 (같은 이름이면 처음 것)
                snapv[tg[0] + ":" + tg[1] + ":last"] = v
            if tg and tg[0].startswith("kb_"):
                kbs[tg[0]] = W.sn[k, j, :nn].clone()
                continue
            if tg and tg[0].startswith("v_"):
                vt = (v[nidc] * (wsg > 0)).sum(1) / 4.0; vb = (v[nidc] * (wsg < 0)).sum(1) / 4.0
                out[f"k{k}_vrate"] = ((vt - vb) / hs.clamp_min(1e-12)).float().cpu().numpy()   # 칸 세로 압축률 (1/s, 음수 = 줄어듦)
                continue
            tags.append([str(q) for q in tg] if tg else ["?", "?", -1])
            R.append(rr(v))
            if prev is None or tg[0].startswith("Ut") or tg[0].startswith("yh"):
                RC.append(torch.full_like(hs, float("nan"))); RF.append(torch.full_like(hs, float("nan")))
            else:
                vC = torch.where(cons, v, prev); vF = torch.where(cons, prev, v)
                RC.append(rr(vC)); RF.append(rr(vF))
            if not (tg and (tg[0].startswith("Ut") or tg[0].startswith("yh"))):
                prev = v
        tags = [t for t in tags if not t[0].startswith("v_")]
        # 절 CM115: 구속 몫 정의별 부분 구속 수 (같은 상태), 앞면 노드, 노드별 입자 수, 확장 노드
        act = mn > 0
        w8k = W.w8[k, :nn] >= 0
        kfin = getattr(s, "kb", None)
        if kfin is not None:
            kbs["kb_최종"] = kfin[:nn].clone()
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
        for nmk, kv in kbs.items():
            kv = kv.double()
            part = act & (kv > 1e-3) & (kv <= 0.999)
            held_ = act & (kv <= 1e-3)
            KBS[nmk] = dict(partial=int(part.sum()), held=int(held_.sum()), free=int((act & (kv > 0.999)).sum()),
                            partial_ext=int((part & w8k).sum()), partial_front=int((part & front).sum()),
                            corner_partial=int(part[nidc].sum()), partial_cnt_med=float(cntn[part].median()) if bool(part.any()) else None)
            out[f"k{k}_{nmk}"] = kv[nidc].half().cpu().numpy()
        out[f"k{k}_kbstat"] = np.array(json.dumps(dict(KBS=KBS, nact=int(act.sum()), front=int(front.sum()), front_corner=int(front[nidc].sum()),
                                                       ext=int((act & w8k).sum()), cnt_med=float(cntn[act].median()), cnt_front_med=float(cntn[front].median()) if bool(front.any()) else None,
                                                       held_part=float(heldp.mean())), ensure_ascii=False))
        # 절 CM114: 꼭짓점 구속 몫·고착·덮어쓴 변위
        kb = getattr(s, "kb", None)
        if kb is not None:
            out[f"k{k}_kb"] = kb[:nn][nidc].half().cpu().numpy()
        out[f"k{k}_stk"] = (s.an[:nn] / s.mqn[:nn].clamp_min(1e-30)).clamp(0, 1)[nidc].half().cpu().numpy()
        out[f"k{k}_sk"] = sk[nidc].half().cpu().numpy()
        out[f"k{k}_mnc"] = mn[nidc].float().cpu().numpy()
        Ut_ = snapv.get("Ut_관성예측:rt_ga_step0")
        ymt = snapv.get("y:mt_floor_n")
        yh_ = snapv.get("yh_기준:rt_grad")
        if Ut_ is not None and ymt is not None:
            dm_ = (ymt - Ut_)[nidc]
            out[f"k{k}_dmt"] = dm_.float().cpu().numpy()                                     # mt_floor_n 이 덮어쓴 법선 변위 (m)
            out[f"k{k}_dmt_tb"] = ((dm_ * (wsg > 0)).sum(1) / 4.0 - (dm_ * (wsg < 0)).sum(1) / 4.0).float().cpu().numpy()   # 위 평균 − 아래 평균 (음수 = 칸을 줄임)
        if Ut_ is not None and yh_ is not None:
            out[f"k{k}_dyh"] = (yh_ - Ut_)[nidc].float().cpu().numpy()
        out[f"k{k}_r"] = torch.stack(R).float().cpu().numpy()
        out[f"k{k}_rC"] = torch.stack(RC).float().cpu().numpy()
        out[f"k{k}_rF"] = torch.stack(RF).float().cpu().numpy()
        # 평가별 항 (절 CM111 정의)
        me = W.me[k, :nn].double()
        FE = []; ev_ok = []
        for e in range(EV):
            sl = k * EV + e
            if itb[sl] < 0:
                continue
            it2 = float(itf[sl]); en = W.en[sl, :nn].double(); gn = W.gn[sl, :nn].double(); acc = W.acc[sl, :nbu, :nn]
            T = [mn * it2 * en + tn * mn * gz, -tn * mn * gz, (me - mn) * it2 * en] + [tn * acc[b] for b in range(nbu)] + [me * it2 * en + tn * acc.sum(0), gn]
            FE.append(torch.stack([(g_[nidc] * wsg).sum(1) for g_ in T], 0).float().cpu().numpy()); ev_ok.append(e)
        if FE:
            out[f"k{k}_F"] = np.stack(FE, 0)
        steps.append(dict(k=k, tags=tags, ev=ev_ok, ncons=int(cons[mn > 0].sum()), nact=int((mn > 0).sum())))
    # 마지막 걸음 끝 = 프레임 끝 U
    out["rEnd"] = rr(s.U[:nn].double() @ n_).float().cpu().numpy()
    out["terms"] = np.array(["관성", "중력", "질량하한"] + names + ["합", "풀이기잔차"])
    os.makedirs(os.path.join(OUT[0], "sp"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT[0], "sp", f"f{fr:03d}.npz"), steps=np.array(json.dumps(steps, ensure_ascii=False)), **out)
    gs = W.gs.cpu().tolist(); gc = W.gc.cpu().tolist()
    o.update(ncell=int(len(hs)), steps=steps, snapdrop=INFO["n_snapdrop"], g2=[[gs[2 * k], gs[2 * k + 1], gc[2 * k + 1]] for k in range(KSG) if gs[2 * k] >= 0])
    return o


def install(out, fr0=3, fr1=12):
    FR[0], FR[1] = int(fr0), int(fr1)
    if os.environ.get("AF_SP_FR"):                                                          # 절 CM116: 기록 프레임 범위 (예 "1,50")
        FR[0], FR[1] = (int(q) for q in os.environ["AF_SP_FR"].split(","))
    from . import fem_an_rt as ART
    OUT[0] = out
    LOG[0] = open(os.path.join(out, "sp.jsonl"), "w")
    _wrap(ART.RTSolver)
    print(f"[절 CM113] startpt 켬 -- 걸음 시작 단계·뉴턴 갱신별 칸 r, 평가별 항 (결과 불변), 프레임 {FR}", flush=True)
    _dump()
