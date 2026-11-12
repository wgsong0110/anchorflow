"""절 CM111 (사용자 11-12): H9 wolf 칸 세로 압축 -- 증분 포텐셜 항별 일반화 힘 기록 (결과 불변: 풀이 버퍼는 읽기만, 이 모듈 버퍼에만 씀; 별도 cupy 모듈, 같은 스트림·그래프 안).

뉴턴 기울기 (rt_grad): g = M_eff/τ² η + T f  (η = 바뀐 변수, y = ŷ + T η; ŷ 에 τ² g_중력 이 들어 있음; f = 요소·벌점·장벽 커널이 더한 힘 기울기).
기록 (가장 안쪽 = 풀이기 s.fn 사전을 감쌈 -> 모든 s.k 경로의 실제 띄우기를 봄):
  - 창 = 뉴턴 평가 하나: rt_zero_d(f) (f 를 0 으로) 뒤 ~ rt_grad 앞. 창 안에서 인자에 f 포인터가 있는 커널마다 띄우기 앞·뒤로 f − prev 의 법선 성분 (n·Δf) 을
    이름별 통 acc[b] 에 더함 (앞 = '~이름' 통: 그 커널 앞에 s.k 밖에서 f 를 바꾼 몫, 뒤 = '이름' 통). rt_grad 앞에서 한 번 더 ('~rt_grad').
    -> Σ_b acc[b] = n·f (rt_grad 가 읽는 f). 통 0 = 요소 커널 (탄성: rt_elem·aq_elem·ao_elem·ao_elem_s), 통 1 = rt_bar_n (장벽), 나머지 = 처음 본 순서로 이름.
  - rt_grad 앞: n·η (en), 그 평가의 1/τ² (rt_grad 인자), 뉴턴 평가 번호·걸음 번호. 걸음 첫 평가에 sk·pm·M_eff·확장 표시 (w8b) 사본.
    rt_grad 뒤: 풀이기 잔차 r0 (= −g) 의 법선 성분 -> 검사 (우리가 다시 모은 합과 같아야 함).
  - 모든 기록 커널은 풀이기 AF_SKIP (허용치 조기 종료·시간 예산이 건너뛴 반복) 을 지킴 -> 끝 칸 = 실제로 돈 마지막 평가.
  - 칸 둘: 시작 = 프레임 첫 걸음의 첫 평가 (걸음 시작점), 끝 = 마지막으로 돈 평가 (덜 풀린 걸음은 마지막 갱신 한 번 전 = 절 CM107 '걸음 끝 잔차' 와 같은 반복).
호스트 (s.step 뒤, 그래프 밖): 노드별 기울기 법선 성분 (η 공간; T = I 인 노드는 y 공간과 같음)
    관성 (중력 뺌) = m/τ² (n·η) + t_n m (g·n) · 중력 = −t_n m (g·n) · 질량 하한 = (M_eff − m)/τ² (n·η) · 통 b = t_n acc[b] · 합 = M_eff/τ² (n·η) + t_n Σ acc
    (t_n = n·T n: pm 0 -> sk, pm 1 -> 1, pm 2 -> 0).
  칸 = 입자 받침 base + (fx ≥ 1) 의 낮은 모서리 8 꼭짓점 노드 (절 CM112 cellh 와 같음). 꼭짓점을 걸음 시작 높이 순으로 아래 4 · 위 4,
  d_c = (아래 +n, 위 −n)/√8. 일반화 힘 F = −g·d_c = (Σ_위 n·g − Σ_아래 n·g)/√8 (양수 = 칸을 누름, 음수 = 벌림).
  r = 프레임 끝 두께 / 시작 두께 (꼭짓점 높이 최대 − 최소, 끝 = + 프레임 끝 U), 층 = floor(칸 바닥 시작 높이 / dx).
  -> <out>/cf/f<프레임>.npz (칸별), <out>/cf.jsonl (프레임 요약), <out>/cfinfo.json.
"""
import json
import math
import os

import numpy as np
import torch

INFO = dict(n_inst=0, n_alloc=0, n_wrapfn=0, n_win=0, n_delta=0, n_fin=0, n_rec=0, n_line=0, err=None, names=[], over=[], skipptr=None, cls=None)
OUT = ["."]
LOG = [None]
NBK = 12
ELEM = ("rt_elem", "aq_elem", "ao_elem", "ao_elem_s")

SRC = r'''
extern "C" {
__device__ __forceinline__ bool cfs(const int* S) { return S != 0 && *S != 0; }
__global__ void cf_reset(int capN, int nbk, double* prev, double* acc, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= capN) return;
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
}
'''


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _dump():
    try:
        with open(os.path.join(OUT[0], "cfinfo.json"), "w") as f_:
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
        s.fn = {n: s.mod.get_function(n) for n in ("cf_reset", "cf_delta", "cf_fin", "cf_g0", "cf_node", "cf_cp", "cf_frame")}
        s.capN = 0
        s.kstep = -1; s.evi = 0; s.win = False; s.skip = None

    def alloc(s, sv):
        capN = int(sv.capN)
        if capN == s.capN:
            return
        dev = sv.dev
        s.capN = capN
        s.prev = torch.zeros(capN, 3, device=dev, dtype=torch.float64)
        s.acc = torch.zeros(3, NBK, capN, device=dev, dtype=torch.float64)                # [시작 · 끝 · 창 작업 통]
        s.en = torch.zeros(2, capN, device=dev, dtype=torch.float32)
        s.gn = torch.zeros(2, capN, device=dev, dtype=torch.float32)
        s.sk = torch.zeros(2, capN, device=dev, dtype=torch.float32)
        s.pm = torch.zeros(2, capN, device=dev, dtype=torch.int32)
        s.me = torch.zeros(2, capN, device=dev, dtype=torch.float32)
        s.w8 = torch.full((2, capN), -1, device=dev, dtype=torch.int32)
        s.itb = torch.full((2,), -1, device=dev, dtype=torch.int32)
        s.itf = torch.zeros(2, device=dev, dtype=torch.float32)
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


def _pre(sv, W, name, args):
    """원 커널 앞 일을 하고, 뒤에 할 일 (함수 또는 None) 을 돌려줌."""
    i32, f32 = np.int32, np.float32
    capN = W.capN
    if name == "rt_reset":                                                                     # 프레임 시작 (record 첫 커널)
        W.kstep, W.evi, W.win = -1, 0, False
        return lambda: W.go("cf_frame", 1, (P(W.itb),), 1)
    if name == "rt_ga_step0":                                                                  # 걸음 시작
        def _p():
            W.kstep += 1; W.evi = 0; W.win = False
        return _p
    if W.kstep < 0:
        return None
    S = _skip(sv, W)
    nr, pn = _fl(sv)
    NR = [f32(q) for q in nr]
    fp = sv.f.data_ptr()
    if name == "rt_zero_d" and len(args) >= 5 and int(args[4]) == fp:                        # 창 시작: f = 0
        def _p():
            W.go("cf_reset", capN, (i32(capN), i32(NBK), P(W.prev), P(W.acc[2]), S))
            W.win = True; INFO["n_win"] += 1
        return _p
    if not W.win:
        return None
    if name.startswith("rt_grad"):                                                             # 창 끝
        IPp = args[0]
        W.go("cf_delta", capN, (IPp, P(sv.f), P(W.prev), P(W.acc[2, _bucket("~rt_grad")]), *NR, S))
        it2 = float(args[2])
        W.go("cf_fin", capN, (IPp, i32(capN), i32(NBK), P(W.acc[2]), P(W.acc[1]), args[8], *NR, P(W.en[1]), P(W.itb), i32(W.kstep * 100 + W.evi), P(W.itf), f32(it2), S))
        if W.evi == 0:
            w8b = getattr(sv, "w8b", None)
            W.go("cf_node", capN, (IPp, args[3], args[4], args[1], P(w8b), P(W.sk[1]), P(W.pm[1]), P(W.me[1]), P(W.w8[1]), S))
        INFO["n_fin"] += 1
        k0 = (W.kstep == 0 and W.evi == 0)

        def _p():
            W.go("cf_g0", capN, (IPp, args[-1], *NR, P(W.gn[1]), S))
            if k0:
                W.go("cf_cp", capN, (IPp, i32(capN), i32(NBK), P(W.acc[1]), P(W.acc[0]), P(W.en[1]), P(W.en[0]), P(W.gn[1]), P(W.gn[0]), P(W.sk[1]), P(W.sk[0]),
                                     P(W.pm[1]), P(W.pm[0]), P(W.me[1]), P(W.me[0]), P(W.w8[1]), P(W.w8[0]), P(W.itb), P(W.itf), S))
            W.evi += 1; W.win = False
        return _p
    if _has_ptr(args, fp):                                                                     # 창 안 f 쓰는 (쓸 수 있는) 커널
        IPp = P(sv.IP)
        bn = "탄성" if name in ELEM else name
        W.go("cf_delta", capN, (IPp, P(sv.f), P(W.prev), P(W.acc[2, _bucket("~" + bn)]), *NR, S))
        INFO["n_delta"] += 1
        return lambda: W.go("cf_delta", capN, (IPp, P(sv.f), P(W.prev), P(W.acc[2, _bucket(bn)]), *NR, S))
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
    IP = s.IP[:16].cpu().numpy().astype(np.int64)
    nn = int(IP[9]); nb = IP[6:9].astype(np.int64); bmin = IP[0:3].astype(np.int64)
    nb1, nb2 = int(nb[1]), int(nb[2]); nbx = int(nb[0] * nb[1] * nb[2])
    dev = s.U.device
    nr, pn = _fl(s)
    n_ = torch.tensor(nr, device=dev, dtype=torch.float64); n_ = n_ / n_.norm()
    A = s.A.double().reshape(3, 3)
    dx = float(A[:, 0].norm())
    fr = int(round(float(t) / float(s.h))) + 1
    itb = W.itb.cpu().tolist(); itf = W.itf.cpu().tolist()
    o = dict(fr=fr, t=float(t), nn=nn, nbx=nbx, itb=itb, itf=itf, names=list(INFO["names"]), err=INFO["err"])
    if itb[0] < 0 or itb[1] < 0:
        o["skip"] = "기록 없음"
        return o
    # 노드 시작·끝 높이
    ids = torch.arange(nn, device=dev, dtype=torch.int64)
    ijk = torch.stack([ids // (nb1 * nb2), (ids // nb2) % nb1, ids % nb2], 1) + torch.tensor(bmin, device=dev)
    x0 = ijk.double() @ A.T
    h0 = x0 @ n_ - pn
    hE = h0 + s.U[:nn].double() @ n_
    # 칸 (입자 받침)
    N = int(s.N)
    Np = s.Np[:N].long(); fx = s.fx[:N]
    c = Np + (fx >= 1.0).long()
    OFF = torch.tensor([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.int64)
    nid = ((c[:, None, 0] + OFF[None, :, 0]) * nb1 + (c[:, None, 1] + OFF[None, :, 1])) * nb2 + (c[:, None, 2] + OFF[None, :, 2])
    okp = (nid >= 0).all(1) & (nid < nn).all(1)
    key = torch.where(okp, nid[:, 0], torch.full_like(nid[:, 0], -1))
    uk, inv = torch.unique(key, return_inverse=True)
    first = torch.full((len(uk),), N, device=dev, dtype=torch.int64).scatter_reduce(0, inv, torch.arange(N, device=dev), reduce="amin")
    sel = first[uk >= 0]
    nidc = nid[sel]                                                                          # [C, 8]
    a0 = h0[nidc]; aE = hE[nidc]
    hs = a0.max(1).values - a0.min(1).values; he = aE.max(1).values - aE.min(1).values
    r = he / hs.clamp_min(1e-12)
    hb = a0.min(1).values
    lay = torch.clamp(torch.floor(hb / dx + 1e-6), min=0).long()
    order = torch.argsort(a0, dim=1)
    wsg = torch.empty_like(a0)
    wsg.scatter_(1, order[:, :4], -1.0 / math.sqrt(8.0)); wsg.scatter_(1, order[:, 4:], 1.0 / math.sqrt(8.0))
    mn = s.mn[:nn].double()
    gz = float(sum(float(q) * float(p) for q, p in zip(s.g, n_.tolist())))
    names = list(INFO["names"])
    nbu = len(names)
    out = {}
    stat = {}
    for si, sl in ((0, "시작"), (1, "끝")):
        it2 = float(itf[si])
        en = W.en[si, :nn].double(); gn = W.gn[si, :nn].double()
        sk = W.sk[si, :nn].double(); pm = W.pm[si, :nn]; me = W.me[si, :nn].double(); w8 = W.w8[si, :nn]
        tn = torch.where(pm == 0, sk, torch.where(pm == 2, torch.zeros_like(sk), torch.ones_like(sk)))
        acc = W.acc[si, :nbu, :nn]
        T = [("관성", mn * it2 * en + tn * mn * gz), ("중력", -tn * mn * gz), ("질량하한", (me - mn) * it2 * en)]
        for b, nm in enumerate(names):
            T.append((nm, tn * acc[b]))
        tot = me * it2 * en + tn * acc.sum(0)
        T.append(("합", tot)); T.append(("풀이기잔차", gn))
        Fm = torch.stack([(g_[nidc] * wsg).sum(1) for _, g_ in T], 0)                        # [항, C]
        out[sl] = Fm.float().cpu().numpy()
        act = mn > 0
        dif = (tot - gn)[act].abs()
        stat[sl] = dict(it2=it2, code=int(itb[si]), chk_max=float(dif.max()) if bool(act.any()) else 0.0, chk_ref=float(gn[act].abs().max()) if bool(act.any()) else 0.0,
                        pm_other=int(((pm != 0) & (pm != 1) & (pm != 2) & act).sum()), nT=int(((tn != 1) & act).sum()), nw8=int(((w8 >= 0) & act).sum()),
                        sumabs={nm: float(g_[act].abs().sum()) for nm, g_ in T})
        if si == 1:
            ce = ((w8[nidc] >= 0).sum(1)).short(); ct = ((tn[nidc] != 1).sum(1)).short()
    terms = ["관성", "중력", "질량하한"] + names + ["합", "풀이기잔차"]
    cm = mn[nidc].sum(1)
    os.makedirs(os.path.join(OUT[0], "cf"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT[0], "cf", f"f{fr:03d}.npz"), terms=np.array(terms), r=r.float().cpu().numpy(), lay=lay.short().cpu().numpy(),
                        hb=hb.float().cpu().numpy(), hs=hs.float().cpu().numpy(), m=cm.float().cpu().numpy(), ne=ce.cpu().numpy(), nt=ct.cpu().numpy(),
                        F0=out["시작"], F1=out["끝"], dx=dx, gz=gz)
    o.update(ncell=int(len(sel)), npart_bad=int((~okp).sum()), dx=dx, stat=stat, r_med=float(r.median()), r_p10=float(torch.quantile(r.float(), 0.1)) if len(r) else None)
    return o


def install(out):
    from . import fem_an_rt as ART
    OUT[0] = out
    LOG[0] = open(os.path.join(out, "cf.jsonl"), "w")
    _wrap(ART.RTSolver)
    print("[절 CM111] cellforce 켬 -- 칸 세로 압축 모드 항별 일반화 힘 기록 (결과 불변)", flush=True)
    _dump()
