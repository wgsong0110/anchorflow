"""절 CM67 묶음 mfix (11-08 위임): 저질량 '고정 노드' 정의 고치기 -- (1) 연속 고정 (관성 덧질량) · (2) 장벽 표본점이 기대는 노드는 고정에서 뺌.

원인 (절 CM64): 지금 정의는 노드 질량 m_n < m̄ (= mlow × 첫 프레임 노드 질량 중앙, mlow 1e-3) 이면 관성 예측 ŷ 에 고정 (pm 2, 미지수 아님), 위면 완전 자유.
  튐 = 문턱 바로 위 (질량 비 1e-3~5e-2) 의 풀린 노드가 거의 빈 칸에서 한 걸음에 0.5~3 m (87 %).
  가라앉음 = 이탈 입자의 27 노드 무게 0.29~0.69 가 고정 노드 -> 고정 노드는 ŷ (중력 낙하) 로 바닥을 무시, 장벽은 미지수만 밂 -> A2 가 기준을 올리고 R2 가 이어 받는 톱니.

정의 (새 문턱·클램프 없음; m̄ 은 기존 mlow 그대로 스케일로만):
  (1) 연속 고정 (AF_MF1=1): 고정을 없애고, 질량 있는 모든 노드 (m_n > 0) 의 관성 항 질량을 m_n -> m_n + m̄ 로 (풀이 에너지·기울기·헤시안·선탐색이 같은 s.mn 을 읽음).
      관성 에너지 (m_n + m̄)/(2τ²) |y_n − ŷ_n|² -- m̄ 은 ŷ (관성 예측) 에 묶는 덧질량. 노드 하나에 힘 f 가 걸리면 걸음 변위 = τ² f / (m_n + m̄) = w · (자유 해),
      w = m_n / (m_n + m̄): m_n ≪ m̄ 에서 w ≈ m_n/m̄ (질량 비에 비례하는 풀이 몫), m_n = m̄ 에서 0.5, m_n ≫ m̄ 에서 1. 고정 (w = 0) 은 m̄ -> ∞ 극한, 지금의 0 -> 1 불연속이 없어짐.
      에너지 꼴이라 장벽 (무한 장벽 + 실현 가능 보폭) 이 그 노드도 밂 -- 풀이 뒤 섞기 (y ← ŷ + w (y* − ŷ)) 는 장벽 보장을 깨므로 안 씀.
      질량 0 노드 (빈 노드) 는 지금처럼 고정. 속도·가속도 (rt_vn·ao_adiv) 와 구속 몫 (ao_pbc_k) 은 원래 질량 (덧질량은 풀이 관성 항에만).
  (2) 장벽 노드 (AF_MF2=1): 걸음마다 장벽 표본점 q (팔분 칸 입자 질량 중심, 칸 질량 > 0) 중 걸음 시작 위치 또는 관성 예측 위치 (띠 고착 판정 wb_floor_q·rt_floor_q 와 같은 자리)
      에서 d < d̂ (장벽 받침 안 = 장벽 힘이 걸릴 수 있는 점) 인 점의 27 노드 (W̄ 무게 > 0) 를 표시해, 저질량 고정에서 뺌 (질량 그대로 풀이에 넣음). d̂ 는 장벽 자체 받침.
      (1) 과 함께면 표시 노드는 덧질량도 안 받음 (원래 질량으로 풀이) -- '고정 몫' 에서 완전히 뺌.
  둘 다 노드 위치를 바닥에 묶지 않음 (노드 바닥 아님): (2) 는 그 노드를 미지수로 두어 장벽이 표본점을 통해 밀게 할 뿐.
  mthr = 0 인 씬 (레고·ficus: --mlow 없음) 은 아무것도 안 함 (커널 하나도 안 띄움 -> 결과 그대로).

진단 (결과 불변, 프레임 끝 그래프 밖): mf.jsonl -- 원래 질량으로 가벼운 노드 수, 그중 고정 (pm 2) 수, 장벽 표시 노드 수, 표시인데 가벼운 노드 수·그중 고정 수,
  덧질량 받은 노드 수, 검사 (s0k) 덧질량 차 최대 오차.
"""
import json
import os

import numpy as np
import torch

C1 = int(os.environ.get("AF_MF1", "0"))
C2 = int(os.environ.get("AF_MF2", "0"))
ST = dict(on=False, log=None, nset=0, nmark=0, said=False)
FLOORN = ("ao_floor_n", "mt_floor_n")
PBCK = ("ao_pbc_k", "mt_pbc_kphi")

MF_SRC = r'''
// ---- 절 CM67 mfix: 저질량 고정 노드 정의 (연속 덧질량 · 장벽 노드 빼기) ----
__global__ void mf_save(const int* IP, const float* mn, float* mnb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    mnb[n] = mn[n];
}
// c1: 풀이 관성 질량 mn = mnb + add (mnb > 0, 표시 아닌 노드); mnx = 고정 판정 (floor_n 의 'mn < mthr') 이 볼 값: 표시 노드 (m > 0) 는 mthr (고정 아님), 아니면 mn
__global__ void mf_set(const int* IP, const float* mnb, float* mn, float* mnx, const unsigned char* bf, float add, float mthr, int c1, int c2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mnb[n];
    int b = (c2 && bf[n]) ? 1 : 0;
    float mm = (c1 && m > 0.f && !b) ? (m + add) : m;
    if (c1) mn[n] = mm;
    mnx[n] = (b && m > 0.f) ? mthr : mm;
}
__global__ void mf_zero(const int* IP, unsigned char* bf) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bf[n] = 0;
}
'''
MF_BAR_SRC = r'''
#ifdef E3W
#define MFW(q_, oc_, o_) E3W(q_, o_)
#else
#define MFW(q_, oc_, o_) BW[(oc_)*27+(o_)]
#endif
// 장벽 표본점 (x̄_q + Σ W̄ ...) 이 걸음 시작 (U) 또는 관성 예측 (U + thinv (y − U), y = 예측) 에서 d < d̂ 이면 그 27 노드 표시
__global__ void mf_mark(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, unsigned char* bf, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3]; if (!(bm > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xs[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
    float xp[3] = {xs[0], xs[1], xs[2]};
    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];
        for (int a=0;a<3;a++) { xs[a] += w*U[id*3+a]; xp[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); } }
    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;
    if (!(ds < dh) && !(dp < dh)) return;
    atomicAdd(&cnt[0], 1);
    for (int o=0;o<27;o++) if (MFW(q, oc, o) > 0.f) bf[ids[o]] = 1;
}
'''


def _append(src):
    if "mf_save(" in src:
        return src
    add = MF_SRC + (MF_BAR_SRC if "__device__ float* AF_BX;" in src else "")
    i = src.rstrip().rfind("}")
    return src[:i] + add + src[i:]


def install(out):
    """실행기 runpy 순간 (다른 훅 뒤, 가장 바깥) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    ST.update(on=True)
    ST["log"] = open(os.path.join(out, "mf.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))
    P = ART.P
    f32, i32 = np.float32, np.int32

    def on(s):
        return float(getattr(s, "mthr", 0.0) or 0.0) > 0.0 and (C1 or C2)

    def _alloc(s):
        cap = int(s.capN)
        if getattr(s, "mfb", None) is None or s.mfb.numel() < cap:
            s.mfb = torch.zeros(cap, device=s.dev, dtype=torch.float32)
            s.mfx = torch.zeros(cap, device=s.dev, dtype=torch.float32)
            s.mfbf = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        if getattr(s, "mfcnt", None) is None:
            s.mfcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)

    def mark(s, ok, y, U, thv, nr, pn):
        dh = None
        wb = getattr(s, "wbar", None)
        if isinstance(wb, dict) and "DH" in wb:
            dh = float(wb["DH"])
        elif getattr(s, "bar", False) and s.floor:
            dh = float(s.floor["DH"])
        ok(s, "mf_zero", int(s.capN), (P(s.IP), P(s.mfbf)), 256)
        if dh is None or not C2:
            return
        capQ = int(s.capQc) * 8
        ok(s, "mf_mark", capQ, (P(s.IP), P(s.cb), P(s.BW), y, U, thv, *nr, pn, f32(dh), P(s.mfbf), P(s.mfcnt)), 128)
        ST["nmark"] += 1

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if getattr(s, "_mfin", False) or name.startswith("mf_") or not on(s):
                return _ok(s, name, n, args, b)
            s._mfin = True
            try:
                _alloc(s)
                if name in FLOORN:                                                          # 앞: 덧질량·고정 판정 값 (걸음마다 같은 값 -> 쪼갠 걸음에서도 한 번만 더한 셈)
                    ok_ = _ok
                    ok_(s, "mf_set", int(s.capN), (P(s.IP), P(s.mfb), P(s.mn), P(s.mfx), P(s.mfbf), f32(s.mthr), f32(s.mthr), i32(C1), i32(C2)), 256)
                    ST["nset"] += 1
                    a = list(args)
                    pm_ = P(s.mn)
                    ix = [i for i in range(len(a) - 1) if isinstance(a[i], np.uint64) and a[i] == pm_ and isinstance(a[i + 1], np.float32)]
                    assert len(ix) == 1, ("mfix: floor_n 질량 인자", name, ix)
                    a[ix[0]] = P(s.mfx)
                    args = tuple(a)
                elif name in PBCK and C1:                                                   # 구속 몫은 원래 질량
                    a = list(args)
                    assert isinstance(a[1], np.uint64) and a[1] == P(s.mn), ("mfix: pbc 질량 인자", name)
                    a[1] = P(s.mfb)
                    args = tuple(a)
            finally:
                s._mfin = False
            r = _ok(s, name, n, args, b)
            s._mfin = True
            try:
                if name == "rt_vn":                                                         # 뒤: P2G 가 끝난 원래 질량 사본
                    _ok(s, "mf_save", int(s.capN), (P(s.IP), P(s.mn), P(s.mfb)), 256)
                elif name == "wb_floor_q":                                                  # 옛 실행기 장벽 판 (IP, cb, BW, mq, y, U, thv, n0, n1, n2, PN, DH, an)
                    mark(s, _ok, args[4], args[5], args[6], args[7:10], args[10])
                elif name == "rt_floor_q" and float(args[15]) > 0:                          # 새 실행기 (IP, cb, BW, xq0, mq, y, U, thinv, p0..2, n0..2, PN, DXG, an)
                    mark(s, _ok, args[5], args[6], args[7], args[11:14], args[14])
            finally:
                s._mfin = False
            return r
        cls.k = k

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)

    dep = dict(d=0)

    def outer(fn):
        def w(s, t, h, *a, **kw):
            dep["d"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep["d"] -= 1
            if dep["d"] == 0:
                try:
                    post(s, int(round(t / h)) + 1)
                except Exception:
                    import traceback
                    with open(os.path.join(out, "mf_err.txt"), "a") as f_:
                        f_.write(traceback.format_exc()[-3000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    print(f"[절 CM67] mfix 켬 -- (1) 연속 고정 (관성 덧질량 m̄ = mthr) {C1} · (2) 장벽 표본점 27 노드 고정 빼기 {C2}", flush=True)


@torch.no_grad()
def post(s, fi):
    if getattr(s, "mfb", None) is None:
        if not ST["said"]:
            ST["said"] = True
            ST["log"].write(json.dumps(dict(f=fi, off=True, mthr=float(getattr(s, "mthr", 0.0) or 0.0))) + "\n")
            ST["log"].flush()
        return
    torch.cuda.synchronize()
    nn = int(s.IP[9].item())
    mb = s.mfb[:nn]
    mn = s.mn[:nn]
    pm = s.pm[:nn]
    bf = s.mfbf[:nn].bool()
    mt = float(s.mthr)
    lt = (mb > 0) & (mb < mt)
    fx = getattr(s, "fixd", None)
    fxd = fx[:nn].bool() if fx is not None and fx.numel() >= nn else torch.zeros_like(bf)
    med = float(mb[mb > 0].median()) if bool((mb > 0).any()) else 0.0
    rec = dict(f=fi, nn=nn, nact=int((mb > 0).sum()), nlight=int(lt.sum()), nlight_fix=int((lt & (pm == 2)).sum()),
               nflag=int(bf.sum()), nflag_light=int((bf & lt).sum()), nflag_light_fix=int((bf & lt & (pm == 2)).sum()),
               naug=int((mn != mb).sum()), nfix_act=int(((mb > 0) & (pm == 2) & ~fxd).sum()), mthr=mt, mthr_med=(mt / med if med > 0 else None),
               nset=ST["nset"], nmark=ST["nmark"])
    if C1:
        sel = (mb > 0) & ~(bf if C2 else torch.zeros_like(bf))
        rec["aug_err"] = float((mn[sel].double() - mb[sel].double() - mt).abs().max()) if bool(sel.any()) else None
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()
