"""절 CM40 수박 깨짐 모양 -- 진단 (결과 불변 읽기) + 고침 후보 (수박 옛 실행기 exe/fem_ao_rt_sim.py 만; 플래그를 안 주면 이 모듈을 안 부름).

진단 (--wsh_rec N): 프레임 끝마다 입자 (원래 순서) 의 몸 모양 (높이·퍼짐·질량중심), logJp 분포·손상 위치, 응력 (탄성 F_e 에서 압력·전단),
  N 프레임까지는 가우스 점 수준 (풀이가 실제로 힘을 계산하는 상태) 통계: 프레임 시작 모으기 (팔분 입자 부피 가중 평균 = aq_p2g_s 와 같은 식,
  입자 소속은 이 프레임 fx·pcd·cidx) 가 손상 입자를 얼마나 희석하는지, 커밋 뒤 가우스 점 응력이 손상 점에서 실제로 줄었는지, 보간 교체 (fpi_p) 가
  손상 부피를 얼마나 바꾸는지. 같은 입자 통계 함수를 정답 (exe/wmshape_gt.py) 에도 씀.
  --wsh_sv 프레임들: 그 프레임 입자 위치·logJp 를 float16 npz 로 (비결정 흩어짐·정답 겹침·바닥 미끄럼 계산용).

후보 (--wsh_pl 1, 정의 doc 절 CM40.2): 입자 logJp 를 '프레임 끝 가우스 점 logJp 보간으로 교체' (fpi_p) 대신 '프레임 시작 입자 값 + 가우스 점 증분의 보간'
  ljp ← lj_start + I(Sq_end) − I(Sq_start)  (I = fpi_p 와 같은 칸 안 삼선형 가중, Sq_start = 그 프레임 aq_gauss_s 직후 값) -- 입자 수준 손상 이질성을 매 프레임
  평균→보간으로 지우지 않음 (F 는 fpi_p 그대로). aq_p2g_s 앞 · aq_gauss_s 뒤 · fpi_p 뒤에 커널 셋 (AORTSolver.k 감쌈); 끄면 소스·커널 열 그대로.
"""
import json
import math
import os

import numpy as np
import torch

LJ0 = -0.04
FLOOR = 0.02
VOX = (2.0 / 300.0, 0.01, 0.02)                                                     # 정답 격자 dx · 셀 3 가우스 점 간격 · 셀 3 칸 (m)
HB = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 10.0)                                          # 손상 위치: 바닥 위 높이 / 처음 높이
RB = (0.0, 0.4, 0.7, 0.9, 1.1, 10.0)                                               # 손상 위치: 처음 질량중심 축에서 거리 / 처음 반경


def stress(F, mu, lam):
    """F (n,3,3) -> J, 정답 식 NHB 키르히호프 압력 p (압축 +)·전단 q = √(3/2)|dev τ|, 우리 식 SNH 같은 둘 (주값 아닌 행렬식)."""
    F = F.double()
    J = torch.linalg.det(F)
    B = F @ F.transpose(1, 2)
    trB = B.diagonal(dim1=1, dim2=2).sum(-1)
    dB = B - trB[:, None, None] / 3.0 * torch.eye(3, device=F.device, dtype=F.dtype)
    nd = dB.flatten(1).norm(dim=1)
    kap, ls = 2.0 * mu / 3.0 + lam, lam + mu
    Jp = J.clamp_min(1e-12)
    return dict(J=J, p_nhb=-0.5 * kap * (J * J - 1.0), q_nhb=math.sqrt(1.5) * mu * Jp ** (-2.0 / 3.0) * nd,
                p_snh=-(mu * trB / 3.0 + (ls * (J - 1.0) - mu) * J), q_snh=math.sqrt(1.5) * mu * nd)


def _q(t, qs):
    if t.numel() == 0:
        return [None] * len(qs)
    t = t.float()
    if t.numel() > 8_000_000:
        t = t[torch.randperm(t.numel(), device=t.device)[:8_000_000]]
    return [float(v) for v in torch.quantile(t, torch.tensor(qs, device=t.device))]


def _hist(v, edges):
    if v.numel() == 0:
        return [0.0] * (len(edges) - 1)
    e = torch.tensor(edges, device=v.device, dtype=v.dtype)
    i = torch.bucketize(v.clamp(min=edges[0], max=edges[-1] - 1e-9), e, right=True) - 1
    c = torch.bincount(i.clamp(0, len(edges) - 2), minlength=len(edges) - 1).double()
    return [float(q) for q in (c / c.sum())]


def _vox(x, lj, dm, s):
    """크기 s 의 상자로 묶을 때: 손상 입자 중 상자 평균 logJp > 0 인 몫 (평균 모으기로도 손상으로 보이는 몫), 상자 손상 몫 ≥ 0.5 인 몫 (국소 집중)."""
    if int(dm.sum()) == 0:
        return [None, None]
    k = torch.floor(x / s).long()
    k = k - k.min(0).values
    mx = k.max(0).values + 1
    key = (k[:, 0] * mx[1] + k[:, 1]) * mx[2] + k[:, 2]
    u, inv = torch.unique(key, return_inverse=True)
    n = torch.bincount(inv, minlength=u.numel()).double()
    sl = torch.zeros(u.numel(), device=x.device, dtype=torch.float64).index_add_(0, inv, lj.double())
    sd = torch.bincount(inv, weights=dm.double(), minlength=u.numel())
    mb, fb = sl / n, sd / n
    return [float((mb[inv[dm]] > 0).double().mean()), float((fb[inv[dm]] >= 0.5).double().mean())]


def pstats(x, F, lj, lj_prev, mu, lam, xi, ref, floor=FLOOR):
    """입자 통계 (정답·우리 같은 함수). ref = dict(c0=[x,y] 처음 질량중심, H0 처음 높이, R0 처음 반경). 입자 질량은 같다고 봄 (같은 입자 집합)."""
    ok = torch.isfinite(x).all(-1) & torch.isfinite(F.flatten(1)).all(-1)
    if lj is not None:
        ok &= torch.isfinite(lj)
    x, F = x[ok].double(), F[ok]
    h = x[:, 2] - floor
    c0 = torch.tensor(ref["c0"], device=x.device, dtype=torch.float64)
    r = (x[:, :2] - c0).norm(dim=1)
    hq = _q(h, [0.005, 0.5, 0.995])
    foot = r[h < 0.03]
    xyq = [_q(x[:, a], [0.005, 0.995]) for a in (0, 1)]
    o = dict(n=int(ok.sum()), zc=float(x[:, 2].mean()), H=hq[2], hmin=hq[0], hmed=hq[1], r95=_q(r, [0.95])[0], r99=_q(r, [0.99])[0],
             foot95=(_q(foot, [0.95])[0] if foot.numel() > 100 else None), nfoot=int(foot.numel()),
             wx=xyq[0][1] - xyq[0][0], wy=xyq[1][1] - xyq[1][0], cxy=[float(q) for q in x[:, :2].mean(0)])
    st = stress(F.reshape(-1, 3, 3), mu, lam)
    J = st["J"]
    o["Jq"] = _q(J, [0.01, 0.1, 0.5, 0.9, 0.99])
    o["J090"], o["J111"] = float((J < 0.9).double().mean()), float((J > 1.11).double().mean())
    if lj is None:
        return o
    l = lj[ok].double()
    Jt = J * torch.exp(l - LJ0)                                                      # 전체 부피 비 = 탄성 J x 소성 Jp (처음 logJp −0.04 기준)
    o["Jtq"] = _q(Jt, [0.01, 0.1, 0.5, 0.9, 0.99])
    o["ljq"] = _q(l, [0.01, 0.1, 0.5, 0.9, 0.99])
    dm = l > 0
    o.update(d0=float(dm.double().mean()), d04=float((l > 0.4).double().mean()), c02=float((l < -0.2).double().mean()))
    H0, R0 = float(ref["H0"]), float(ref["R0"])
    o["dh"] = _hist(h[dm] / H0, HB)
    o["dr"] = _hist(r[dm] / R0, RB)
    if lj_prev is not None:
        nw = dm & ~(lj_prev[ok].double() > 0)
        o["nnew"] = int(nw.sum())
        o["nh"] = _hist(h[nw] / H0, HB)
        o["nr"] = _hist(r[nw] / R0, RB)
    o["vox"] = [_vox(x, l, dm, s) for s in VOX]
    kap = 2.0 * mu / 3.0 + lam
    p0 = kap * (1e-5 + torch.sinh(xi * torch.clamp(-l, min=0.0)))
    for nm in ("p_nhb", "q_nhb", "p_snh", "q_snh"):
        v = st[nm]
        o[nm] = dict(d=_q(v[dm], [0.1, 0.5, 0.9]), u=_q(v[~dm], [0.1, 0.5, 0.9]))
    o["p0"] = dict(d=_q(p0[dm], [0.5]), u=_q(p0[~dm], [0.5]))
    return o


def ref_of(x0, floor=FLOOR):
    x0 = x0[torch.isfinite(x0).all(-1)].double()
    c0 = x0[:, :2].mean(0)
    return dict(c0=[float(q) for q in c0], H0=float(torch.quantile((x0[:, 2] - floor).float(), 0.995)),
                R0=float(torch.quantile((x0[:, :2] - c0).norm(dim=1).float(), 0.99)))


class Rec:
    """실행기 안 진단 (프레임 끝, 읽기만)."""

    def __init__(s, S, out, mu, lam, xi, x0, gmax, sv):
        s.S, s.out, s.mu, s.lam, s.xi, s.gmax = S, out, float(mu), float(lam), float(xi), int(gmax)
        s.sv = set(int(q) for q in sv.split(",") if q) if sv else set()
        s.ref = ref_of(x0.to(S.x.device))
        s.lprev = None                                                               # 원래 순서, 앞 프레임 끝 입자 logJp
        s.gprev = None
        os.makedirs(os.path.join(out, "wsh"), exist_ok=True)
        json.dump(s.ref, open(os.path.join(out, "wsh", "ref.json"), "w"))

    def _orig(s, t):
        o = torch.empty_like(t)
        o[s.S.perm] = t
        return o

    def gauss(s, lprev_int):
        S = s.S
        nq = int(S.IP[12])
        Sq = S.Sq.view(-1, 4)[:nq]
        Vq = S.Vq[:nq].double()
        Fq = (S.Fe[:nq, 18:27] if getattr(S, "Fe", None) is not None else S.Fq[:nq]).reshape(-1, 3, 3)   # 프레임 끝 탄성 F_e = (I + ∇U) F_q (fpi_q 의 X; F_q 자체는 커밋 뒤 프레임 시작 기준)
        fx = S.fx.reshape(-1, 3)
        c = S.cidx[S.pcd.long()].long()
        ok = (c >= 0) & torch.isfinite(lprev_int)
        oc = (fx[:, 0] >= 1.0).long() * 4 + (fx[:, 1] >= 1.0).long() * 2 + (fx[:, 2] >= 1.0).long()
        q = (c * 8 + oc)[ok]
        w = S.vol[ok].double()
        lp = lprev_int[ok].double()
        sw = torch.zeros(nq, device=w.device, dtype=torch.float64).index_add_(0, q, w)
        sl = torch.zeros(nq, device=w.device, dtype=torch.float64).index_add_(0, q, w * lp)
        has = sw > 0
        S0 = torch.where(has, sl / sw.clamp_min(1e-30), torch.zeros_like(sl))
        kap = 2.0 * s.mu / 3.0 + s.lam
        p0p = kap * (1e-5 + torch.sinh(s.xi * torch.clamp(-lp, min=0.0)))
        sp0 = torch.zeros(nq, device=w.device, dtype=torch.float64).index_add_(0, q, w * p0p)
        mp0 = sp0 / sw.clamp_min(1e-30)                                              # 입자 응집력 평균 (= --ljq p0 가 쓰는 값)
        p0m = kap * (1e-5 + torch.sinh(s.xi * torch.clamp(-S0, min=0.0)))            # 평균 logJp 의 응집력 (지금 풀이가 쓰는 값)
        dmp = lp > 0
        nd = torch.zeros(nq, device=w.device, dtype=torch.float64).index_add_(0, q, w * dmp.double())
        anyd = has & (nd > 0)
        Se = Sq[:, 1].double()
        act = has & (Vq > 0)
        st = stress(Fq, s.mu, s.lam)
        de = act & (Se > 0)
        ue = act & ~(Se > 0)
        dS = Se - S0
        o = dict(nq=int(act.sum()),
                 dil=float((S0[q[dmp]] <= 0).double().mean()) if int(dmp.sum()) else None,           # 프레임 시작 손상 입자 중 그 가우스 점 평균이 손상 아님 (힘에 연화 안 보임) 몫
                 dil_w=float((w[dmp] * (S0[q[dmp]] <= 0).double()).sum() / w[dmp].sum()) if int(dmp.sum()) else None,
                 p0r=_q((p0m / mp0.clamp_min(1e-30))[anyd], [0.1, 0.5, 0.9]),                          # 손상 입자를 품은 점: 평균 logJp 응집력 / 입자 응집력 평균
                 fd=float((nd[anyd] / sw[anyd]).median()) if int(anyd.sum()) else None,                 # 그 점들의 손상 입자 부피 몫 중앙
                 Vd0=float(Vq[act & (S0 > 0)].sum() / Vq[act].sum()), Vde=float(Vq[de].sum() / Vq[act].sum()),
                 rm_up=float((dS[act] > 1e-4).double().mean()), rm_dn=float((dS[act] < -1e-4).double().mean()),
                 Smax0=float(S0[act].max()), Smaxe=float(Se[act].max()),
                 qd=_q(st["q_snh"][de], [0.1, 0.5, 0.9]), qu=_q(st["q_snh"][ue], [0.1, 0.5, 0.9]),
                 pd=_q(st["p_snh"][de], [0.1, 0.5, 0.9]), pu=_q(st["p_snh"][ue], [0.1, 0.5, 0.9]),
                 Jd=_q(st["J"][de], [0.1, 0.5, 0.9]), Ju=_q(st["J"][ue], [0.1, 0.5, 0.9]))
        if s.gprev is not None:                                                      # 앞 프레임 끝 (커밋 뒤) 손상 점 부피 몫 -> 보간 교체·다시 모으기 뒤 (이 프레임 시작)
            o["Vde_prev"] = s.gprev
        s.gprev = o["Vde"]
        return o

    def frame(s, fi):
        S = s.S
        t = fi + 1
        if int(S.IP[13]) != 0:
            return None
        xo = S.x_orig()
        Fo = s._orig(S.F.reshape(S.F.shape[0], -1)).reshape(-1, 3, 3)
        lo = s._orig(S.ljp)
        o = pstats(xo, Fo, lo, s.lprev, s.mu, s.lam, s.xi, s.ref)
        if t <= s.gmax and s.lprev is not None:
            try:
                o["g"] = s.gauss(s.lprev[S.perm])
            except Exception as ex:                                                  # 진단 실패는 결과에 영향 없음
                o["g"] = dict(err=repr(ex))
        if t in s.sv:
            np.savez_compressed(os.path.join(s.out, "wsh", f"xs_{t:03d}.npz"), x=xo.cpu().numpy().astype(np.float16), lj=lo.cpu().numpy().astype(np.float16))
        s.lprev = lo.clone()
        return o


# ---------------- 후보 --wsh_pl ----------------
PL_SRC = r'''
// ---- 절 CM40 --wsh_pl: 입자 logJp = 프레임 시작 값 + 가우스 점 증분 보간 (fpi_p 와 같은 가중) ----
__global__ void wsh_lo(int N, const int* IP, const float* ljp, float* lo) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    lo[p] = ljp[p];
}
__global__ void wsh_s0(const int* IP, const float* Sq, float* S0) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    S0[q] = Sq[(size_t)q*4+1];
}
__global__ void wsh_pl(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* S0, const float* lo, float* ljp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    if (c < 0) { ljp[p] = lo[p]; return; }
    const float GG = 0.28867513459481287f;
    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);
    float s0 = 0.f;
    for (int oc=0; oc<8; oc++) {
        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);
        if (!(w > 0.f)) continue;
        s0 += w*S0[(size_t)(c*8 + oc)];
    }
    float v = lo[p] + (ljp[p] - s0);
    if (isfinite(v)) ljp[p] = v;
}
'''


def install_pl():
    from . import fem_ao_rt as AORT
    if getattr(AORT, "_wsh_pl", False):
        return
    AORT._wsh_pl = True
    o_src = AORT._src

    def _src(*a, **kw):
        r = o_src(*a, **kw).rstrip()
        i = r.rfind("}")
        return r[:i] + PL_SRC + "\n}\n"
    AORT._src = _src
    o_alloc = AORT.AORTSolver.alloc

    def alloc(s, need):
        o_alloc(s, need)
        s.wshS0 = torch.zeros(s.capQc * 8, device=s.dev, dtype=torch.float32)
        s.wshLO = torch.zeros(s.N, device=s.dev, dtype=torch.float32)
    AORT.AORTSolver.alloc = alloc
    ok = AORT.AORTSolver.k
    P = AORT.ART.P
    i32 = np.int32

    def k(s, name, n, args, b=128):
        if name == "aq_p2g_s":
            ok(s, "wsh_lo", s.N, (i32(s.N), P(s.IP), P(s.ljp), P(s.wshLO)), 128)
        r = ok(s, name, n, args, b)
        if name == "aq_gauss_s":
            ok(s, "wsh_s0", s.capQc * 8, (P(s.IP), P(s.Sq), P(s.wshS0)), 128)
        elif name == "fpi_p":
            ok(s, "wsh_pl", s.N, (i32(s.N), P(s.IP), P(s.fx), P(s.pcd), P(s.cidx), P(s.wshS0), P(s.wshLO), P(s.ljp)), 128)
        return r
    AORT.AORTSolver.k = k
