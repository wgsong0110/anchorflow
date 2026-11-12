"""절 CM121 묶음 trlm (조정자 11-12): 뉴턴 걸음의 전역 받아들임 판정 (신뢰 영역 비 판정) + 전역 Levenberg-Marquardt 이동 (모래 판).

배경 (절 CM120): 일관 접선 (+ 점별 Levenberg) 헤시안은 꼬리 수축을 줄이나 첫 반복에서 잔차를 키워 수렴 판이 발산 -- 지금 모래 판의 걸음은
  장벽 실현 가능 보폭 (2^-k) 만 보고 받아들임 (에너지 선탐색 없음; 있던 선탐색 에너지는 고정 공회전 탄성이라 모래 포텐셜과 다름).
정의 (새 문턱 없음, 상수는 신뢰 영역 교과서 값):
  - 판정 포텐셜 E(y) = 관성 ½ m/τ² |η|² (rt_lsyQ 와 같은 식, 중력은 ŷ 안) + Σ_q V_q W(ε̂_q),  W = μ Σ ε̂_i² + ½ λ (Σ ε̂_i)²,
    ε̂ = 반환 사상 뒤 Hencky 로그 변형 (ao_dpf: 탄성 log σ · 꼬리 tr/3 + s e_i · 꼭짓점 0) -- ∂W/∂ε̂ 가 힘 커널의 τ̂ 와 같은 식 (반환 사상 뒤 실제 포텐셜).
    장벽 에너지는 넣지 않음 (장벽은 실현 가능 보폭으로 따로 지킴).
  - 후보 c_i = c_b 2^-i (c_b = 장벽 실현 가능 보폭, i = 0..NC−1). 예측 감소 pred(c) = −gd (c − c²/2) (이차 모형 m(c) = c gd + ½ c² δᵀHδ, PCG 해에서 δᵀHδ = −gd),
    실제 감소 ared = E(0) − E(c). 비 ρ = ared/pred ≥ 1/4 인 가장 큰 c 를 받아들임 (Nocedal & Wright 알고리즘 4.1 의 η = 1/4).
  - 전역 Levenberg: 헤시안 곱·V 순환의 질량 항을 m (1 + λ_LM) 로 (= H + λ_LM M, 기울기는 그대로). 걸음 시작 λ_LM = 0, ν = 2.
    첫 후보가 ρ > 3/4 로 받아들여지면 λ_LM ← λ_LM/3, 줄인 후보거나 아무것도 안 되면 λ_LM ← (λ_LM > 0 ? λ_LM ν : 1), ν ← 2ν (Nielsen 1999).
    아무 후보도 안 되면 가장 작은 후보를 받아들임 (걸음은 늘 진행; 다음 반복이 큰 λ_LM 로 짧아짐).
훅: rt_grad 뒤 질량 이동 배열, rt_ap·층 0 rt_mg_node·rt_mg_pw1 의 질량 인자를 이동 배열로, rt_lsup 앞에 후보 에너지·판정, ctl(1) 에 λ_LM 초기화.
기록: 판정마다 (ρ_고른, 고른 i, λ_LM, ared, pred) -> trlm.npy (끝에서 한 번).
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, npick=0, err=None)
MARK = "// ---- 절 CM121 trlm"
HCAP = 400000
TR_SRC = r'''
// ---- 절 CM121 trlm (lib/anchorflow/fem_trlm.py)
#ifdef AF_ADAPT
#define TR_SKIP if (AF_SKIP) return;
#else
#define TR_SKIP
#endif
__global__ void tr_init(double* TR) { TR[0] = 0.0; TR[1] = 2.0; }
__global__ void tr_meff(const int* IP, const float* meff, const double* TR, float* ms) {
    TR_SKIP
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    ms[n] = meff[n]*(float)(1.0 + TR[0]);
}
__global__ void tr_fin(double* SC, int NC) {
    TR_SKIP
    double cb = SC[14], c = cb;
    for (int i=0;i<NC;i++) { SC[21+i] = SC[20] + c*SC[33] + c*c*SC[34]; c *= 0.5; }
}
// 후보 c = 0 (i = -1 -> SC[10]) 과 c_i = SC[14] 2^-i (SC[40+i]) 의 반환 사상 뒤 Hencky 에너지
__global__ void __launch_bounds__(128) tr_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, float cdp, double* SC, int NC) {
    TR_SKIP
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    float cbs = (float)SC[14];
    for (int i=-1;i<NC;i++) {
        float c = (i < 0) ? 0.f : ldexpf(cbs, -i);
        float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + c*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
        float U[9], S[3], V[9]; ssvdf(X, U, S, V);
        float sgn[3], sg[3], ep[3], ed[3], tr, rho, s;
        int st = ao_dpf(S, cdp, sgn, sg, ep, ed, tr, rho, s);
        double eh[3], tre = 0.0, s2 = 0.0;
        for (int k=0;k<3;k++) { eh[k] = (st == 2) ? 0.0 : ((st == 1) ? (double)(tr/3.f + s*ed[k]) : (double)ep[k]); tre += eh[k]; s2 += eh[k]*eh[k]; }
        double en = on ? (double)vq*((double)mu*s2 + 0.5*(double)lam*tre*tre) : 0.0;
        if (!isfinite(en)) en = 1e30;
        blocksum_add(en, (i < 0) ? &SC[10] : &SC[40+i]);
    }
}
__global__ void tr_pick(double* SC, int NC, double* TR, int* hp, double* H, int cap) {
    TR_SKIP
    double E0 = SC[20] + SC[10], gd = SC[9], cb = SC[14];
    int pick = -1; double rp = 0.0, ar = 0.0, pr = 0.0, c = cb, cl = cb;
    if (gd < 0.0) {
        for (int i=0;i<NC;i++) { double E = SC[21+i] + SC[40+i]; double pred = -gd*(c - 0.5*c*c); double ared = E0 - E;
            if (isfinite(E) && E < 1e29 && pred > 0.0 && ared >= 0.25*pred) { pick = i; rp = ared/pred; ar = ared; pr = pred; cl = c; break; }
            cl = c; c *= 0.5; }
    }
    if (pick == 0 && rp > 0.75) { TR[0] = TR[0]/3.0; TR[1] = 2.0; }
    else if (pick != 0) { TR[0] = (TR[0] > 0.0) ? TR[0]*TR[1] : 1.0; TR[1] *= 2.0; }
    SC[14] = cl;
    int k = hp[0]; hp[0] = k + 1;
    if (k < cap) { H[k*6] = rp; H[k*6+1] = pick; H[k*6+2] = TR[0]; H[k*6+3] = ar; H[k*6+4] = pr; H[k*6+5] = cb; }
}
'''


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def add_src(src):
    if MARK in src or "ao_dpf(" not in src or "rt_lsyQ" not in src:
        return src
    ST["nsrc"] += 1
    s = src.rstrip(); i = s.rfind("}")
    return s[:i] + MARK + "\n" + TR_SRC + "\n}\n"


def _dump():
    try:
        with open(os.path.join(ST["out"], "trinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


_SOL = []


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out)
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.RTSolver
    oa, octl, ok = R.alloc, R.ctl, R.k
    i32, f32 = np.int32, np.float32

    def on(s):
        return getattr(s, "trms", None) is not None and getattr(s, "_mat", None) == 2

    def alloc(s, need):
        r = oa(s, need)
        try:
            if getattr(s, "_mat", None) == 2:
                s.trms = torch.zeros(s.capN, device=s.dev, dtype=torch.float32)
                if getattr(s, "TR", None) is None:
                    s.TR = torch.zeros(8, device=s.dev, dtype=torch.float64)
                    s.trH = torch.zeros(HCAP * 6, device=s.dev, dtype=torch.float64)
                    s.trhp = torch.zeros(1, device=s.dev, dtype=torch.int32)
                    _SOL.append(s)
                ST["nalloc"] += 1
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if op == 1 and on(s):
            ok(s, "tr_init", 1, (P(s.TR),), 1)
        return r

    def k(s, name, n, args, b=128):
        if not on(s):
            return ok(s, name, n, args, b)
        pm_ = int(P(s.meff))
        if name == "rt_ap" and int(args[1]) == pm_:
            args = (args[0], P(s.trms)) + tuple(args[2:])
        elif name == "rt_mg_node" and int(args[0]) == 0 and int(args[3]) == pm_:
            args = tuple(args[:3]) + (P(s.trms),) + tuple(args[4:])
        elif name == "rt_mg_pw1" and int(args[0]) == 0 and int(args[2]) == pm_:
            args = tuple(args[:2]) + (P(s.trms),) + tuple(args[3:])
        elif name == "rt_lsup":
            s._tr_it2 = getattr(s, "_tr_it2", f32(1.0))
            NC = i32(max(int(getattr(s, "ls_n", 0) or 0), 6))
            nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
            ok(s, "rt_sc_clear", 1, (P(s.SC), i32(9), i32(11)), 1)
            ok(s, "rt_sc_clear", 1, (P(s.SC), i32(20), i32(35)), 1)
            ok(s, "rt_sc_clear", 1, (P(s.SC), i32(40), i32(40 + int(NC))), 1)
            ok(s, "rt_lsyQ", s.capN, (P(s.IP), P(s.meff), s._tr_it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), P(None)), 256)
            ok(s, "tr_fin", 1, (P(s.SC), NC), 1)
            capQ = s.Fq.numel() // 9
            ok(s, "tr_elem", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), *[f32(q) for q in nr],
                                    f32(s.mu), f32(s.lam), f32(s.cdp), P(s.SC), NC), 128)
            ok(s, "tr_pick", 1, (P(s.SC), NC, P(s.TR), P(s.trhp), P(s.trH), i32(HCAP)), 1)
            ST["npick"] += 1
        r = ok(s, name, n, args, b)
        if name == "rt_grad":
            s._tr_it2 = args[2]
            ok(s, "tr_meff", s.capN, (args[0], P(s.meff), P(s.TR), P(s.trms)), 256)
        return r

    R.alloc, R.ctl, R.k = alloc, ctl, k

    def fin():
        try:
            if _SOL:
                s = _SOL[-1]
                n_ = min(int(s.trhp.item()), HCAP)
                np.save(os.path.join(out, "trlm.npy"), s.trH[: n_ * 6].view(-1, 6).cpu().numpy())
                ST["nlog"] = n_
        except Exception as e:
            ST["err"] = "fin " + repr(e)
        _dump()
    import atexit
    atexit.register(fin)
    print("[절 CM121] trlm 켬 -- 반환 사상 뒤 포텐셜 비 판정 (ρ ≥ 1/4) + 전역 Levenberg (질량 이동)", flush=True)
    _dump()
