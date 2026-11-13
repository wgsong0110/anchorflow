"""절 CM127 고침 xjac (`--xjd 1`, 기본 0 = 모듈 안 부름): 확장 접은 뉴턴 선형계 Ã = Eᵀ A E 의 전처리를 Ã 블록 대각 야코비로.

측정 (절 CM127, lib/anchorflow/fem_linwhy.py): Ã 는 대칭·양정치 (u·Ãv 비대칭 1e-8, λmin > 0), MG V 순환 M 도 대칭·양정치. 그러나 V 순환은 확장을 모르는 A 의
  층 0 블록 대각 D 와 거친 층 (평균 F 로 다시 만든 탄성 연산자) 이라, 확장 주인 노드 줄 (Ã_jj 가 A_jj 의 수백~수만 배) 과 거친 보정이 Ã 보다 무른 방향에서
  M Ã 고윳값이 1e4 (수렴 판) ~ 1e8 (예산 판) 까지 넘침 -> PCG 5·20 회 에너지 노름 오차 0.97~1.0 (거의 안 풂). 같은 계에서 Ã 블록 대각 야코비는 λ(D̃⁻¹Ã) ≤ 30~60,
  에너지 오차 5 회 0.01 (수렴 판 첫 반복)·0.74~0.76 (나중 반복·예산 판). 근사 D̃ (아래 정의) 와 탐침으로 잰 정확 D̃ 의 PCG 이력이 같음.
정의 (새 문턱·클램프 없음): D̃_j = D_j + T_j (Σ_o c_oj² D_o) T_j  (D = rt_grad 의 m/τ² I + T B T 블록, 외부 노드 o 는 T = I, c_oj = 확장 가중) -- 절 CM94 wolfh13 층 0 정의와 같음.
  전처리 z = D̃⁻¹ r (내부 노드), 외부 노드 z = 0 (자유도 아님; PCG 가 p 를 펼침). 풀리는 방정식·수렴 해는 그대로, 전처리만 바뀜.
훅 (실행기 runpy 순간, 가장 바깥): k('rt_grad') 뒤 (확장 접기 다음) D̃⁻¹ 를 s.Dinv 자리에 · s.mg_vc(0, r, z) -> z = D̃⁻¹ r · mg_power 생략 (V 순환 안 씀).
확장이 꺼진 판 (w8b 없음, κ 1 씬) 은 아무것도 안 바꿈.
"""
import json
import os

import numpy as np

ST = dict(on=0, out=".", nsrc=0, ngrad=0, nvc=0, npw=0)
MARK = "// ---- 절 CM127 xjac"
XJ_SRC = r'''
// ---- 절 CM127 xjac: Ã 블록 대각 야코비 (lib/anchorflow/fem_xjac.py)
__global__ void xj_acc(const int* IP, const int* base, const float* w, const float* Dinv, float* S) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    float Do[9]; inv3(Dinv + n*9, Do);
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float w2 = ww*ww;
          for (int k=0;k<9;k++) atomicAdd(&S[q*9+k], w2*Do[k]); } } }
}
__global__ void xj_fin(const int* IP, const int* base, const float* sk, const int* pm, float n0, float n1, float n2, float* S, float* Dinv) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] >= 0) return;
    float s[9]; bool any = false;
    for (int k=0;k<9;k++) { s[k] = S[n*9+k]; S[n*9+k] = 0.f; any |= (s[k] != 0.f); }
    if (!any) return;
    float D[9]; inv3(Dinv + n*9, D);
    float T[9]; Tmat(sk[n], pm[n], n0, n1, n2, T);
    float TS[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += T[a*3+k]*s[k*3+b]; TS[a*3+b] = t; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += TS[a*3+k]*T[k*3+b]; D[a*3+b] += t; }
    float Di[9]; inv3(D, Di);
    for (int k=0;k<9;k++) Dinv[n*9+k] = Di[k];
}
__global__ void xj_apply(const int* IP, const int* base, const float* Dinv, const float* r, float* z) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (base[n] >= 0) { z[n*3] = 0.f; z[n*3+1] = 0.f; z[n*3+2] = 0.f; return; }
    const float* D = Dinv + n*9;
    float rr[3] = {r[n*3], r[n*3+1], r[n*3+2]};
    for (int a=0;a<3;a++) z[n*3+a] = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2];
}
'''


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def add_src(src):
    if MARK in src or "w8_expand" not in src:
        return src
    assert "inv3(" in src and "Tmat(" in src, "xjac: inv3·Tmat 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + MARK + "\n" + XJ_SRC + "\n}\n"


def _dump():
    try:
        with open(os.path.join(ST["out"], "xjinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out, mode=1):
    import torch
    from . import fem_an_rt as ART
    ST.update(on=int(mode), out=out)
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.RTSolver
    oa, ok, ovc, opw = R.alloc, R.k, R.mg_vc, R.mg_power

    def on(s):
        return getattr(s, "w8b", None) is not None and getattr(s, "_xjS", None) is not None

    def alloc(s, need):
        r = oa(s, need)
        if getattr(s, "w8b", None) is not None:
            s._xjS = torch.zeros(s.capN * 9, device=s.dev, dtype=torch.float32)
        else:
            s._xjS = None
        return r

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if name in ("rt_grad", "rt_grad_c") and on(s):                                 # args: IP mn it2 sk pm n0 n1 n2 eta f B Dinv r z p dl Tp SC r0
            ok(s, "xj_acc", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[11], P(s._xjS)), 256)
            ok(s, "xj_fin", s.capN, (P(s.IP), P(s.w8b), args[3], args[4], *args[5:8], P(s._xjS), args[11]), 256)
            ST["ngrad"] += 1
        return r

    def mg_vc(s, l, r, z):
        if l == 0 and on(s):
            ok(s, "xj_apply", s.capN, (P(s.IP), P(s.w8b), P(s.Dinv), P(r), P(z)), 256)
            ST["nvc"] += 1
            return
        return ovc(s, l, r, z)

    def mg_power(s, l):
        if on(s):
            ST["npw"] += 1
            return
        return opw(s, l)
    R.alloc, R.k, R.mg_vc, R.mg_power = alloc, k, mg_vc, mg_power
    import atexit
    atexit.register(_dump)
    print("[절 CM127] xjac 켬 -- 확장 판 전처리 = Ã 블록 대각 야코비 D̃ (D_j + T Σ c² D_o T), V 순환·거듭제곱법 안 씀", flush=True)
    _dump()
