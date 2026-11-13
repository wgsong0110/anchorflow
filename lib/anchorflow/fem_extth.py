"""절 CM133 묶음 extth (조정자 11-13, 사용자 wolf 완전 위임): 확장 B-스플라인 T1E2 (절 CM81 fem_wolfh8) 의 외부 노드를 위상 판정 (묶음 / 자유) 대신
노드 채움 비 θ 로 원래 자유도와 확장 자유도를 연속으로 섞음 (절 CM131.5 (나)) -- 새 플래그 --xth, 기본 0 (기본 판 불변).

배경 (절 CM131·CM132·CM133.1): 예산 안 H9 정체 = 확장 주인 노드에 접힌 잔차. 접혀 오는 외부 노드는 무겁고 꽉 찬 노드 (바닥 위 얇게 퍼진 층·몸 안 압축 층) --
  T1 (26 이웃 칸 모두 점유) 이 위아래 한 칸 (3.5 cm) 을 요구해 얇은 층에는 내부 칸이 없음 (CM133.1: 두께 < 6 cm 기둥의 98~100 % 가 내부 칸 0).
정의 (새 문턱·클램프·벌점·노드 바닥 없음):
  변수 바꿈 y = E u (뉴턴 걸음 증분, wolfh8 의 아핀 확장 자리 그대로): 내부·고립 노드 y_j = u_j (T 그대로),
    묶인 외부 노드 o (w8_class 가 base ≥ 0 으로 고른 노드, 묶는 블록·라그랑주 계수 c_oj 는 wolfh8 그대로):
      Δy_o = θ_o Δu_o + (1 − θ_o) Σ_j c_oj (T Δu)_j.
    θ_o = 1 이면 원래 자유도 (확장 없음), θ_o = 0 이면 지금 확장 (Δy_o = Σ c Δy_j). 0 < θ < 1 이면 E 는 가역 -- 수렴 해는 확장 끈 문제와 같고,
    예산 (뉴턴 3 평가 · PCG 5) 안에서 외부 노드가 자기 잔차로 움직이는 몫이 θ (y 공간 전처리 대각 그대로라 자기 걸음 ∝ θ²), 나머지는 주인 블록을 따라감.
  θ (--xth 1, 기본 정의): 노드 기저 적분 중 점유 칸 안 몫 θ_o = ∫_{Ω_h} N_o / ∫ N_o, Ω_h = 점유 매듭 칸 합집합 (T1 판정·가우스 적분과 같은 칸 집합 cb).
    2 차 B-스플라인 축 적분의 칸 몫 (1/6, 2/3, 1/6) 의 텐서 곱을 점유 칸마다 그 27 노드에 더함 -> 0 ≤ θ ≤ 1 이 정의에서 나옴 (클램프 아님).
    지시 식 m_o / (ρ̄ ∫N_o) 에서 '그 칸의 재질 밀도 몫' 을 점유 (0/1) 로 둔 것: 측정 (CM133.1) 상 wolf 는 칸당 입자 약 8 개라 m_o/(ρ_재질 h³) 가 내부 노드에서도
    중앙 0.4, 바닥 쌓인 층에서 4~9 (입자 몰림) -- 질량 비는 '채움' 이 아니라 입자 표본 밀도를 잼.
  θ (--xth 2, 지시 식 그대로): θ_o = m_o / (ρ̄ h³), ρ̄ = 그 걸음 전체 질량 / 점유 칸 부피 (= Σ m / (IP[11] h³), 점유 영역 평균 밀도). θ > 1 (몰린 노드) 도 그대로 씀 (가역 섞기, 클램프 없음).
구현: wolfh8 의 접기·펼치기 자리 그대로 이름만 바꿔 부름 (fem_wolfh8.install 을 감싸 R.k 를 안쪽에서 감쌈):
  - 걸음 시작 w8_class 뒤: θ 계산 (xt_thz · xt_thc (xth 1) / xt_msum · xt_thm (xth 2)).
  - w8_fold -> xt_fold: r_o <- θ_o r_o, 주인 r_j += T_j (1 − θ_o) c_oj r_o (= Eᵀ r).
  - w8_expand -> xt_expand: (u 공간) p_o 를 pu 에 두고 p_o = Tp_o = θ_o pu_o + (1 − θ_o) Σ c_oj Tp_j (= E p; 외부 노드 T 는 w8_free 가 항등으로 둠).
  - rt_cg2 앞: xt_restore p_o <- pu_o (PCG 갱신 p = z + β p 는 u 공간; dl += α p · rt_ap 의 M p 는 펼친 y 공간 그대로).
  MG 전처리는 접은 r (u 공간, 외부 노드 몫 θ r_o) 에 원래 V 순환 그대로 (대칭 양정치 고정 연산자라 PCG 성립). 선탐색·장벽 보폭은 y 공간 dl 그대로.
기록: xtinfo.json, xt.jsonl (걸음 시작마다 묶인 외부 노드 수 · θ 평균 · 질량 가중 θ · θ 분위 근사 (θ < 0.25 · 0.25~0.5 · 0.5~0.75 · ≥ 0.75 · > 1 노드 수)).
"""
import atexit
import json
import os
import weakref

import numpy as np
import torch

ST = dict(th=0, out=".", nsrc=0, nth=0, nfold=0, nexp=0, nrest=0, nalloc=0, said=False, err=None)
MARK = "// ---- 절 CM133 extth"
HCAP = 8192
NH = 12
_SOL = []

XT_SRC = r'''
// ---- 절 CM133 extth (lib/anchorflow/fem_extth.py)
__global__ void xt_thz(const int* IP, float* th, double* MS) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9]) th[n] = 0.f; if (n == 0) MS[0] = 0.0; }
__global__ void xt_thc(const int* IP, const int* cb, float* th) {                     // xth 1: 점유 칸 c 의 27 노드에 ∫_c N_o / ∫ N_o (축 몫 1/6 · 2/3 · 1/6 텐서 곱)
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    const float W1[3] = {1.f/6.f, 2.f/3.f, 1.f/6.f};
    int nb1 = IP[7], nb2 = IP[8];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int d=0;d<3;d++)
        atomicAdd(&th[((cb[c*3]+a)*nb1 + (cb[c*3+1]+b))*nb2 + (cb[c*3+2]+d)], W1[a]*W1[b]*W1[d]);
}
__global__ void xt_msum(const int* IP, const float* mn, double* MS) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = (n < IP[9]) ? (double)mn[n] : 0.0;
    blocksum_add(s, &MS[0]);
}
__global__ void xt_thm(const int* IP, const float* mn, const double* MS, float* th) {   // xth 2: θ = m / (ρ̄ h³), ρ̄ h³ = Σ m / 점유 칸 수
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    double mc = MS[0]/(double)(IP[11] > 0 ? IP[11] : 1);
    th[n] = (mc > 0.0) ? (float)((double)mn[n]/mc) : 1.f;
}
__global__ void xt_stat(const int* IP, const int* base, const float* th, const float* mn, double* XS) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13] || base[n] < 0) return;
    float t = th[n];
    atomicAdd(&XS[0], 1.0); atomicAdd(&XS[1], (double)t); atomicAdd(&XS[2], (double)mn[n]*t); atomicAdd(&XS[3], (double)mn[n]);
    atomicAdd(&XS[t < 0.25f ? 4 : (t < 0.5f ? 5 : (t < 0.75f ? 6 : 7))], 1.0);
    if (t > 1.f) atomicAdd(&XS[8], 1.0);
}
__global__ void xt_log(double* XS, int* hp, double* H, int cap) {
    int s = hp[0]; hp[0] = s + 1;
    if (s < cap) for (int i=0;i<12;i++) H[s*12 + i] = XS[i];
    for (int i=0;i<12;i++) XS[i] = 0.0;
}
__global__ void xt_fold(const int* IP, const int* base, const float* w, const float* th, float* v, float* acc) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    float t = th[n], s = 1.f - t;
    float vv[3] = {v[n*3], v[n*3+1], v[n*3+2]};
    v[n*3] = t*vv[0]; v[n*3+1] = t*vv[1]; v[n*3+2] = t*vv[2];
    if (s == 0.f) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          ww *= s;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) atomicAdd(&acc[q*3+d], ww*vv[d]); } } }
}
__global__ void xt_expand(const int* IP, const int* base, const float* w, const float* th, float* p, float* Tp, float* pu) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t = th[n], s = 1.f - t;
    float e[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) e[d] += ww*Tp[q*3+d]; } } }
    for (int d=0;d<3;d++) { float u = p[n*3+d]; pu[n*3+d] = u; float y = t*u + s*e[d]; Tp[n*3+d] = y; p[n*3+d] = y; }
}
__global__ void xt_restore(const int* IP, const int* base, const float* pu, float* p) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    for (int d=0;d<3;d++) p[n*3+d] = pu[n*3+d];
}
'''


def add_src(src):
    if MARK in src:
        return src
    assert "blocksum_add(" in src, "extth: blocksum_add 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + XT_SRC + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _on(s):
    return getattr(s, "w8b", None) is not None and getattr(s, "_w8on", None) is not None and getattr(s, "xtth", None) is not None


def install(th=0, out="."):
    """감싸개 import 때. fem_wolfh8.install 을 감싸 그 직전에 RTSolver.k (안쪽)·build_src, 직후에 alloc 을 감쌈 (절 CM132 extfix 와 같은 자리)."""
    from . import fem_wolfh8 as W8
    ST.update(th=int(th), out=out)
    if not ST["th"]:
        return
    w8_install = W8.install

    def w8_install_x(out_, D=2, T=0, X=0):
        from . import fem_an_rt as ART
        R = ART.RTSolver
        ob = ART.build_src
        ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
        ok0 = R.k

        def k(s, name, n, args, b=128):
            if not _on(s):
                return ok0(s, name, n, args, b)
            i32 = np.int32
            if name == "w8_fold":                                                      # args: IP base w v acc
                ST["nfold"] += 1
                return ok0(s, "xt_fold", n, (args[0], args[1], args[2], P(s.xtth), args[3], args[4]), b)
            if name == "w8_expand":                                                    # args: IP base w p Tp
                ST["nexp"] += 1
                return ok0(s, "xt_expand", n, (args[0], args[1], args[2], P(s.xtth), args[3], args[4], P(s.xtpu)), b)
            if name == "rt_cg2":                                                       # args: IP SC sk pm n0 n1 n2 z p Tp KTz -- p 를 u 공간으로 되돌린 뒤 갱신
                ok0(s, "xt_restore", s.capN, (P(s.IP), P(s.w8b), P(s.xtpu), args[8]), 256)
                ST["nrest"] += 1
                return ok0(s, name, n, args, b)
            r = ok0(s, name, n, args, b)
            if name == "w8_class":
                ok0(s, "xt_thz", s.capN, (P(s.IP), P(s.xtth), P(s.xtMS)), 256)
                if ST["th"] == 1:
                    ok0(s, "xt_thc", s.capQc, (P(s.IP), P(s.cb), P(s.xtth)), 128)
                else:
                    ok0(s, "xt_msum", s.capN, (P(s.IP), P(s.mn), P(s.xtMS)), 256)
                    ok0(s, "xt_thm", s.capN, (P(s.IP), P(s.mn), P(s.xtMS), P(s.xtth)), 256)
                ok0(s, "xt_stat", s.capN, (P(s.IP), P(s.w8b), P(s.xtth), P(s.mn), P(s.xtXS)), 256)
                ok0(s, "xt_log", 1, (P(s.xtXS), P(s.xthp), P(s.xtH), i32(HCAP)), 1)
                ST["nth"] += 1
                if not ST["said"]:
                    ST["said"] = True
                    print(f"[절 CM133 extth] 걸음 시작 훅 -- 외부 노드 섞기 θ 정의 {ST['th']} (1 = 점유 칸 안 기저 적분 몫, 2 = m/(ρ̄ h³)); 접기·펼치기 θ 가중, rt_cg2 앞 u 공간 p 되돌림", flush=True)
                    _dump()
            return r
        R.k = k
        w8_install(out_, D=D, T=T, X=X)
        oa = R.alloc

        def alloc(s, need):
            r = oa(s, need)
            if getattr(s, "w8b", None) is not None:
                dev = s.dev
                s.xtth = torch.zeros(s.capN, device=dev, dtype=torch.float32)
                s.xtpu = torch.zeros(s.capN * 3, device=dev, dtype=torch.float32)
                s.xtMS = torch.zeros(4, device=dev, dtype=torch.float64)
                s.xtXS = torch.zeros(NH, device=dev, dtype=torch.float64)
                if getattr(s, "xtH", None) is None:
                    s.xtH = torch.zeros(HCAP, NH, device=dev, dtype=torch.float64)
                    s.xthp = torch.zeros(1, device=dev, dtype=torch.int32)
                    _SOL.append(weakref.ref(s))
                ST["nalloc"] += 1
            return r
        R.alloc = alloc
        print(f"[절 CM133] extth 켬 -- 외부 노드 θ 섞기 (θ 정의 {ST['th']}; wolfh8 차수 {D} 내부 칸 정의 {T})", flush=True)

    W8.install = w8_install_x
    atexit.register(_dump)


def _dump():
    try:
        rows = []
        for wr in _SOL:
            s = wr()
            if s is None:
                continue
            n = int(s.xthp.item())
            H = s.xtH[:min(n, HCAP)].cpu().numpy()
            for i, h in enumerate(H):
                no = h[0]
                rows.append(dict(k=i, outer=int(no), th_mean=(h[1] / no if no > 0 else None), th_mw=(h[2] / h[3] if h[3] > 0 else None),
                                 n_lt25=int(h[4]), n_25_50=int(h[5]), n_50_75=int(h[6]), n_ge75=int(h[7]), n_gt1=int(h[8])))
        if rows:
            with open(os.path.join(ST["out"], "xt.jsonl"), "w") as f_:
                for r_ in rows:
                    f_.write(json.dumps(r_) + "\n")
    except Exception as e_:
        ST["err"] = f"{type(e_).__name__}: {e_}"
    try:
        with open(os.path.join(ST["out"], "xtinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
