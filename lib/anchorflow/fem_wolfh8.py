"""절 CM81 묶음 wolfh8 (11-08 위임): wolf 가 정답보다 높게 쌓임 8 -- 확장 B-스플라인 (Höllig 2001, de Prenter 2017) 으로 가벼운 노드 자유도를 없앰.

배경 (절 CM70·CM77·CM79): wolf 높이 (f50 600, 정답 533) 는 P1X 벌점 강성이 흐르는 모래를 붙잡아서 생김. 강성을 빼면 (절 CM77 (K)) 낙하는 정답 수준이 되나
  가벼운 자유 노드 (질량 m̄ 의 2~6 배) 가 튀어 발산 4/4. 확장 B-스플라인은 그런 노드를 이웃 내부 노드의 다항 외삽으로 묶어 자유도에서 없앰.
정의 (새 문턱·클램프 없음; 나누는 기준은 칸 점유 위상):
  - 팔분 칸 점유: 칸 c (2 차 B-스플라인 매듭 칸, 노드 b..b+2 가 받침) 의 팔분 칸 o 가 입자 부피 > 0 (rt_gauss 의 A[q][27] = 팔분 칸 실제 입자 부피).
    내부 칸 = 팔분 8 개가 모두 점유된 칸 (적분 영역 = 점유 팔분 칸 합집합에 완전히 들어간 매듭 칸 -- Höllig 의 '내부 칸').
  - 내부 노드 = 받침 (매듭 칸 b ∈ [n−2, n]^3) 안에 내부 칸이 하나라도 있는 활성 노드 (m > 0). 외부 노드 = 나머지 활성 노드.
  - 외부 노드 o 는 가까운 내부 칸 c* 하나에 묶음: 후보 = 받침 블록 (노드 b..b+2) 이 o 의 26 이웃에 닿는 내부 칸 (블록 가운데 b+1 과 o 의 칸 거리 ≤ 2;
    거리 ≤ 1 이면 o 는 내부라 정의상 2), 그중 o 와 블록 가운데의 유클리드 거리 최소 (같으면 칸 번호 작은 것). c* 의 27 노드는 정의상 모두 내부.
    그런 칸이 없는 외부 노드 (내부 칸이 없는 조각) = '고립' -- 그대로 둠 (원래 자유도·원래 고정 판정).
  - 외삽 가중 = 축별 라그랑주 다항 (블록 노드 −1·0·1, o 의 축 거리 x ∈ {−2..2}; |x| ≤ 1 축은 그 노드 하나) 의 텐서 곱:
      차수 2 (기본, 2 차 B-스플라인과 같은 차수): x = 2 -> (1, −3, 3), x = −2 -> (3, −3, 1)
      차수 1 (대조): 가까운 두 노드 선형, x = 2 -> (0, −1, 2);  차수 0 (대조): 가까운 노드 하나 (가중 1).
  - 묶는 것은 뉴턴 걸음 증분 (아핀 확장): 걸음 시작점 y0 (원래 시작점 규칙 그대로 -- 장벽 실현 가능) 에서 y_o = y0_o + Σ_j c_oj (y_j − y0_j).
    외부 노드의 관성 몫은 자기 질량·자기 예측 ŷ_o 그대로 (½ m_o |y_o − ŷ_o|²/τ²) -- 줄어든 에너지가 원래 에너지에 y = E u 를 넣은 것 그대로.
훅 (실행기 runpy 순간, 가장 바깥):
  - 걸음 시작 (s.ctl(1), 시작점 다 정한 뒤 뉴턴 고리 직전): 칸 점유 -> 내부 칸 -> 노드 분류·가중 (w8_icell·w8_class), 외부 노드 변수 바꿈을 항등으로
    (pm 0, sk 1, η = y − ŷ -> y = ŷ + T η 그대로).
  - 뉴턴 기울기 (rt_grad 뒤): 잔차 접기 r_j += T_j Σ_o c_oj r_o, r_o = 0 (w8_fold·w8_fold2) + |g|² (SC[1]) 를 접은 잔차로 다시 (허용치·판정이 줄어든 문제 기준).
    선탐색 g·δ 는 원래 r0 (안 접음) · 펼친 δ = 줄어든 기울기 · 줄어든 δ 와 같음.
  - PCG: 방향 펼치기 (rt_pinit·rt_cg2 뒤) Tp_o = p_o = Σ_j c_oj (T p)_j (w8_expand); 헤시안 곱 접기 (rt_ap 뒤) Ap 를 같은 식으로 접음.
    p·Ap (rt_ap 가 접기 전에 더함) 는 펼친 p 의 원래 이차형식 = 줄어든 이차형식이라 그대로. MG 전처리 = r_o = 0 인 원래 V 순환 (주 부분행렬, 대칭 양정치 유지).
  - dl += α p (펼친 p) 라 δ_o 는 늘 펼친 값 -> 선탐색·장벽 보폭·rt_lsup 의 y = ŷ + T(η + c δ) 가 외부 노드에서 자동으로 확장 관계를 지킴.
  모든 새 커널은 허용치 조기 종료 (AF_SKIP) 를 지킴 (멈춘 반복에서 아무것도 안 함).
기록: w8info.json (설치·훅 수), w8.jsonl (프레임마다 내부 칸·내부·외부·고립 노드 수와 질량, 외삽 축 수 분포).
"""
import json
import os

import numpy as np
import torch

ST = dict(D=2, T=0, X=0, out=".", nsrc=0, nclass=0, nfold=0, nexp=0, ngg=0, nalloc=0, nstep=0, said=False, err=None)
MARK = "// ---- 절 CM81 wolfh8"

W8_SRC = r'''
// ---- 절 CM81 wolfh8: 확장 B-스플라인 (lib/anchorflow/fem_wolfh8.py)
__device__ __forceinline__ void w8_wts(int x, int deg, float* w) {          // 블록 가운데에서 축 거리 x 인 노드의 블록 노드 (−1, 0, 1) 가중
    w[0] = 0.f; w[1] = 0.f; w[2] = 0.f;
    if (x >= -1 && x <= 1) { w[x+1] = 1.f; return; }                          // (차수 0 은 |x| > 2 에도 끝 노드 하나)
    float t = (float)x;
    if (deg >= 2) { w[0] = 0.5f*t*(t - 1.f); w[1] = (1.f - t)*(1.f + t); w[2] = 0.5f*t*(t + 1.f); }
    else if (deg == 1) { if (x > 0) { w[1] = 1.f - t; w[2] = t; } else { w[0] = -t; w[1] = 1.f + t; } }
    else { if (x > 0) w[2] = 1.f; else w[0] = 1.f; }
}
__global__ void w8_icell(const int* IP, const int* cb, const float* A, int* ic, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int o=0;o<8;o++) if (!(A[(size_t)(c*8 + o)*28 + 27] > 0.f)) return;
    int n1 = IP[7]-2, n2 = IP[8]-2;
    ic[(cb[c*3]*n1 + cb[c*3+1])*n2 + cb[c*3+2]] = 1;
    atomicAdd(&cnt[3], 1);
}
__global__ void w8_icell2(const int* IP, const int* cb, const int* occ, int* ic, int* cnt, int face) {   // 내부 칸 정의 t 1: 점유 칸 중 26 이웃 칸이 모두 점유 (칸 점유 집합의 위상 내부), t 2 (face 1): 면 이웃 6 칸만
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int a0 = cb[c*3], b0 = cb[c*3+1], c0_ = cb[c*3+2];
    for (int a=a0-1;a<=a0+1;a++) for (int b=b0-1;b<=b0+1;b++) for (int d=c0_-1;d<=c0_+1;d++) {
        if (face && (a != a0) + (b != b0) + (d != c0_) > 1) continue;
        if (a < 0 || b < 0 || d < 0 || a >= n0 || b >= n1 || d >= n2) return;
        if (!occ[(a*n1 + b)*n2 + d]) return; }
    ic[(a0*n1 + b0)*n2 + c0_] = 1;
    atomicAdd(&cnt[3], 1);
}
__global__ void w8_class(const int* IP, const float* mn, const int* ic, int deg, int* base, float* w, int* cnt, float* cf, int xiso) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    base[n] = -1;
    float m = mn[n];
    if (!(m > 0.f) || IP[13]) return;
    int nb1 = IP[7], nb2 = IP[8];
    int c0 = IP[6]-2, c1 = nb1-2, c2 = nb2-2;
    int i = n/(nb1*nb2), j = (n/nb2)%nb1, k = n%nb2;
    for (int a=i-2;a<=i;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-2;b<=j;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-2;c<=k;c++) { if (c < 0 || c >= c2) continue;
          if (ic[(a*c1 + b)*c2 + c]) { atomicAdd(&cnt[0], 1); atomicAdd(&cf[0], m); return; } } } }
    int best = -1, bd = 1 << 30, ba = 0, bb = 0, bc = 0;
    for (int a=i-3;a<=i+1;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-3;b<=j+1;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-3;c<=k+1;c++) { if (c < 0 || c >= c2) continue;
          int cd = (a*c1 + b)*c2 + c;
          if (!ic[cd]) continue;
          int x = i-a-1, y = j-b-1, z = k-c-1, d = x*x + y*y + z*z;
          if (d < bd || (d == bd && cd < best)) { bd = d; best = cd; ba = a; bb = b; bc = c; } } } }
    int dg = deg;
    if (best < 0 && xiso) {                                                            // 절 CM81 x 1: 고립 노드 = 가장 가까운 내부 칸 (칸 거리 껍질 3, 4, ... 차례) 의 가장 가까운 블록 노드에 0 차 (평행 이동)
        int R = c0 > c1 ? c0 : c1; R = R > c2 ? R : c2;
        for (int d=3; d<=R+1 && best < 0; d++) {
          for (int a=i-1-d;a<=i-1+d;a++) { if (a < 0 || a >= c0) continue;
            for (int b=j-1-d;b<=j-1+d;b++) { if (b < 0 || b >= c1) continue;
              int ea = (a == i-1-d || a == i-1+d || b == j-1-d || b == j-1+d);
              for (int c=k-1-d;c<=k-1+d;c += (ea ? 1 : 2*d)) { if (c < 0 || c >= c2) continue;
                int cd = (a*c1 + b)*c2 + c;
                if (!ic[cd]) continue;
                int x = i-a-1, y = j-b-1, z = k-c-1, dd = x*x + y*y + z*z;
                if (dd < bd || (dd == bd && cd < best)) { bd = dd; best = cd; ba = a; bb = b; bc = c; } } } } }
        if (best >= 0) { dg = 0; atomicAdd(&cnt[8], 1); atomicAdd(&cf[3], m); }
    }
    if (best < 0) { atomicAdd(&cnt[2], 1); atomicAdd(&cf[2], m); return; }           // 고립 (가까운 내부 칸 없음) -- 그대로
    atomicAdd(&cnt[1], 1); atomicAdd(&cf[1], m);
    int xs[3] = {i-ba-1, j-bb-1, k-bc-1}, ne = 0;
    for (int a=0;a<3;a++) if (xs[a] < -1 || xs[a] > 1) ne++;
    atomicAdd(&cnt[4 + ne], 1);                                                       // 외삽 축 수 1·2·3 -> cnt[5..7]
    base[n] = (ba*nb1 + bb)*nb2 + bc;
    w8_wts(xs[0], dg, w + n*9); w8_wts(xs[1], dg, w + n*9 + 3); w8_wts(xs[2], dg, w + n*9 + 6);
}
__global__ void w8_free(const int* IP, const int* base, float* sk, int* pm, const float* yh, const float* y, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    sk[n] = 1.f; pm[n] = 0;
    for (int a=0;a<3;a++) eta[n*3+a] = y[n*3+a] - yh[n*3+a];
}
__global__ void w8_fold(const int* IP, const int* base, const float* w, float* v, float* acc) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    float vv[3] = {v[n*3], v[n*3+1], v[n*3+2]};
    v[n*3] = 0.f; v[n*3+1] = 0.f; v[n*3+2] = 0.f;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) atomicAdd(&acc[q*3+d], ww*vv[d]); } } }
}
__global__ void w8_fold2(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, float* acc, float* v) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float a3[3] = {acc[n*3], acc[n*3+1], acc[n*3+2]};
    if (a3[0] == 0.f && a3[1] == 0.f && a3[2] == 0.f) return;
    acc[n*3] = 0.f; acc[n*3+1] = 0.f; acc[n*3+2] = 0.f;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, a3, t);
    for (int d=0;d<3;d++) v[n*3+d] += t[d];
}
__global__ void w8_gg(const int* IP, const float* r, double* SC) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) for (int d=0;d<3;d++) { double q = r[n*3+d]; s += q*q; }
    blocksum_add(s, &SC[1]);
}
__global__ void w8_expand(const int* IP, const int* base, const float* w, float* p, float* Tp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) t[d] += ww*Tp[q*3+d]; } } }
    for (int d=0;d<3;d++) { Tp[n*3+d] = t[d]; p[n*3+d] = t[d]; }
}
'''


def add_src(src):
    if MARK in src:
        return src
    assert "Tapply(" in src and "blocksum_add(" in src, "wolfh8: Tapply·blocksum_add 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + W8_SRC + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _on(s):
    return getattr(s, "_w8on", None) is not None and getattr(s, "w8b", None) is not None


def install(out, D=2, T=0, X=0):
    from . import fem_an_rt as ART
    ST.update(D=int(D), T=int(T), X=int(X), out=out)
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.RTSolver
    oa, ok, oc, os_ = R.alloc, R.k, R.ctl, R.step

    def alloc(s, need):
        r = oa(s, need)
        dev = s.dev
        s.w8ic = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.w8b = torch.full((s.capN,), -1, device=dev, dtype=torch.int32)
        s.w8w = torch.zeros(s.capN * 9, device=dev, dtype=torch.float32)
        s.w8acc = torch.zeros(s.capN * 3, device=dev, dtype=torch.float32)
        s.w8cnt = torch.zeros(16, device=dev, dtype=torch.int32)
        s.w8cf = torch.zeros(4, device=dev, dtype=torch.float32)
        s._w8on = True
        ST["nalloc"] += 1
        return r

    def fold(s, v, sk, pm, nr):
        i32 = np.int32
        ok(s, "w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), v, P(s.w8acc)), 256)
        ok(s, "w8_fold2", s.capN, (P(s.IP), sk, pm, *nr, P(s.w8acc), v), 256)
        ST["nfold"] += 1

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if not _on(s):
            return r
        i32 = np.int32
        if name == "rt_grad":                                                         # args: IP mn it2 sk pm n0 n1 n2 eta f B Dinv r z p dl Tp SC r0
            fold(s, args[12], args[3], args[4], tuple(args[5:8]))
            ok(s, "rt_sc_clear", 1, (args[17], i32(1), i32(2)), 1)
            ok(s, "w8_gg", s.capN, (P(s.IP), args[12], args[17]), 256)
            ST["ngg"] += 1
        elif name == "rt_ap":                                                         # args: IP mn it2 sk pm n0 n1 n2 p KTp Ap SC Kb
            fold(s, args[10], args[3], args[4], tuple(args[5:8]))
        elif name == "rt_pinit":                                                      # args: IP z sk pm n0 n1 n2 p Tp
            ok(s, "w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[7], args[8]), 256)
            ST["nexp"] += 1
        elif name == "rt_cg2":                                                        # args: IP SC sk pm n0 n1 n2 z p Tp KTz
            ok(s, "w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[8], args[9]), 256)
            ST["nexp"] += 1
        return r

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 1 and _on(s):
            assert getattr(s, "nl", 1) > 1, "wolfh8: MG 판 (--mg_levels > 1) 만 (rt_grad 의 z·p 를 MG 가 다시 씀)"
            i32 = np.int32
            ok(s, "rt_zero_i", s.capC, (i32(s.capC), P(s.w8ic)), 256)
            if ST["T"] in (1, 2):
                ok(s, "w8_icell2", s.capQc, (P(s.IP), P(s.cb), P(s.occ), P(s.w8ic), P(s.w8cnt), i32(1 if ST["T"] == 2 else 0)), 128)
            else:
                ok(s, "w8_icell", s.capQc, (P(s.IP), P(s.cb), P(s.Aacc), P(s.w8ic), P(s.w8cnt)), 128)
            ok(s, "w8_class", s.capN, (P(s.IP), P(s.mn), P(s.w8ic), i32(ST["D"]), P(s.w8b), P(s.w8w), P(s.w8cnt), P(s.w8cf), i32(ST["X"])), 256)
            ok(s, "w8_free", s.capN, (P(s.IP), P(s.w8b), P(s.sk), P(s.pm), P(s.yh), P(s.y), P(s.eta)), 256)
            ST["nclass"] += 1
            if not ST["said"]:
                ST["said"] = True
                print(f"[절 CM81 wolfh8] 확장 훅 -- 차수 {ST['D']}, 걸음 시작 분류·변수 바꿈 항등, rt_grad 접기 + |g|² 다시, rt_pinit·rt_cg2 펼치기, rt_ap 접기 (MG 층 {s.nl})", flush=True)
                _dump()
        return r

    def step(s, t):
        on = _on(s)
        if on:
            s.w8cnt.zero_(); s.w8cf.zero_()
        r = os_(s, t)
        if on:
            try:
                c = s.w8cnt.tolist(); m = s.w8cf.tolist()
                mt = sum(m[:3])
                with open(os.path.join(ST["out"], "w8.jsonl"), "a") as f_:
                    f_.write(json.dumps(dict(f=ST["nstep"], icell=c[3], ncell=int(s.IP[11]), inner=c[0], outer=c[1], iso=c[2], ax1=c[5], ax2=c[6], ax3=c[7], xiso=c[8], m_xiso=m[3],
                                             m_inner=m[0], m_outer=m[1], m_iso=m[2], mf_outer=(m[1] / mt if mt > 0 else None), mf_iso=(m[2] / mt if mt > 0 else None), redo=bool(r))) + "\n")
            except Exception as e_:
                ST["err"] = f"{type(e_).__name__}: {e_}"
        ST["nstep"] += 1
        return r
    R.alloc, R.k, R.ctl, R.step = alloc, k, ctl, step
    print(f"[절 CM81] wolfh8 켬 -- 확장 B-스플라인 차수 {ST['D']} (내부 칸 정의 {ST['T']}: 0 = 팔분 8 개 점유, 1 = 26 이웃 칸 모두 점유, 2 = 면 이웃 6 칸 모두 점유; 외부 노드 = 받침에 내부 칸 없는 활성 노드, 증분 아핀 확장)", flush=True)
    import atexit
    atexit.register(_dump)


def _dump():
    try:
        with open(os.path.join(ST["out"], "w8info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
