"""절 CM94 묶음 wolfh13 (11-08 위임): 확장 B-스플라인을 반영한 MG 전처리기 (풀리는 방정식·수렴 해는 그대로, 전처리만 바뀜).

배경 (절 CM93): H8 wolf 부피 붕괴의 주원인은 덜 풀린 걸음 (걸음 끝 잔차 비 H5 의 약 3 천 배), 확장 T1E2 가 덜 풀림을 키움 (첫 프레임 잔차 14 대 확장 끔 0.004).
  지금 (fem_wolfh8) 은 줄어든 문제 Ã = Eᵀ A E (E = 외부 노드를 내부 노드 외삽으로 묶는 확장) 를 PCG 로 풀면서 전처리는 r_o = 0 인 원래 A 의 V 순환
  (주 부분행렬 A_II 근사) -- 외부 노드가 접혀 들어오는 몫 (A_IO E + Eᵀ A_OI + Eᵀ A_OO E, 외삽 가중 |c| 가 최대 27) 을 모름.
정의 (de Prenter 2017 'Condition number analysis and preconditioning of the finite cell method' · 2019 'Multigrid solvers for immersed finite element
  methods and immersed isogeometric analysis' 의 '확장/잘린 받침 자유도를 전처리에 반영' 을 이 실행기 MG 에 옮김; 새 문턱·클램프 없음):
  - 층 0 (고운 층): 블록 대각 = Eᵀ D E 의 블록 대각, 곧 내부 노드 j 에 D̃_j = M_j + T_j (Σ_o c_oj² M_o) T_j (M = m/τ² I + B, B = 요소·장벽 블록 대각,
    T = 장벽 변수 바꿈; 외부 노드는 T = I). 외삽 받는 것이 없는 노드는 rt_grad 의 D⁻¹ 그대로 (비트 같음), 외부 노드는 D̃⁻¹ = 0 (자유도 아님).
    평활 (ω D̃⁻¹ 야코비) 의 잔차는 접은 잔차 Eᵀ (r − A E z), 곱 전에 z 를 펼침 (z_o = Σ c_oj (T z)_j), ω 는 같은 거듭제곱법을 D̃⁻¹ Ã 에.
  - 층 사이 전달: 연장 Q = I_내부 T P E_c (거친 층 보정을 고운 층에 연장한 뒤 외부 노드는 다시 펼침), 제한 = Qᵀ (고운 잔차를 접은 뒤 외부 0 으로 제한, 거친 층에서 다시 접음).
  - 거친 층 l ≥ 1 (모드 1): 같은 정의의 거친 층 확장 E_l -- 거친 칸 점유 (rt_mg_fin 의 칸 부피 > 0), 내부 칸 = 점유 칸 중 26 이웃 (T2 면 6) 이 모두 점유
    (fem_wolfh8 t 1·2 와 같은 위상 규칙), 외부 노드 = 받침에 내부 칸 없는 활성 노드 (거친 질량 > 0), 가까운 내부 칸 블록에 같은 차수 외삽.
    거친 연산자 Ã_l = E_lᵀ A_l E_l, 블록 대각 E_lᵀ D_l E_l (D_l 은 장벽 덩어리 K n nᵀ 포함 -- 원래 Db 와 같은 것), ω 는 같은 거듭제곱법.
    모드 2 = 층 0 만 (거친 층은 원래 V 순환) -- 가르기용.
  수렴 해는 같음: PCG 가 푸는 것은 그대로 Ã δ = Eᵀ (−g), 전처리 (대칭 V 순환) 만 바뀜.
훅 (실행기 runpy 순간, 가장 바깥; fem_wolfh8 이 켜진 판 (κ 0) 에서만 -- κ 1 씬은 할당도 커널도 없음):
  - alloc: 층 0 S·임시, 거친 층마다 분류·가중·S·D̃·임시 버퍼.
  - k('rt_grad') 뒤 (fem_wolfh8 의 잔차 접기 다음): 층 0 D̃⁻¹ (s.Dinv 를 그 자리에서), 거친 층 D̃⁻¹ (뉴턴 반복마다 -- 장벽 Kb 가 반복마다 바뀜).
  - mg_build: 거친 층 분류 (프레임마다, 층 차원·부피 정해진 뒤), 거친 층 거듭제곱법 (원래 --mg_lreuse 규칙 그대로) 을 Ã_l 로.
  - mg_power(0): Ã 로. _mg_vc(0): 아래 V 순환 (원래 커널 rt_mg_node·rt_mg_restrict3·rt_mg_prolong·mg_Az 를 그대로 쓰고 접기·펼치기·D̃ 야코비만 더함).
  모든 새 커널은 허용치 조기 종료 (AF_SKIP) 를 지킴. 그래프 캡처 안에서 호스트 동기화 없음 (기록은 장치 누적 -> 끝에서 한 번 읽음).
기록: xpinfo.json (설치·할당·훅 수, 장치 누적 수: 층 0 D̃ 바뀐 노드, 거친 층별 내부 칸·내부·외부·고립 노드·D̃ 바뀐 노드).
"""
import json
import os

import numpy as np
import torch

ST = dict(mode=1, out=".", nsrc=0, nalloc=0, ngrad=0, nvc=0, npw0=0, npwl=0, nbuild=0, ncls=0, said=False, err=None, dev=None)
MARK = "// ---- 절 CM94 wolfh13"
_SOLVERS = []

XP_SRC = r'''
// ---- 절 CM94 wolfh13: 확장 반영 전처리기 (lib/anchorflow/fem_wolfh13.py)
#ifdef AF_ADAPT
#define XP_SKIP if (AF_SKIP) return;
#else
#define XP_SKIP
#endif
// 층 0 노드 번호 (노드 압축 판이면 ACT 목록, 아니면 0..IP[9]-1)
__device__ __forceinline__ int xp_cnt(const int* ACT, const int* IP) { return ACT ? ACT[0] : IP[9]; }
__device__ __forceinline__ int xp_nid(const int* ACT, int i) { return ACT ? ACT[1+i] : i; }
// 층 0: 외부 노드 o 의 M_o = m_o/τ² I + B_o 를 묶인 블록 노드 j 에 c_oj² 배로 (S_j)
__global__ void xp_s0(const int* IP, const int* base, const float* w, const float* mn, float itau2, const float* B, float* S) {
    XP_SKIP
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    float M[9]; float Mi = mn[n]*itau2;
    for (int k=0;k<9;k++) M[k] = B[n*9+k] + ((k % 4 == 0) ? Mi : 0.f);
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float w2 = ww*ww;
          for (int k=0;k<9;k++) atomicAdd(&S[q*9+k], w2*M[k]); } } }
}
// 층 0: D̃⁻¹ = (m/τ² I + T (B + S) T)⁻¹ (S 가 0 인 노드는 rt_grad 값 그대로), 외부 노드 0; S 비움
__global__ void xp_d0(const int* IP, const int* base, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2,
                      const float* B, float* S, float* Dinv, int* cnt) {
    XP_SKIP
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float Sn[9]; bool nz = false;
    for (int k=0;k<9;k++) { Sn[k] = S[n*9+k]; nz = nz || (Sn[k] != 0.f); S[n*9+k] = 0.f; }
    if (base[n] >= 0) { for (int k=0;k<9;k++) Dinv[n*9+k] = 0.f; return; }
    if (!nz) return;
    float T[9]; Tmat(sk[n], pm[n], n0, n1, n2, T);
    float Bs[9]; for (int k=0;k<9;k++) Bs[k] = B[n*9+k] + Sn[k];
    float Mi = mn[n]*itau2, TB[9], M[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += T[a*3+k]*Bs[k*3+b]; TB[a*3+b] = t; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = (a==b) ? Mi : 0.f; for (int k=0;k<3;k++) t += TB[a*3+k]*T[k*3+b]; M[a*3+b] = t; }
    float Di[9]; inv3(M, Di);
    for (int k=0;k<9;k++) Dinv[n*9+k] = Di[k];
    atomicAdd(&cnt[0], 1);
}
// 층 0 제한 입력 접기 마무리: v_j += T_j T_j acc_j (v = T (r − A z), 외부 몫은 w8_fold 가 acc 로 옮기고 0)
__global__ void xp_f2TT(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, float* acc, float* v) {
    XP_SKIP
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float a3[3] = {acc[n*3], acc[n*3+1], acc[n*3+2]};
    if (a3[0] == 0.f && a3[1] == 0.f && a3[2] == 0.f) return;
    acc[n*3] = 0.f; acc[n*3+1] = 0.f; acc[n*3+2] = 0.f;
    float t[3], u[3]; Tapply(sk[n], pm[n], n0, n1, n2, a3, t); Tapply(sk[n], pm[n], n0, n1, n2, t, u);
    for (int d=0;d<3;d++) v[n*3+d] += u[d];
}
// 층 0 야코비: z += ω D̃⁻¹ e (e = 접은 잔차), Tz = T z (외부 노드는 건너뜀 -- 뒤에서 펼침)
__global__ void xp_jac0(const int* IP, const int* ACT, const int* base, const float* D, const double* W, const float* e, const float* sk, const int* pm,
                        float n0, float n1, float n2, float* z, float* Tz) {
    XP_SKIP
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= xp_cnt(ACT, IP)) return;
    int n = xp_nid(ACT, i);
    if (base[n] >= 0) return;
    float w = (float)W[0]; const float* d = D + n*9;
    float ee[3] = {e[n*3], e[n*3+1], e[n*3+2]}, zz[3];
    for (int a=0;a<3;a++) { zz[a] = z[n*3+a] + w*(d[a*3]*ee[0] + d[a*3+1]*ee[1] + d[a*3+2]*ee[2]); z[n*3+a] = zz[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, t); for (int a=0;a<3;a++) Tz[n*3+a] = t[a];
}
// 층 0 거듭제곱법: out = m/τ² z + T kt (KT 비움; rt_mg_pw1 의 Az 와 같은 식)
__global__ void xp_az0(const int* IP, const int* ACT, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2,
                       const float* z, float* KT, float* out) {
    XP_SKIP
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= xp_cnt(ACT, IP)) return;
    int n = xp_nid(ACT, i);
    float kt[3] = {KT[n*3], KT[n*3+1], KT[n*3+2]};
    KT[n*3] = 0.f; KT[n*3+1] = 0.f; KT[n*3+2] = 0.f;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t);
    float Mi = mn[n]*itau2;
    for (int a=0;a<3;a++) out[n*3+a] = Mi*z[n*3+a] + t[a];
}
// 층 0 거듭제곱법: y = D̃⁻¹ (접은 Az) (외부 0), SCM[0..2] += z·y, z·z, y·y (외부 노드 빼고)
__global__ void xp_pwy0(const int* IP, const int* ACT, const int* base, const float* D, const float* z, const float* e, float* y, double* SCM) {
    XP_SKIP
    int cnt = xp_cnt(ACT, IP);
    if (blockIdx.x*blockDim.x >= cnt) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    if (i < cnt) {
        int n = xp_nid(ACT, i);
        if (base[n] >= 0) { y[n*3] = 0.f; y[n*3+1] = 0.f; y[n*3+2] = 0.f; }
        else { const float* d = D + n*9; float ee[3] = {e[n*3], e[n*3+1], e[n*3+2]};
            for (int a=0;a<3;a++) { float v = d[a*3]*ee[0] + d[a*3+1]*ee[1] + d[a*3+2]*ee[2]; y[n*3+a] = v; s1 += (double)z[n*3+a]*v; s2 += (double)z[n*3+a]*z[n*3+a]; s3 += (double)v*v; } }
    }
    blocksum_add(s1, &SCM[0]); blocksum_add(s2, &SCM[1]); blocksum_add(s3, &SCM[2]);
}
// ---- 거친 층 l (LM[l*8]: 0..2 노드 상자, 3 노드 수, 4..6 칸 상자, 7 가우스 점 수) -- T 없음
__global__ void xl_icell(int l, const int* LM, const float* Vq, int* ic, int face, int* cnt) {
    XP_SKIP
    const int* L = LM + l*8;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int nc = L[7] >> 3;
    if (c >= nc) return;
    int n0 = L[4], n1 = L[5], n2 = L[6];
    int cz = c % n2, cy = (c / n2) % n1, cx = c / (n1*n2);
    int in = 1;
    for (int a=cx-1;a<=cx+1 && in;a++) for (int b=cy-1;b<=cy+1 && in;b++) for (int d=cz-1;d<=cz+1;d++) {
        if (face && (a != cx) + (b != cy) + (d != cz) > 1) continue;
        if (a < 0 || b < 0 || d < 0 || a >= n0 || b >= n1 || d >= n2) { in = 0; break; }
        if (!(Vq[(size_t)((a*n1 + b)*n2 + d)*8] > 0.f)) { in = 0; break; } }
    ic[c] = in;
    if (in) atomicAdd(&cnt[3], 1);
}
__device__ __forceinline__ void xl_wts(int x, int deg, float* w) {          // fem_wolfh8 w8_wts 와 같은 식
    w[0] = 0.f; w[1] = 0.f; w[2] = 0.f;
    if (x >= -1 && x <= 1) { w[x+1] = 1.f; return; }
    float t = (float)x;
    if (deg >= 2) { w[0] = 0.5f*t*(t - 1.f); w[1] = (1.f - t)*(1.f + t); w[2] = 0.5f*t*(t + 1.f); }
    else if (deg == 1) { if (x > 0) { w[1] = 1.f - t; w[2] = t; } else { w[0] = -t; w[1] = 1.f + t; } }
    else { if (x > 0) w[2] = 1.f; else w[0] = 1.f; }
}
__global__ void xl_class(int l, const int* LM, const float* mc, const int* ic, int deg, int* base, float* w, int* cnt) {   // fem_wolfh8 w8_class (고립 그대로 판) 를 층 차원으로
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3]) return;
    base[n] = -1;
    float m = mc[n];
    if (!(m > 0.f)) return;
    int nb1 = L[1], nb2 = L[2];
    int c0 = L[4], c1 = L[5], c2 = L[6];
    int i = n/(nb1*nb2), j = (n/nb2)%nb1, k = n%nb2;
    for (int a=i-2;a<=i;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-2;b<=j;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-2;c<=k;c++) { if (c < 0 || c >= c2) continue;
          if (ic[(a*c1 + b)*c2 + c]) { atomicAdd(&cnt[0], 1); return; } } } }
    int best = -1, bd = 1 << 30, ba = 0, bb = 0, bc = 0;
    for (int a=i-3;a<=i+1;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-3;b<=j+1;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-3;c<=k+1;c++) { if (c < 0 || c >= c2) continue;
          int cd = (a*c1 + b)*c2 + c;
          if (!ic[cd]) continue;
          int x = i-a-1, y = j-b-1, z = k-c-1, d = x*x + y*y + z*z;
          if (d < bd || (d == bd && cd < best)) { bd = d; best = cd; ba = a; bb = b; bc = c; } } } }
    if (best < 0) { atomicAdd(&cnt[2], 1); return; }
    atomicAdd(&cnt[1], 1);
    base[n] = (ba*nb1 + bb)*nb2 + bc;
    xl_wts(i-ba-1, deg, w + n*9); xl_wts(j-bb-1, deg, w + n*9 + 3); xl_wts(k-bc-1, deg, w + n*9 + 6);
}
__global__ void xl_fold(int l, const int* LM, const int* base, const float* w, float* v, float* acc) {   // acc_j += Σ_o c_oj v_o, v_o = 0
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3] || base[n] < 0) return;
    float vv[3] = {v[n*3], v[n*3+1], v[n*3+2]};
    v[n*3] = 0.f; v[n*3+1] = 0.f; v[n*3+2] = 0.f;
    int nb1 = L[1], nb2 = L[2], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) atomicAdd(&acc[q*3+d], ww*vv[d]); } } }
}
__global__ void xl_fold2(int l, const int* LM, float* acc, float* v) {
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3]) return;
    float a3[3] = {acc[n*3], acc[n*3+1], acc[n*3+2]};
    if (a3[0] == 0.f && a3[1] == 0.f && a3[2] == 0.f) return;
    acc[n*3] = 0.f; acc[n*3+1] = 0.f; acc[n*3+2] = 0.f;
    for (int d=0;d<3;d++) v[n*3+d] += a3[d];
}
__global__ void xl_expand(int l, const int* LM, const int* base, const float* w, float* z) {   // z_o = Σ_j c_oj z_j
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3] || base[n] < 0) return;
    int nb1 = L[1], nb2 = L[2], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) t[d] += ww*z[q*3+d]; } } }
    for (int d=0;d<3;d++) z[n*3+d] = t[d];
}
// 거친 층 M = m/τ² I + B (+ K n nᵀ, 장벽 덩어리가 이 층 연산에 들어가는 판) -- rt_mg_D·rt_mg_dbar 가 뒤집는 것과 같은 행렬
__device__ __forceinline__ void xl_M(int n, const float* mc, float itau2, const float* B, const float* Kb, float n0, float n1, float n2, float* M) {
    float Mi = mc[n]*itau2, K = Kb ? Kb[n] : 0.f, nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) M[a*3+b] = B[n*9+a*3+b] + ((a==b) ? Mi : 0.f) + K*nn[a]*nn[b];
}
__global__ void xl_s(int l, const int* LM, const int* base, const float* w, const float* mc, float itau2, const float* B, const float* Kb, float n0, float n1, float n2, float* S) {
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3] || base[n] < 0) return;
    float M[9]; xl_M(n, mc, itau2, B, Kb, n0, n1, n2, M);
    int nb1 = L[1], nb2 = L[2], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float w2 = ww*ww;
          for (int k=0;k<9;k++) atomicAdd(&S[q*9+k], w2*M[k]); } } }
}
__global__ void xl_d(int l, const int* LM, const int* base, const float* mc, float itau2, const float* B, const float* Kb, float n0, float n1, float n2,
                     const float* Din, float* S, float* Dt, int* cnt) {
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3]) return;
    float Sn[9]; bool nz = false;
    for (int k=0;k<9;k++) { Sn[k] = S[n*9+k]; nz = nz || (Sn[k] != 0.f); S[n*9+k] = 0.f; }
    if (base[n] >= 0) { for (int k=0;k<9;k++) Dt[n*9+k] = 0.f; return; }
    if (!nz) { for (int k=0;k<9;k++) Dt[n*9+k] = Din[n*9+k]; return; }
    float M[9]; xl_M(n, mc, itau2, B, Kb, n0, n1, n2, M);
    for (int k=0;k<9;k++) M[k] += Sn[k];
    float Di[9]; inv3(M, Di);
    for (int k=0;k<9;k++) Dt[n*9+k] = Di[k];
    atomicAdd(&cnt[4], 1);
}
__global__ void xl_jac(int l, const int* LM, const int* base, const float* D, const double* W, const float* e, float* z) {
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3] || base[n] >= 0) return;
    float w = (float)W[l]; const float* d = D + n*9;
    float ee[3] = {e[n*3], e[n*3+1], e[n*3+2]};
    for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*ee[0] + d[a*3+1]*ee[1] + d[a*3+2]*ee[2]);
}
__global__ void xl_az(int l, const int* LM, const float* mc, float itau2, const float* z, float* KT, float* out) {
    XP_SKIP
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3]) return;
    float Mi = mc[n]*itau2;
    for (int a=0;a<3;a++) { out[n*3+a] = Mi*z[n*3+a] + KT[n*3+a]; KT[n*3+a] = 0.f; }
}
__global__ void xl_pwy(int l, const int* LM, const int* base, const float* D, const float* z, const float* e, float* y, double* SCM) {
    XP_SKIP
    const int* L = LM + l*8;
    int nn = L[3];
    if (blockIdx.x*blockDim.x >= nn) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    if (n < nn) {
        if (base[n] >= 0) { y[n*3] = 0.f; y[n*3+1] = 0.f; y[n*3+2] = 0.f; }
        else { const float* d = D + n*9; float ee[3] = {e[n*3], e[n*3+1], e[n*3+2]};
            for (int a=0;a<3;a++) { float v = d[a*3]*ee[0] + d[a*3+1]*ee[1] + d[a*3+2]*ee[2]; y[n*3+a] = v; s1 += (double)z[n*3+a]*v; s2 += (double)z[n*3+a]*z[n*3+a]; s3 += (double)v*v; } }
    }
    blocksum_add(s1, &SCM[l*4]); blocksum_add(s2, &SCM[l*4+1]); blocksum_add(s3, &SCM[l*4+2]);
}
'''


def add_src(src):
    if MARK in src:
        return src
    assert "Tapply(" in src and "blocksum_add(" in src and "inv3(" in src and "Tmat(" in src, "wolfh13: Tapply·blocksum_add·inv3·Tmat 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + XP_SRC + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def install(out, mode=1):
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wolfh8 as W8
    ST.update(mode=int(mode), out=out)
    assert ST["mode"] in (1, 2), ST["mode"]
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.RTSolver
    oa, ok, ovc, opw, omb = R.alloc, R.k, R._mg_vc, R.mg_power, R.mg_build
    i32, f32 = np.int32, np.float32

    def on(s):
        return getattr(s, "_xpon", False) and W8._on(s)

    def coarse(s):
        return ST["mode"] == 1

    def nrf(s):
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        return [f32(q) for q in nr]

    def it2m(s):
        return f32(1.0 / (getattr(s, "mg_tau", s.tau) ** 2))

    def alloc(s, need):
        r = oa(s, need)
        if W8._on(s) and getattr(s, "nl", 1) > 1 and not isinstance(s, AORT.AORTSolver):
            assert not s.pk and not s.mg_fuse and not s.mg_csolve and not getattr(s, "mg_occ", False), "wolfh13: --pk·--mg_fuse·--mg_csolve·--mg_occ 없는 MG 판만"
            dev = s.dev
            Z = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.float32)
            s.xpS = Z(s.capN * 9); s.xpT = Z(s.capN * 3)
            old = getattr(s, "xpc", None)
            s.xpc = torch.zeros(64, device=dev, dtype=torch.int32)
            if old is not None:
                s.xpc.copy_(old)
            s.XL = []
            for L in s.ML:
                cn, cq = L["capn"], L["capq"]
                s.XL.append(dict(base=torch.full((cn,), -1, device=dev, dtype=torch.int32), w=Z(cn * 9), acc=Z(cn * 3), ic=torch.zeros(cq // 8, device=dev, dtype=torch.int32),
                                 S=Z(cn * 9), Dt=Z(cn * 9), tmp=Z(cn * 3)))
            s._xpon = True
            ST["nalloc"] += 1
            if s not in _SOLVERS:
                _SOLVERS.append(s)
        return r

    # ---------------- 블록 대각 D̃ (뉴턴 반복마다, rt_grad 뒤)
    def d0(s, args):
        IP = args[0]
        s.k("xp_s0", s.capN, (IP, P(s.w8b), P(s.w8w), args[1], args[2], args[10], P(s.xpS)), b=256)
        s.k("xp_d0", s.capN, (IP, P(s.w8b), args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[10], P(s.xpS), args[11], P(s.xpc)), b=256)

    def dl(s, l):
        L, X = s.ML[l - 1], s.XL[l - 1]
        mgb = getattr(s, "_mgb_on", False)
        Kb = L["Kb"] if mgb else None
        Din = L["Db"] if mgb else L["D"]
        nr = nrf(s); it2 = it2m(s)
        c = s.xpc[8 + (l - 1) * 8:]
        s.k("xl_s", L["capn"], (i32(l), P(s.LM), P(X["base"]), P(X["w"]), P(L["mc"]), it2, P(L["B"]), P(Kb), *nr, P(X["S"])), b=256)
        s.k("xl_d", L["capn"], (i32(l), P(s.LM), P(X["base"]), P(L["mc"]), it2, P(L["B"]), P(Kb), *nr, P(Din), P(X["S"]), P(X["Dt"]), P(c)), b=256)

    def classify(s):
        from . import fem_wolfh8 as W8_
        face = 1 if W8_.ST["T"] == 2 else 0
        for l in range(1, s.nl):
            L, X = s.ML[l - 1], s.XL[l - 1]
            c = s.xpc[8 + (l - 1) * 8:]
            s.k("xl_icell", L["capq"] // 8, (i32(l), P(s.LM), P(L["Vq"]), P(X["ic"]), i32(face), P(c)), b=128)
            s.k("xl_class", L["capn"], (i32(l), P(s.LM), P(L["mc"]), P(X["ic"]), i32(W8_.ST["D"]), P(X["base"]), P(X["w"]), P(c)), b=256)
            dl(s, l)
        s._xpcl = True
        ST["ncls"] += 1

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if name == "rt_grad" and on(s):
            d0(s, args)
            if coarse(s):
                for l in range(1, s.nl):
                    dl(s, l)
            ST["ngrad"] += 1
        return r

    def mg_build(s):
        if not (on(s) and coarse(s)):
            return omb(s)
        s._xpcl = False
        r = omb(s)
        if not s._xpcl:
            classify(s)
        ST["nbuild"] += 1
        return r

    # ---------------- 층 0 펼치기·접기
    def expand0(s, z, Tz):
        s.k("w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(z), P(Tz)), b=256)

    def fold0(s, v):
        nr = nrf(s)
        s.k("w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(v), P(s.w8acc)), b=256)
        s.k("w8_fold2", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nr, P(s.w8acc), P(v)), b=256)

    def fold0TT(s, v):
        nr = nrf(s)
        s.k("w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(v), P(s.w8acc)), b=256)
        s.k("xp_f2TT", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nr, P(s.w8acc), P(v)), b=256)

    def expandl(s, l, z):
        X = s.XL[l - 1]
        s.k("xl_expand", s.ML[l - 1]["capn"], (i32(l), P(s.LM), P(X["base"]), P(X["w"]), P(z)), b=256)

    def foldl(s, l, v):
        X, cn = s.XL[l - 1], s.ML[l - 1]["capn"]
        s.k("xl_fold", cn, (i32(l), P(s.LM), P(X["base"]), P(X["w"]), P(v), P(X["acc"])), b=256)
        s.k("xl_fold2", cn, (i32(l), P(s.LM), P(X["acc"]), P(v)), b=256)

    def act(s):
        return P(s.actl) if s.ncomp else P(None)

    def restrict(s, l, src, dst):
        Lc = s.ML[l]
        s.k(*(("rt_mg_restrict3", Lc["capn"] * 16) if s.mv2 else ("rt_mg_restrict", Lc["capn"])), (i32(l), P(s.LM), P(src), P(dst), i32(3)), b=256)

    # ---------------- 거듭제곱법 (ω = mg_omega / λ_max(D̃⁻¹ Ã))
    def power0(s):
        it2 = it2m(s); nr = nrf(s)
        z, y, KT = s.m0t, s.zo, s.m0KT
        s.k("rt_fill_rand", s.capN, (i32(0), P(s.LM), P(z)), b=256)
        for it in range(s.mg_pow):
            s.k("rt_mg_pw0", 1, (i32(0), P(s.SCM)), b=1)
            s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(3), P(s.mn), f32(0), P(s.sk), P(s.pm), *nr, P(None), P(s.W), P(None), P(None), P(z), P(s.m0u), P(None)), b=256)
            expand0(s, z, s.m0u)
            s.mg_Az(0, z, KT, tz_ready=True, bar=True)
            s.k("xp_az0", s.capN, (P(s.IP), act(s), P(s.meff), it2, P(s.sk), P(s.pm), *nr, P(z), P(KT), P(s.xpT)), b=256)
            fold0(s, s.xpT)
            s.k("xp_pwy0", s.capN, (P(s.IP), act(s), P(s.w8b), P(s.Dinv), P(z), P(s.xpT), P(y), P(s.SCM)), b=256)
            s.k("rt_mg_pw2", s.capN, (i32(0), P(s.LM), P(y), P(z), P(s.SCM), P(s.W), f32(s.mg_omega), i32(1 if it == s.mg_pow - 1 else 0)), b=256)
        ST["npw0"] += 1

    def powerl(s, l):
        if not getattr(s, "_xpcl", False):
            classify(s)
        it2 = it2m(s)
        L, X = s.ML[l - 1], s.XL[l - 1]
        z, y, KT, cn = L["z"], L["y"], L["KT"], L["capn"]
        s.k("rt_fill_rand", cn, (i32(l), P(s.LM), P(z)), b=256)
        for it in range(s.mg_pow):
            s.k("rt_mg_pw0", 1, (i32(l), P(s.SCM)), b=1)
            expandl(s, l, z)
            s.mg_Az(l, z, KT, bar=True)
            s.k("xl_az", cn, (i32(l), P(s.LM), P(L["mc"]), it2, P(z), P(KT), P(X["tmp"])), b=256)
            foldl(s, l, X["tmp"])
            s.k("xl_pwy", cn, (i32(l), P(s.LM), P(X["base"]), P(X["Dt"]), P(z), P(X["tmp"]), P(y), P(s.SCM)), b=256)
            s.k("rt_mg_pw2", cn, (i32(l), P(s.LM), P(y), P(z), P(s.SCM), P(s.W), f32(s.mg_omega), i32(1 if it == s.mg_pow - 1 else 0)), b=256)
        ST["npwl"] += 1

    def mg_power(s, l):
        if not on(s):
            return opw(s, l)
        if l == 0:
            return power0(s)
        if coarse(s):
            return powerl(s, l)
        return opw(s, l)

    # ---------------- V 순환
    def kb0(s):
        return s.bKn if (getattr(s, "bar", False) and s.mgbar and not getattr(s, "_pkc", None)) else None

    def vc0(s, r, z):
        it2 = it2m(s); nr = nrf(s)
        KT, KB = s.m0KT, kb0(s)
        node = lambda mode, rr, kt, zz, rs: s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(mode), P(s.meff), it2, P(s.sk), P(s.pm), *nr, P(s.Dinv), P(s.W),
                                                                       P(rr), P(kt), P(zz), P(rs), P(KB)), b=256)
        node(6, r, None, z, s.m0u)                                                     # z = ω D̃⁻¹ r (외부 0), m0u = T z
        expand0(s, z, s.m0u)                                                           # z_o = Σ c (T z)_j
        s.mg_Az(0, z, KT, tz_ready=True); node(7, r, KT, z, s.m0u)                     # m0u = T (r − A z)
        fold0TT(s, s.m0u)                                                              # 제한 입력 = T Eᵀ (r − A z), 외부 0
        restrict(s, 0, s.m0u, s.ML[0]["r"])
        Lc = s.ML[0]
        if coarse(s):
            vcl(s, 1, Lc["r"], Lc["z"])
        else:
            s.mg_vc(1, Lc["r"], Lc["z"])                                               # 모드 2: 거친 층은 원래 V 순환
        s.k("rt_mg_prolong", s.capN, (i32(0), P(s.LM), P(Lc["z"]), P(s.m0u), i32(0)), b=256)
        if s.fuse2:
            node(9, s.m0u, None, z, s.m0u)                                             # z += T P z_c, m0u = T z
        else:
            node(5, s.m0u, None, z, None); node(3, None, None, z, s.m0u)
        expand0(s, z, s.m0u)
        s.mg_Az(0, z, KT, tz_ready=True); node(2, r, KT, z, s.xpT)                     # e = r − A z
        fold0(s, s.xpT)                                                                # e ← Eᵀ e
        s.k("xp_jac0", s.capN, (P(s.IP), act(s), P(s.w8b), P(s.Dinv), P(s.W), P(s.xpT), P(s.sk), P(s.pm), *nr, P(z), P(s.m0u)), b=256)
        expand0(s, z, s.m0u)
        ST["nvc"] += 1

    def smoothl(s, l, r, z):
        L, X = s.ML[l - 1], s.XL[l - 1]
        s.mg_Az(l, z, L["KT"])
        nodel(s, l, 2, r, L["KT"], z, X["tmp"])
        foldl(s, l, X["tmp"])
        s.k("xl_jac", L["capn"], (i32(l), P(s.LM), P(X["base"]), P(X["Dt"]), P(s.W), P(X["tmp"]), P(z)), b=256)
        expandl(s, l, z)

    def nodel(s, l, mode, rr, kt, zz, rs):
        L, X = s.ML[l - 1], s.XL[l - 1]
        mgb = getattr(s, "_mgb_on", False)
        KB = L["Kb"] if (getattr(s, "bar", False) and s.mgbar and not getattr(s, "_pkc", None) and mgb) else None
        s.k("rt_mg_node", L["capn"], (i32(l), P(s.LM), i32(mode), P(L["mc"]), it2m(s), P(s.sk), P(s.pm), *nrf(s), P(X["Dt"]), P(s.W),
                                      P(rr), P(kt), P(zz), P(rs), P(KB)), b=256)

    def vcl(s, l, r, z):
        L = s.ML[l - 1]
        foldl(s, l, r)                                                                 # 거친 층 잔차 E_lᵀ r
        nodel(s, l, 0, r, None, z, None); expandl(s, l, z)
        if l == s.nl - 1:
            for _ in range(s.mg_nc - 1):
                smoothl(s, l, r, z)
            return
        s.mg_Az(l, z, L["KT"]); nodel(s, l, 2, r, L["KT"], z, L["t"])
        foldl(s, l, L["t"])
        restrict(s, l, L["t"], s.ML[l]["r"])
        vcl(s, l + 1, s.ML[l]["r"], s.ML[l]["z"])
        s.k("rt_mg_prolong", L["capn"], (i32(l), P(s.LM), P(s.ML[l]["z"]), P(z), i32(1)), b=256)
        expandl(s, l, z)
        smoothl(s, l, r, z)

    def _mg_vc(s, l, r, z):
        if on(s) and l == 0:
            return vc0(s, r, z)
        return ovc(s, l, r, z)

    R.alloc, R.k, R.mg_build, R.mg_power, R._mg_vc = alloc, k, mg_build, mg_power, _mg_vc
    print(f"[절 CM94] wolfh13 켬 -- 확장 반영 전처리기 모드 {ST['mode']} (1 = 층 0 + 거친 층 확장, 2 = 층 0 만): 블록 대각 Eᵀ D E, 접은 잔차 야코비, 전달 Q = I T P E_c (확장 판 (κ 0) 에서만)", flush=True)
    import atexit
    atexit.register(_dump)
    _dump()


def _dump():
    try:
        if _SOLVERS:
            s = _SOLVERS[-1]
            c = s.xpc.tolist()
            ST["dev"] = dict(d0_mod=c[0], lv={l: dict(zip(("inner", "outer", "iso", "icell", "dmod"), c[8 + (l - 1) * 8: 8 + (l - 1) * 8 + 5])) for l in range(1, s.nl)})
    except Exception as e_:
        ST["dev_err"] = f"{type(e_).__name__}: {e_}"
    try:
        with open(os.path.join(ST["out"], "xpinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
