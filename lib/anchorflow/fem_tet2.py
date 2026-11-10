"""절 CM103 (사용자 11-10 지시·승인): 사면체 판 (절 CM102 T9S) 에 (1) 구조적 뒤집힘 방지 · (2) 립시츠 (유계 왜곡) 장벽.

정의 (doc/fem_realtime.md 절 CM103.0):
- 사면체 (적분점) 변형 기울기 X = F_q + A F_q, A = Σ_o y_o ⊗ ∇λ_o (사면체 안에서 상수, 실행기 요소 커널과 같은 식). det X 는 이 X 의 정확한 행렬식.
- 장벽 에너지 (적분점마다 V_q κ ψ_b(σ), κ = c (2μ + λ), σ = X 의 부호 있는 특이값):
    ψ_b = Σ_i [ b(σ_i − σ_lo; d̂_lo) + b(σ_hi − σ_i; d̂_hi) ] + b(det X; d̂_J),   b(d; d̂) = −(d − d̂)² ln(d/d̂) (0 < d < d̂, IPC 꼴), d ≥ d̂ 이면 0, d ≤ 0 이면 +∞.
  주값 기울기·헤시안·비틀림/뒤집힘 고윳값 ((g_i ± g_j)/(σ_i ± σ_j)) 을 각 요소 커널의 주값 응력 (ph·p)·주값 헤시안 D (투영 전)·λ_t·λ_f (투영 전) 에 더함
  (에너지 선탐색 판은 에너지 rt_elem·rt_elem_lsN 에도). 투영 (음수 고윳값 0) 은 원래 커널 그대로 합에 적용. 클램프 없음.
- 보폭 (닫힌 꼴): 뉴턴 반복 경계마다 (요소 커널 앞·걸음 끝 커널 앞) 직전 상태 y_a 에서 지금 y_b 로 가는 선분 y(a) = y_a + a (y_b − y_a) 에서
  모든 적분점 det X(a) = c0 + c1 a + c2 a² + c3 a³ 의 (0, 1] 첫 근의 최솟값 α (카르다노/삼각 공식, 배정밀도), a0 = min(1, 0.9 α);
  유계 왜곡은 a0, a0/2, ... (16 개) 중 '새 위반 없음' (y_a 에서 지키던 det > 0 · σ_min > σ_lo · σ_max < σ_hi 를 깨지 않음) 인 가장 큰 a.
  a < 1 이면 y ← y_a + a (y_b − y_a), η ← η_a + a (η_b − η_a) (y = ŷ + T η 가 선형이라 같은 점).
- 시작점: 걸음 첫 요소 커널 앞에서 같은 규칙을 기준점 (y = 0 즉 X = F_q, 바닥 묶인 노드는 rt_ga_y0 와 같은 사영) -> 시작점 ŷ 선분에 적용 (섞기 a).
- 기록 (--t2dg 1): 걸음 끝마다 행 하나 (det ≤ 0 수·최소 det·띠 밖 수·최소/최대 σ·장벽 에너지·참고 Hencky 에너지·반복별 보폭 자름 통계) -> t2.jsonl.
  모드 0 (대조 판) = 요소 커널을 바꾸지 않고 보폭도 적용하지 않음, 같은 기록만 (결과 불변).
"""
import hashlib
import json
import math
import os
import re

import numpy as np

NC = 16
MAXROW = 64
NR_ = 24
STEP0 = {"rt_step0", "rt_step0g", "rt_ga_step0", "ao_step0_ga"}
ENDK = {"rt_stepend", "rt_stepend_g", "rt_ga_end", "rt_ga_end_uf", "rt_ga_end_ufl", "ao_stepend_ga", "ao_stepend_th", "ao_stepend_uf", "ao_stepend_ufl", "en_stepend"}
ELEM = {"rt_elem": 8, "ao_elem": 9, "ao_elem_s": 9, "aq_elem": None}                 # 이름: 모드 인자 자리 (None = 모드 없음, 늘 기울기)

CFG = dict(mode=None, slo=None, shi=None, dlo=None, dhi=None, ja=None, c=1.0, dg=1)
INFO = dict(mode=None, mods=[], hooks={}, err=None, solvers=[])
OUT = ["."]


def defs():
    c = CFG
    return ("\n#define AF_T2 %d\n#define AF_T2_SLO %.9e\n#define AF_T2_SHI %.9e\n#define AF_T2_DLO %.9e\n#define AF_T2_DHI %.9e\n#define AF_T2_JA %.9e\n#define AF_T2_C %.9e\n"
            % (int(c["mode"]), c["slo"], c["shi"], c["dlo"], c["dhi"], c["ja"], c["c"]))


DEV = r'''
#ifndef AF_T2_DEF
#define AF_T2_DEF
#define T2_INF __longlong_as_double(0x7ff0000000000000LL)
// 절 CM103 장벽 (IPC 꼴 로그 장벽) -- b(d; d̂) = −(d − d̂)² ln(d/d̂), 0 < d < d̂
__device__ __forceinline__ double t2_b(double d, double dh, double* b1, double* b2) {
    *b1 = 0.0; *b2 = 0.0;
    if (!(d < dh)) return 0.0;
    if (!(d > 0.0)) return T2_INF;
    double r = d - dh, L = log(d/dh);
    *b1 = -(2.0*r*L + r*r/d);
    *b2 = -(2.0*L + 4.0*r/d - r*r/(d*d));
    return -r*r*L;
}
// ψ_b(σ) (κ 곱함), 기울기 g_i = ∂ψ/∂σ_i, 헤시안 D (3x3). 정의역 밖이면 +∞ 이고 g·D = 0.
__device__ __noinline__ double t2_psi(const float* Sf, double kap, double* g, double* D) {
    double S[3] = {(double)Sf[0], (double)Sf[1], (double)Sf[2]};
    double e = 0.0, b, b1, b2; bool bad = false;
    for (int k=0;k<9;k++) D[k] = 0.0;
    for (int i=0;i<3;i++) {
        g[i] = 0.0;
        b = t2_b(S[i] - AF_T2_SLO, AF_T2_DLO, &b1, &b2); if (!isfinite(b)) bad = true; e += b; g[i] += b1; D[i*4] += b2;
        b = t2_b(AF_T2_SHI - S[i], AF_T2_DHI, &b1, &b2); if (!isfinite(b)) bad = true; e += b; g[i] -= b1; D[i*4] += b2;
    }
    double J = S[0]*S[1]*S[2];
    b = t2_b(J, AF_T2_JA, &b1, &b2); if (!isfinite(b)) bad = true; e += b;
    double cj[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    for (int i=0;i<3;i++) { g[i] += b1*cj[i]; for (int j=0;j<3;j++) D[i*3+j] += b2*cj[i]*cj[j] + ((i != j) ? b1*S[3-i-j] : 0.0); }
    if (bad || !isfinite(e)) { for (int i=0;i<3;i++) g[i] = 0.0; for (int k=0;k<9;k++) D[k] = 0.0; return T2_INF; }
    for (int i=0;i<3;i++) g[i] *= kap;
    for (int k=0;k<9;k++) D[k] *= kap;
    return kap*e;
}
__device__ __forceinline__ double t2_be(const float* S, double kap) { double g[3], D[9]; return t2_psi(S, kap, g, D); }
__device__ __forceinline__ void t2_bp(const float* S, double kap, float* p) { double g[3], D[9]; t2_psi(S, kap, g, D); for (int i=0;i<3;i++) p[i] += (float)g[i]; }
__device__ __forceinline__ void t2_bD(const float* S, double kap, float D_[3][3]) { double g[3], D[9]; t2_psi(S, kap, g, D); for (int i=0;i<3;i++) for (int j=0;j<3;j++) D_[i][j] += (float)D[i*3+j]; }
__device__ __forceinline__ void t2_btf(const float* S, double kap, int i, int j, float* lt, float* lf) {
    double g[3], D[9]; t2_psi(S, kap, g, D);
    double si = S[i], sj = S[j];
    double den = si + sj; if (fabs(den) > 1e-12) *lt += (float)((g[i] + g[j])/den);
    double dd = si - sj; *lf += (fabs(dd) > 1e-6*(fabs(si) + fabs(sj))) ? (float)((g[i] - g[j])/dd) : (float)(D[i*4] - D[i*3+j]);
}
#endif
'''

KER = r'''
// ---- 절 CM103 보폭·시작점·기록 커널 ----
__device__ __forceinline__ void t2_X(int q, const int* IP, const int* cb, const float* BG, const float* Fq, const float* y, double* X) {
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    double F[9]; for (int k=0;k<9;k++) F[k] = (double)Fq[(size_t)q*9+k];
    double A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o];
        for (int a=0;a<3;a++) { double ya = (double)y[(size_t)id*3+a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
}
__device__ __forceinline__ double t2_det(const double* A) { return A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]); }
__device__ __forceinline__ void t2_cof(const double* A, double* C) {
    C[0] = A[4]*A[8]-A[5]*A[7]; C[1] = A[5]*A[6]-A[3]*A[8]; C[2] = A[3]*A[7]-A[4]*A[6];
    C[3] = A[2]*A[7]-A[1]*A[8]; C[4] = A[0]*A[8]-A[2]*A[6]; C[5] = A[1]*A[6]-A[0]*A[7];
    C[6] = A[1]*A[5]-A[2]*A[4]; C[7] = A[2]*A[3]-A[0]*A[5]; C[8] = A[0]*A[4]-A[1]*A[3];
}
__device__ __forceinline__ double t2_pv(double c0, double c1, double c2, double c3, double t) { return ((c3*t + c2)*t + c1)*t + c0; }
__device__ __forceinline__ double t2_pol(double c0, double c1, double c2, double c3, double r) {   // 뉴턴 다듬기 2 번
    for (int k=0;k<2;k++) { double p = t2_pv(c0,c1,c2,c3,r), dp = (3.0*c3*r + 2.0*c2)*r + c1; if (dp != 0.0 && isfinite(p/dp)) r -= p/dp; }
    return r;
}
// det(A + tB) = c0 + c1 t + c2 t² + c3 t³ 의 (0, 1] 첫 근 (없으면 2). 닫힌 꼴 (삼각·카르다노·근의 공식) + 수치 확인 (p(1) ≤ 0 인데 근을 못 찾으면 이분법).
__device__ double t2_root(double c0, double c1, double c2, double c3) {
    double sc = fabs(c0) + fabs(c1) + fabs(c2) + fabs(c3), best = 2.0;
    double r[3]; int nr = 0;
    if (fabs(c3) > 1e-12*sc) {
        double a = c2/c3, b = c1/c3, c = c0/c3;
        double Q = (a*a - 3.0*b)/9.0, R = (2.0*a*a*a - 9.0*a*b + 27.0*c)/54.0;
        if (R*R < Q*Q*Q) { double th = acos(fmin(fmax(R/sqrt(Q*Q*Q), -1.0), 1.0)), sq = -2.0*sqrt(Q);
            r[0] = sq*cos(th/3.0) - a/3.0; r[1] = sq*cos((th + 2.0*3.14159265358979323846)/3.0) - a/3.0; r[2] = sq*cos((th - 2.0*3.14159265358979323846)/3.0) - a/3.0; nr = 3; }
        else { double A_ = -copysign(cbrt(fabs(R) + sqrt(R*R - Q*Q*Q)), R); double B_ = (A_ != 0.0) ? Q/A_ : 0.0; r[0] = A_ + B_ - a/3.0; nr = 1; }
    } else if (fabs(c2) > 1e-12*sc) {
        double D = c1*c1 - 4.0*c2*c0;
        if (D >= 0.0) { double qq = -0.5*(c1 + copysign(sqrt(D), c1)); r[0] = qq/c2; r[1] = (qq != 0.0) ? c0/qq : 2.0; nr = 2; }
    } else if (fabs(c1) > 1e-12*sc) { r[0] = -c0/c1; nr = 1; }
    for (int k=0;k<nr;k++) { double t = t2_pol(c0,c1,c2,c3,r[k]); if (t > 0.0 && t <= 1.0 && t < best) best = t; }
    if (best > 1.0 && !(t2_pv(c0,c1,c2,c3,1.0) > 0.0)) {                      // 수치 확인: 끝점이 뒤집혔는데 근을 못 찾음 -> 첫 부호 바뀜을 표본 + 이분법
        double lo = 0.0, hi = 1.0;
        for (int k=1;k<=64;k++) { double t = k/64.0; if (!(t2_pv(c0,c1,c2,c3,t) > 0.0)) { hi = t; lo = (k-1)/64.0; break; } }
        for (int k=0;k<50;k++) { double m = 0.5*(lo + hi); if (t2_pv(c0,c1,c2,c3,m) > 0.0) lo = m; else hi = m; }
        best = hi;
    }
    return best;
}
__global__ void __launch_bounds__(128) t2_cap(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* ya, const float* yb, int* TI) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    double A[9], Xb[9], B[9], CA[9], CB[9];
    t2_X(q, IP, cb, BG, Fq, ya, A); t2_X(q, IP, cb, BG, Fq, yb, Xb);
    for (int k=0;k<9;k++) B[k] = Xb[k] - A[k];
    double c0 = t2_det(A);
    if (!(c0 > 0.0)) { atomicAdd(&TI[2], 1); return; }
    t2_cof(A, CA); t2_cof(B, CB);
    double c1 = 0.0, c2 = 0.0; for (int k=0;k<9;k++) { c1 += CA[k]*B[k]; c2 += CB[k]*A[k]; }
    double c3 = t2_det(B);
    double t = t2_root(c0, c1, c2, c3);
    if (t <= 1.0) atomicMin(&TI[0], __float_as_int((float)t));
}
__device__ __forceinline__ void t2_sv(const double* Xd, float* S) { float X[9], U[9], V[9]; for (int k=0;k<9;k++) X[k] = (float)Xd[k]; ssvdf(X, U, S, V); }
__global__ void __launch_bounds__(128) t2_band(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* ya, const float* yb, int* TI, int nc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    double A[9], Xb[9], X[9]; float S[3];
    t2_X(q, IP, cb, BG, Fq, ya, A); t2_X(q, IP, cb, BG, Fq, yb, Xb);
    t2_sv(A, S);
    bool od = (double)S[0]*S[1]*S[2] > 0.0, ol = S[2] > (float)AF_T2_SLO, oh = S[0] < (float)AF_T2_SHI;
    if (!(ol && oh)) atomicAdd(&TI[3], 1);                                   // 기준점이 이미 띠 밖인 점 수
    float a = fminf(1.f, 0.9f*__int_as_float(TI[0]));
    int kq = nc;
    for (int k=0;k<nc;k++) {
        for (int i=0;i<9;i++) X[i] = A[i] + (double)a*(Xb[i] - A[i]);
        t2_sv(X, S);
        bool ok = (!od || (double)S[0]*S[1]*S[2] > 0.0) && (!ol || S[2] > (float)AF_T2_SLO) && (!oh || S[0] < (float)AF_T2_SHI);
        if (ok) { kq = k; break; }
        a *= 0.5f;
    }
    if (kq > 0) atomicMax(&TI[1], kq);
}
// ACC: 0 반복 확인 수 · 1 보폭 자른 수 (a < 1) · 2 최소 a · 3 시작점 섞은 수 · 4 시작점 최소 a · 5 det 근으로 자른 수 · 6 띠로 더 줄인 수 · 7 실현 가능 후보 없음
//      8 시작점 기준이 뒤집힌 점 최대 · 9 관성 예측 기준이 (det·띠) 어겨 y = 0 기준으로 섞은 수 · 10·11 걸음 시작 F_q (y 0) 의 det ≤ 0 · 띠 밖 점 수
// start: 0 반복 경계, 1 시작점 (기준 = 걸음 시작 관성 예측 Ũ), 2 시작점 대체 (기준 = y 0; 1 의 기준에 뒤집힌 점이 있을 때만 씀)
__global__ void t2_pick(int* TI, float* T, double* ACC, int start, int apply, int nc) {
    if (start == 2) { ACC[10] = fmax(ACC[10], (double)TI[2]); ACC[11] = fmax(ACC[11], (double)TI[3]); }   // y 0 기준 (= 걸음 시작 F_q) 의 det ≤ 0 · 띠 밖 점 수
    if (start == 2 && T[2] == 0.f) { T[0] = 1.f; TI[0] = __float_as_int(2.f); TI[1] = 0; TI[2] = 0; TI[3] = 0; return; }
    float ad = __int_as_float(TI[0]); float a0 = fminf(1.f, 0.9f*ad); int k = TI[1];
    float a = (k >= nc) ? 0.f : ldexpf(a0, -k);
    bool use = true;
    if (start == 1) { T[2] = (float)(TI[2] + TI[3]); use = (TI[2] == 0 && TI[3] == 0); }   // 관성 예측 기준이 det·띠 어느 쪽이든 어기면 y 0 기준으로
    T[0] = (apply && use) ? a : 1.f; T[1] = a;
    if (use) {
        ACC[0] += 1.0; if (a < 1.f) ACC[1] += 1.0; ACC[2] = fmin(ACC[2], (double)a);
        if (a0 < 1.f) ACC[5] += 1.0; if (k > 0) ACC[6] += 1.0; if (k >= nc) ACC[7] += 1.0;
        if (start) { if (a < 1.f) ACC[3] += 1.0; ACC[4] = fmin(ACC[4], (double)a); ACC[8] = fmax(ACC[8], (double)TI[2]); }
        if (start == 2) ACC[9] += 1.0;
    }
    TI[0] = __float_as_int(2.f); TI[1] = 0; TI[2] = 0; TI[3] = 0;
}
__global__ void t2_apply(const int* IP, const float* T, const float* ya, const float* ea, float* y, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float a = T[0]; if (!(a < 1.f)) return;
    for (int c=0;c<3;c++) { size_t i = (size_t)n*3+c; y[i] = ya[i] + a*(y[i] - ya[i]); eta[i] = ea[i] + a*(eta[i] - ea[i]); }
}
__global__ void t2_copy(const int* IP, const float* y, const float* eta, float* ya, float* ea) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int c=0;c<3;c++) { size_t i = (size_t)n*3+c; ya[i] = y[i]; ea[i] = eta[i]; }
}
// 기준점 (rt_ga_y0 와 같은 식, 목표 Ur (없으면 0)): y = ŷ + T η 꼴로 목표에 가장 가까운 점 (고정 노드 pm 2 는 지금 값 그대로)
__global__ void t2_ref(const int* IP, const float* Ur, const float* yh, const float* sk, const int* pm, float n0, float n1, float n2, float* ea, float* ya) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int p = pm[n];
    if (p == 2) return;
    float s = sk[n];
    float isk = (s > 1e-6f) ? 1.f/s : 0.f;
    float dd[3]; for (int a=0;a<3;a++) dd[a] = (Ur ? Ur[n*3+a] : 0.f) - yh[n*3+a];
    float e[3];
#if defined(AF_FSTK) || defined(AF_FUP)
    if (p < 0) { float sn_ = __int_as_float(-p - 1); float isn_ = (sn_ > 1e-6f) ? 1.f/sn_ : 0.f;
        float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = isn_*dn*nn[a] + isk*(dd[a] - dn*nn[a]); } else
#endif
    if (p == 0) { for (int a=0;a<3;a++) e[a] = isk*dd[a]; }
    else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = dn*nn[a] + isk*(dd[a] - dn*nn[a]); }
    float t[3]; Tapply(s, p, n0, n1, n2, e, t);
    for (int a=0;a<3;a++) { ea[n*3+a] = e[a]; ya[n*3+a] = yh[n*3+a] + t[a]; }
}
__global__ void t2_acc0(double* ACC) { for (int k=0;k<16;k++) ACC[k] = 0.0; ACC[2] = 1.0; ACC[4] = 1.0; }
__device__ __forceinline__ int t2_key(float f) { int b = __float_as_int(f); return (b >= 0) ? b : (b ^ 0x7fffffff); }
__global__ void t2_rowinit(const int* TI, double* R, int* RI, int maxrow) {
    int r = TI[10]; if (r >= maxrow) r = maxrow - 1;
    for (int k=0;k<24;k++) R[r*24+k] = 0.0;
    RI[r*4] = 0x7fffffff; RI[r*4+1] = 0x7fffffff; RI[r*4+2] = (int)0x80000000; RI[r*4+3] = (int)0x80000000;
}
__device__ __forceinline__ double t2_ws(double v) { for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float t2_wmin(float v) { for (int o=16;o>0;o>>=1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, o)); return v; }
__device__ __forceinline__ float t2_wmax(float v) { for (int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o)); return v; }
// 걸음 끝 기록 (결과 불변): R 행 = 0 점 수 · 1 det ≤ 0 · 2 띠 밖 · 3 σ_min < σ_lo · 4 σ_max > σ_hi · 5 장벽 에너지 (유한 합) · 6 장벽 +∞ 점 · 7 Hencky 참고 에너지 · 8 장벽 활성 점
//   RI = 최소 det · 최소 σ · 최대 σ (순서 보존 키) · 최대 ψ_b 밀도 키
__global__ void __launch_bounds__(128) t2_diag(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, const int* TI, double* R, int* RI, int maxrow) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && Vq[q < IP[12] ? q : 0] > 0.f;
    double c[9] = {0,0,0,0,0,0,0,0,0}; float dmin = 3e38f, smin = 3e38f, smax = -3e38f, pmax = -3e38f;
    if (on) {
        double X[9]; float S[3]; t2_X(q, IP, cb, BG, Fq, y, X); t2_sv(X, S);
        double vq = Vq[q], J = (double)S[0]*S[1]*S[2], kap = AF_T2_C*(2.0*(double)mu + (double)lam);
        c[0] = 1.0; c[1] = (J <= 0.0) ? 1.0 : 0.0;
        bool lo = S[2] < (float)AF_T2_SLO, hi = S[0] > (float)AF_T2_SHI;
        c[2] = (lo || hi) ? 1.0 : 0.0; c[3] = lo ? 1.0 : 0.0; c[4] = hi ? 1.0 : 0.0;
        double eb = t2_be(S, kap);
        if (isfinite(eb)) { c[5] = vq*eb; pmax = (float)eb; } else { c[6] = 1.0; pmax = 3e38f; }
        if (J > 0.0) { double l0 = log((double)fabsf(S[0])), l1 = log((double)fabsf(S[1])), l2 = log((double)fabsf(S[2])), lj = l0 + l1 + l2;
            c[7] = vq*((double)mu*(l0*l0 + l1*l1 + l2*l2) + 0.5*(double)lam*lj*lj); }
        c[8] = (!isfinite(eb) || eb > 0.0) ? 1.0 : 0.0;
        dmin = (float)J; smin = S[2]; smax = S[0];
    }
    int r = TI[10]; if (r >= maxrow) r = maxrow - 1;
    for (int k=0;k<9;k++) { double v = t2_ws(c[k]); if ((threadIdx.x & 31) == 0 && v != 0.0) atomicAdd(&R[r*24+k], v); }
    dmin = t2_wmin(dmin); smin = t2_wmin(smin); smax = t2_wmax(smax); pmax = t2_wmax(pmax);
    if ((threadIdx.x & 31) == 0) { atomicMin(&RI[r*4], t2_key(dmin)); atomicMin(&RI[r*4+1], t2_key(smin)); atomicMax(&RI[r*4+2], t2_key(smax)); atomicMax(&RI[r*4+3], t2_key(pmax)); }
}
// 걸음 끝 갱신 뒤 U (입자 F 갱신에 쓰이는 걸음 끝 변위) 의 X = F_q + ∇U F_q: det ≤ 0 · 띠 밖 점 수 -> R 21·22
__global__ void __launch_bounds__(128) t2_diagU(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* U, const int* TI, double* R, int maxrow) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && Vq[q < IP[12] ? q : 0] > 0.f;
    double c0 = 0.0, c1 = 0.0;
    if (on) { double X[9]; float S[3]; t2_X(q, IP, cb, BG, Fq, U, X); t2_sv(X, S);
        c0 = ((double)S[0]*S[1]*S[2] <= 0.0) ? 1.0 : 0.0; c1 = (S[2] < (float)AF_T2_SLO || S[0] > (float)AF_T2_SHI) ? 1.0 : 0.0; }
    int r = TI[10]; if (r >= maxrow) r = maxrow - 1;
    c0 = t2_ws(c0); c1 = t2_ws(c1);
    if ((threadIdx.x & 31) == 0) { if (c0 != 0.0) atomicAdd(&R[r*24+21], c0); if (c1 != 0.0) atomicAdd(&R[r*24+22], c1); }
}
__global__ void t2_rowfin(int* TI, const double* ACC, double* R, int maxrow) {
    int r = TI[10]; if (r >= maxrow) r = maxrow - 1;
    for (int k=0;k<12;k++) R[r*24+9+k] = ACC[k];
    TI[11] += 1; if (TI[10] < maxrow - 1) TI[10] += 1; else TI[12] += 1;
}
'''


def _bodies(src):
    out = []
    for m in re.finditer(r"(?:__global__|__device__)[\w\s]*?void\s+(?:__launch_bounds__\(\d+\)\s+)?(\w+)\s*\(", src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        d, j = 0, i
        while j < len(src):
            if src[j] == "{":
                d += 1
            elif src[j] == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        if j < len(src):
            out.append((m.group(1), i, j + 1))
    return out


KAP_ML = "(AF_T2_C*(2.0*(double)mu + (double)lam))"
KAP_PRM = "(AF_T2_C*(2.0*(double)prm[0] + (double)prm[1]))"
R_LT = r"(?<![\w.])lt = isfinite\(lt\) \? fmaxf\(lt, 0\.f\) : 0\.f;"
R_JAC = r"(?<![\w.])jac3f\(D, Ev\);"
R_VQ = r"(?<![\w.])float vq = on \? Vq\[qq\] : 0\.f;"


def _rules(name):
    """함수 이름 -> [(규칙 이름, 정규식, 바꿈)]."""
    base = re.sub(r"_c$", "", name)
    if base == "rt_elem":
        k = KAP_ML
        return [("E", r"(?<![\w.])blocksum_add\(e, E\);", "if (on && vq > 0.f) e += (double)vq*t2_be(S, %s); blocksum_add(e, E);" % k),
                ("P", r"(?<![\w.])float P\[9\];", "t2_bp(S, %s, ph); float P[9];" % k),
                ("D", R_JAC, "t2_bD(S, %s, D); jac3f(D, Ev);" % k),
                ("T", R_LT, "t2_btf(S, %s, i, j, &lt, &lf); lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f;" % k)]
    if base == "rt_elem_lsN":
        return [("E", r"(?<![\w.])blocksum_add\(en, &SC\[40\+i\]\);", "if (on && vq > 0.f) en += (double)vq*t2_be(S, %s); blocksum_add(en, &SC[40+i]);" % KAP_ML)]
    if base in ("ao_elem", "ao_elem_s", "aq_elem"):
        k = KAP_PRM if base == "aq_elem" else KAP_ML
        return [("P", R_VQ, "t2_bp(S, %s, p); float vq = on ? Vq[qq] : 0.f;" % k),
                ("D", R_JAC, "t2_bD(S, %s, D); jac3f(D, Ev);" % k),
                ("T", R_LT, "t2_btf(S, %s, i, j, &lt, &lf); lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f;" % k)]
    return None


def transform(src):
    """해 모듈 (ssvdf·qnodes·Tapply 있음) 이면: 정의 + 장벽 함수 앞에, 보폭·기록 커널 끝에; 모드 1 이면 요소 커널에 장벽 몫."""
    if not ("void ssvdf(" in src and "void qnodes(" in src and "void Tapply(" in src and 'extern "C" {' in src):
        return src, None
    cnt = {}
    s = src
    if CFG["mode"] == 1:
        for name, a, b in reversed(_bodies(s)):
            R = _rules(name)
            if R is None:
                continue
            body = s[a:b]
            for rn, pat, rep in R:
                body, n = re.subn(pat, rep, body)
                cnt[name + ":" + rn] = cnt.get(name + ":" + rn, 0) + n
            s = s[:a] + body + s[b:]
    k = s.index('extern "C" {') + len('extern "C" {')
    s = s[:k] + defs() + DEV + s[k:]
    e = s.rstrip()
    assert e.endswith("}"), "모듈 끝이 extern C 닫는 괄호가 아님"
    j = s.rindex("}")
    s = s[:j] + KER + "\n" + s[j:]
    cnt["ker"] = 1
    return s, cnt


def _dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], "tet2info.json"), "w"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


# ---------------- 파이썬 훅 (해 인스턴스 s.k · s.step) ----------------
def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


class Hook:
    def __init__(h, s):
        import torch
        h.s, h.torch = s, torch
        h.k0 = s.k
        h.phase = 0
        h.busy = False
        h.buf = None
        h.mu = h.lam = None
        h.f = open(os.path.join(OUT[0], "t2.jsonl"), "a") if CFG["dg"] else None
        h.ncall = 0

    def _alloc(h):
        s, torch = h.s, h.torch
        ny = s.y.numel()
        if h.buf is not None and h.buf["ya"].numel() == ny:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[절 CM103] 캡처 중 버퍼 할당 (데움 없이 캡처?)")
        dev = s.y.device
        TI = torch.zeros(16, dtype=torch.int32, device=dev); TI[0] = int(np.float32(2.0).view(np.int32))
        old = h.buf
        h.buf = dict(ya=torch.zeros_like(s.y), ea=torch.zeros_like(s.eta), yi=torch.zeros_like(s.y), ei=torch.zeros_like(s.eta), TI=TI, T=torch.ones(4, dtype=torch.float32, device=dev),
                     ACC=torch.zeros(16, dtype=torch.float64, device=dev), R=torch.zeros(MAXROW * NR_, dtype=torch.float64, device=dev),
                     RI=torch.zeros(MAXROW * 4, dtype=torch.int32, device=dev))
        if old is not None:                                                                 # 다시 할당 (용량 2 배): 행 버퍼 이어서
            h.buf["TI"].copy_(old["TI"]); h.buf["R"].copy_(old["R"]); h.buf["RI"].copy_(old["RI"]); h.buf["ACC"].copy_(old["ACC"])
        if h.mu is None:
            mu, lam = getattr(s, "mu", None), getattr(s, "lam", None)
            prm = getattr(s, "prm", None)
            p01 = [float(prm[0]), float(prm[1])] if (prm is not None and hasattr(prm, "numel") and prm.numel() >= 2) else None
            if mu is None and p01 is not None:
                mu, lam = p01
            h.mu, h.lam = float(mu), float(lam)
            INFO["solvers"].append(dict(cls=type(s).__name__, mu=h.mu, lam=h.lam, prm01=p01, kappa=CFG["c"] * (2 * h.mu + h.lam)))
            _dump()

    def L(h, name, n, args, b=128):
        h.busy = True
        try:
            h.k0(name, n, args, b)
        finally:
            h.busy = False

    def nq(h):
        return int(h.s.Fq.numel() // 9)

    def nn(h):
        return int(h.s.y.numel() // 3)

    def check(h, start):
        s, B = h.s, h.buf
        i32, f32 = np.int32, np.float32
        ap = 1 if CFG["mode"] == 1 else 0
        fl = getattr(s, "floor", None)
        nr = fl["NR"] if fl else [0.0, 0.0, 1.0]
        for st in ((1, 2) if start else (0,)):
            if st:                                                                          # 기준점: 1 = 걸음 시작 관성 예측 (yi), 2 = y 0
                h.L("t2_copy", h.nn(), (_P(s.IP), _P(s.y), _P(s.eta), _P(B["ya"]), _P(B["ea"])), 256)
                h.L("t2_ref", h.nn(), (_P(s.IP), _P(B["yi"]) if st == 1 else _P(None), _P(s.yh), _P(s.sk), _P(s.pm), f32(nr[0]), f32(nr[1]), f32(nr[2]), _P(B["ea"]), _P(B["ya"])), 256)
            qa = (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(B["ya"]), _P(s.y), _P(B["TI"]))
            h.L("t2_cap", h.nq(), qa, 128)
            h.L("t2_band", h.nq(), qa + (i32(NC),), 128)
            h.L("t2_pick", 1, (_P(B["TI"]), _P(B["T"]), _P(B["ACC"]), i32(st), i32(ap), i32(NC)), 1)
            if ap:
                h.L("t2_apply", h.nn(), (_P(s.IP), _P(B["T"]), _P(B["ya"]), _P(B["ea"]), _P(s.y), _P(s.eta)), 256)

    def snap(h):
        s, B = h.s, h.buf
        h.L("t2_copy", h.nn(), (_P(s.IP), _P(s.y), _P(s.eta), _P(B["ya"]), _P(B["ea"])), 256)

    def diag(h):
        s, B = h.s, h.buf
        i32, f32 = np.int32, np.float32
        h.L("t2_rowinit", 1, (_P(B["TI"]), _P(B["R"]), _P(B["RI"]), i32(MAXROW)), 1)
        h.L("t2_diag", h.nq(), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(s.y), f32(h.mu), f32(h.lam), _P(B["TI"]), _P(B["R"]), _P(B["RI"]), i32(MAXROW)), 128)

    def diag_end(h, Uptr):
        s, B = h.s, h.buf
        i32 = np.int32
        h.L("t2_diagU", h.nq(), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), Uptr, _P(B["TI"]), _P(B["R"]), i32(MAXROW)), 128)
        h.L("t2_rowfin", 1, (_P(B["TI"]), _P(B["ACC"]), _P(B["R"]), i32(MAXROW)), 1)

    def k(h, name, n, args, b=128):
        if h.busy:
            return h.k0(name, n, args, b)
        s = h.s
        hk = INFO["hooks"]
        try:
            if name in STEP0:
                h.k0(name, n, args, b)
                h._alloc()
                h.L("t2_acc0", 1, (_P(h.buf["ACC"]),), 1)
                h.L("t2_copy", h.nn(), (_P(s.IP), _P(s.y), _P(s.eta), _P(h.buf["yi"]), _P(h.buf["ei"])), 256)   # 걸음 시작 관성 예측 Ũ (시작점 기준)
                h.phase = 1
                hk["step0:" + name] = hk.get("step0:" + name, 0) + 1
                return
            if name in ELEM and h.phase:
                mi = ELEM[name]
                ony = int(args[5]) == s.y.data_ptr()
                md = int(args[mi]) if mi is not None else 1
                if ony and (md & 1):
                    h._alloc()
                    h.check(h.phase == 1)
                    h.snap()
                    hk[("start:" if h.phase == 1 else "iter:") + name] = hk.get(("start:" if h.phase == 1 else "iter:") + name, 0) + 1
                    h.phase = 2
            elif name in ENDK and h.phase == 2:
                h.check(False)
                h.phase = 0
                hk["end:" + name] = hk.get("end:" + name, 0) + 1
                if CFG["dg"]:
                    h.diag()
                    h.k0(name, n, args, b)
                    h.diag_end(args[7])                                                     # 걸음 끝 갱신 뒤 U (rt_ga_end*·ao_stepend*: 일곱째 인자)
                    return
        except Exception as e_:
            INFO["err"] = f"{name}: {e_}"[:600]; _dump(); raise
        return h.k0(name, n, args, b)

    def read(h, t):
        """프레임 (s.step 한 번) 뒤 걸음 행 읽기 -> t2.jsonl."""
        if h.buf is None or h.f is None:
            return
        B = h.buf
        TI = B["TI"].cpu().numpy()
        nr = int(TI[10]) + (1 if TI[12] > 0 else 0)
        nr = min(nr, MAXROW)
        R = B["R"][:nr * NR_].view(nr, NR_).cpu().numpy() if nr else np.zeros((0, NR_))
        RI = B["RI"][:nr * 4].view(nr, 4).cpu().numpy() if nr else np.zeros((0, 4), np.int32)

        def fk(k):
            k = int(k)
            b = k if k >= 0 else (k ^ 0x7fffffff)
            return float(np.array([b], dtype=np.int32).view(np.float32)[0])
        rows = []
        for i in range(nr):
            r = R[i]
            rows.append(dict(nq=int(r[0]), ndet=int(r[1]), nband=int(r[2]), nlo=int(r[3]), nhi=int(r[4]), Eb=float(r[5]), ninf=int(r[6]), Eh=float(r[7]), nact=int(r[8]),
                             nchk=int(r[9]), ncap=int(r[10]), amin=float(r[11]), nstart=int(r[12]), astart=float(r[13]), ndetcap=int(r[14]), nbandcap=int(r[15]),
                             nnofeas=int(r[16]), ninv0=int(r[17]), nfb=int(r[18]), nfq_det=int(r[19]), nfq_band=int(r[20]), nU_det=int(r[21]), nU_band=int(r[22]), dmin=fk(RI[i][0]), smin=fk(RI[i][1]), smax=fk(RI[i][2]), pbmax=fk(RI[i][3])))
        h.f.write(json.dumps(dict(c=h.ncall, t=float(t), steps=rows, over=int(TI[12]))) + "\n"); h.f.flush()
        h.ncall += 1
        B["TI"][10:13].zero_()
        _dump()


def _install_hook(s):
    if getattr(s, "_t2h", None) is not None:
        return
    h = Hook(s)
    s._t2h = h
    s.k = h.k
    if CFG["dg"] and hasattr(s, "step"):
        st0 = s.step

        def step(t, *a, **kw):
            r = st0(t, *a, **kw)
            h.read(t)
            return r
        s.step = step


def install(out, mode, band, c=1.0, dg=1):
    import cupy as cp
    OUT[0] = out
    CFG.update(mode=int(mode), slo=band[0], shi=band[1], dlo=band[2], dhi=band[3], ja=band[4], c=float(c), dg=int(dg))
    INFO.update(mode=int(mode), cfg=dict(CFG))
    try:
        os.remove(os.path.join(out, "t2.jsonl"))
    except OSError:
        pass
    base = cp.RawModule
    if not getattr(base, "_tet2", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            if isinstance(code, str):
                try:
                    code2, cnt = transform(code)
                except Exception as e_:
                    INFO["err"] = str(e_)[:600]; _dump(); raise
                if cnt is not None:
                    INFO["mods"].append(dict(sha=hashlib.sha1(code.encode()).hexdigest()[:12], rules=cnt)); _dump()
                    if "code" in kw:
                        kw["code"] = code2
                    else:
                        a[0] = code2
            return base(*a, **kw)
        RawModule._tet2 = True
        cp.RawModule = RawModule
    from . import fem_an_rt as ART
    mods = [ART]
    for nm in ("fem_an_rt_v0", "fem_ao_rt", "fem_mat_rt"):
        try:
            mods.append(__import__("anchorflow." + nm, fromlist=[nm]))
        except Exception as e_:
            INFO.setdefault("imp_err", []).append(f"{nm}: {e_}"[:200])
    for M in mods:
        for cn in ("RTSolver", "AORTSolver", "RTAdapt", "MatRT"):
            C = getattr(M, cn, None)
            if C is None or getattr(C.__init__, "_tet2", False):
                continue
            f0 = C.__init__

            def init(s, *a, _f0=f0, **kw):
                _f0(s, *a, **kw)
                _install_hook(s)
            init._tet2 = True
            C.__init__ = init
    print(f"[절 CM103] tet2 켬 -- 모드 {mode} ({'장벽·보폭' if mode == 1 else '기록만 (결과 불변)'}), 띠 σ [{band[0]:.4f}, {band[1]:.4f}] 활성 폭 {band[2]:.4f}·{band[3]:.4f}, "
          f"det 활성 {band[4]:.4f}, κ = {c} (2μ+λ), 기록 {dg}", flush=True)
    _dump()


def summarize(path):
    """t2.jsonl -> 판 요약 (같은 t 는 마지막 시도만)."""
    L = [json.loads(l) for l in open(path)]
    by = {}
    for q in L:
        by[round(q["t"], 9)] = q
    S = [r for k in sorted(by) for r in by[k]["steps"]]
    if not S:
        return {}
    return dict(nstep=len(S), steps_det=sum(r["ndet"] > 0 for r in S), max_ndet=max(r["ndet"] for r in S), min_det=min(r["dmin"] for r in S),
                steps_band=sum(r["nband"] > 0 for r in S), max_nband=max(r["nband"] for r in S), min_s=min(r["smin"] for r in S), max_s=max(r["smax"] for r in S),
                nchk=sum(r["nchk"] for r in S), ncap=sum(r["ncap"] for r in S), amin=min(r["amin"] for r in S), nstart=sum(r["nstart"] for r in S),
                nnofeas=sum(r["nnofeas"] for r in S), nfb=sum(r.get("nfb", 0) for r in S), steps_fq=sum((r.get("nfq_det", 0) + r.get("nfq_band", 0)) > 0 for r in S), max_nfq_det=max(r.get("nfq_det", 0) for r in S), max_nfq_band=max(r.get("nfq_band", 0) for r in S), max_nU_det=max(r.get("nU_det", 0) for r in S), max_nU_band=max(r.get("nU_band", 0) for r in S), eb_share_max=max((r["Eb"] / (r["Eb"] + r["Eh"]) if (r["Eb"] + r["Eh"]) > 0 else 0.0) for r in S),
                eb_share_mean=float(np.mean([(r["Eb"] / (r["Eb"] + r["Eh"]) if (r["Eb"] + r["Eh"]) > 0 else 0.0) for r in S])), ninf=sum(r["ninf"] for r in S))
