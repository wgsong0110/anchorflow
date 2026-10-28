"""명시적 FEM -- 2 차 B-스플라인 (MPM 표준) 판 커널 (사용자 10-28 "해봐", doc/fem_realtime.md 절 M).

절 L 의 명시적 판 (선형 사면체 barycentric, 노드 질량 Σλm 이면 발산) 을 **MPM 과 같은 2 차 B-스플라인 보간 (--interp bspline2)** 으로 바꾼다.
격자 = 기존 노드 격자 (simplex.lattice(node_h, HZ_CUBE node_h) -- 정육면체 격자, 간격 dx = node_h/√2, 몸대각선 z; simplex.bspline27 와 같은 가중치):
  y = Ai (x - lo), base = floor(y - ½), f = y - base,  w = [½(1.5-f)², 0.75-(f-1)², ½(f-0.5)²] (축별 곱, 27 노드),  ∇_x w = (∂w/∂y)ᵀ Ai.
한 서브스텝 (h_s = h/N, PG p2g2p 와 같은 순서):
  P2G   m_n = Σ w m_p,  (mv)_n = Σ w m_p v_p + h_s f_n          (질량 정의 = Σ w m 그대로)
        f_n = −Σ_q V_q τ_q ∇w_n(x_q)   (τ = P(F) Fᵀ, 고정 FCR -- 사면체 판과 같은 ψ·P 식, σ 하한 0.01)
          quad particle : 적분점 = 입자 (MPM 식, 입자 F_p 로 τ_p, ∇w 는 입자 위치)
          quad cell     : 적분점 = 셀 대표 (base 가 같은 입자 = 같은 27 노드 지지 = 셀; F̄_c = Σ V_p F_p / Σ V_p, V_c = Σ V_p,
                          ∇w 는 셀 안 K 점 평균 -- K 1: 셀 중심, K 8: 2x2x2 가우스점)
        바닥 후보 입자 (프레임 시작 분할, 절 L 과 같은 정의) 는 탄성 적분점에서 빠지고 프레임 단위 바닥 에너지의 기울기를 (w, ∇w) 로 노드에 -- f -= ∂E_fl/∂δU.
  격자  v_n = ((mv)_n / m_n + h_s g) · 감쇠,  --cfg_bc 노드는 v_bc     (PG: grid_normalization_and_gravity -> add_damping_via_grid -> grid BC 와 같은 순서)
  G2P   v_p = Σ w v_n (PIC -- 기존 sub 경로와 같다),  x_p += h_s v_p,  F_p ← (I + h_s Σ v_n ⊗ ∇w_n) F_p
        (quad cell: 바닥 후보 아닌 입자는 ∇w_n 대신 그 셀의 ḡ_n -- 힘과 같은 연산자, 사면체 판에서 입자 F 를 그 사면체 ∇λ 로 갱신하는 것과 같은 꼴)
프레임 끝 바닥 후보 매핑은 절 L 의 ex_near_fin (fem_ex_cuda) 그대로.
커널은 계산 정밀도 R (float/double) 템플릿: 기준 판 = double 따로따로, 최적화 판 = float 융합 (G2P + 다음 P2G 한 커널).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import fem_rt_cuda as RTC

_BS = r'''
// ---------------- 2 차 B-스플라인 가중치 (simplex.bspline27 과 같은 식) ----------------
// gpar = [lo(3), Ai(9)] (R), ipar = [n0, n1, n2] -- 밀집 격자 상자. 밖이면 1 (base 를 상자 안으로 자른다)
template <typename R> __device__ __forceinline__ int bs_base(const R* x, const R* gpar, const int* ipar, int* b, R W[3][3], R D[3][3]) {
    int ovf = 0;
    R q0 = x[0]-gpar[0], q1 = x[1]-gpar[1], q2 = x[2]-gpar[2];
    for (int k=0;k<3;k++) {
        R y = gpar[3+k*3]*q0 + gpar[3+k*3+1]*q1 + gpar[3+k*3+2]*q2;
        int bi = (int)floor(y - (R)0.5);
        int nk = ipar[k];
        if (bi < 0) { bi = 0; ovf = 1; }
        if (bi > nk-3) { bi = nk-3; ovf = 1; }
        R f = y - (R)bi;
        b[k] = bi;
        W[0][k] = (R)0.5*((R)1.5-f)*((R)1.5-f); W[1][k] = (R)0.75-(f-(R)1)*(f-(R)1); W[2][k] = (R)0.5*(f-(R)0.5)*(f-(R)0.5);
        D[0][k] = f-(R)1.5; D[1][k] = (R)-2*(f-(R)1); D[2][k] = f-(R)0.5;
    }
    return ovf;
}
// 노드 o = (i, j, l) 의 w 와 ∇_x w
template <typename R> __device__ __forceinline__ R bs_wg(const R W[3][3], const R D[3][3], const R* Ai, int i, int j, int l, R* gx) {
    R gy0 = D[i][0]*W[j][1]*W[l][2], gy1 = W[i][0]*D[j][1]*W[l][2], gy2 = W[i][0]*W[j][1]*D[l][2];
    for (int c=0;c<3;c++) gx[c] = gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c];
    return W[i][0]*W[j][1]*W[l][2];
}
// τ = P(X) Xᵀ (고정 FCR, fem_ex ex_tet_force_body 와 같은 식)
template <typename R> __device__ __forceinline__ void bs_kirch(const R* X, R mu, R lam, R* Gm) {
    R C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    R l[3]; symeig3(C, l);
    R J = 1;
    for (int k=0;k<3;k++){ R s = sqrt(l[k] > (R)1e-12 ? l[k] : (R)1e-12); s = s > (R)0.01 ? s : (R)0.01; J *= s; }
    R Q[9], cf[9], P[9];
    polar(X, Q); cof3(X, cf);
    R dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    R sg = (dt>0)?(R)1:((dt<0)?(R)-1:(R)0);
    for (int k=0;k<9;k++) P[k] = (R)2*mu*(X[k]-Q[k]) + lam*(J-(R)1)*sg*cf[k];
    mmt3(P, X, Gm);
}
// G2P (PIC): v = Σ w v_n, x += h_s v, F <- (I + h_s Σ v_n ⊗ ∇w) F
// gK (선택, 셀 대표 적분점 판): 바닥 후보가 아닌 입자의 F 는 셀 기울기 표 ḡ_n 로 (I + h_s Σ v_n ⊗ ḡ_n) -- 힘 f_n = -V_c τ(F̄_c) ḡ_n 과 같은 연산자
// (F 를 입자 ∇w 로 갱신하면 힘·변형 연산자가 달라 강성이 비대칭 -> 어떤 N 에서도 첫 프레임 발산: 스윕 1 실측)
template <typename R, typename G> __device__ __forceinline__ void bs_g2p_body(const R* x, const float* Fp, const G* vg, const R* gpar, const int* ipar, R hs,
                                                                             R* xn, R* vn, R* Fn, int* szp, const R* gK = nullptr) {
    int b[3]; R W[3][3], D[3][3];
    if (bs_base<R>(x, gpar, ipar, b, W, D)) atomicAdd(&szp[3], 1);
    int n1 = ipar[1], n2 = ipar[2];
    const R* Ai = gpar + 3;
    R v[3] = {0,0,0}, Gv[9] = {0,0,0,0,0,0,0,0,0};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        R gx[3]; R w = bs_wg<R>(W, D, Ai, i, j, l, gx);
        if (gK) { int o = (i*3+j)*3+l; gx[0] = gK[o*3]; gx[1] = gK[o*3+1]; gx[2] = gK[o*3+2]; }
        long long n = ((long long)(b[0]+i)*n1 + (b[1]+j))*n2 + (b[2]+l);
        for (int a=0;a<3;a++) { R va = (R)vg[n*3+a]; v[a] += w*va; for (int c=0;c<3;c++) Gv[a*3+c] += va*gx[c]; }
    }
    R A[9], F[9];
    for (int k=0;k<9;k++) { A[k] = hs*Gv[k] + ((k%4==0)?(R)1:(R)0); F[k] = (R)Fp[k]; }
    mm3(A, F, Fn);
    for (int k=0;k<3;k++) { vn[k] = v[k]; xn[k] = x[k] + hs*v[k]; }
}
// P2G: 질량·운동량 + (quad 0) 입자 적분점 탄성 충격량 h_s f / (quad 1) 셀 V·VF 합 / (quad -1) 운동량만
template <typename R, typename G> __device__ __forceinline__ void bs_p2g_body(const R* x, const R* v, const R* F, R mp, R Vp, int near, int quad,
                                                                             const R* gpar, const int* ipar, R hs, R mu, R lam,
                                                                             G* gm, G* gmv, G* cV, G* cVF, int* szp) {
    int b[3]; R W[3][3], D[3][3];
    if (bs_base<R>(x, gpar, ipar, b, W, D)) atomicAdd(&szp[3], 1);
    int n1 = ipar[1], n2 = ipar[2];
    const R* Ai = gpar + 3;
    bool fp = (!near) && quad == 0;
    R Gm[9];
    if (fp) { bs_kirch<R>(F, mu, lam, Gm); for (int k=0;k<9;k++) Gm[k] *= -hs*Vp; }
    if ((!near) && quad == 1) {
        long long c = ((long long)b[0]*n1 + b[1])*n2 + b[2];
        atomicAdd(&cV[c], (G)Vp);
        for (int k=0;k<9;k++) atomicAdd(&cVF[c*9+k], (G)(Vp*F[k]));
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        R gx[3]; R w = bs_wg<R>(W, D, Ai, i, j, l, gx);
        long long n = ((long long)(b[0]+i)*n1 + (b[1]+j))*n2 + (b[2]+l);
        R wm = w*mp;
        atomicAdd(&gm[n], (G)wm);
        for (int c=0;c<3;c++) {
            R t = wm*v[c];
            if (fp) t += Gm[c*3]*gx[0] + Gm[c*3+1]*gx[1] + Gm[c*3+2]*gx[2];
            atomicAdd(&gmv[n*3+c], (G)t);
        }
    }
}

// 바닥 후보 입자의 프레임 바닥 에너지 기울기 (fem_ex ex_near_force_body 의 sub 갈래와 같은 식) -> gy[12] (변위 3 + 변위 기울기 9, (I+G)ᵀ 곱 포함)
template <typename R> __device__ __forceinline__ void bs_near_gy(int q, long long p, const float* x0c, const float* F0c, const float* cc,
                              const float* xcur, const float* Fcur, const float* F0inv, const float* mp, const float* vol, R h,
                              R p0, R p1, R p2, R n0, R n1, R n2, R eps_st, R eps_sp, R xn, R mu, R lam, R* gy) {
    R u[3], G[9], xx[3], c[3], IGt[9];
    for (int k=0;k<3;k++){ xx[k]=x0c[q*3+k]; c[k]=cc[q*3+k]; }
    for (int k=0;k<3;k++) u[k] = (R)xcur[p*3+k] - xx[k];
    R Fc[9], Fi[9]; for (int k=0;k<9;k++){ Fc[k]=Fcur[p*9+k]; Fi[k]=F0inv[q*9+k]; }
    mm3(Fc, Fi, IGt);
    for (int k=0;k<9;k++) G[k] = IGt[k] - ((k%4==0)?(R)1:(R)0);
    R pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMapT<R> M; near_map(xx, u, G, c, pt, nv, eps_st, eps_sp, xn, M);
    R Jf[9]; mm3(M.Mn, M.K1, Jf);
    R F[9]; for (int k=0;k<9;k++) F[k]=F0c[q*9+k];
    R X[9]; mm3(Jf, F, X);
    R C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    R l[3]; symeig3(C, l);
    R J=1; for (int k=0;k<3;k++){ R s=fmax(sqrt(fmax(l[k],(R)1e-12)),(R)0.01); J*=s; }
    R V = vol[p], k_ = (R)mp[p]/(h*h);
    R Q[9], cf[9], P[9], Pc[9];
    polar(X, Q); cof3(X, cf);
    R dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    R sg = (dt>0)?(R)1:((dt<0)?(R)-1:(R)0);
    for (int k=0;k<9;k++) P[k] = (R)2*mu*(X[k]-Q[k]) + lam*(J-(R)1)*sg*cf[k];
    mmt3(P, F, Pc); for (int k=0;k<9;k++) Pc[k]*=V;
    R NK1[9]; mm3(M.Nm, M.K1, NK1);
    for (int c3=0;c3<3;c3++) {
        R t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.r[i];
        t -= u[c3] - c[c3];
        R Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
        mm3(M.Mn, dK, MdK);
        R pj=0; for (int k=0;k<9;k++) pj += Pc[k]*(M.g1*M.n[c3]*NK1[k] + MdK[k]);
        gy[c3] = k_*t + pj;
    }
    R MJ[9]; mm3(M.Mn, M.Jst, MJ);
    R gG[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++){ R t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gG[a*3+b]=t; }
    R T2[9]; mmt3(gG, IGt, T2);
    for (int k=0;k<9;k++) gy[3+k]=T2[k];
}
template <typename R, typename G> __device__ __forceinline__ void bs_near_body(int q, int nn_, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* xcur, const float* Fcur, const float* F0inv, const float* mp, const float* vol, double h,
                              double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                              double mu, double lam, const R* gpar, const int* ipar, R hs, G* gmv, int* szp) {
    if (q >= nn_) return;
    long long p = nidx[q];
    R gy[12];
    bs_near_gy<R>(q, p, x0c, F0c, cc, xcur, Fcur, F0inv, mp, vol, (R)h, (R)p0, (R)p1, (R)p2, (R)n0, (R)n1, (R)n2, (R)eps_st, (R)eps_sp, (R)xn,
                  (R)mu, (R)lam, gy);
    R xp[3] = {(R)xcur[p*3], (R)xcur[p*3+1], (R)xcur[p*3+2]};
    int b[3]; R W[3][3], D[3][3];
    bs_base<R>(xp, gpar, ipar, b, W, D);
    int n1_ = ipar[1], n2_ = ipar[2];
    const R* Ai = gpar + 3;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        R gx[3]; R w = bs_wg<R>(W, D, Ai, i, j, l, gx);
        long long n = ((long long)(b[0]+i)*n1_ + (b[1]+j))*n2_ + (b[2]+l);
        for (int c=0;c<3;c++) {
            R t = w*gy[c] + gx[0]*gy[3+c*3] + gx[1]*gy[3+c*3+1] + gx[2]*gy[3+c*3+2];
            atomicAdd(&gmv[n*3+c], (G)(-hs*t));
        }
    }
}

template <typename R, typename G> __device__ __forceinline__ void bs_cell_body(long long c, const int* ipar, G* cV, G* cVF, const R* gK, R mu, R lam, R hs, G* gmv) {
    int n0 = ipar[0], n1 = ipar[1], n2 = ipar[2];
    if (c >= (long long)n0*n1*n2) return;
    R V = (R)cV[c];
    if (!(V > (R)0)) return;
    R Fb[9]; for (int k=0;k<9;k++) { Fb[k] = (R)cVF[c*9+k]/V; cVF[c*9+k] = (G)0; }
    cV[c] = (G)0;
    R Gm[9]; bs_kirch<R>(Fb, mu, lam, Gm);
    int b2 = c % n2, b1 = (c / n2) % n1, b0 = c / ((long long)n1*n2);
    int o = 0;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++, o++) {
        long long n = ((long long)(b0+i)*n1 + (b1+j))*n2 + (b2+l);
        for (int cc=0;cc<3;cc++) {
            R t = Gm[cc*3]*gK[o*3] + Gm[cc*3+1]*gK[o*3+1] + Gm[cc*3+2]*gK[o*3+2];
            atomicAdd(&gmv[n*3+cc], (G)(-hs*V*t));
        }
    }
}
template <typename R, typename G> __device__ __forceinline__ void bs_node_body(long long n, const int* ipar, G* gm, G* gmv, G* vg, R g0, R g1, R g2, R hs, R damp,
                                                                              const unsigned char* fx, const float* vbc) {
    if (n >= (long long)ipar[0]*ipar[1]*ipar[2]) return;
    R m = (R)gm[n];
    R gg[3] = {g0, g1, g2};
    for (int c=0;c<3;c++) {
        R v = (m > (R)0) ? ((R)gmv[n*3+c]/m + hs*gg[c])*damp : (R)0;
        if (fx && fx[n]) v = (R)vbc[n*3+c];
        vg[n*3+c] = (G)v; gmv[n*3+c] = (G)0;
    }
    gm[n] = (G)0;
}
// 프레임 분할 (절 L ex_split 과 같은 정의: 예측 변위 u0 = Σ w (H v_n + H² g) 의 끝 높이 s/ε ≤ 20 이면 바닥 후보)
template <typename R, typename G> __device__ __forceinline__ void bs_split_body(int p, const float* x, const G* vg, const R* gpar, const int* ipar, R H, R g0, R g1, R g2,
                                                                               int floor, R p0, R p1, R p2, R q0, R q1, R q2, R eps_sp, R eps_st,
                                                                               int* nslot, long long* nidx, int* szp) {
    bool near = false;
    if (floor) {
        R xp[3] = {(R)x[p*3], (R)x[p*3+1], (R)x[p*3+2]};
        int b[3]; R W[3][3], D[3][3];
        bs_base<R>(xp, gpar, ipar, b, W, D);
        int n1 = ipar[1], n2 = ipar[2];
        R u[3] = {0,0,0}, gg[3] = {g0,g1,g2};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            R w = W[i][0]*W[j][1]*W[l][2];
            long long n = ((long long)(b[0]+i)*n1 + (b[1]+j))*n2 + (b[2]+l);
            for (int c=0;c<3;c++) u[c] += w*(H*(R)vg[n*3+c] + H*H*gg[c]);
        }
        R s = (xp[0]+u[0]-p0)*q0 + (xp[1]+u[1]-p1)*q1 + (xp[2]+u[2]-p2)*q2;
        if (eps_sp > (R)0 && s/eps_sp <= (R)20) near = true;
        if (eps_st > (R)0 && s/eps_st <= (R)20) near = true;
    }
    if (near) { int q = atomicAdd(&szp[2], 1); nidx[q] = p; nslot[p] = q; }
    else nslot[p] = -1;
}

extern "C" {
// ---- 기준 판 (double, 따로따로) ----
__global__ void bs_g2p_d(int N, const float* xa, const float* Fa, const double* vg, const double* gpar, const int* ipar, double hs,
                         float* xb, float* vb, float* Fb, int* szp, const double* gK, const int* nslot) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double x[3] = {xa[p*3], xa[p*3+1], xa[p*3+2]}, xn[3], vn[3], Fn[9];
    bs_g2p_body<double, double>(x, Fa + p*9, vg, gpar, ipar, hs, xn, vn, Fn, szp, (gK && !(nslot && nslot[p] >= 0)) ? gK : nullptr);
    for (int k=0;k<3;k++) { xb[p*3+k] = (float)xn[k]; vb[p*3+k] = (float)vn[k]; }
    for (int k=0;k<9;k++) Fb[p*9+k] = (float)Fn[k];
}
__global__ void bs_p2g_d(int N, const float* xa, const float* va, const float* Fa, const float* m, const float* vol, const int* nslot, int quad,
                         const double* gpar, const int* ipar, double hs, double mu, double lam, double* gm, double* gmv, double* cV, double* cVF, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double x[3], v[3], F[9];
    for (int k=0;k<3;k++) { x[k] = xa[p*3+k]; v[k] = va[p*3+k]; }
    for (int k=0;k<9;k++) F[k] = Fa[p*9+k];
    bs_p2g_body<double, double>(x, v, F, (double)m[p], (double)vol[p], nslot ? (nslot[p] >= 0) : 0, quad, gpar, ipar, hs, mu, lam, gm, gmv, cV, cVF, szp);
}
__global__ void bs_near_d(const int* szp, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                          const float* xcur, const float* Fcur, const float* F0inv, const float* mp, const float* vol, double h,
                          double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                          double mu, double lam, const double* gpar, const int* ipar, double hs, double* gmv) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bs_near_body<double, double>(q, szp[2], nidx, x0c, F0c, cc, xcur, Fcur, F0inv, mp, vol, h, p0, p1, p2, n0, n1, n2, eps_st, eps_sp, xn, mu, lam,
                                 gpar, ipar, hs, gmv, (int*)szp);
}
__global__ void bs_cell_d(const int* ipar, double* cV, double* cVF, const double* gK, double mu, double lam, double hs, double* gmv) {
    long long c = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    bs_cell_body<double, double>(c, ipar, cV, cVF, gK, mu, lam, hs, gmv);
}
__global__ void bs_node_d(const int* ipar, double* gm, double* gmv, double* vg, double g0, double g1, double g2, double hs, double damp,
                          const unsigned char* fx, const float* vbc) {
    long long n = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    bs_node_body<double, double>(n, ipar, gm, gmv, vg, g0, g1, g2, hs, damp, fx, vbc);
}
__global__ void bs_split_d(int N, const float* x, const double* vg, const double* gpar, const int* ipar, double H, double g0, double g1, double g2,
                           int floor, double p0, double p1, double p2, double q0, double q1, double q2, double eps_sp, double eps_st,
                           int* nslot, long long* nidx, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bs_split_body<double, double>(p, x, vg, gpar, ipar, H, g0, g1, g2, floor, p0, p1, p2, q0, q1, q2, eps_sp, eps_st, nslot, nidx, szp);
}

// ---- 최적화 판 (float, 융합) ----
// G2P (do_g2p: xa·Fa + 격자 v -> xb·vb·Fb) + 그 상태의 P2G (do_p2g) 한 커널. do_g2p = 0 이면 xa·va·Fa 를 그대로 P2G.
__global__ void bs_g2p_p2g_f(int N, const float* xa, const float* va, const float* Fa, const float* vg, const float* gpar, const int* ipar, float hs,
                             int do_g2p, int do_p2g, float* xb, float* vb, float* Fb,
                             const float* m, const float* vol, const int* nslot, int quad, float mu, float lam,
                             float* gm, float* gmv, float* cV, float* cVF, int* szp, const float* gK) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float x[3], v[3], F[9];
    if (do_g2p) {
        float x0[3] = {xa[p*3], xa[p*3+1], xa[p*3+2]};
        bs_g2p_body<float, float>(x0, Fa + p*9, vg, gpar, ipar, hs, x, v, F, szp, (gK && !(nslot && nslot[p] >= 0)) ? gK : nullptr);
        for (int k=0;k<3;k++) { xb[p*3+k] = x[k]; vb[p*3+k] = v[k]; }
        for (int k=0;k<9;k++) Fb[p*9+k] = F[k];
    } else {
        for (int k=0;k<3;k++) { x[k] = xa[p*3+k]; v[k] = va[p*3+k]; }
        for (int k=0;k<9;k++) F[k] = Fa[p*9+k];
    }
    if (!do_p2g) return;
    bs_p2g_body<float, float>(x, v, F, m[p], vol[p], nslot ? (nslot[p] >= 0) : 0, quad, gpar, ipar, hs, mu, lam, gm, gmv, cV, cVF, szp);
}
__global__ void bs_near_f(const int* szp, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                          const float* xcur, const float* Fcur, const float* F0inv, const float* mp, const float* vol, double h,
                          double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                          double mu, double lam, const float* gpar, const int* ipar, float hs, float* gmv) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bs_near_body<float, float>(q, szp[2], nidx, x0c, F0c, cc, xcur, Fcur, F0inv, mp, vol, h, p0, p1, p2, n0, n1, n2, eps_st, eps_sp, xn, mu, lam,
                               gpar, ipar, hs, gmv, (int*)szp);
}
__global__ void bs_cell_f(const int* ipar, float* cV, float* cVF, const float* gK, float mu, float lam, float hs, float* gmv) {
    long long c = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    bs_cell_body<float, float>(c, ipar, cV, cVF, gK, mu, lam, hs, gmv);
}
__global__ void bs_node_f(const int* ipar, float* gm, float* gmv, float* vg, float g0, float g1, float g2, float hs, float damp,
                          const unsigned char* fx, const float* vbc) {
    long long n = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    bs_node_body<float, float>(n, ipar, gm, gmv, vg, g0, g1, g2, hs, damp, fx, vbc);
}
__global__ void bs_split_f(int N, const float* x, const float* vg, const float* gpar, const int* ipar, float H, float g0, float g1, float g2,
                           int floor, float p0, float p1, float p2, float q0, float q1, float q2, float eps_sp, float eps_st,
                           int* nslot, long long* nidx, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bs_split_body<float, float>(p, x, vg, gpar, ipar, H, g0, g1, g2, floor, p0, p1, p2, q0, q1, q2, eps_sp, eps_st, nslot, nidx, szp);
}
}
'''

_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + _BS).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def launch(name, n, args, b=128):
    with RTC._Stream():
        mod().get_function(name)(((int(n) + b - 1) // b,), (b,), args)


def cell_grad_table(Ai, K):
    """셀 대표 적분점의 ∇w 표 [27,3] (x 좌표): 셀 안 K 점 (1: 중심 f=1, 8: 2x2x2 가우스점 f = 1 ± 1/(2√3)) 의 ∂w/∂y 평균 -> (·)ᵀ Ai."""
    if int(K) == 1:
        pts = [(1.0, 1.0, 1.0)]
    elif int(K) == 8:
        e = 0.5 / math.sqrt(3.0)
        pts = [(1 + a * e, 1 + b * e, 1 + c * e) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    else:
        raise SystemExit(f"--K {K} 은 1 또는 8")
    acc = np.zeros((27, 3))
    for f in pts:
        W = [[0.5 * (1.5 - f[k]) ** 2, 0.75 - (f[k] - 1) ** 2, 0.5 * (f[k] - 0.5) ** 2] for k in range(3)]
        D = [[f[k] - 1.5, -2.0 * (f[k] - 1), f[k] - 0.5] for k in range(3)]
        o = 0
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    acc[o] += [D[0][i] * W[1][j] * W[2][l], W[0][i] * D[1][j] * W[2][l], W[0][i] * W[1][j] * D[2][l]]
                    o += 1
    acc /= len(pts)
    return torch.from_numpy(acc @ Ai.detach().double().cpu().numpy())   # ∇_x w = (∂w/∂y)ᵀ Ai  (행벡터)
