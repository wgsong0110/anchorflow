"""명시적 FEM (사용자 10-28 지시, doc/fem_realtime.md 절 L) 커널: 암시적 실시간 판 (fem_rt_cuda) 과 같은 모델 --
사면체 적분점 (셀별 F̄_T = 입자 부피 가중 평균, P2G), 고정 FCR 탄성, 같은 관성 (질량)·중력·감쇠, 입자 바닥 (stick -> softplus 매핑) --
을 뉴턴 대신 명시적 (symplectic Euler) 으로 푼다.

바닥 후보 (매핑이 항등이 아닌 입자, 프레임 시작에 관성 예측 끝 높이 s/ε ≤ 20 -- 암시적 판과 같은 분할) 는 사면체 F̄ 에서 빠지고,
기존 입자 바닥 에너지를 **프레임 단위 그대로** (걸음 h, 프레임 시작 x0·F0, c = h v + h² g) 둔다:
    E_fl(u, G) = m/(2h²)(|q(x0+u) - x0 - c|² - |u - c|²) + V ψ(J_f F0)      (u·G = 프레임 시작부터의 변위·변위 기울기)
(첫 항: 매핑이 바꾼 관성 몫 -- 매핑 없는 관성은 노드 질량 동역학이 갖는다; 둘째: 이 입자의 탄성 (매핑 J_f 포함)).
매핑 (softplus 이동 sp(s) - s, stick) 은 한 번 쓸 때마다 같은 양을 옮기는 걸음 정의라 서브스텝마다 쓰면 N 배로 밀어 낸다 (진단: N=64 첫 서브스텝 |v| 160 m/s)
-> 매핑은 프레임 끝 G2P 에 한 번 (암시적 판과 같다), 서브스텝에는 그 에너지의 음의 기울기를 힘으로.
한 서브스텝 (h_s = h / N):
  f   = -∂Ψ_far/∂x - ∂E_fl/∂x   (지금 상태에서 -- 명시적)
  v  <- (v + h_s (f / m_n + g)) · 감쇠,  x_n += h_s v  (노드 질량 = Σ λ m_p, 속도 = 질량 가중 P2G; --cfg_bc 묶인 노드는 v_bc)
  sub (MPM 식)  : 매 서브스텝 P2G (질량·속도·F̄) -> 힘 -> 선형 G2P (x += Σλ ΔU, F <- (I + Σ ΔU ⊗ ∇λ) F, v = ΔU / h_s)
  lag (라그랑주): 프레임 동안 격자 노드가 움직인다 (U 누적, 셀 F = A_T(U) F̄_T), 바닥 후보는 u = Σλ U 보간, 프레임 끝 선형 G2P (v = Σλ v_n)
  프레임 끝 (둘 다): 바닥 후보 입자만 x = q(x0 + u), F = J_f F0, v += (q - x0 - u)/h.
커널 소스는 fem_rt_cuda._SRC (극분해·고유값·near_map 장치 함수) 뒤에 붙여 한 모듈로 컴파일한다.
"""
from __future__ import annotations

import numpy as np
import torch

from . import fem_rt_cuda as RTC

_EX = r'''
// 진단 전용 배율 (절 N.2, --near_diag): [0] 바닥 후보 관성 보정 항 m/h² (J_qᵀ r - (u - c)), [1] 변위 쪽 탄성 항 pj, [2] 변위 기울기 쪽 탄성 항 gG. 기본 1 (원래 식)
__device__ float EXD[3] = {1.f, 1.f, 1.f};
template <typename R, typename S>
__device__ __forceinline__ void ex_tet_force_body(int t, const long long* tets, const float* U, const S* gl, const S* FT, const S* VT, R mu, R lam, float* f) {
    R V = (R)VT[t];
    if (!(V > (R)0)) return;
    long long n[4]; R G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=(R)gl[t*12+k];
    R A[9] = {1,0,0, 0,1,0, 0,0,1};
    if (U) {
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { R u=(R)U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    }
    R Fb[9]; for (int k=0;k<9;k++) Fb[k]=(R)FT[t*9+k];
    R X[9]; mm3(A, Fb, X);
    R C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    R l[3]; symeig3(C, l);
    R J = 1;
    for (int k=0;k<3;k++){ R s = sqrt(l[k] > (R)1e-12 ? l[k] : (R)1e-12); s = s > (R)0.01 ? s : (R)0.01; J *= s; }
    R Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    R dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    R sg = (dt>0)?(R)1:((dt<0)?(R)-1:(R)0);
    for (int k=0;k<9;k++) P[k] = (R)2*mu*(X[k]-Q[k]) + lam*(J-(R)1)*sg*cf[k];
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        R s = 0; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], (float)(-V*s));
    }
}

template <typename R> __device__ __forceinline__ void ex_near_force_body(int q, int nn_, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, R h,
                              R p0, R p1, R p2, R n0, R n1, R n2, R eps_st, R eps_sp, R xn, R mu, R lam, float* gn) {
    if (q >= nn_) return;
    long long p = nidx[q];
    long long nd[4]; R lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3], IGt[9]={1,0,0,0,1,0,0,0,1};
    const float* glp = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; }
    for (int k=0;k<12;k++) gl[k]=glp[k];
    for (int k=0;k<3;k++){ xx[k]=x0c[q*3+k]; c[k]=cc[q*3+k]; }
    if (Ucum) {
        for (int i=0;i<4;i++) for (int a=0;a<3;a++){ R Ua=(R)Ucum[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
    } else {
        for (int k=0;k<3;k++) u[k] = (R)xcur[p*3+k] - xx[k];
        R Fc[9], Fi[9]; for (int k=0;k<9;k++){ Fc[k]=Fcur[p*9+k]; Fi[k]=F0inv[q*9+k]; }
        mm3(Fc, Fi, IGt);
        for (int k=0;k<9;k++) G[k] = IGt[k] - ((k%4==0)?(R)1:(R)0);
    }
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
    R gy[12];
    R NK1[9]; mm3(M.Nm, M.K1, NK1);
    for (int c3=0;c3<3;c3++) {
        R t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.r[i];
        t -= u[c3] - c[c3];                                            // - 매핑 없는 관성 (노드 질량 동역학이 이미 갖는 몫)
        R Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
        mm3(M.Mn, dK, MdK);
        R pj=0; for (int k=0;k<9;k++) pj += Pc[k]*(M.g1*M.n[c3]*NK1[k] + MdK[k]);
        gy[c3] = (R)EXD[0]*k_*t + (R)EXD[1]*pj;
    }
    R MJ[9]; mm3(M.Mn, M.Jst, MJ);
    R gG[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++){ R t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gG[a*3+b]=t; }
    if (!Ucum) { R T2[9]; mmt3(gG, IGt, T2); for (int k=0;k<9;k++) gG[k]=T2[k]; }     // sub: ∂/∂δU 는 gG (I+G)ᵀ
    for (int k=0;k<9;k++) gy[3+k]=(R)EXD[2]*gG[k];
    for (int i=0;i<4;i++) for (int c3=0;c3<3;c3++) {
        R t = lm[i]*gy[c3];
        for (int b=0;b<3;b++) t += gl[i*3+b]*gy[3+c3*3+b];
        atomicAdd(&gn[nd[i]*3+c3], (float)t);
    }
}


extern "C" {
// szp (선택): 장치 크기 [M, T, nn, 넘침] -- 그래프 재생에서 프레임마다 바뀌는 크기를 장치에서 읽는다 (없으면 인자 값)
__global__ void ex_tet_force_dd(const int* szp, int Targ, const long long* tets, const float* U, const double* gl, const double* FT, const double* VT,
                                double mu, double lam, float* f) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < T) ex_tet_force_body<double, double>(t, tets, U, gl, FT, VT, mu, lam, f);
}
__global__ void ex_tet_force_ff(const int* szp, int Targ, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                float mu, float lam, float* f) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < T) ex_tet_force_body<float, float>(t, tets, U, gl, FT, VT, mu, lam, f);
}

__global__ void ex_node_step(const int* szp, int Marg, const float* mn, const float* f, const float* gn, float* vn, float* dU, float* Ucum,
                             float g0, float g1, float g2, float hs, float damp, const unsigned char* fixd, const float* vbc) {
    int M = szp ? szp[0] : Marg;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        float v;
        if (fixd && fixd[n]) v = vbc[n*3+c];
        else if (m > 0.f) v = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v = 0.f;
        vn[n*3+c] = v; dU[n*3+c] = hs*v;
        if (Ucum) Ucum[n*3+c] += hs*v;
    }
}

// 바닥 후보 입자의 프레임 바닥 에너지 (기존 정의, 프레임 걸음 h·프레임 시작 x0·F0·c = h v + h² g 그대로):
//   E_fl = m/(2h²)(|q(x0+u) - x0 - c|² - |u - c|²) + V ψ(J_f(u, G) F0)
//   (매핑이 항등이면 첫 항 0; 탄성은 이 입자들이 사면체 F̄ 에서 빠져 있으므로 여기서), u·G = 프레임 시작부터의 변위·변위 기울기
//   lag: u = Σλ U_cum, G = Σ U_cum ⊗ ∇λ (변수 U_cum)       sub: u = x - x0, G = F F0⁻¹ - I (변수 = 지금 서브스텝 노드 변위 δU, δG = D (I + G))
//   기울기를 노드로 gn (힘 = -gn) -- 서브스텝마다 지금 상태에서 (명시적)
__global__ void ex_near_force(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, double h,
                              double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                              double mu, double lam, float* gn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    ex_near_force_body<double>(q, szp ? szp[2] : nnarg, nidx, x0c, F0c, cc, Ucum, xcur, Fcur, F0inv, rows, lamb, glT, tinv, mp, vol, h,
                               p0, p1, p2, n0, n1, n2, eps_st, eps_sp, xn, mu, lam, gn);
}
// float32 대수 판 (암시적 판 --near32 / --eg32 와 같은 방식)
__global__ void ex_near_force32(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, double h,
                              double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                              double mu, double lam, float* gn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    ex_near_force_body<float>(q, szp ? szp[2] : nnarg, nidx, x0c, F0c, cc, Ucum, xcur, Fcur, F0inv, rows, lamb, glT, tinv, mp, vol, (float)h,
                              (float)p0, (float)p1, (float)p2, (float)n0, (float)n1, (float)n2, (float)eps_st, (float)eps_sp, (float)xn, (float)mu, (float)lam, gn);
}

// 프레임 끝 바닥 후보 입자: 기존 G2P 매핑을 프레임 변위에 한 번 (암시적 판 G2P 와 같은 식) -- x2 = q(x0 + u), F2 = J_f F0,
// v2 = (선형 G2P 가 이미 쓴 v2) + (q - x0 - u)/h  (매핑이 옮긴 만큼을 프레임 속도로)
__global__ void ex_near_fin(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c,
                            const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                            const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                            double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn, double h,
                            float* x2, float* v2, float* F2) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    double u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c0[3]={0,0,0};
    for (int k=0;k<3;k++) xx[k]=x0c[q*3+k];
    if (Ucum) {
        const float* gl = glT + tinv[p]*12;
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; } }
    } else {
        for (int k=0;k<3;k++) u[k] = (double)xcur[p*3+k] - xx[k];
        double Fc[9], Fi[9], IG[9]; for (int k=0;k<9;k++){ Fc[k]=Fcur[p*9+k]; Fi[k]=F0inv[q*9+k]; }
        mm3(Fc, Fi, IG);
        for (int k=0;k<9;k++) G[k] = IG[k] - ((k%4==0)?1.0:0.0);
    }
    double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMap M; near_map(xx, u, G, c0, pt, nv, eps_st, eps_sp, xn, M);
    double Jf[9]; mm3(M.Mn, M.K1, Jf);
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)(xx[k] + M.r[k]); v2[p*3+k] = (float)((double)v2[p*3+k] + (M.r[k] - u[k])/h); }
    double F[9]; for (int k=0;k<9;k++) F[k]=F0c[q*9+k];
    double F2d[9]; mm3(Jf, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}

// (lag) 프레임 끝 선형 G2P: x + Σλ U, v = Σλ v_n, F <- (I + Σ U ⊗ ∇λ) F  (바닥 후보는 뒤이어 ex_near_fin 이 덮어쓴다)
__global__ void ex_g2p_lag(int N, const float* x, const float* Ucum, const float* Vn, const long long* rows, const float* lamb, const float* glT,
                           const long long* tinv, const float* Fp, float* x2, float* v2, float* F2, float* C2, int vq) {
    // C2 (선택, --affine): 입자 속도 기울기 C = (Σ v_n ⊗ ∇λ_n) A⁻¹ (사면체 안 선형 속도장의 현재 배치 기울기) -- 다음 프레임 P2G 의 아핀 전달 (APIC 와 같은 꼴)
    //   (정답 PG 의 APIC C = B D⁻¹, B = Σ w v_i (x_i - x_p)ᵀ, D = Σ w (x_i - x_p)(x_i - x_p)ᵀ 와 같은 값: 선형 사면체 보간은 아핀장에 정확하므로
    //    B = (Σ v_i ⊗ ∇λ_i) D 이고, D 는 입자가 면·모서리 위면 특이하므로 닫힌 꼴로 쓴다)
    // vq = 1 (절 Q (a) --rep_w tet4): 입자 속도 v = ¼ Σ v_n (P2G 사면체 등분 가중 m_p/4 와 같은 가중) -- 위치·F 는 그대로 λ·A
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1}, Gv[9]={0,0,0,0,0,0,0,0,0};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; double Va=(double)Vn[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*Va;
            for (int b=0;b<3;b++) { A[a*3+b]+=Ua*gl[i*3+b]; Gv[a*3+b]+=Va*gl[i*3+b]; } } }
    if (vq) { for (int a=0;a<3;a++) { double t=0; for (int i=0;i<4;i++) t += (double)Vn[rows[p*4+i]*3+a]; vv[a] = 0.25*t; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
    if (C2) {
        double cf[9]; cof3(A, cf);
        double dA = A[0]*cf[0]+A[1]*cf[1]+A[2]*cf[2];
        double Ci[9];
        if (fabs(dA) > 1e-12) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ci[i*3+j] = cf[j*3+i]/dA; }   // A⁻¹ = cofᵀ / det
        else { for (int k=0;k<9;k++) Ci[k] = (k%4==0)?1.0:0.0; }
        double Cc[9]; mm3(Gv, Ci, Cc);
        for (int k=0;k<9;k++) C2[p*9+k] = (float)Cc[k];
    }
}
// ======== 절 O: lag 판 속도 단위 바닥 (--floor vel, 사용자 10-28 "2번") ========
// 정답 (PG mpm_solver_warp add_surface_collider, surface "sticky"): 격자 노드 (x_n - p)·n < 0 이면 grid_v_out = 0 (세 성분 모두),
//   서브스텝마다 격자 정규화·중력·감쇠 뒤, G2P 앞.
// 여기서는 판정을 **입자** 로: 바닥 후보 입자의 지금 보간 위치 x_p + Σλ U (서브스텝 시작) 의 높이 s = (x - p)·n < 0 이면 접촉.
// 접촉 입자가 걸린 사면체 네 노드에 P2G 와 같은 질량 가중 w_pn (사면체 등분 0.25 m_p / Σλm 판은 λ m_p) 을 쌓는다 -> acc_n.
__global__ void ex_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                               const float* m, int tetmass, float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*Ucum[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// lag 노드 걸음 (ex_node_step 과 같은 식) + 속도 바닥 + 쓴 뒤 f·gn·acc 비우기 (채우기 커널 없음):
//   v = (v + h_s((f - gn)/m + g))·감쇠  (--cfg_bc 묶인 노드 = v_bc)
//   acc_n > 0 (접촉 입자가 걸린 노드):
//     mode 0 (질량 비율 고착): v <- (1 - α) v,  α = min(acc_n / m_n, 1)  -- 접촉 입자 몫의 운동량을 0 으로 (정답 sticky 의 v = 0 을 접촉 입자 질량에만)
//     mode 1 (노드 전체 고착): v <- 0  -- 정답 sticky 와 같은 노드 속도 0 (판정만 입자)
//     mode 2 (절 AG 미끄럼, --vfl_mode slip): 접선 성분 그대로 (np = 1 이면 안쪽 법선 성분만 제거 -- GF/PG bounding_box 의 한쪽 방향 속도 0)
//     np = 1 (비관통): 그 뒤 v·n < 0 이면 법선 성분 제거 -> 접촉 입자의 보간 속도 Σλ v_n (볼록 결합) 의 법선 성분 ≥ 0
//   U += h_s v
__global__ void ex_node_step_lag(const int* szp, const float* mn, float* f, float* gn, float* vn, float* Ucum, float g0, float g1, float g2, float hs, float damp,
                                 const unsigned char* fixd, const float* vbc, float* acc, int mode, int np_, float n0, float n1, float n2, int nodisp) {
    // nodisp = 1 (절 R --lim): 속도만 쓰고 U 갱신은 뒤이은 색칠 이동 제한 커널 (ex_lim_color) 이 한다
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    float v[3];
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f) v[c] = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v[c] = 0.f;
        f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f;
    }
    if (acc) {
        float a = acc[n];
        acc[n] = 0.f;
        if (a > 0.f && !(fixd && fixd[n])) {
            float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f));   // mode 2 (절 AG slip): 접선 그대로, 아래 비관통만 (정답 bounding_box: 바깥 법선 성분만 0)
            for (int c=0;c<3;c++) v[c] *= k;
            if (np_) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
        }
    }
    for (int c=0;c<3;c++) { vn[n*3+c] = v[c]; if (!nodisp) Ucum[n*3+c] += hs*v[c]; }
}
// ======== 절 R: lag 판 셀 뒤집힘 방지 (사용자 10-29 "두 가지 다") ========
// (1) FCR + log J 장벽: 에너지 밀도에 κ μ b(J; Ĵ) (phys_resid._barrier_b, 연장 linear: J < 1e-3 Ĵ 에서 값·기울기 이어 붙임) 를 더한다.
//     fbon 0: J = det(A_T F̄_T) (셀 변형 전체 = 기존 F 장벽 꼴), 1: J = det A_T (프레임 안 셀 사상 = 기존 셀 장벽 꼴)
//     b'(J) = -2 (J - Ĵ) ln(J/Ĵ) - (J - Ĵ)² / J  (J < Ĵ), 0 (J ≥ Ĵ)
__device__ __forceinline__ float bar_db(float d, float jh) {
    if (!(d < jh)) return 0.f;
    float eps = 1e-3f*jh;
    float dc = d < eps ? eps : d;
    float k = dc - jh, l = logf(dc/jh);
    return -2.f*k*l - k*k/dc;
}
// 사면체 힘 (ex_tet_force_body<float> 와 같은 FCR 식) + 장벽 + 뒤집힘 계수 rst[0] += 1 (점유 사면체, det A_T ≤ 0 인 서브스텝마다)
__global__ void ex_tet_force_R(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                               float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// (2') 상자 판 (--lim_mode box, 서브스텝당 커널 2 개 -- 색 8 번보다 싸다): det(A0 + Σ α_i d_i ⊗ ∇λ_i) 는 각 α_i 에 대해 아핀 (행렬식 보조정리) 이라
//   상자 [0,s]⁴ 위 최솟값은 꼭짓점 (부분집합 S 의 노드만 s 배로 움직인 15 + 1 배치) 에서 난다. 사면체마다 꼭짓점 det ≥ thr 를 지키는 최대 s_T 를 찾고
//   (s = 1 이 되면 끝, 아니면 이분 14 번 -- lo 는 늘 유효), 노드 α_n = min_T s_T (원자 min, 음 아닌 float 비트 순서 = 정수 순서).
//   모든 노드가 α_n ≤ s_T 이므로 실제 배치는 그 사면체의 상자 안 -> det A_T ≥ thr 보장 (Jacobi, 순서 무관·결정적). thr = min(jmin, 지금 det) (더 줄이지만 않게).
//   (--cfg_bc 묶인 노드는 α 1 로 움직인다 -- 그 노드가 든 사면체는 보장 밖)
__device__ __forceinline__ float det_sub(const float* A0, const float* Mo, int S, float s) {
    float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
    return det3(B);
}
__global__ void ex_lim_box_tet(const int* szp, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum, float hs, float jmin,
                               unsigned int* alph, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float Mo[36];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
    float D0 = det3(A0);
    float thr = fminf(jmin, D0);
    bool ok = true;
    for (int S=1; S<16 && ok; S++) if (det_sub(A0, Mo, S, 1.f) < thr) ok = false;
    if (ok) return;
    float lo = 0.f, hi = 1.f;
    for (int it=0; it<14; it++) {
        float mid = 0.5f*(lo+hi); bool g = true;
        for (int S=1; S<16 && g; S++) if (det_sub(A0, Mo, S, mid) < thr) g = false;
        if (g) lo = mid; else hi = mid;
    }
    unsigned int bits = __float_as_uint(lo);
    for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
    atomicAdd(&rst[7], 1.f);
}
__global__ void ex_lim_box_apply(const int* szp, const float* mn, float* vn, float* Ucum, float hs, unsigned int* alph, int vmode, const unsigned char* fixd, float* rst) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float al = __uint_as_float(alph[n]);
    alph[n] = 0x3f800000u;                                         // 1.0f (다음 서브스텝)
    if (fixd && fixd[n]) al = 1.f;
    float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
    for (int c=0;c<3;c++) Ucum[n*3+c] += al*hs*v[c];
    if (al < 1.f) {
        float m = mn[n], w = 1.f - al;
        atomicAdd(&rst[1], 1.f);
        if (vmode == 0) {
            for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
            float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
            atomicAdd(&rst[6], w*m*sqrtf(v2));
        }
    }
}
// (2) 노드 이동 제한용 노드 -> 사면체 목록 (프레임마다, 격자 만든 뒤): nt[n*24 + s] = t*4 + (n 의 사면체 안 꼭짓점 번호), ntc[n] = 개수 (Kuhn 최대 24)
__global__ void ex_node_tets(const int* szp, const long long* tets, int* ntc, int* nt, int* ovf) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    for (int i=0;i<4;i++) { long long n = tets[t*4+i]; int s = atomicAdd(&ntc[n], 1); if (s < 24) nt[n*24+s] = t*4+i; else atomicAdd(ovf, 1); }
}
// (2) 색 col 의 노드만 (Kuhn 격자 지표 홀짝 8 색: 한 사면체의 네 노드는 지표 차가 {0,1}³\0 이라 모두 다른 색 -> 같은 색 노드는 사면체를 공유하지 않는다)
//   이번 서브스텝 변위 d = h_s v_n 을 α ∈ [0,1] 배로 줄여, 이 노드가 든 점유 사면체 (V_T > 0) 마다 det A_T ≥ jmin 을 지킨다.
//   한 노드만 움직이면 det(A + α d ⊗ ∇λ_i) = det A + α dᵀ cof(A) ∇λ_i (행렬식 보조정리) 로 α 에 선형 -> 닫힌 꼴 최대 α.
//   다른 색 노드는 그 사이 고정 (앞 색은 이미 갱신, 뒤 색은 아직) -- 색 순서 0..7 로 Gauss-Seidel, 같은 색 안은 병렬·순서 무관 (결정적).
//   vmode 0: v ← α v (속도도 이동과 맞춘다 -- 운동량 (1-α) m v·운동에너지 ½ m |v|² (1-α²) 를 rst 에 기록), 1: v 그대로 (이동만 줄임)
//   rst: [1] 제한된 (노드·서브스텝) 수, [2..4] 줄인 운동량 Σ (1-α) m v, [5] 줄인 운동에너지, [6] Σ |(1-α) m v|
__global__ void ex_lim_color(const int* szp, int col, const unsigned char* ncol, const int* ntc, const int* nt, const long long* tets, const float* gl,
                             const float* VT, const float* mn, float* vn, float* Ucum, float hs, float jmin, int vmode, const unsigned char* fixd, float* rst) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M || ncol[n] != col) return;
    float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
    float d[3] = {hs*v[0], hs*v[1], hs*v[2]};
    float al = 1.f;
    if (!(fixd && fixd[n])) {
        int k = min(ntc[n], 24);
        for (int s=0;s<k;s++) {
            int e = nt[n*24+s]; int t = e >> 2, i = e & 3;
            if (!(VT[t] > 0.f)) continue;
            const float* G = gl + (long long)t*12;
            float A[9] = {1,0,0, 0,1,0, 0,0,1};
            for (int j=0;j<4;j++) { long long q = tets[(long long)t*4+j]; for (int c=0;c<3;c++) { float u=Ucum[q*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[j*3+b]; } }
            float cA[9]; cof3(A, cA);
            float D0 = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
            float r = 0.f;
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) r += d[a]*cA[a*3+b]*G[i*3+b];
            if (r < 0.f) { float am = D0 > jmin ? (D0 - jmin)/(-r) : 0.f; al = fminf(al, am); }
        }
        al = fminf(fmaxf(al, 0.f), 1.f);
    }
    for (int c=0;c<3;c++) Ucum[n*3+c] += al*d[c];
    if (al < 1.f) {
        float m = mn[n], w = 1.f - al;
        atomicAdd(&rst[1], 1.f);
        if (vmode == 0) {
            for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
            float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
            atomicAdd(&rst[6], w*m*sqrtf(v2));
        }
    }
}
// ======== 절 Z: A_T 특이값 비율 σ1/σ3 (사용자 10-29 "특이값 비율로 두 방식 각각") ========
// r(A) = σ1(A)/σ3(A) (AᵀA 고윳값 l0 ≤ l1 ≤ l2 -> sqrt(l2/l0)). 클램프 판정에서는 det A ≤ 0 이면 r = +inf (뒤집힌 배치는 비율 무한대로 본다).
__device__ __forceinline__ float sr_ratio(const float* A) {
    float d = det3(A);
    if (!(d > 0.f)) return __int_as_float(0x7f800000);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
    float l[3]; symeig3(C, l);
    return sqrtf(fmaxf(l[2], 0.f)/fmaxf(l[0], 1e-30f));
}
// (1) 클램프 (--lim_sr c, 절 R 상자 판과 같은 Jacobi 틀: 사면체별 s_T -> 노드 α_n = min_T s_T (원자 min) -> ex_lim_box_apply 가 이동·속도 α 배):
//   판정 feas(s): 상자 [0,s]⁴ 의 꼭짓점 15 개 (부분집합 S 의 노드만 s 배 이동) 모두 r ≤ thrR (= max(c, 지금 r)) [+ jon 이면 det ≥ thrJ (= min(Ĵ, 지금 det)), 기존 상자 판 그대로].
//   빠른 통과: ‖Σ α_i d_i ∇λ_iᵀ‖₂ ≤ ρ = Σ |d_i||∇λ_i| (모든 α ∈ [0,1]⁴) -> Weyl 로 σ1 ≤ σ1⁰ + ρ, σ3 ≥ σ3⁰ - ρ; (σ1⁰+ρ) ≤ thrR (σ3⁰-ρ) 이면 비율은 상자 전체에서 보장 (꼭짓점 검사 생략).
//   아니면 s = 1 꼭짓점 검사, 실패하면 이분 14 번. 비율은 α 에 아핀이 아니라 꼭짓점 판정은 상자 내부를 보장하지 않는다 -> 이동 뒤 실제 비율 위반을 ex_sr_viol 로 센다 (rst[8]).
__device__ __forceinline__ bool sr_feas(const float* A0, const float* Mo, float s, float thrR, int jon, float thrJ) {
    for (int S=1; S<16; S++) {
        float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
        if (jon && det3(B) < thrJ) return false;
        if (sr_ratio(B) > thrR) return false;
    }
    return true;
}
__global__ void ex_lim_sr_tet(const int* szp, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum, float hs,
                              float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float Mo[36];
    float rho = 0.f;
    for (int i=0;i<4;i++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
        for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
        rho += sqrtf(dd*gg);
    }
    float D0 = det3(A0);
    float r0 = sr_ratio(A0);
    float thrR = fmaxf(csr, r0);
    float thrJ = fminf(jmin, D0);
    bool ok = true;
    if (jon) { for (int S=1; S<16 && ok; S++) if (det_sub(A0, Mo, S, 1.f) < thrJ) ok = false; }
    if (ok && D0 > 0.f) {
        float C[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
        float l[3]; symeig3(C, l);
        float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
        if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) return;          // Weyl: 상자 전체 보장
    }
    if (sr_feas(A0, Mo, 1.f, thrR, jon, thrJ)) return;
    float lo = 0.f, hi = 1.f;
    for (int it=0; it<14; it++) {
        float mid = 0.5f*(lo+hi);
        if (sr_feas(A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
    }
    unsigned int bits = __float_as_uint(lo);
    for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
    atomicAdd(&rst[7], 1.f);
}
// 클램프 검산 (--lim_sr_chk, 측정 판에는 끔): 이동 뒤 점유 사면체의 실제 r > c (1 + 1e-3) 인 (사면체 x 서브스텝) 수 -> rst[8], det ≤ 0 수 -> rst[9]
__global__ void ex_sr_viol(const int* szp, const long long* tets, const float* gl, const float* VT, const float* Ucum, float csr, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long q = tets[t*4+i]; for (int c=0;c<3;c++) { float u=Ucum[q*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*gl[t*12+i*3+b]; } }
    float r = sr_ratio(A);
    if (!(det3(A) > 0.f)) atomicAdd(&rst[9], 1.f);
    else if (r > csr*1.001f) atomicAdd(&rst[8], 1.f);
}
// (2) 로그 장벽 (--sb κ --sb_c c --sb_dhat d̂): 사면체 에너지에 V_T κ μ b(c - r(A_T); d̂) (b 는 기존 bar_db 꼴 -(d-d̂)² ln(d/d̂), d < 1e-3 d̂ 선형 연장).
//   ∂r/∂A = A (P_max/(σ1σ3) - σ1 P_min/σ3³),  P_k = AᵀA 의 고유 사영 (단순 고윳값: Π_{j≠k}(C - l_j I)/(l_k - l_j), 겹친 고윳값: 두 차원 사영의 1/2 -- 대칭 부분기울기)
//   ∂E/∂A = -κ μ b'(d) ∂r/∂A  (b' < 0 이라 r 을 줄이는 쪽), 사면체 힘 f_i = -V (∂E/∂A) ∇λ_i. 장벽 부분은 double.
//   나머지 (FCR 탄성·뒤집힘 계수·det 장벽 fbk) 는 ex_tet_force_R 와 같다.
__device__ __forceinline__ double bar_db_d(double d, double jh) {
    if (!(d < jh)) return 0.0;
    double eps = 1e-3*jh;
    double dc = d < eps ? eps : d;
    double k = dc - jh, l = log(dc/jh);
    return -2.0*k*l - k*k/dc;
}
__device__ __forceinline__ void sr_grad(const float* Af, double* Gr, double* rr) {   // Gr = ∂r/∂A, rr = r
    double A[9]; for (int k=0;k<9;k++) A[k] = (double)Af[k];
    double C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
    double l[3]; symeig3(C, l);
    double l0 = fmax(l[0], 1e-30), l1 = l[1], l2 = l[2];
    double s1 = sqrt(l2), s3 = sqrt(l0);
    *rr = s1/s3;
    double tol = 1e-6*l2;
    double Pmax[9], Pmin[9], I3[9] = {1,0,0, 0,1,0, 0,0,1};
    double X0[9], X1[9], X2[9];
    for (int k=0;k<9;k++) { X0[k] = C[k] - l0*I3[k]; X1[k] = C[k] - l1*I3[k]; X2[k] = C[k] - l2*I3[k]; }
    if (l2 - l1 > tol) { double Q[9]; mm3(X1, X0, Q); double s = 1.0/((l2-l1)*(l2-l0)); for (int k=0;k<9;k++) Pmax[k] = Q[k]*s; }
    else { double s = 0.5/(l2-l0); for (int k=0;k<9;k++) Pmax[k] = X0[k]*s; }
    if (l1 - l0 > tol) { double Q[9]; mm3(X2, X1, Q); double s = 1.0/((l0-l2)*(l0-l1)); for (int k=0;k<9;k++) Pmin[k] = Q[k]*s; }
    else { double s = 0.5/(l0-l2); for (int k=0;k<9;k++) Pmin[k] = X2[k]*s; }
    double W[9]; double a = 1.0/(s1*s3), b = s1/(s3*s3*s3);
    for (int k=0;k<9;k++) W[k] = a*Pmax[k] - b*Pmin[k];
    mm3(A, W, Gr);
}
__global__ void ex_tet_force_SR(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, float srk, float src, float srd) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    if (srk > 0.f) {
        float CA[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) CA[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
        float la[3]; symeig3(CA, la);
        float rq = sqrtf(fmaxf(la[2], 0.f)/fmaxf(la[0], 1e-30f));
        if ((float)src - rq < 1.01f*srd) {                                     // 활성 근처만 double 로 다시
            double Gr[9], r;
            sr_grad(A, Gr, &r);
            double db = bar_db_d((double)src - r, (double)srd);
            if (db != 0.0) {
                double s = -(double)srk*(double)mu*db;
                for (int k=0;k<9;k++) Gm[k] += (float)(s*Gr[k]);
                if (rst) atomicAdd(&rst[10], 1.f);
            }
        }
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// ======== 절 P: lag 판 K 서브스텝마다 재평균 (--rep K --rep_mode vel, 사용자 10-29 "해") ========
// 속도만 재평균 (입자 이동·재배치 없음, 같은 rows/λ·같은 노드 질량): v_p = Σλ v_n (G2P, PIC) -> (mv)_n = Σ w_pn v_p, v_n = (mv)_n / m_n (P2G)
//   w_pn = P2G 와 같은 질량 가중 (사면체 등분 0.25 m_p / Σλm 판은 λ m_p). mv 는 호출 전에 비워 둔다.
// 절 Q 선택지:
//   g2pq = 1 ((a) --rep_w tet4): G2P 도 P2G 와 같은 사면체 등분 가중 v_p = ¼ Σ v_n
//   tetmass = 0 + 정규화 질량 ml = Σλ m_p ((a) --rep_w lam): P2G 를 G2P 와 같은 λ m_p 가중 (동역학 노드 질량 m_T/4 는 그대로, 정규화만 Σλm)
//   glT ((b) --affine, APIC): C_p = Σ v_n ⊗ ∇λ_n (정답 PG 의 C = B D⁻¹ 와 같은 값), P2G 값 v_p + C_p (x_n - x_p) -- 같은 rows 로 되돌리므로 x_n - x_p 는
//     기준 배치 (npos - x0) 로 계산 (지금 배치의 A_T (X_n - X_p) 와 C A_T = Σ v ⊗ ∇λ 가 상쇄 -- 같은 값). 이 경우 각 기여가 정확히 v_n 이 되어 재평균은 항등이다.
__global__ void ex_rep_vel_p2g(int N, const long long* rows, const float* lamb, const float* m, int tetmass, const float* vn, float* mv,
                               int g2pq, const float* glT, const long long* tinv, const float* npos, const float* x0) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    long long nd[4]; float lm[4]; float vp[3] = {0,0,0};
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; float q = g2pq ? 0.25f : lm[i]; for (int c=0;c<3;c++) vp[c] += q*vn[nd[i]*3+c]; }
    float Gv[9] = {0,0,0,0,0,0,0,0,0};
    if (glT) { const float* gl = glT + tinv[p]*12;
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float vc = vn[nd[i]*3+c]; for (int b=0;b<3;b++) Gv[c*3+b] += vc*gl[i*3+b]; } }
    float mp = m[p];
    for (int i=0;i<4;i++){ float w=(tetmass ? 0.25f : lm[i])*mp;
        float d[3] = {0,0,0};
        if (glT) for (int b=0;b<3;b++) d[b] = npos[nd[i]*3+b] - x0[p*3+b];
        for (int c=0;c<3;c++) atomicAdd(&mv[nd[i]*3+c], w*(vp[c] + Gv[c*3]*d[0] + Gv[c*3+1]*d[1] + Gv[c*3+2]*d[2])); }
}
__global__ void ex_rep_vel_fin(const int* szp, const float* mn, float* mv, float* vn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float m = mn[n];
    for (int c=0;c<3;c++) { if (m > 1e-30f) vn[n*3+c] = mv[n*3+c]/m; mv[n*3+c] = 0.f; }
}
// 프레임 시작 바닥 후보 압축 배열: x0, F0, F0⁻¹, c = h v + h² g
__global__ void ex_near_init(const int* szp, int nnarg, const long long* nidx, const float* x, const float* v, const float* Fp, float h, float g0, float g1, float g2,
                             float* x0c, float* F0c, float* F0inv, float* cc) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float gg[3] = {g0, g1, g2};
    for (int k=0;k<3;k++){ x0c[q*3+k]=x[p*3+k]; cc[q*3+k]=h*v[p*3+k] + h*h*gg[k]; }
    double A[9], C[9];
    for (int k=0;k<9;k++){ A[k]=Fp[p*9+k]; F0c[q*9+k]=Fp[p*9+k]; }
    cof3(A, C);
    double det = A[0]*C[0]+A[1]*C[1]+A[2]*C[2];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) F0inv[q*9+i*3+j] = (float)(C[j*3+i]/det);
}

// (sub) 서브스텝 G2P: 매핑 포함 (floor=1, g2p_part 와 같은 식) 또는 선형, v = (q - x)/h_s * damp
__global__ void ex_g2p_sub(int N, const float* x, const float* dU, const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                           const float* Fp, int floor, double p0, double p1, double p2, double n0, double n1, double n2,
                           double eps_st, double eps_sp, double xn, double hs, double damp, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c0[3]={0,0,0};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; } }
    for (int k=0;k<3;k++) xx[k]=x[p*3+k];
    double Jf[9], r[3];
    if (floor) {
        double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
        NearMap M; near_map(xx, u, G, c0, pt, nv, eps_st, eps_sp, xn, M);
        mm3(M.Mn, M.K1, Jf);
        for (int k=0;k<3;k++) r[k]=M.r[k];
    } else {
        for (int k=0;k<9;k++) Jf[k] = G[k] + ((k%4==0)?1.0:0.0);
        for (int k=0;k<3;k++) r[k]=u[k];
    }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)(xx[k] + r[k]); v2[p*3+k] = (float)(r[k]/hs*damp); }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(Jf, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}

// ---------------- 최적화 판 격자: 셀 점유 표시 -> (팽창) -> 노드·사면체 압축, 서브스텝마다 장치 표로 locate ----------------
// gpar = [lo0, lo1, lo2, Ai(9)], ipar = [n0, n1, n2]
__global__ void ex_mark_cells(int N, const float* x, const float* gpar, const int* ipar, unsigned char* cmark) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {x[p*3]-gpar[0], x[p*3+1]-gpar[1], x[p*3+2]-gpar[2]};
    const float* Ai = gpar + 3;
    int nl[3] = {n0, n1, n2}, ci[3];
    for (int k=0;k<3;k++){ float y = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2]; int c=(int)floorf(y); ci[k]=min(max(c,0),nl[k]-2); }
    cmark[((long long)ci[0]*n1 + ci[1])*n2 + ci[2]] = 1;
}
__global__ void ex_dilate(const int* ipar, int r, const unsigned char* cin, unsigned char* cout) {
    long long cidx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (cidx >= P) return;
    int c2 = cidx % n2, c1 = (cidx / n2) % n1, c0 = cidx / ((long long)n1*n2);
    if (c0 > n0-2 || c1 > n1-2 || c2 > n2-2) { cout[cidx]=0; return; }
    unsigned char on = 0;
    for (int a=-r;a<=r && !on;a++) for (int b=-r;b<=r && !on;b++) for (int d=-r;d<=r && !on;d++) {
        int e0=c0+a, e1=c1+b, e2=c2+d;
        if (e0<0||e1<0||e2<0||e0>n0-2||e1>n1-2||e2>n2-2) continue;
        if (cin[((long long)e0*n1+e1)*n2+e2]) on = 1;
    }
    cout[cidx] = on;
}
__global__ void ex_mark_nt(const int* ipar, const unsigned char* cmark, unsigned char* nmark, unsigned char* tmark) {
    long long cidx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (cidx >= P || !cmark[cidx]) return;
    int c2 = cidx % n2, c1 = (cidx / n2) % n1, c0 = cidx / ((long long)n1*n2);
    for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int d=0;d<2;d++) nmark[((long long)(c0+a)*n1+(c1+b))*n2+(c2+d)] = 1;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) if (a!=b) tmark[cidx*9+a*3+b] = 1;
}
// 압축 사면체 꼭짓점: locate_k 와 같은 순서 (기준 꼭짓점, +e_pr0, +e_pr1, +e_pr2)
__global__ void ex_build_tets(const int* ipar, const unsigned char* tmark, const long long* tcm, const long long* ncm, long long* tets) {
    long long s = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (s >= P*9 || !tmark[s]) return;
    long long cidx = s / 9; int a = (s % 9) / 3, b = s % 3, c = 3 - a - b;
    int v[3] = {(int)(cidx / ((long long)n1*n2)), (int)((cidx / n2) % n1), (int)(cidx % n2)};
    long long t = tcm[s];
    int pr[3] = {a, b, c};
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        tets[t*4+r] = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
    }
}
// 서브스텝 locate (locate_k 와 같은 식) + 압축 번호; 팽창 영역 밖이면 넘침 표시 szp[3]
__global__ void ex_locate(int N, const float* x, const float* gpar, const int* ipar, const long long* ncm, const long long* tcm,
                          const unsigned char* tmark, long long* rows, float* lamb, long long* pinv, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {x[p*3]-gpar[0], x[p*3+1]-gpar[1], x[p*3+2]-gpar[2]};
    const float* Ai = gpar + 3;
    float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
    int nl[3] = {n0, n1, n2};
    int ci[3]; float f[3];
    for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; f[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
    int pr[3] = {0,1,2};
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    if (f[pr[1]] < f[pr[2]]) { int t=pr[1]; pr[1]=pr[2]; pr[2]=t; }
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    float s0=f[pr[0]], s1=f[pr[1]], s2=f[pr[2]];
    lamb[p*4]=1.f-s0; lamb[p*4+1]=s0-s1; lamb[p*4+2]=s1-s2; lamb[p*4+3]=s2;
    long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
    long long tt = cell*9 + pr[0]*3 + pr[1];
    if (!tmark[tt]) { atomicAdd(&szp[3], 1); }
    pinv[p] = tmark[tt] ? tcm[tt] : 0;
    int v[3] = {ci[0], ci[1], ci[2]};
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        long long fl = ((long long)v[0]*n1 + v[1])*n2 + v[2];
        long long cm = ncm[fl];
        rows[p*4+r] = cm >= 0 ? cm : 0;
    }
}
// P2G 질량·운동량 (float 원자 합)
// tetmass = 0: 노드 질량 Σ λ m_p (기본, 암시적 판 관성 행렬의 행 합), 1: 사면체 등분 Σ_T m_T / 4 (정의 변경 -- 선형 사면체 표준 질량 집중)
__global__ void ex_p2g_mv(int N, const long long* rows, const float* lamb, const float* m, const float* v, float* mn, float* mv, int tetmass,
                          const float* Cp, const float* xp, const float* npos, float* ml) {
    // Cp (선택, --affine): 운동량에 m w C (x_n - x_p) -- 아핀 속도장은 노드 값이 정확히 v(x_n)
    // ml (선택, 절 Q (a) --rep_w lam): 속도 전달 가중을 G2P 와 같은 λ m_p 로 (운동량 mv 에 λ m_p v, 정규화 질량 ml = Σ λ m_p), 노드 질량 mn 은 tetmass 정의 그대로
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float mp = m[p];
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float w=(tetmass ? 0.25f : lamb[p*4+i])*mp; atomicAdd(&mn[nd], w);
        if (ml) { w = lamb[p*4+i]*mp; atomicAdd(&ml[nd], w); }
        float d[3] = {0,0,0};
        if (Cp) for (int b=0;b<3;b++) d[b] = npos[nd*3+b] - xp[p*3+b];
        for (int c=0;c<3;c++) { float vc = v[p*3+c]; if (Cp) vc += Cp[p*9+c*3]*d[0] + Cp[p*9+c*3+1]*d[1] + Cp[p*9+c*3+2]*d[2];
            atomicAdd(&mv[nd*3+c], w*vc); } }
}
__global__ void ex_vfin(const int* szp, int Marg, const float* mn, const float* mv, float* vn, const float* ml) {
    // ml (선택, --rep_w lam): 속도 = 운동량 / Σλm (그 값이 0 인 노드 -- 모든 입자의 λ 가 0 -- 는 속도 0)
    int M = szp ? szp[0] : Marg;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float m = ml ? ml[n] : mn[n];
    for (int c=0;c<3;c++) vn[n*3+c] = m > 1e-30f ? mv[n*3+c]/m : 0.f;
}
// 분할 (CudaTetqK 와 같은 정의: 예측 변위 u0 = Σλ (H v_n + H² g) 의 끝 높이 s/ε ≤ 20 이면 바닥 후보) + 항등 입자만 사면체 V·V F 합
__global__ void ex_split(int N, const float* x, const long long* rows, const float* lamb, const long long* pinv, const float* vn, float H,
                         float g0, float g1, float g2, int floor, float p0, float p1, float p2, float q0, float q1, float q2, float eps_sp, float eps_st,
                         const float* vol, const float* Fp, float* VTs, float* FTs, int* nslot, long long* nidx, int* szp, int keep, int near_in_tet) {
    // near_in_tet = 1 (--floor vel, 절 O): 바닥 후보도 사면체 F̄ 에 넣는다 (에너지 바닥이 없으므로 탄성은 사면체가 갖는다) -- 후보 표시는 접촉 판정용
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bool near = false;
    if (keep) {
        // 절 P (--rep full): 프레임 안 재 P2G 는 keep=1 -- 바닥 후보 표시는 프레임 것 그대로, near_in_tet=1 (--floor vel) 이면 후보도 F̄ 에
        if (nslot[p] < 0 || near_in_tet) { long long t = pinv[p]; float V = vol[p]; atomicAdd(&VTs[t], V); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]); }
        return;
    }
    if (floor) {
        float u[3]={0,0,0}, gg[3]={g0,g1,g2};
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int c=0;c<3;c++) u[c] += lm*(H*vn[nd*3+c] + H*H*gg[c]); }
        float s = (x[p*3]+u[0]-p0)*q0 + (x[p*3+1]+u[1]-p1)*q1 + (x[p*3+2]+u[2]-p2)*q2;
        if (eps_sp > 0.f && s/eps_sp <= 20.f) near = true;
        if (eps_st > 0.f && s/eps_st <= 20.f) near = true;
    }
    if (near) {
        int q = atomicAdd(&szp[2], 1);
        nidx[q] = p; nslot[p] = q;
        if (near_in_tet) { long long t = pinv[p]; float V = vol[p]; atomicAdd(&VTs[t], V); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]); }
    } else {
        nslot[p] = -1;
        long long t = pinv[p]; float V = vol[p];
        atomicAdd(&VTs[t], V);
        for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]);
    }
}
__global__ void ex_ffin(const int* szp, int Targ, const float* VTs, const float* FTs, float* FT) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    for (int k=0;k<9;k++) FT[t*9+k] = V > 0.f ? FTs[t*9+k]/V : ((k%4==0)?1.f:0.f);
}
__global__ void ex_gather(const int* szp, int nnarg, const long long* nidx, const float* x, const float* Fp, float* xs, float* Fn) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    for (int k=0;k<3;k++) xs[q*3+k]=x[p*3+k];
    for (int k=0;k<9;k++) Fn[q*9+k]=Fp[p*9+k];
}
// 사면체 ∇λ float (tet_geo 의 float 판)
__global__ void ex_tet_geo32(const int* szp, const long long* tets, const float* npos, float* gl) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    double X[4][3];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) X[i][c] = (double)npos[tets[t*4+i]*3+c];
    double D[9];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) D[r*3+c] = X[c+1][r]-X[0][r];
    double C[9]; cof3(D, C);
    double det = D[0]*C[0]+D[1]*C[1]+D[2]*C[2];
    double g[12];
    for (int i=0;i<3;i++) for (int b=0;b<3;b++) g[(i+1)*3+b] = C[b*3+i]/det;
    for (int b=0;b<3;b++) g[b] = -(g[3+b]+g[6+b]+g[9+b]);
    for (int k=0;k<12;k++) gl[t*12+k] = (float)g[k];
}
// ======== sub 판 융합 커널 (최적화: 값은 같은 식, 소비하며 비우기 -> 채우기 커널 없음) ========
// 사면체 힘 + F̄ 나누기 융합: F̄ = FTs/VTs 를 그 자리에서, 쓴 뒤 VTs·FTs 를 0 으로 (다음 서브스텝 P2G 가 다시 쌓는다)
__global__ void ex_tet_force_sub(const int* szp, const long long* tets, const float* gl, float* FTs, float* VTs, float mu, float lam, float* f) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    if (!(V > 0.f)) return;
    float Fb[9]; for (int k=0;k<9;k++) { Fb[k] = FTs[t*9+k]/V; FTs[t*9+k] = 0.f; }
    VTs[t] = 0.f;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float X[9]; for (int k=0;k<9;k++) X[k]=Fb[k];
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// 노드 걸음 + 속도 나누기 융합: v = mv/m + h_s((f - gn)/m + g), 쓴 뒤 mn·mv·f·gn 을 0 으로
__global__ void ex_node_step_sub(const int* szp, float* mn, float* mv, float* f, float* gn, float* dU, float g0, float g1, float g2, float hs,
                                 const unsigned char* fixd, const float* vbc) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        float v;
        if (fixd && fixd[n]) v = vbc[n*3+c];
        else if (m > 0.f) v = mv[n*3+c]/m + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]);
        else v = 0.f;
        dU[n*3+c] = hs*v;
        mv[n*3+c] = 0.f; f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f;
    }
    mn[n] = 0.f;
}
// 선형 G2P (float) + 다음 서브스텝 locate + P2G (질량·운동량, 항등 입자 V·VF) 융합
__global__ void ex_g2p_p2g(int N, const float* xa, const float* Fa, const float* dU, long long* rows, float* lamb, long long* pinv, const float* glT,
                           float hs, float damp, float* xb, float* vb, float* Fb, int do_p2g,
                           const float* gpar, const int* ipar, const long long* ncm, const long long* tcm, const unsigned char* tmark,
                           const float* m, const float* vol, const int* nslot, int tetmass, float* mn, float* mv, float* VTs, float* FTs, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float u[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    const float* gl = glT + pinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++){ float Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
    float xn[3], vn_[3], F[9], F2[9];
    for (int k=0;k<3;k++) { xn[k] = xa[p*3+k] + u[k]; vn_[k] = u[k]/hs*damp; xb[p*3+k] = xn[k]; vb[p*3+k] = vn_[k]; }
    for (int k=0;k<9;k++) F[k]=Fa[p*9+k];
    mm3(A, F, F2);
    for (int k=0;k<9;k++) Fb[p*9+k] = F2[k];
    if (!do_p2g) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {xn[0]-gpar[0], xn[1]-gpar[1], xn[2]-gpar[2]};
    const float* Ai = gpar + 3;
    float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
    int nl[3] = {n0, n1, n2};
    int ci[3]; float fr[3];
    for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; fr[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
    int pr[3] = {0,1,2};
    if (fr[pr[0]] < fr[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    if (fr[pr[1]] < fr[pr[2]]) { int t=pr[1]; pr[1]=pr[2]; pr[2]=t; }
    if (fr[pr[0]] < fr[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    float s0=fr[pr[0]], s1=fr[pr[1]], s2=fr[pr[2]];
    float lmb[4] = {1.f-s0, s0-s1, s1-s2, s2};
    long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
    long long tt = cell*9 + pr[0]*3 + pr[1];
    if (!tmark[tt]) atomicAdd(&szp[3], 1);
    long long t = tmark[tt] ? tcm[tt] : 0;
    pinv[p] = t;
    int v[3] = {ci[0], ci[1], ci[2]};
    float mp = m[p];
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        long long cm = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
        long long nd = cm >= 0 ? cm : 0;
        rows[p*4+r] = nd; lamb[p*4+r] = lmb[r];
        float w = (tetmass ? 0.25f : lmb[r])*mp;
        atomicAdd(&mn[nd], w);
        for (int c=0;c<3;c++) atomicAdd(&mv[nd*3+c], w*vn_[c]);
    }
    if (nslot[p] < 0) {
        float V = vol[p];
        atomicAdd(&VTs[t], V);
        for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*F2[k]);
    }
}

// 정렬된 입자 (같은 사면체 연속) 의 워프 구간 합: 같은 키가 이어진 레인끼리 포함 누적합 (5 단계), 구간 끝 레인만 원자 합
__device__ __forceinline__ float seg_scan(float v, int key, int lane) {
    for (int d=1; d<32; d<<=1) {
        float o = __shfl_up_sync(0xffffffff, v, d);
        int ko = __shfl_up_sync(0xffffffff, key, d);
        if (lane >= d && ko == key) v += o;
    }
    return v;
}
// ex_g2p_p2g 와 같은 식, P2G 원자 합만 워프 구간 합으로 (입자는 프레임 시작에 사면체 번호로 정렬) -- 모든 레인이 끝까지 간다
__global__ void ex_g2p_p2g_seg(int N, const float* xa, const float* Fa, const float* dU, long long* rows, float* lamb, long long* pinv, const float* glT,
                           float hs, float damp, float* xb, float* vb, float* Fb, int do_p2g,
                           const float* gpar, const int* ipar, const long long* ncm, const long long* tcm, const unsigned char* tmark,
                           const float* m, const float* vol, const int* nslot, int tetmass, float* mn, float* mv, float* VTs, float* FTs, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = p < N;
    float vn_[3] = {0,0,0}, F2[9] = {0,0,0,0,0,0,0,0,0}, xn[3] = {0,0,0};
    if (on) {
        float u[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
        const float* gl = glT + pinv[p]*12;
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++){ float Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
        float F[9];
        for (int k=0;k<3;k++) { xn[k] = xa[p*3+k] + u[k]; vn_[k] = u[k]/hs*damp; xb[p*3+k] = xn[k]; vb[p*3+k] = vn_[k]; }
        for (int k=0;k<9;k++) F[k]=Fa[p*9+k];
        mm3(A, F, F2);
        for (int k=0;k<9;k++) Fb[p*9+k] = F2[k];
    }
    if (!do_p2g) return;
    int key = -1 - lane;                                       // 꺼진 레인은 고유 키
    long long nd4[4] = {0,0,0,0};
    float wv[4] = {0,0,0,0};
    float V = 0.f;
    long long t = 0;
    if (on) {
        int n0=ipar[0], n1=ipar[1], n2=ipar[2];
        float q[3] = {xn[0]-gpar[0], xn[1]-gpar[1], xn[2]-gpar[2]};
        const float* Ai = gpar + 3;
        float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
        int nl[3] = {n0, n1, n2};
        int ci[3]; float fr[3];
        for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; fr[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
        int pr[3] = {0,1,2};
        if (fr[pr[0]] < fr[pr[1]]) { int tt_=pr[0]; pr[0]=pr[1]; pr[1]=tt_; }
        if (fr[pr[1]] < fr[pr[2]]) { int tt_=pr[1]; pr[1]=pr[2]; pr[2]=tt_; }
        if (fr[pr[0]] < fr[pr[1]]) { int tt_=pr[0]; pr[0]=pr[1]; pr[1]=tt_; }
        float s0=fr[pr[0]], s1=fr[pr[1]], s2=fr[pr[2]];
        float lmb[4] = {1.f-s0, s0-s1, s1-s2, s2};
        long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
        long long tt = cell*9 + pr[0]*3 + pr[1];
        if (!tmark[tt]) atomicAdd(&szp[3], 1);
        t = tmark[tt] ? tcm[tt] : 0;
        pinv[p] = t;
        key = (int)t;
        int v[3] = {ci[0], ci[1], ci[2]};
        float mp = m[p];
        for (int r=0;r<4;r++) {
            if (r>0) v[pr[r-1]] += 1;
            long long cm = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
            nd4[r] = cm >= 0 ? cm : 0;
            rows[p*4+r] = nd4[r]; lamb[p*4+r] = lmb[r];
            wv[r] = (tetmass ? 0.25f : lmb[r])*mp;
        }
        if (nslot[p] < 0) V = vol[p];
    }
    int kn = __shfl_down_sync(0xffffffff, key, 1);
    bool last = (lane == 31) || (kn != key);
    // 같은 사면체 = 같은 네 노드 (같은 순서) 이므로 키 하나로 노드 합도 묶는다
    for (int r=0;r<4;r++) {
        float sw = seg_scan(wv[r], key, lane);
        float sx = seg_scan(wv[r]*vn_[0], key, lane), sy = seg_scan(wv[r]*vn_[1], key, lane), sz = seg_scan(wv[r]*vn_[2], key, lane);
        if (on && last) { long long nd = nd4[r]; atomicAdd(&mn[nd], sw); atomicAdd(&mv[nd*3], sx); atomicAdd(&mv[nd*3+1], sy); atomicAdd(&mv[nd*3+2], sz); }
    }
    float sV = seg_scan(V, key, lane);
    float sF[9]; for (int k=0;k<9;k++) sF[k] = seg_scan(V*F2[k], key, lane);
    if (on && last && sV > 0.f) { atomicAdd(&VTs[t], sV); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], sF[k]); }
}
// ======== 절 S: lag 판 P2G 를 2 차 B-스플라인으로 (--p2g bs, 사용자 10-29) ========
// 격자 = 사면체 메시 노드 격자 (Kuhn 정육면체 격자, 지표 좌표 y = Ai (x - lo), 노드 = 정수 y, 셀 c = [c, c+1]³, 셀 중심 = c + ½).
// 축마다 2 차 B-스플라인 N(t) (|t| < ½: ¾ - t², ½ ≤ |t| < 3/2: ½ (3/2 - |t|)²), 분할 1 (Σ_k N(y - k) = 1, 어떤 정수 격자에서나).
//   노드:  base = floor(y - ½), f = y - base, 노드 base..base+2 의 w = [½(1.5-f)², ¾-(f-1)², ½(f-½)²]   (fem_bs_cuda 와 같은 식)
//   셀:   셀 중심 격자는 y_c = y - ½ 의 정수 격자 -> 같은 식을 y - ½ 에 (셀 base_c..base_c+2 = 주변 3³ 셀)
__device__ __forceinline__ void bsl_w(float y, int& b, float* W, float* D) {
    b = (int)floorf(y - 0.5f);
    float f = y - (float)b;
    W[0] = 0.5f*(1.5f-f)*(1.5f-f); W[1] = 0.75f-(f-1.f)*(f-1.f); W[2] = 0.5f*(f-0.5f)*(f-0.5f);
    D[0] = f-1.5f; D[1] = -2.f*(f-1.f); D[2] = f-0.5f;
}
__device__ __forceinline__ void bsl_y(const float* x, const float* gpar, float* y) {
    float q[3] = {x[0]-gpar[0], x[1]-gpar[1], x[2]-gpar[2]};
    const float* Ai = gpar + 3;
    for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
}
// 프레임 시작 P2G (입자마다 한 번):
//   m_n += w(x_n - x_p) m_p,  (mv)_n += w m_p v_p                                  (27 노드, 압축 번호 ncm; 메시 밖 노드면 넘침 szp[3])
//   cV_c += w(x_c - x_p) V_p,  cVF_c += w(x_c - x_p) V_p F_p                         (27 셀, 밀집 셀 번호; 메시 밖 셀이면 넘침)
//   바닥 후보 (floor=1): 프레임 끝 예측 높이 (x + H v + H² g - p)·n ≤ margin 이면 nidx 에 (접촉 판정 대상 표시만 -- 판정은 서브스텝마다 정확히)
__global__ void bsl_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const float* gpar, const int* ipar,
                        const long long* ncm, const unsigned char* cmk, float* mn, float* mv, float* cV, float* cVF, int* szp,
                        int floor, float H, float g0, float g1, float g2, float p0, float p1, float p2, float q0, float q1, float q2, float margin,
                        long long* nidx, int nrm, float* bst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float xp[3] = {x[p*3], x[p*3+1], x[p*3+2]}, vp[3] = {v[p*3], v[p*3+1], v[p*3+2]};
    float y[3]; bsl_y(xp, gpar, y);
    float mp = m[p], Vp = vol[p];
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    int ovf = 0;
    // 절 T (nrm = 1): 메시에 있는 노드에만 쓰고, 입자마다 메시 안 노드 가중 합 Win 으로 나눠 Σ_n w = 1 (총 질량·운동량 보존). 메시 밖 몫은 넘침이 아니다.
    float sn = 1.f;
    if (nrm) {
        float Win = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
            if (ncm[((long long)a0*n1 + a1)*n2 + a2] < 0) continue;
            Win += W[0][i]*W[1][j]*W[2][l];
        }
        if (Win > 0.f) { sn = 1.f/Win; if (bst) atomicAdd(&bst[2], (1.f - Win)*mp); } else ovf = 1;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) { if (!nrm) ovf = 1; continue; }
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) { if (!nrm) ovf = 1; continue; }
        float w = W[0][i]*W[1][j]*W[2][l]*sn*mp;
        atomicAdd(&mn[nd], w);
        for (int c=0;c<3;c++) atomicAdd(&mv[nd*3+c], w*vp[c]);
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[p*9+k];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    // 절 T (nrm = 1): 셀도 메시에 있는 셀에만, 메시 안 셀 가중 합 Cin 으로 나눠 Σ_c w = 1 (총 부피 보존)
    float sc = 1.f;
    if (nrm) {
        float Cin = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
            if (!cmk[((long long)a0*n1 + a1)*n2 + a2]) continue;
            Cin += W[0][i]*W[1][j]*W[2][l];
        }
        if (Cin > 0.f) { sc = 1.f/Cin; if (bst) atomicAdd(&bst[3], (1.f - Cin)*Vp); } else ovf = 1;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) { if (!nrm) ovf = 1; continue; }
        long long c = ((long long)a0*n1 + a1)*n2 + a2;
        if (!cmk[c]) { if (!nrm) ovf = 1; continue; }
        float w = W[0][i]*W[1][j]*W[2][l]*sc*Vp;
        atomicAdd(&cV[c], w);
        for (int k=0;k<9;k++) atomicAdd(&cVF[c*9+k], w*Fp[k]);
    }
    if (ovf) atomicAdd(&szp[3], 1);
    if (floor) {
        float gg[3] = {g0, g1, g2};
        float s = 0.f, nn[3] = {q0, q1, q2}, pp[3] = {p0, p1, p2};
        for (int c=0;c<3;c++) s += (xp[c] + H*vp[c] + H*H*gg[c] - pp[c])*nn[c];
        if (s <= margin) { int q = atomicAdd(&szp[2], 1); nidx[q] = p; }
    }
}
// 사면체 상태: 셀 (정육면체 = Kuhn 사면체 6 개, 같은 부피) 의 F̄_c = cVF_c / cV_c, V_T = cV_c / 6.
//   네 노드 중 질량 0 인 노드가 있으면 그 사면체는 힘에서 뺀다 (V_T = 0; 질량 0 노드에 힘을 걸 수 없다) -- 뺀 부피는 bst[0] 에 기록.
__global__ void bsl_tfin(const int* szp, const long long* tcell, const long long* tets, const float* mn, const float* cV, const float* cVF,
                         float* VTs, float* FT, float* bst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long c = tcell[t];
    float V = cV[c];
    if (V > 0.f) {
        bool z = false;
        for (int i=0;i<4;i++) if (!(mn[tets[t*4+i]] > 0.f)) z = true;
        if (z) { VTs[t] = 0.f; atomicAdd(&bst[0], V/6.f); atomicAdd(&bst[1], 1.f); for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f; return; }
        VTs[t] = V/6.f;
        for (int k=0;k<9;k++) FT[t*9+k] = cVF[c*9+k]/V;
    } else {
        VTs[t] = 0.f;
        for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f;
    }
}
// 사면체 -> 밀집 셀 번호 (격자 만들 때)
__global__ void ex_tet_cell(const int* ipar, const unsigned char* tmark, const long long* tcm, long long* tcell) {
    long long s = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long P = (long long)ipar[0]*ipar[1]*ipar[2];
    if (s >= P*9 || !tmark[s]) return;
    tcell[tcm[s]] = s / 9;
}
// 입자의 프레임 변위 보간: bs = 1 -> u = Σ w(x_n - x_p⁰) U_n (B-스플라인), 0 -> Σλ U (사면체)
__device__ __forceinline__ void bsl_interp_u(long long p, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                             const float* gpar, const int* ipar, const long long* ncm, int bs, float* u, int nrm) {
    u[0]=u[1]=u[2]=0.f;
    if (!bs) {
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm*Ucum[nd*3+a]; }
        return;
    }
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    float Win = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) continue;
        float w = W[0][i]*W[1][j]*W[2][l];
        Win += w;
        for (int a=0;a<3;a++) u[a] += w*Ucum[nd*3+a];
    }
    if (nrm && Win > 0.f) for (int a=0;a<3;a++) u[a] /= Win;                     // 절 T: P2G 와 같은 정규화 (메시 안 노드 가중 합)
}
// 속도 바닥 접촉 (절 O 와 같은 판정·분배, 질량 가중만 B-스플라인): 바닥 후보 입자의 지금 위치 x_p⁰ + u (G2P 와 같은 보간) 높이 < 0 이면
//   P2G 와 같은 질량 가중 w m_p 를 27 노드 acc 에 -> ex_node_step_lag 가 frac 고착 (1 - acc/m) + 비관통
__global__ void bsl_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                const float* gpar, const int* ipar, const long long* ncm, const float* m, int g2pbs,
                                float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon, int nrm) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3]; bsl_interp_u(p, x, Ucum, rows, lamb, gpar, ipar, ncm, g2pbs, u, nrm);
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    int N0=ipar[0], N1=ipar[1], N2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    float mp = m[p];
    if (nrm) {                                                                   // 절 T: P2G 와 같은 정규화 가중 (메시 안 노드 합 = 1)
        float Win = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>N0-1||a1>N1-1||a2>N2-1) continue;
            if (ncm[((long long)a0*N1 + a1)*N2 + a2] < 0) continue;
            Win += W[0][i]*W[1][j]*W[2][l];
        }
        if (Win > 0.f) mp /= Win;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-1||a1>N1-1||a2>N2-1) continue;
        long long nd = ncm[((long long)a0*N1 + a1)*N2 + a2];
        if (nd < 0) continue;
        atomicAdd(&acc[nd], W[0][i]*W[1][j]*W[2][l]*mp);
    }
    if (ncon) atomicAdd(ncon, 1);
}
// 프레임 끝 B-스플라인 G2P (질량 가중과 같은 w): x = x⁰ + Σ w U_n, v = Σ w v_n, F ← (I + Σ U_n ⊗ ∇w_n(x⁰)) F
__global__ void bsl_g2p(int N, const float* x, const float* Ucum, const float* Vn, const float* gpar, const int* ipar, const long long* ncm,
                        const float* Fp, float* x2, float* v2, float* F2, int nrm) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    const float* Ai = gpar + 3;
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    // 절 T (nrm = 1): 메시 안 노드만, 정규화 가중 w~ = w / Win, 기울기 ∇w~ = (∇w - w~ ∇Win) / Win (Σ ∇w~ = 0 -- 강체 이동이면 A = I)
    double Win = 1.0, gW[3] = {0,0,0};
    if (nrm) {
        Win = 0.0;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
            if (ncm[((long long)a0*n1 + a1)*n2 + a2] < 0) continue;
            Win += (double)(W[0][i]*W[1][j]*W[2][l]);
            float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
            for (int c=0;c<3;c++) gW[c] += gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c];
        }
        if (!(Win > 0.0)) { Win = 1.0; gW[0]=gW[1]=gW[2]=0.0; }
    }
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) continue;
        double w = (double)(W[0][i]*W[1][j]*W[2][l]) / Win;
        float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
        double gx[3]; for (int c=0;c<3;c++) gx[c] = ((double)(gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c]) - w*gW[c]) / Win;
        for (int a=0;a<3;a++) { double Ua = Ucum[nd*3+a]; u[a] += w*Ua; vv[a] += w*(double)Vn[nd*3+a]; for (int c=0;c<3;c++) A[a*3+c] += Ua*gx[c]; }
    }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}
// ======== 절 W (사용자 10-29): 셀 중심 커널 하나로 질량·운동량·F̄·V 모두 셀에, 노드는 일반 FEM 집중 질량 m_T/4 ========
// 셀 가중 = 절 T 의 셀 가중 그대로 (셀 중심 2 차 B-스플라인, 메시 안 셀만, 입자마다 메시 안 합 Cin 으로 정규화 -> Σ_c w~ = 1):
//   cM_c += w~ m_p, cMV_c += w~ m_p v_p, cV_c += w~ V_p, cVF_c += w~ V_p F_p.  바닥 후보 표시는 bsl_p2g 와 같다.
//   bst[2] = 메시 밖으로 갈 몫의 질량 Σ(1-Cin) m_p, bst[3] = 부피 Σ(1-Cin) V_p (정규화로 메시 안 셀에 옮긴 양)
__global__ void bsc_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const float* gpar, const int* ipar,
                        const unsigned char* cmk, float* cM, float* cMV, float* cV, float* cVF, int* szp,
                        int floor, float H, float g0, float g1, float g2, float p0, float p1, float p2, float q0, float q1, float q2, float margin,
                        long long* nidx, float* bst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float xp[3] = {x[p*3], x[p*3+1], x[p*3+2]}, vp[3] = {v[p*3], v[p*3+1], v[p*3+2]};
    float y[3]; bsl_y(xp, gpar, y);
    float mp = m[p], Vp = vol[p];
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    float Cin = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
        if (!cmk[((long long)a0*n1 + a1)*n2 + a2]) continue;
        Cin += W[0][i]*W[1][j]*W[2][l];
    }
    if (!(Cin > 0.f)) { atomicAdd(&szp[3], 1); return; }
    float sc = 1.f/Cin;
    if (bst) { atomicAdd(&bst[2], (1.f - Cin)*mp); atomicAdd(&bst[3], (1.f - Cin)*Vp); }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[p*9+k];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
        long long c = ((long long)a0*n1 + a1)*n2 + a2;
        if (!cmk[c]) continue;
        float w = W[0][i]*W[1][j]*W[2][l]*sc;
        float wm = w*mp, wv = w*Vp;
        atomicAdd(&cM[c], wm);
        for (int k=0;k<3;k++) atomicAdd(&cMV[c*3+k], wm*vp[k]);
        atomicAdd(&cV[c], wv);
        for (int k=0;k<9;k++) atomicAdd(&cVF[c*9+k], wv*Fp[k]);
    }
    if (floor) {
        float gg[3] = {g0, g1, g2};
        float s = 0.f, nn[3] = {q0, q1, q2}, pp[3] = {p0, p1, p2};
        for (int c=0;c<3;c++) s += (xp[c] + H*vp[c] + H*H*gg[c] - pp[c])*nn[c];
        if (s <= margin) { int q = atomicAdd(&szp[2], 1); nidx[q] = p; }
    }
}
// 셀 -> 사면체 -> 노드 (일반 FEM 집중 질량): 셀 = 같은 부피 Kuhn 사면체 6 개 -> m_T = cM_c/6, (mv)_T = cMV_c/6, V_T = cV_c/6, F̄_T = cVF_c/cV_c;
//   노드 m_n = Σ_{T∋n} m_T/4, (mv)_n = Σ_{T∋n} (mv)_T/4  (속도는 ex_vfin 이 (mv)_n / m_n)
__global__ void bsc_tfin(const int* szp, const long long* tcell, const long long* tets, const float* cM, const float* cMV, const float* cV, const float* cVF,
                         float* VTs, float* FT, float* mn, float* mv) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long c = tcell[t];
    float V = cV[c];
    if (V > 0.f) { VTs[t] = V/6.f; for (int k=0;k<9;k++) FT[t*9+k] = cVF[c*9+k]/V; }
    else { VTs[t] = 0.f; for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f; }
    float q = cM[c]*(1.f/24.f);
    float pv[3] = {cMV[c*3]*(1.f/24.f), cMV[c*3+1]*(1.f/24.f), cMV[c*3+2]*(1.f/24.f)};
    if (q > 0.f) for (int i=0;i<4;i++) { long long nd = tets[t*4+i]; atomicAdd(&mn[nd], q); for (int k=0;k<3;k++) atomicAdd(&mv[nd*3+k], pv[k]); }
}
// 속도 바닥 접촉 (절 O·T 와 같은 판정 -- 지금 위치 = G2P 와 같은 보간; 분배는 P2G 와 같은 질량 경로): 접촉 입자 m_p 를 셀 가중 w~_pc 로 셀 accC 에,
//   bsc_acc_tet 이 사면체마다 accC_c/24 를 네 노드 acc 에 (= 노드 질량과 같은 m_T/4 경로) -> ex_node_step_lag 의 frac 고착 (1 - acc/m) + 비관통
__global__ void bsc_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                const float* gpar, const int* ipar, const long long* ncm, const unsigned char* cmk, const float* m, int g2pbs,
                                float p0, float p1, float p2, float n0, float n1, float n2, float* accC, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3]; bsl_interp_u(p, x, Ucum, rows, lamb, gpar, ipar, ncm, g2pbs, u, 1);
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    int N0=ipar[0], N1=ipar[1], N2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    float Cin = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-2||a1>N1-2||a2>N2-2) continue;
        if (!cmk[((long long)a0*N1 + a1)*N2 + a2]) continue;
        Cin += W[0][i]*W[1][j]*W[2][l];
    }
    if (!(Cin > 0.f)) return;
    float mp = m[p]/Cin;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-2||a1>N1-2||a2>N2-2) continue;
        long long c = ((long long)a0*N1 + a1)*N2 + a2;
        if (!cmk[c]) continue;
        atomicAdd(&accC[c], W[0][i]*W[1][j]*W[2][l]*mp);
    }
    if (ncon) atomicAdd(ncon, 1);
}
__global__ void bsc_acc_tet(const int* szp, const long long* tcell, const long long* tets, const float* accC, float* acc) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float q = accC[tcell[t]]*(1.f/24.f);
    if (q > 0.f) for (int i=0;i<4;i++) atomicAdd(&acc[tets[t*4+i]], q);
}
// 입자 재배열 (프레임 시작): dst[i] = src[perm[i]] (행 길이 k 의 float)
__global__ void ex_gather_rows(int N, int k, const long long* perm, const float* src, float* dst) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long long)N*k) return;
    long long r = i / k; int c = i % k;
    dst[i] = src[perm[r]*k + c];
}
// 되돌리기 (프레임 끝): dst[perm[i]] = src[i]
__global__ void ex_scatter_rows(int N, int k, const long long* perm, const float* src, float* dst) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long long)N*k) return;
    long long r = i / k; int c = i % k;
    dst[perm[r]*k + c] = src[i];
}
// ======== 절 V: 노드 적분 FEM (사용자 10-29 "그것도 일단 해 보라고 해") -- 평균 노드 변형 사면체 (Bonet & Burton 1998, Dohrmann et al. 2000) ========
// 틀·질량·바닥·F̄ 상태·G2P 는 절 R lag 판 그대로. 바뀐 것은 내부력 하나:
//   서브스텝마다
//   (1) nint_cell : F_T = A_T(U) F̄_T,  사면체 버퍼 X_T = (V_T/4) F_T                (V_T > 0 사면체만)
//   (2) nint_node : 노드 -> 사면체 목록으로 S_n = Σ_{T∋n} X_T, V_n = Σ V_T/4, F_n = S_n / V_n (= Σ V_T F_T / Σ V_T),
//                   P_n = P_FCR(F_n) (ex_tet_force_R 와 같은 FCR 식)
//   (3) nint_force: f_a -= (V_T/4) [(1-β) Σ_{n∈T} P_n + 4β P_FCR(F_T)] F̄_Tᵀ ∇λ_a
//   = 에너지 E(U) = (1-β) Σ_n V_n ψ(F_n) + β Σ_T V_T ψ(F_T) 의 -∂E/∂U_a 그대로
//     (∂F_n/∂U_a·δ = Σ_{T∋n,a} (V_T/4)/V_n δ ⊗ (F̄_Tᵀ ∇λ_a) 이라 V_n 이 상쇄된다). β = 0 이 순수 노드 적분, β > 0 은 셀 적분 혼합 (영에너지 모드 안정화 선택지).
//   사면체마다 Σ_a ∇λ_a = 0 -> 내력 합 0 (운동량 보존), E 가 회전 불변 -> 각운동량 보존.
__device__ __forceinline__ void fcr_P(const float* X, float mu, float lam, float* P) {
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
}
__device__ __forceinline__ float nint_AX(int t, const long long* tets, const float* U, const float* gl, const float* FT, long long* n, float* G, float* X) {
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    mm3(A, Fb, X);
    return det3(A);
}
// (1) 사면체마다 (V_T/4) F_T 를 사면체 버퍼에 (원자 합 없음 -- 노드 쪽이 목록으로 모은다; 목록 순서는 ex_node_tets 원자 슬롯이라 합 순서는 실행마다 다를 수 있다)
__global__ void nint_cell(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, float* XT, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12], X[9];
    float dA = nint_AX(t, tets, U, gl, FT, n, G, X);
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float q = 0.25f*V;
    for (int k=0;k<9;k++) XT[t*9+k] = q*X[k];
}
// (2) 노드마다 노드 -> 사면체 목록 (ex_node_tets, 프레임마다) 으로 S_n = Σ (V_T/4) F_T, V_n = Σ V_T/4 -> F_n = S_n / V_n, P_n = P_FCR(F_n)
__global__ void nint_node(const int* szp, const int* ntc, const int* nt, const float* VT, const float* XT, float* Pn, float* Vn, float mu, float lam) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    int k = min(ntc[n], 24);
    float S[9] = {0,0,0,0,0,0,0,0,0}, v = 0.f;
    for (int s=0;s<k;s++) {
        int t = nt[n*24+s] >> 2;
        float V = VT[t];
        if (!(V > 0.f)) continue;
        v += 0.25f*V;
        for (int c=0;c<9;c++) S[c] += XT[(long long)t*9+c];
    }
    float P[9] = {0,0,0,0,0,0,0,0,0};
    if (v > 0.f) {
        float X[9]; for (int c=0;c<9;c++) X[c] = S[c]/v;
        fcr_P(X, mu, lam, P);
    }
    for (int c=0;c<9;c++) Pn[n*9+c] = P[c];
    if (Vn) Vn[n] = v;
}
__global__ void nint_force(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, const float* Pn,
                           float mu, float lam, float beta, float* f) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float S[9];
    for (int k=0;k<9;k++) S[k] = (1.f-beta)*(Pn[n[0]*9+k] + Pn[n[1]*9+k] + Pn[n[2]*9+k] + Pn[n[3]*9+k]);
    if (beta > 0.f) {
        long long n2[4]; float G2[12], X[9], PT[9];
        nint_AX(t, tets, U, gl, FT, n2, G2, X);
        fcr_P(X, mu, lam, PT);
        for (int k=0;k<9;k++) S[k] += 4.f*beta*PT[k];
    }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float Gm[9]; mmt3(S, Fb, Gm);
    float q = 0.25f*V;
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -q*s);
    }
}
// ======== 절 AA: 셀 에너지 = 셀 안 입자 에너지 평균의 2 차 근사 (사용자 10-30 "셀 평균 ψ 의 근사로") ========
//   E_T = V_T [ ψ(X̄) + ½ ⟨∂²ψ(X̄), S⟩ ],  X̄ = A_T F̄_T,  S = (A⊗I) Σ_F (A⊗I)ᵀ  (= vec(A F_p) 의 공분산, vec 은 행 우선 9)
//   Σ_F = Σ_p V_p vec(F_p - F̄) vec(F_p - F̄)ᵀ / V_T  -- 프레임 시작 P2G 에서 한 번 (두 번째 패스, 대칭 9x9 위 삼각 45 개 저장)
//   -∂E_T/∂x = -V_T [ (P(X̄) + ∂_X q) F̄ᵀ + G_S ] ∇λ_i
//     G_S[a][b] = Σ_c Σ_m H[(a,c)][m] T[m][(b,c)],  T = (A⊗I) Σ_F   (∂S/∂A 몫, H = ∂²ψ(X̄))
//     ∂_X q (S 고정, ∂³ψ 몫 -- 근사 없음, 닫힌 식):
//       λ 몫 q_λ = λ/2 [ cᵀ S c + 2 (det X - sg) X:C_S ]  (c = cof X, C_S = Σ_q cof(s_q) = S 의 2x2 소행렬식 합)
//            ∂_X q_λ = λ [ coflin(X, Sc) + (X:C_S) cof X + (det X - sg) C_S ]
//       μ 몫 q_μ = μ [ tr S - h ],  h = tr(M⁻¹ B S Bᵀ),  M = tr(Ŝ) I - Ŝ (X = R Ŝ 극분해), B s = ax(Rᵀ s - sᵀ R)
//            (s : DR[s] = aᵀ M⁻¹ a 에서 -- FCR 의 μ 헤시안 2μ(I - DR)),  ∂h/∂X_m = -⟨dM_m, M⁻¹KM⁻¹⟩ + 2 Σ_i ax_i(dR_mᵀ y_i - y_iᵀ dR_m)
//            (dR = R [w]×, w = M⁻¹ B e_m, dŜ = sym(Rᵀ e_m - [w]× Ŝ), K = B S Bᵀ, y_i = (S Bᵀ M⁻¹) 의 i 열)
//   H: hmode 0 = 기존 해석 헤시안 장치 함수 kmanual (전진 미분 극분해 10 회), 1 = 같은 헤시안의 닫힌 식 (H s = 2μ(s - R[M⁻¹ B s]×) + λ[(c:s) c + (det - sg) coflin(X, s)])
//   σ_min(X̄) ≤ 0.01 이거나 tr Σ_F = 0 (입자 1 개 셀 등) 이면 보정 0 (= 기존 ψ(X̄)).
__device__ __forceinline__ int cq_si(int i, int j) { if (i > j) { int t=i; i=j; j=t; } return i*9 - (i*(i-1))/2 + (j-i); }
__global__ void ex_cov_acc(int N, const long long* pinv, const int* nslot, int nit, const float* vol, const float* Fp, const float* FT, float* CS) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (!(nslot[p] < 0 || nit)) return;
    long long t = pinv[p];
    float d[9]; for (int k=0;k<9;k++) d[k] = Fp[p*9+k] - FT[t*9+k];
    float V = vol[p];
    int s = 0;
    for (int i=0;i<9;i++) for (int j=i;j<9;j++) { atomicAdd(&CS[t*45+s], V*d[i]*d[j]); s++; }
}
__global__ void ex_cov_fin(const int* szp, const float* VT, float* CS) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    float iv = V > 0.f ? 1.f/V : 0.f;
    for (int k=0;k<45;k++) CS[(long long)t*45+k] *= iv;
}
__device__ __forceinline__ void cq_ax(const float* Z, float* a) { a[0] = Z[7]-Z[5]; a[1] = Z[2]-Z[6]; a[2] = Z[3]-Z[1]; }   // ax(Z - Zᵀ)
__device__ __forceinline__ void cq_Bs(const float* R, const float* s, float* a) {   // a = ax(Rᵀ s - sᵀ R)
    float Y[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Y[i*3+j] = R[i]*s[j] + R[3+i]*s[3+j] + R[6+i]*s[6+j];
    cq_ax(Y, a);
}
// 보정 응력: dq (∂_X q, 9) 와 G_S (3x3). 반환 0 이면 보정 없음.
__device__ int cq_corr(const float* A, const float* X, const float* Q, const float* cf, float dt, float sg, float smin, const float* CSt,
                       float mu, float lam, int hmode, float* dq, float* GS) {
    float tr = CSt[cq_si(0,0)] + CSt[cq_si(1,1)] + CSt[cq_si(2,2)] + CSt[cq_si(3,3)] + CSt[cq_si(4,4)] + CSt[cq_si(5,5)] + CSt[cq_si(6,6)] + CSt[cq_si(7,7)] + CSt[cq_si(8,8)];
    if (!(tr > 0.f) || !(smin > 0.01f)) return 0;
    float W1[81], W2[81];                       // W1 = Σ_F -> S_X,  W2 = T
    for (int i=0;i<9;i++) for (int j=0;j<9;j++) W1[i*9+j] = CSt[cq_si(i,j)];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) for (int j=0;j<9;j++)
        W2[(r*3+c)*9+j] = A[r*3]*W1[c*9+j] + A[r*3+1]*W1[(3+c)*9+j] + A[r*3+2]*W1[(6+c)*9+j];
    for (int i=0;i<9;i++) for (int r=0;r<3;r++) for (int c=0;c<3;c++)                   // S_X = T Lᵀ (W1 덮어씀 -- Σ_F 는 더 안 쓴다)
        W1[i*9+r*3+c] = W2[i*9+c]*A[r*3] + W2[i*9+3+c]*A[r*3+1] + W2[i*9+6+c]*A[r*3+2];
    // 극분해 Ŝ = sym(Qᵀ X), M = tr Ŝ I - Ŝ, M⁻¹
    float Sh[9];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Sh[i*3+j] = Q[i]*X[j] + Q[3+i]*X[3+j] + Q[6+i]*X[6+j];
    for (int i=0;i<3;i++) for (int j=i+1;j<3;j++) { float m_ = 0.5f*(Sh[i*3+j]+Sh[j*3+i]); Sh[i*3+j] = m_; Sh[j*3+i] = m_; }
    float tS = Sh[0]+Sh[4]+Sh[8];
    float Mm[9], Mi[9];
    for (int k=0;k<9;k++) Mm[k] = -Sh[k];
    Mm[0] += tS; Mm[4] += tS; Mm[8] += tS;
    cof3(Mm, Mi); { float dm = Mm[0]*Mi[0]+Mm[1]*Mi[1]+Mm[2]*Mi[2]; float idm = 1.f/dm; for (int k=0;k<9;k++) Mi[k] *= idm; }   // M 대칭 -> cof 대칭 = adj
    float Bm[27];                               // B (3x9): B[:,m] = ax(Rᵀ e_m - e_mᵀ R)
    for (int m=0;m<9;m++) { float e[9] = {0,0,0,0,0,0,0,0,0}; e[m] = 1.f; float a3[3]; cq_Bs(Q, e, a3); Bm[m] = a3[0]; Bm[9+m] = a3[1]; Bm[18+m] = a3[2]; }
    float dd = dt - sg;
    // ---- G_S = Σ_c (H T)[(a,c),(b,c)] ----
    for (int k=0;k<9;k++) GS[k] = 0.f;
    if (hmode == 0) {
        float H[81]; const float I3[9] = {1,0,0, 0,1,0, 0,0,1};
        kmanual(X, I3, mu, lam, 1.f, smin, H);
        for (int i=0;i<9;i++) for (int j=i+1;j<9;j++) { float m_ = 0.5f*(H[i*9+j]+H[j*9+i]); H[i*9+j] = m_; H[j*9+i] = m_; }
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float s = 0.f;
            for (int c=0;c<3;c++) for (int m=0;m<9;m++) s += H[(a*3+c)*9+m]*W2[m*9+b*3+c];
            GS[a*3+b] = s; }
    } else {
        for (int b=0;b<3;b++) for (int c=0;c<3;c++) {          // T 의 열 (b,c) 에 H 를 곱한다
            float s9[9]; for (int m=0;m<9;m++) s9[m] = W2[m*9+b*3+c];
            float a3[3], w[3]; cq_Bs(Q, s9, a3);
            for (int i=0;i<3;i++) w[i] = Mi[i*3]*a3[0] + Mi[i*3+1]*a3[1] + Mi[i*3+2]*a3[2];
            float Wx[9] = {0.f,-w[2],w[1], w[2],0.f,-w[0], -w[1],w[0],0.f};
            float dR[9]; mm3(Q, Wx, dR);
            float cs = 0.f; for (int k=0;k<9;k++) cs += cf[k]*s9[k];
            float cl[9]; coflin(X, s9, cl);
            for (int a=0;a<3;a++) {
                float hs_ = 0.f;
                for (int cc=0;cc<3;cc++) if (cc == c) { int k = a*3+cc; hs_ = 2.f*mu*(s9[k]-dR[k]) + lam*(cs*cf[k] + dd*cl[k]); }
                GS[a*3+b] += hs_;
            }
        }
    }
    // ---- λ 몫 ∂_X q_λ ----
    float Sc[9], CSx[9];
    for (int i=0;i<9;i++) { float s = 0.f; for (int j=0;j<9;j++) s += W1[i*9+j]*cf[j]; Sc[i] = s; }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        int i1 = (i+1)%3, i2 = (i+2)%3, j1 = (j+1)%3, j2 = (j+2)%3;
        CSx[i*3+j] = W1[(i1*3+j1)*9 + i2*3+j2] - W1[(i1*3+j2)*9 + i2*3+j1];
    }
    float XC = 0.f; for (int k=0;k<9;k++) XC += X[k]*CSx[k];
    float cl[9]; coflin(X, Sc, cl);
    for (int k=0;k<9;k++) dq[k] = lam*(cl[k] + XC*cf[k] + dd*CSx[k]);
    // ---- μ 몫 -μ ∂h/∂X ----
    float BS[27];
    for (int l=0;l<3;l++) for (int m=0;m<9;m++) { float s = 0.f; for (int j=0;j<9;j++) s += Bm[l*9+j]*W1[j*9+m]; BS[l*9+m] = s; }
    float K[9];
    for (int l=0;l<3;l++) for (int r=0;r<3;r++) { float s = 0.f; for (int m=0;m<9;m++) s += BS[l*9+m]*Bm[r*9+m]; K[l*3+r] = s; }
    float T1[9], MKM[9]; mm3(Mi, K, T1); mm3(T1, Mi, MKM);
    float Yc[27];                               // Yc[i*9+m] = (S Bᵀ M⁻¹)[m][i]
    for (int i=0;i<3;i++) for (int m=0;m<9;m++) Yc[i*9+m] = BS[m]*Mi[i] + BS[9+m]*Mi[3+i] + BS[18+m]*Mi[6+i];
    for (int m=0;m<9;m++) {
        int k = m/3, j = m%3;
        float a3[3] = {Bm[m], Bm[9+m], Bm[18+m]}, w[3];
        for (int i=0;i<3;i++) w[i] = Mi[i*3]*a3[0] + Mi[i*3+1]*a3[1] + Mi[i*3+2]*a3[2];
        float Wx[9] = {0.f,-w[2],w[1], w[2],0.f,-w[0], -w[1],w[0],0.f};
        float dR[9]; mm3(Q, Wx, dR);
        float WS[9]; mm3(Wx, Sh, WS);
        float dS[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) dS[a*3+b] = ((b==j) ? Q[k*3+a] : 0.f) - WS[a*3+b];
        for (int a=0;a<3;a++) for (int b=a+1;b<3;b++) { float m_ = 0.5f*(dS[a*3+b]+dS[b*3+a]); dS[a*3+b] = m_; dS[b*3+a] = m_; }
        float tdS = dS[0]+dS[4]+dS[8];
        float t1 = 0.f;
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float dM = ((a==b) ? tdS : 0.f) - dS[a*3+b]; t1 += dM*MKM[a*3+b]; }
        float t2 = 0.f;
        for (int i=0;i<3;i++) {
            float Z[9];                         // Z = dRᵀ y_i
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) Z[a*3+b] = dR[a]*Yc[i*9+b] + dR[3+a]*Yc[i*9+3+b] + dR[6+a]*Yc[i*9+6+b];
            float az[3]; cq_ax(Z, az);
            t2 += az[i];
        }
        dq[m] += -mu*(-t1 + 2.f*t2);
    }
    return 1;
}
__global__ void ex_tet_force_CQ(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, const float* CS,
                                float mu, float lam, float* f, float* rst, int hmode, float* gdbg) {
    // gdbg (검산 전용, 보통 null): 사면체마다 [P F̄ᵀ (9), ∂_X q F̄ᵀ (9), G_S (9)]
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (rst) { float dA = det3(A); if (!(dA > 0.f)) atomicAdd(&rst[0], 1.f); }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    float dq[9], GS[9];
    int on = cq_corr(A, X, Q, cf, dt, sg, sqrtf(fmaxf(l[0], 0.f)), CS + (long long)t*45, mu, lam, hmode, dq, GS);
    if (gdbg) { float g0[9]; mmt3(P, Fb, g0); for (int k=0;k<9;k++) { gdbg[(long long)t*27+k] = g0[k]; gdbg[(long long)t*27+9+k] = 0.f; gdbg[(long long)t*27+18+k] = 0.f; }
        if (on) { float g1[9]; mmt3(dq, Fb, g1); for (int k=0;k<9;k++) { gdbg[(long long)t*27+9+k] = g1[k]; gdbg[(long long)t*27+18+k] = GS[k]; } } }
    if (on) for (int k=0;k<9;k++) P[k] += dq[k];
    mmt3(P, Fb, Gm);
    if (on) { for (int k=0;k<9;k++) Gm[k] += GS[k]; if (rst) atomicAdd(&rst[11], 1.f); }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}

// ======== 절 AB (갈래 A, 사용자 10-30 "해볼 수 있는 거 다"): 프레임 끝 G2P 의 입자 F 갱신 방식 ========
// 진단 (절 Y·AA): 입자 F 이웃 벌어짐이 정답의 10~20 배 -- 이웃 사면체의 서로 다른 A_T 를 F_p <- A_T F_p 로 각자 누적. 위치·속도 G2P 는 그대로, F 갱신만 바꾼다.
// fa_tet_acc: 점유 사면체 (V_T > 0) 의 X_T 를 네 노드에 V_T 가중 합 -- NA[n*10+k] += V_T X_T[k], NA[n*10+9] += V_T
//   which 0: X_T = A_T = I + Σ_i U_i ∇λ_iᵀ (프레임 셀 사상), 1: X_T = F̄_T (프레임 시작 셀 평균 F, P2G 것)
__global__ void fa_tet_acc(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, int which, float* NA) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; for (int i=0;i<4;i++) n[i] = tets[t*4+i];
    float X[9];
    if (which == 0) {
        for (int k=0;k<9;k++) X[k] = (k%4==0) ? 1.f : 0.f;
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u = U[n[i]*3+c]; for (int b=0;b<3;b++) X[c*3+b] += u*gl[t*12+i*3+b]; }
    } else {
        for (int k=0;k<9;k++) X[k] = FT[t*9+k];
    }
    for (int i=0;i<4;i++) { for (int k=0;k<9;k++) atomicAdd(&NA[n[i]*10+k], V*X[k]); atomicAdd(&NA[n[i]*10+9], V); }
}
// ex_g2p_lagF: 위치·속도는 ex_g2p_lag (vq 0, C 없음) 와 같은 식, F 갱신만 mode 로:
//   0 기존: F <- A_T F
//   1 bar  (셀 균질화/부분 이완): F <- A_T [(1-β) F + β F̄_T]          (β 1 = 셀 평균으로 덮어씀)
//   2 nbar (이웃 이완): F <- A_T [(1-β) F + β F̃],  F̃ = Σ_i λ_i F̄_n (F̄_n = 노드에 닿는 점유 사면체 F̄_T 의 V_T 가중 평균, NA which 1)
//   3 navg (매끈한 속도 기울기): F <- [(1-β) A_T + β Ã] F,  Ã = Σ_i λ_i Ā_n (Ā_n = 노드에 닿는 점유 사면체 A_T 의 V_T 가중 평균, NA which 0 -- simplex.nodeavg_jac_pre 꼴)
//   4 polar (회전/신장 분리): F = R S, F̃ = R̃ S̃ (극분해) -> F <- A_T R [(1-β) S + β S̃]  (회전은 입자 것, 신장만 이웃 (노드 평균 F̄) 쪽으로)
//   5 navgbar (3 + 2): F <- [(1-β) A_T + β Ã] [(1-β2) F + β2 F̃]  (NA = A 노드 평균, NB = F̄ 노드 평균)
__global__ void ex_g2p_lagF(int N, const float* x, const float* Ucum, const float* Vn, const long long* rows, const float* lamb, const float* glT,
                            const long long* tinv, const float* Fp, float* x2, float* v2, float* F2, const float* FT, const float* NA, int mode, float beta, const float* NB, float beta2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1}, Nt[9]={0,0,0,0,0,0,0,0,0}, Nb[9]={0,0,0,0,0,0,0,0,0};
    long long t = tinv[p];
    const float* gl = glT + t*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; double Va=(double)Vn[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*Va;
            for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } 
        if (mode >= 2) { double W = (double)NA[nd*10+9]; if (W > 0.0) for (int k=0;k<9;k++) Nt[k] += lm*(double)NA[nd*10+k]/W; }
        if (mode == 5) { double W = (double)NB[nd*10+9]; if (W > 0.0) for (int k=0;k<9;k++) Nb[k] += lm*(double)NB[nd*10+k]/W; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double B = beta, G[9], F2d[9];
    if (mode == 1) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*F[k] + B*(double)FT[t*9+k]; mm3(A, G, F2d); }
    else if (mode == 2) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*F[k] + B*Nt[k]; mm3(A, G, F2d); }
    else if (mode == 3) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*A[k] + B*Nt[k]; mm3(G, F, F2d); }
    else if (mode == 4) {
        double R[9], Rt[9], S[9], St[9], RS[9];
        polar(F, R); polar(Nt, Rt);
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { double s1=0, s2=0; for (int k=0;k<3;k++) { s1 += R[k*3+r]*F[k*3+c]; s2 += Rt[k*3+r]*Nt[k*3+c]; } S[r*3+c]=s1; St[r*3+c]=s2; }
        for (int k=0;k<9;k++) G[k] = ((double)1-B)*S[k] + B*St[k];
        mm3(R, G, RS); mm3(A, RS, F2d);
    }
    else if (mode == 5) { double H[9]; for (int k=0;k<9;k++) { G[k] = ((double)1-B)*A[k] + B*Nt[k]; H[k] = ((double)1-(double)beta2)*F[k] + (double)beta2*Nb[k]; } mm3(G, H, F2d); }
    else mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}
}
extern "C" {
// ======== 절 AD: 희소 셀 병합 (사용자 10-30 "해볼 수 있는 거 다", 갈래 B) ========
// 묶음 g (사면체 집합, 루트 r) 의 셀 에너지 E_g = V_g ψ(Ā_g F̄_g),  Ā_g = Σ_{t∈g} V_t A_t / V_g (입자 부피 가중),
//   F̄_g = Σ V_t F̄_t / V_g,  V_g = Σ V_t (입자 부피).  힘 f_{t,i} = -V_t P(Ā F̄) F̄ᵀ ∇λ_{t,i}  (= -∂E_g/∂x, 묶인 사면체마다 입자 부피 V_t 비례 분배 -- 노드 질량 m_T/4 와 같은 비라 강성/질량 비 유지).
//   (처음 판은 기하 부피 = 같은 몫 (V_g/n) 이었는데 입자 1 개 사면체의 가벼운 노드에 큰 강성이 걸려 프레임 2 에 발산 -- 연기 시험)
//   Σ_i ∇λ_{t,i} = 0 -> 운동량 보존, ψ 회전 불변 + Ā 가 강체 회전과 같이 돈다 -> 각운동량 보존 (에너지의 기울기이므로).
// grp[t] = 루트 (묶음 크기 ≥ 2) 또는 -1 (홀로: ex_tet_force_R 과 같은 식). AG = 서브스텝 홀짝 두 벌 (G1 이 이번 벌에 쌓고, G2 가 읽은 뒤 다음 벌을 비운다).
__device__ __forceinline__ void sp_fcr_P(const float* X, float mu, float lam, float* P) {
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
}
__global__ void sp_force_g1(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                            float mu, float lam, float* f, const int* grp, float* AG, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (rst && !(det3(A) > 0.f)) atomicAdd(&rst[0], 1.f);
    int r = grp[t];
    if (r >= 0) { for (int k=0;k<9;k++) atomicAdd(&AG[(long long)r*9+k], V*A[k]); return; }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float P[9], Gm[9]; sp_fcr_P(X, mu, lam, P);
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
__global__ void sp_force_g2(const int* szp, const long long* tets, const float* gl, float mu, float lam, float* f, const int* grp, const float* AG, float* AGz,
                            const float* FG, const float* VG, const float* VT, const float* U, float hg) {
    // hg > 0: 모래시계 억제 γ μ Σ_t V_t |A_t - Ā|²  ->  사면체 t 힘에 -V_t 2 γ μ (A_t - Ā) ∇λ_{t,i} 더함 (Σ_t V_t (A_t - Ā) = 0 이라 Ā 경유 몫 없음)
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    int r = grp[t];
    if (r < 0) return;
    float vg = VG[r];
    float A[9]; for (int k=0;k<9;k++) A[k] = AG[(long long)r*9+k]/vg;
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FG[(long long)r*9+k];
    float X[9]; mm3(A, Fb, X);
    float P[9], Gm[9]; sp_fcr_P(X, mu, lam, P);
    mmt3(P, Fb, Gm);
    float w = VT[t];
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    if (hg > 0.f) {
        float At[9] = {1,0,0, 0,1,0, 0,0,1};
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) At[c*3+b] += u*G[i*3+b]; }
        for (int k=0;k<9;k++) Gm[k] += 2.f*hg*mu*(At[k]-A[k]);
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -w*s);
    }
    if (AGz) for (int k=0;k<9;k++) AGz[(long long)r*9+k] = 0.f;              // 다음 서브스텝 벌 비우기 (같은 묶음 모두 0 을 쓴다)
}
}  // extern "C" (절 AD)
'''

_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + _EX).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def set_diag(vals):
    """진단 배율 EXD (fem_ex_cuda 머리말) 를 장치에 쓴다."""
    import cupy as cp
    ptr = mod().get_global("EXD")
    cp.ndarray((3,), cp.float32, ptr)[:] = cp.asarray(np.asarray(vals, dtype=np.float32))


def P(t):
    """장치 주소 (없으면 0 = null)."""
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def launch(name, n, args, b=128):
    with RTC._Stream():
        mod().get_function(name)(((int(n) + b - 1) // b,), (b,), args)


def kuhn_min_altitude(gl):
    """사면체 ∇λ [T,4,3] -> 최소 꼭짓점-대면 높이 1/|∇λ_i| (CFL 길이)."""
    return float((1.0 / gl.double().norm(dim=-1)).min())
