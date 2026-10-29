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
            float k = (mode == 1) ? 0.f : fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f);
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
}
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
