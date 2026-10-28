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
        gy[c3] = k_*t + pj;
    }
    R MJ[9]; mm3(M.Mn, M.Jst, MJ);
    R gG[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++){ R t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gG[a*3+b]=t; }
    if (!Ucum) { R T2[9]; mmt3(gG, IGt, T2); for (int k=0;k<9;k++) gG[k]=T2[k]; }     // sub: ∂/∂δU 는 gG (I+G)ᵀ
    for (int k=0;k<9;k++) gy[3+k]=gG[k];
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
                           const long long* tinv, const float* Fp, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*(double)Vn[nd*3+a]; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
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
__global__ void ex_p2g_mv(int N, const long long* rows, const float* lamb, const float* m, const float* v, float* mn, float* mv, int tetmass) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float mp = m[p];
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float w=(tetmass ? 0.25f : lamb[p*4+i])*mp; atomicAdd(&mn[nd], w); for (int c=0;c<3;c++) atomicAdd(&mv[nd*3+c], w*v[p*3+c]); }
}
__global__ void ex_vfin(const int* szp, int Marg, const float* mn, const float* mv, float* vn) {
    int M = szp ? szp[0] : Marg;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float m = mn[n];
    for (int c=0;c<3;c++) vn[n*3+c] = m > 1e-30f ? mv[n*3+c]/m : 0.f;
}
// 분할 (CudaTetqK 와 같은 정의: 예측 변위 u0 = Σλ (H v_n + H² g) 의 끝 높이 s/ε ≤ 20 이면 바닥 후보) + 항등 입자만 사면체 V·V F 합
__global__ void ex_split(int N, const float* x, const long long* rows, const float* lamb, const long long* pinv, const float* vn, float H,
                         float g0, float g1, float g2, int floor, float p0, float p1, float p2, float q0, float q1, float q2, float eps_sp, float eps_st,
                         const float* vol, const float* Fp, float* VTs, float* FTs, int* nslot, long long* nidx, int* szp, int keep) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bool near = false;
    if (keep) {
        if (nslot[p] < 0) { long long t = pinv[p]; float V = vol[p]; atomicAdd(&VTs[t], V); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]); }
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
}
'''

_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + _EX).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def P(t):
    """장치 주소 (없으면 0 = null)."""
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def launch(name, n, args, b=128):
    with RTC._Stream():
        mod().get_function(name)(((int(n) + b - 1) // b,), (b,), args)


def kuhn_min_altitude(gl):
    """사면체 ∇λ [T,4,3] -> 최소 꼭짓점-대면 높이 1/|∇λ_i| (CFL 길이)."""
    return float((1.0 / gl.double().norm(dim=-1)).min())
