"""절 AG (사용자 10-30): 확정 실시간 명시적 FEM (절 AD) 에 소성·점성·파괴 -- 정답 솔버 (PhysGaussian / GaussianFluent MPM) 의 재질 식을 그대로 옮긴 커널.

정답 식 출처 (그대로 옮김, 클러스터 사본 /home/dkta/work/{PhysGaussian,GaussianFluent}/mpm_solver_warp/mpm_utils.py):
  재질 번호 (PG MPM_Simulator_WARP 와 같다):
   1 metal      : PG von_mises_return_mapping (항복 ys, hardening 이면 ys += 2μ ξ δγ) + 응력 PG kirchoff_stress_StVK
   2 sand       : PG sand_return_mapping (Drucker-Prager, α = √(2/3)·2 sinφ/(3 - sinφ)) + 응력 PG kirchoff_stress_drucker_prager
   3 foam       : PG viscoplasticity_return_mapping_with_StVK (점소성, plastic_viscosity η, 걸음 dt) + 응력 PG kirchoff_stress_StVK
   5 plasticine : PG von_mises_return_mapping_with_damage (ys -= softening·|Δε|, ys ≤ 0 이면 μ = λ = 0 -- 영구 손상) + 응력 PG kirchoff_stress_FCR
   7 watermelon : GF NonAssociativeCamClay_return_mapping (NACC, logJp 경화 = 수박을 깨뜨리는 장치) + 응력 GF kirchoff_stress_neoHookeanBoarden
  모두 주값 (특이값) 공간에서 같은 식. 응력은 키르히호프 τ = P Fᵀ 로 정답과 대조 (exe/fem_ag_unit.py).
  주의: PG kirchoff_stress_StVK 는 U diag(τ_i) Vᵀ Fᵀ 를 돌려준다 (τ_i = 2μ ln σ_i + λ Σ ln σ) -- 곧 P = U diag(τ_i) Vᵀ (1/σ_i 가 없다). 그대로 옮긴다.

3x3 SVD: C = XᵀX 순환 Jacobi (float) -> V (det +1 로 맞춤), σ 내림차순, U = X V / σ (u2 = u0 x u1, σ3 부호 = det X 부호) -- 정답 wp.svd3 과 같은 꼴 (U·V 회전, 뒤집힘은 마지막 σ 음수).

커널:
  ag_fc   : 절 C 융합 판 c_fc (사면체 힘 + 바닥 접촉) 와 같은 틀. 사면체 블록만 재질 식으로 -- X = A_T F̄_T 를 셀 소성 상태 (입자 부피 가중 평균, 프레임마다)
            로 정답 return mapping 에 넣어 (상태는 바꾸지 않음) 응력을 항복면 안으로 투영한 뒤 그 응력으로 힘. mat 0 이면 c_force_body 그대로 (탄성 판과 같은 식).
            visc > 0 이면 셀 점성 응력 τ_v = 2 η J sym(L), L = (Σ v_n ⊗ ∇λ_n) A⁻¹ (정답 솔버에는 이 항이 없다 -- 구현만, --visc).
  ag_rm   : 프레임 끝 G2P 뒤 입자별 return mapping (정답 식, 상태 갱신: ys·logJp·손상).
  ag_pbc  : 입자 속도 구속 (PG enforce_particle_translation / release_particles_sequentially) 의 노드 몫 섞기 v <- (1 - α) v + α v̄_bc (서브스텝마다, 노드 걸음 뒤).
  ag_test : 단위 대조 -- 같은 F·상태에 return mapping (갱신) + 키르히호프 응력.
prm (장치 float 16): [0] μ [1] λ [2] κ [3] α [4] M [5] β [6] ξ [7] hardening [8] softening [9] η (plastic_viscosity)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import fem_rt_cuda as RTC
from . import fem_ex_cuda as EX
from . import fem_c_cuda as CCU

MAT = {"jelly": 0, "metal": 1, "sand": 2, "foam": 3, "plasticine": 5, "watermelon": 7}

_PL = r'''
extern "C" {
// ---------------- 3x3 SVD (C = XᵀX 순환 Jacobi) ----------------
__device__ __forceinline__ void pl_svd(const float* X, float* U, float* s, float* V) {
    float A[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) A[i*3+k] = X[i]*X[k] + X[3+i]*X[3+k] + X[6+i]*X[6+k];
    for (int k=0;k<9;k++) V[k] = (k%4==0) ? 1.f : 0.f;
    for (int sw=0; sw<10; sw++) {
        float off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
        float dg = A[0]*A[0] + A[4]*A[4] + A[8]*A[8];
        if (off <= 1e-14f*dg || off < 1e-36f) break;
        for (int r=0;r<3;r++) {
            int p = (r==2) ? 1 : 0, q = (r==0) ? 1 : 2;
            float apq = A[p*3+q];
            if (fabsf(apq) < 1e-30f) continue;
            float th = (A[q*3+q] - A[p*3+p]) / (2.f*apq);
            float t = (th >= 0.f ? 1.f : -1.f) / (fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), sn = t*c;
            for (int k=0;k<3;k++) { float akp=A[k*3+p], akq=A[k*3+q]; A[k*3+p]=c*akp - sn*akq; A[k*3+q]=sn*akp + c*akq; }
            for (int k=0;k<3;k++) { float apk=A[p*3+k], aqk=A[q*3+k]; A[p*3+k]=c*apk - sn*aqk; A[q*3+k]=sn*apk + c*aqk; }
            for (int k=0;k<3;k++) { float vkp=V[k*3+p], vkq=V[k*3+q]; V[k*3+p]=c*vkp - sn*vkq; V[k*3+q]=sn*vkp + c*vkq; }
        }
    }
    // 내림차순 정렬 (열 교환)
    float l[3] = {A[0], A[4], A[8]};
    for (int i=0;i<2;i++) for (int j=0;j<2-i;j++) if (l[j] < l[j+1]) {
        float tl=l[j]; l[j]=l[j+1]; l[j+1]=tl;
        for (int k=0;k<3;k++) { float tv=V[k*3+j]; V[k*3+j]=V[k*3+j+1]; V[k*3+j+1]=tv; }
    }
    // V 를 회전으로 (det +1)
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) { V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8]; }
    float xv[3][3];
    for (int j=0;j<3;j++) for (int r=0;r<3;r++) xv[j][r] = X[r*3]*V[j] + X[r*3+1]*V[3+j] + X[r*3+2]*V[6+j];
    float s0 = sqrtf(fmaxf(l[0], 0.f));
    float u0[3], u1[3], u2[3];
    float n0 = sqrtf(xv[0][0]*xv[0][0] + xv[0][1]*xv[0][1] + xv[0][2]*xv[0][2]);
    if (n0 > 1e-20f) { for (int r=0;r<3;r++) u0[r] = xv[0][r]/n0; } else { u0[0]=1.f; u0[1]=0.f; u0[2]=0.f; }
    float d01 = u0[0]*xv[1][0] + u0[1]*xv[1][1] + u0[2]*xv[1][2];
    for (int r=0;r<3;r++) u1[r] = xv[1][r] - d01*u0[r];
    float n1 = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]);
    if (n1 > 1e-20f) { for (int r=0;r<3;r++) u1[r] /= n1; }
    else {   // u0 에 수직인 아무 단위 벡터
        float e[3] = {0.f, 0.f, 0.f}; int k = (fabsf(u0[0]) < 0.6f) ? 0 : ((fabsf(u0[1]) < 0.6f) ? 1 : 2); e[k] = 1.f;
        float d = u0[k]; for (int r=0;r<3;r++) u1[r] = e[r] - d*u0[r];
        float nn = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]); for (int r=0;r<3;r++) u1[r] /= nn;
    }
    u2[0] = u0[1]*u1[2] - u0[2]*u1[1]; u2[1] = u0[2]*u1[0] - u0[0]*u1[2]; u2[2] = u0[0]*u1[1] - u0[1]*u1[0];
    s[0] = n0 > 1e-20f ? n0 : s0;
    s[1] = u1[0]*xv[1][0] + u1[1]*xv[1][1] + u1[2]*xv[1][2];
    s[2] = u2[0]*xv[2][0] + u2[1]*xv[2][1] + u2[2]*xv[2][2];
    for (int r=0;r<3;r++) { U[r*3] = u0[r]; U[r*3+1] = u1[r]; U[r*3+2] = u2[r]; }
}
__device__ __forceinline__ void pl_usv(const float* U, const float* d, const float* V, float* Y) {   // Y = U diag(d) Vᵀ
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) Y[r*3+c] = U[r*3]*d[0]*V[c*3] + U[r*3+1]*d[1]*V[c*3+1] + U[r*3+2]*d[2]*V[c*3+2];
}
// ---------------- return mapping (주값; s 를 덮어쓴다; 바뀌었으면 true) ----------------
//  upd = 0 이면 ys·lj·al 은 지역 사본 (셀 투영), 1 이면 입자 상태 갱신
__device__ __forceinline__ bool pl_rm(int mat, float* s, const float* prm, float& ys, float& lj, float& al, float dt) {
    float mu = prm[0], lam = prm[1];
    if (mat == 1 || mat == 5) {                                     // PG von_mises_return_mapping(_with_damage)
        if (mat == 5 && !(al > 0.f)) return false;                  // 손상 입자: μ = λ = 0 -> cond 0 > ys (≤ 0) -> F_trial 그대로
        float sg[3], ep[3];
        for (int i=0;i<3;i++) { sg[i] = fmaxf(s[i], 0.01f); ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2], tmp = tr/3.f;
        float tau[3]; for (int i=0;i<3;i++) tau[i] = 2.f*mu*ep[i] + lam*tr;
        float st = tau[0] + tau[1] + tau[2];
        float c0 = tau[0] - st/3.f, c1 = tau[1] - st/3.f, c2 = tau[2] - st/3.f;
        if (sqrtf(c0*c0 + c1*c1 + c2*c2) > ys) {
            if (mat == 5 && ys <= 0.f) return false;
            float eh[3]; for (int i=0;i<3;i++) eh[i] = ep[i] - tmp;
            float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]) + 1e-6f;
            float dg = ehn - ys/(2.f*mu);
            for (int i=0;i<3;i++) ep[i] = ep[i] - (dg/ehn)*eh[i];
            float muh = mu;
            if (mat == 5) {
                float r = dg/ehn; float de = fabsf(r)*sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
                ys = ys - prm[8]*de;
                if (ys <= 0.f) { al = 0.f; muh = 0.f; }
            }
            for (int i=0;i<3;i++) s[i] = expf(ep[i]);
            if (prm[7] == 1.f) ys = ys + 2.f*muh*prm[6]*dg;
            return true;
        }
        return false;
    }
    if (mat == 3) {                                                 // PG viscoplasticity_return_mapping_with_StVK
        float sg[3], ep[3], b[3];
        for (int i=0;i<3;i++) { sg[i] = fmaxf(s[i], 0.01f); b[i] = sg[i]*sg[i]; ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2];
        float st[3]; for (int i=0;i<3;i++) st[i] = 2.f*mu*(ep[i] - tr/3.f);
        float stn = sqrtf(st[0]*st[0] + st[1]*st[1] + st[2]*st[2]);
        float y = stn - sqrtf(2.f/3.f)*ys;
        if (y > 0.f) {
            float muh = mu*(b[0] + b[1] + b[2])/3.f;
            float snn = stn - y/(1.f + prm[9]/(2.f*muh*dt));
            for (int i=0;i<3;i++) s[i] = expf((snn/stn)*st[i]/(2.f*mu) + tr/3.f);
            return true;
        }
        return false;
    }
    if (mat == 2) {                                                 // PG sand_return_mapping
        float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fmaxf(fabsf(s[i]), 1e-14f));
        float tr = ep[0] + ep[1] + ep[2];
        float eh[3]; for (int i=0;i<3;i++) eh[i] = ep[i] - tr/3.f;
        float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
        float dg = ehn + (3.f*lam + 2.f*mu)/(2.f*mu)*tr*prm[3];
        if (dg <= 0.f) return false;
        if (tr > 0.f) { s[0] = s[1] = s[2] = 1.f; return true; }
        for (int i=0;i<3;i++) s[i] = expf(ep[i] - eh[i]*(dg/ehn));
        return true;
    }
    if (mat == 7) {                                                 // GF NonAssociativeCamClay_return_mapping
        float kap = prm[2], M = prm[4], beta = prm[5], xi = prm[6];
        float sg[3]; for (int i=0;i<3;i++) sg[i] = fmaxf(s[i], 0.f);
        float p0 = kap*(0.00001f + sinhf(xi*fmaxf(-lj, 0.f)));
        float J = sg[0]*sg[1]*sg[2];
        float B[3]; for (int i=0;i<3;i++) B[i] = sg[i]*sg[i];
        float Bm = (B[0] + B[1] + B[2])/3.f;
        float Jp = powf(J, -2.f/3.f);
        float sh[3]; for (int i=0;i<3;i++) sh[i] = mu*Jp*(B[i] - Bm);
        float prime = kap/2.f*(J - 1.f/J);
        float ptr = -prime*J;
        float ysh = (6.f - 3.f)/2.f*(1.f + 2.f*beta);
        float yph = M*M*(ptr + beta*p0)*(ptr - p0);
        float ssq = sh[0]*sh[0] + sh[1]*sh[1] + sh[2]*sh[2];
        float y = ysh*ssq + yph;
        float sf[3] = {sg[0], sg[1], sg[2]};
        float ljn = lj, pmin = beta*p0;
        int hard = prm[7] > 0.5f;
        if (ptr > p0) {
            float Je = sqrtf(-2.f*p0/kap + 1.f);
            sf[0] = sf[1] = sf[2] = powf(Je, 1.f/3.f);
            if (hard) ljn = lj + logf(J/Je);
        } else if (ptr < -pmin) {
            float Je = sqrtf(2.f*pmin/kap + 1.f);
            sf[0] = sf[1] = sf[2] = powf(Je, 1.f/3.f);
            if (hard) ljn = lj + logf(J/Je);
        } else if (y >= 1e-4f) {
            float shn = fmaxf(sqrtf(ssq), 1e-10f);
            float sqf = sqrtf(-yph/ysh);
            float J2 = powf(J, 2.f/3.f);
            float sc = J2/mu*sqf/shn;
            for (int i=0;i<3;i++) sf[i] = sqrtf(sc*sh[i] + Bm);
            if (hard && p0 > 1e-4f && ptr < p0 - 1e-4f && ptr > 1e-4f - pmin) {
                float pc = (p0 - pmin)*0.5f;
                float qt = sqrtf((6.f - 3.f)/2.f)*shn;
                float dp = pc - ptr, dq = 0.f - qt;
                float dn = fmaxf(sqrtf(dp*dp + dq*dq), 1e-10f);
                dp = dp/dn;                                         // GF 원본: direction_q 는 나누지 않는다 (그대로)
                float C = M*M*(pc + beta*p0)*(pc - p0);
                float Bq = M*M*dp*(2.f*pc - p0 + beta*p0);
                float Aq = M*M*dp*dp + (1.f + 2.f*beta)*dq*dq;
                float disc = Bq*Bq - 4.f*Aq*C;
                float l1 = (-Bq + sqrtf(disc))/(2.f*Aq);
                float l2 = (-Bq - sqrtf(disc))/(2.f*Aq);
                float p1 = pc + l1*dp, p2 = pc + l2*dp;
                float pf = ((ptr - pc)*(p1 - pc) > 0.f) ? p1 : p2;
                float Jef = sqrtf(fabsf(-2.f*pf/kap + 1.f));
                if (Jef > 1e-4f && hard) ljn = lj + logf(J/Jef);
            }
        }
        lj = ljn;
        s[0] = sf[0]; s[1] = sf[1]; s[2] = sf[2];
        return true;                                                // GF 는 늘 U diag(σ_final) Vᵀ 로 다시 짠다
    }
    return false;
}
// ---------------- 응력 주값 P_i (P = U diag(P_i) Vᵀ, 키르히호프 τ = P Fᵀ) ----------------
__device__ __forceinline__ void pl_pi(int mat, const float* s, const float* prm, float al, float* Pi) {
    float mu = prm[0], lam = prm[1];
    if (mat == 1 || mat == 3) {                                     // PG kirchoff_stress_StVK: τ = U diag(τ_i) Vᵀ Fᵀ -> P_i = τ_i
        float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fmaxf(s[i], 0.01f));
        float tr = ep[0] + ep[1] + ep[2];
        for (int i=0;i<3;i++) Pi[i] = 2.f*mu*ep[i] + lam*tr;
    } else if (mat == 2) {                                          // PG kirchoff_stress_drucker_prager: P_i = (2μ ln σ_i + λ Σ ln σ)/σ_i
        float sg[3], ep[3]; for (int i=0;i<3;i++) { sg[i] = fmaxf(fabsf(s[i]), 1e-14f); ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2];
        for (int i=0;i<3;i++) Pi[i] = (2.f*mu*ep[i] + lam*tr)/sg[i];
    } else if (mat == 7) {                                          // GF kirchoff_stress_neoHookeanBoarden: τ = μ J^{-2/3} dev(B) + κ/2 (J² - 1) I
        float kap = prm[2];
        float J = s[0]*s[1]*s[2];
        float Bm = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])/3.f;
        float Jp = powf(J, -2.f/3.f);
        for (int i=0;i<3;i++) { float sd = (fabsf(s[i]) > 1e-6f) ? s[i] : ((s[i] < 0.f) ? -1e-6f : 1e-6f);
            Pi[i] = (mu*Jp*(s[i]*s[i] - Bm) + 0.5f*kap*(J*J - 1.f))/sd; }
    } else {                                                        // PG kirchoff_stress_FCR (jelly, plasticine): P = 2μ(F - R) + λ J (J - 1) F⁻ᵀ
        float J = s[0]*s[1]*s[2];
        float co[3] = {s[1]*s[2], s[0]*s[2], s[0]*s[1]};
        for (int i=0;i<3;i++) Pi[i] = al*(2.f*mu*(s[i] - 1.f) + lam*(J - 1.f)*co[i]);
    }
}
// GF kirchoff_stress_neoHookeanBoarden 를 F_e 에서 바로 (GF 와 같은 식: B = F Fᵀ, J = det F -- SVD 거치지 않아 float 오차가 정답 수준)
__device__ __forceinline__ void pl_nhb_tau(const float* Fe, float mu, float kap, float* tau) {
    float B[9]; mmt3(Fe, Fe, B);
    float J = det3(Fe);
    float tb = (B[0] + B[4] + B[8])/3.f;
    float sd = mu*powf(J, -2.f/3.f), sv = 0.5f*kap*(J*J - 1.f);
    for (int k=0;k<9;k++) tau[k] = sd*(B[k] - ((k%4==0) ? tb : 0.f)) + ((k%4==0) ? sv : 0.f);
}
// ---------------- 사면체 힘 (c_force_body 틀 + 재질) ----------------
__device__ __forceinline__ void ag_force_body(int t, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                              float mu, float lam, float* f, float* rst, int fpol,
                                              int mat, const float* prm, const float* ysT, const float* ljT, float dtc, const float* vn, float visc, int cflag,
                                              float* FeC, float* XpC, float* ysC, float* ljC) {
    // cflag (진단 전용): 1 = 셀 투영 끔 (재질 응력만, return mapping 은 프레임 끝 입자에만), 2 = 응력을 FCR 로 (투영은 그대로)
    if (mat == 0 && !(visc > 0.f)) { c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, 0.f, 0.f, 0, rst, fpol, (float*)0); return; }
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
    float Gm[9];
    if (mat == 0) {
        float C[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
        float l[3]; symeig3(C, l);
        float J = 1.f;
        for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
        float Q[9], cf[9], P[9];
        if (fpol) c_polar_fast(X, Q); else polar(X, Q);
        cof3(X, cf);
        float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
        float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
        for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
        mmt3(P, Fb, Gm);
    } else {
        float Uu[9], s[3], Vv[9];
        float ys = ysT ? ysT[t] : 0.f, lj = ljT ? ljT[t] : 0.f, al = 1.f;
        if (FeC) {                                                  // 셀 증분 return mapping (정답처럼 서브스텝마다): F_trial = X X_prev⁻¹ F_e,prev, 셀 상태 갱신
            float Xp[9], Fp[9]; for (int k=0;k<9;k++) { Xp[k] = XpC[t*9+k]; Fp[k] = FeC[t*9+k]; }
            float cX[9]; cof3(Xp, cX); float dX = det3(Xp); float id = (fabsf(dX) > 1e-20f) ? 1.f/dX : 0.f;
            float Xi[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Xi[i*3+j] = cX[j*3+i]*id;
            float D[9]; mm3(X, Xi, D);
            float Ft[9]; mm3(D, Fp, Ft);
            ys = ysC[t]; lj = ljC[t];
            pl_svd(Ft, Uu, s, Vv);
            bool ch = pl_rm(mat, s, prm, ys, lj, al, dtc);
            float Fe[9]; if (ch) pl_usv(Uu, s, Vv, Fe); else for (int k=0;k<9;k++) Fe[k] = Ft[k];
            for (int k=0;k<9;k++) { FeC[t*9+k] = Fe[k]; XpC[t*9+k] = X[k]; }
            ysC[t] = ys; ljC[t] = lj;
            // 응력은 F_e 에서, 힘은 Gm = P(F_e) F_e⁻ᵀ ... 대신 τ X⁻ᵀ F̄ᵀ: P_X = τ X⁻ᵀ (현재 배치 응력은 τ/J 로 같다)
            float tau[9];
            if (mat == 7 && !(cflag & 2)) pl_nhb_tau(Fe, prm[0], prm[2], tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(Fe, U2, s2, V2); float Pi[3]; pl_pi((cflag & 2) ? 5 : mat, s2, prm, al, Pi);
                   float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, Fe, tau); }
            float cXX[9]; cof3(X, cXX); float dXX = det3(X); float idd = (fabsf(dXX) > 1e-20f) ? 1.f/dXX : 0.f;
            float XiT[9]; for (int k=0;k<9;k++) XiT[k] = cXX[k]*idd;                       // X⁻ᵀ = cof(X)/det
            float PX[9]; mm3(tau, XiT, PX);
            mmt3(PX, Fb, Gm);
            goto visc_part;
        }
        pl_svd(X, Uu, s, Vv);
        if (!(cflag & 1)) pl_rm(mat, s, prm, ys, lj, al, dtc);    // 셀 투영 (상태는 지역 사본 -- 입자 상태는 프레임 끝 ag_rm 만 바꾼다)
        if (mat == 7 && !(cflag & 2)) {                             // τ 를 F_e 에서 바로: Gm = τ F_e⁻ᵀ F̄ᵀ = τ (F̄ F_e⁻¹)ᵀ
            float Fe[9]; pl_usv(Uu, s, Vv, Fe);
            float tau[9]; pl_nhb_tau(Fe, prm[0], prm[2], tau);
            float cF[9]; cof3(Fe, cF);
            float dF = det3(Fe); float id = (fabsf(dF) > 1e-20f) ? 1.f/dF : 0.f;
            float Fi[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Fi[i*3+j] = cF[j*3+i]*id;     // F_e⁻¹
            float M_[9]; mm3(Fb, Fi, M_);                            // F̄ F_e⁻¹
            mmt3(tau, M_, Gm);
        } else {
            float Pi[3]; pl_pi((cflag & 2) ? 5 : mat, s, prm, al, Pi);
            float P[9]; pl_usv(Uu, Pi, Vv, P);
            mmt3(P, Fb, Gm);
        }
    }
visc_part:
    if (visc > 0.f && vn) {                                         // 점성 (정답 솔버에는 없음): τ_v = 2 η J sym(L), L = (Σ v_n ⊗ ∇λ_n) A⁻¹, 힘 -V τ_v A⁻ᵀ ∇λ
        float Gv[9] = {0,0,0,0,0,0,0,0,0};
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float vv = vn[n[i]*3+c]; for (int b=0;b<3;b++) Gv[c*3+b] += vv*G[i*3+b]; }
        float id = (fabsf(dA) > 1e-12f) ? 1.f/dA : 0.f;
        float Ai[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ai[i*3+j] = cA[j*3+i]*id;     // A⁻¹ = cofᵀ / det
        float L[9]; mm3(Gv, Ai, L);
        float J = det3(X);
        float tv[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) tv[i*3+j] = visc*J*(L[i*3+j] + L[j*3+i]);
        float Gw[9]; mmt3(tv, Ai, Gw);
        for (int k=0;k<9;k++) Gm[k] += Gw[k];
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float sacc = 0.f; for (int b=0;b<3;b++) sacc += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*sacc);
    }
}
// c_fc 와 같은 틀 (사면체 블록 nbT 개 + 바닥 후보 블록) -- 사면체 블록만 ag_force_body
__global__ void ag_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                      float mu, float lam, float* f, float* rst, int fpol,
                      int mat, const float* prm, const float* ysT, const float* ljT, float dtc, const float* vn, float visc, int cflag,
                      float* FeC, float* XpC, float* ysC, float* ljC,
                      const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                      float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
    if ((int)blockIdx.x < nbT) {
        int i0 = blockIdx.x*blockDim.x + threadIdx.x;
        int t;
        if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
        else { if (i0 >= szp[1]) return; t = i0; }
        ag_force_body(t, tets, U, gl, FT, VT, mu, lam, f, rst, fpol, mat, prm, ysT, ljT, dtc, vn, visc, cflag, FeC, XpC, ysC, ljC);
        return;
    }
    int q = (blockIdx.x - nbT)*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float sd = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(sd < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// 셀 증분 상태 초기화 (서브스텝 그래프 첫 커널): F_e = X_prev = F̄, 상태 = 입자 평균
__global__ void ag_cinit(const int* szp, const float* FT, const float* ysT, const float* ljT, float* FeC, float* XpC, float* ysC, float* ljC) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    for (int k=0;k<9;k++) { FeC[t*9+k] = FT[t*9+k]; XpC[t*9+k] = FT[t*9+k]; }
    ysC[t] = ysT ? ysT[t] : 0.f; ljC[t] = ljT ? ljT[t] : 0.f;
}
// 프레임 끝 입자 return mapping (상태 갱신) -- rmc[0] += 투영된 입자 수, rmc[1] += 새로 손상된 입자 수
__global__ void ag_rm(int N, float* F, float* ys, float* lj, float* al, int mat, const float* prm, float dt, float* rmc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float X[9]; for (int k=0;k<9;k++) X[k] = F[p*9+k];
    float Uu[9], s[3], Vv[9]; pl_svd(X, Uu, s, Vv);
    float y_ = ys ? ys[p] : 0.f, l_ = lj ? lj[p] : 0.f, a_ = al ? al[p] : 1.f, a0 = a_;
    bool ch = pl_rm(mat, s, prm, y_, l_, a_, dt);
    if (ys) ys[p] = y_;
    if (lj) lj[p] = l_;
    if (al) al[p] = a_;
    if (ch) {
        float Y[9]; pl_usv(Uu, s, Vv, Y);
        for (int k=0;k<9;k++) F[p*9+k] = Y[k];
        if (rmc && mat != 7) atomicAdd(&rmc[0], 1.f);
    }
    if (rmc && a0 > 0.f && !(a_ > 0.f)) atomicAdd(&rmc[1], 1.f);
}
// 입자 속도 구속의 노드 몫 섞기 (노드 걸음 뒤): v <- (1 - α) v + pv,  pv = α v̄_bc (프레임마다 P2G 질량 가중으로)
__global__ void ag_pbc(const int* szp, float* vn, const float* pa, const float* pv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= szp[0]) return;
    float a = pa[n];
    if (!(a > 0.f)) return;
    for (int c=0;c<3;c++) vn[n*3+c] = (1.f - a)*vn[n*3+c] + pv[n*3+c];
}
// 단위 대조: return mapping (갱신) + 키르히호프 응력 τ = P F_eᵀ (대칭화)
__global__ void ag_test(int N, const float* Fin, float* ys, float* lj, float* al, int mat, const float* prm, float dt, float* Fout, float* tau) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float X[9]; for (int k=0;k<9;k++) X[k] = Fin[p*9+k];
    float Uu[9], s[3], Vv[9]; pl_svd(X, Uu, s, Vv);
    float y_ = ys[p], l_ = lj[p], a_ = al[p];
    bool ch = pl_rm(mat, s, prm, y_, l_, a_, dt);
    ys[p] = y_; lj[p] = l_; al[p] = a_;
    float Fe[9];
    if (ch) pl_usv(Uu, s, Vv, Fe); else for (int k=0;k<9;k++) Fe[k] = X[k];
    for (int k=0;k<9;k++) Fout[p*9+k] = Fe[k];
    float T[9];
    if (mat == 7) pl_nhb_tau(Fe, prm[0], prm[2], T);
    else {
        float U2[9], s2[3], V2[9]; pl_svd(Fe, U2, s2, V2);
        float Pi[3]; pl_pi(mat, s2, prm, a_, Pi);
        float P[9]; pl_usv(U2, Pi, V2, P);
        mmt3(P, Fe, T);
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) tau[p*9+i*3+j] = 0.5f*(T[i*3+j] + T[j*3+i]);
}
}
'''

_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + _PL).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def launch(name, n, args, b=128, grid=None):
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        mod().get_function(name)((g,), (b,), args)


P = EX.P


def params(cfg, mu, lam, dev="cuda"):
    """정답 config -> (재질 번호, prm 장치 텐서 [16], 설명 dict). 기본값은 정답 솔버 것 (PG: friction 25°, hardening 0, ξ 0, softening 0.1;
    GF: friction 45°, β 1, logJp 초기 -0.04 -- 수박 config 는 모두 명시)."""
    mname = cfg.get("material", "jelly")
    if mname not in MAT:
        raise SystemExit(f"--plast: 재질 '{mname}' 은 아직 없다 ({sorted(MAT)})")
    mat = MAT[mname]
    gf = mat == 7
    phi = float(cfg.get("friction_angle", 45.0 if gf else 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    Mn = alpha * 3.0 / math.sqrt(2.0 / (6.0 - 3.0))
    kap = 2.0 * mu / 3.0 + lam
    pr = dict(mu=mu, lam=lam, kappa=kap, alpha=alpha, M=Mn, beta=float(cfg.get("beta", 1.0)), xi=float(cfg.get("xi", 0.0)),
              hardening=float(cfg.get("hardening", 0.0)), softening=float(cfg.get("softening", 0.1)), eta=float(cfg.get("plastic_viscosity", 0.0)),
              yield_stress=float(cfg.get("yield_stress", 0.0)), logJp0=float(cfg.get("alpha_0", -0.04)), friction_angle=phi)
    v = [pr["mu"], pr["lam"], pr["kappa"], pr["alpha"], pr["M"], pr["beta"], pr["xi"], pr["hardening"], pr["softening"], pr["eta"]]
    t = torch.zeros(16, device=dev, dtype=torch.float32)
    t[:len(v)] = torch.tensor(v, dtype=torch.float32)
    return mat, t, pr
