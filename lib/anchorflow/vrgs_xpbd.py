"""VR-GS (Jiang et al., SIGGRAPH 2024, arXiv 2401.16663) 직접 구현 -- XPBD + 고정 위상 사면체 케이지에 3DGS 임베딩 (탄성).

코드가 공개되지 않아 논문 본문에 있는 것은 그대로, 없는 것은 표준 XPBD 관행으로 메운다 (doc/pareto.md 절 P.1 에 전부 적음).

논문에 있는 것
  - 시뮬레이터 XPBD (Macklin et al. 2016): 서브스텝마다 예측 -> 구속 반복 -> 속도 = 위치 차 / dt, 라그랑주 승수 λ 는 서브스텝마다 0.
  - 탄성 = 변형 에너지 구속 ("strain energy constraint"), 물성 E·ν·ρ.
  - 가우시안 중심 = 담고 있는 사면체의 무게중심 보간 μ = Σ w_i x_i (rest 에서 정한 w), 공분산 Σ = F Σ0 Fᵀ (F = Ds Dm⁻¹).
  - 케이지 정점 수 10K~30K.
  - 바닥 충돌 (물성 조정에 씀). 서브스텝당 반복 수는 장면마다 1~50 (표 1), PhysGaussian 비교는 dt 1e-4 s·반복 11.
논문에 없는 것 (표준 관행)
  - 케이지: 논문은 VDB -> marching cubes -> TetGen. 여기는 가우시안 중심 점유 격자 (1 칸 팽창) -> 칸마다 Kuhn 6 사면체 (meshgs/tetcage.py 와 같은 구조).
    점유 격자 칸 크기는 정점 수가 상한 (기본 30K) 이하인 가장 작은 칸.
  - 변형 에너지 구속의 꼴: 정답 재질 (fixed corotated, ψ = μ‖F−R‖² + λ/2 (det F − 1)²) 을 두 구속으로 나눠 에너지를 정확히 맞춤
      C_R = ‖F − R‖_F, α_R = 1/(2μV);  C_V = det F − 1, α_V = 1/(λV)   (½C_R²/α_R + ½C_V²/α_V = V ψ)
    사면체마다 두 구속을 2x2 로 함께 푼다 (Macklin & Müller 2021 의 결합 풀이). R 은 Müller et al. 2016 반복 회전 추출 (사면체별 사원수 따뜻한 시작, 5 회).
  - 풀이 순서: 가우스-자이델, 사면체 탐욕 색칠 (같은 색은 꼭짓점을 공유하지 않음) -- GPU XPBD 표준.
  - 질량: 사면체 집중 질량 ρ V_T / 4 (ρ = 사면체 안 가우시안의 입자 밀도 평균, 없으면 config 밀도).
  - 바닥: 정점 비관통 z ≥ 바닥 (반복마다 투영) + 고착 (닿은 정점은 이번 서브스텝 접선 이동 취소) = 정답 sticky collider.
  - 고정 경계 (ficus cuboid): rest 위치가 상자 안인 정점 역질량 0. 시간 구간 cuboid (pillow2sofa) 는 PG 와 같이 start <= t < end 동안만 (프레임 시작 시각 판정).
  - 감쇠: 씬 config grid_v_damping_scale 를 같은 시간율로 정점 속도에 (서브스텝마다 s^(dt/substep_dt)); lego 는 1 (없음).
  - 초기 속도: 입자 속도의 질량·무게중심 가중 평균 (한 번, 프레임 0).
서브스텝 안에서는 입자를 쓰지 않는다 (가우시안 갱신은 프레임 끝 한 번).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import fem_rt_cuda as RTC

_SRC = r"""
extern "C" {
__device__ __forceinline__ void qmat(const float* q, float* R) {   // q = (x,y,z,w) -> R 행 우선
  float x=q[0],y=q[1],z=q[2],w=q[3];
  R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-z*w);   R[2]=2*(x*z+y*w);
  R[3]=2*(x*y+z*w);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-x*w);
  R[6]=2*(x*z-y*w);   R[7]=2*(y*z+x*w);   R[8]=1-2*(x*x+y*y);
}

// [blx] 3x3 SVD (야코비: A = Fᵀ F 고유분해 -> V, σ; U = F V / σ, Gram-Schmidt; U·V 회전, 뒤집힘이면 σ3 < 0 -- warp svd3 와 같은 부호 규약)
__device__ void svd3x3(const float* F, float* U, float* S, float* V) {
  float A[9];
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { float a = 0; for (int k = 0; k < 3; k++) a += F[3*k+i]*F[3*k+j]; A[3*i+j] = a; }
  float Q[9] = {1.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f,1.f};
  for (int sw = 0; sw < 12; sw++) {
    float off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
    if (off < 1e-12f * (A[0]*A[0] + A[4]*A[4] + A[8]*A[8]) + 1e-35f) break;
    for (int pq = 0; pq < 3; pq++) {
      int p = pq == 2 ? 1 : 0, q = pq == 0 ? 1 : 2;
      float apq = A[3*p+q]; if (fabsf(apq) < 1e-35f) continue;
      float th = (A[3*q+q] - A[3*p+p]) / (2.0f*apq);
      float t = (th >= 0 ? 1.0f : -1.0f) / (fabsf(th) + sqrtf(th*th + 1.0f));
      float c = 1.0f / sqrtf(t*t + 1.0f), sn = t*c;
      for (int k = 0; k < 3; k++) { float akp = A[3*k+p], akq = A[3*k+q]; A[3*k+p] = c*akp - sn*akq; A[3*k+q] = sn*akp + c*akq; }
      for (int k = 0; k < 3; k++) { float apk = A[3*p+k], aqk = A[3*q+k]; A[3*p+k] = c*apk - sn*aqk; A[3*q+k] = sn*apk + c*aqk; }
      for (int k = 0; k < 3; k++) { float qkp = Q[3*k+p], qkq = Q[3*k+q]; Q[3*k+p] = c*qkp - sn*qkq; Q[3*k+q] = sn*qkp + c*qkq; }
    }
  }
  float ev[3] = {A[0], A[4], A[8]}; int o[3] = {0, 1, 2};
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2 - i; j++) if (ev[o[j]] < ev[o[j+1]]) { int tt = o[j]; o[j] = o[j+1]; o[j+1] = tt; }
  float v[3][3];
  for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) v[c][r] = Q[3*r+o[c]];
  // det V = +1
  float dv = v[0][0]*(v[1][1]*v[2][2]-v[1][2]*v[2][1]) - v[0][1]*(v[1][0]*v[2][2]-v[1][2]*v[2][0]) + v[0][2]*(v[1][0]*v[2][1]-v[1][1]*v[2][0]);
  if (dv < 0) for (int r = 0; r < 3; r++) v[2][r] = -v[2][r];
  float fv[3][3];
  for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) fv[c][r] = F[3*r]*v[c][0] + F[3*r+1]*v[c][1] + F[3*r+2]*v[c][2];
  float u[3][3];
  float n0 = sqrtf(fv[0][0]*fv[0][0] + fv[0][1]*fv[0][1] + fv[0][2]*fv[0][2]);
  if (n0 > 1e-30f) { for (int r = 0; r < 3; r++) u[0][r] = fv[0][r] / n0; } else { u[0][0] = 1; u[0][1] = 0; u[0][2] = 0; }
  float d1 = u[0][0]*fv[1][0] + u[0][1]*fv[1][1] + u[0][2]*fv[1][2];
  float w1[3] = {fv[1][0] - d1*u[0][0], fv[1][1] - d1*u[0][1], fv[1][2] - d1*u[0][2]};
  float n1 = sqrtf(w1[0]*w1[0] + w1[1]*w1[1] + w1[2]*w1[2]);
  if (n1 > 1e-30f) { for (int r = 0; r < 3; r++) u[1][r] = w1[r] / n1; }
  else { float e[3] = {0, 0, 0}; e[fabsf(u[0][0]) < 0.9f ? 0 : 1] = 1; float dd = e[0]*u[0][0] + e[1]*u[0][1] + e[2]*u[0][2];
         float w[3] = {e[0]-dd*u[0][0], e[1]-dd*u[0][1], e[2]-dd*u[0][2]}; float nn = sqrtf(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]); for (int r = 0; r < 3; r++) u[1][r] = w[r]/nn; }
  u[2][0] = u[0][1]*u[1][2] - u[0][2]*u[1][1]; u[2][1] = u[0][2]*u[1][0] - u[0][0]*u[1][2]; u[2][2] = u[0][0]*u[1][1] - u[0][1]*u[1][0];
  S[0] = n0; S[1] = (u[1][0]*fv[1][0] + u[1][1]*fv[1][1] + u[1][2]*fv[1][2]); S[2] = (u[2][0]*fv[2][0] + u[2][1]*fv[2][1] + u[2][2]*fv[2][2]);
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { U[3*r+c] = u[c][r]; V[3*r+c] = v[c][r]; }
}

__global__ void vg_predict(int n, float* x, float* xp, float* v, const float* w, float gx, float gy, float gz, float dt, float damp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float a = x[3*i], b = x[3*i+1], c = x[3*i+2];
  xp[3*i] = a; xp[3*i+1] = b; xp[3*i+2] = c;
  if (w[i] <= 0.f) return;
  float vx = v[3*i]*damp + gx*dt, vy = v[3*i+1]*damp + gy*dt, vz = v[3*i+2]*damp + gz*dt;
  v[3*i] = vx; v[3*i+1] = vy; v[3*i+2] = vz;
  x[3*i] = a + vx*dt; x[3*i+1] = b + vy*dt; x[3*i+2] = c + vz*dt;
}

// [paper_remeasure2] 구동 정점 (PG enforce_particle_velocity_rotation): 질량 무한 (w = 0) 정점을 프레임 속도 vd 로 움직임
__global__ void vg_kin(int n, float* x, const float* vd, const unsigned char* kin, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || !kin[i]) return;
  x[3*i] += vd[3*i]*dt; x[3*i+1] += vd[3*i+1]*dt; x[3*i+2] += vd[3*i+2]*dt;
}

__global__ void vg_zero(int n, float* lam) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) lam[i] = 0.f;
}

// 같은 색 사면체 [off, off+cnt)
__global__ void vg_solve(int off, int cnt, const int* T, float* x, const float* w, const float* Dmi, const float* aR, const float* aV,
                         float* lam, float* q, int nrot, int mat, const unsigned char* dead) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= cnt) return;
  int t = off + k;
  if (dead[t]) return;                                       // [blx] PG plasticine 손상: 항복 응력 ≤ 0 이면 mu = lam = 0 (응력 없음)
  int id[4]; float p[4][3], wi[4];
  for (int a = 0; a < 4; a++) { id[a] = T[4*t+a]; wi[a] = w[id[a]]; for (int r = 0; r < 3; r++) p[a][r] = x[3*id[a]+r]; }
  float D[9]; for (int r = 0; r < 3; r++) for (int j = 0; j < 3; j++) D[3*r+j] = p[j+1][r] - p[0][r];
  const float* M = Dmi + 9*t;
  float F[9];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) F[3*r+c] = D[3*r]*M[c] + D[3*r+1]*M[3+c] + D[3*r+2]*M[6+c];
  // 회전 추출 (Müller 2016)
  float qq[4] = {q[4*t], q[4*t+1], q[4*t+2], q[4*t+3]};
  float R[9];
  for (int it = 0; it < nrot; it++) {
    qmat(qq, R);
    float om[3] = {0.f, 0.f, 0.f}; float dd = 0.f;
    for (int c = 0; c < 3; c++) {
      float rc0 = R[c], rc1 = R[3+c], rc2 = R[6+c], f0 = F[c], f1 = F[3+c], f2 = F[6+c];
      om[0] += rc1*f2 - rc2*f1; om[1] += rc2*f0 - rc0*f2; om[2] += rc0*f1 - rc1*f0;
      dd += rc0*f0 + rc1*f1 + rc2*f2;
    }
    float s = 1.f / (fabsf(dd) + 1e-9f);
    om[0] *= s; om[1] *= s; om[2] *= s;
    float ang = sqrtf(om[0]*om[0] + om[1]*om[1] + om[2]*om[2]);
    if (ang < 1e-9f) break;
    float ha = 0.5f*ang, sa = sinf(ha)/ang, ca = cosf(ha);
    float ax = om[0]*sa, ay = om[1]*sa, az = om[2]*sa;   // dq * qq
    float nx = ca*qq[0] + ax*qq[3] + ay*qq[2] - az*qq[1];
    float ny = ca*qq[1] + ay*qq[3] + az*qq[0] - ax*qq[2];
    float nz = ca*qq[2] + az*qq[3] + ax*qq[1] - ay*qq[0];
    float nw = ca*qq[3] - ax*qq[0] - ay*qq[1] - az*qq[2];
    float nn = rsqrtf(nx*nx + ny*ny + nz*nz + nw*nw);
    qq[0] = nx*nn; qq[1] = ny*nn; qq[2] = nz*nn; qq[3] = nw*nn;
  }
  qmat(qq, R);
  q[4*t] = qq[0]; q[4*t+1] = qq[1]; q[4*t+2] = qq[2]; q[4*t+3] = qq[3];
  float G[9]; float CR = 0.f;
  for (int i = 0; i < 9; i++) { G[i] = F[i] - R[i]; CR += G[i]*G[i]; }
  CR = sqrtf(CR);
  // cof F 열: c0 = f1 x f2, c1 = f2 x f0, c2 = f0 x f1 (f_j = F 열)
  float f0[3] = {F[0], F[3], F[6]}, f1[3] = {F[1], F[4], F[7]}, f2[3] = {F[2], F[5], F[8]};
  float Cf[9];
  Cf[0] = f1[1]*f2[2]-f1[2]*f2[1]; Cf[3] = f1[2]*f2[0]-f1[0]*f2[2]; Cf[6] = f1[0]*f2[1]-f1[1]*f2[0];
  Cf[1] = f2[1]*f0[2]-f2[2]*f0[1]; Cf[4] = f2[2]*f0[0]-f2[0]*f0[2]; Cf[7] = f2[0]*f0[1]-f2[1]*f0[0];
  Cf[2] = f0[1]*f1[2]-f0[2]*f1[1]; Cf[5] = f0[2]*f1[0]-f0[0]*f1[2]; Cf[8] = f0[0]*f1[1]-f0[1]*f1[0];
  float J = f0[0]*Cf[0] + f0[1]*Cf[3] + f0[2]*Cf[6];
  float CV = J - 1.f;
  if (mat == 2) {                                            // [blx] PG 모래 탄성 = Hencky (kirchoff_stress_drucker_prager: τ = 2μ log σ + λ tr log σ):
    // ψ V = ½ C1²/α_R + ½ C2²/α_V,  C1 = |ε|, C2 = tr ε (ε = log|σ|), α 는 FCR 과 같은 1/(2μV)·1/(λV); ∂C1/∂F = U diag(ε/σ) Vᵀ / |ε|, ∂C2/∂F = U diag(1/σ) Vᵀ
    float U_[9], S_[3], V_[9];
    svd3x3(F, U_, S_, V_);
    float e_[3], is_[3];
    for (int i = 0; i < 3; i++) { float sa = fmaxf(fabsf(S_[i]), 1e-14f); e_[i] = logf(sa); is_[i] = 1.f / (S_[i] >= 0.f ? sa : -sa); }
    CR = sqrtf(e_[0]*e_[0] + e_[1]*e_[1] + e_[2]*e_[2]);
    CV = e_[0] + e_[1] + e_[2];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
      float g = 0.f, h = 0.f;
      for (int i = 0; i < 3; i++) { g += U_[3*r+i] * e_[i] * is_[i] * V_[3*c+i]; h += U_[3*r+i] * is_[i] * V_[3*c+i]; }
      G[3*r+c] = g; Cf[3*r+c] = h;                            // 아래 공통 식: gR = G Dm⁻ᵀ / CR, gV = Cf Dm⁻ᵀ
    }
  }
  // 정점 기울기: H = G Dm⁻ᵀ, 정점 j+1 = H 열 j, 정점 0 = −합
  float gR[4][3], gV[4][3];
  bool useR = CR > 1e-7f;
  float iC = useR ? 1.f / CR : 0.f;
  for (int r = 0; r < 3; r++) {
    float sR = 0.f, sV = 0.f;
    for (int j = 0; j < 3; j++) {
      float hR = (G[3*r]*M[3*j] + G[3*r+1]*M[3*j+1] + G[3*r+2]*M[3*j+2]) * iC;
      float hV = Cf[3*r]*M[3*j] + Cf[3*r+1]*M[3*j+1] + Cf[3*r+2]*M[3*j+2];
      gR[j+1][r] = hR; gV[j+1][r] = hV; sR += hR; sV += hV;
    }
    gR[0][r] = -sR; gV[0][r] = -sV;
  }
  float A11 = aR[t], A22 = aV[t], A12 = 0.f;
  for (int a = 0; a < 4; a++) for (int r = 0; r < 3; r++) {
    A11 += wi[a]*gR[a][r]*gR[a][r]; A22 += wi[a]*gV[a][r]*gV[a][r]; A12 += wi[a]*gR[a][r]*gV[a][r];
  }
  float lR = lam[2*t], lV = lam[2*t+1];
  float b1 = -(CR + aR[t]*lR), b2 = -(CV + aV[t]*lV);
  float dR = 0.f, dV = 0.f;
  float det = A11*A22 - A12*A12;
  if (useR && fabsf(det) > 1e-30f) { dR = (b1*A22 - b2*A12) / det; dV = (A11*b2 - A12*b1) / det; }
  else if (A22 > 0.f) { dV = b2 / A22; }
  lam[2*t] = lR + dR; lam[2*t+1] = lV + dV;
  for (int a = 0; a < 4; a++) {
    if (wi[a] <= 0.f) continue;
    for (int r = 0; r < 3; r++) x[3*id[a]+r] = p[a][r] + wi[a]*(gR[a][r]*dR + gV[a][r]*dV);
  }
}

// [blx] 서브스텝 끝 반환 사상 (PG mpm_utils.py: sand_return_mapping · von_mises_return_mapping_with_damage 줄 단위 대조). F_e = Ds Dmi (Dmi = Dm⁻¹ F_p⁻¹),
//   F_e 새 값 = U diag(s_new) Vᵀ = F_e V diag(s_new/σ) Vᵀ  ->  Dmi <- Dmi V diag(s_new/σ) Vᵀ (위치는 안 바뀜, F_p 만 흡수).
__global__ void vg_plast(int nt, const int* T, const float* x, float* Dmi, float* ys, unsigned char* dead, int mat, const float* ml, float cdp, float soft) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nt || dead[t]) return;
  float p[4][3];
  for (int a = 0; a < 4; a++) for (int r = 0; r < 3; r++) p[a][r] = x[3*T[4*t+a]+r];
  float D[9]; for (int r = 0; r < 3; r++) for (int j = 0; j < 3; j++) D[3*r+j] = p[j+1][r] - p[0][r];
  float* M = Dmi + 9*t;
  float F[9];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) F[3*r+c] = D[3*r]*M[c] + D[3*r+1]*M[3+c] + D[3*r+2]*M[6+c];
  float U[9], S[3], V[9];
  svd3x3(F, U, S, V);
  float mu = ml[0], la = ml[1];
  float sn[3];
  if (mat == 2) {                                            // sand_return_mapping
    float e[3]; for (int i = 0; i < 3; i++) e[i] = logf(fmaxf(fabsf(S[i]), 1e-14f));
    float tr = e[0] + e[1] + e[2];
    float eh[3] = {e[0] - tr/3.f, e[1] - tr/3.f, e[2] - tr/3.f};
    float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
    float dg = ehn + cdp * tr;                               // cdp = (3λ+2μ)/(2μ) α
    if (dg <= 0.f) return;                                   // F_elastic = F_trial
    if (tr > 0.f) { sn[0] = sn[1] = sn[2] = 1.f; }           // F_elastic = U Vᵀ
    else { for (int i = 0; i < 3; i++) sn[i] = expf(e[i] - eh[i] * (dg / ehn)); }
  } else if (mat == 5) {                                     // von_mises_return_mapping_with_damage
    float sg[3] = {fmaxf(S[0], 0.01f), fmaxf(S[1], 0.01f), fmaxf(S[2], 0.01f)};
    float e[3] = {logf(sg[0]), logf(sg[1]), logf(sg[2])};
    float temp = (e[0] + e[1] + e[2]) / 3.f;
    float tau[3]; for (int i = 0; i < 3; i++) tau[i] = 2.f*mu*e[i] + la*(e[0] + e[1] + e[2]);
    float st = tau[0] + tau[1] + tau[2];
    float cd[3] = {tau[0] - st/3.f, tau[1] - st/3.f, tau[2] - st/3.f};
    if (!(sqrtf(cd[0]*cd[0] + cd[1]*cd[1] + cd[2]*cd[2]) > ys[t])) return;
    if (ys[t] <= 0.f) return;
    float eh[3] = {e[0] - temp, e[1] - temp, e[2] - temp};
    float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]) + 1e-6f;
    float dg = ehn - ys[t] / (2.f*mu);
    float dl = 0.f;
    for (int i = 0; i < 3; i++) { float d = (dg / ehn) * eh[i]; e[i] -= d; dl += d*d; }
    ys[t] = ys[t] - soft * sqrtf(dl);
    if (ys[t] <= 0.f) dead[t] = 1;                           // model.mu = model.lam = 0
    for (int i = 0; i < 3; i++) sn[i] = expf(e[i]);
  } else return;
  float r_[3];
  for (int i = 0; i < 3; i++) { float sa = fmaxf(fabsf(S[i]), 1e-14f); r_[i] = sn[i] / (S[i] >= 0.f ? sa : -sa); }
  float W[9];                                                // W = V diag(r) Vᵀ
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { float w = 0.f; for (int i = 0; i < 3; i++) w += V[3*a+i] * r_[i] * V[3*b+i]; W[3*a+b] = w; }
  float Mn[9];
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Mn[3*a+b] = M[3*a]*W[b] + M[3*a+1]*W[3+b] + M[3*a+2]*W[6+b];
  for (int i = 0; i < 9; i++) M[i] = Mn[i];
}

__global__ void vg_floor(int n, float* x, const float* xp, const float* w, float fz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  if (w[i] > 0.f && x[3*i+2] < fz) { x[3*i+2] = fz; x[3*i] = xp[3*i]; x[3*i+1] = xp[3*i+1]; }
}

__global__ void vg_vel(int n, const float* x, const float* xp, float* v, float idt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  for (int r = 0; r < 3; r++) v[3*i+r] = (x[3*i+r] - xp[3*i+r]) * idt;
}

__global__ void vg_skin(int n, const int* tid, const float* bw, const int* T, const float* x, float* out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= n) return;
  int t = tid[p];
  float s0 = 0.f, s1 = 0.f, s2 = 0.f;
  for (int a = 0; a < 4; a++) { int v = T[4*t+a]; float ww = bw[4*p+a]; s0 += ww*x[3*v]; s1 += ww*x[3*v+1]; s2 += ww*x[3*v+2]; }
  out[3*p] = s0; out[3*p+1] = s1; out[3*p+2] = s2;
}
}
"""
_MOD = [None]


def _mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=_SRC, options=("--std=c++14", "--use_fast_math"))
    return _MOD[0]


_P = RTC._cp
# Kuhn 6 사면체 (주 대각 0-6), meshgs/tetcage.py cube_to_tets 와 같음
_CORN = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def _occ(x, h, dil=1):
    lo = x.min(0).values - (dil + 1) * h
    idx = torch.floor((x - lo) / h).long()
    R = (idx.max(0).values + dil + 2).tolist()
    occ = torch.zeros(R, dtype=torch.bool, device=x.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dil > 0:
        occ = torch.nn.functional.max_pool3d(occ.float()[None, None], 2 * dil + 1, 1, dil)[0, 0] > 0
    return occ, lo, idx


def build_cage(x, h, dil=1):
    """점유 격자 (칸 h, dil 칸 팽창) -> 정점 [V,3], 사면체 [T,4], 입자 결속 (사면체 번호, 무게중심 4)."""
    dev = x.device
    occ, lo, idx = _occ(x, h, dil)
    R = occ.shape
    cell = occ[:-1, :-1, :-1].clone()
    for (a, b, c) in _CORN[1:]:
        cell &= occ[a:R[0] - 1 + a, b:R[1] - 1 + b, c:R[2] - 1 + c]
    ci = torch.nonzero(cell)                                                            # [C,3]
    off = torch.tensor(_CORN, device=dev)
    vid = (ci[:, None, :] + off[None]).reshape(-1, 3)
    key = (vid[:, 0] * R[1] + vid[:, 1]) * R[2] + vid[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)
    V = torch.stack([uk // (R[1] * R[2]), (uk // R[2]) % R[1], uk % R[2]], -1).double() * h + lo.double()
    ck = inv.reshape(-1, 8)
    T = torch.stack([ck[:, list(t)] for t in _TETS], 1).reshape(-1, 4)                  # 칸 c 의 사면체 6c..6c+5
    # 입자 결속: 입자가 든 칸 = 정점 idx..idx+1 칸 (팽창 1 이면 반드시 있음)
    cid = torch.full((R[0] - 1, R[1] - 1, R[2] - 1), -1, dtype=torch.long, device=dev)
    cid[ci[:, 0], ci[:, 1], ci[:, 2]] = torch.arange(ci.shape[0], device=dev)
    pc = cid[idx[:, 0].clamp(max=R[0] - 2), idx[:, 1].clamp(max=R[1] - 2), idx[:, 2].clamp(max=R[2] - 2)]
    assert bool((pc >= 0).all()), "결속 실패: 입자가 든 칸이 케이지에 없음"
    xd = x.double()
    best_t = torch.zeros(x.shape[0], dtype=torch.long, device=dev)
    best_w = torch.zeros(x.shape[0], 4, dtype=torch.float64, device=dev)
    best_m = torch.full((x.shape[0],), -1e30, dtype=torch.float64, device=dev)
    for k in range(6):
        t = pc * 6 + k
        P = V[T[t]]                                                                     # [N,4,3]
        A = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        l = torch.linalg.solve(A, (xd - P[:, 0]).unsqueeze(-1)).squeeze(-1)
        wv = torch.cat([1.0 - l.sum(-1, keepdim=True), l], -1)
        m = wv.min(-1).values
        upd = m > best_m
        best_t = torch.where(upd, t, best_t); best_w = torch.where(upd[:, None], wv, best_w); best_m = torch.where(upd, m, best_m)
    return V, T, best_t, best_w, float(best_m.min())


def cage_for_max_verts(x, vmax, dil=1):
    """정점 수 ≤ vmax 인 가장 작은 칸 h (이분 탐색, 로그 척도)."""
    ext = float((x.max(0).values - x.min(0).values).max())
    lo_h, hi_h = ext / 400.0, ext / 4.0

    def nv(h):
        occ, _, _ = _occ(x, h, dil)
        return int(occ.sum())                                                           # 정점 수 상계 (점유 점 수)
    for _ in range(30):
        mid = math.sqrt(lo_h * hi_h)
        if nv(mid) > vmax:
            lo_h = mid
        else:
            hi_h = mid
    h = hi_h
    while True:
        V, T, tid, bw, mm = build_cage(x, h, dil)
        if V.shape[0] <= vmax:
            return h, V, T, tid, bw, mm
        h *= 1.01


def greedy_color(T, nv):
    """탐욕 색칠 (꼭짓점 공유 사면체는 다른 색) -- 공간 순서 (사면체 번호 = 칸 순서) 그대로."""
    Tn = T.cpu().numpy()
    vm = np.zeros(nv, dtype=np.uint64)
    col = np.empty(Tn.shape[0], dtype=np.int64)
    one = np.uint64(1)
    for t in range(Tn.shape[0]):
        a, b, c, d = Tn[t]
        m = int(vm[a] | vm[b] | vm[c] | vm[d])
        k = (~m & (m + 1)).bit_length() - 1
        assert k < 64
        col[t] = k
        bit = one << np.uint64(k)
        vm[a] |= bit; vm[b] |= bit; vm[c] |= bit; vm[d] |= bit
    return torch.from_numpy(col)


class VRGS:
    """케이지 XPBD. x0 [Np,3] 입자 rest, v0 [Np,3], m [Np] 입자 질량, rho_p [Np] 입자 밀도."""

    def __init__(s, x0, v0, m, rho_p, E, nu, g, frame_dt, N, iters, vmax=30000, floor_z=None, fix_boxes=(), damp_scale=1.0, damp_dt=None,
                 h=None, nrot=5, rot_bcs=(), fix_boxes_t=(), origin=None, plast=None, mov_boxes=()):
        dev = "cuda"
        s.Np = x0.shape[0]
        x0 = x0.to(dev).double()
        if h is None:
            s.h, V, T, tid, bw, s.bind_min = cage_for_max_verts(x0, vmax)
        else:
            s.h = h
            V, T, tid, bw, s.bind_min = build_cage(x0, h)
        s.nv, s.nt = V.shape[0], T.shape[0]
        col = greedy_color(T, s.nv).to(dev)
        order = torch.argsort(col, stable=True)
        inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=dev)
        T = T[order]; tid = inv[tid]
        cnt = torch.bincount(col, minlength=int(col.max()) + 1).tolist()
        s.colors = [(int(sum(cnt[:c])), int(cnt[c])) for c in range(len(cnt)) if cnt[c] > 0]
        P = V[T]
        Dm = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        s.vol = (torch.linalg.det(Dm).abs() / 6.0)
        s.Dmi = torch.linalg.inv(Dm).float().contiguous().reshape(-1)
        s.Dmi0 = s.Dmi.clone()                                                          # [blx] rest Dm⁻¹ (총 F = Ds Dmi0); s.Dmi = Dm⁻¹ F_p⁻¹ (소성이면 서브스텝마다 바뀜)
        # [blx] 재질 확장 (원 논문은 탄성 전용): plast = stab_bl_common.plastic_params 꼴 dict (material sand | plasticine), 없으면 탄성 (예전 그대로)
        s.plast = plast
        s.mat = {"sand": 2, "plasticine": 5}.get((plast or {}).get("material"), 0)
        if s.mat == 5:
            assert int(plast.get("hardening", 0)) == 0, "PG hardening 1 은 미지원 (config 에 없으면 0)"
        s.ys0 = float((plast or {}).get("yield_stress", 0.0))
        s.ys = torch.full((s.nt,), s.ys0, device=dev, dtype=torch.float32)
        s.dead = torch.zeros(s.nt, device=dev, dtype=torch.uint8)
        s.ml = torch.zeros(2, device=dev, dtype=torch.float32)
        # 질량: ρ V_T / 4, ρ = 사면체에 결속된 입자 밀도 평균 (없으면 전체 질량 가중 밀도)
        rho_p = rho_p.to(dev).double()
        rs = torch.zeros(s.nt, device=dev, dtype=torch.float64).index_add_(0, tid, rho_p)
        rc = torch.zeros(s.nt, device=dev, dtype=torch.float64).index_add_(0, tid, torch.ones_like(rho_p))
        rho_t = torch.where(rc > 0, rs / rc.clamp(min=1), (m.to(dev).double().sum() / (m.to(dev).double() / rho_p).sum()))
        Mv = torch.zeros(s.nv, device=dev, dtype=torch.float64).index_add_(0, T.reshape(-1), (rho_t * s.vol / 4.0).repeat_interleave(4))
        w = 1.0 / Mv
        s.fixed = torch.zeros(s.nv, dtype=torch.bool, device=dev)
        for (pt, sz) in fix_boxes:
            s.fixed |= ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1)
        # 회전 구동기 (rot_bcs = [(점, 단위 법선, 반 높이, 반지름, [(회전 몫, 이동 몫, 시작, 끝), ..]), ..]): 처음 위치가 원기둥 안인 정점은 질량 무한 + 프레임 속도 구동
        s.rotg = list(rot_bcs)
        s.kin = torch.zeros(s.nv, dtype=torch.bool, device=dev)
        s.kin_g = []
        for (pt, nv_, hh, rr, wins) in s.rotg:
            pt_ = torch.tensor(pt, device=dev, dtype=torch.float64); n_ = torch.tensor(nv_, device=dev, dtype=torch.float64)
            off = V - pt_; ax = off @ n_; rp = off - ax[:, None] * n_
            msk = (ax.abs() < hh) & (rp.norm(dim=-1) < rr)
            s.kin_g.append(msk); s.kin |= msk
        s.fixed |= s.kin
        # [final_pa_bl] 시간 구간 고정 상자 (PG cuboid start <= t < end): rest 위치가 상자 안인 정점은 그 구간 동안만 역질량 0, 뒤엔 자유 (프레임 시작 시각으로 판정)
        s.fixw = []
        for (pt, sz, t0, t1) in fix_boxes_t:
            mw = ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1) & ~s.fixed
            s.fixw.append((mw, float(t0), float(t1)))
        # [blx] 구간 등속 구동 상자 (PG enforce_particle_translation 속도 ≠ 0; bread 당김): rest 위치가 상자 안인 정점은 start <= t < end 동안 질량 무한 + 속도 v 로 이동, 뒤엔 자유 (속도 유지)
        s.movw = []
        for (pt, sz, t0, t1, vel) in mov_boxes:
            mw = ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1) & ~s.fixed
            s.movw.append((mw, float(t0), float(t1), [float(q) for q in vel]))
        s.w_free = w.float().contiguous().clone()
        s.w_free[s.fixed] = 0.0
        w[s.fixed] = 0.0
        for (mw, t0, t1) in s.fixw:
            if t0 <= 1e-6 < t1:
                w[mw] = 0.0
        for (mw, t0, t1, _v) in s.movw:
            if t0 <= 1e-6 < t1:
                w[mw] = 0.0
        s.fr = 0
        s.vd = torch.zeros(s.nv, 3, device=dev, dtype=torch.float32)
        s.kin_u8 = s.kin.to(torch.uint8).contiguous()
        s.mass_cage = float(Mv.sum()); s.mass_part = float(m.sum())
        # 초기 속도: 입자 속도의 질량·무게중심 (음수는 0) 가중 평균
        m_ = m.to(dev).double(); v0 = v0.to(dev).double()
        wp_ = bw.clamp(min=0.0) * m_[:, None]
        Ms = torch.zeros(s.nv, device=dev, dtype=torch.float64).index_add_(0, T[tid].reshape(-1), wp_.reshape(-1))
        Ps = torch.zeros(s.nv, 3, device=dev, dtype=torch.float64).index_add_(0, T[tid].reshape(-1), (wp_[:, :, None] * v0[:, None, :]).reshape(-1, 3))
        vbar = (m_[:, None] * v0).sum(0) / m_.sum()
        vv = torch.where(Ms[:, None] > 0, Ps / Ms.clamp(min=1e-300)[:, None], vbar[None])
        vv[s.fixed] = 0.0
        vv[w == 0.0] = 0.0
        for (mw, t0, t1, vel) in s.movw:
            if t0 <= 1e-6 < t1:
                vv[mw] = torch.tensor(vel, device=dev, dtype=torch.float64)
        # [final_pa_bl] 좌표 원점 옮김 (평행 이동만; 동역학 불변): float32 위치에서 서브스텝 변위 g dt² 가 위치의 반 ulp 보다 작으면 (pillow2sofa: g 5, dt 1e-4 -> 5e-8,
        #   z >= 1 의 반 ulp 6e-8) 정지에서 출발한 정점이 반올림으로 제자리에 머물고 v = (x − xp)/dt = 0 이 되어 영영 안 떨어진다 -> 시뮬 내부 좌표를 origin 기준으로.
        #   출력 xo 는 프레임 끝 (시간 밖) 에 origin 을 더해 원래 좌표. origin None = 예전 그대로.
        s.origin = None if origin is None else torch.as_tensor(origin, device=dev, dtype=torch.float64)
        if s.origin is not None:
            V = V - s.origin
        s.X0 = V.float().contiguous(); s.V0 = vv.float().contiguous()
        s.x = s.X0.clone(); s.xp = s.X0.clone(); s.v = s.V0.clone()
        s.w = w.float().contiguous()
        s.T = T.int().contiguous(); s.tid = tid.int().contiguous(); s.bw = bw.float().contiguous()
        s.lam = torch.zeros(2 * s.nt, device=dev); s.q = torch.zeros(s.nt, 4, device=dev); s.q[:, 3] = 1.0
        s.xo = torch.zeros(s.Np, 3, device=dev)
        s.g = [float(q) for q in g]; s.frame_dt = float(frame_dt); s.nu = float(nu)
        s.fz = -1e30 if floor_z is None else float(floor_z) - (0.0 if s.origin is None else float(s.origin[2]))
        s.damp_scale, s.damp_dt = float(damp_scale), damp_dt
        s.nrot = int(nrot)
        s.aR = torch.zeros(s.nt, device=dev); s.aV = torch.zeros(s.nt, device=dev)
        s.set_setting(N, iters)
        s.set_E(E)
        s.g_sub = s.g_skin = None

    def set_setting(s, N, iters):
        s.N, s.iters = int(N), int(iters)
        s.dt = s.frame_dt / s.N
        s.damp = (s.damp_scale ** (s.dt / s.damp_dt)) if (s.damp_dt and s.damp_scale < 1.0) else 1.0
        s.g_sub = s.g_skin = None

    def set_E(s, E):
        s.E = float(E)
        mu = s.E / (2.0 * (1.0 + s.nu)); la = s.E * s.nu / ((1.0 + s.nu) * (1.0 - 2.0 * s.nu))
        dt2 = s.dt * s.dt
        s.aR.copy_((1.0 / (2.0 * mu * s.vol) / dt2).float())
        s.aV.copy_((1.0 / (la * s.vol) / dt2).float())
        s.ml.copy_(torch.tensor([mu, la], dtype=torch.float32))

    def reset(s):
        s.fr = 0
        s.x.copy_(s.X0); s.xp.copy_(s.X0); s.v.copy_(s.V0); s.q.zero_(); s.q[:, 3] = 1.0
        s.Dmi.copy_(s.Dmi0); s.ys.fill_(s.ys0); s.dead.zero_()
        if s.fixw or s.movw:
            s._pin()

    def _pin(s):
        """[final_pa_bl] 프레임 시작 시각 t 에 켜진 구간 고정 상자의 정점 역질량 0 (+ 속도 0), 꺼진 상자는 원래 역질량. 그래프는 s.w 를 그대로 읽음."""
        t = s.fr * s.frame_dt
        w = s.w_free.clone()
        on = torch.zeros_like(s.fixed)
        for (mw, t0, t1) in s.fixw:
            if t0 <= t + 1e-6 < t1:
                on |= mw
        w[on] = 0.0
        s.v[on] = 0.0
        if s.movw:                                                                       # [blx] 구동 중인 상자: 질량 무한 + vg_kin 으로 v dt 씩 (s.vd), 끝나면 자유
            kin = s.kin.clone()
            vd = s.vd.clone() if s.rotg else torch.zeros_like(s.vd)
            for (mw, t0, t1, vel) in s.movw:
                if t0 <= t + 1e-6 < t1:
                    w[mw] = 0.0; kin |= mw
                    vd[mw] = torch.tensor(vel, device=vd.device, dtype=vd.dtype)
                    s.v[mw] = vd[mw]
            s.kin_u8.copy_(kin.to(torch.uint8)); s.vd.copy_(vd)
        s.w.copy_(w)

    def _rec_sub(s):
        M = _mod()
        f_pred, f_zero, f_solve, f_floor, f_vel, f_kin = (M.get_function(k) for k in ("vg_predict", "vg_zero", "vg_solve", "vg_floor", "vg_vel", "vg_kin"))
        B = 128
        gv = ((s.nv + B - 1) // B,)
        i32, f32 = np.int32, np.float32
        with RTC._Stream():
            f_pred(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.v), _P(s.w), f32(s.g[0]), f32(s.g[1]), f32(s.g[2]), f32(s.dt), f32(s.damp)))
            if s.rotg or s.movw:
                f_kin(gv, (B,), (i32(s.nv), _P(s.x), _P(s.vd), _P(s.kin_u8), f32(s.dt)))
            f_zero(((2 * s.nt + B - 1) // B,), (B,), (i32(2 * s.nt), _P(s.lam)))
            for _ in range(s.iters):
                for (o, c) in s.colors:
                    f_solve(((c + B - 1) // B,), (B,), (i32(o), i32(c), _P(s.T), _P(s.x), _P(s.w), _P(s.Dmi), _P(s.aR), _P(s.aV), _P(s.lam), _P(s.q), i32(s.nrot),
                                                       i32(s.mat), _P(s.dead)))
                if s.fz > -1e29:
                    f_floor(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.w), f32(s.fz)))
            if s.mat:                                                                     # [blx] 서브스텝 끝 반환 사상 (구속 반복 뒤, 속도 갱신 전; 위치 불변)
                M.get_function("vg_plast")(((s.nt + B - 1) // B,), (B,), (i32(s.nt), _P(s.T), _P(s.x), _P(s.Dmi), _P(s.ys), _P(s.dead), i32(s.mat), _P(s.ml),
                                                                    f32(s.plast.get("cdp", 0.0)), f32(s.plast.get("softening", 0.1))))
            f_vel(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.v), f32(1.0 / s.dt)))

    def _rec_skin(s):
        f = _mod().get_function("vg_skin")
        with RTC._Stream():
            f(((s.Np + 127) // 128,), (128,), (np.int32(s.Np), _P(s.tid), _P(s.bw), _P(s.T), _P(s.x), _P(s.xo)))

    def capture(s):
        bk = (s.x.clone(), s.xp.clone(), s.v.clone(), s.q.clone(), s.Dmi.clone(), s.ys.clone(), s.dead.clone())
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            s._rec_sub(); s._rec_skin()                                                   # 데움 (함수 적재)
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        s.g_sub = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g_sub):
            s._rec_sub()
        s.g_skin = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g_skin):
            s._rec_skin()
        torch.cuda.synchronize()
        s.x.copy_(bk[0]); s.xp.copy_(bk[1]); s.v.copy_(bk[2]); s.q.copy_(bk[3]); s.Dmi.copy_(bk[4]); s.ys.copy_(bk[5]); s.dead.copy_(bk[6])

    def frame(s):
        """한 프레임 (N 서브스텝 + 가우시안 중심 갱신). 돌려줌: GPU 시간 ms (CUDA 이벤트), 입자 위치 s.xo."""
        if s.g_sub is None:
            s.capture()
        if s.rotg:
            s._drive()
        if s.fixw or s.movw:
            s._pin()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(s.N):
            s.g_sub.replay()
        s.g_skin.replay()
        e1.record()
        torch.cuda.synchronize()
        if s.origin is not None:
            s.xo += s.origin.float()
        s.fr += 1
        return e0.elapsed_time(e1)

    def _drive(s):
        """프레임 시각 [t, t+h) 와 겹친 몫 f 만큼 곱한 회전 구동 속도 (PG: 시각 start <= t < end, v = rs (r_perp x n) + ts n, r 은 현재 위치) 를 구동 정점에."""
        t = s.fr * s.frame_dt
        vd = torch.zeros(s.nv, 3, device=s.x.device, dtype=torch.float32)
        for (pt, nv_, hh, rr, wins), msk in zip(s.rotg, s.kin_g):
            rs = ts = 0.0
            for (r_, t_, t0, t1) in wins:
                f = max(0.0, min(t + s.frame_dt, t1) - max(t, t0)) / s.frame_dt
                rs += r_ * f; ts += t_ * f
            n_ = torch.tensor(nv_, device=s.x.device, dtype=torch.float32)
            off = s.x - torch.tensor(pt, device=s.x.device, dtype=torch.float32) + (0.0 if s.origin is None else s.origin.float())
            rp = off - (off @ n_)[:, None] * n_
            v = rs * torch.cross(rp, n_.expand_as(rp), dim=-1) + ts * n_
            vd[msk] = v[msk]
        s.vd.copy_(vd)

    def F_part(s):
        """입자 (가우시안) 마다 결속 사면체의 변형 기울기 F = Ds Dm^-1 [Np,3,3] (렌더 공분산 F Σ0 Fᵀ, 논문 그대로; 시간 밖)."""
        P = s.x[s.T.long()]
        Ds = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        return (Ds @ s.Dmi0.reshape(-1, 3, 3))[s.tid.long()]                         # [blx] 총 F (소성 포함 렌더 공분산·지표) -- 탄성이면 예전과 같음

    def info(s):
        return dict(h=s.h, nv=s.nv, nt=s.nt, ncolor=len(s.colors), nfixed=int(s.fixed.sum()), nfixed_t=[int(q[0].sum()) for q in s.fixw],
                    nmov=[int(q[0].sum()) for q in s.movw], mat=s.mat, plast=s.plast, ndead=int(s.dead.sum()), mass_cage=s.mass_cage, mass_part=s.mass_part,
                    bind_min=s.bind_min, N=s.N, iters=s.iters, dt=s.dt, damp=s.damp, E=s.E, nu=s.nu)
