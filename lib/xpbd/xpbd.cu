//@@ _SRC 17195

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

// [final_fix_vrgs] 물체-물체 충돌 (XPBD 표준 정점 비관통 부등식 구속, 논문에 세부 없음 -- doc/pareto.md 절 P.1c)
//   해시 격자 (칸 = 접촉 거리 d = 케이지 칸 h) 에 정점을 넣고, 다른 성분 (케이지 연결 성분 = 물체) 정점 j 와 |x_i − x_j| < d 이면
//   C = |x_i − x_j| − d >= 0 를 강성 무한 (α = 0) 으로 투영. 마찰 없음이 기본 (fr 0); fr 1 = 접선 상대 변위 취소 (고착, 예전 판).
//   야코비 + 구속 평균 (Macklin et al. 2014 Unified Particle Physics 4.2): 정점마다 접촉 보정 합 / 접촉 수.
__device__ __forceinline__ unsigned vg_hash(int ix, int iy, int iz, unsigned mask) {
  return (((unsigned)ix * 73856093u) ^ ((unsigned)iy * 19349663u) ^ ((unsigned)iz * 83492791u)) & mask;
}
__global__ void vg_hclear(int n, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) cnt[i] = 0;
}
__global__ void vg_hins(int n, const float* x, const int* comp, float ih, unsigned mask, int cap, int* cnt, int* slot) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || comp[i] < 0) return;
  unsigned b = vg_hash((int)floorf(x[3*i]*ih), (int)floorf(x[3*i+1]*ih), (int)floorf(x[3*i+2]*ih), mask);
  int k = atomicAdd(&cnt[b], 1);
  if (k < cap) slot[b*cap + k] = i;
}
__global__ void vg_contact(int n, const float* x, const float* xp, const float* w, const int* comp, float ih, unsigned mask, int cap,
                           const int* cnt, const int* slot, float d, float fr, float* dx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float ax = 0.f, ay = 0.f, az = 0.f; int nc = 0;
  float wi = w[i];
  if (wi > 0.f && comp[i] >= 0) {
    float xi = x[3*i], yi = x[3*i+1], zi = x[3*i+2];
    float ui = xi - xp[3*i], vi = yi - xp[3*i+1], si = zi - xp[3*i+2];
    int cx = (int)floorf(xi*ih), cy = (int)floorf(yi*ih), cz = (int)floorf(zi*ih);
    int ci = comp[i];
    float d2 = d*d;
    for (int ox = -1; ox <= 1; ox++) for (int oy = -1; oy <= 1; oy++) for (int oz = -1; oz <= 1; oz++) {
      int qx = cx+ox, qy = cy+oy, qz = cz+oz;
      unsigned b = vg_hash(qx, qy, qz, mask);
      int m = min(cnt[b], cap);
      for (int k = 0; k < m; k++) {
        int j = slot[b*cap + k];
        if (comp[j] == ci) continue;
        float xj = x[3*j], yj = x[3*j+1], zj = x[3*j+2];
        if ((int)floorf(xj*ih) != qx || (int)floorf(yj*ih) != qy || (int)floorf(zj*ih) != qz) continue;   // 해시 충돌로 같은 버킷에 든 다른 칸 정점 (중복 방지)
        float rx = xi-xj, ry = yi-yj, rz = zi-zj;
        float r2 = rx*rx + ry*ry + rz*rz;
        if (r2 >= d2 || r2 < 1e-20f) continue;
        float r = sqrtf(r2), nx = rx/r, ny = ry/r, nz = rz/r;
        float wj = w[j];
        float sw = wi / (wi + wj);
        float pen = d - r;
        float dux = ui - (xj - xp[3*j]), duy = vi - (yj - xp[3*j+1]), duz = si - (zj - xp[3*j+2]);
        float dn = dux*nx + duy*ny + duz*nz;
        float tx = dux - dn*nx, ty = duy - dn*ny, tz = duz - dn*nz;
        ax += sw*(pen*nx - fr*tx); ay += sw*(pen*ny - fr*ty); az += sw*(pen*nz - fr*tz);   // fr 0 = 마찰 없음 (기본), 1 = 고착
        nc++;
      }
    }
  }
  float inv = nc > 0 ? 1.f / (float)nc : 0.f;
  dx[3*i] = ax*inv; dx[3*i+1] = ay*inv; dx[3*i+2] = az*inv;
}
__global__ void vg_addx(int n, float* x, const float* dx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  x[3*i] += dx[3*i]; x[3*i+1] += dx[3*i+1]; x[3*i+2] += dx[3*i+2];
}
// [final_fix_vrgs] PG bounding_box (mpm_solver_warp add_bounding_box: 벽에서 padding 3 칸 안 격자 노드의 바깥 방향 속도 0 = 미끄럼 벽) -> 정점 축별 [lo, hi] 투영 (접선 자유)
__global__ void vg_box(int n, float* x, const float* w, float lx, float ly, float lz, float hx, float hy, float hz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || w[i] <= 0.f) return;
  x[3*i] = fminf(fmaxf(x[3*i], lx), hx); x[3*i+1] = fminf(fmaxf(x[3*i+1], ly), hy); x[3*i+2] = fminf(fmaxf(x[3*i+2], lz), hz);
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

