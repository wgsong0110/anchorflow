// A differentiable MPM substep, for fitting a coarse particle set to a fine one.
//
// The chain's middle stage has been an anchor simulator: a few hundred anchors
// carrying positions and velocities, with every Gaussian's deformation gradient
// rebuilt from the arrangement at each step. F is the one quantity MPM carries
// forward and that state cannot, and it costs 2.6-5.0% -- branch MPM with
// identical positions and velocities but the anchor-reconstructed F and the
// trajectories separate by that much in thirty frames. Fitting cannot remove it.
//
// A coarse MPM has the same state as the reference and accumulates F the same
// way, so what it gives up is resolution rather than a kind of information. To
// fit one, its step has to be differentiable in the particles' rest positions,
// volumes and stiffness.
//
// This is DreamPhysics' p2g2p written to carry a backward. It is deliberately a
// port rather than a rewrite: same quadratic B-spline weights, same APIC
// transfer, same Fixed Corotated Kirchhoff stress, same order of operations, so
// exe/verify_mpmstep.py can hold it against the warp solver and see a
// difference that is only arithmetic order.
//
//   p2g   scatter mass and momentum, and the elastic impulse, onto the grid
//   grid  normalise by mass, add gravity
//   g2p   gather velocity back, form C and the velocity gradient, advance x,
//         accumulate F <- (I + dt grad_v) F
//   sig   Kirchhoff stress from the updated F, for the next p2g
//
// The grid is the awkward part of the backward. Forward, p2g is an atomic
// scatter over 27 cells per particle; its adjoint is a gather, which is the
// cheap direction. g2p is a gather forward, so its adjoint scatters -- and that
// one runs over the grid rather than the particles to stay contention-free.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define FULL_MASK 0xffffffffu

// ---------------- small dense linear algebra --------------------------------

__device__ inline void m3_mul(const float A[9], const float B[9], float C[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
      C[i * 3 + j] = s;
    }
}

__device__ inline void m3_mulT(const float A[9], const float B[9], float C[9]) {
  // A * B^T
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[j * 3 + k];
      C[i * 3 + j] = s;
    }
}

__device__ inline float m3_det(const float F[9]) {
  return F[0] * (F[4] * F[8] - F[5] * F[7])
       - F[1] * (F[3] * F[8] - F[5] * F[6])
       + F[2] * (F[3] * F[7] - F[4] * F[6]);
}

__device__ inline float m3_fro(const float F[9]) {
  float s = 0.f;
  for (int i = 0; i < 9; ++i) s += F[i] * F[i];
  return sqrtf(s);
}

__device__ inline void m3_inv(const float F[9], float out[9]) {
  float d = m3_det(F);
  float id = (fabsf(d) > 1e-30f) ? 1.0f / d : 0.0f;
  out[0] = (F[4] * F[8] - F[5] * F[7]) * id;
  out[1] = (F[2] * F[7] - F[1] * F[8]) * id;
  out[2] = (F[1] * F[5] - F[2] * F[4]) * id;
  out[3] = (F[5] * F[6] - F[3] * F[8]) * id;
  out[4] = (F[0] * F[8] - F[2] * F[6]) * id;
  out[5] = (F[2] * F[3] - F[0] * F[5]) * id;
  out[6] = (F[3] * F[7] - F[4] * F[6]) * id;
  out[7] = (F[1] * F[6] - F[0] * F[7]) * id;
  out[8] = (F[0] * F[4] - F[1] * F[3]) * id;
}

// The rotation from F = R S. The reference takes it from an SVD; this uses the
// same scaled Newton iteration the rest of this project does, which agrees to
// float precision and, unlike a 3x3 SVD, has a derivative that is a few matrix
// products rather than a branch on degenerate singular values.
__device__ void polar_R(const float F[9], int iters, float ridge, float R[9]) {
  float n = fmaxf(m3_fro(F), 1e-12f);
  for (int i = 0; i < 9; ++i) R[i] = F[i];
  R[0] += ridge * n; R[4] += ridge * n; R[8] += ridge * n;
  for (int it = 0; it < iters; ++it) {
    float Ri[9];
    m3_inv(R, Ri);
    float g = powf(fmaxf(fabsf(m3_det(R)), 1e-12f), -1.0f / 3.0f);
    float ig = 1.0f / g;
    float nx[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        nx[i * 3 + j] = 0.5f * (g * R[i * 3 + j] + ig * Ri[j * 3 + i]);
    for (int i = 0; i < 9; ++i) R[i] = nx[i];
  }
}

// quadratic B-spline weights and their derivatives, exactly as the reference
__device__ inline void bspline(const float gp[3], int base[3], float w[3][3],
                                float dw[3][3]) {
  for (int d = 0; d < 3; ++d) {
    base[d] = (int)(gp[d] - 0.5f);
    float fx = gp[d] - (float)base[d];
    float a = 1.5f - fx, b = fx - 1.0f, c = fx - 0.5f;
    w[d][0] = 0.5f * a * a;
    w[d][1] = 0.75f - b * b;
    w[d][2] = 0.5f * c * c;
    dw[d][0] = fx - 1.5f;
    dw[d][1] = -2.0f * (fx - 1.0f);
    dw[d][2] = fx - 0.5f;
  }
}

// ---------------- forward ---------------------------------------------------

__global__ void p2g_kernel(
    const float* __restrict__ x, const float* __restrict__ v,
    const float* __restrict__ C, const float* __restrict__ stress,
    const float* __restrict__ vol, const float* __restrict__ mass,
    float* __restrict__ grid_v, float* __restrict__ grid_m,
    int N, int G, float dx, float inv_dx, float dt) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float gp[3] = {x[p * 3] * inv_dx, x[p * 3 + 1] * inv_dx, x[p * 3 + 2] * inv_dx};
  int base[3]; float w[3][3], dw[3][3];
  bspline(gp, base, w, dw);
  float fx[3];
  for (int d = 0; d < 3; ++d) fx[d] = gp[d] - (float)base[d];
  float m = mass[p], vl = vol[p];

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        int ix = base[0] + i, iy = base[1] + j, iz = base[2] + k;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= G || iy >= G || iz >= G) continue;
        float wt = w[0][i] * w[1][j] * w[2][k];
        float dwt[3] = {dw[0][i] * w[1][j] * w[2][k] * inv_dx,
                        w[0][i] * dw[1][j] * w[2][k] * inv_dx,
                        w[0][i] * w[1][j] * dw[2][k] * inv_dx};
        float dpos[3] = {((float)i - fx[0]) * dx, ((float)j - fx[1]) * dx,
                         ((float)k - fx[2]) * dx};
        long long g = (((long long)ix * G) + iy) * G + iz;
        for (int a = 0; a < 3; ++a) {
          float Cd = 0.f;
          for (int b = 0; b < 3; ++b) Cd += C[p * 9 + a * 3 + b] * dpos[b];
          float ef = 0.f;
          for (int b = 0; b < 3; ++b) ef += stress[p * 9 + a * 3 + b] * dwt[b];
          atomicAdd(&grid_v[g * 3 + a], wt * m * (v[p * 3 + a] + Cd) - dt * vl * ef);
        }
        atomicAdd(&grid_m[g], wt * m);
      }
}

__global__ void grid_kernel(float* __restrict__ grid_v,
                             const float* __restrict__ grid_m,
                             const float* __restrict__ grav, long long n, float dt) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  float m = grid_m[g];
  if (m > 1e-15f) {
    float inv = 1.0f / m;
    for (int a = 0; a < 3; ++a) grid_v[g * 3 + a] = grid_v[g * 3 + a] * inv + dt * grav[a];
  } else {
    for (int a = 0; a < 3; ++a) grid_v[g * 3 + a] = 0.f;
  }
}

__global__ void g2p_kernel(
    const float* __restrict__ x, const float* __restrict__ F,
    const float* __restrict__ grid_v, const unsigned char* __restrict__ fixed,
    float* __restrict__ x_out, float* __restrict__ v_out,
    float* __restrict__ C_out, float* __restrict__ F_out,
    int N, int G, float inv_dx, float dt) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float gp[3] = {x[p * 3] * inv_dx, x[p * 3 + 1] * inv_dx, x[p * 3 + 2] * inv_dx};
  int base[3]; float w[3][3], dw[3][3];
  bspline(gp, base, w, dw);
  float fx[3];
  for (int d = 0; d < 3; ++d) fx[d] = gp[d] - (float)base[d];

  float nv[3] = {0.f, 0.f, 0.f}, nC[9] = {0.f}, nF[9] = {0.f};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        int ix = base[0] + i, iy = base[1] + j, iz = base[2] + k;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= G || iy >= G || iz >= G) continue;
        float wt = w[0][i] * w[1][j] * w[2][k];
        float dwt[3] = {dw[0][i] * w[1][j] * w[2][k] * inv_dx,
                        w[0][i] * dw[1][j] * w[2][k] * inv_dx,
                        w[0][i] * w[1][j] * dw[2][k] * inv_dx};
        float dpos[3] = {(float)i - fx[0], (float)j - fx[1], (float)k - fx[2]};
        long long g = (((long long)ix * G) + iy) * G + iz;
        for (int a = 0; a < 3; ++a) {
          float gv = grid_v[g * 3 + a];
          nv[a] += gv * wt;
          for (int b = 0; b < 3; ++b) {
            nC[a * 3 + b] += gv * dpos[b] * (wt * inv_dx * 4.0f);
            nF[a * 3 + b] += gv * dwt[b];
          }
        }
      }
  bool fx_ = fixed && fixed[p];
  for (int a = 0; a < 3; ++a) {
    float vv = fx_ ? 0.f : nv[a];
    v_out[p * 3 + a] = vv;
    x_out[p * 3 + a] = x[p * 3 + a] + dt * vv;
  }
  for (int i = 0; i < 9; ++i) C_out[p * 9 + i] = fx_ ? 0.f : nC[i];
  // F <- (I + dt grad_v) F
  float A[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i * 3 + j] = (i == j ? 1.0f : 0.0f) + dt * nF[i * 3 + j];
  float Fo[9];
  m3_mul(A, F + p * 9, Fo);
  for (int i = 0; i < 9; ++i) F_out[p * 9 + i] = Fo[i];
}

// Kirchhoff stress, Fixed Corotated:  tau = 2 mu (F - R) F^T + lam J (J - 1) I
__global__ void stress_kernel(
    const float* __restrict__ F, const float* __restrict__ mu,
    const float* __restrict__ lam, float* __restrict__ stress,
    int N, int polar_iters, float ridge) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float Fl[9], R[9];
  for (int i = 0; i < 9; ++i) Fl[i] = F[p * 9 + i];
  polar_R(Fl, polar_iters, ridge, R);
  float J = m3_det(Fl);
  float D[9];
  for (int i = 0; i < 9; ++i) D[i] = Fl[i] - R[i];
  float DFt[9];
  m3_mulT(D, Fl, DFt);
  float m = mu[p], l = lam[p], c = l * J * (J - 1.0f);
  float s[9];
  for (int i = 0; i < 9; ++i) s[i] = 2.0f * m * DFt[i] + ((i % 4 == 0) ? c : 0.f);
  // the reference symmetrises before use
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      stress[p * 9 + i * 3 + j] = 0.5f * (s[i * 3 + j] + s[j * 3 + i]);
}

// ---------------- host ------------------------------------------------------

#define CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")

std::vector<torch::Tensor> substep(
    torch::Tensor x, torch::Tensor v, torch::Tensor C, torch::Tensor F,
    torch::Tensor vol, torch::Tensor mass, torch::Tensor mu, torch::Tensor lam,
    torch::Tensor grav, torch::Tensor fixed,
    int64_t G, double dx, double dt, int64_t polar_iters, double ridge) {
  CHECK(x); CHECK(v); CHECK(C); CHECK(F); CHECK(vol); CHECK(mass);
  CHECK(mu); CHECK(lam); CHECK(grav);
  const int N = x.size(0);
  auto o = x.options();
  auto stress = torch::empty({N, 3, 3}, o);
  const int T = 256;
  stress_kernel<<<(N + T - 1) / T, T>>>(
      F.data_ptr<float>(), mu.data_ptr<float>(), lam.data_ptr<float>(),
      stress.data_ptr<float>(), N, (int)polar_iters, (float)ridge);

  const long long n = (long long)G * G * G;
  auto grid_v = torch::zeros({(int64_t)n, 3}, o);
  auto grid_m = torch::zeros({(int64_t)n}, o);
  p2g_kernel<<<(N + T - 1) / T, T>>>(
      x.data_ptr<float>(), v.data_ptr<float>(), C.data_ptr<float>(),
      stress.data_ptr<float>(), vol.data_ptr<float>(), mass.data_ptr<float>(),
      grid_v.data_ptr<float>(), grid_m.data_ptr<float>(),
      N, (int)G, (float)dx, (float)(1.0 / dx), (float)dt);

  const int TG = 256;
  grid_kernel<<<(int)((n + TG - 1) / TG), TG>>>(
      grid_v.data_ptr<float>(), grid_m.data_ptr<float>(),
      grav.data_ptr<float>(), n, (float)dt);

  auto x2 = torch::empty_like(x), v2 = torch::empty_like(v);
  auto C2 = torch::empty_like(C), F2 = torch::empty_like(F);
  g2p_kernel<<<(N + T - 1) / T, T>>>(
      x.data_ptr<float>(), F.data_ptr<float>(), grid_v.data_ptr<float>(),
      fixed.numel() ? fixed.data_ptr<unsigned char>() : nullptr,
      x2.data_ptr<float>(), v2.data_ptr<float>(), C2.data_ptr<float>(),
      F2.data_ptr<float>(), N, (int)G, (float)(1.0 / dx), (float)dt);
  return {x2, v2, C2, F2, stress, grid_v, grid_m};
}


// ---------------- backward ---------------------------------------------------
//
// Four adjoints, in the reverse of the order above. Every one of them is the
// transpose of a linear map the forward already builds, except the stress, whose
// nonlinearity is the polar factor.
//
// dR/dF for the polar decomposition. Writing F = R S with S symmetric, a
// perturbation dF gives dR = R W where W is skew, and W solves
//
//     (tr(S) I - S) w = 2 * axl(skew(R^T dF))
//
// with w the axial vector of W. That is a 3x3 solve, and it is the whole of the
// nonlinearity: everything else in the stress is products. S = R^T F.
__device__ void polar_adjoint(const float F[9], const float R[9],
                               const float gR[9], float gF[9]) {
  // S = R^T F
  float S[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += R[k * 3 + i] * F[k * 3 + j];
      S[i * 3 + j] = s;
    }
  float trS = S[0] + S[4] + S[8];
  // A = tr(S) I - S, the operator acting on the axial vector
  float A[9];
  for (int i = 0; i < 9; ++i) A[i] = -S[i];
  A[0] += trS; A[4] += trS; A[8] += trS;

  // the adjoint of R = R(F) applied to gR: first pull gR through R, giving a
  // matrix whose skew part drives the solve
  float RtG[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += R[k * 3 + i] * gR[k * 3 + j];
      RtG[i * 3 + j] = s;
    }
  // b = axl(skew(R^T gR)) -- the same map the forward's W came from, transposed
  float b[3] = {0.5f * (RtG[7] - RtG[5]),
                0.5f * (RtG[2] - RtG[6]),
                0.5f * (RtG[3] - RtG[1])};
  // solve A^T y = b. A is symmetric here (S is), so A^T = A
  float Ai[9];
  m3_inv(A, Ai);
  float y[3];
  for (int i = 0; i < 3; ++i) {
    float s = 0.f;
    for (int k = 0; k < 3; ++k) s += Ai[i * 3 + k] * b[k];
    y[i] = s;
  }
  // W = 2 * skew(y), and dF enters through skew(R^T dF), so
  //   gF = R * (2 * skew(y))^T mapped back: gF = R * Wt, Wt = -2 skew(y)
  float Wt[9] = {0.f, y[2], -y[1],
                 -y[2], 0.f, y[0],
                 y[1], -y[0], 0.f};
  for (int i = 0; i < 9; ++i) Wt[i] *= -1.0f;
  float out[9];
  m3_mul(R, Wt, out);
  for (int i = 0; i < 9; ++i) gF[i] = out[i];
}

// tau = 2 mu sym((F - R) F^T) + lam J (J - 1) I
__global__ void stress_bwd_kernel(
    const float* __restrict__ F, const float* __restrict__ mu,
    const float* __restrict__ lam, const float* __restrict__ gs,
    float* __restrict__ gF, float* __restrict__ gmu, float* __restrict__ glam,
    int N, int polar_iters, float ridge) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float Fl[9], R[9];
  for (int i = 0; i < 9; ++i) Fl[i] = F[p * 9 + i];
  polar_R(Fl, polar_iters, ridge, R);
  float J = m3_det(Fl);
  float m = mu[p], l = lam[p];

  // the forward symmetrised, so pull that back first
  float g[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      g[i * 3 + j] = 0.5f * (gs[p * 9 + i * 3 + j] + gs[p * 9 + j * 3 + i]);

  // d/dF of 2 mu (F - R) F^T, holding R fixed: 2 mu (g F + g^T (F - R))
  float gF_l[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float a = 0.f, b = 0.f;
      for (int k = 0; k < 3; ++k) {
        a += g[i * 3 + k] * Fl[j * 3 + k];        // g * F, contracted on F's cols
        b += g[k * 3 + i] * (Fl[k * 3 + j] - R[k * 3 + j]);
      }
      gF_l[i * 3 + j] = 2.0f * m * (a + b);
    }
  // and through R: d/dR of 2 mu (F - R) F^T is -2 mu g F, then dR/dF
  float gR[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float a = 0.f;
      for (int k = 0; k < 3; ++k) a += g[i * 3 + k] * Fl[j * 3 + k];
      gR[i * 3 + j] = -2.0f * m * a;
    }
  float gFr[9];
  polar_adjoint(Fl, R, gR, gFr);

  // the volumetric term: lam J (J-1) I, and dJ/dF = J F^-T
  float trg = g[0] + g[4] + g[8];
  float c = l * (2.0f * J - 1.0f) * trg;
  float Fi[9];
  m3_inv(Fl, Fi);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      gF_l[i * 3 + j] += gFr[i * 3 + j] + c * J * Fi[j * 3 + i];

  for (int i = 0; i < 9; ++i) gF[p * 9 + i] = gF_l[i];

  // the moduli are fitted too
  float dm = 0.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float a = 0.f;
      for (int k = 0; k < 3; ++k) a += (Fl[i * 3 + k] - R[i * 3 + k]) * Fl[j * 3 + k];
      dm += g[i * 3 + j] * 2.0f * a;
    }
  gmu[p] = dm;
  glam[p] = trg * J * (J - 1.0f);
}

// g2p forward gathers from the grid, so its adjoint scatters onto it. Run per
// particle with atomics: the grid has a cell per particle-neighbourhood rather
// than a few hundred slots, so contention is nothing like the anchor case.
__global__ void g2p_bwd_kernel(
    const float* __restrict__ x, const float* __restrict__ F,
    const float* __restrict__ grid_v, const unsigned char* __restrict__ fixed,
    const float* __restrict__ gx2, const float* __restrict__ gv2,
    const float* __restrict__ gC2, const float* __restrict__ gF2,
    float* __restrict__ g_grid_v, float* __restrict__ gx, float* __restrict__ gF_in,
    int N, int G, float inv_dx, float dt) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float gp[3] = {x[p * 3] * inv_dx, x[p * 3 + 1] * inv_dx, x[p * 3 + 2] * inv_dx};
  int base[3]; float w[3][3], dw[3][3];
  bspline(gp, base, w, dw);
  float fxv[3];
  for (int d = 0; d < 3; ++d) fxv[d] = gp[d] - (float)base[d];
  bool fx_ = fixed && fixed[p];

  // F_out = (I + dt nF) F_in, so gF_in = (I + dt nF)^T gF2 and
  // g_nF = dt * gF2 F_in^T
  float nF[9] = {0.f};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        int ix = base[0] + i, iy = base[1] + j, iz = base[2] + k;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= G || iy >= G || iz >= G) continue;
        float dwt[3] = {dw[0][i] * w[1][j] * w[2][k] * inv_dx,
                        w[0][i] * dw[1][j] * w[2][k] * inv_dx,
                        w[0][i] * w[1][j] * dw[2][k] * inv_dx};
        long long g = (((long long)ix * G) + iy) * G + iz;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            nF[a * 3 + b] += grid_v[g * 3 + a] * dwt[b];
      }
  float A[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i * 3 + j] = (i == j ? 1.0f : 0.0f) + dt * nF[i * 3 + j];
  float gF2l[9], Fin[9];
  for (int i = 0; i < 9; ++i) { gF2l[i] = gF2[p * 9 + i]; Fin[i] = F[p * 9 + i]; }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += A[k * 3 + i] * gF2l[k * 3 + j];
      gF_in[p * 9 + i * 3 + j] = s;
    }
  float gnF[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += gF2l[i * 3 + k] * Fin[j * 3 + k];
      gnF[i * 3 + j] = dt * s;
    }

  // v_out and x_out = x + dt v_out share the same source
  float gv[3];
  for (int a = 0; a < 3; ++a)
    gv[a] = fx_ ? 0.f : (gv2[p * 3 + a] + dt * gx2[p * 3 + a]);
  // x also appears directly in x_out
  for (int a = 0; a < 3; ++a) atomicAdd(&gx[p * 3 + a], gx2[p * 3 + a]);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        int ix = base[0] + i, iy = base[1] + j, iz = base[2] + k;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= G || iy >= G || iz >= G) continue;
        float wt = w[0][i] * w[1][j] * w[2][k];
        float dwt[3] = {dw[0][i] * w[1][j] * w[2][k] * inv_dx,
                        w[0][i] * dw[1][j] * w[2][k] * inv_dx,
                        w[0][i] * w[1][j] * dw[2][k] * inv_dx};
        float dpos[3] = {(float)i - fxv[0], (float)j - fxv[1], (float)k - fxv[2]};
        long long g = (((long long)ix * G) + iy) * G + iz;
        for (int a = 0; a < 3; ++a) {
          float acc = gv[a] * wt;
          if (!fx_)
            for (int b = 0; b < 3; ++b)
              acc += gC2[p * 9 + a * 3 + b] * dpos[b] * (wt * inv_dx * 4.0f);
          for (int b = 0; b < 3; ++b) acc += gnF[a * 3 + b] * dwt[b];
          atomicAdd(&g_grid_v[g * 3 + a], acc);
        }
      }
}

__global__ void grid_bwd_kernel(
    const float* __restrict__ grid_v_in, const float* __restrict__ grid_m,
    const float* __restrict__ g_out, float* __restrict__ g_in, long long n) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  float m = grid_m[g];
  float inv = (m > 1e-15f) ? 1.0f / m : 0.0f;
  for (int a = 0; a < 3; ++a) g_in[g * 3 + a] = g_out[g * 3 + a] * inv;
}

// p2g forward scatters, so its adjoint gathers -- the cheap direction, and no
// atomics on the particle side at all
__global__ void p2g_bwd_kernel(
    const float* __restrict__ x, const float* __restrict__ v,
    const float* __restrict__ C, const float* __restrict__ stress,
    const float* __restrict__ vol, const float* __restrict__ mass,
    const float* __restrict__ g_grid, float* __restrict__ gv,
    float* __restrict__ gC, float* __restrict__ gstress,
    float* __restrict__ gvol, float* __restrict__ gx,
    int N, int G, float dx, float inv_dx, float dt) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  float gp[3] = {x[p * 3] * inv_dx, x[p * 3 + 1] * inv_dx, x[p * 3 + 2] * inv_dx};
  int base[3]; float w[3][3], dw[3][3];
  bspline(gp, base, w, dw);
  float fxv[3];
  for (int d = 0; d < 3; ++d) fxv[d] = gp[d] - (float)base[d];
  float m = mass[p], vl = vol[p];

  float av[3] = {0.f, 0.f, 0.f}, aC[9] = {0.f}, as[9] = {0.f}, avol = 0.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        int ix = base[0] + i, iy = base[1] + j, iz = base[2] + k;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= G || iy >= G || iz >= G) continue;
        float wt = w[0][i] * w[1][j] * w[2][k];
        float dwt[3] = {dw[0][i] * w[1][j] * w[2][k] * inv_dx,
                        w[0][i] * dw[1][j] * w[2][k] * inv_dx,
                        w[0][i] * w[1][j] * dw[2][k] * inv_dx};
        float dpos[3] = {((float)i - fxv[0]) * dx, ((float)j - fxv[1]) * dx,
                         ((float)k - fxv[2]) * dx};
        long long g = (((long long)ix * G) + iy) * G + iz;
        for (int a = 0; a < 3; ++a) {
          float gg = g_grid[g * 3 + a];
          av[a] += gg * wt * m;
          for (int b = 0; b < 3; ++b) {
            aC[a * 3 + b] += gg * wt * m * dpos[b];
            as[a * 3 + b] += -gg * dt * vl * dwt[b];
            avol += -gg * dt * stress[p * 9 + a * 3 + b] * dwt[b];
          }
        }
      }
  for (int a = 0; a < 3; ++a) gv[p * 3 + a] = av[a];
  for (int i = 0; i < 9; ++i) { gC[p * 9 + i] = aC[i]; gstress[p * 9 + i] = as[i]; }
  gvol[p] = avol;
  // x's effect through the weights is left out deliberately: the fit moves the
  // particles' REST positions, and a step's dependence on where the weights land
  // is the stiff, high-frequency part of that derivative. See __init__.py.
}

std::vector<torch::Tensor> substep_backward(
    torch::Tensor gx2, torch::Tensor gv2, torch::Tensor gC2, torch::Tensor gF2,
    torch::Tensor x, torch::Tensor v, torch::Tensor C, torch::Tensor F,
    torch::Tensor stress, torch::Tensor grid_v, torch::Tensor grid_m,
    torch::Tensor vol, torch::Tensor mass, torch::Tensor mu, torch::Tensor lam,
    torch::Tensor fixed, int64_t G, double dx, double dt,
    int64_t polar_iters, double ridge) {
  const int N = x.size(0);
  auto o = x.options();
  const long long n = (long long)G * G * G;
  auto g_grid_out = torch::zeros({(int64_t)n, 3}, o);
  auto gx = torch::zeros_like(x);
  auto gF_in = torch::zeros_like(F);
  const int T = 256;
  g2p_bwd_kernel<<<(N + T - 1) / T, T>>>(
      x.data_ptr<float>(), F.data_ptr<float>(), grid_v.data_ptr<float>(),
      fixed.numel() ? fixed.data_ptr<unsigned char>() : nullptr,
      gx2.data_ptr<float>(), gv2.data_ptr<float>(), gC2.data_ptr<float>(),
      gF2.data_ptr<float>(), g_grid_out.data_ptr<float>(),
      gx.data_ptr<float>(), gF_in.data_ptr<float>(), N, (int)G,
      (float)(1.0 / dx), (float)dt);

  auto g_grid_in = torch::empty({(int64_t)n, 3}, o);
  const int TG = 256;
  grid_bwd_kernel<<<(int)((n + TG - 1) / TG), TG>>>(
      grid_v.data_ptr<float>(), grid_m.data_ptr<float>(),
      g_grid_out.data_ptr<float>(), g_grid_in.data_ptr<float>(), n);

  auto gv = torch::zeros_like(v), gC = torch::zeros_like(C);
  auto gstress = torch::zeros_like(stress), gvol = torch::zeros_like(vol);
  p2g_bwd_kernel<<<(N + T - 1) / T, T>>>(
      x.data_ptr<float>(), v.data_ptr<float>(), C.data_ptr<float>(),
      stress.data_ptr<float>(), vol.data_ptr<float>(), mass.data_ptr<float>(),
      g_grid_in.data_ptr<float>(), gv.data_ptr<float>(), gC.data_ptr<float>(),
      gstress.data_ptr<float>(), gvol.data_ptr<float>(), gx.data_ptr<float>(),
      N, (int)G, (float)dx, (float)(1.0 / dx), (float)dt);

  auto gmu = torch::zeros_like(mu), glam = torch::zeros_like(lam);
  auto gF_s = torch::zeros_like(F);
  stress_bwd_kernel<<<(N + T - 1) / T, T>>>(
      F.data_ptr<float>(), mu.data_ptr<float>(), lam.data_ptr<float>(),
      gstress.data_ptr<float>(), gF_s.data_ptr<float>(), gmu.data_ptr<float>(),
      glam.data_ptr<float>(), N, (int)polar_iters, (float)ridge);
  gF_in = gF_in + gF_s;
  return {gx, gv, gC, gF_in, gvol, gmu, glam};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("substep", &substep, "One differentiable MPM substep, forward (CUDA)");
  m.def("substep_backward", &substep_backward, "Its adjoint (CUDA)");
}
