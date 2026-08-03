// Fully fused anchor-elastodynamics step (forward energy + analytic backward
// force), replacing the entire per-step torch op chain in
// lib/anchorflow/anchor_mpm.py (_weights -> _shape_match -> polar -> energy ->
// autograd.grad): one thread per Gaussian does RBF weights, shape-matching
// A/B assembly, symmetric-3x3 eigendecomposition (device copy of the verified
// lib/eigen3x3 solver), identity-fallback F construction, closed-form polar
// decomposition, Fixed Corotated energy -- and the backward kernel computes
// the anchor force ANALYTICALLY (no autograd graph at all):
//
//   With per-step-frozen weights (standard discretization choice -- weights
//   come from LAGGED gaussian positions anyway, see anchor_mpm.py's lag
//   convention), F_i = A_i @ Binv_i + C_i is LINEAR in anchor positions, so
//     dE/dp_m = sum_i V_i * w_im * G_i @ (q_im - qbar_i),
//     G_i = P(F_i) @ Binv_i^T,
//     P(F) = 2 mu (F - R) + lam (J - 1) J F^{-T}   (Fixed Corotated PK1)
//   accumulated with atomicAdd. This removes ~30 small kernel launches plus
//   a full autograd construct/backward per substep.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")

#define MAX_K 8

// ---------------- device: symmetric 3x3 eigh (copy of lib/eigen3x3's verified
// solver: scale-normalized Smith's method + robust eigenvectors + repeated-
// eigenvalue Gram-Schmidt fallback). Keep in sync with eigen3x3_cuda.cu. ----
__device__ void sym3x3_eigh(const float B[9], float lo[3], float Vo[9]) {
  float a = B[0], d = B[1], e = B[2];
  float b = B[4], f = B[5];
  float c = B[8];
  float scale = fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c),
                fmaxf(fabsf(d), fmaxf(fabsf(e), fabsf(f)))));
  if (scale < 1e-30f) {
    lo[0] = lo[1] = lo[2] = 0.f;
    for (int i = 0; i < 9; ++i) Vo[i] = 0.f;
    Vo[0] = Vo[4] = Vo[8] = 1.f;
    return;
  }
  float inv_scale = 1.0f / scale;
  a *= inv_scale; b *= inv_scale; c *= inv_scale;
  d *= inv_scale; e *= inv_scale; f *= inv_scale;
  float p1 = d * d + e * e + f * f;
  if (p1 < 1e-12f) {
    float vals[3] = {a, b, c};
    int order[3] = {0, 1, 2};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2 - i; ++j)
        if (vals[order[j]] > vals[order[j + 1]]) { int t = order[j]; order[j] = order[j + 1]; order[j + 1] = t; }
    for (int i = 0; i < 3; ++i) lo[i] = vals[order[i]] * scale;
    for (int i = 0; i < 9; ++i) Vo[i] = 0.f;
    for (int i = 0; i < 3; ++i) Vo[order[i] * 3 + i] = 1.f;
    return;
  }
  float q = (a + b + c) / 3.0f;
  float pa = a - q, pb = b - q, pc = c - q;
  float p2 = pa * pa + pb * pb + pc * pc + 2.0f * p1;
  float p = sqrtf(p2 / 6.0f);
  float inv_p = 1.0f / p;
  float Ba = pa * inv_p, Bb = pb * inv_p, Bc = pc * inv_p;
  float Bd = d * inv_p, Be = e * inv_p, Bf = f * inv_p;
  float detB = Ba * (Bb * Bc - Bf * Bf) - Bd * (Bd * Bc - Bf * Be) + Be * (Bd * Bf - Bb * Be);
  float r = fmaxf(-1.0f, fminf(1.0f, detB / 2.0f));
  float phi = acosf(r) / 3.0f;
  float eig2 = q + 2.0f * p * cosf(phi);
  float eig0 = q + 2.0f * p * cosf(phi + 2.0944f);
  float eig1 = 3.0f * q - eig0 - eig2;
  lo[0] = eig0; lo[1] = eig1; lo[2] = eig2;
  for (int k = 0; k < 3; ++k) {
    float lam = lo[k];
    float rows[3][3] = {{a - lam, d, e}, {d, b - lam, f}, {e, f, c - lam}};
    float best_norm = -1.0f, best[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; ++i) {
      int j = (i + 1) % 3;
      float cx = rows[i][1] * rows[j][2] - rows[i][2] * rows[j][1];
      float cy = rows[i][2] * rows[j][0] - rows[i][0] * rows[j][2];
      float cz = rows[i][0] * rows[j][1] - rows[i][1] * rows[j][0];
      float nrm = cx * cx + cy * cy + cz * cz;
      if (nrm > best_norm) { best_norm = nrm; best[0] = cx; best[1] = cy; best[2] = cz; }
    }
    float inv_n = best_norm > 1e-12f ? rsqrtf(best_norm) : 0.0f;
    Vo[0 * 3 + k] = best[0] * inv_n;
    Vo[1 * 3 + k] = best[1] * inv_n;
    Vo[2 * 3 + k] = best[2] * inv_n;
  }
  for (int k = 0; k < 3; ++k) {
    float vk0 = Vo[k], vk1 = Vo[3 + k], vk2 = Vo[6 + k];
    if (vk0 * vk0 + vk1 * vk1 + vk2 * vk2 > 0.5f) continue;
    float bestv[3] = {0.f, 0.f, 0.f};
    float best_len = -1.f;
    for (int axis = 0; axis < 3; ++axis) {
      float t[3] = {0.f, 0.f, 0.f};
      t[axis] = 1.f;
      for (int j = 0; j < 3; ++j) {
        if (j == k) continue;
        float vj[3] = {Vo[j], Vo[3 + j], Vo[6 + j]};
        if (vj[0] * vj[0] + vj[1] * vj[1] + vj[2] * vj[2] < 0.5f) continue;
        float dot = t[0] * vj[0] + t[1] * vj[1] + t[2] * vj[2];
        t[0] -= dot * vj[0]; t[1] -= dot * vj[1]; t[2] -= dot * vj[2];
      }
      float len2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
      if (len2 > best_len) { best_len = len2; bestv[0] = t[0]; bestv[1] = t[1]; bestv[2] = t[2]; }
    }
    float inv_l = best_len > 1e-12f ? rsqrtf(best_len) : 0.0f;
    Vo[k] = bestv[0] * inv_l;
    Vo[3 + k] = bestv[1] * inv_l;
    Vo[6 + k] = bestv[2] * inv_l;
  }
  for (int i = 0; i < 3; ++i) lo[i] *= scale;
}

__device__ inline void mat3_mul(const float A[9], const float B[9], float C[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
      C[i * 3 + j] = s;
    }
}

__device__ inline float mat3_det(const float F[9]) {
  return F[0] * (F[4] * F[8] - F[5] * F[7])
       - F[1] * (F[3] * F[8] - F[5] * F[6])
       + F[2] * (F[3] * F[7] - F[4] * F[6]);
}

// polar rotation R from F via eigh(F^T F): R = F V diag(1/sqrt(l)) V^T
__device__ void polar_R(const float F[9], float R[9]) {
  float FtF[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += F[k * 3 + i] * F[k * 3 + j];
      FtF[i * 3 + j] = s;
    }
  float l[3], V[9];
  sym3x3_eigh(FtF, l, V);
  float Sinv[9];
  float inv_sqrt[3];
  for (int i = 0; i < 3; ++i) inv_sqrt[i] = rsqrtf(fmaxf(l[i], 1e-8f));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += V[i * 3 + k] * inv_sqrt[k] * V[j * 3 + k];
      Sinv[i * 3 + j] = s;
    }
  mat3_mul(F, Sinv, R);
}

// ---------------- forward: per-Gaussian energy + F + skinned position -------
__global__ void anchorstep_forward_kernel(
    const float* __restrict__ gaussian_canonical,   // [N,3]
    const float* __restrict__ gaussian_pos_prev,    // [N,3] (weight lag)
    const float* __restrict__ anchor_pos,           // [M,3] current
    const float* __restrict__ anchor_rest,          // [M,3] canonical
    const long* __restrict__ nn_idx,                // [N,K]
    const float* __restrict__ volume,               // [N]
    const float* __restrict__ mu_p,   // [N] per-particle Lame mu
    const float* __restrict__ lam_p,  // [N] per-particle Lame lambda
    float radius, int N, int K, float eig_floor_frac,
    int stage,   // profiling only: 1=stop after weights+A/B, 2=+shape-match eigh
                 // (F built), 3=full (polar eigh + energy + G/c). Lets us
                 // attribute cost to the two eigendecompositions instead of
                 // guessing which part dominates.
    float* __restrict__ out_w,        // [N,K]  saved for backward
    float* __restrict__ out_Binv,     // [N,9]  effective inverse map (saved)
    float* __restrict__ out_qbar,     // [N,3]  weighted rest centroid offset (saved)
    float* __restrict__ out_F,        // [N,9]
    float* __restrict__ out_pos,      // [N,3]  skinned gaussian position
    float* __restrict__ out_psi,      // [N]    energy density
    float* __restrict__ out_R,        // [N,9]  polar rotation, saved for backward
    float* __restrict__ out_G,        // [N,9]  G = P(F) @ Binv^T, per-Gaussian
    float* __restrict__ out_c) {      // [N,3]  rest centroid + qbar, per-Gaussian
  // out_G/out_c exist so the gather backward is a pure 9-mult dot product per
  // (gaussian, anchor-slot) pair. A first gather attempt recomputed the whole
  // stress->G chain inside the per-pair loop, doing that work K=8x per Gaussian
  // instead of once -- it removed the atomics but was 4x SLOWER overall.
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  const long* idx = nn_idx + (long)n * K;
  float w[MAX_K];
  float wsum = 0.f;
  float gx = gaussian_pos_prev[n * 3], gy = gaussian_pos_prev[n * 3 + 1], gz = gaussian_pos_prev[n * 3 + 2];
  float inv_2r2 = 1.0f / (2.0f * radius * radius);
  for (int k = 0; k < K; ++k) {
    long a = idx[k];
    float dx = gx - anchor_pos[a * 3], dy = gy - anchor_pos[a * 3 + 1], dz = gz - anchor_pos[a * 3 + 2];
    w[k] = expf(-(dx * dx + dy * dy + dz * dz) * inv_2r2) + 1e-8f;
    wsum += w[k];
  }
  float rc[3] = {0.f, 0.f, 0.f}, cc[3] = {0.f, 0.f, 0.f};
  for (int k = 0; k < K; ++k) {
    w[k] /= wsum;
    long a = idx[k];
    for (int d = 0; d < 3; ++d) {
      rc[d] += w[k] * anchor_rest[a * 3 + d];
      cc[d] += w[k] * anchor_pos[a * 3 + d];
    }
  }
  float A[9] = {0}, B[9] = {0};
  for (int k = 0; k < K; ++k) {
    long a = idx[k];
    float q[3], p[3];
    for (int d = 0; d < 3; ++d) {
      q[d] = anchor_rest[a * 3 + d] - rc[d];
      p[d] = anchor_pos[a * 3 + d] - cc[d];
    }
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        A[i * 3 + j] += w[k] * p[i] * q[j];
        B[i * 3 + j] += w[k] * q[i] * q[j];
      }
  }
  if (stage == 1) {   // profiling: stop before any eigendecomposition
    out_psi[n] = A[0] + B[0];   // touch results so nothing is optimized away
    return;
  }
  float l[3], V[9];
  sym3x3_eigh(B, l, V);
  float lmax = fmaxf(l[2], 1e-12f);
  // Binv_eff = V diag(m_j/l_j) V^T ; C = V diag(1-m_j) V^T  (identity fallback)
  float Binv[9], Cfix[9];
  float m[3], invl[3];
  for (int j = 0; j < 3; ++j) {
    m[j] = (l[j] > eig_floor_frac * lmax) ? 1.f : 0.f;
    invl[j] = m[j] / fmaxf(l[j], 1e-20f);
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float sb = 0.f, sc = 0.f;
      for (int k = 0; k < 3; ++k) {
        sb += V[i * 3 + k] * invl[k] * V[j * 3 + k];
        sc += V[i * 3 + k] * (1.f - m[k]) * V[j * 3 + k];
      }
      Binv[i * 3 + j] = sb;
      Cfix[i * 3 + j] = sc;
    }
  float F[9];
  mat3_mul(A, Binv, F);
  for (int i = 0; i < 9; ++i) F[i] += Cfix[i];

  // skinned position: cc + F @ (X - rc)
  float X0[3] = {gaussian_canonical[n * 3] - rc[0],
                 gaussian_canonical[n * 3 + 1] - rc[1],
                 gaussian_canonical[n * 3 + 2] - rc[2]};
  for (int i = 0; i < 3; ++i)
    out_pos[n * 3 + i] = cc[i] + F[i * 3] * X0[0] + F[i * 3 + 1] * X0[1] + F[i * 3 + 2] * X0[2];

  if (stage == 2) {   // profiling: F built (1 eigh), skip polar eigh + energy
    out_psi[n] = F[0];
    return;
  }
  // Fixed Corotated energy density
  float R[9];
  polar_R(F, R);
  float J = mat3_det(F);
  float frob2 = 0.f;
  for (int i = 0; i < 9; ++i) { float dfr = F[i] - R[i]; frob2 += dfr * dfr; }
  float mu = mu_p[n], lam = lam_p[n];
  out_psi[n] = mu * frob2 + 0.5f * lam * (J - 1.f) * (J - 1.f);
  for (int i = 0; i < 9; ++i) out_R[n * 9 + i] = R[i];

  // per-Gaussian PK1 stress -> G = P @ Binv^T (used by both backward variants)
  float cof[9];
  cof[0]=F[4]*F[8]-F[5]*F[7]; cof[1]=F[5]*F[6]-F[3]*F[8]; cof[2]=F[3]*F[7]-F[4]*F[6];
  cof[3]=F[2]*F[7]-F[1]*F[8]; cof[4]=F[0]*F[8]-F[2]*F[6]; cof[5]=F[1]*F[6]-F[0]*F[7];
  cof[6]=F[1]*F[5]-F[2]*F[4]; cof[7]=F[2]*F[3]-F[0]*F[5]; cof[8]=F[0]*F[4]-F[1]*F[3];
  float coefJ = lam * (J - 1.f);
  float Pk[9];
  for (int i = 0; i < 9; ++i) Pk[i] = 2.f * mu * (F[i] - R[i]) + coefJ * cof[i];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float sg = 0.f;
      for (int k = 0; k < 3; ++k) sg += Pk[i * 3 + k] * Binv[j * 3 + k];
      out_G[n * 9 + i * 3 + j] = sg;
    }

  // saved-for-backward
  for (int k = 0; k < K; ++k) out_w[n * K + k] = w[k];
  for (int i = 0; i < 9; ++i) { out_Binv[n * 9 + i] = Binv[i]; out_F[n * 9 + i] = F[i]; }
  // qbar = weighted rest offset mean = sum_k w_k q_k (needed by backward)
  float qbar[3] = {0.f, 0.f, 0.f};
  for (int k = 0; k < K; ++k) {
    long a = idx[k];
    for (int d = 0; d < 3; ++d) qbar[d] += w[k] * (anchor_rest[a * 3 + d] - rc[d]);
  }
  for (int d = 0; d < 3; ++d) out_qbar[n * 3 + d] = qbar[d];
  // c = rc + qbar: the gather backward needs q_im = anchor_rest[m] - rc - qbar,
  // so fold both per-Gaussian terms into one vector it can subtract directly.
  for (int d = 0; d < 3; ++d) out_c[n * 3 + d] = rc[d] + qbar[d];
}

// ---------------- backward: analytic anchor force -----------------------
// dE/dp_m = sum_i V_i * w_im * G_i @ (q_im - qbar_i),  G_i = P(F_i) @ Binv_i^T
//
// TWO implementations:
//  (a) scatter (below): one thread per Gaussian, atomicAdd into [M,3].
//      Measured to be CONTENTION-bound at small M -- 4.1M atomicAdds landing
//      on M*3 addresses serialize badly: backward took 0.383 ms at M=512 vs
//      0.160 ms at M=4096 for IDENTICAL arithmetic. Since sparse anchors are
//      the whole point of this method, that penalty hits exactly the regime
//      we care about (and is why replacing MPM's grid only bought ~1.8x:
//      MPM's p2g atomics spread over ~1M grid nodes and barely contend).
//  (b) gather (anchorstep_backward_gather_kernel): one BLOCK per anchor,
//      walking a prebuilt anchor->Gaussian CSR (the connectivity is fixed at
//      construction, so the CSR is built once) and block-reducing. Zero
//      atomics, so it is contention-free at any M.
__global__ void anchorstep_backward_kernel(
    const float* __restrict__ anchor_rest,          // [M,3]
    const long* __restrict__ nn_idx,                // [N,K]
    const float* __restrict__ volume,               // [N]
    const float* __restrict__ w,                    // [N,K]
    const float* __restrict__ Binv,                 // [N,9]
    const float* __restrict__ qbar,                 // [N,3]
    const float* __restrict__ F_in,                 // [N,9]
    const float* __restrict__ R_in,                 // [N,9] polar R from forward
    const float* __restrict__ mu_p, const float* __restrict__ lam_p,
    int N, int K,
    float* __restrict__ grad_anchor) {              // [M,3] += dE/dp
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const long* idx = nn_idx + (long)n * K;
  const float* F = F_in + (long)n * 9;

  // P = 2 mu (F - R) + lam (J-1) J F^{-T}
  // R comes from the forward pass (it already ran the same polar decomposition
  // for the energy); recomputing it here was the single biggest cost in the
  // step, since polar_R runs a full symmetric 3x3 eigendecomposition.
  const float* R = R_in + (long)n * 9;
  float J = mat3_det(F);
  // cofactor(F) = J * F^{-T} (works even at small J)
  float cof[9];
  cof[0] = F[4] * F[8] - F[5] * F[7];
  cof[1] = F[5] * F[6] - F[3] * F[8];
  cof[2] = F[3] * F[7] - F[4] * F[6];
  cof[3] = F[2] * F[7] - F[1] * F[8];
  cof[4] = F[0] * F[8] - F[2] * F[6];
  cof[5] = F[1] * F[6] - F[0] * F[7];
  cof[6] = F[1] * F[5] - F[2] * F[4];
  cof[7] = F[2] * F[3] - F[0] * F[5];
  cof[8] = F[0] * F[4] - F[1] * F[3];
  // NOTE cof above is the transpose-of-cofactor layout: cof[i*3+j] = d(det)/dF[i*3+j]
  float P[9];
  float mu = mu_p[n], lam = lam_p[n];
  float coef = lam * (J - 1.f);
  for (int i = 0; i < 9; ++i) P[i] = 2.f * mu * (F[i] - R[i]) + coef * cof[i];

  const float* Bi = Binv + (long)n * 9;
  float G[9];   // G = P @ Binv^T
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += P[i * 3 + k] * Bi[j * 3 + k];
      G[i * 3 + j] = s;
    }
  float Vn = volume[n];
  // rest centroid rc = (weighted mean of rest anchors); q_im = rest_m - rc.
  // qbar (saved) = sum w q = (sum w rest) - rc -> and rc = sum w rest, so
  // qbar == 0 analytically; saved anyway for exactness under fp rounding.
  float rc[3] = {0.f, 0.f, 0.f};
  for (int k = 0; k < K; ++k) {
    long a = idx[k];
    float wk = w[n * K + k];
    for (int d = 0; d < 3; ++d) rc[d] += wk * anchor_rest[a * 3 + d];
  }
  const float* qb = qbar + (long)n * 3;
  for (int k = 0; k < K; ++k) {
    long a = idx[k];
    float wk = w[n * K + k];
    float qm[3];
    for (int d = 0; d < 3; ++d) qm[d] = anchor_rest[a * 3 + d] - rc[d] - qb[d];
    for (int i = 0; i < 3; ++i) {
      float g = Vn * wk * (G[i * 3] * qm[0] + G[i * 3 + 1] * qm[1] + G[i * 3 + 2] * qm[2]);
      atomicAdd(&grad_anchor[a * 3 + i], g);
    }
  }
}


// Contention-free backward: one block per anchor walking a prebuilt
// anchor->(gaussian, slot) CSR. All the expensive per-Gaussian work (stress,
// G = P @ Binv^T, rest centroid) was already done ONCE in the forward kernel
// and handed over via G/c, so each pair here costs only a 3x3 matvec.
__global__ void anchorstep_backward_gather_kernel(
    const float* __restrict__ anchor_rest,   // [M,3]
    const float* __restrict__ volume,        // [N]
    const float* __restrict__ w,             // [N,K]
    const float* __restrict__ G_in,          // [N,9]
    const float* __restrict__ c_in,          // [N,3]
    const int* __restrict__ csr_off,         // [M+1]
    const int* __restrict__ csr_gid,         // [nnz]
    const int* __restrict__ csr_slot,        // [nnz]
    int K, float* __restrict__ grad_anchor) {
  int m = blockIdx.x;
  int start = csr_off[m], end = csr_off[m + 1];
  float ax = anchor_rest[m * 3], ay = anchor_rest[m * 3 + 1], az = anchor_rest[m * 3 + 2];
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
  for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
    int n = csr_gid[e];
    float wk = w[(long)n * K + csr_slot[e]] * volume[n];
    const float* G = G_in + (long)n * 9;
    const float* c = c_in + (long)n * 3;
    float q0 = ax - c[0], q1 = ay - c[1], q2 = az - c[2];
    acc0 += wk * (G[0] * q0 + G[1] * q1 + G[2] * q2);
    acc1 += wk * (G[3] * q0 + G[4] * q1 + G[5] * q2);
    acc2 += wk * (G[6] * q0 + G[7] * q1 + G[8] * q2);
  }
  __shared__ float sm[3 * 32];
  float accs[3] = {acc0, acc1, acc2};
  for (int d = 0; d < 3; ++d) {
    float v = accs[d];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((threadIdx.x & 31) == 0) sm[d * 32 + (threadIdx.x >> 5)] = v;
  }
  __syncthreads();
  if (threadIdx.x < 3) {
    int nwarp = (blockDim.x + 31) / 32;
    float t = 0.f;
    for (int i = 0; i < nwarp; ++i) t += sm[threadIdx.x * 32 + i];
    grad_anchor[m * 3 + threadIdx.x] = t;
  }
}

torch::Tensor anchorstep_backward_gather(
    torch::Tensor anchor_rest, torch::Tensor volume, torch::Tensor w,
    torch::Tensor G, torch::Tensor c, torch::Tensor csr_off,
    torch::Tensor csr_gid, torch::Tensor csr_slot, int64_t K, int64_t M) {
  auto grad_anchor = torch::empty({M, 3}, G.options());
  anchorstep_backward_gather_kernel<<<M, 128>>>(
      anchor_rest.contiguous().data_ptr<float>(),
      volume.contiguous().data_ptr<float>(),
      w.contiguous().data_ptr<float>(),
      G.contiguous().data_ptr<float>(),
      c.contiguous().data_ptr<float>(),
      csr_off.contiguous().data_ptr<int>(),
      csr_gid.contiguous().data_ptr<int>(),
      csr_slot.contiguous().data_ptr<int>(),
      (int)K, grad_anchor.data_ptr<float>());
  return grad_anchor;
}

std::vector<torch::Tensor> anchorstep_forward(
    torch::Tensor gaussian_canonical, torch::Tensor gaussian_pos_prev,
    torch::Tensor anchor_pos, torch::Tensor anchor_rest, torch::Tensor nn_idx,
    torch::Tensor volume, torch::Tensor mu_p, torch::Tensor lam_p,
    double radius, double eig_floor_frac, int64_t stage) {
  CHECK_CUDA(anchor_pos);
  int N = gaussian_canonical.size(0);
  int K = nn_idx.size(1);
  TORCH_CHECK(K <= MAX_K, "K must be <= ", MAX_K);
  auto opt = anchor_pos.options();
  auto out_w = torch::empty({N, K}, opt);
  auto out_Binv = torch::empty({N, 9}, opt);
  auto out_qbar = torch::empty({N, 3}, opt);
  auto out_F = torch::empty({N, 9}, opt);
  auto out_pos = torch::empty({N, 3}, opt);
  auto out_psi = torch::empty({N}, opt);
  auto out_R = torch::empty({N, 9}, opt);
  auto out_G = torch::empty({N, 9}, opt);
  auto out_c = torch::empty({N, 3}, opt);
  int threads = 256, blocks = (N + threads - 1) / threads;
  anchorstep_forward_kernel<<<blocks, threads>>>(
      gaussian_canonical.contiguous().data_ptr<float>(),
      gaussian_pos_prev.contiguous().data_ptr<float>(),
      anchor_pos.contiguous().data_ptr<float>(),
      anchor_rest.contiguous().data_ptr<float>(),
      nn_idx.contiguous().data_ptr<long>(),
      volume.contiguous().data_ptr<float>(),
      mu_p.contiguous().data_ptr<float>(), lam_p.contiguous().data_ptr<float>(),
      (float)radius, N, K, (float)eig_floor_frac, (int)stage,
      out_w.data_ptr<float>(), out_Binv.data_ptr<float>(), out_qbar.data_ptr<float>(),
      out_F.data_ptr<float>(), out_pos.data_ptr<float>(), out_psi.data_ptr<float>(),
      out_R.data_ptr<float>(), out_G.data_ptr<float>(), out_c.data_ptr<float>());
  return {out_w, out_Binv, out_qbar, out_F, out_pos, out_psi, out_R, out_G, out_c};
}

torch::Tensor anchorstep_backward(
    torch::Tensor anchor_rest, torch::Tensor nn_idx, torch::Tensor volume,
    torch::Tensor w, torch::Tensor Binv, torch::Tensor qbar, torch::Tensor F,
    torch::Tensor R, torch::Tensor mu_p, torch::Tensor lam_p, int64_t M) {
  int N = nn_idx.size(0);
  int K = nn_idx.size(1);
  auto grad_anchor = torch::zeros({M, 3}, F.options());
  int threads = 256, blocks = (N + threads - 1) / threads;
  anchorstep_backward_kernel<<<blocks, threads>>>(
      anchor_rest.contiguous().data_ptr<float>(),
      nn_idx.contiguous().data_ptr<long>(),
      volume.contiguous().data_ptr<float>(),
      w.contiguous().data_ptr<float>(),
      Binv.contiguous().data_ptr<float>(),
      qbar.contiguous().data_ptr<float>(),
      F.contiguous().data_ptr<float>(),
      R.contiguous().data_ptr<float>(),
      mu_p.contiguous().data_ptr<float>(), lam_p.contiguous().data_ptr<float>(),
      N, K,
      grad_anchor.data_ptr<float>());
  return grad_anchor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &anchorstep_forward, "Fused anchor elastodynamics forward (CUDA)",
        pybind11::arg("gaussian_canonical"), pybind11::arg("gaussian_pos_prev"),
        pybind11::arg("anchor_pos"), pybind11::arg("anchor_rest"), pybind11::arg("nn_idx"),
        pybind11::arg("volume"), pybind11::arg("mu_p"), pybind11::arg("lam_p"),
        pybind11::arg("radius"), pybind11::arg("eig_floor_frac"), pybind11::arg("stage") = 3);
  m.def("backward", &anchorstep_backward, "Analytic anchor force backward, atomic scatter (CUDA)");
  m.def("backward_gather", &anchorstep_backward_gather, "Analytic anchor force backward, contention-free CSR gather (CUDA)");
}
