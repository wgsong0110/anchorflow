// Fused per-substep hot path for the SPARSE anchor discretisation
// (lib/anchorflow/anchor_sparse.py), which lib/anchorstep cannot serve: that
// kernel assumes a fixed K neighbours per Gaussian, isotropic RBF weights and
// one material stiffness, and the fitted set has none of those -- anchors carry
// an orientation and three extents, membership is whatever falls inside
// G(x) > c, and every anchor has its own stiffness multiplier.
//
// What the torch path spends its time on is not the arithmetic but the
// materialisation. Both the rest scatter and the deformation scatter build a
// [P,3,3] tensor of outer products and hand it to index_add_; at P in the
// millions that is tens of megabytes written and read back per substep, for a
// result that is only [N,3,3]. Here the outer product is accumulated in
// registers and never reaches memory.
//
// The pair list arrives sorted by Gaussian -- anchor_sparse.refresh() ends with
// argsort(g * M + a) -- so it is already CSR by row, and row offsets are the
// only thing that has to be built. The scatter back onto anchors gets its own
// anchor-major CSR rather than atomics: there are a few hundred anchors and
// millions of pairs, so atomicAdd would serialise on M*3 addresses, which is
// the same contention lib/anchorstep documents (0.383 ms at M=512 against
// 0.160 ms at M=4096 for identical work).
//
// FORWARD ONLY, and deliberately: nothing differentiates through it. The fit
// needs gradients and keeps the torch path; this is what a trained simulator
// runs, where the parameters are fixed and only the anchor positions move.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define FULL_MASK 0xffffffffu

// ---------------- device: small dense linear algebra -------------------------

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

// inverse via the adjugate. Returns the determinant so the caller can see a
// singular matrix rather than silently dividing by zero
__device__ inline float mat3_inv(const float F[9], float out[9]) {
  float d = mat3_det(F);
  float inv_d = (fabsf(d) > 1e-30f) ? 1.0f / d : 0.0f;
  out[0] = (F[4] * F[8] - F[5] * F[7]) * inv_d;
  out[1] = (F[2] * F[7] - F[1] * F[8]) * inv_d;
  out[2] = (F[1] * F[5] - F[2] * F[4]) * inv_d;
  out[3] = (F[5] * F[6] - F[3] * F[8]) * inv_d;
  out[4] = (F[0] * F[8] - F[2] * F[6]) * inv_d;
  out[5] = (F[2] * F[3] - F[0] * F[5]) * inv_d;
  out[6] = (F[3] * F[7] - F[4] * F[6]) * inv_d;
  out[7] = (F[1] * F[6] - F[0] * F[7]) * inv_d;
  out[8] = (F[0] * F[4] - F[1] * F[3]) * inv_d;
  return d;
}

__device__ inline float mat3_fro(const float F[9]) {
  float s = 0.f;
  for (int i = 0; i < 9; ++i) s += F[i] * F[i];
  return sqrtf(s);
}

// symmetric 3x3 eigh, in ascending order. Copy of lib/eigen3x3's verified
// solver (scale-normalised Smith's method with a Gram-Schmidt fallback for
// repeated eigenvalues); kept in sync with anchorstep_cuda.cu's copy. Only the
// inverted-element correction below needs it, which is a rare branch.
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

// The nearest rotation to F, by the same scaled Newton iteration the torch path
// uses (anchor_fit.polar_R) rather than the closed form via eigh(F^T F): the
// two agree, but keeping the iteration means the kernel and the reference
// differ only in arithmetic order.
//
//   R <- (gamma R + gamma^-1 R^-T) / 2,  gamma = |det R|^(-1/3)
//
// and the ridge in the direction of the identity is what keeps a singular F
// from producing infinities.
__device__ void closest_rotation(const float F[9], int iters, float ridge, float R[9]) {
  float n = fmaxf(mat3_fro(F), 1e-12f);
  for (int i = 0; i < 9; ++i) R[i] = F[i];
  R[0] += ridge * n; R[4] += ridge * n; R[8] += ridge * n;
  for (int it = 0; it < iters; ++it) {
    float Rinv[9];
    mat3_inv(R, Rinv);
    float d = fabsf(mat3_det(R));
    float g = powf(fmaxf(d, 1e-12f), -1.0f / 3.0f);
    float inv_g = 1.0f / g;
    float next[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        // Rinv^T[i][j] == Rinv[j][i]
        next[i * 3 + j] = 0.5f * (g * R[i * 3 + j] + inv_g * Rinv[j * 3 + i]);
    for (int i = 0; i < 9; ++i) R[i] = next[i];
  }
  // An element that has inverted has a polar factor that is a reflection, and
  // Fixed Corotated built on a reflection treats the inverted state as an
  // energy minimum and drives it further in -- which is what used to take this
  // simulator to NaN. Negating the smallest-stretch singular direction is the
  // standard repair.
  if (mat3_det(R) < 0.f) {
    float FtF[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        float s = 0.f;
        for (int k = 0; k < 3; ++k) s += F[k * 3 + i] * F[k * 3 + j];
        FtF[i * 3 + j] = s;
      }
    float l[3], V[9];
    sym3x3_eigh(FtF, l, V);
    float u[3] = {V[0], V[3], V[6]};            // column 0: the smallest stretch
    float H[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        H[i * 3 + j] = (i == j ? 1.f : 0.f) - 2.f * u[i] * u[j];
    float RH[9];
    mat3_mul(R, H, RH);
    for (int i = 0; i < 9; ++i) R[i] = RH[i];
  }
}

__device__ inline float warp_allsum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off);
  return v;
}

// ---------------- kernel 1: shape matching, one warp per Gaussian ------------
//
// Two passes over the row. The first needs only the weighted centroid, which
// every lane then needs to form its own outer product, so it is reduced with a
// butterfly rather than to a single lane. Anchor positions are re-read in the
// second pass instead of being held: there are a few hundred anchors and the
// whole array is a few kilobytes, so it sits in L1.
__global__ void deform_kernel(
    const float* __restrict__ p,        // [M,3] anchor positions, the only input that moves
    const int*   __restrict__ row_off,  // [N+1]
    const int*   __restrict__ pair_a,   // [P]
    const float* __restrict__ w,        // [P]  normalised, per pair
    const float* __restrict__ q,        // [P,3] rest offsets from the rest centroid
    const float* __restrict__ Binv,     // [N,9]
    const float* __restrict__ blocked,  // [N,9]
    float* __restrict__ Fout,           // [N,9]
    float* __restrict__ ccout,          // [N,3]
    int N) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_id >= N) return;
  int lo = row_off[warp_id], hi = row_off[warp_id + 1];

  float cx = 0.f, cy = 0.f, cz = 0.f;
  for (int j = lo + lane; j < hi; j += 32) {
    int a = pair_a[j];
    float wj = w[j];
    cx += wj * p[a * 3 + 0];
    cy += wj * p[a * 3 + 1];
    cz += wj * p[a * 3 + 2];
  }
  cx = warp_allsum(cx); cy = warp_allsum(cy); cz = warp_allsum(cz);

  float A[9];
  for (int i = 0; i < 9; ++i) A[i] = 0.f;
  for (int j = lo + lane; j < hi; j += 32) {
    int a = pair_a[j];
    float wj = w[j];
    float pq[3] = {p[a * 3 + 0] - cx, p[a * 3 + 1] - cy, p[a * 3 + 2] - cz};
    float qj[3] = {q[j * 3 + 0], q[j * 3 + 1], q[j * 3 + 2]};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) A[i * 3 + k] += wj * pq[i] * qj[k];
  }
  for (int i = 0; i < 9; ++i) A[i] = warp_allsum(A[i]);

  if (lane == 0) {
    float Bi[9], F[9];
    for (int i = 0; i < 9; ++i) Bi[i] = Binv[warp_id * 9 + i];
    mat3_mul(A, Bi, F);
    for (int i = 0; i < 9; ++i) Fout[warp_id * 9 + i] = F[i] + blocked[warp_id * 9 + i];
    ccout[warp_id * 3 + 0] = cx;
    ccout[warp_id * 3 + 1] = cy;
    ccout[warp_id * 3 + 2] = cz;
  }
}

// ---------------- kernel 2: skinning ----------------------------------------
// x = cc + F (Xc - rc). One thread per Gaussian; nothing here is a reduction.
__global__ void skin_kernel(
    const float* __restrict__ F,        // [N,9]
    const float* __restrict__ cc,       // [N,3]
    const float* __restrict__ Xc,       // [N,3] canonical Gaussian centres
    const float* __restrict__ rc,       // [N,3] rest centroid
    float* __restrict__ out,            // [N,3]
    int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float d[3] = {Xc[n * 3 + 0] - rc[n * 3 + 0],
                Xc[n * 3 + 1] - rc[n * 3 + 1],
                Xc[n * 3 + 2] - rc[n * 3 + 2]};
  for (int i = 0; i < 3; ++i) {
    float s = cc[n * 3 + i];
    for (int k = 0; k < 3; ++k) s += F[n * 9 + i * 3 + k] * d[k];
    out[n * 3 + i] = s;
  }
}

// ---------------- kernel 3: Fixed Corotated stress --------------------------
//
// One thread per Gaussian, not one warp: the polar iteration is serial 3x3 work
// and giving it a warp would leave thirty-one lanes idle for the length of it.
//
//   P = 2 mu (F - R) + lam (J - 1) J F^-T
//
// with mu and lam already carrying the per-anchor stiffness the fit learned
// (blended onto Gaussians on the python side, where the weights are constant).
// The volumetric term holds J F^-T, unbounded as an element approaches zero
// volume, so the inverse is taken with a ridge scaled to F rather than left to
// produce an infinity the integrator then has to survive.
__global__ void stress_kernel(
    const float* __restrict__ F,        // [N,9]
    const float* __restrict__ Binv,     // [N,9]
    const float* __restrict__ mu,       // [N] already times stiffness
    const float* __restrict__ lam,      // [N] already times stiffness
    float* __restrict__ PB,             // [N,9]  P @ Binv
    int N, int polar_iters, float polar_ridge) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float Fl[9];
  for (int i = 0; i < 9; ++i) Fl[i] = F[n * 9 + i];

  float R[9];
  closest_rotation(Fl, polar_iters, polar_ridge, R);

  float J = mat3_det(Fl);
  float nrm = fmaxf(mat3_fro(Fl), 1e-12f);
  float Fr[9];
  for (int i = 0; i < 9; ++i) Fr[i] = Fl[i];
  Fr[0] += 1e-6f * nrm; Fr[4] += 1e-6f * nrm; Fr[8] += 1e-6f * nrm;
  float Finv[9];
  mat3_inv(Fr, Finv);

  float m = mu[n], l = lam[n];
  float coef = l * (J - 1.0f) * J;
  float P[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      // Finv^T[i][j] == Finv[j][i]
      P[i * 3 + j] = 2.0f * m * (Fl[i * 3 + j] - R[i * 3 + j]) + coef * Finv[j * 3 + i];

  float Bi[9], out[9];
  for (int i = 0; i < 9; ++i) Bi[i] = Binv[n * 9 + i];
  mat3_mul(P, Bi, out);
  for (int i = 0; i < 9; ++i) PB[n * 9 + i] = out[i];
}

// ---------------- kernel 4: gather onto anchors -----------------------------
//
// One warp per anchor, walking that anchor's own list of pairs. The obvious
// alternative -- one thread per pair with an atomicAdd -- puts millions of
// updates onto a few hundred addresses and spends its time serialising.
__global__ void gather_kernel(
    const float* __restrict__ PB,       // [N,9]
    const float* __restrict__ q,        // [P,3]
    const float* __restrict__ w,        // [P]
    const float* __restrict__ vol,      // [N]
    const int*   __restrict__ pair_g,   // [P]
    const int*   __restrict__ acsr_off, // [M+1]
    const int*   __restrict__ acsr_pair,// [P] indices into the pair arrays
    float* __restrict__ f,              // [M,3]
    int M) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_id >= M) return;
  int lo = acsr_off[warp_id], hi = acsr_off[warp_id + 1];

  float fx = 0.f, fy = 0.f, fz = 0.f;
  for (int t = lo + lane; t < hi; t += 32) {
    int j = acsr_pair[t];
    int g = pair_g[j];
    float s = -(vol[g] * w[j]);
    float qj[3] = {q[j * 3 + 0], q[j * 3 + 1], q[j * 3 + 2]};
    float r0 = 0.f, r1 = 0.f, r2 = 0.f;
    for (int k = 0; k < 3; ++k) {
      r0 += PB[g * 9 + 0 * 3 + k] * qj[k];
      r1 += PB[g * 9 + 1 * 3 + k] * qj[k];
      r2 += PB[g * 9 + 2 * 3 + k] * qj[k];
    }
    fx += s * r0; fy += s * r1; fz += s * r2;
  }
  fx = warp_allsum(fx); fy = warp_allsum(fy); fz = warp_allsum(fz);
  if (lane == 0) {
    f[warp_id * 3 + 0] = fx;
    f[warp_id * 3 + 1] = fy;
    f[warp_id * 3 + 2] = fz;
  }
}


// ---------------- backward: the pair-level reductions -----------------------
//
// The fit differentiates through all of this, and the forward kernels above are
// no use to it -- so it ran the torch path, where one substep costs 121 ms
// against the kernel's 2.15. Nearly all of that is the [P,3,3] tensor of outer
// products being built, scattered, and then built again in reverse: 3.6M pairs
// times nine floats is 129 MB written and read per direction per substep.
//
// What is fused here is only the pair-level part. The per-Gaussian 3x3 work --
// the polar factor, the determinant, the inverse -- stays in autograd: it is
// 171k tiny problems rather than 3.6M, and its derivatives are the delicate
// ones. The split is clean because the two meet at F and at P@Binv.
//
// Writing A_n = sum_j w_j (p_j - cc_n) q_j^T with cc_n = sum_j w_j p_j and
// S_n = sum_j w_j q_j, and letting gcc' = gcc - gA S:
//
//   dL/dp_j = w_j (gA (q_j - S) + gcc)
//   dL/dw_j = (p_j - cc)^T gA q_j + gcc' . p_j
//   dL/dq_j = w_j gA^T (p_j - cc)
//
// S is a per-Gaussian reduction, so the forward returns it rather than making
// the backward recompute it.

__global__ void deform_fwd_kernel(
    const float* __restrict__ p, const int* __restrict__ row_off,
    const int* __restrict__ pair_a, const float* __restrict__ w,
    const float* __restrict__ q, float* __restrict__ ccout,
    float* __restrict__ Aout, float* __restrict__ Sout, int N) {
  int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (n >= N) return;
  int lo = row_off[n], hi = row_off[n + 1];

  float cx = 0.f, cy = 0.f, cz = 0.f, sx = 0.f, sy = 0.f, sz = 0.f;
  for (int j = lo + lane; j < hi; j += 32) {
    int a = pair_a[j];
    float wj = w[j];
    cx += wj * p[a * 3 + 0]; cy += wj * p[a * 3 + 1]; cz += wj * p[a * 3 + 2];
    sx += wj * q[j * 3 + 0]; sy += wj * q[j * 3 + 1]; sz += wj * q[j * 3 + 2];
  }
  cx = warp_allsum(cx); cy = warp_allsum(cy); cz = warp_allsum(cz);
  sx = warp_allsum(sx); sy = warp_allsum(sy); sz = warp_allsum(sz);

  float A[9];
  for (int i = 0; i < 9; ++i) A[i] = 0.f;
  for (int j = lo + lane; j < hi; j += 32) {
    int a = pair_a[j];
    float wj = w[j];
    float d[3] = {p[a * 3 + 0] - cx, p[a * 3 + 1] - cy, p[a * 3 + 2] - cz};
    float qj[3] = {q[j * 3 + 0], q[j * 3 + 1], q[j * 3 + 2]};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) A[i * 3 + k] += wj * d[i] * qj[k];
  }
  for (int i = 0; i < 9; ++i) A[i] = warp_allsum(A[i]);
  if (lane == 0) {
    for (int i = 0; i < 9; ++i) Aout[n * 9 + i] = A[i];
    ccout[n * 3 + 0] = cx; ccout[n * 3 + 1] = cy; ccout[n * 3 + 2] = cz;
    Sout[n * 3 + 0] = sx; Sout[n * 3 + 1] = sy; Sout[n * 3 + 2] = sz;
  }
}

// per pair: dL/dw and dL/dq need no reduction at all, and dL/dp is a scatter
// onto a few hundred anchors -- so it takes the anchor-major list, as the force
// gather does, rather than atomics
__global__ void deform_bwd_pair_kernel(
    const float* __restrict__ p, const float* __restrict__ w,
    const float* __restrict__ q, const float* __restrict__ cc,
    const float* __restrict__ S, const float* __restrict__ gcc,
    const float* __restrict__ gA, const int* __restrict__ pair_a,
    const int* __restrict__ pair_g, float* __restrict__ gw,
    float* __restrict__ gq, int P) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= P) return;
  int g = pair_g[j], a = pair_a[j];
  float d[3], qj[3], gAn[9], gc[3], Sn[3];
  for (int i = 0; i < 3; ++i) {
    d[i] = p[a * 3 + i] - cc[g * 3 + i];
    qj[i] = q[j * 3 + i];
    gc[i] = gcc[g * 3 + i];
    Sn[i] = S[g * 3 + i];
  }
  for (int i = 0; i < 9; ++i) gAn[i] = gA[g * 9 + i];

  // gcc' = gcc - gA S
  float gcp[3];
  for (int i = 0; i < 3; ++i) {
    float s = 0.f;
    for (int k = 0; k < 3; ++k) s += gAn[i * 3 + k] * Sn[k];
    gcp[i] = gc[i] - s;
  }
  // dL/dw_j = d^T gA q + gcc' . p
  float gAq[3];
  for (int i = 0; i < 3; ++i) {
    float s = 0.f;
    for (int k = 0; k < 3; ++k) s += gAn[i * 3 + k] * qj[k];
    gAq[i] = s;
  }
  float acc = 0.f;
  for (int i = 0; i < 3; ++i) acc += d[i] * gAq[i] + gcp[i] * p[a * 3 + i];
  gw[j] = acc;
  // dL/dq_j = w gA^T d
  float wj = w[j];
  for (int k = 0; k < 3; ++k) {
    float s = 0.f;
    for (int i = 0; i < 3; ++i) s += gAn[i * 3 + k] * d[i];
    gq[j * 3 + k] = wj * s;
  }
}

// dL/dp_j = w_j (gA (q_j - S) + gcc), gathered per anchor
__global__ void deform_bwd_p_kernel(
    const float* __restrict__ w, const float* __restrict__ q,
    const float* __restrict__ S, const float* __restrict__ gcc,
    const float* __restrict__ gA, const int* __restrict__ pair_g,
    const int* __restrict__ acsr_off, const int* __restrict__ acsr_pair,
    float* __restrict__ gp, int M) {
  int a = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (a >= M) return;
  int lo = acsr_off[a], hi = acsr_off[a + 1];
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int t = lo + lane; t < hi; t += 32) {
    int j = acsr_pair[t], g = pair_g[j];
    float wj = w[j];
    float u[3];
    for (int i = 0; i < 3; ++i) u[i] = q[j * 3 + i] - S[g * 3 + i];
    float r[3];
    for (int i = 0; i < 3; ++i) {
      float s = gcc[g * 3 + i];
      for (int k = 0; k < 3; ++k) s += gA[g * 9 + i * 3 + k] * u[k];
      r[i] = wj * s;
    }
    ax += r[0]; ay += r[1]; az += r[2];
  }
  ax = warp_allsum(ax); ay = warp_allsum(ay); az = warp_allsum(az);
  if (lane == 0) { gp[a * 3 + 0] = ax; gp[a * 3 + 1] = ay; gp[a * 3 + 2] = az; }
}

// ---- the force gather, and its backward ------------------------------------
//
//   f_a = sum_{j: a_j = a} s_j (PB_{g_j} q_j),   s_j = -vol_{g_j} w_j
//
//   dL/dPB_g = sum_{j: g_j = g} s_j (gf_{a_j} q_j^T)     (per Gaussian, CSR)
//   dL/dw_j  = -vol_g (gf_{a_j} . (PB_g q_j))
//   dL/dq_j  = s_j PB_g^T gf_{a_j}

__global__ void gather_bwd_pb_kernel(
    const float* __restrict__ gf, const float* __restrict__ q,
    const float* __restrict__ w, const float* __restrict__ vol,
    const int* __restrict__ row_off, const int* __restrict__ pair_a,
    float* __restrict__ gPB, int N) {
  int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (n >= N) return;
  int lo = row_off[n], hi = row_off[n + 1];
  float acc[9];
  for (int i = 0; i < 9; ++i) acc[i] = 0.f;
  float vg = vol[n];
  for (int j = lo + lane; j < hi; j += 32) {
    int a = pair_a[j];
    float s = -(vg * w[j]);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        acc[i * 3 + k] += s * gf[a * 3 + i] * q[j * 3 + k];
  }
  for (int i = 0; i < 9; ++i) acc[i] = warp_allsum(acc[i]);
  if (lane == 0)
    for (int i = 0; i < 9; ++i) gPB[n * 9 + i] = acc[i];
}

__global__ void gather_bwd_pair_kernel(
    const float* __restrict__ gf, const float* __restrict__ PB,
    const float* __restrict__ q, const float* __restrict__ w,
    const float* __restrict__ vol, const int* __restrict__ pair_a,
    const int* __restrict__ pair_g, float* __restrict__ gw,
    float* __restrict__ gq, int P) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= P) return;
  int g = pair_g[j], a = pair_a[j];
  float qj[3], gfa[3];
  for (int i = 0; i < 3; ++i) { qj[i] = q[j * 3 + i]; gfa[i] = gf[a * 3 + i]; }
  float PBq[3];
  for (int i = 0; i < 3; ++i) {
    float s = 0.f;
    for (int k = 0; k < 3; ++k) s += PB[g * 9 + i * 3 + k] * qj[k];
    PBq[i] = s;
  }
  float dot = 0.f;
  for (int i = 0; i < 3; ++i) dot += gfa[i] * PBq[i];
  float vg = vol[g];
  gw[j] = -vg * dot;
  float s = -(vg * w[j]);
  for (int k = 0; k < 3; ++k) {
    float t = 0.f;
    for (int i = 0; i < 3; ++i) t += PB[g * 9 + i * 3 + k] * gfa[i];
    gq[j * 3 + k] = s * t;
  }
}

// ---------------- host ------------------------------------------------------

#define CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")

static void launch_deform(const torch::Tensor& p, const torch::Tensor& row_off,
                           const torch::Tensor& pair_a, const torch::Tensor& w,
                           const torch::Tensor& q, const torch::Tensor& Binv,
                           const torch::Tensor& blocked, torch::Tensor& F,
                           torch::Tensor& cc, int N) {
  const int threads = 128;                 // four warps, four Gaussians
  const int blocks = (N * 32 + threads - 1) / threads;
  deform_kernel<<<blocks, threads>>>(
      p.data_ptr<float>(), row_off.data_ptr<int>(), pair_a.data_ptr<int>(),
      w.data_ptr<float>(), q.data_ptr<float>(), Binv.data_ptr<float>(),
      blocked.data_ptr<float>(), F.data_ptr<float>(), cc.data_ptr<float>(), N);
}

// the student's per-frame cost: anchors in, Gaussian cloud out
std::vector<torch::Tensor> sparse_skin(
    torch::Tensor p, torch::Tensor row_off, torch::Tensor pair_a,
    torch::Tensor w, torch::Tensor q, torch::Tensor Binv, torch::Tensor blocked,
    torch::Tensor Xc, torch::Tensor rc) {
  CHECK(p); CHECK(row_off); CHECK(pair_a); CHECK(w); CHECK(q);
  CHECK(Binv); CHECK(blocked); CHECK(Xc); CHECK(rc);
  const int N = Binv.size(0);
  auto opts = p.options();
  auto F = torch::empty({N, 3, 3}, opts);
  auto cc = torch::empty({N, 3}, opts);
  auto out = torch::empty({N, 3}, opts);
  launch_deform(p, row_off, pair_a, w, q, Binv, blocked, F, cc, N);
  const int threads = 256;
  skin_kernel<<<(N + threads - 1) / threads, threads>>>(
      F.data_ptr<float>(), cc.data_ptr<float>(), Xc.data_ptr<float>(),
      rc.data_ptr<float>(), out.data_ptr<float>(), N);
  return {out, F};
}

// the simulator's per-substep cost: anchors in, anchor forces out
torch::Tensor sparse_force(
    torch::Tensor p, torch::Tensor row_off, torch::Tensor pair_a,
    torch::Tensor pair_g, torch::Tensor w, torch::Tensor q,
    torch::Tensor Binv, torch::Tensor blocked, torch::Tensor vol,
    torch::Tensor mu, torch::Tensor lam,
    torch::Tensor acsr_off, torch::Tensor acsr_pair,
    int64_t M, int64_t polar_iters, double polar_ridge) {
  CHECK(p); CHECK(row_off); CHECK(pair_a); CHECK(pair_g); CHECK(w); CHECK(q);
  CHECK(Binv); CHECK(blocked); CHECK(vol); CHECK(mu); CHECK(lam);
  CHECK(acsr_off); CHECK(acsr_pair);
  const int N = Binv.size(0);
  auto opts = p.options();
  auto F = torch::empty({N, 3, 3}, opts);
  auto cc = torch::empty({N, 3}, opts);
  auto PB = torch::empty({N, 3, 3}, opts);
  auto f = torch::empty({(int64_t)M, 3}, opts);

  launch_deform(p, row_off, pair_a, w, q, Binv, blocked, F, cc, N);
  const int threads = 256;
  stress_kernel<<<(N + threads - 1) / threads, threads>>>(
      F.data_ptr<float>(), Binv.data_ptr<float>(), mu.data_ptr<float>(),
      lam.data_ptr<float>(), PB.data_ptr<float>(), N,
      (int)polar_iters, (float)polar_ridge);
  const int gthreads = 128;
  const int gblocks = ((int)M * 32 + gthreads - 1) / gthreads;
  gather_kernel<<<gblocks, gthreads>>>(
      PB.data_ptr<float>(), q.data_ptr<float>(), w.data_ptr<float>(),
      vol.data_ptr<float>(), pair_g.data_ptr<int>(), acsr_off.data_ptr<int>(),
      acsr_pair.data_ptr<int>(), f.data_ptr<float>(), (int)M);
  return f;
}

// the deformation gradient on its own, for verification against the torch path
std::vector<torch::Tensor> sparse_deform(
    torch::Tensor p, torch::Tensor row_off, torch::Tensor pair_a,
    torch::Tensor w, torch::Tensor q, torch::Tensor Binv, torch::Tensor blocked) {
  CHECK(p); CHECK(row_off); CHECK(pair_a); CHECK(w); CHECK(q);
  CHECK(Binv); CHECK(blocked);
  const int N = Binv.size(0);
  auto opts = p.options();
  auto F = torch::empty({N, 3, 3}, opts);
  auto cc = torch::empty({N, 3}, opts);
  launch_deform(p, row_off, pair_a, w, q, Binv, blocked, F, cc, N);
  return {F, cc};
}


// ---- differentiable pair reductions, for the fit ---------------------------

std::vector<torch::Tensor> deform_fwd(
    torch::Tensor p, torch::Tensor row_off, torch::Tensor pair_a,
    torch::Tensor w, torch::Tensor q, int64_t N) {
  CHECK(p); CHECK(row_off); CHECK(pair_a); CHECK(w); CHECK(q);
  auto opts = p.options();
  auto cc = torch::empty({(int64_t)N, 3}, opts);
  auto A = torch::empty({(int64_t)N, 3, 3}, opts);
  auto S = torch::empty({(int64_t)N, 3}, opts);
  const int threads = 128;
  const int blocks = ((int)N * 32 + threads - 1) / threads;
  deform_fwd_kernel<<<blocks, threads>>>(
      p.data_ptr<float>(), row_off.data_ptr<int>(), pair_a.data_ptr<int>(),
      w.data_ptr<float>(), q.data_ptr<float>(), cc.data_ptr<float>(),
      A.data_ptr<float>(), S.data_ptr<float>(), (int)N);
  return {cc, A, S};
}

std::vector<torch::Tensor> deform_bwd(
    torch::Tensor gcc, torch::Tensor gA, torch::Tensor p, torch::Tensor w,
    torch::Tensor q, torch::Tensor cc, torch::Tensor S,
    torch::Tensor pair_a, torch::Tensor pair_g,
    torch::Tensor acsr_off, torch::Tensor acsr_pair, int64_t M) {
  CHECK(gcc); CHECK(gA); CHECK(p); CHECK(w); CHECK(q); CHECK(cc); CHECK(S);
  const int P = (int)w.size(0);
  auto opts = p.options();
  auto gw = torch::empty({P}, opts);
  auto gq = torch::empty({P, 3}, opts);
  auto gp = torch::empty({(int64_t)M, 3}, opts);
  const int t1 = 256;
  deform_bwd_pair_kernel<<<(P + t1 - 1) / t1, t1>>>(
      p.data_ptr<float>(), w.data_ptr<float>(), q.data_ptr<float>(),
      cc.data_ptr<float>(), S.data_ptr<float>(), gcc.data_ptr<float>(),
      gA.data_ptr<float>(), pair_a.data_ptr<int>(), pair_g.data_ptr<int>(),
      gw.data_ptr<float>(), gq.data_ptr<float>(), P);
  const int t2 = 128;
  deform_bwd_p_kernel<<<((int)M * 32 + t2 - 1) / t2, t2>>>(
      w.data_ptr<float>(), q.data_ptr<float>(), S.data_ptr<float>(),
      gcc.data_ptr<float>(), gA.data_ptr<float>(), pair_g.data_ptr<int>(),
      acsr_off.data_ptr<int>(), acsr_pair.data_ptr<int>(),
      gp.data_ptr<float>(), (int)M);
  return {gp, gw, gq};
}

torch::Tensor gather_fwd(
    torch::Tensor PB, torch::Tensor q, torch::Tensor w, torch::Tensor vol,
    torch::Tensor pair_g, torch::Tensor acsr_off, torch::Tensor acsr_pair,
    int64_t M) {
  CHECK(PB); CHECK(q); CHECK(w); CHECK(vol);
  auto f = torch::empty({(int64_t)M, 3}, PB.options());
  const int threads = 128;
  gather_kernel<<<((int)M * 32 + threads - 1) / threads, threads>>>(
      PB.data_ptr<float>(), q.data_ptr<float>(), w.data_ptr<float>(),
      vol.data_ptr<float>(), pair_g.data_ptr<int>(), acsr_off.data_ptr<int>(),
      acsr_pair.data_ptr<int>(), f.data_ptr<float>(), (int)M);
  return f;
}

std::vector<torch::Tensor> gather_bwd(
    torch::Tensor gf, torch::Tensor PB, torch::Tensor q, torch::Tensor w,
    torch::Tensor vol, torch::Tensor row_off, torch::Tensor pair_a,
    torch::Tensor pair_g, int64_t N) {
  CHECK(gf); CHECK(PB); CHECK(q); CHECK(w); CHECK(vol);
  const int P = (int)w.size(0);
  auto opts = PB.options();
  auto gPB = torch::empty({(int64_t)N, 3, 3}, opts);
  auto gw = torch::empty({P}, opts);
  auto gq = torch::empty({P, 3}, opts);
  const int t1 = 128;
  gather_bwd_pb_kernel<<<((int)N * 32 + t1 - 1) / t1, t1>>>(
      gf.data_ptr<float>(), q.data_ptr<float>(), w.data_ptr<float>(),
      vol.data_ptr<float>(), row_off.data_ptr<int>(), pair_a.data_ptr<int>(),
      gPB.data_ptr<float>(), (int)N);
  const int t2 = 256;
  gather_bwd_pair_kernel<<<(P + t2 - 1) / t2, t2>>>(
      gf.data_ptr<float>(), PB.data_ptr<float>(), q.data_ptr<float>(),
      w.data_ptr<float>(), vol.data_ptr<float>(), pair_a.data_ptr<int>(),
      pair_g.data_ptr<int>(), gw.data_ptr<float>(), gq.data_ptr<float>(), P);
  return {gPB, gw, gq};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("skin", &sparse_skin, "Fused sparse-anchor skinning (CUDA, forward only)");
  m.def("force", &sparse_force, "Fused sparse-anchor elastic force (CUDA, forward only)");
  m.def("deform", &sparse_deform, "Shape-matching F and centroid (CUDA, forward only)");
  m.def("deform_fwd", &deform_fwd, "Pair reduction to (centroid, A, S) (CUDA)");
  m.def("deform_bwd", &deform_bwd, "Its backward, to (p, w, q) (CUDA)");
  m.def("gather_fwd", &gather_fwd, "Scatter P@Binv onto anchor forces (CUDA)");
  m.def("gather_bwd", &gather_bwd, "Its backward, to (PB, w, q) (CUDA)");
}
