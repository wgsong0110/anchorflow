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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("skin", &sparse_skin, "Fused sparse-anchor skinning (CUDA, forward only)");
  m.def("force", &sparse_force, "Fused sparse-anchor elastic force (CUDA, forward only)");
  m.def("deform", &sparse_deform, "Shape-matching F and centroid (CUDA, forward only)");
}
