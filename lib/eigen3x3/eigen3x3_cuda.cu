// Fused batched closed-form symmetric 3x3 eigendecomposition (forward + backward),
// replacing torch.linalg.eigh for the anchor_mpm shape-matching B-matrix solve
// (lib/anchorflow/anchor_mpm.py) -- eigh is called once per Gaussian per physics
// substep there, and a general LAPACK-backed eigh is not well suited to many tiny
// (3x3) batched matrices (kernel-launch/dispatch overhead dominates the actual
// FLOPs). Closed-form trigonometric eigensolver (Smith's method / Eberly's "A
// Robust Eigensolver for 3x3 Symmetric Matrices") avoids that entirely.
//
// NOT YET NUMERICALLY VERIFIED against torch.linalg.eigh (no local GPU on this
// dev machine) -- run exe/verify_eigen3x3.py on an instance before trusting this
// for anything beyond a forward-only speed check. Do not wire into anchor_mpm.py's
// hot path until that parity check passes.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---- forward: closed-form eigenvalues + eigenvectors for a batch of symmetric 3x3 ----
// B layout per-item: [a, d, e, d, b, f, e, f, c] (row-major 3x3, symmetric)
// eigval output ascending [l0<=l1<=l2]; eigvec columns are the corresponding eigenvectors.
__global__ void eigh3x3_forward_kernel(
    const float* __restrict__ B, float* __restrict__ eigval, float* __restrict__ eigvec, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* Bn = B + n * 9;
  float a = Bn[0], d = Bn[1], e = Bn[2];
  float b = Bn[4], f = Bn[5];
  float c = Bn[8];

  float p1 = d * d + e * e + f * f;
  float* lo = eigval + n * 3;
  float* Vo = eigvec + n * 9;

  if (p1 < 1e-12f) {
    // already diagonal -- sort a,b,c ascending, eigenvectors = permuted identity
    float vals[3] = {a, b, c};
    int order[3] = {0, 1, 2};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2 - i; ++j)
        if (vals[order[j]] > vals[order[j + 1]]) { int t = order[j]; order[j] = order[j + 1]; order[j + 1] = t; }
    for (int i = 0; i < 3; ++i) lo[i] = vals[order[i]];
    for (int i = 0; i < 9; ++i) Vo[i] = 0.f;
    for (int i = 0; i < 3; ++i) Vo[order[i] * 3 + i] = 1.f;  // column i = e_{order[i]}
    return;
  }

  float q = (a + b + c) / 3.0f;
  float pa = a - q, pb = b - q, pc = c - q;
  float p2 = pa * pa + pb * pb + pc * pc + 2.0f * p1;
  float p = sqrtf(p2 / 6.0f);
  float inv_p = 1.0f / p;
  // B' = (A - qI) / p ; r = det(B') / 2
  float Ba = pa * inv_p, Bb = pb * inv_p, Bc = pc * inv_p;
  float Bd = d * inv_p, Be = e * inv_p, Bf = f * inv_p;
  float detB = Ba * (Bb * Bc - Bf * Bf) - Bd * (Bd * Bc - Bf * Be) + Be * (Bd * Bf - Bb * Be);
  float r = detB / 2.0f;
  r = fmaxf(-1.0f, fminf(1.0f, r));
  float phi = acosf(r) / 3.0f;

  // Smith's method (see e.g. Eberly, "A Robust Eigensolver for 3x3 Symmetric
  // Matrices"): the three roots of the depressed characteristic cubic are
  // 2cos(phi), 2cos(phi+2pi/3), 2cos(phi+4pi/3) for phi=acos(r)/3 in
  // [0,pi/3]. Numerically (e.g. phi=pi/6): 2cos(phi)=1.73 (largest),
  // 2cos(phi+2pi/3)=-1.73 (smallest), 2cos(phi+4pi/3)=0 (middle) -- so the
  // SMALLEST root uses +2pi/3, not +4pi/3 (that phase gives the MIDDLE
  // root instead). Bug caught via exe/verify_eigen3x3.py: reconstruction
  // V@diag(L)@V^T was correct (some valid decomposition), but raw
  // eigenvalues vs. torch.linalg.eigh were wrong for ~90% of a random
  // batch with no correlation to near-degenerate eigenvalue gaps --
  // i.e. a systematic formula error, not an ordering/degeneracy edge case.
  float eig2 = q + 2.0f * p * cosf(phi);                    // largest
  float eig0 = q + 2.0f * p * cosf(phi + 2.0944f);          // smallest (phi + 2pi/3)
  float eig1 = 3.0f * q - eig0 - eig2;                      // middle (trace = 3q)

  lo[0] = eig0; lo[1] = eig1; lo[2] = eig2;

  // eigenvectors via (A - lambda I) row cross products, robust row selection
  for (int k = 0; k < 3; ++k) {
    float lam = lo[k];
    float m00 = a - lam, m01 = d, m02 = e;
    float m10 = d, m11 = b - lam, m12 = f;
    float m20 = e, m21 = f, m22 = c - lam;
    // try all 3 row-pairs, keep the cross product with largest norm (most robust)
    float rows[3][3] = {{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}};
    float best_norm = -1.0f, best[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; ++i) {
      int j = (i + 1) % 3;
      float cx = rows[i][1] * rows[j][2] - rows[i][2] * rows[j][1];
      float cy = rows[i][2] * rows[j][0] - rows[i][0] * rows[j][2];
      float cz = rows[i][0] * rows[j][1] - rows[i][1] * rows[j][0];
      float nrm = cx * cx + cy * cy + cz * cz;
      if (nrm > best_norm) { best_norm = nrm; best[0] = cx; best[1] = cy; best[2] = cz; }
    }
    float inv_n = best_norm > 1e-20f ? rsqrtf(best_norm) : 0.0f;
    Vo[0 * 3 + k] = best[0] * inv_n;
    Vo[1 * 3 + k] = best[1] * inv_n;
    Vo[2 * 3 + k] = best[2] * inv_n;
  }
}

// ---- backward: standard symmetric-eigendecomposition VJP ----
// dA = V ( diag(gL) + E ⊙ (V^T gV) )_sym V^T ,  E_ij = 1/(lambda_j - lambda_i), i!=j else 0
__global__ void eigh3x3_backward_kernel(
    const float* __restrict__ grad_eigval, const float* __restrict__ grad_eigvec,
    const float* __restrict__ eigval, const float* __restrict__ eigvec,
    float* __restrict__ grad_B, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* gL = grad_eigval + n * 3;
  const float* gV = grad_eigvec + n * 9;
  const float* L = eigval + n * 3;
  const float* V = eigvec + n * 9;

  // VtgV = V^T @ gV  (3x3)
  float VtgV[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += V[k * 3 + i] * gV[k * 3 + j];
      VtgV[i][j] = s;
    }

  float inner[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        inner[i][j] = gL[i];
      } else {
        float denom = L[j] - L[i];
        float Eij = (fabsf(denom) > 1e-6f) ? (1.0f / denom) : 0.0f;  // guard near-degenerate eigs
        inner[i][j] = Eij * VtgV[i][j];
      }
    }

  // dA = V @ inner @ V^T
  float tmp[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += V[i * 3 + k] * inner[k][j];
      tmp[i][j] = s;
    }
  float* gB = grad_B + n * 9;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += tmp[i][k] * V[j * 3 + k];
      gB[i * 3 + j] = s;
    }
  // symmetrize (only the symmetric part of dA is meaningful for symmetric input)
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j) {
      float avg = 0.5f * (gB[i * 3 + j] + gB[j * 3 + i]);
      gB[i * 3 + j] = avg;
      gB[j * 3 + i] = avg;
    }
}

std::vector<torch::Tensor> eigh3x3_forward(torch::Tensor B) {
  CHECK_CUDA(B); CHECK_CONTIGUOUS(B);
  int N = B.size(0);
  auto eigval = torch::empty({N, 3}, B.options());
  auto eigvec = torch::empty({N, 3, 3}, B.options());
  int threads = 256, blocks = (N + threads - 1) / threads;
  eigh3x3_forward_kernel<<<blocks, threads>>>(
      B.data_ptr<float>(), eigval.data_ptr<float>(), eigvec.data_ptr<float>(), N);
  return {eigval, eigvec};
}

torch::Tensor eigh3x3_backward(torch::Tensor grad_eigval, torch::Tensor grad_eigvec,
                                 torch::Tensor eigval, torch::Tensor eigvec) {
  CHECK_CUDA(grad_eigval); CHECK_CUDA(grad_eigvec); CHECK_CUDA(eigval); CHECK_CUDA(eigvec);
  int N = eigval.size(0);
  auto grad_B = torch::empty({N, 3, 3}, eigval.options());
  int threads = 256, blocks = (N + threads - 1) / threads;
  eigh3x3_backward_kernel<<<blocks, threads>>>(
      grad_eigval.contiguous().data_ptr<float>(), grad_eigvec.contiguous().data_ptr<float>(),
      eigval.data_ptr<float>(), eigvec.data_ptr<float>(), grad_B.data_ptr<float>(), N);
  return grad_B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &eigh3x3_forward, "Batched closed-form symmetric 3x3 eigh forward (CUDA)");
  m.def("backward", &eigh3x3_backward, "Batched symmetric 3x3 eigh backward / VJP (CUDA)");
}
