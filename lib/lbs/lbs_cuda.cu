// Fused anchor-LBS position blend (forward + backward).
// pos[n] = sum_k w[n,k] * ( R[j] @ (x[n]-a_rest[j]) + a_rest[j] + (a_now[j]-a_rest[j]) )
// with j = idx[n,k].  Anchor rotations R are treated as constants (they are computed
// under no_grad in the torch reference), so the only differentiable input is a_now:
//   d pos[n] / d a_now[j] = sum_{k: idx[n,k]==j} w[n,k]        (scatter-add in backward)
// Reference: anchorflow.warp.lbs_warp (parity-tested).
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename S>
__global__ void lbs_forward_kernel(
    const S* __restrict__ x,        // [N,3]
    const S* __restrict__ w,        // [N,K]
    const long* __restrict__ idx,   // [N,K]
    const S* __restrict__ a_rest,   // [M,3]
    const S* __restrict__ a_now,    // [M,3]
    const S* __restrict__ R,        // [M,3,3]
    S* __restrict__ out,            // [N,3]
    int N, int K, int M) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  S xn0 = x[n*3+0], xn1 = x[n*3+1], xn2 = x[n*3+2];
  S o0 = 0, o1 = 0, o2 = 0;
  for (int k = 0; k < K; ++k) {
    long j = idx[n*K+k];
    S wk = w[n*K+k];
    S ar0 = a_rest[j*3+0], ar1 = a_rest[j*3+1], ar2 = a_rest[j*3+2];
    S d0 = xn0-ar0, d1 = xn1-ar1, d2 = xn2-ar2;
    const S* Rj = R + j*9;
    // R_j @ (x - a_rest)
    S rx0 = Rj[0]*d0 + Rj[1]*d1 + Rj[2]*d2;
    S rx1 = Rj[3]*d0 + Rj[4]*d1 + Rj[5]*d2;
    S rx2 = Rj[6]*d0 + Rj[7]*d1 + Rj[8]*d2;
    // + a_rest + (a_now - a_rest) == rx + a_now
    o0 += wk * (rx0 + a_now[j*3+0]);
    o1 += wk * (rx1 + a_now[j*3+1]);
    o2 += wk * (rx2 + a_now[j*3+2]);
  }
  out[n*3+0] = o0; out[n*3+1] = o1; out[n*3+2] = o2;
}

template <typename S>
__global__ void lbs_backward_kernel(
    const S* __restrict__ grad_out, // [N,3]
    const S* __restrict__ w,        // [N,K]
    const long* __restrict__ idx,   // [N,K]
    S* __restrict__ grad_a_now,     // [M,3]
    int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  S g0 = grad_out[n*3+0], g1 = grad_out[n*3+1], g2 = grad_out[n*3+2];
  for (int k = 0; k < K; ++k) {
    long j = idx[n*K+k];
    S wk = w[n*K+k];
    atomicAdd(&grad_a_now[j*3+0], wk*g0);
    atomicAdd(&grad_a_now[j*3+1], wk*g1);
    atomicAdd(&grad_a_now[j*3+2], wk*g2);
  }
}

torch::Tensor lbs_forward(torch::Tensor x, torch::Tensor w, torch::Tensor idx,
                          torch::Tensor a_rest, torch::Tensor a_now, torch::Tensor R) {
  int N = x.size(0), K = w.size(1), M = a_rest.size(0);
  auto out = torch::empty_like(x);
  int threads = 256, blocks = (N + threads - 1) / threads;
  lbs_forward_kernel<float><<<blocks, threads>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), idx.data_ptr<long>(),
      a_rest.data_ptr<float>(), a_now.data_ptr<float>(), R.data_ptr<float>(),
      out.data_ptr<float>(), N, K, M);
  return out;
}

torch::Tensor lbs_backward(torch::Tensor grad_out, torch::Tensor w, torch::Tensor idx, int M) {
  int N = w.size(0), K = w.size(1);
  auto grad_a_now = torch::zeros({M, 3}, grad_out.options());
  int threads = 256, blocks = (N + threads - 1) / threads;
  lbs_backward_kernel<float><<<blocks, threads>>>(
      grad_out.data_ptr<float>(), w.data_ptr<float>(), idx.data_ptr<long>(),
      grad_a_now.data_ptr<float>(), N, K);
  return grad_a_now;
}


// ---------------------------------------------------------------------------
// Fused covariance warp (forward only).
//   qg[n]   = normalize( sum_k w[n,k] * sign_k * quat[idx[n,k]] )   (wxyz)
//   sign_k  = sign(dot(quat[idx[n,k]], quat[idx[n,0]]))   (0 -> +1)
//   Rg[n]   = quat_to_matrix(qg[n])
//   out6[n] = mat3_to_cov6( Rg * cov6_to_mat3(cov6[n]) * Rg^T )
// Parity target: warp._blend_quat + geom.quat_to_matrix + Rg@S@Rg^T.
// The torch path materialises [N,K,4] and several [N,3,3] tensors (~90MB+ of
// traffic per frame at N=1.85M); this reads cov6+quat and writes cov6 only.
// Anchor rotations are detached in the reference (Procrustes under no_grad) and
// this is used under no_grad, so no backward is needed.
template <typename S>
__global__ void cov_warp_kernel(
    const S* __restrict__ quat,     // [M,4] wxyz
    const S* __restrict__ w,        // [N,K]
    const long* __restrict__ idx,   // [N,K]
    const S* __restrict__ cov6,     // [N,6]
    S* __restrict__ out6,           // [N,6]
    int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  // --- weighted, sign-aligned quaternion mean -------------------------------
  long j0 = idx[n*K+0];
  S r0 = quat[j0*4+0], r1 = quat[j0*4+1], r2 = quat[j0*4+2], r3 = quat[j0*4+3];
  S q0 = 0, q1 = 0, q2 = 0, q3 = 0;
  for (int k = 0; k < K; ++k) {
    long j = idx[n*K+k];
    S wk = w[n*K+k];
    S a0 = quat[j*4+0], a1 = quat[j*4+1], a2 = quat[j*4+2], a3 = quat[j*4+3];
    S d = a0*r0 + a1*r1 + a2*r2 + a3*r3;
    S sg = (d > 0) ? (S)1 : ((d < 0) ? (S)-1 : (S)1);   // sign(0) -> +1
    q0 += wk * sg * a0; q1 += wk * sg * a1;
    q2 += wk * sg * a2; q3 += wk * sg * a3;
  }
  S nrm = sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  if (nrm > (S)1e-12) { q0 /= nrm; q1 /= nrm; q2 /= nrm; q3 /= nrm; }
  else { q0 = 1; q1 = q2 = q3 = 0; }

  // --- quat(wxyz) -> R ------------------------------------------------------
  S tx = 2*q1, ty = 2*q2, tz = 2*q3;
  S twx = tx*q0, twy = ty*q0, twz = tz*q0;
  S txx = tx*q1, txy = ty*q1, txz = tz*q1;
  S tyy = ty*q2, tyz = tz*q2, tzz = tz*q3;
  S R00 = 1-(tyy+tzz), R01 = txy-twz,     R02 = txz+twy;
  S R10 = txy+twz,     R11 = 1-(txx+tzz), R12 = tyz-twx;
  S R20 = txz-twy,     R21 = tyz+twx,     R22 = 1-(txx+tyy);

  // --- S from cov6 [xx,xy,xz,yy,yz,zz] --------------------------------------
  S sxx = cov6[n*6+0], sxy = cov6[n*6+1], sxz = cov6[n*6+2];
  S syy = cov6[n*6+3], syz = cov6[n*6+4], szz = cov6[n*6+5];

  // --- M = R*S  then  C = M*R^T (symmetric; only 6 entries stored) ----------
  S m00 = R00*sxx + R01*sxy + R02*sxz;
  S m01 = R00*sxy + R01*syy + R02*syz;
  S m02 = R00*sxz + R01*syz + R02*szz;
  S m10 = R10*sxx + R11*sxy + R12*sxz;
  S m11 = R10*sxy + R11*syy + R12*syz;
  S m12 = R10*sxz + R11*syz + R12*szz;
  S m20 = R20*sxx + R21*sxy + R22*sxz;
  S m21 = R20*sxy + R21*syy + R22*syz;
  S m22 = R20*sxz + R21*syz + R22*szz;

  out6[n*6+0] = m00*R00 + m01*R01 + m02*R02;   // xx
  out6[n*6+1] = m00*R10 + m01*R11 + m02*R12;   // xy
  out6[n*6+2] = m00*R20 + m01*R21 + m02*R22;   // xz
  out6[n*6+3] = m10*R10 + m11*R11 + m12*R12;   // yy
  out6[n*6+4] = m10*R20 + m11*R21 + m12*R22;   // yz
  out6[n*6+5] = m20*R20 + m21*R21 + m22*R22;   // zz
}

torch::Tensor cov_warp(torch::Tensor quat, torch::Tensor w, torch::Tensor idx,
                       torch::Tensor cov6) {
  const int N = w.size(0), K = w.size(1);
  auto out6 = torch::empty({N, 6}, cov6.options());
  const int threads = 256, blocks = (N + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(cov6.scalar_type(), "cov_warp", ([&] {
    cov_warp_kernel<scalar_t><<<blocks, threads>>>(
        quat.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
        idx.data_ptr<long>(), cov6.data_ptr<scalar_t>(),
        out6.data_ptr<scalar_t>(), N, K);
  }));
  return out6;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lbs_forward, "LBS position blend forward");
  m.def("backward", &lbs_backward, "LBS position blend backward (grad a_now)");
  m.def("cov_warp", &cov_warp, "Fused covariance warp (quat blend + R S R^T)");
}
