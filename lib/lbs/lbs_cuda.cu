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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lbs_forward, "LBS position blend forward");
  m.def("backward", &lbs_backward, "LBS position blend backward (grad a_now)");
}
