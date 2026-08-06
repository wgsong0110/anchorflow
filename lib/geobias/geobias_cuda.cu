// Fused relative-geometry attention bias.
//
// For every ordered pair of anchors (i, j) the model turns the relative offset
// and distance into a per-head additive bias on the attention logit:
//
//     f_ij = [ (p_i - p_j) / s , log1p(|p_i - p_j| / s) ]        4 values
//     b_ij = W2 * silu(W1 f_ij + c1) + c2                        H values
//
// with a hidden width of 32 and H = 4 heads. The parameters are 292 numbers,
// but the map is applied to B*M*M pairs, and eager PyTorch materialises the
// intermediates: at B=8, M=512 the hidden activation alone is
// [8,512,512,32] = 268 MB, written in forward and read again in backward.
// Measured, this one 292-parameter layer was 22.3 ms of a 64 ms training
// iteration -- 35% -- entirely in memory traffic.
//
// Here each pair's chain runs in registers and only the H outputs reach memory,
// 336 MB of traffic down to 34 MB. The backward recomputes the forward instead
// of storing it: the input is a function of the anchor positions, which are
// already live, so there is nothing worth keeping.
//
// The arithmetic is unchanged. Floating-point association order differs (the
// hidden sum is accumulated in a different order than cuBLAS would), so results
// agree to float32 rounding rather than bitwise -- exe/verify_geobias.py checks
// forward and both gradients against the eager implementation.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define HID 32
#define IN 4

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float dsilu(float x) {
  float s = 1.f / (1.f + __expf(-x));
  return s * (1.f + x * (1.f - s));
}

// f = [rel/s, log1p(d/s)] for the pair (i, j) of batch b
__device__ __forceinline__ void pair_feature(
    const float* __restrict__ pos, int b, int i, int j, int M, float inv_s, float* f) {
  const float* pi = pos + ((long)b * M + i) * 3;
  const float* pj = pos + ((long)b * M + j) * 3;
  float rx = pi[0] - pj[0], ry = pi[1] - pj[1], rz = pi[2] - pj[2];
  float d = sqrtf(rx * rx + ry * ry + rz * rz);
  f[0] = rx * inv_s; f[1] = ry * inv_s; f[2] = rz * inv_s;
  f[3] = log1pf(d * inv_s);
}

template <int H>
__global__ void geobias_forward_kernel(
    const float* __restrict__ pos,      // [B,M,3]
    const float* __restrict__ w1,       // [HID,IN]
    const float* __restrict__ b1,       // [HID]
    const float* __restrict__ w2,       // [H,HID]
    const float* __restrict__ b2,       // [H]
    float inv_s, int B, int M,
    float* __restrict__ out) {          // [B,H,M,M]
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * M * M;
  if (idx >= total) return;
  int j = idx % M;
  int i = (idx / M) % M;
  int b = idx / ((long)M * M);

  float f[IN];
  pair_feature(pos, b, i, j, M, inv_s, f);

  float acc[H];
#pragma unroll
  for (int h = 0; h < H; ++h) acc[h] = b2[h];
#pragma unroll
  for (int u = 0; u < HID; ++u) {
    float z = b1[u];
#pragma unroll
    for (int c = 0; c < IN; ++c) z += w1[u * IN + c] * f[c];
    float a = silu(z);
#pragma unroll
    for (int h = 0; h < H; ++h) acc[h] += w2[h * HID + u] * a;
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    out[((long)b * H + h) * M * M + (long)i * M + j] = acc[h];
}

// Backward. Recomputes the hidden activations, accumulates parameter gradients
// with atomics (292 slots, so contention is irrelevant next to the pair count)
// and scatters the position gradient onto i and j.
template <int H>
__global__ void geobias_backward_kernel(
    const float* __restrict__ pos, const float* __restrict__ w1,
    const float* __restrict__ b1, const float* __restrict__ w2,
    const float* __restrict__ b2, const float* __restrict__ gout,   // [B,H,M,M]
    float inv_s, int B, int M,
    float* __restrict__ gpos, float* __restrict__ gw1,
    float* __restrict__ gb1, float* __restrict__ gw2, float* __restrict__ gb2) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * M * M;
  if (idx >= total) return;
  int j = idx % M;
  int i = (idx / M) % M;
  int b = idx / ((long)M * M);

  float f[IN];
  pair_feature(pos, b, i, j, M, inv_s, f);

  float go[H];
#pragma unroll
  for (int h = 0; h < H; ++h) {
    go[h] = gout[((long)b * H + h) * M * M + (long)i * M + j];
    atomicAdd(&gb2[h], go[h]);
  }

  float gf[IN] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int u = 0; u < HID; ++u) {
    float z = b1[u];
#pragma unroll
    for (int c = 0; c < IN; ++c) z += w1[u * IN + c] * f[c];
    float a = silu(z);
    float ga = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
      atomicAdd(&gw2[h * HID + u], go[h] * a);
      ga += w2[h * HID + u] * go[h];
    }
    float gz = ga * dsilu(z);
    atomicAdd(&gb1[u], gz);
#pragma unroll
    for (int c = 0; c < IN; ++c) {
      atomicAdd(&gw1[u * IN + c], gz * f[c]);
      gf[c] += w1[u * IN + c] * gz;
    }
  }

  // d f / d p:  the first three entries are (p_i - p_j)/s, the fourth is
  // log1p(d/s), whose derivative w.r.t. the offset is r / (d * (s + d)).
  const float* pi = pos + ((long)b * M + i) * 3;
  const float* pj = pos + ((long)b * M + j) * 3;
  float rx = pi[0] - pj[0], ry = pi[1] - pj[1], rz = pi[2] - pj[2];
  float d = sqrtf(rx * rx + ry * ry + rz * rz);
  float k = (d > 1e-20f) ? (gf[3] * inv_s / (d * (1.f + d * inv_s))) : 0.f;
  float g0 = gf[0] * inv_s + k * rx;
  float g1 = gf[1] * inv_s + k * ry;
  float g2 = gf[2] * inv_s + k * rz;
  float* gi = gpos + ((long)b * M + i) * 3;
  float* gj = gpos + ((long)b * M + j) * 3;
  atomicAdd(gi + 0, g0); atomicAdd(gi + 1, g1); atomicAdd(gi + 2, g2);
  atomicAdd(gj + 0, -g0); atomicAdd(gj + 1, -g1); atomicAdd(gj + 2, -g2);
}

torch::Tensor forward(torch::Tensor pos, torch::Tensor w1, torch::Tensor b1,
                      torch::Tensor w2, torch::Tensor b2, double inv_s) {
  int B = pos.size(0), M = pos.size(1), H = w2.size(0);
  TORCH_CHECK(H == 4, "geobias is compiled for 4 heads");
  TORCH_CHECK(w1.size(0) == HID && w1.size(1) == IN, "hidden must be 32x4");
  auto out = torch::empty({B, H, M, M}, pos.options());
  long total = (long)B * M * M;
  int threads = 256, blocks = (total + threads - 1) / threads;
  geobias_forward_kernel<4><<<blocks, threads>>>(
      pos.contiguous().data_ptr<float>(), w1.contiguous().data_ptr<float>(),
      b1.contiguous().data_ptr<float>(), w2.contiguous().data_ptr<float>(),
      b2.contiguous().data_ptr<float>(), (float)inv_s, B, M, out.data_ptr<float>());
  return out;
}

std::vector<torch::Tensor> backward(torch::Tensor pos, torch::Tensor w1,
                                    torch::Tensor b1, torch::Tensor w2,
                                    torch::Tensor b2, torch::Tensor gout,
                                    double inv_s) {
  int B = pos.size(0), M = pos.size(1);
  auto gpos = torch::zeros_like(pos);
  auto gw1 = torch::zeros_like(w1), gb1 = torch::zeros_like(b1);
  auto gw2 = torch::zeros_like(w2), gb2 = torch::zeros_like(b2);
  long total = (long)B * M * M;
  int threads = 256, blocks = (total + threads - 1) / threads;
  geobias_backward_kernel<4><<<blocks, threads>>>(
      pos.contiguous().data_ptr<float>(), w1.contiguous().data_ptr<float>(),
      b1.contiguous().data_ptr<float>(), w2.contiguous().data_ptr<float>(),
      b2.contiguous().data_ptr<float>(), gout.contiguous().data_ptr<float>(),
      (float)inv_s, B, M, gpos.data_ptr<float>(), gw1.data_ptr<float>(),
      gb1.data_ptr<float>(), gw2.data_ptr<float>(), gb2.data_ptr<float>());
  return {gpos, gw1, gb1, gw2, gb2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "fused geometry bias forward");
  m.def("backward", &backward, "fused geometry bias backward");
}
