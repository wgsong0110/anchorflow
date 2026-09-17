// 변형 모델 한 프레임의 두 병목을 융합한다.
//
// 어느 쪽도 연산이 모자라서 느린 것이 아니다. 둘 다 **중간 텐서를 전역 메모리에
// 썼다가 다시 읽는** 것이 비용의 전부였다. 실측(입자 138 만, 앵커 512, k=16):
//
//   kNN        행렬곱 6.2 ms + topk 33 ms = 40 ms.  [N,512] 점수판(1.4 GB)을
//              만들었다가 topk 가 다시 훑는다. 앵커를 공유 메모리에 올리고 고른
//              k 개를 레지스터에 들고 있으면 점수판 자체가 필요 없다.
//   스키닝+J   24 ms.  p[idx], dp[idx], dvec 같은 [N,k,3] 텐서를 여러 번 만든다.
//              한 입자를 한 스레드가 맡아 k 를 두 번 돌면 전부 레지스터에 남는다.
//
// 값은 파이토치 경로와 같아야 한다 -- exe/verify_deformcuda.py 가 대조한다.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")

// ---------------------------------------------------------------- kNN
// 앵커 전부를 공유 메모리에 올린다 (512 x 3 float = 6 KB). 입자 하나가 스레드
// 하나이고, 가장 가까운 K 개를 **삽입 정렬**로 레지스터 배열에 유지한다. K 가
// 16 이하라 배열이 레지스터에 남고, 전역 메모리에는 결과만 쓴다.
template <int KMAX>
__global__ void knn_kernel(const float* __restrict__ x,
                           const float* __restrict__ p,
                           int N, int M, int K,
                           long* __restrict__ oidx,
                           float* __restrict__ odist) {
    extern __shared__ float sp[];                 // [M,3]
    for (int i = threadIdx.x; i < M * 3; i += blockDim.x) sp[i] = p[i];
    __syncthreads();

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float px = x[3 * n], py = x[3 * n + 1], pz = x[3 * n + 2];

    float bd[KMAX];
    int   bi[KMAX];
#pragma unroll
    for (int i = 0; i < KMAX; ++i) { bd[i] = FLT_MAX; bi[i] = 0; }

    for (int m = 0; m < M; ++m) {
        const float dx = px - sp[3 * m];
        const float dy = py - sp[3 * m + 1];
        const float dz = pz - sp[3 * m + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= bd[K - 1]) continue;            // 최악값보다 크면 볼 것 없다
        int j = K - 1;
        while (j > 0 && bd[j - 1] > d2) { bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; --j; }
        bd[j] = d2; bi[j] = m;
    }
    for (int i = 0; i < K; ++i) {
        oidx[(long)n * K + i] = bi[i];
        odist[(long)n * K + i] = sqrtf(fmaxf(bd[i], 0.f));
    }
}

std::vector<torch::Tensor> knn(torch::Tensor x, torch::Tensor p, int K) {
    CHECK(x); CHECK(p);
    const int N = x.size(0), M = p.size(0);
    TORCH_CHECK(K >= 1 && K <= 32, "K must be in [1,32]");
    auto oidx = torch::empty({N, K}, x.options().dtype(torch::kLong));
    auto odist = torch::empty({N, K}, x.options());
    const int T = 128;
    const int B = (N + T - 1) / T;
    const size_t shm = (size_t)M * 3 * sizeof(float);
    if (K <= 8)
        knn_kernel<8><<<B, T, shm>>>(x.data_ptr<float>(), p.data_ptr<float>(),
                                     N, M, K, oidx.data_ptr<long>(),
                                     odist.data_ptr<float>());
    else if (K <= 16)
        knn_kernel<16><<<B, T, shm>>>(x.data_ptr<float>(), p.data_ptr<float>(),
                                      N, M, K, oidx.data_ptr<long>(),
                                      odist.data_ptr<float>());
    else
        knn_kernel<32><<<B, T, shm>>>(x.data_ptr<float>(), p.data_ptr<float>(),
                                      N, M, K, oidx.data_ptr<long>(),
                                      odist.data_ptr<float>());
    return {oidx, odist};
}

// ------------------------------------------------- 스키닝 + 야코비안
// phi(x) = x + sum_a w_a(x) dp_a,  J = I + sum_a dp_a (x) grad w_a.
//
// 파이토치 경로와 같은 식을 쓴다 (lib/anchorflow/deform.py 의 skin_with_jacobian).
// k 를 두 번 돈다: 한 번째는 softmax 의 분모와 tau 를, 두 번째는 가중합과 J 를.
// 사이의 값은 전부 레지스터에 있으므로 [N,k,*] 텐서가 하나도 생기지 않는다.
template <int KMAX>
__global__ void skinj_kernel(const float* __restrict__ x,
                             const float* __restrict__ p,
                             const float* __restrict__ dp,
                             const float* __restrict__ log_r,
                             const float* __restrict__ log_t,
                             const long* __restrict__ idx,
                             int N, int K, float h, float tau_min,
                             float* __restrict__ out,
                             float* __restrict__ J) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float xx = x[3 * n], xy = x[3 * n + 1], xz = x[3 * n + 2];
    const float inv_h2 = 1.f / (h * h);

    float dvx[KMAX], dvy[KMAX], dvz[KMAX], gg[KMAX], uu[KMAX], tt[KMAX];
    int   aid[KMAX];
    float gmax = -FLT_MAX, umax = -FLT_MAX;
    for (int i = 0; i < K; ++i) {
        const int m = (int)idx[(long)n * K + i];
        aid[i] = m;
        const float dx = xx - p[3 * m], dy = xy - p[3 * m + 1], dz = xz - p[3 * m + 2];
        dvx[i] = dx; dvy[i] = dy; dvz[i] = dz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float r2 = expf(2.f * log_r[m]);
        gg[i] = -0.5f * d2 / fmaxf(r2, 1e-12f);       // g_a
        uu[i] = -0.5f * d2 * inv_h2;                  // u 의 logit
        tt[i] = expf(log_t[m]);
        gmax = fmaxf(gmax, gg[i]);
        umax = fmaxf(umax, uu[i]);
    }
    // u = softmax(-d^2/2h^2), tau = sum u_a t_a
    float usum = 0.f;
    for (int i = 0; i < K; ++i) { uu[i] = expf(uu[i] - umax); usum += uu[i]; }
    float tau = 0.f;
    for (int i = 0; i < K; ++i) { uu[i] /= usum; tau += uu[i] * tt[i]; }
    tau = fmaxf(tau, tau_min);

    // w = softmax(g / tau).  g/tau 의 최댓값은 g 의 최댓값에서 온다 (tau > 0).
    float wsum = 0.f, w[KMAX];
    for (int i = 0; i < K; ++i) { w[i] = expf((gg[i] - gmax) / tau); wsum += w[i]; }
    for (int i = 0; i < K; ++i) w[i] /= wsum;

    // grad tau = -(1/h^2) [ sum t_a u_a dv_a - (sum t_a u_a) sum u_b dv_b ]
    float sux = 0, suy = 0, suz = 0, tux = 0, tuy = 0, tuz = 0, tusum = 0;
    for (int i = 0; i < K; ++i) {
        sux += uu[i] * dvx[i]; suy += uu[i] * dvy[i]; suz += uu[i] * dvz[i];
        const float tu = tt[i] * uu[i];
        tusum += tu;
        tux += tu * dvx[i]; tuy += tu * dvy[i]; tuz += tu * dvz[i];
    }
    const float gtx = -(tux - tusum * sux) * inv_h2;
    const float gty = -(tuy - tusum * suy) * inv_h2;
    const float gtz = -(tuz - tusum * suz) * inv_h2;

    // sum w_b grad g_b,  sum w_b g_b,  sum w_a dp_a,  sum w_a g_a dp_a,
    // 그리고 J 의 첫 항 sum (w_a/r_a^2) dp_a (x) dv_a
    float swgx = 0, swgy = 0, swgz = 0, wg = 0;
    float wdx = 0, wdy = 0, wdz = 0, wgdx = 0, wgdy = 0, wgdz = 0;
    float M00 = 0, M01 = 0, M02 = 0, M10 = 0, M11 = 0, M12 = 0,
          M20 = 0, M21 = 0, M22 = 0;
    for (int i = 0; i < K; ++i) {
        const int m = aid[i];
        const float r2 = fmaxf(expf(2.f * log_r[m]), 1e-12f);
        const float wr = w[i] / r2;
        swgx -= wr * dvx[i]; swgy -= wr * dvy[i]; swgz -= wr * dvz[i];
        wg += w[i] * gg[i];
        const float ax = dp[3 * m], ay = dp[3 * m + 1], az = dp[3 * m + 2];
        wdx += w[i] * ax; wdy += w[i] * ay; wdz += w[i] * az;
        const float wgi = w[i] * gg[i];
        wgdx += wgi * ax; wgdy += wgi * ay; wgdz += wgi * az;
        const float bx = wr * ax, by = wr * ay, bz = wr * az;
        M00 += bx * dvx[i]; M01 += bx * dvy[i]; M02 += bx * dvz[i];
        M10 += by * dvx[i]; M11 += by * dvy[i]; M12 += by * dvz[i];
        M20 += bz * dvx[i]; M21 += bz * dvy[i]; M22 += bz * dvz[i];
    }
    out[3 * n] = xx + wdx; out[3 * n + 1] = xy + wdy; out[3 * n + 2] = xz + wdz;

    const float it = 1.f / tau, it2 = it * it;
    const float Gx = swgx * it - wg * gtx * it2;
    const float Gy = swgy * it - wg * gty * it2;
    const float Gz = swgz * it - wg * gtz * it2;
    float* Jn = J + 9 * (long)n;
    Jn[0] = 1.f - M00 * it - wgdx * gtx * it2 - wdx * Gx;
    Jn[1] =     - M01 * it - wgdx * gty * it2 - wdx * Gy;
    Jn[2] =     - M02 * it - wgdx * gtz * it2 - wdx * Gz;
    Jn[3] =     - M10 * it - wgdy * gtx * it2 - wdy * Gx;
    Jn[4] = 1.f - M11 * it - wgdy * gty * it2 - wdy * Gy;
    Jn[5] =     - M12 * it - wgdy * gtz * it2 - wdy * Gz;
    Jn[6] =     - M20 * it - wgdz * gtx * it2 - wdz * Gx;
    Jn[7] =     - M21 * it - wgdz * gty * it2 - wdz * Gy;
    Jn[8] = 1.f - M22 * it - wgdz * gtz * it2 - wdz * Gz;
}

std::vector<torch::Tensor> skin_jacobian(torch::Tensor x, torch::Tensor p,
                                         torch::Tensor dp, torch::Tensor log_r,
                                         torch::Tensor log_t, torch::Tensor idx,
                                         double h, double tau_min) {
    CHECK(x); CHECK(p); CHECK(dp); CHECK(log_r); CHECK(log_t); CHECK(idx);
    const int N = x.size(0), K = idx.size(1);
    TORCH_CHECK(K >= 1 && K <= 32, "K must be in [1,32]");
    auto out = torch::empty_like(x);
    auto J = torch::empty({N, 3, 3}, x.options());
    const int T = 128, B = (N + T - 1) / T;
    auto go = [&](auto tag) {
        constexpr int KM = decltype(tag)::value;
        skinj_kernel<KM><<<B, T>>>(x.data_ptr<float>(), p.data_ptr<float>(),
                                   dp.data_ptr<float>(), log_r.data_ptr<float>(),
                                   log_t.data_ptr<float>(), idx.data_ptr<long>(),
                                   N, K, (float)h, (float)tau_min,
                                   out.data_ptr<float>(), J.data_ptr<float>());
    };
    if (K <= 8) go(std::integral_constant<int, 8>{});
    else if (K <= 16) go(std::integral_constant<int, 16>{});
    else go(std::integral_constant<int, 32>{});
    return {out, J};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "k nearest anchors, selection kept in registers");
    m.def("skin_jacobian", &skin_jacobian, "skinning and its Jacobian, fused");
}
