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
template <int K>
__global__ void knn_kernel(const float* __restrict__ x,
                           const float* __restrict__ p,
                           int N, int M,
                           long* __restrict__ oidx,
                           float* __restrict__ odist) {
    extern __shared__ float sp[];                 // [M,3]
    for (int i = threadIdx.x; i < M * 3; i += blockDim.x) sp[i] = p[i];
    __syncthreads();

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float px = x[3 * n], py = x[3 * n + 1], pz = x[3 * n + 2];

    // K 가 **컴파일 타임 상수**여야 한다. 런타임 값으로 두면 배열에 동적 색인이
    // 남아 레지스터가 아니라 로컬 메모리로 내려가고, k=16 에서 파이토치 경로보다
    // 세 배 느려졌다 (144 ms 대 40 ms).
    float bd[K];
    int   bi[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { bd[i] = FLT_MAX; bi[i] = 0; }

    for (int m = 0; m < M; ++m) {
        const float dx = px - sp[3 * m];
        const float dy = py - sp[3 * m + 1];
        const float dz = pz - sp[3 * m + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= bd[K - 1]) continue;            // 최악값보다 크면 볼 것 없다
        // 삽입 정렬을 완전히 펼친다 -- 조건부 이동만 남아 분기도 동적 색인도 없다.
        // 먼저 들어갈 자리를 "나보다 작은 것의 개수"로 세고, 그 위만 한 칸씩 민다.
        int pos = 0;
#pragma unroll
        for (int j = 0; j < K; ++j) pos += (bd[j] < d2);
#pragma unroll
        for (int j = K - 1; j > 0; --j) {
            const bool sh = (j > pos);
            bd[j] = sh ? bd[j - 1] : bd[j];
            bi[j] = sh ? bi[j - 1] : bi[j];
        }
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const bool put = (j == pos);
            bd[j] = put ? d2 : bd[j];
            bi[j] = put ? m : bi[j];
        }
    }
#pragma unroll
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
#define KNN_CASE(KK)                                                          \
    case KK: knn_kernel<KK><<<B, T, shm>>>(x.data_ptr<float>(),               \
                 p.data_ptr<float>(), N, M, oidx.data_ptr<long>(),            \
                 odist.data_ptr<float>()); break;
    switch (K) {
        KNN_CASE(4) KNN_CASE(6) KNN_CASE(8) KNN_CASE(12) KNN_CASE(16)
        KNN_CASE(24) KNN_CASE(32)
        default: TORCH_CHECK(false, "K must be one of 4,6,8,12,16,24,32");
    }
#undef KNN_CASE
    return {oidx, odist};
}

// ------------------------------------------------- 스키닝 + 야코비안
// phi(x) = x + sum_a w_a(x) dp_a,  J = I + sum_a dp_a (x) grad w_a.
//
// 파이토치 경로와 같은 식을 쓴다 (lib/anchorflow/deform.py 의 skin_with_jacobian).
// k 를 두 번 돈다: 한 번째는 softmax 의 분모와 tau 를, 두 번째는 가중합과 J 를.
// 사이의 값은 전부 레지스터에 있으므로 [N,k,*] 텐서가 하나도 생기지 않는다.
template <int K>
__global__ void skinj_kernel(const float* __restrict__ x,
                             const float* __restrict__ p,
                             const float* __restrict__ dp,
                             const float* __restrict__ log_r,
                             const float* __restrict__ log_t,
                             const long* __restrict__ idx,
                             int N, float h, float tau_min,
                             float* __restrict__ out,
                             float* __restrict__ J) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float xx = x[3 * n], xy = x[3 * n + 1], xz = x[3 * n + 2];
    const float inv_h2 = 1.f / (h * h);

    float dvx[K], dvy[K], dvz[K], gg[K], uu[K], tt[K];
    int   aid[K];
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
    float wsum = 0.f, w[K];
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
#define SKIN_CASE(KK)                                                         \
    case KK: skinj_kernel<KK><<<B, T>>>(x.data_ptr<float>(),                  \
                 p.data_ptr<float>(), dp.data_ptr<float>(),                   \
                 log_r.data_ptr<float>(), log_t.data_ptr<float>(),            \
                 idx.data_ptr<long>(), N, (float)h, (float)tau_min,           \
                 out.data_ptr<float>(), J.data_ptr<float>()); break;
    switch (K) {
        SKIN_CASE(4) SKIN_CASE(6) SKIN_CASE(8) SKIN_CASE(12) SKIN_CASE(16)
        SKIN_CASE(24) SKIN_CASE(32)
        default: TORCH_CHECK(false, "K must be one of 4,6,8,12,16,24,32");
    }
#undef SKIN_CASE
    return {out, J};
}

// ---------------------------------------------------------------- 집계
// 앵커별로 자기 가우시안들을 질량 가중으로 요약한다. 파이토치 쪽은 [N*k, 41]
// 짜리 텐서를 만들어 index_add_ 로 흩뿌리는데, 입자 138 만 x k=16 이면 그것만
// 3.6 GB 다 -- 실측 68 ms 로 연산이 아니라 그 통행량이 전부였다.
//
// 여기서는 블록마다 공유 메모리에 [M, D] 누적기를 두고, 블록이 맡은 짝을 전부
// 더한 뒤 앵커당 한 번만 전역으로 원자합한다. 원자 경합이 (짝 수)에서
// (블록 수)로 줄고 중간 텐서가 없다.
//
// D 가 큰 2 차 모멘트는 공유 메모리 한도 때문에 두 번에 나눠 돈다.

template <int D>
__global__ void agg_kernel(const float* __restrict__ val,   // [N*k, D] 는 만들지 않는다
                           const float* __restrict__ x,
                           const float* __restrict__ X,
                           const float* __restrict__ v,
                           const float* __restrict__ m,
                           const long* __restrict__ idx,
                           const float* __restrict__ cx,    // 2차에서만 쓴다
                           const float* __restrict__ cX,
                           const float* __restrict__ cv,
                           int N, int K, int M, int phase,
                           float* __restrict__ out) {
    extern __shared__ float acc[];                  // [M, D]
    for (int i = threadIdx.x; i < M * D; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();

    const long total = (long)N * K;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x) {
        const int n = (int)(t / K);
        const int aa = (int)idx[t];
        const float w = m[n];
        float* dst = acc + (long)aa * D;
        if (phase == 0) {
            atomicAdd(dst + 0, w);
            atomicAdd(dst + 1, w * x[3 * n]);
            atomicAdd(dst + 2, w * x[3 * n + 1]);
            atomicAdd(dst + 3, w * x[3 * n + 2]);
            atomicAdd(dst + 4, w * X[3 * n]);
            atomicAdd(dst + 5, w * X[3 * n + 1]);
            atomicAdd(dst + 6, w * X[3 * n + 2]);
            atomicAdd(dst + 7, w * v[3 * n]);
            atomicAdd(dst + 8, w * v[3 * n + 1]);
            atomicAdd(dst + 9, w * v[3 * n + 2]);
            atomicAdd(dst + 10, 1.f);
        } else {
            const float dx0 = x[3 * n] - cx[3 * aa];
            const float dx1 = x[3 * n + 1] - cx[3 * aa + 1];
            const float dx2 = x[3 * n + 2] - cx[3 * aa + 2];
            const float dX0 = X[3 * n] - cX[3 * aa];
            const float dX1 = X[3 * n + 1] - cX[3 * aa + 1];
            const float dX2 = X[3 * n + 2] - cX[3 * aa + 2];
            const float dv0 = v[3 * n] - cv[3 * aa];
            const float dv1 = v[3 * n + 1] - cv[3 * aa + 1];
            const float dv2 = v[3 * n + 2] - cv[3 * aa + 2];
            if (phase == 1) {                       // S (9) + L (3)
                atomicAdd(dst + 0, w * dx0 * dx0); atomicAdd(dst + 1, w * dx0 * dx1);
                atomicAdd(dst + 2, w * dx0 * dx2); atomicAdd(dst + 3, w * dx1 * dx0);
                atomicAdd(dst + 4, w * dx1 * dx1); atomicAdd(dst + 5, w * dx1 * dx2);
                atomicAdd(dst + 6, w * dx2 * dx0); atomicAdd(dst + 7, w * dx2 * dx1);
                atomicAdd(dst + 8, w * dx2 * dx2);
                atomicAdd(dst + 9,  w * (dx1 * dv2 - dx2 * dv1));
                atomicAdd(dst + 10, w * (dx2 * dv0 - dx0 * dv2));
                atomicAdd(dst + 11, w * (dx0 * dv1 - dx1 * dv0));
            } else {                                // A (9) + B (9)
                atomicAdd(dst + 0, w * dx0 * dX0); atomicAdd(dst + 1, w * dx0 * dX1);
                atomicAdd(dst + 2, w * dx0 * dX2); atomicAdd(dst + 3, w * dx1 * dX0);
                atomicAdd(dst + 4, w * dx1 * dX1); atomicAdd(dst + 5, w * dx1 * dX2);
                atomicAdd(dst + 6, w * dx2 * dX0); atomicAdd(dst + 7, w * dx2 * dX1);
                atomicAdd(dst + 8, w * dx2 * dX2);
                atomicAdd(dst + 9,  w * dX0 * dX0); atomicAdd(dst + 10, w * dX0 * dX1);
                atomicAdd(dst + 11, w * dX0 * dX2); atomicAdd(dst + 12, w * dX1 * dX0);
                atomicAdd(dst + 13, w * dX1 * dX1); atomicAdd(dst + 14, w * dX1 * dX2);
                atomicAdd(dst + 15, w * dX2 * dX0); atomicAdd(dst + 16, w * dX2 * dX1);
                atomicAdd(dst + 17, w * dX2 * dX2);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M * D; i += blockDim.x)
        if (acc[i] != 0.f) atomicAdd(out + i, acc[i]);
}

std::vector<torch::Tensor> aggregate_moments(
        torch::Tensor x, torch::Tensor X, torch::Tensor v, torch::Tensor m,
        torch::Tensor idx, int M) {
    CHECK(x); CHECK(X); CHECK(v); CHECK(m); CHECK(idx);
    const int N = x.size(0), K = idx.size(1);
    auto opt = x.options();
    auto g1 = torch::zeros({M, 11}, opt);
    // 블록마다 마지막에 [M,D] 전체를 전역 원자합으로 흘린다. 블록이 많으면 그
    // 흘리기가 비용을 지배한다 -- 2048 블록이면 1150 만 번이다. 블록을 줄이고
    // 각 블록이 격자 보폭으로 더 많은 짝을 맡게 하면 그 수가 그만큼 준다.
    const int T = 256;
    const int B = std::min<long>(160, ((long)N * K + T - 1) / T);
    agg_kernel<11><<<B, T, (size_t)M * 11 * sizeof(float)>>>(
        nullptr, x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), idx.data_ptr<long>(), nullptr, nullptr, nullptr,
        N, K, M, 0, g1.data_ptr<float>());

    auto W = g1.select(1, 0).clamp_min(1e-12).unsqueeze(-1);
    auto cx = (g1.slice(1, 1, 4) / W).contiguous();
    auto cX = (g1.slice(1, 4, 7) / W).contiguous();
    auto cv = (g1.slice(1, 7, 10) / W).contiguous();

    auto g2 = torch::zeros({M, 12}, opt);
    agg_kernel<12><<<B, T, (size_t)M * 12 * sizeof(float)>>>(
        nullptr, x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), idx.data_ptr<long>(), cx.data_ptr<float>(),
        cX.data_ptr<float>(), cv.data_ptr<float>(), N, K, M, 1,
        g2.data_ptr<float>());
    auto g3 = torch::zeros({M, 18}, opt);
    agg_kernel<18><<<B, T, (size_t)M * 18 * sizeof(float)>>>(
        nullptr, x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), idx.data_ptr<long>(), cx.data_ptr<float>(),
        cX.data_ptr<float>(), cv.data_ptr<float>(), N, K, M, 2,
        g3.data_ptr<float>());
    return {g1, g2, g3};
}

// ---------------------------------------------------------------- FPS
// 반복 M 번, 각 반복이 (전체 최댓값 찾기) + (거리 갱신)이다. 파이토치로는 반복마다
// argmax 와 minimum 이 따로 실행되어 커널이 M x 2 번 뜬다 (실측 136 ms).
// 여기서는 두 일을 한 커널에 합치고, 블록별 부분 최댓값만 전역에 남긴다.

__global__ void fps_step(const float* __restrict__ x, float* __restrict__ d,
                         int N, int last, float* __restrict__ bval,
                         int* __restrict__ bidx) {
    extern __shared__ char smem[];
    float* sv = (float*)smem;
    int* si = (int*)(sv + blockDim.x);
    const float lx = x[3 * last], ly = x[3 * last + 1], lz = x[3 * last + 2];
    float best = -1.f; int bi = 0;
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += gridDim.x * blockDim.x) {
        const float dx = x[3 * n] - lx, dy = x[3 * n + 1] - ly,
                    dz = x[3 * n + 2] - lz;
        const float nd = sqrtf(dx * dx + dy * dy + dz * dz);
        const float cur = fminf(d[n], nd);
        d[n] = cur;
        if (cur > best) { best = cur; bi = n; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sv[threadIdx.x + s] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { bval[blockIdx.x] = sv[0]; bidx[blockIdx.x] = si[0]; }
}

torch::Tensor fps_cuda(torch::Tensor x, int M, int first) {
    CHECK(x);
    const int N = x.size(0);
    auto d = torch::full({N}, 1e30f, x.options());
    auto idx = torch::empty({M}, x.options().dtype(torch::kLong));
    auto hidx = torch::empty({M}, torch::dtype(torch::kLong));
    const int T = 256, B = std::min(1024, (N + T - 1) / T);
    auto bval = torch::empty({B}, x.options());
    auto bidx = torch::empty({B}, x.options().dtype(torch::kInt));
    int last = first;
    hidx[0] = last;
    for (int i = 1; i < M; ++i) {
        fps_step<<<B, T, T * (sizeof(float) + sizeof(int))>>>(
            x.data_ptr<float>(), d.data_ptr<float>(), N, last,
            bval.data_ptr<float>(), bidx.data_ptr<int>());
        // 블록이 1024 개뿐이라 마지막 축약은 호스트로 가져와도 싸다
        last = bidx[bval.argmax()].item<int>();
        hidx[i] = last;
    }
    idx.copy_(hidx);
    return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "k nearest anchors, selection kept in registers");
    m.def("skin_jacobian", &skin_jacobian, "skinning and its Jacobian, fused");
    m.def("aggregate_moments", &aggregate_moments,
          "per-anchor mass-weighted moments, accumulated in shared memory");
    m.def("fps", &fps_cuda, "farthest point sampling, one kernel per pick");
}
