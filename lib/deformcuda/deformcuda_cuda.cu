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
        if (m < 0) {   // 복셀 경로에서 빈 이웃 자리. 가중치 0 이 되게 둔다
            dvx[i] = dvy[i] = dvz[i] = 0.f;
            gg[i] = -FLT_MAX / 4; uu[i] = -FLT_MAX / 4; tt[i] = 1.f;
            continue;
        }
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
        if (m < 0) continue;
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

// ------------------------------------------------- 복셀 이웃 탐색
// 앵커를 복셀 다운샘플링으로 뽑으면, 어느 가우시안이 어느 앵커에 속하는지가
// **탐색 없이** 정해진다 -- 자기 복셀과 이웃 26 칸이 곧 후보다. 512 개 앵커
// 전부와 거리를 재던 kNN 이 27 개 후보로 줄고, FPS 도 필요 없다.
//
// 키는 정렬된 유일 복셀 키 배열이라 이진 탐색으로 앵커 번호를 얻는다 (M 이 수백
// 이므로 열 번 남짓). 빈 복셀은 -1 로 두고, 스키닝 커널이 그것을 건너뛴다.
__device__ __forceinline__ int key_find(const long* __restrict__ keys, int M, long q) {
    int lo = 0, hi = M - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const long v = keys[mid];
        if (v == q) return mid;
        if (v < q) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

// ------------------------------------------------- 복셀 해시 테이블
// 앵커 집합을 만드는 데 정렬(torch.unique)을 쓰면 키를 파이토치에서 만들어야 하고
// L 개 격자면 그것이 L 배로 든다 -- 실측으로 앙상블 비용의 47% 가 거기였다.
// 여기서는 열린 주소 해시로 O(N) 에 번호를 매긴다. 조회도 이진 탐색(열 번 남짓)이
// 아니라 한두 번의 탐침이 된다.
//
// 빈 칸은 EMPTY(=LONG_MIN) 로 두고 atomicCAS 로 자리를 잡는다. 자리를 잡은 스레드만
// 번호를 하나 발급받는다 (atomicAdd on counter).

#define VOX_EMPTY (-0x7FFFFFFFFFFFFFFFLL)

__device__ __forceinline__ unsigned vox_hash(long k, unsigned mask) {
    unsigned long long h = (unsigned long long)k * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
    return (unsigned)(h) & mask;
}

__device__ __forceinline__ int vox_insert(long* __restrict__ tab,
                                          int* __restrict__ slot,
                                          int* __restrict__ cnt,
                                          unsigned mask, long k) {
    unsigned h = vox_hash(k, mask);
    for (unsigned i = 0; i <= mask; ++i) {
        const long old = atomicCAS((unsigned long long*)(tab + h),
                                   (unsigned long long)VOX_EMPTY,
                                   (unsigned long long)k);
        if (old == VOX_EMPTY) { slot[h] = atomicAdd(cnt, 1); return h; }
        if (old == k) return h;
        h = (h + 1) & mask;
    }
    return -1;
}

__device__ __forceinline__ int vox_find(const long* __restrict__ tab,
                                        const int* __restrict__ slot,
                                        unsigned mask, long k) {
    unsigned h = vox_hash(k, mask);
    for (unsigned i = 0; i <= mask; ++i) {
        const long v = tab[h];
        if (v == k) return slot[h];
        if (v == VOX_EMPTY) return -1;
        h = (h + 1) & mask;
    }
    return -1;
}

__global__ void vox_hash_build(const float* __restrict__ x,
                               const float* __restrict__ offs,
                               int N, int L, int D1, int D2, long stride,
                               float ox, float oy, float oz, float cell,
                               long* __restrict__ tab, int* __restrict__ slot,
                               int* __restrict__ cnt, unsigned mask) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += gridDim.x * blockDim.x) {
        const float px = x[3 * n], py = x[3 * n + 1], pz = x[3 * n + 2];
        for (int l = 0; l < L; ++l) {
            const int gx = (int)floorf((px - ox - offs[3 * l] * cell) / cell);
            const int gy = (int)floorf((py - oy - offs[3 * l + 1] * cell) / cell);
            const int gz = (int)floorf((pz - oz - offs[3 * l + 2] * cell) / cell);
            vox_insert(tab, slot, cnt, mask,
                       (long)l * stride + ((long)gx * D1 + gy) * D2 + gz);
        }
    }
}

std::vector<torch::Tensor> voxel_hash(torch::Tensor x, torch::Tensor offs,
                                      int D1, int D2, long stride,
                                      double ox, double oy, double oz,
                                      double cell, long table_size) {
    CHECK(x); CHECK(offs);
    const int N = x.size(0), L = offs.size(0);
    unsigned T = 1;
    while ((long)T < table_size) T <<= 1;
    auto tab = torch::full({(long)T}, VOX_EMPTY, x.options().dtype(torch::kLong));
    auto slot = torch::full({(long)T}, -1, x.options().dtype(torch::kInt));
    auto cnt = torch::zeros({1}, x.options().dtype(torch::kInt));
    const int TH = 256, B = std::min<long>(1024, (N + TH - 1) / TH);
    vox_hash_build<<<B, TH>>>(x.data_ptr<float>(), offs.data_ptr<float>(),
                              N, L, D1, D2, stride,
                              (float)ox, (float)oy, (float)oz, (float)cell,
                              tab.data_ptr<long>(), slot.data_ptr<int>(),
                              cnt.data_ptr<int>(), T - 1);
    // 테이블과 번호를 그대로 돌려준다. 조회는 정렬된 배열의 이진 탐색이 아니라
    // 탐침 한두 번이고, 앵커 위치 배열은 번호 순서이므로 따로 정렬할 것이 없다.
    // keys_by_slot 은 파이토치 예비 경로와 대조할 때만 쓴다 (M 개라 사소하다).
    const int M = cnt.item<int>();
    auto used = tab.ne(VOX_EMPTY);
    auto kb = torch::empty({M}, x.options().dtype(torch::kLong));
    kb.index_put_({slot.masked_select(used).to(torch::kLong)},
                  tab.masked_select(used));
    return {tab, slot, cnt, kb};
}


template <int K>
__global__ void voxel_knn_kernel(const float* __restrict__ x,
                                 const float* __restrict__ p,
                                 const long* __restrict__ keys,
                                 const float* __restrict__ offs,   // [L,3] 칸 단위
                                 const long* __restrict__ htab,
                                 const int* __restrict__ hslot, unsigned hmask,
                                 int N, int M, int D1, int D2, int L, long stride,
                                 float ox, float oy, float oz, float cell,
                                 int R,
                                 long* __restrict__ oidx,
                                 float* __restrict__ odist) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float px = x[3 * n], py = x[3 * n + 1], pz = x[3 * n + 2];

    float bd[K];
    int   bi[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { bd[i] = FLT_MAX; bi[i] = -1; }

    // 격자 L 개를 한 번에 돈다. 입자 좌표를 한 번만 읽고, 키에 격자 번호를 실어
    // 하나의 정렬된 배열에서 함께 찾는다 -- 파이썬 루프로 L 번 도는 것과 값은
    // 같지만 입자 읽기와 커널 실행이 공유된다.
    for (int l = 0; l < L; ++l) {
      const float axo = ox + offs[3 * l] * cell;
      const float ayo = oy + offs[3 * l + 1] * cell;
      const float azo = oz + offs[3 * l + 2] * cell;
      const int cx = (int)floorf((px - axo) / cell);
      const int cy = (int)floorf((py - ayo) / cell);
      const int cz = (int)floorf((pz - azo) / cell);
      for (int dz = -R; dz <= R; ++dz)
        for (int dy = -R; dy <= R; ++dy)
          for (int dx = -R; dx <= R; ++dx) {
            const long q = (long)l * stride
                           + ((long)(cx + dx) * D1 + (cy + dy)) * D2 + (cz + dz);
            const int a = vox_find(htab, hslot, hmask, q);
            if (a < 0) continue;
            const float ex = px - p[3 * a], ey = py - p[3 * a + 1],
                        ez = pz - p[3 * a + 2];
            const float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 >= bd[K - 1]) continue;
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
                bi[j] = put ? a : bi[j];
            }
          }
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
        oidx[(long)n * K + i] = bi[i];
        odist[(long)n * K + i] = (bi[i] < 0) ? 0.f : sqrtf(fmaxf(bd[i], 0.f));
    }
}

std::vector<torch::Tensor> voxel_knn(torch::Tensor x, torch::Tensor p,
                                     torch::Tensor tab, torch::Tensor slot,
                                     torch::Tensor offs,
                                     int D1, int D2, long stride,
                                     double ox, double oy, double oz,
                                     double cell, int K, int R) {
    CHECK(x); CHECK(p); CHECK(tab); CHECK(slot); CHECK(offs);
    const int N = x.size(0), M = p.size(0), L = offs.size(0);
    const unsigned hmask = (unsigned)tab.size(0) - 1;
    auto oidx = torch::empty({N, K}, x.options().dtype(torch::kLong));
    auto odist = torch::empty({N, K}, x.options());
    const int T = 128, B = (N + T - 1) / T;
#define VK_CASE(KK)                                                           \
    case KK: voxel_knn_kernel<KK><<<B, T>>>(x.data_ptr<float>(),              \
                 p.data_ptr<float>(), nullptr, offs.data_ptr<float>(),        \
                 tab.data_ptr<long>(), slot.data_ptr<int>(), hmask,           \
                 N, M, D1, D2, L, stride,                                     \
                 (float)ox, (float)oy, (float)oz, (float)cell, R,             \
                 oidx.data_ptr<long>(), odist.data_ptr<float>()); break;
    switch (K) {
        VK_CASE(4) VK_CASE(6) VK_CASE(8) VK_CASE(12) VK_CASE(16)
        VK_CASE(24) VK_CASE(32)
        default: TORCH_CHECK(false, "K must be one of 4,6,8,12,16,24,32");
    }
#undef VK_CASE
    return {oidx, odist};
}

// ------------------------------------------------- 복셀 생성·집계 융합
// 파이토치 경로는 torch.unique (138 만 개 int64 정렬) 로 앵커 번호를 매기고,
// index_add_ 로 [N, 41] 을 흩뿌린다. 부드러운 배정에서는 짝이 N x 27 = 3700 만
// 이 되어 unique 를 그 위에서 또 돌리므로 181 ms 가 나왔다.
//
// 여기서는 두 가지를 바꾼다.
//   앵커 집합은 **점유 복셀**로 고정한다. 부드러운 배정에서도 앵커를 새로 만들지
//   않고, 이웃 중 점유된 칸에만 B-스플라인 가중으로 뿌린다 -- 가중이 0 으로 죽는
//   자리에는 어차피 기여가 없다.
//   번호 부여는 정렬 대신 **정렬된 유일 키 배열 위의 이진 탐색**이다. 그 배열은
//   하드 배정의 unique 한 번으로 얻고, 부드러운 쪽과 앙상블이 함께 쓴다.

__device__ __forceinline__ float bspline_w(float t) {
    const float a = fabsf(t);
    if (a < 0.5f) return 0.75f - a * a;
    if (a < 1.5f) { const float b = 1.5f - a; return 0.5f * b * b; }
    return 0.f;
}

// phase 0: (질량, 질량x위치, 질량x정준, 질량x속도, 개수) = 11
// phase 1: (S 9, L 3) = 12,   phase 2: (A 9, B 9) = 18
template <int D>
__global__ void vox_moment_kernel(const float* __restrict__ x,
                                  const float* __restrict__ X,
                                  const float* __restrict__ v,
                                  const float* __restrict__ m,
                                  const long* __restrict__ keys,
                                  const float* __restrict__ offs,
                                  const float* __restrict__ cx,
                                  const float* __restrict__ cX,
                                  const float* __restrict__ cv,
                                  int N, int M, int D1, int D2, int L, long stride,
                                  float ox, float oy, float oz, float cell,
                                  int soft, int phase, int use_shared,
                                  const long* __restrict__ htab,
                                  const int* __restrict__ hslot, unsigned hmask,
                                  float* __restrict__ out) {
    // 공유 메모리에 [M,D] 누적기를 두는 것이 원자 경합을 줄이지만, M 이 크면
    // 48 KB 한도를 넘어 실행이 아예 안 된다 (앵커 756 x 18 = 54 KB). 그때는
    // 전역으로 바로 원자합한다 -- 느리지만 돈다.
    extern __shared__ float acc[];
    if (use_shared) {
        for (int i = threadIdx.x; i < M * D; i += blockDim.x) acc[i] = 0.f;
        __syncthreads();
    }

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += gridDim.x * blockDim.x) {
        const float px = x[3 * n], py = x[3 * n + 1], pz = x[3 * n + 2];
        const int R = soft ? 1 : 0;
        for (int l = 0; l < L; ++l) {
        const float axo = ox + offs[3 * l] * cell;
        const float ayo = oy + offs[3 * l + 1] * cell;
        const float azo = oz + offs[3 * l + 2] * cell;
        const float ux = (px - axo) / cell - 0.5f;
        const float uy = (py - ayo) / cell - 0.5f;
        const float uz = (pz - azo) / cell - 0.5f;
        const int cxi = (int)floorf((px - axo) / cell);
        const int cyi = (int)floorf((py - ayo) / cell);
        const int czi = (int)floorf((pz - azo) / cell);
        for (int dz = -R; dz <= R; ++dz)
          for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                const int gx = cxi + dx, gy = cyi + dy, gz = czi + dz;
                float w = m[n];
                if (soft) {
                    w *= bspline_w(ux - gx) * bspline_w(uy - gy)
                         * bspline_w(uz - gz);
                    if (w <= 1e-9f) continue;
                }
                const long q = (long)l * stride + ((long)gx * D1 + gy) * D2 + gz;
                const int aa = vox_find(htab, hslot, hmask, q);
                if (aa < 0) continue;
                float* dst = (use_shared ? acc : out) + (long)aa * D;
                if (phase == 0) {
                    atomicAdd(dst + 0, w);
                    atomicAdd(dst + 1, w * px); atomicAdd(dst + 2, w * py);
                    atomicAdd(dst + 3, w * pz);
                    atomicAdd(dst + 4, w * X[3 * n]);
                    atomicAdd(dst + 5, w * X[3 * n + 1]);
                    atomicAdd(dst + 6, w * X[3 * n + 2]);
                    atomicAdd(dst + 7, w * v[3 * n]);
                    atomicAdd(dst + 8, w * v[3 * n + 1]);
                    atomicAdd(dst + 9, w * v[3 * n + 2]);
                    atomicAdd(dst + 10, 1.f);
                } else {
                    const float ax = px - cx[3 * aa], ay = py - cx[3 * aa + 1],
                                az = pz - cx[3 * aa + 2];
                    const float bx = X[3 * n] - cX[3 * aa],
                                by = X[3 * n + 1] - cX[3 * aa + 1],
                                bz = X[3 * n + 2] - cX[3 * aa + 2];
                    const float ex = v[3 * n] - cv[3 * aa],
                                ey = v[3 * n + 1] - cv[3 * aa + 1],
                                ez = v[3 * n + 2] - cv[3 * aa + 2];
                    if (phase == 1) {
                        atomicAdd(dst + 0, w * ax * ax); atomicAdd(dst + 1, w * ax * ay);
                        atomicAdd(dst + 2, w * ax * az); atomicAdd(dst + 3, w * ay * ax);
                        atomicAdd(dst + 4, w * ay * ay); atomicAdd(dst + 5, w * ay * az);
                        atomicAdd(dst + 6, w * az * ax); atomicAdd(dst + 7, w * az * ay);
                        atomicAdd(dst + 8, w * az * az);
                        atomicAdd(dst + 9,  w * (ay * ez - az * ey));
                        atomicAdd(dst + 10, w * (az * ex - ax * ez));
                        atomicAdd(dst + 11, w * (ax * ey - ay * ex));
                    } else {
                        atomicAdd(dst + 0, w * ax * bx); atomicAdd(dst + 1, w * ax * by);
                        atomicAdd(dst + 2, w * ax * bz); atomicAdd(dst + 3, w * ay * bx);
                        atomicAdd(dst + 4, w * ay * by); atomicAdd(dst + 5, w * ay * bz);
                        atomicAdd(dst + 6, w * az * bx); atomicAdd(dst + 7, w * az * by);
                        atomicAdd(dst + 8, w * az * bz);
                        atomicAdd(dst + 9,  w * bx * bx); atomicAdd(dst + 10, w * bx * by);
                        atomicAdd(dst + 11, w * bx * bz); atomicAdd(dst + 12, w * by * bx);
                        atomicAdd(dst + 13, w * by * by); atomicAdd(dst + 14, w * by * bz);
                        atomicAdd(dst + 15, w * bz * bx); atomicAdd(dst + 16, w * bz * by);
                        atomicAdd(dst + 17, w * bz * bz);
                    }
                }
            }
        }
    }
    if (use_shared) {
        __syncthreads();
        for (int i = threadIdx.x; i < M * D; i += blockDim.x)
            if (acc[i] != 0.f) atomicAdd(out + i, acc[i]);
    }
}

std::vector<torch::Tensor> voxel_moments(
        torch::Tensor x, torch::Tensor X, torch::Tensor v, torch::Tensor m,
        torch::Tensor tab, torch::Tensor slot, int M, torch::Tensor offs,
        int D1, int D2, long stride,
        double ox, double oy, double oz, double cell, bool soft) {
    CHECK(x); CHECK(X); CHECK(v); CHECK(m); CHECK(tab); CHECK(slot); CHECK(offs);
    const int N = x.size(0), L = offs.size(0);
    const unsigned hmask = (unsigned)tab.size(0) - 1;
    auto opt = x.options();
    const int T = 256, B = std::min<long>(160, (N + T - 1) / T);
    // 48 KB 를 넘으면 공유 누적기를 포기한다
    auto shm = [&](int MM, int D) -> size_t {
        const size_t b = (size_t)MM * D * sizeof(float);
        return b <= 47000 ? b : 0;
    };
    auto usesh = [&](int MM, int D) { return shm(MM, D) > 0 ? 1 : 0; };
    auto g1 = torch::zeros({M, 11}, opt);
    vox_moment_kernel<11><<<B, T, shm(M, 11)>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), nullptr, offs.data_ptr<float>(),
        nullptr, nullptr, nullptr, N, M, D1, D2, L, stride,
        (float)ox, (float)oy, (float)oz, (float)cell,
        soft ? 1 : 0, 0, usesh(M, 11),
        tab.data_ptr<long>(), slot.data_ptr<int>(), hmask, g1.data_ptr<float>());
    auto W = g1.select(1, 0).clamp_min(1e-12).unsqueeze(-1);
    auto cx = (g1.slice(1, 1, 4) / W).contiguous();
    auto cX = (g1.slice(1, 4, 7) / W).contiguous();
    auto cv = (g1.slice(1, 7, 10) / W).contiguous();
    auto g2 = torch::zeros({M, 12}, opt);
    vox_moment_kernel<12><<<B, T, shm(M, 12)>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), nullptr, offs.data_ptr<float>(),
        cx.data_ptr<float>(), cX.data_ptr<float>(), cv.data_ptr<float>(),
        N, M, D1, D2, L, stride,
        (float)ox, (float)oy, (float)oz, (float)cell, soft ? 1 : 0, 1,
        usesh(M, 12),
        tab.data_ptr<long>(), slot.data_ptr<int>(), hmask, g2.data_ptr<float>());
    auto g3 = torch::zeros({M, 18}, opt);
    vox_moment_kernel<18><<<B, T, shm(M, 18)>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), nullptr, offs.data_ptr<float>(),
        cx.data_ptr<float>(), cX.data_ptr<float>(), cv.data_ptr<float>(),
        N, M, D1, D2, L, stride,
        (float)ox, (float)oy, (float)oz, (float)cell, soft ? 1 : 0, 2,
        usesh(M, 18),
        tab.data_ptr<long>(), slot.data_ptr<int>(), hmask, g3.data_ptr<float>());
    return {g1, g2, g3, cx, cX, cv};
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

// ---- 집계 2 판: 짝이 아니라 **입자**마다 한 스레드 ----------------------------
// 1 판은 스레드 하나가 (입자, 이웃) 짝 하나를 맡아서, 같은 입자의 x/X/v/m 을
// K=16 번 다시 읽었다. 입자마다 맡기면 한 번만 읽는다 -- 전역 통행량이 16 분의 1.
// 덧붙여 2 차 모멘트 두 판(12 + 18)을 하나(30)로 합쳤다. 공유 메모리가 48 KB 를
// 넘어가므로 sm_80+ 의 옵트인(최대 99 KB)을 켠다. 짝 목록을 한 번 덜 읽는다.
// 색인은 int32 로 받는다 -- long 이면 2200 만 개가 176 MB 이고, 이 커널은 대역이
// 정한다.
template <int D, int PH>
__global__ void agg_np(const float* __restrict__ x, const float* __restrict__ X,
                       const float* __restrict__ v, const float* __restrict__ m,
                       const int* __restrict__ idx,
                       const float* __restrict__ cx, const float* __restrict__ cX,
                       const float* __restrict__ cv,
                       int N, int K, int M, int use_shared,
                       float* __restrict__ out) {
    extern __shared__ float acc[];
    float* dstbase = use_shared ? acc : out;
    if (use_shared) {
        for (int i = threadIdx.x; i < M * D; i += blockDim.x) acc[i] = 0.f;
        __syncthreads();
    }
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += gridDim.x * blockDim.x) {
        const float w = m[n];
        const float x0 = x[3 * n], x1 = x[3 * n + 1], x2 = x[3 * n + 2];
        float X0 = 0.f, X1 = 0.f, X2 = 0.f, v0 = 0.f, v1 = 0.f, v2 = 0.f;
        if (PH != 1) { X0 = X[3 * n]; X1 = X[3 * n + 1]; X2 = X[3 * n + 2]; }
        if (PH != 2) { v0 = v[3 * n]; v1 = v[3 * n + 1]; v2 = v[3 * n + 2]; }
        if (PH == 0) { X0 = X[3 * n]; X1 = X[3 * n + 1]; X2 = X[3 * n + 2]; }
        const long base = (long)n * K;
        for (int j = 0; j < K; ++j) {
            const int aa = idx[base + j];
            float* dst = dstbase + (long)aa * D;
            if (PH == 0) {
                atomicAdd(dst + 0, w);
                atomicAdd(dst + 1, w * x0);
                atomicAdd(dst + 2, w * x1);
                atomicAdd(dst + 3, w * x2);
                atomicAdd(dst + 4, w * X0);
                atomicAdd(dst + 5, w * X1);
                atomicAdd(dst + 6, w * X2);
                atomicAdd(dst + 7, w * v0);
                atomicAdd(dst + 8, w * v1);
                atomicAdd(dst + 9, w * v2);
                atomicAdd(dst + 10, 1.f);
            } else {                       // PH 1: S,L / PH 2: A,B / PH 3: 둘 다
                const float dx0 = x0 - cx[3 * aa];
                const float dx1 = x1 - cx[3 * aa + 1];
                const float dx2 = x2 - cx[3 * aa + 2];
                const float dX0 = X0 - cX[3 * aa];
                const float dX1 = X1 - cX[3 * aa + 1];
                const float dX2 = X2 - cX[3 * aa + 2];
                const float dv0 = v0 - cv[3 * aa];
                const float dv1 = v1 - cv[3 * aa + 1];
                const float dv2 = v2 - cv[3 * aa + 2];
                // S = dx (x) dx 와 B = dX (x) dX 는 **대칭**이라 위 삼각 6 개만
                // 더하면 된다. 원자합 수가 41 개에서 35 개로 준다 -- 이 커널은
                // 공유 메모리 원자합 발행률이 정하므로 그만큼 그대로 빨라진다.
                const int o = (PH == 3) ? 9 : 0;
                if (PH == 1 || PH == 3) {
                    atomicAdd(dst + 0, w * dx0 * dx0); atomicAdd(dst + 1, w * dx0 * dx1);
                    atomicAdd(dst + 2, w * dx0 * dx2); atomicAdd(dst + 3, w * dx1 * dx1);
                    atomicAdd(dst + 4, w * dx1 * dx2); atomicAdd(dst + 5, w * dx2 * dx2);
                    atomicAdd(dst + 6, w * (dx1 * dv2 - dx2 * dv1));
                    atomicAdd(dst + 7, w * (dx2 * dv0 - dx0 * dv2));
                    atomicAdd(dst + 8, w * (dx0 * dv1 - dx1 * dv0));
                }
                if (PH == 2 || PH == 3) {
                    atomicAdd(dst + o + 0, w * dx0 * dX0); atomicAdd(dst + o + 1, w * dx0 * dX1);
                    atomicAdd(dst + o + 2, w * dx0 * dX2); atomicAdd(dst + o + 3, w * dx1 * dX0);
                    atomicAdd(dst + o + 4, w * dx1 * dX1); atomicAdd(dst + o + 5, w * dx1 * dX2);
                    atomicAdd(dst + o + 6, w * dx2 * dX0); atomicAdd(dst + o + 7, w * dx2 * dX1);
                    atomicAdd(dst + o + 8, w * dx2 * dX2);
                    atomicAdd(dst + o + 9,  w * dX0 * dX0); atomicAdd(dst + o + 10, w * dX0 * dX1);
                    atomicAdd(dst + o + 11, w * dX0 * dX2); atomicAdd(dst + o + 12, w * dX1 * dX1);
                    atomicAdd(dst + o + 13, w * dX1 * dX2); atomicAdd(dst + o + 14, w * dX2 * dX2);
                }
            }
        }
    }
    if (use_shared) {
        __syncthreads();
        for (int i = threadIdx.x; i < M * D; i += blockDim.x)
            if (acc[i] != 0.f) atomicAdd(out + i, acc[i]);
    }
}

static int agg_shared_ok(size_t bytes, const void* fn) {
    // sm_80+ 는 블록당 동적 공유 메모리를 옵트인으로 99 KB 까지 늘릴 수 있다.
    static int cap = -1;
    if (cap < 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    }
    if ((int)bytes > cap) return 0;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    return 1;
}

std::vector<torch::Tensor> aggregate_moments2(
        torch::Tensor x, torch::Tensor X, torch::Tensor v, torch::Tensor m,
        torch::Tensor idx, int M, int merge) {
    CHECK(x); CHECK(X); CHECK(v); CHECK(m); CHECK(idx);
    const int N = x.size(0), K = idx.size(1);
    auto opt = x.options();
    auto i32 = idx.to(torch::kInt).contiguous();
    const int* ip = i32.data_ptr<int>();

    const int T = 256;
    const int B = std::min<long>(320, ((long)N + T - 1) / T);

    auto g1 = torch::zeros({M, 11}, opt);
    size_t sh1 = (size_t)M * 11 * sizeof(float);
    int us1 = agg_shared_ok(sh1, (const void*)agg_np<11, 0>);
    agg_np<11, 0><<<B, T, us1 ? sh1 : 0>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), ip, nullptr, nullptr, nullptr, N, K, M, us1,
        g1.data_ptr<float>());

    auto W = g1.select(1, 0).clamp_min(1e-12).unsqueeze(-1);
    auto cx = (g1.slice(1, 1, 4) / W).contiguous();
    auto cX = (g1.slice(1, 4, 7) / W).contiguous();
    auto cv = (g1.slice(1, 7, 10) / W).contiguous();

    // 위 삼각 6 개 -> 3x3 9 개로 되펴는 색인
    auto sy = torch::tensor({0, 1, 2, 1, 3, 4, 2, 4, 5},
                            opt.dtype(torch::kLong)).to(x.device());
    if (merge) {
        // 한 판으로 합치면 짝 목록을 한 번 덜 읽지만 공유 메모리가 커져 SM 당
        // 블록이 하나로 떨어진다. 실측 13.3 ms 대 8.8 ms 로 오히려 느렸다.
        auto g23 = torch::zeros({M, 24}, opt);
        size_t sh2 = (size_t)M * 24 * sizeof(float);
        int us2 = agg_shared_ok(sh2, (const void*)agg_np<24, 3>);
        agg_np<24, 3><<<B, T, us2 ? sh2 : 0>>>(
            x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
            m.data_ptr<float>(), ip, cx.data_ptr<float>(), cX.data_ptr<float>(),
            cv.data_ptr<float>(), N, K, M, us2, g23.data_ptr<float>());
        auto g2 = torch::cat({g23.slice(1, 0, 6).index_select(1, sy),
                              g23.slice(1, 6, 9)}, 1).contiguous();
        auto g3 = torch::cat({g23.slice(1, 9, 18),
                              g23.slice(1, 18, 24).index_select(1, sy)},
                             1).contiguous();
        return {g1, g2, g3};
    }
    auto g2s = torch::zeros({M, 9}, opt);
    size_t sha = (size_t)M * 9 * sizeof(float);
    int usa = agg_shared_ok(sha, (const void*)agg_np<9, 1>);
    agg_np<9, 1><<<B, T, usa ? sha : 0>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), ip, cx.data_ptr<float>(), cX.data_ptr<float>(),
        cv.data_ptr<float>(), N, K, M, usa, g2s.data_ptr<float>());
    auto g3s = torch::zeros({M, 15}, opt);
    size_t shb = (size_t)M * 15 * sizeof(float);
    int usb = agg_shared_ok(shb, (const void*)agg_np<15, 2>);
    agg_np<15, 2><<<B, T, usb ? shb : 0>>>(
        x.data_ptr<float>(), X.data_ptr<float>(), v.data_ptr<float>(),
        m.data_ptr<float>(), ip, cx.data_ptr<float>(), cX.data_ptr<float>(),
        cv.data_ptr<float>(), N, K, M, usb, g3s.data_ptr<float>());
    auto g2 = torch::cat({g2s.slice(1, 0, 6).index_select(1, sy),
                          g2s.slice(1, 6, 9)}, 1).contiguous();
    auto g3 = torch::cat({g3s.slice(1, 0, 9),
                          g3s.slice(1, 9, 15).index_select(1, sy)},
                         1).contiguous();
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
    m.def("aggregate_moments2", &aggregate_moments2,
          "same moments, one thread per particle; merge=1 folds the two "
          "second-order passes into one",
          py::arg("x"), py::arg("X"), py::arg("v"), py::arg("m"),
          py::arg("idx"), py::arg("M"), py::arg("merge") = 0);
    m.def("fps", &fps_cuda, "farthest point sampling, one kernel per pick");
    m.def("voxel_hash", &voxel_hash,
          "assign voxel ids with an open-addressing hash, no sort");
    m.def("voxel_moments", &voxel_moments,
          "voxel anchor moments, hard or B-spline, on the occupied set");
    m.def("voxel_knn", &voxel_knn,
          "nearest anchors among the 3x3x3 voxel neighbourhood, no search");
}
