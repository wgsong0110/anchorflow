/*
 * Per-node Selective SSM step (Mamba-style, ZOH discretization)
 *
 * Forward per node i, dim d:
 *   A = -exp(A_log[d, n])          (stable negative)
 *   dA[m,d,n] = exp(dt[m,d] * A)  = exp(-dt[m,d] * exp(A_log[d,n]))
 *   dB[m,d,n] = dt[m,d] * B[m,n]
 *   h_new[m,d,n] = dA * h[m,d,n] + dB * u[m,d]
 *   y[m,d]    = sum_n( C[m,n] * h_new[m,d,n] ) + D_vec[d] * u[m,d]
 *
 * Dimensions: M nodes, D model dim, N state dim
 * Grid: ceil(M*D / BLOCK), Block: BLOCK
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK 256

// ── Forward ──────────────────────────────────────────────────────────────── //

__global__ void ssm_step_fwd_kernel(
    const float * __restrict__ u,       // [M, D]
    const float * __restrict__ h,       // [M, D, N]
    const float * __restrict__ dt,      // [M, D]  (after softplus)
    const float * __restrict__ B,       // [M, N]
    const float * __restrict__ C,       // [M, N]
    const float * __restrict__ A_log,   // [D, N]  log(-A)
    const float * __restrict__ D_vec,   // [D]
    float * __restrict__ h_new,         // [M, D, N]
    float * __restrict__ y,             // [M, D]
    int M, int D, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m   = idx / D;
    int d   = idx % D;
    if (m >= M) return;

    float u_val  = u[m * D + d];
    float dt_val = dt[m * D + d];
    float y_val  = D_vec[d] * u_val;

    int h_base = m * D * N + d * N;
    int B_base = m * N;
    int A_base = d * N;

    for (int n = 0; n < N; n++) {
        float neg_A   = expf(A_log[A_base + n]);          // exp(A_log) = -A > 0
        float dA      = expf(-dt_val * neg_A);             // ZOH: exp(dt * A)
        float dB      = dt_val * B[B_base + n];
        float h_val   = dA * h[h_base + n] + dB * u_val;
        h_new[h_base + n] = h_val;
        y_val += C[B_base + n] * h_val;
    }
    y[m * D + d] = y_val;
}

// ── Backward ─────────────────────────────────────────────────────────────── //

__global__ void ssm_step_bwd_kernel(
    const float * __restrict__ u,
    const float * __restrict__ h,
    const float * __restrict__ dt,
    const float * __restrict__ B,
    const float * __restrict__ C,
    const float * __restrict__ A_log,
    const float * __restrict__ D_vec,
    const float * __restrict__ h_new,
    const float * __restrict__ dh_new,  // [M, D, N]  upstream grad for h_new
    const float * __restrict__ dy,      // [M, D]     upstream grad for y
    float * __restrict__ du,
    float * __restrict__ dh,
    float * __restrict__ ddt,
    float * __restrict__ dB,            // atomicAdd
    float * __restrict__ dC,            // atomicAdd
    float * __restrict__ dA_log,        // atomicAdd
    float * __restrict__ dD_vec,        // atomicAdd
    int M, int D, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m   = idx / D;
    int d   = idx % D;
    if (m >= M) return;

    float u_val  = u[m * D + d];
    float dt_val = dt[m * D + d];
    float dy_val = dy[m * D + d];

    // skip connection: y += D_vec[d] * u  =>  du += D_vec * dy; dD += u * dy
    float du_val  = D_vec[d] * dy_val;
    float ddt_val = 0.f;
    atomicAdd(&dD_vec[d], u_val * dy_val);

    int h_base = m * D * N + d * N;
    int B_base = m * N;
    int A_base = d * N;

    for (int n = 0; n < N; n++) {
        float A_log_val  = A_log[A_base + n];
        float neg_A      = expf(A_log_val);
        float dA         = expf(-dt_val * neg_A);
        float B_val      = B[B_base + n];
        float C_val      = C[B_base + n];
        float h_old      = h[h_base + n];
        float h_new_val  = h_new[h_base + n];
        float dh_new_val = dh_new[h_base + n];

        // total upstream on h_new[m,d,n]:
        //   from next-step's h gradient (dh_new) + from y via C
        float dh_new_tot = dh_new_val + C_val * dy_val;

        // h_new = dA * h_old + dB * u
        dh[h_base + n]  = dA * dh_new_tot;
        du_val          += (dt_val * B_val) * dh_new_tot;
        ddt_val         += (-neg_A * dA * h_old + B_val * u_val) * dh_new_tot;

        // dA_log: d/d(A_log) exp(-dt * exp(A_log)) = -dt * exp(A_log) * dA
        atomicAdd(&dA_log[A_base + n], -dt_val * neg_A * dA * h_old * dh_new_tot);
        atomicAdd(&dB[B_base + n],      dt_val * u_val * dh_new_tot);
        atomicAdd(&dC[B_base + n],      h_new_val * dy_val);
    }

    du[m * D + d]  = du_val;
    ddt[m * D + d] = ddt_val;
}

// ── C++ launchers ─────────────────────────────────────────────────────────── //

std::vector<torch::Tensor> ssm_step_forward(
    torch::Tensor u,        // [M, D]
    torch::Tensor h,        // [M, D, N]
    torch::Tensor dt,       // [M, D]
    torch::Tensor B,        // [M, N]
    torch::Tensor C,        // [M, N]
    torch::Tensor A_log,    // [D, N]
    torch::Tensor D_vec     // [D]
) {
    int M = u.size(0), D = u.size(1), N = h.size(2);
    auto h_new = torch::empty_like(h);
    auto y     = torch::empty_like(u);

    int total  = M * D;
    int blocks = (total + BLOCK - 1) / BLOCK;
    ssm_step_fwd_kernel<<<blocks, BLOCK>>>(
        u.data_ptr<float>(), h.data_ptr<float>(),
        dt.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        A_log.data_ptr<float>(), D_vec.data_ptr<float>(),
        h_new.data_ptr<float>(), y.data_ptr<float>(),
        M, D, N);

    return {h_new, y};
}

std::vector<torch::Tensor> ssm_step_backward(
    torch::Tensor u, torch::Tensor h, torch::Tensor dt,
    torch::Tensor B, torch::Tensor C, torch::Tensor A_log, torch::Tensor D_vec,
    torch::Tensor h_new,
    torch::Tensor dh_new, torch::Tensor dy
) {
    int M = u.size(0), D = u.size(1), N = h.size(2);

    auto du     = torch::zeros_like(u);
    auto dh     = torch::zeros_like(h);
    auto ddt    = torch::zeros_like(dt);
    auto dB     = torch::zeros_like(B);
    auto dC     = torch::zeros_like(C);
    auto dA_log = torch::zeros_like(A_log);
    auto dD_vec = torch::zeros_like(D_vec);

    int total  = M * D;
    int blocks = (total + BLOCK - 1) / BLOCK;
    ssm_step_bwd_kernel<<<blocks, BLOCK>>>(
        u.data_ptr<float>(), h.data_ptr<float>(),
        dt.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        A_log.data_ptr<float>(), D_vec.data_ptr<float>(), h_new.data_ptr<float>(),
        dh_new.data_ptr<float>(), dy.data_ptr<float>(),
        du.data_ptr<float>(), dh.data_ptr<float>(), ddt.data_ptr<float>(),
        dB.data_ptr<float>(), dC.data_ptr<float>(),
        dA_log.data_ptr<float>(), dD_vec.data_ptr<float>(),
        M, D, N);

    return {du, dh, ddt, dB, dC, dA_log, dD_vec};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &ssm_step_forward,  "SSM step forward");
    m.def("backward", &ssm_step_backward, "SSM step backward");
}
