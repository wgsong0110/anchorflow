"""ours 가우스-뉴턴 곱 (CE·J_elᵀJ_el + J_inᵀJ_in) v 를 CUDA 커널 한 번으로 (CuPy RawKernel, NVRTC).

입자마다 국소 매개 16 개 (노드 4 × (u 3, ρ 1)) 에 대한 잔차 13 개 (√CE·wv·(F−R) 9, √CE·wv·sla·(det F−1) 1, sq·dy 3) 의
야코비안을 그 자리에서 해석적으로 구해 Jᵀ(J v) 를 노드로 흩뿌린다 -- fused_ip._local_res 와 같은 식 (smu 1, R 은 선형화 점에서 고정이라
미분에 들어가지 않는다). Warp 0.10 은 커널 안 원소 대입·삼항식이 없어 CUDA C 로 썼다.
"""
import cupy as cp
import torch

_SRC = r'''
__device__ __forceinline__ void psi_d(float r, float rho, float hl, float aa, float* o) {
    // (psi, dpsi/dr, dpsi/drho, d(dpsi/dr)/drho) -- Lattice.yJ 의 psi_raw·dpsi, 자르기 1e-6 포함
    float psi, dpsi, p_r, dp_r;
    if (r < rho) {
        float t = r / rho;
        psi = 1.f - aa * t * t;
        dpsi = -2.f * aa * r / (rho * rho);
        p_r = 2.f * aa * r * r / (rho * rho * rho);
        dp_r = 4.f * aa * r / (rho * rho * rho);
    } else {
        float hh = 0.5f * hl, q = 1.f + (r - rho) / hh;
        psi = (1.f - aa) / q;
        dpsi = -(1.f - aa) / hh / (q * q);
        p_r = (1.f - aa) / (q * q) / hh;
        dp_r = -2.f * (1.f - aa) / (hh * hh * q * q * q);
    }
    if (psi <= 1e-6f) { o[0] = 1e-6f; o[1] = 0.f; o[2] = 0.f; o[3] = 0.f; return; }
    o[0] = psi; o[1] = dpsi; o[2] = p_r; o[3] = dp_r;
}

// ρ_k 방향 한 단위에 대한 dW_m (스칼라) 와 d(dW_m) (vec3)
__device__ __forceinline__ void rho_dir(int k, const float* W, float dg[4][3], const float* S, float G,
                                         float dgj, const float* ddg, float dWs[4], float ddW[4][3]) {
    for (int m = 0; m < 4; ++m) {
        dWs[m] = ((m == k) ? dgj : 0.f) / G - W[m] * dgj / G;
        for (int c = 0; c < 3; ++c)
            ddW[m][c] = ((m == k) ? ddg[c] : 0.f) / G - dg[m][c] * dgj / (G * G)
                        - (dWs[m] * S[c] + W[m] * ddg[c]) / G + W[m] * S[c] * dgj / (G * G);
    }
}

extern "C" __global__ void gn_hv(const int* rows, const float* r, const float* dr, const float* w, const float* dw,
                                 const float* Fp, const float* wv, const float* sq, const float* th, const float* v,
                                 int M3, float hl, float aa, float sla, float sce, int P, float* out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int nd[4]; float W[4], wk[4], pr[4], dpr[4], g[4], dg[4][3], dW[4][3], u[4][3], a[4][3], drk[4][3], dwk[4][3];
    float S[3] = {0.f, 0.f, 0.f}, G = 0.f;
    for (int k = 0; k < 4; ++k) {
        nd[k] = rows[4 * p + k];
        float x = th[M3 + nd[k]];
        float sg = 1.f / (1.f + expf(-x));
        float rho = hl * (0.05f + 0.95f * sg), drho = hl * 0.95f * sg * (1.f - sg);
        float o[4]; psi_d(r[4 * p + k], rho, hl, aa, o);
        wk[k] = w[4 * p + k];
        g[k] = wk[k] * o[0]; pr[k] = o[2] * drho; dpr[k] = o[3] * drho;
        for (int c = 0; c < 3; ++c) {
            drk[k][c] = dr[(4 * p + k) * 3 + c]; dwk[k][c] = dw[(4 * p + k) * 3 + c];
            dg[k][c] = dwk[k][c] * o[0] + wk[k] * o[1] * drk[k][c];
            S[c] += dg[k][c];
            u[k][c] = th[3 * nd[k] + c];
        }
        G += g[k];
    }
    G = fmaxf(G, 1e-12f);
    float Fq[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Fq[i][j] = Fp[9 * p + 3 * i + j];
    float J[3][3] = {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
    for (int k = 0; k < 4; ++k) {
        W[k] = g[k] / G;
        for (int c = 0; c < 3; ++c) dW[k][c] = dg[k][c] / G - W[k] * S[c] / G;
    }
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) J[i][j] += u[k][i] * dW[k][j];
    float F[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) F[i][j] = J[i][0] * Fq[0][j] + J[i][1] * Fq[1][j] + J[i][2] * Fq[2][j];
    float cof[3][3];
    cof[0][0] = F[1][1] * F[2][2] - F[1][2] * F[2][1]; cof[0][1] = F[1][2] * F[2][0] - F[1][0] * F[2][2]; cof[0][2] = F[1][0] * F[2][1] - F[1][1] * F[2][0];
    cof[1][0] = F[0][2] * F[2][1] - F[0][1] * F[2][2]; cof[1][1] = F[0][0] * F[2][2] - F[0][2] * F[2][0]; cof[1][2] = F[0][1] * F[2][0] - F[0][0] * F[2][1];
    cof[2][0] = F[0][1] * F[1][2] - F[0][2] * F[1][1]; cof[2][1] = F[0][2] * F[1][0] - F[0][0] * F[1][2]; cof[2][2] = F[0][0] * F[1][1] - F[0][1] * F[1][0];
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 3; ++j) a[k][j] = Fq[0][j] * dW[k][0] + Fq[1][j] * dW[k][1] + Fq[2][j] * dW[k][2];   // Fpᵀ dW_k
    // ---- J v
    float dF[3][3] = {{0.f}}, dy[3] = {0.f, 0.f, 0.f};
    float dWs[4], ddW[4][3], ddg[3];
    for (int k = 0; k < 4; ++k) {
        float vk[3] = {v[3 * nd[k]], v[3 * nd[k] + 1], v[3 * nd[k] + 2]};
        for (int i = 0; i < 3; ++i) { dy[i] += W[k] * vk[i]; for (int j = 0; j < 3; ++j) dF[i][j] += vk[i] * a[k][j]; }
        float vr = v[M3 + nd[k]];
        if (vr != 0.f) {
            float dgj = wk[k] * pr[k];
            for (int c = 0; c < 3; ++c) ddg[c] = dwk[k][c] * pr[k] + wk[k] * dpr[k] * drk[k][c];
            rho_dir(k, W, dg, S, G, dgj, ddg, dWs, ddW);
            float dJ[3][3] = {{0.f}};
            for (int m = 0; m < 4; ++m)
                for (int i = 0; i < 3; ++i) { dy[i] += vr * dWs[m] * u[m][i]; for (int j = 0; j < 3; ++j) dJ[i][j] += u[m][i] * ddW[m][j]; }
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                dF[i][j] += vr * (dJ[i][0] * Fq[0][j] + dJ[i][1] * Fq[1][j] + dJ[i][2] * Fq[2][j]);
        }
    }
    float ce = sce * wv[p], s = sq[p];
    float e[3][3], ddet = 0.f, ein[3];
    for (int i = 0; i < 3; ++i) { ein[i] = s * dy[i]; for (int j = 0; j < 3; ++j) { e[i][j] = ce * dF[i][j]; ddet += cof[i][j] * dF[i][j]; } }
    ddet *= ce * sla;
    float cs = ce * sla;
    // ---- Jᵀ (J v)
    for (int k = 0; k < 4; ++k) {
        for (int c = 0; c < 3; ++c) {
            float ea = 0.f, ca = 0.f;
            for (int j = 0; j < 3; ++j) { ea += e[c][j] * a[k][j]; ca += cof[c][j] * a[k][j]; }
            atomicAdd(&out[3 * nd[k] + c], ce * ea + ddet * cs * ca + W[k] * s * ein[c]);
        }
        float dgj = wk[k] * pr[k];
        for (int c = 0; c < 3; ++c) ddg[c] = dwk[k][c] * pr[k] + wk[k] * dpr[k] * drk[k][c];
        rho_dir(k, W, dg, S, G, dgj, ddg, dWs, ddW);
        float dJ[3][3] = {{0.f}}, dyr[3] = {0.f, 0.f, 0.f};
        for (int m = 0; m < 4; ++m)
            for (int i = 0; i < 3; ++i) { dyr[i] += dWs[m] * u[m][i]; for (int j = 0; j < 3; ++j) dJ[i][j] += u[m][i] * ddW[m][j]; }
        float gr = 0.f;
        for (int i = 0; i < 3; ++i) {
            gr += s * ein[i] * dyr[i];
            for (int j = 0; j < 3; ++j) {
                float dFr = dJ[i][0] * Fq[0][j] + dJ[i][1] * Fq[1][j] + dJ[i][2] * Fq[2][j];
                gr += ce * e[i][j] * dFr + ddet * cs * cof[i][j] * dFr;
            }
        }
        atomicAdd(&out[M3 + nd[k]], gr);
    }
}
'''
_SRC = _SRC + '\nextern "C" __global__ void gn_block(const int* rows, const float* r, const float* dr, const float* w, const float* dw,\n                                    const float* Fp, const float* wv, const float* sq, const float* th,\n                                    int M3, float hl, float aa, float sla, float sce, int P, float* B) {\n    // 노드마다 4x4 (u_x, u_y, u_z, ρ) 가우스-뉴턴 블록 Σ_p J_kᵀ J_k 를 모은다 (B: [노드 수, 16])\n    int p = blockIdx.x * blockDim.x + threadIdx.x;\n    if (p >= P) return;\n    int nd[4]; float W[4], wk[4], pr[4], dpr[4], g[4], dg[4][3], dW[4][3], u[4][3], a[4][3], drk[4][3], dwk[4][3];\n    float S[3] = {0.f, 0.f, 0.f}, G = 0.f;\n    for (int k = 0; k < 4; ++k) {\n        nd[k] = rows[4 * p + k];\n        float sg = 1.f / (1.f + expf(-th[M3 + nd[k]]));\n        float rho = hl * (0.05f + 0.95f * sg), drho = hl * 0.95f * sg * (1.f - sg);\n        float o[4]; psi_d(r[4 * p + k], rho, hl, aa, o);\n        wk[k] = w[4 * p + k]; g[k] = wk[k] * o[0]; pr[k] = o[2] * drho; dpr[k] = o[3] * drho;\n        for (int c = 0; c < 3; ++c) {\n            drk[k][c] = dr[(4 * p + k) * 3 + c]; dwk[k][c] = dw[(4 * p + k) * 3 + c];\n            dg[k][c] = dwk[k][c] * o[0] + wk[k] * o[1] * drk[k][c]; S[c] += dg[k][c]; u[k][c] = th[3 * nd[k] + c];\n        }\n        G += g[k];\n    }\n    G = fmaxf(G, 1e-12f);\n    float Fq[3][3]; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Fq[i][j] = Fp[9 * p + 3 * i + j];\n    float J[3][3] = {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};\n    for (int k = 0; k < 4; ++k) { W[k] = g[k] / G; for (int c = 0; c < 3; ++c) dW[k][c] = dg[k][c] / G - W[k] * S[c] / G; }\n    for (int k = 0; k < 4; ++k) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) J[i][j] += u[k][i] * dW[k][j];\n    float F[3][3]; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) F[i][j] = J[i][0] * Fq[0][j] + J[i][1] * Fq[1][j] + J[i][2] * Fq[2][j];\n    float cof[3][3];\n    cof[0][0] = F[1][1]*F[2][2]-F[1][2]*F[2][1]; cof[0][1] = F[1][2]*F[2][0]-F[1][0]*F[2][2]; cof[0][2] = F[1][0]*F[2][1]-F[1][1]*F[2][0];\n    cof[1][0] = F[0][2]*F[2][1]-F[0][1]*F[2][2]; cof[1][1] = F[0][0]*F[2][2]-F[0][2]*F[2][0]; cof[1][2] = F[0][1]*F[2][0]-F[0][0]*F[2][1];\n    cof[2][0] = F[0][1]*F[1][2]-F[0][2]*F[1][1]; cof[2][1] = F[0][2]*F[1][0]-F[0][0]*F[1][2]; cof[2][2] = F[0][0]*F[1][1]-F[0][1]*F[1][0];\n    for (int k = 0; k < 4; ++k) for (int j = 0; j < 3; ++j) a[k][j] = Fq[0][j] * dW[k][0] + Fq[1][j] * dW[k][1] + Fq[2][j] * dW[k][2];\n    float ce = sce * wv[p], s = sq[p], cs = ce * sla;\n    for (int k = 0; k < 4; ++k) {\n        // 열 4 개: u_c (c=0..2), ρ_k. 각 열 = 잔차 13 개\n        float col[4][13];\n        for (int c = 0; c < 3; ++c) {\n            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) col[c][3 * i + j] = (i == c) ? ce * a[k][j] : 0.f;\n            float cd = 0.f; for (int j = 0; j < 3; ++j) cd += cof[c][j] * a[k][j];\n            col[c][9] = cs * cd;\n            for (int i = 0; i < 3; ++i) col[c][10 + i] = (i == c) ? s * W[k] : 0.f;\n        }\n        float dgj = wk[k] * pr[k], ddg[3], dWs[4], ddW[4][3];\n        for (int c = 0; c < 3; ++c) ddg[c] = dwk[k][c] * pr[k] + wk[k] * dpr[k] * drk[k][c];\n        rho_dir(k, W, dg, S, G, dgj, ddg, dWs, ddW);\n        float dJ[3][3] = {{0.f}}, dyr[3] = {0.f, 0.f, 0.f};\n        for (int m = 0; m < 4; ++m) for (int i = 0; i < 3; ++i) { dyr[i] += dWs[m] * u[m][i]; for (int j = 0; j < 3; ++j) dJ[i][j] += u[m][i] * ddW[m][j]; }\n        float dd = 0.f;\n        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {\n            float dFr = dJ[i][0] * Fq[0][j] + dJ[i][1] * Fq[1][j] + dJ[i][2] * Fq[2][j];\n            col[3][3 * i + j] = ce * dFr; dd += cof[i][j] * dFr;\n        }\n        col[3][9] = cs * dd;\n        for (int i = 0; i < 3; ++i) col[3][10 + i] = s * dyr[i];\n        for (int x = 0; x < 4; ++x) for (int y = 0; y < 4; ++y) {\n            float acc = 0.f; for (int q = 0; q < 13; ++q) acc += col[x][q] * col[y][q];\n            atomicAdd(&B[16 * nd[k] + 4 * x + y], acc);\n        }\n    }\n}\n'
_K = cp.RawKernel(_SRC, "gn_hv")
_KB = cp.RawKernel(_SRC, "gn_block")


def _c(t, dt=None):
    t = t.contiguous() if dt is None else t.to(dt).contiguous()
    return cp.asarray(t)


def prep(rows, r, dr, w, dw, Fp, wv, sq):
    """바깥 반복마다 한 번: 커널 입력을 cupy 보기로 (복사 없음, rows 만 int32 로)."""
    return (_c(rows, torch.int32), _c(r), _c(dr), _c(w), _c(dw), _c(Fp), _c(wv), _c(sq))


def hv(pre, th, v, M3, hl, aa, sla, sce):
    out = torch.zeros_like(v)
    P = pre[0].shape[0]
    blk = 128
    _K(((P + blk - 1) // blk,), (blk,), (*pre, cp.asarray(th.contiguous()), cp.asarray(v.contiguous()), cp.int32(M3),
                                       cp.float32(hl), cp.float32(aa), cp.float32(sla), cp.float32(sce), cp.int32(P),
                                       cp.asarray(out)))
    return out


_SRC_EG = r"""
__device__ __forceinline__ void psi_d2(float r, float rho, float hl, float aa, float* o) {
    float psi, dpsi, p_r, dp_r;
    if (r < rho) { float t = r / rho; psi = 1.f - aa * t * t; dpsi = -2.f * aa * r / (rho * rho);
                   p_r = 2.f * aa * r * r / (rho * rho * rho); dp_r = 4.f * aa * r / (rho * rho * rho); }
    else { float hh = 0.5f * hl, q = 1.f + (r - rho) / hh; psi = (1.f - aa) / q; dpsi = -(1.f - aa) / hh / (q * q);
           p_r = (1.f - aa) / (q * q) / hh; dp_r = -2.f * (1.f - aa) / (hh * hh * q * q * q); }
    if (psi <= 1e-6f) { o[0] = 1e-6f; o[1] = 0.f; o[2] = 0.f; o[3] = 0.f; return; }
    o[0] = psi; o[1] = dpsi; o[2] = p_r; o[3] = dp_r;
}
__device__ __forceinline__ float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}
__device__ __forceinline__ void cof3(const float F[3][3], float c[3][3]) {
    c[0][0] = F[1][1]*F[2][2]-F[1][2]*F[2][1]; c[0][1] = F[1][2]*F[2][0]-F[1][0]*F[2][2]; c[0][2] = F[1][0]*F[2][1]-F[1][1]*F[2][0];
    c[1][0] = F[0][2]*F[2][1]-F[0][1]*F[2][2]; c[1][1] = F[0][0]*F[2][2]-F[0][2]*F[2][0]; c[1][2] = F[0][1]*F[2][0]-F[0][0]*F[2][1];
    c[2][0] = F[0][1]*F[1][2]-F[0][2]*F[1][1]; c[2][1] = F[0][2]*F[1][0]-F[0][0]*F[1][2]; c[2][2] = F[0][0]*F[1][1]-F[0][1]*F[1][0];
}

// 증분 포텐셜 값(정규화)과 기울기 -- fused_ip.objective 와 같은 식. grad 가 0 이면 값만.
extern "C" __global__ void ip_eg(const int* rows, const float* r, const float* dr, const float* w, const float* dw,
        const float* Fp, const float* X, const float* xt, const float* VOL, const float* MASS, const float* th,
        int M3, float hl, float aa, float gx, float gy, float gz, float zf, int use_floor, float kfl, float hdt,
        float mu, float la, float norm, int P, int want_grad, double* val, int* bad, float* grad) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int nd[4]; float W[4], wk[4], pr[4], dpr[4], g[4], dg[4][3], dW[4][3], u[4][3], drk[4][3], dwk[4][3];
    float S[3] = {0.f, 0.f, 0.f}, G = 0.f;
    for (int k = 0; k < 4; ++k) {
        nd[k] = rows[4 * p + k];
        float sg = 1.f / (1.f + expf(-th[M3 + nd[k]]));
        float rho = hl * (0.05f + 0.95f * sg), drho = hl * 0.95f * sg * (1.f - sg);
        float o[4]; psi_d2(r[4 * p + k], rho, hl, aa, o);
        wk[k] = w[4 * p + k]; g[k] = wk[k] * o[0]; pr[k] = o[2] * drho; dpr[k] = o[3] * drho;
        for (int c = 0; c < 3; ++c) {
            drk[k][c] = dr[(4 * p + k) * 3 + c]; dwk[k][c] = dw[(4 * p + k) * 3 + c];
            dg[k][c] = dwk[k][c] * o[0] + wk[k] * o[1] * drk[k][c]; S[c] += dg[k][c]; u[k][c] = th[3 * nd[k] + c];
        }
        G += g[k];
    }
    G = fmaxf(G, 1e-12f);
    float J[3][3] = {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}, dy[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < 4; ++k) {
        W[k] = g[k] / G;
        for (int c = 0; c < 3; ++c) dW[k][c] = dg[k][c] / G - W[k] * S[c] / G;
    }
    for (int k = 0; k < 4; ++k) for (int i = 0; i < 3; ++i) { dy[i] += W[k] * u[k][i]; for (int j = 0; j < 3; ++j) J[i][j] += u[k][i] * dW[k][j]; }
    float Fq[3][3], F[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Fq[i][j] = Fp[9 * p + 3 * i + j];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) F[i][j] = J[i][0] * Fq[0][j] + J[i][1] * Fq[1][j] + J[i][2] * Fq[2][j];
    // 극분해 R: 크기 조정 Newton 8 번 (fused_ip.polar_newton 과 같다)
    float R[3][3], c[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R[i][j] = F[i][j];
    for (int it = 0; it < 8; ++it) {
        float d = det3(R); float ds = (fabsf(d) > 1e-20f) ? d : 1e-20f;
        float gm = powf(fabsf(ds), -1.f / 3.f);
        cof3(R, c);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R[i][j] = 0.5f * (gm * R[i][j] + c[i][j] / (ds * gm));
    }
    float dF = det3(F);
    // 자르기·뒤집힘 판정: S = Rᵀ F 의 대각 최소 < 0.0105 또는 det F ≤ 1e-6
    float smin = 1e30f;
    for (int i = 0; i < 3; ++i) { float sii = R[0][i] * F[0][i] + R[1][i] * F[1][i] + R[2][i] * F[2][i]; smin = fminf(smin, sii); }
    if (dF <= 1e-6f || smin < 0.0105f) atomicAdd(bad, 1);
    float e2 = 0.f;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { float t = F[i][j] - R[i][j]; e2 += t * t; }
    float psi = mu * e2 + 0.5f * la * (dF - 1.f) * (dF - 1.f);
    float x[3] = {X[3 * p] + dy[0], X[3 * p + 1] + dy[1], X[3 * p + 2] + dy[2]};
    float dd[3] = {x[0] - xt[3 * p], x[1] - xt[3 * p + 1], x[2] - xt[3 * p + 2]};
    float m = MASS[p], ih2 = 1.f / (hdt * hdt);
    float e = 0.5f * ih2 * m * (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) - m * (dd[0] * gx + dd[1] * gy + dd[2] * gz) + VOL[p] * psi;
    float pen = 0.f;
    if (use_floor) { pen = fmaxf(zf - x[2], 0.f); e += kfl * 0.5f * ih2 * m * pen * pen; }
    atomicAdd(val, (double)(e * norm));
    if (!want_grad) return;
    // ---- 기울기: dE/dF = V (2μ(F−R) + λ(J−1) cof F), dE/dx = m/h² (x−x̃) − m g + 바닥
    cof3(F, c);
    float P_[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) P_[i][j] = norm * VOL[p] * (2.f * mu * (F[i][j] - R[i][j]) + la * (dF - 1.f) * c[i][j]);
    float gx_[3] = {norm * (m * ih2 * dd[0] - m * gx), norm * (m * ih2 * dd[1] - m * gy), norm * (m * ih2 * dd[2] - m * gz)};
    if (use_floor) gx_[2] += norm * (-kfl * ih2 * m * pen);
    for (int k = 0; k < 4; ++k) {
        float a[3];
        for (int j = 0; j < 3; ++j) a[j] = Fq[0][j] * dW[k][0] + Fq[1][j] * dW[k][1] + Fq[2][j] * dW[k][2];
        for (int cc = 0; cc < 3; ++cc) {
            float pa = P_[cc][0] * a[0] + P_[cc][1] * a[1] + P_[cc][2] * a[2];
            atomicAdd(&grad[3 * nd[k] + cc], pa + W[k] * gx_[cc]);
        }
        // ρ_k
        float dgj = wk[k] * pr[k], ddg[3];
        for (int cc = 0; cc < 3; ++cc) ddg[cc] = dwk[k][cc] * pr[k] + wk[k] * dpr[k] * drk[k][cc];
        float dJ[3][3] = {{0.f}}, dyr[3] = {0.f, 0.f, 0.f};
        for (int mm = 0; mm < 4; ++mm) {
            float dWs = ((mm == k) ? dgj : 0.f) / G - W[mm] * dgj / G;
            for (int cc = 0; cc < 3; ++cc) {
                float ddW = ((mm == k) ? ddg[cc] : 0.f) / G - dg[mm][cc] * dgj / (G * G) - (dWs * S[cc] + W[mm] * ddg[cc]) / G + W[mm] * S[cc] * dgj / (G * G);
                for (int i = 0; i < 3; ++i) dJ[i][cc] += u[mm][i] * ddW;
            }
            for (int i = 0; i < 3; ++i) dyr[i] += dWs * u[mm][i];
        }
        float gr = gx_[0] * dyr[0] + gx_[1] * dyr[1] + gx_[2] * dyr[2];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
            gr += P_[i][j] * (dJ[i][0] * Fq[0][j] + dJ[i][1] * Fq[1][j] + dJ[i][2] * Fq[2][j]);
        atomicAdd(&grad[M3 + nd[k]], gr);
    }
}
"""
_KEG = cp.RawKernel(_SRC_EG, "ip_eg")


def eg(pre, X, xt, VOL, MASS, th, M3, hl, aa, g, zf, use_floor, kfl, hdt, mu, la, norm, want_grad=True):
    """(값 float, 자르기·뒤집힘 입자 수 int, 기울기 torch 또는 None) -- fused_ip.objective 와 같은 식."""
    rows, r, dr, w, dw, Fp = pre[:6]
    P = rows.shape[0]
    val = cp.zeros(1, dtype=cp.float64); bad = cp.zeros(1, dtype=cp.int32)
    grad = torch.zeros_like(th) if want_grad else torch.zeros(1, device=th.device)
    blk = 128
    _KEG(((P + blk - 1) // blk,), (blk,), (rows, r, dr, w, dw, Fp, cp.asarray(X.contiguous()), cp.asarray(xt.contiguous()),
                                         cp.asarray(VOL.contiguous()), cp.asarray(MASS.contiguous()), cp.asarray(th.contiguous()),
                                         cp.int32(M3), cp.float32(hl), cp.float32(aa), cp.float32(g[0]), cp.float32(g[1]), cp.float32(g[2]),
                                         cp.float32(zf), cp.int32(1 if use_floor else 0), cp.float32(kfl), cp.float32(hdt),
                                         cp.float32(mu), cp.float32(la), cp.float32(norm), cp.int32(P), cp.int32(1 if want_grad else 0),
                                         val, bad, cp.asarray(grad)))
    vb = cp.asnumpy(cp.concatenate([val, bad.astype(cp.float64)]))
    return float(vb[0]), int(vb[1]), (grad if want_grad else None)



def block_inv(pre, th, M3, hl, aa, sla, sce, eps):
    """노드별 4x4 가우스-뉴턴 블록 (+eps I) 의 역 [노드 수, 4, 4] -- 블록 야코비 전처리."""
    nn = M3 // 3
    B = torch.zeros(nn, 16, device=th.device)
    P = pre[0].shape[0]
    blk = 128
    _KB(((P + blk - 1) // blk,), (blk,), (*pre, cp.asarray(th.contiguous()), cp.int32(M3), cp.float32(hl), cp.float32(aa),
                                        cp.float32(sla), cp.float32(sce), cp.int32(P), cp.asarray(B)))
    B = B.reshape(nn, 4, 4) + eps * torch.eye(4, device=th.device)
    B = B + 1e-30 * torch.eye(4, device=th.device)
    return torch.linalg.inv(B.double()).float()


def block_apply(Binv, r, M3):
    """r (u 3·노드 + ρ 노드) 에 블록 역을 곱한다."""
    nn = M3 // 3
    z = torch.cat([r[:M3].reshape(nn, 3), r[M3:M3 + nn, None]], 1)
    y = (Binv @ z[..., None]).squeeze(-1)
    return torch.cat([y[:, :3].reshape(-1), y[:, 3]])
