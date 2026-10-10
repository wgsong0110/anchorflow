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
_K = cp.RawKernel(_SRC, "gn_hv")


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
