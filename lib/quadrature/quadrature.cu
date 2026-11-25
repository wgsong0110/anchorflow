//@@ SRC 4809

// ---- 절 CM194 모래시계 억제 벌점 (fusedgs/fem_gq1.py): E = ½ kg V_q ‖Σ_n y_n ⊗ g_n‖², g_n = F_c^0ᵀ (b_g,n − b_c,n) ----
__device__ __forceinline__ void hg_g(const float* F, const float* B8, const float* Bc, int oc, int o, float* g) {
    float d0 = B8[(oc*27+o)*3] - Bc[o*3], d1 = B8[(oc*27+o)*3+1] - Bc[o*3+1], d2 = B8[(oc*27+o)*3+2] - Bc[o*3+2];
    g[0] = F[0]*d0 + F[3]*d1 + F[6]*d2; g[1] = F[1]*d0 + F[4]*d1 + F[7]*d2; g[2] = F[2]*d0 + F[5]*d1 + F[8]*d2;
}
__global__ void __launch_bounds__(128) hg_force(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg,
                                                const float* y, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float w = on ? kg*Vq[qq] : 0.f;
    float H[9] = {0,0,0,0,0,0,0,0,0};
    if (f || E) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = on ? y[id*3+a] : 0.f; H[a*3] += ya*g[0]; H[a*3+1] += ya*g[1]; H[a*3+2] += ya*g[2]; } }
    if (E) { double e = 0.0; for (int k=0;k<9;k++) e += (double)H[k]*(double)H[k]; blocksum_add(0.5*(double)w*e, E); }
    if (f) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        for (int a=0;a<3;a++) { double s = seg8d((double)(w*(H[a*3]*g[0] + H[a*3+1]*g[1] + H[a*3+2]*g[2]))); if (on && head) atomicAdd(&f[ids[o]*3+a], s); } }
    if (B) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        float s = seg8f(w*(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])); if (on && head) for (int a=0;a<3;a++) atomicAdd(&B[ids[o]*9+a*4], s); }
}
__global__ void __launch_bounds__(128) hg_mv(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg, const float* v, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float w = on ? kg*Vq[qq] : 0.f;
    float H[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        for (int a=0;a<3;a++) { float va = on ? v[id*3+a] : 0.f; H[a*3] += va*g[0]; H[a*3+1] += va*g[1]; H[a*3+2] += va*g[2]; } }
    for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        for (int a=0;a<3;a++) { float s = seg8f(w*(H[a*3]*g[0] + H[a*3+1]*g[1] + H[a*3+2]*g[2])); if (on && head) atomicAdd(&out[ids[o]*3+a], s); } }
}
// 선탐색 후보 c_i = 2^-i (i0 ≤ i < NC): y_c = ŷ + T(η + c_i δ) -> SC[40+i] += E_hg(y_c)  (rt_elem_lsN 과 같은 후보·같은 T; T 선형이라 H(c) = H0 + c Hd)
__global__ void __launch_bounds__(128) hg_ls(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg,
                                             const float* yh, const float* eta, const float* dl, const float* sk, const int* pm, float n0, float n1, float n2,
                                             double* SC, int NC, int i0, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    double w = on ? (double)kg*(double)Vq[qq] : 0.0;
    double H0[9] = {0,0,0,0,0,0,0,0,0}, Hd[9] = {0,0,0,0,0,0,0,0,0};
    if (on) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        float t0[3], td[3], e0[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, ed[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]};
        Tapply(sk[id], pm[id], n0, n1, n2, e0, t0); Tapply(sk[id], pm[id], n0, n1, n2, ed, td);
        for (int a=0;a<3;a++) { double ya = (double)yh[id*3+a] + (double)t0[a], da = (double)td[a];
            for (int b=0;b<3;b++) { H0[a*3+b] += ya*(double)g[b]; Hd[a*3+b] += da*(double)g[b]; } } }
    double c = 1.0; for (int i=0;i<i0;i++) c *= 0.5;
    for (int i=i0;i<NC;i++) {
        double e = 0.0; for (int k=0;k<9;k++) { double h = H0[k] + c*Hd[k]; e += h*h; }
        blocksum_add(0.5*w*e, &SC[40+i]);
        c *= 0.5;
    }
}

//@@ HM_SRC 1613

extern "C" {
__device__ __forceinline__ double hm_seg8(double s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double hm_psi(const double* G, double r) {
    double t = G[0] + G[4] + G[8], s = 0.0;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { double e = 0.5*(G[a*3+b] + G[b*3+a]); s += e*e; }
    return s + r*t*t;
}
// 점유 칸마다 8 레인 (칸 c = t >> 3, 점 g = t & 7): G_g = Σ_n U_n ⊗ b_g,n (월드). out[0] += Σ ψ(G_g − Ḡ), out[1] += Σ ψ(G_g), out[2] += 칸 수
__global__ void hm_cell(const int* IP, const int* cb, const float* U, const float* B8, double r, double* out) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int nc = IP[11];
    if (blockIdx.x*blockDim.x >= nc*8) return;
    bool on = (t < nc*8) && !IP[13];
    int c = on ? (t >> 3) : 0, g = t & 7;
    int nb1 = IP[7], nb2 = IP[8];
    double G[9] = {0,0,0,0,0,0,0,0,0};
    if (on) for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; int id = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        const float* b = B8 + (g*27+o)*3;
        for (int a=0;a<3;a++) { double ua = (double)U[id*3+a]; G[a*3] += ua*b[0]; G[a*3+1] += ua*b[1]; G[a*3+2] += ua*b[2]; } }
    double D[9]; for (int k=0;k<9;k++) D[k] = G[k] - 0.125*hm_seg8(G[k]);
    double eh = on ? hm_psi(D, r) : 0.0, et = on ? hm_psi(G, r) : 0.0;
    if (!(eh == eh) || !(et == et)) { eh = 0.0; et = 0.0; if (on && g == 0) atomicAdd(&out[3], 1.0); }
    atomicAdd(&out[0], eh); atomicAdd(&out[1], et);
    if (on && g == 0) atomicAdd(&out[2], 1.0);
}
}

//@@ q27_Q27_HELP 1076

// ---- 절 CM101 q27: 칸마다 가우스 3x3x3 (fusedgs/fem_q27.py). 점 o = 9 i + 3 j + k (o < 27), 칸당 자리 32 (27..31 빈 자리: 무게 0)
__device__ __constant__ float Q27D[3] = {-0.38729833462074170f, 0.f, 0.38729833462074170f};
__device__ __constant__ float Q27W[32] = {%s};
__device__ __forceinline__ int q27_i(float f) { float u = f - 0.5f; return (u < 0.27777777777777778f) ? 0 : ((u < 0.72222222222222222f) ? 1 : 2); }
__device__ __forceinline__ int q27_oc(float f0, float f1, float f2) { return q27_i(f0)*9 + q27_i(f1)*3 + q27_i(f2); }
__device__ __forceinline__ float q27_d(int oc, int a) { if (oc >= 27) return 0.f; int i = (a == 0) ? oc/9 : ((a == 1) ? (oc/3)%%3 : oc%%3); return Q27D[i]; }
// 보간 교체 축 몫: 이웃한 두 가우스 점 사이 선형 (u < ½ 이면 점 0–1, 아니면 1–2), 바깥은 끝 점 값 (2x2x2 판의 같은 규칙)
__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { float u = f - 0.5f; const float x0 = 0.11270166537925831f, x2 = 0.88729833462074169f;
    if (u < 0.5f) { *lo = 0; *t = fminf(fmaxf((u - x0)/(0.5f - x0), 0.f), 1.f); } else { *lo = 1; *t = fminf(fmaxf((u - 0.5f)/(x2 - 0.5f), 0.f), 1.f); } }

//@@ q27_Q1_AX 103
__device__ __forceinline__ int q27_i(float f) { return 1; }   // 절 CM179 1x1x1: 모든 입자 -> 가운데 점 (o = 13)
//@@ q27_Q1_AX2 125
__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { *lo = 1; *t = 0.f; }   // 절 CM179 1x1x1: 보간 = 가운데 점 값 하나
//@@ q27b_Q27_HELP 627

// ---- 절 CM183 q27b (fusedgs/fem_q27b.py): 칸마다 가우스 NP x NP x NP. 점 o = NP*NP*i + NP*j + k (o < NP^3), 칸당 자리 SLOTS (나머지 빈 자리: 무게 0)
__device__ __constant__ float Q27D[%(NP)d] = {%(D)s};
__device__ __constant__ float Q27W[%(S)d] = {%(W)s};
%(QI)s
__device__ __forceinline__ int q27_oc(float f0, float f1, float f2) { return q27_i(f0)*%(NP2)d + q27_i(f1)*%(NP)d + q27_i(f2); }
__device__ __forceinline__ float q27_d(int oc, int a) { if (oc >= %(NPT)d) return 0.f; int i = (a == 0) ? oc/%(NP2)d : ((a == 1) ? (oc/%(NP)d)%%%(NP)d : oc%%%(NP)d); return Q27D[i]; }
// 보간 교체 축 몫: 이웃한 두 가우스 점 사이 선형, 바깥은 끝 점 값 (2x2x2 판의 같은 규칙)
%(QX)s

//@@ _QI3 147
__device__ __forceinline__ int q27_i(float f) { float u = f - 0.5f; return (u < 0.27777777777777778f) ? 0 : ((u < 0.72222222222222222f) ? 1 : 2); }
//@@ _QX3 309
__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { float u = f - 0.5f; const float x0 = 0.11270166537925831f, x2 = 0.88729833462074169f;
    if (u < 0.5f) { *lo = 0; *t = fminf(fmaxf((u - x0)/(0.5f - x0), 0.f), 1.f); } else { *lo = 1; *t = fminf(fmaxf((u - 0.5f)/(x2 - 0.5f), 0.f), 1.f); } }
//@@ q27b_Q1_AX 103
__device__ __forceinline__ int q27_i(float f) { return 1; }   // 절 CM179 1x1x1: 모든 입자 -> 가운데 점 (o = 13)
//@@ q27b_Q1_AX2 125
__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { *lo = 1; *t = 0.f; }   // 절 CM179 1x1x1: 보간 = 가운데 점 값 하나
//@@ FI_SRC 1275

// ---- 절 CM92 finterp: 보간 교체의 부피 진단 + 부피 보존 보간 (모드 1 J 산술, 모드 2 J 증분) -- fpi_p 앞에 넣음 ----
__device__ __forceinline__ float fi_det(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }
// 프레임 시작 (rt_gauss 뒤): 가우스 점 J_q0, 입자 J_p0
__global__ void fi_jg(const int* IP, const float* Fq, float* Jq0) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    Jq0[q] = fi_det(Fq + (size_t)q*9);
}
__global__ void fi_jp(int N, const float* F, float* Jp0) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    Jp0[p] = fi_det(F + (size_t)p*9);
}
// 커밋 앞: J(A F_q) (ao_commit·aq_commit·fpi_q 와 같은 X)
__global__ void fi_jx(const int* IP, const int* cb, const float* BG, const float* Fq, const float* U, float* JqX) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9], X[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; X[k] = F[k]; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    JqX[q] = fi_det(X);
}

