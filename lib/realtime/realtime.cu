//@@ TB_SRC 1550

extern "C" {
__device__ unsigned long long TB_T0, TB_TP, TB_TL, TB_TE;
__device__ unsigned long long TB_MIT[8], TB_MTL[8];
__device__ int TB_NEWF, TB_RUN, TB_CUT, TB_FR, TB_GATES, TB_CS;
__device__ __forceinline__ unsigned long long tb_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void tb_t0() {
    unsigned long long n = tb_now();
    if (TB_NEWF) { TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }
    TB_RUN = 0; TB_TP = n; TB_CS = 0;
}
__global__ void tb_tick(const int* CT, int op) {
    unsigned long long n = tb_now();
    if (TB_RUN) { unsigned long long d = n - TB_TP; if (d > TB_MIT[TB_FR]) TB_MIT[TB_FR] = d; }
    TB_TP = n;
    if (op == 8) { TB_TL = n; TB_RUN = 0; } else TB_RUN = (CT[0] == 0);
}
__global__ void tb_gate(int* CT, int* SKIP, unsigned long long B) {
    if (CT[0]) return;
    unsigned long long ci = 0ull, ct = 0ull;
    for (int i = 0; i < 8; i++) { if (TB_MIT[i] > ci) ci = TB_MIT[i]; if (TB_MTL[i] > ct) ct = TB_MTL[i]; }
    TB_GATES++;
    if (tb_now() - TB_T0 + ci + ct > B) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; }
}
// 예산으로 자른 걸음은 '허용치로 멈춤' 이 아니라 '반복 상한에 닿음' 과 같게 판정되도록 뉴턴 고리 끝 (ctl 8, 판정 cj_end·uf_end 앞) 에서 CT[0] 을 0 으로 되돌림
__global__ void tb_unmark(int* CT) { if (TB_CS) { CT[0] = 0; TB_CS = 0; } }
__global__ void tb_end() { unsigned long long n = tb_now(); unsigned long long d = n - TB_TL; if (TB_TL > TB_T0 && d > TB_MTL[TB_FR]) TB_MTL[TB_FR] = d; TB_TE = n; }
}

//@@ X3E_SRC 249

extern "C" {
__device__ int X3E_REDO, X3E_N;
__global__ void x3e_chk(const double* SC, const double* CD, int* WON, double T) {
    if (*WON) return;
    double g2 = SC[1], F2 = CD[2];
    if (!(g2 <= T*F2)) { *WON = 1; X3E_REDO = 1; X3E_N++; }
}
}

//@@ W3_SRC 623

extern "C" {
__global__ void s3_box(const int* IP, const int* mi, const float* mf, float* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int n1 = IP[7], n2 = IP[8], n0 = IP[6];
    int i = n/(n1*n2), j = (n/n2) % n1, k = n % n2;
    float s = 0.f;
    for (int a=-1;a<=1;a++) for (int b=-1;b<=1;b++) for (int c=-1;c<=1;c++) {
        int ii = i+a, jj = j+b, kk = k+c;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= n0 || jj >= n1 || kk >= n2) continue;
        int m = (ii*n1 + jj)*n2 + kk;
        s += mi ? (mi[m] != 0 ? 1.f : 0.f) : mf[m];
    }
    out[n] = s*(1.f/27.f);
}
}

//@@ speed4_SP_SRC 314

extern "C" {
#define SPN 65536
__device__ unsigned long long SP_T[SPN];
__device__ int SP_C[SPN];
__device__ int SP_N;
__global__ void sp_st(int c) {
    int i = SP_N;
    if (i < SPN) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); SP_T[i] = t; SP_C[i] = c; }
    SP_N = i + 1;
}
}

//@@ WS4_K 156

__global__ void ws4_chk(const int* fl) {
    if (AF_SKIP) return;
    WS4_NROUND++;
    if (!WS4_DN && fl[0] + fl[1] == 0) { WS4_DN = 1; WS4_NCLEAN++; }
}

//@@ XB_SRC 753

extern "C" {
__global__ void xb_x(const int* IP, const int* ufl, const float* ufg, float T, int* out) {      // 'x': 넘침 · C3.nfail(K 1) > 0 (같은 식, float 비교)
    out[0] = IP[13]; out[1] = (T > 0.f) ? ((ufl[0] > 0 && !(ufg[0] <= T)) ? 1 : 0) : ufl[0];
}
__global__ void xb_r(const int* IP, const double* SC, const double* CD, double T, int* out) {    // 'r': 넘침 · !(‖g‖² ≤ T F²) (반복 0 시작 잔차)
    out[2] = IP[13]; out[3] = !(SC[1] <= T*CD[2]) ? 1 : 0;
}
__global__ void xb_cp(const float* src, float* dst, const int* IP, int ci, int mul) {           // 수만큼만 복사 (IP[ci]·mul 원소)
    long long n = (long long)IP[ci]*mul;
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) dst[i] = src[i];
}
}

//@@ YL4_K 146

__global__ void yl4_chk(const int* fl) {
    if (AF_SKIP) return;
    YL4_NCHK++;
    if (!YL4_DN && fl[0] == 0) { YL4_DN = 1; YL4_NCLEAN++; }
}

//@@ speed5_SP_SRC 315

extern "C" {
#define SPN 65536
__device__ unsigned long long SP_T[SPN];
__device__ int SP_C[SPN];
__device__ int SP_N;
__global__ void sp5_st(int c) {
    int i = SP_N;
    if (i < SPN) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); SP_T[i] = t; SP_C[i] = c; }
    SP_N = i + 1;
}
}

//@@ P_SRC 3207

// ---- 절 CM72 (P) 장벽 표본점 거르기 ----
__device__ int* S7B;
__device__ int S7ON;
__device__ int S7RDY;
__device__ unsigned long long* S7VIOL;
__device__ float S7MARG;
__device__ __forceinline__ int s7key(float f) { int i = __float_as_int(f); return i >= 0 ? i : (i ^ 0x7fffffff); }
__device__ __forceinline__ float s7unkey(int k) { return __int_as_float(k >= 0 ? k : (k ^ 0x7fffffff)); }
#define S7_CNT(f_, v_) if (S7ON == 2) { atomicAdd(&S7VIOL[2], 1ULL); if (f_) { atomicAdd(&S7VIOL[1], 1ULL); if (v_) atomicAdd(&S7VIOL[0], 1ULL); } }
// sk1·sk2 = 뒤 커널이 바로 돌아가는 장치 표시 (AF_SKIP·WS4_DN) 자리 -- 켜져 있으면 덩이 표를 안 만들고 준비 표시 0
__global__ void s7_bmin(const int* IP, const float* U, const float* ya, const float* yb, float thinv, float n0, float n1, float n2, const int* sk1, const int* sk2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if ((sk1 && *sk1) || (sk2 && *sk2)) { if (n == 0) S7RDY = 0; return; }
    if (n == 0) S7RDY = 1;
    if (n >= IP[9] || IP[13]) return;
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float u0 = U[n*3], u1 = U[n*3+1], u2 = U[n*3+2];
    float ta = (u0 + thinv*(ya[n*3] - u0))*n0 + (u1 + thinv*(ya[n*3+1] - u1))*n1 + (u2 + thinv*(ya[n*3+2] - u2))*n2;
    float t = ta; bool ok = isfinite(ta);
    if (yb) { float tb = (u0 + thinv*(yb[n*3] - u0))*n0 + (u1 + thinv*(yb[n*3+1] - u1))*n1 + (u2 + thinv*(yb[n*3+2] - u2))*n2; ok = ok && isfinite(tb); t = fminf(t, tb); }
    if (!ok) t = __int_as_float(0xff800000);
    int b = ((i >> 2)*((nb1 + 3) >> 2) + (j >> 2))*((nb2 + 3) >> 2) + (k >> 2);
    atomicMin(&S7B[b], s7key(t));
}
__global__ void s7_brst(const int* IP, int cap) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b == 0) S7RDY = 0;
    int nb = ((IP[6] + 3) >> 2)*((IP[7] + 3) >> 2)*((IP[8] + 3) >> 2);
    if (b < nb && b < cap) S7B[b] = 0x7f800000;
}
// (Y) 시작점 후보 에너지 나눠 셈
__device__ int S7YI0 = 0;
__device__ int S7YI1 = 1073741824;
__device__ int S7YON = 0;
__device__ int S7YGT = 0;
__device__ int S7YK0 = 0;
__device__ int S7YM = 0;
__global__ void s7_yset(int i0, int i1, int on) { S7YI0 = i0; S7YI1 = i1; S7YON = on; }
__global__ void s7_ychk(const double* SC, const int* kk, int M, int cnt) {
    double E0 = SC[0]; int g = 0;
    for (int i=0;i<M;i++) { double E1 = SC[1+i]; if (isfinite(E1) && E1 <= E0) { g = 1; break; } }
    S7YGT = g; S7YK0 = kk[0]; S7YM = M;
    if (cnt) { atomicAdd(&S7VIOL[3], 1ULL); if (g) atomicAdd(&S7VIOL[4], 1ULL); }
}
__global__ void s7_ycmp(const int* kk) { if (S7YGT && !(kk[0] >= S7YK0 && kk[0] < S7YK0 + S7YM)) atomicAdd(&S7VIOL[5], 1ULL); }
// 점 q 의 거리 하한 > 문턱 T + 여유 이면 참 (S7RDY 0 = 앞에 s7_bmin 이 안 돌았음 -> 거짓)
__device__ __forceinline__ bool s7far(int q, const int* IP, const int* cb, float db, float T) {
    if (S7ON == 0 || S7RDY == 0 || IP[13]) return false;
    int c = q >> 3; int b0 = cb[c*3], b1 = cb[c*3+1], b2 = cb[c*3+2];
    int B1 = (IP[7] + 3) >> 2, B2 = (IP[8] + 3) >> 2;
    int m = 0x7fffffff;
    for (int x = b0 >> 2; x <= (b0 + 2) >> 2; x++) for (int y = b1 >> 2; y <= (b1 + 2) >> 2; y++) for (int z = b2 >> 2; z <= (b2 + 2) >> 2; z++) m = min(m, S7B[(x*B1 + y)*B2 + z]);
    float mn = s7unkey(m);
    return db + mn > T + S7MARG;
}

//@@ S8_HDR 2684

// ---- 절 CM74 W: 워프 모으기 (같은 열쇠 레인끼리 셔플 합 -> 대표 하나만 원자 합) ----
__device__ int S8CHK;                       // 1 = 확인 판 (그림자 원자 합)
__device__ float* S8P[24];                  // 칸 k: [3k] sh1 (원래 방식 합), [3k+1] sha (|항| 합), [3k+2] shn (항 수)
__device__ unsigned long long* S8LD;        // [2k] 레인 수, [2k+1] 대표 수
__device__ __forceinline__ unsigned s8_act(int p, int N) { int b = p & ~31; int r = N - b; return r >= 32 ? 0xffffffffu : ((1u << r) - 1u); }
__device__ __forceinline__ float s8_gsum(unsigned g, float v) { float s = 0.f; unsigned r = g; while (r) { int l = __ffs(r) - 1; r &= r - 1; s += __shfl_sync(g, v, l); } return s; }
__device__ __forceinline__ bool s8_lead(unsigned g) { return (int)(threadIdx.x & 31) == __ffs(g) - 1; }
__device__ __forceinline__ void s8_sh(int k, size_t off, float v) { if (S8CHK) { atomicAdd(S8P[3*k] + off, v); atomicAdd(S8P[3*k+1] + off, fabsf(v)); atomicAdd(S8P[3*k+2] + off, 1.f); } }
__device__ __forceinline__ void s8_ld(int k, bool l) { if (S8CHK && S8LD) { atomicAdd(S8LD + 2*k, 1ull); if (l) atomicAdd(S8LD + 2*k + 1, 1ull); } }
extern "C" __global__ void s8_pre(const float* real, float* sh1, float* sha, float* shn, long long n) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) { float v = real[i]; sh1[i] = v; sha[i] = fabsf(v); shn[i] = 1.f; }
}
// st[k*6]: 0 어김 수, 1 비교 원소 (항 2 개 이상), 2 max 정규화 차 (float 비트), 3 max |Δ|/Σ|항| (float 비트), 4 비유한 다름, 5 비트 다름 원소
extern "C" __global__ void s8_cmp(const float* real, const float* sh1, const float* sha, const float* shn, long long n, unsigned long long* st, int k) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) {
        float a = real[i], b = sh1[i], s = sha[i], c = shn[i];
        if (c < 1.5f) { if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 5, 1ull); continue; }
        atomicAdd(st + k*6 + 1, 1ull);
        if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 5, 1ull);
        if (!isfinite(a) || !isfinite(b)) { if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 4, 1ull); continue; }
        double d = fabs((double)a - (double)b), u = 5.9604644775390625e-8, bnd = 2.0 * ((double)c - 1.0) * u * (double)s;
        if (d > bnd * 1.0001 + 1e-38) atomicAdd(st + k*6, 1ull);
        float r = (float)(d / fmax(((double)c - 1.0) * u * (double)s, 1e-300)), q = (float)(d / fmax((double)s, 1e-300));
        atomicMax((unsigned long long*)(st + k*6 + 2), (unsigned long long)__float_as_uint(r));
        atomicMax((unsigned long long*)(st + k*6 + 3), (unsigned long long)__float_as_uint(q));
    }
}

//@@ HDR 2056

// ---- 절 CM76 speed9 ----
__device__ int S9DG = 0;
__device__ int S9CM = 0;
__device__ int S9YL = 0;
__device__ int S9EW = 0;
__device__ unsigned long long* S9C;
__device__ int* S9L;
__device__ unsigned int* S9N;
__device__ double* S9SC;
__device__ double* S9LG;
__device__ int* S9LN;
__device__ double S9GP = 0.0;
__device__ double S9ETA = 0.0;
__device__ double S9RF = 0.0;
__device__ double S9EWG = 0.9;
__device__ double S9EWM = 0.9;
__device__ __forceinline__ void s9_cnt(bool a, bool b) {
    unsigned ma = __ballot_sync(0xffffffffu, a), mb = __ballot_sync(0xffffffffu, b);
    if ((threadIdx.x & 31) == 0) { if (ma) atomicAdd(&S9C[0], (unsigned long long)__popc(ma)); if (mb) atomicAdd(&S9C[1], (unsigned long long)__popc(mb)); }
}
__device__ __forceinline__ void s9_push(int q, float d0, float d1, bool b) {
    unsigned m = __ballot_sync(0xffffffffu, b);
    if (!m) return;
    int l = threadIdx.x & 31; unsigned base = 0;
    if (l == 0) base = atomicAdd(S9N, (unsigned)__popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (b) { unsigned i = base + __popc(m & ((1u << l) - 1u)); S9L[(size_t)i*3] = q; S9L[(size_t)i*3+1] = __float_as_int(d0); S9L[(size_t)i*3+2] = __float_as_int(d1); }
}
__device__ __forceinline__ void s9_lg(int code, double v) {
    if (!S9LG || !S9LN) return;
    int i = atomicAdd(S9LN, 1);
    if (i < 1024) { S9LG[2*i] = (double)code; S9LG[2*i+1] = v; }
}
__device__ __forceinline__ double s9_ew(int it, double eP2, double g2) {
    if (!S9EW || !(eP2 >= 0.0)) return eP2;
    double eta;
    if (it == 0 || !(S9GP > 0.0) || !isfinite(g2)) eta = sqrt(eP2);
    else { double r = g2/S9GP; eta = S9EWG*r; double sg = S9EWG*S9ETA*S9ETA; if (sg > 0.1) eta = fmax(eta, sg); eta = fmin(eta, S9EWM); }
    S9GP = g2; S9ETA = eta;
    return eta*eta;
}
__device__ __forceinline__ int s9_ylf(float d0, float d1, float dh, float c) {
    if (!(d0 > 0.f) || !(d1 < d0)) return 0;
    int i = 0;
    for (; i<59; i++) { float d = d0 + c*(d1 - d0); if (d0 < dh ? (d == d0) : !(d < dh)) break; c *= 0.5f; }
    return i;
}

//@@ TAIL 3245

// ---- 절 CM76 speed9 커널 ----
__global__ void s9_set(int cm) { S9CM = cm; }
__global__ void s9_rst(int full) { int i = threadIdx.x; if (i == 0) *S9N = 0; if (full && i < 128) S9SC[i] = 0.0; }
__global__ void __launch_bounds__(128) s9_y0b(const float* Vq, float dh, float ks, const int* kk, double* SC, int NE) { if (AF_SKIP) return;
    unsigned n = *S9N;
    if (blockIdx.x*blockDim.x >= n) return;
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = i < n;
    int q = on ? S9L[(size_t)i*3] : 0;
    float d0 = on ? __int_as_float(S9L[(size_t)i*3+1]) : 0.f, d1 = on ? __int_as_float(S9L[(size_t)i*3+2]) : 0.f;
    double kap = on ? (double)ks*(double)Vq[q] : 0.0;
    double e = 0.0;
    if (on && d0 < dh) { if (d0 > 0.f) { double dd = (double)d0 - dh; e = -kap*dd*dd*log((double)d0/dh); } else e = 1e30; }
    blocksum_add(e, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int j=0;j<NE;j++) {
        e = 0.0;
        if (on) { float d = d0 + c*(d1 - d0);
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; } }
        blocksum_add(e, &SC[1+j]);
        c *= 0.5f;
    }
}
__device__ int s9_pick(const double* S, int NE) { double E0 = S[0]; for (int i=0;i<NE;i++) { double E1 = S[1+i]; if (isfinite(E1) && E1 <= E0) return i; } return -1; }
__global__ void s9_cmp(const double* SC, int NE) { if (AF_SKIP) return;
    unsigned n = *S9N;
    S9C[8] += 1; S9C[4] += n;
    double u = 1.1102230246251565e-16, nn = (n > 1) ? (double)(n - 1) : 0.0;
    float r1 = __uint_as_float((unsigned)S9C[13]), r2 = __uint_as_float((unsigned)S9C[14]);
    for (int k=0;k<=NE;k++) {
        double a = SC[k], b = S9SC[k], c = S9SC[64+k];
        double s1 = fmax(fabs(a), fabs(b)), s2 = fmax(fabs(a), fabs(c));
        double d1 = fabs(a - b), d2 = fabs(a - c);
        if (!(d1 <= 2.0*nn*u*s1*1.0001 + 1e-300)) S9C[9] += 1;
        if (!(d2 <= 2.0*nn*u*s2*1.0001 + 1e-300)) S9C[12] += 1;
        if (nn > 0.0 && s1 > 0.0) r1 = fmaxf(r1, (float)(d1/(nn*u*s1)));
        if (nn > 0.0 && s2 > 0.0) r2 = fmaxf(r2, (float)(d2/(nn*u*s2)));
    }
    S9C[13] = __float_as_uint(r1); S9C[14] = __float_as_uint(r2);
    int pa = s9_pick(SC, NE), pb = s9_pick(S9SC, NE), pc = s9_pick(S9SC + 64, NE);
    if (pa != pb) S9C[10] += 1;
    if (pa != pc) S9C[11] += 1;
}
__global__ void s9_cont(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* ys, float* eta, float* y) { if (AF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int p = pm[n];
    if (p == 2) return;
    float s = sk[n], isk = (s > 1e-6f) ? 1.f/s : 0.f;
    float dd[3], et[3]; for (int a=0;a<3;a++) dd[a] = ys[n*3+a] - yh[n*3+a];
    float nv[3] = {n0, n1, n2};
    if (p == 0) { for (int a=0;a<3;a++) et[a] = isk*dd[a]; }
    else if (p == 3) { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; for (int a=0;a<3;a++) et[a] = (dd[a] - dn*nv[a]) + isk*dn*nv[a]; }
    else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; for (int a=0;a<3;a++) et[a] = dn*nv[a] + isk*(dd[a] - dn*nv[a]); }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { eta[n*3+a] = et[a]; y[n*3+a] = yh[n*3+a] + ty[a]; }
}

//@@ _TAIL 599

// 절 CM56 비트 2: 이 시도에서 예산으로 자른 걸음이면 --ue_loc 노드 집합을 몸 전체로 (ul_st 앞; 갱신 커널이 덜 풀린 걸음 (cjl) 일 때만 집합을 씀)
__global__ void tc_all(const int* IP, int* m) {
    if (!TC_CK) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) atomicAdd(&TC_NALL, 1);
    if (n < IP[9]) m[n] = 1;
}
// 진단 (결과 불변): rt_ctl op 3 뒤 반복 it 의 시작 ‖g‖² (SC[1]), 힘 척도 F² (CD[2]), 멈춤 표시 (CT[0]) -- 시도마다 덮어씀 (마지막 시도가 남음)
__global__ void tc_rec(const double* SC, const double* CD, const int* CT, int it) {
    if (it >= TC_NIT) return;
    TC_IT[3*it] = SC[1]; TC_IT[3*it+1] = CD[2]; TC_IT[3*it+2] = (double)CT[0];
}

