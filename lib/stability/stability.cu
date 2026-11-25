//@@ ghost1_GP_SRC 6757

// ---- 절 CM47 ghost penalty (fusedgs/fem_ghost1.py) ----
__device__ __forceinline__ float gp_m(int a, int b) { const float M[9] = {6.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 54.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 6.f/120.f}; return M[a*3+b]; }
__device__ __forceinline__ float gp_cl(int l) { return (l == 0) ? -1.f : ((l == 1) ? 3.f : ((l == 2) ? -3.f : 1.f)); }
__device__ __forceinline__ void gp_ids(const int* IP, const int* cb, int fc, int* ids) {   // 면 fc = c*3 + 축 -> 36 노드 (층 l 0..3 x 가로 i, j 0..2)
    int c = fc / 3, a = fc % 3, t1 = (a == 0) ? 1 : 0, t2 = (a == 2) ? 1 : 2;
    int nb1 = IP[7], nb2 = IP[8];
    for (int l=0;l<4;l++) for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += l; x[t1] += i; x[t2] += j;
        ids[l*9 + i*3 + j] = (x[0]*nb1 + x[1])*nb2 + x[2];
    }
}
__device__ __forceinline__ void gp_D(const int* ids, const float* u, float D[9][3]) {
    for (int t=0;t<9;t++) for (int a=0;a<3;a++) { float s = 0.f; for (int l=0;l<4;l++) s += gp_cl(l)*u[ids[l*9+t]*3+a]; D[t][a] = s; }
}
__device__ __forceinline__ void gp_MD(float D[9][3], float MD[9][3]) {
    for (int t=0;t<9;t++) for (int a=0;a<3;a++) { float s = 0.f; for (int t2=0;t2<9;t2++) s += gp_m(t/3, t2/3)*gp_m(t%3, t2%3)*D[t2][a]; MD[t][a] = s; }
}
__global__ void gp_zero(int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    cnt[0] = 0; cnt[1] = 0;
}
__global__ void gp_cut(const int* IP, const int* cb, const int* cidx, int* cut, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) k = 1;
    }
    cut[c] = k;
    if (k) atomicAdd(&cnt[0], 1);
}
__global__ void gp_face(const int* IP, const int* cb, const int* cidx, const int* cut, int* cnt, int* fl) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int c = i / 3, a = i % 3;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
    if (x[a] >= n[a]) return;
    int c2 = cidx[(x[0]*n[1] + x[1])*n[2] + x[2]];
    if (c2 < 0 || !(cut[c] || cut[c2])) return;
    int s = atomicAdd(&cnt[1], 1);
    fl[s] = i;
}
// f += k G y (f64), B 대각 += k c_l² M_tt (블록 I3), E += ½ k yᵀ G y
__global__ void gp_force(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* y, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float D[9][3], MD[9][3]; gp_D(ids, y, D); gp_MD(D, MD);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) e += 0.5*(double)k*(double)D[t][a]*(double)MD[t][a];
        if (f) for (int l=0;l<4;l++) for (int t=0;t<9;t++) for (int a=0;a<3;a++) atomicAdd(&f[ids[l*9+t]*3+a], (double)(k*gp_cl(l)*MD[t][a]));
        if (B) for (int l=0;l<4;l++) for (int t=0;t<9;t++) { float v = k*gp_cl(l)*gp_cl(l)*gp_m(t/3, t/3)*gp_m(t%3, t%3); int id = ids[l*9+t];
            atomicAdd(&B[id*9], v); atomicAdd(&B[id*9+4], v); atomicAdd(&B[id*9+8], v); }
    }
    if (E) blocksum_add(e, E);
}
// out += k G dy (f32)
__global__ void gp_mv(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* dy, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= cnt[1]) return;
    int ids[36]; gp_ids(IP, cb, fl[s], ids);
    float D[9][3], MD[9][3]; gp_D(ids, dy, D); gp_MD(D, MD);
    for (int l=0;l<4;l++) for (int t=0;t<9;t++) for (int a=0;a<3;a++) atomicAdd(&out[ids[l*9+t]*3+a], k*gp_cl(l)*MD[t][a]);
}
// 선탐색 후보 y_c = ŷ + T(η + c δ) = a + c b (T 선형) -> Q_aa·Q_ab·Q_bb (GQ 0..2)
__global__ void gp_ls(const int* IP, const int* cb, const int* cnt, const int* fl, const float* yh, const float* eta, const float* dl, const float* sk, const int* pm,
                      float n0, float n1, float n2, double* GQ, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double qa = 0.0, qab = 0.0, qb = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float Da[9][3], Db[9][3];
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) { Da[t][a] = 0.f; Db[t][a] = 0.f; }
        for (int l=0;l<4;l++) for (int t=0;t<9;t++) { int id = ids[l*9+t];
            float e[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, d[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]}, te[3], td[3];
            Tapply(sk[id], pm[id], n0, n1, n2, e, te); Tapply(sk[id], pm[id], n0, n1, n2, d, td);
            for (int a=0;a<3;a++) { Da[t][a] += gp_cl(l)*(yh[id*3+a] + te[a]); Db[t][a] += gp_cl(l)*td[a]; } }
        float Ma[9][3], Mb[9][3]; gp_MD(Da, Ma); gp_MD(Db, Mb);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) { qa += (double)Da[t][a]*Ma[t][a]; qab += (double)Da[t][a]*Mb[t][a]; qb += (double)Db[t][a]*Mb[t][a]; }
    }
    blocksum_add(qa, &GQ[0]); blocksum_add(qab, &GQ[1]); blocksum_add(qb, &GQ[2]);
}
__global__ void gp_lsfin(double* SC, int NC, int i0, const double* gate, double* GQ, float k) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) { double cd = (double)c; SC[40+i] += 0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[2]); c *= 0.5f; }
    GQ[0] = 0.0; GQ[1] = 0.0; GQ[2] = 0.0;
}
// 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀ G y (en_phi_q 와 같은 MO 문)
__global__ void gp_phi(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* y, const int* MO, double* PH) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (!MO[0]) return;
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float D[9][3], MD[9][3]; gp_D(ids, y, D); gp_MD(D, MD);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) e += 0.5*(double)k*(double)D[t][a]*(double)MD[t][a];
    }
    blocksum_add(e, &PH[1]);
}

//@@ GQ_SRC 22064

// ---- 절 CM53 ghost4 칸별 커널 (fusedgs/fem_ghost4.py; 절 CM51 ghost3 커널 + 면 한정 방식 2·3) ----
__constant__ float GQ_M[9] = {6.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 54.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 6.f/120.f};
// 축 a, 층 l (0..2 = 칸 노드), 가로 (i, j) -> 칸 노드 번호 (칸 기준 오프셋 (o0, o1, o2) -> o0*9 + o1*3 + o2); t1 = (a==0)?1:0, t2 = (a==2)?1:2 (gp_ids 와 같은 배치)
__device__ __forceinline__ int gq_cn(int a, int l, int i, int j) { return (a == 0) ? (l*9 + i*3 + j) : ((a == 1) ? (i*9 + l*3 + j) : (i*9 + j*3 + l)); }
// 바깥 층 (l = 3) 노드 id: base + 3 s_a + i s_t1 + j s_t2
__device__ __forceinline__ int gq_en(int base, int a, int i, int j, int s0, int s1) {
    return (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
}
// MD = M D Mᵀ (3x3, M 대칭)
__device__ __forceinline__ void gq_MDM(const float D[9], float MD[9]) {
    float T[9];
#pragma unroll
    for (int i=0;i<3;i++)
#pragma unroll
        for (int j=0;j<3;j++) T[i*3+j] = GQ_M[i*3]*D[j] + GQ_M[i*3+1]*D[3+j] + GQ_M[i*3+2]*D[6+j];
#pragma unroll
    for (int i=0;i<3;i++)
#pragma unroll
        for (int j=0;j<3;j++) MD[i*3+j] = T[i*3]*GQ_M[j*3] + T[i*3+1]*GQ_M[j*3+1] + T[i*3+2]*GQ_M[j*3+2];
}
__device__ __forceinline__ int gq_base(const int* IP, const int* cb, int c, int* s0, int* s1) {
    int nb1 = IP[7], nb2 = IP[8]; *s1 = nb2; *s0 = nb1*nb2;
    return (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
}
// ---- 절 CM55 ghost5: 칸 점유 φ_c = min(1, 칸 입자 부피 합 / h³) (rt_gauss 가 읽는 칸 누적 C[c*30+28] = 그 칸 가우스 점 부피 Vq 합, h³ = |det A|), 칸 밀도 ρ_c = C[29]/C[28]
__global__ void gq5_cell(const int* IP, const float* C, const float* Am, float* phi, float* rhoc) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    float det = Am[0]*(Am[4]*Am[8] - Am[5]*Am[7]) - Am[1]*(Am[3]*Am[8] - Am[5]*Am[6]) + Am[2]*(Am[3]*Am[7] - Am[4]*Am[6]);
    float V = C[(size_t)c*30 + 28];
    phi[c] = fminf(V / fabsf(det), 1.f);
    rhoc[c] = (V > 0.f) ? C[(size_t)c*30 + 29] / V : 0.f;
}
// 면 가중 fw[c*3+a] (면 = 칸 c 와 +a 이웃 c2, 표지 비트가 선 면만): 1 = φ_max (1 − φ_min), 2 = Frei 외 2024 Σ_K ½ w_max^{1−2κ_K}, 3 = 1 − ½(φ_a + φ_b), 4 = φ_max, 5 = 1 − φ_min, 6 = 1 − φ_max, 그 밖 = 1
__global__ void gq5_w(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const float* phi, int mode, float wmax, float* fw) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int m = fm[c];
    for (int a=0;a<3;a++) {
        float w = 0.f;
        if (m & (1 << a)) {
            int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
            int c2 = (x[a] < n[a]) ? cidx[(x[0]*n[1] + x[1])*n[2] + x[2]] : -1;
            float pa = phi[c], pb = (c2 >= 0) ? phi[c2] : 0.f, lo = fminf(pa, pb), hi = fmaxf(pa, pb);
            if (mode == 1) w = hi*(1.f - lo);
            else if (mode == 2) w = 0.5f*powf(wmax, 1.f - 2.f*pa) + 0.5f*powf(wmax, 1.f - 2.f*pb);
            else if (mode == 3) w = 1.f - 0.5f*(pa + pb);
            else if (mode == 4) w = hi;
            else if (mode == 5) w = 1.f - lo;
            else if (mode == 6) w = 1.f - hi;
            else w = 1.f;
        }
        fw[c*3+a] = w;
    }
}
// ---- 면 표지 (프레임마다 rt_gauss 뒤): 경계 칸 -> 두꺼운 칸 -> 칸별 3 비트 면 표지; cnt = [경계 칸, 면 (빼기 뒤), 면 (빼기 앞), 두꺼운 칸]
__global__ void gq_zero(int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    cnt[0] = 0; cnt[1] = 0; cnt[2] = 0; cnt[3] = 0;
}
__global__ void gq_cut(const int* IP, const int* cb, const int* cidx, unsigned char* cut, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) k = 1;
    }
    cut[c] = (unsigned char)k;
    if (k) atomicAdd(&cnt[0], 1);
}
// 두꺼운 칸 (비트 0): thin 0·3 = 모두; thin 1 = 닫힌 27 이웃에 내부 칸 (활성·경계 아님) 이 있음 (3x3x3 열림);
// thin 2 = 이 커널은 비트 1 (5x5x5 내부 칸 = 체비셰프 반경 2 안 124 칸이 모두 활성) 만 쓰고, gq_thk2 가 비트 0 = 반경 2 안에 5x5x5 내부 칸이 있음 (5x5x5 열림)
__global__ void gq_thk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk, int thin, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int k = 0;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2];
    if (thin == 0 || thin == 3) k = 1;
    else if (thin == 1 || thin == 4 || thin == 5) {                     // 절 CM55: 방식 4·5 도 3x3x3 두꺼운 칸 (비트 0) 에서 시작
        for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
            int X = x+dx, Y = y+dy, Z = z+dz;
            if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
            int c2 = cidx[(X*n1 + Y)*n2 + Z];
            if (c2 >= 0 && !cut[c2]) k = 1;
        }
    } else {
        int in = 1;
        for (int dx=-2;dx<=2 && in;dx++) for (int dy=-2;dy<=2 && in;dy++) for (int dz=-2;dz<=2 && in;dz++) {
            int X = x+dx, Y = y+dy, Z = z+dz;
            if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) in = 0;
        }
        thk[c] = (unsigned char)(in << 1);
        return;
    }
    thk[c] = (unsigned char)k;
    if (k && cnt) atomicAdd(&cnt[3], 1);
}
__global__ void gq_thk2(const int* IP, const int* cb, const int* cidx, unsigned char* thk, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-2;dx<=2 && !k;dx++) for (int dy=-2;dy<=2 && !k;dy++) for (int dz=-2;dz<=2 && !k;dz++) {
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
        int c2 = cidx[(X*n1 + Y)*n2 + Z];
        if (c2 >= 0 && (thk[c2] & 2)) k = 1;
    }
    thk[c] = (unsigned char)((thk[c] & 2) | k);
    if (k && cnt) atomicAdd(&cnt[3], 1);
}
// 절 CM55 방식 4: 비트 2 = 닫힌 27 이웃에 두꺼운 칸 (비트 0) 이 있음 (= 두꺼운 칸과 이차 B-스플라인 기저 (노드) 를 공유하는 칸; 3x3x3 열림의 3x3x3 팽창)
__global__ void gq_dil(const int* IP, const int* cb, const int* cidx, unsigned char* thk) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
        int c2 = cidx[(X*n1 + Y)*n2 + Z];
        if (c2 >= 0 && (thk[c2] & 1)) k = 1;
    }
    if (k) thk[c] |= 4;                                                        // 비트 0 은 다른 스레드가 읽으므로 비트 2 만 더함 (읽는 비트와 겹치지 않음)
}
// 벌점 면: 절 CM47 면 (축 이웃 두 칸 활성, 하나 이상 경계 칸) 중 thin 3 = 한쪽 내부 칸·한쪽 경계 칸 (cut 다름), 그 밖 = 두 칸 모두 두꺼움 (비트 0)
__global__ void gq_face(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, const unsigned char* thk, unsigned char* fm, int thin, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int m = 0, na = 0, nk = 0;
    for (int a=0;a<3;a++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
        if (x[a] >= n[a]) continue;
        int c2 = cidx[(x[0]*n[1] + x[1])*n[2] + x[2]];
        if (c2 < 0 || !(cut[c] || cut[c2])) continue;
        na++;
        bool on = (thin == 3) ? (cut[c] != cut[c2]) : ((thin == 4) ? ((thk[c] & 4) && (thk[c2] & 4)) : ((thin == 5) ? ((thk[c] & 1) || (thk[c2] & 1)) : ((thk[c] & 1) && (thk[c2] & 1))));
        if (on) { m |= (1 << a); nk++; }   // 절 CM55 방식 4 = 두 칸 모두 두꺼운 칸과 기저 공유, 5 = 한쪽 이상 두꺼운 칸
    }
    fm[c] = (unsigned char)m;
    if (cnt) { if (nk) atomicAdd(&cnt[1], nk); if (na) atomicAdd(&cnt[2], na); }
}
// ---- 구속 노드에 닿는 면 빼기 (FIXX; rt_bc_box 뒤): 면 표지 비트 a 의 면 36 노드 (칸 27 노드 중 그 축 층 0..2 + 바깥 층 9) 중 fixd 가 있으면 비트를 지움, cnt[1] 에서 뺌
__global__ void gq_fix(const int* IP, const int* cb, const unsigned char* fixd, unsigned char* fm, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    int cf = 0;
    for (int o=0;o<27;o++) if (fixd[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)]) { cf = 1; break; }
    int m2 = m, nr = 0;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        int f = cf;
        for (int i=0;i<3 && !f;i++) for (int j=0;j<3 && !f;j++) if (fixd[gq_en(base, a, i, j, s0, s1)]) f = 1;
        if (f) { m2 &= ~(1 << a); nr++; }
    }
    if (m2 != m) { fm[c] = (unsigned char)m2; if (cnt) atomicSub(&cnt[1], nr); }
}
// ---- 곱: out += kk G dy (f32), kk = k + k_M/τ² (걸음 안) 또는 k
__global__ void __launch_bounds__(128) gq_mv(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float kk, const float* dy, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
#pragma unroll 1
    for (int q=0;q<3;q++) {
        float u[27], acc[27];
#pragma unroll
        for (int o=0;o<27;o++) { u[o] = dy[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q]; acc[o] = 0.f; }
#pragma unroll
        for (int a=0;a<3;a++) {
            if (!(m & (1 << a))) continue;
            float wf = fw ? fw[c*3+a] : 1.f;                                   // 절 CM55 면 가중
            float D[9], MD[9];
#pragma unroll
            for (int i=0;i<3;i++)
#pragma unroll
                for (int j=0;j<3;j++) {
                    float ue = dy[gq_en(base, a, i, j, s0, s1)*3 + q];
                    D[i*3+j] = -u[gq_cn(a,0,i,j)] + 3.f*u[gq_cn(a,1,i,j)] - 3.f*u[gq_cn(a,2,i,j)] + ue;
                }
            gq_MDM(D, MD);
#pragma unroll
            for (int i=0;i<3;i++)
#pragma unroll
                for (int j=0;j<3;j++) {
                    float v = kk*wf*MD[i*3+j];
                    acc[gq_cn(a,0,i,j)] -= v; acc[gq_cn(a,1,i,j)] += 3.f*v; acc[gq_cn(a,2,i,j)] -= 3.f*v;
                    atomicAdd(&out[gq_en(base, a, i, j, s0, s1)*3 + q], v);
                }
        }
#pragma unroll
        for (int o=0;o<27;o++) if (acc[o] != 0.f) atomicAdd(&out[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q], acc[o]);
    }
}
// ---- 힘·블록 대각·에너지: f += G (k y + cm (y − ŷ)) (f64), B 대각 += (k + cm) c_l² M_ii M_jj, E += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ)  (yh 없으면 cm 몫 없음)
__global__ void __launch_bounds__(128) gq_force(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[11]) return;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    int m = (!IP[13] && c < IP[11]) ? fm[c] : 0;
    if (m) {
        int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
        bool hm = (yh != 0) && (cm != 0.f);
        float kk = k + (hm ? cm : 0.f);
        if (f || E) {
#pragma unroll 1
        for (int q=0;q<3;q++) {
            float u[27], w[27], acc[27];
#pragma unroll
            for (int o=0;o<27;o++) { int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3); float yy = y[id*3+q]; u[o] = yy + (y0 ? y0[id*3+q] : 0.f); w[o] = hm ? (yy - yh[id*3+q]) : 0.f; acc[o] = 0.f; }   // 절 CM57: 강성 몫 = y + u0 (보존 판), 질량 몫 = y − ŷ
#pragma unroll
            for (int a=0;a<3;a++) {
                if (!(m & (1 << a))) continue;
                float wf = fw ? fw[c*3+a] : 1.f;                               // 절 CM55 면 가중
                float Dy[9], Dd[9], Dz[9], MD[9];
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        int ie = gq_en(base, a, i, j, s0, s1);
                        float ye = y[ie*3 + q], ue = ye + (y0 ? y0[ie*3 + q] : 0.f), we = hm ? (ye - yh[ie*3 + q]) : 0.f;
                        Dy[i*3+j] = -u[gq_cn(a,0,i,j)] + 3.f*u[gq_cn(a,1,i,j)] - 3.f*u[gq_cn(a,2,i,j)] + ue;
                        Dd[i*3+j] = -w[gq_cn(a,0,i,j)] + 3.f*w[gq_cn(a,1,i,j)] - 3.f*w[gq_cn(a,2,i,j)] + we;
                        Dz[i*3+j] = wf*(k*Dy[i*3+j] + (hm ? cm*Dd[i*3+j] : 0.f));
                    }
                if (E) {
                    gq_MDM(Dy, MD); float s = 0.f;
#pragma unroll
                    for (int t=0;t<9;t++) s += Dy[t]*MD[t];
                    e += 0.5*(double)k*(double)wf*(double)s;
                    if (hm) { gq_MDM(Dd, MD); s = 0.f;
#pragma unroll
                        for (int t=0;t<9;t++) s += Dd[t]*MD[t];
                        e += 0.5*(double)cm*(double)wf*(double)s; }
                }
                if (f) {
                    gq_MDM(Dz, MD);
#pragma unroll
                    for (int i=0;i<3;i++)
#pragma unroll
                        for (int j=0;j<3;j++) {
                            float v = MD[i*3+j];
                            acc[gq_cn(a,0,i,j)] -= v; acc[gq_cn(a,1,i,j)] += 3.f*v; acc[gq_cn(a,2,i,j)] -= 3.f*v;
                            atomicAdd(&f[gq_en(base, a, i, j, s0, s1)*3 + q], (double)v);
                        }
                }
            }
            if (f) {
#pragma unroll
                for (int o=0;o<27;o++) if (acc[o] != 0.f) atomicAdd(&f[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q], (double)acc[o]);
            }
        }
        }
        if (B) {
            float bd[27];
#pragma unroll
            for (int o=0;o<27;o++) bd[o] = 0.f;
#pragma unroll
            for (int a=0;a<3;a++) {
                if (!(m & (1 << a))) continue;
                float wf = fw ? fw[c*3+a] : 1.f;                               // 절 CM55 면 가중
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        float mm = kk*wf*GQ_M[i*4]*GQ_M[j*4];
                        bd[gq_cn(a,0,i,j)] += mm; bd[gq_cn(a,1,i,j)] += 9.f*mm; bd[gq_cn(a,2,i,j)] += 9.f*mm;
                        int id = gq_en(base, a, i, j, s0, s1);
                        atomicAdd(&B[id*9], mm); atomicAdd(&B[id*9+4], mm); atomicAdd(&B[id*9+8], mm);
                    }
            }
#pragma unroll
            for (int o=0;o<27;o++) if (bd[o] != 0.f) { int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3); atomicAdd(&B[id*9], bd[o]); atomicAdd(&B[id*9+4], bd[o]); atomicAdd(&B[id*9+8], bd[o]); }
        }
    }
    if (E) blocksum_add(e, E);
}
// ---- 선탐색 후보: y_c = ŷ + T(η + c δ) = a + c b, y_c − ŷ = e + c b (e = Tη) -> GQ[0..4] = Q_aa, Q_ab, Q_bb, Q_ee, Q_eb (면마다 D M Dᵀ, 칸별 스레드)
__global__ void __launch_bounds__(128) gq_ls(const int* IP, const int* cb, const unsigned char* fm, const float* fw, const float* yh, const float* y0, const float* eta, const float* dl, const float* sk, const int* pm,
                      float n0, float n1, float n2, double* GQ, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[11]) return;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    double qa = 0.0, qab = 0.0, qb = 0.0, qe = 0.0, qeb = 0.0;
    int m = (!IP[13] && c < IP[11]) ? fm[c] : 0;
    if (m) {
        int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
#pragma unroll 1
        for (int a=0;a<3;a++) {
            if (!(m & (1 << a))) continue;
            double wf = fw ? (double)fw[c*3+a] : 1.0;                          // 절 CM55 면 가중
            float Da[27], Db[27], De[27];
#pragma unroll
            for (int t=0;t<27;t++) { Da[t] = 0.f; Db[t] = 0.f; De[t] = 0.f; }
#pragma unroll 1
            for (int l=0;l<4;l++) {
                float cl = (l == 0) ? -1.f : ((l == 1) ? 3.f : ((l == 2) ? -3.f : 1.f));
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        int id = (l < 3) ? (base + ((a == 0) ? l : i)*s0 + ((a == 0) ? i : ((a == 1) ? l : j))*s1 + ((a == 2) ? l : j)) : gq_en(base, a, i, j, s0, s1);
                        float ev[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, dv[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]}, te[3], td[3];
                        Tapply(sk[id], pm[id], n0, n1, n2, ev, te); Tapply(sk[id], pm[id], n0, n1, n2, dv, td);
#pragma unroll
                        for (int q=0;q<3;q++) { int t = (i*3+j)*3 + q; Da[t] += cl*(yh[id*3+q] + te[q] + (y0 ? y0[id*3+q] : 0.f)); Db[t] += cl*td[q]; De[t] += cl*te[q]; }
                    }
            }
#pragma unroll
            for (int q=0;q<3;q++) {
                float A[9], Bv[9], Ev[9], MB[9], MA[9];
#pragma unroll
                for (int t=0;t<9;t++) { A[t] = Da[t*3+q]; Bv[t] = Db[t*3+q]; Ev[t] = De[t*3+q]; }
                gq_MDM(Bv, MB); gq_MDM(A, MA);
                float sa = 0.f, sab = 0.f, sb = 0.f, se = 0.f, seb = 0.f;
#pragma unroll
                for (int t=0;t<9;t++) { sa += A[t]*MA[t]; sab += A[t]*MB[t]; sb += Bv[t]*MB[t]; seb += Ev[t]*MB[t]; }
                gq_MDM(Ev, MA);
#pragma unroll
                for (int t=0;t<9;t++) se += Ev[t]*MA[t];
                qa += wf*sa; qab += wf*sab; qb += wf*sb; qe += wf*se; qeb += wf*seb;
            }
        }
    }
    blocksum_add(qa, &GQ[0]); blocksum_add(qab, &GQ[1]); blocksum_add(qb, &GQ[2]); blocksum_add(qe, &GQ[3]); blocksum_add(qeb, &GQ[4]);
}
__global__ void gq_lsfin(double* SC, int NC, int i0, const double* gate, double* GQ, float k, float cm) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) { double cd = (double)c; SC[40+i] += 0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[2]) + 0.5*(double)cm*(GQ[3] + 2.0*cd*GQ[4] + cd*cd*GQ[2]); c *= 0.5f; }
    GQ[0] = 0.0; GQ[1] = 0.0; GQ[2] = 0.0; GQ[3] = 0.0; GQ[4] = 0.0;
}
// ---- 절 CM57 ghost6 보존 판: 기준 형상 변위 u0 (걸음 시작 격자 노드) = 입자 (x_p − X_p) 를 F_p 로 노드까지 아핀 연장해 질량·기저 가중 평균
//   u0_i = Σ_p w_ip m_p [ (x_p − X_p) + (I − F_p⁻¹)(ξ_i − x_p) ] / Σ_p w_ip m_p   (ξ_i − x_p = A (o − fx_p), X_p = 첫 프레임 입자 위치, F_p = 걸음 시작 입자 F)
//   x = φ(X) 가 아핀 (강체 운동·균일 변형) 이면 괄호가 입자와 무관한 ξ_i 의 아핀 함수 -> u0 아핀 -> 면 3 차 차분 0.
//   det F ≤ 0 (또는 비유한) 입자는 아핀 몫 없이 (x_p − X_p) 만 (P2G 의 ng 규칙과 같은 판정; 셈 nbadp).
__global__ void g6_zero(const int* IP, float* num, float* den, int* nbadp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && nbadp) *nbadp = 0;
    if (n >= IP[9]) return;
    num[n*3] = 0.f; num[n*3+1] = 0.f; num[n*3+2] = 0.f; den[n] = 0.f;
}
__global__ void g6_p2g(int N, const int* IP, const float* Am, const int* pb, const float* fx, const float* x, const float* X0, const float* F, const float* m, float* num, float* den, int* nbadp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    const float* f = F + (size_t)p*9;
    float c00 = f[4]*f[8] - f[5]*f[7], c01 = f[5]*f[6] - f[3]*f[8], c02 = f[3]*f[7] - f[4]*f[6];
    float det = f[0]*c00 + f[1]*c01 + f[2]*c02;
    float C[9] = {0,0,0,0,0,0,0,0,0};
    if (det > 0.f && isfinite(det)) {                                               // F⁻¹ = adj(F)/det, C = I − F⁻¹
        float id = 1.f/det;
        float Fi[9] = {c00*id, (f[2]*f[7] - f[1]*f[8])*id, (f[1]*f[5] - f[2]*f[4])*id,
                       c01*id, (f[0]*f[8] - f[2]*f[6])*id, (f[2]*f[3] - f[0]*f[5])*id,
                       c02*id, (f[1]*f[6] - f[0]*f[7])*id, (f[0]*f[4] - f[1]*f[3])*id};
        for (int k=0;k<9;k++) C[k] = ((k % 4) == 0 ? 1.f : 0.f) - Fi[k];
    } else if (nbadp) atomicAdd(nbadp, 1);
    float d[3] = {x[p*3] - X0[p*3], x[p*3+1] - X0[p*3+1], x[p*3+2] - X0[p*3+2]};
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        float g[3] = {(float)i - fx[p*3], (float)j - fx[p*3+1], (float)k - fx[p*3+2]};
        float r[3]; for (int a=0;a<3;a++) r[a] = Am[a*3]*g[0] + Am[a*3+1]*g[1] + Am[a*3+2]*g[2];
        for (int a=0;a<3;a++) atomicAdd(&num[nid*3+a], ww*(d[a] + C[a*3]*r[0] + C[a*3+1]*r[1] + C[a*3+2]*r[2]));
        atomicAdd(&den[nid], ww);
    }
}
__global__ void g6_fin(const int* IP, const float* num, const float* den, float* u0) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float dd = den[n];
    for (int a=0;a<3;a++) u0[n*3+a] = (dd > 0.f) ? num[n*3+a]/dd : 0.f;
}
// ---- 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ) (en_phi_n 과 같은 MO 문)
__global__ void __launch_bounds__(128) gq_phi(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, const int* MO, double* PH) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (!MO[0]) return;
    gq_force(IP, cb, fm, fw, k, cm, y, yh, y0, (double*)0, (float*)0, PH + 1);
}

//@@ E3_SRC 1545

// ---- 절 CM45 W̄: 표본점 q 의 입자 G2P 기저 질량 평균 W̄_q,o = Σ_p m_p ŵ_o(x_p) / Σ m_p (AF_WQ, 프레임 시작) ----
__global__ void e3_wq(int N, const int* IP, const int* pb, const float* fx, const float* m, const int* pcd, const int* cidx, const float* mn, int g) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    size_t q = (size_t)c*8 + oc;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float ww[27], s = 0.f;
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        ww[o] = w[0][i]*w[1][j]*w[2][k]*(g == 2 ? mn[nid] : 1.f);
        s += ww[o];
    }
    float sc = m[p]*((g == 2 && s > 0.f) ? 1.f/s : 1.f);                      // G 판 G2P 와 같은 나눔 (Σ w m = 0 이면 G2P 변위 0 -> 몫 0)
    for (int o=0;o<27;o++) atomicAdd(&E3W(q, o), ww[o]*sc);
}
__global__ void e3_wqz(const int* IP) {                                    // 쓰는 칸만 0 (o 별 앞 IP[12] 개)
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    if (i >= nq*27) return;
    E3W(i % nq, i / nq) = 0.f;
}
__global__ void e3_wqn(const int* IP) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    if (i >= nq*27) return;
    int q = i % nq, o = i / nq;
    float bm = AF_BX[q*4+3];
    E3W(q, o) = (bm > 0.f) ? E3W(q, o)/bm : 0.f;
}

//@@ E5_SRC 2670

// ---- 절 CM49 R·A: 표본점 걸음 끝 장벽 기준 높이 이어 받기 (AF_BX 위) ----
__device__ __forceinline__ size_t e5_q(int p, const float* fx, const int* pcd, const int* cidx) {
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    return (size_t)c*8 + oc;
}
__global__ void e5_z(const int* IP, float* R, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q == 0) { cnt[0] = 0; cnt[1] = 0; cnt[2] = 0; }
    if (q >= IP[12]) return;
    R[q] = 0.f;
}
__global__ void e5_rp(int N, const int* IP, const float* fx, const float* m, const int* pcd, const int* cidx, const float* hr, float* R) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    atomicAdd(&R[e5_q(p, fx, pcd, cidx)], m[p]*hr[p]);
}
// 시작: d_q = 실제 질량 평균 높이, d̃_q = R ? Σ m h / Σ m : d_q; A 면 d̃ ≤ 0 -> d̂. δ_q = d̃ − d_q, x̄_q += δ_q n. cnt[0] = d̃ ≤ 0 (옮기기 전) 표본점 수, cnt[1] = A 로 옮긴 수
__global__ void e5_sh(const int* IP, const float* R, float* DQ, float n0, float n1, float n2, float pn, float dh, int useR, int lift, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f)) { DQ[q] = 0.f; return; }
    float da = (AF_BX[q*4]*n0 + AF_BX[q*4+1]*n1 + AF_BX[q*4+2]*n2)/bm - pn;
    float dr = useR ? R[q]/bm : da;
    if (!(dr > 0.f)) { atomicAdd(&cnt[0], 1); if (lift) { dr = dh; atomicAdd(&cnt[1], 1); } }
    float de = dr - da;
    DQ[q] = de;
    if (de != 0.f) { AF_BX[q*4] += bm*de*n0; AF_BX[q*4+1] += bm*de*n1; AF_BX[q*4+2] += bm*de*n2; }
}
// 끝 (G2P 뒤): S_q = Σ m d(x_p) (그 프레임 칸), h_p = δ_q + S_q / m_q
__global__ void e5_ep(int N, const int* IP, const float* fx, const float* m, const int* pcd, const int* cidx, const float* x, float n0, float n1, float n2, float pn, float* S) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    atomicAdd(&S[e5_q(p, fx, pcd, cidx)], m[p]*(x[p*3]*n0 + x[p*3+1]*n1 + x[p*3+2]*n2 - pn));
}
// R2 (rmode 2): 장벽 받침 안 (d̃_q 끝 < d̂ -- 장벽 에너지가 걸린 점) 만 표본점 기준 높이를 이어 받고, 받침 밖은 입자 실제 높이 (δ 0 으로 돌아감)
__global__ void e5_eh(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* S, const float* DQ, float* hr,
                      const float* x, float n0, float n1, float n2, float pn, float dh, int rmode) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    size_t q = e5_q(p, fx, pcd, cidx);
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f)) return;
    float de = DQ[q] + S[q]/bm;
    hr[p] = (rmode == 2 && !(de < dh)) ? (x[p*3]*n0 + x[p*3+1]*n1 + x[p*3+2]*n2 - pn) : de;
}

//@@ W5_SRC 13355

// ---- 절 CM70 wolfh5 (fusedgs/fem_wolfh5.py): 방출 뒤 낙하 매달림 진단 + 정답 받침 구속 몫 S_구속 ----
__device__ __forceinline__ int w5_bin(float dcm) {
    return (dcm < -6.f) ? 0 : ((dcm < -3.f) ? 1 : ((dcm < 0.f) ? 2 : ((dcm < 3.f) ? 3 : ((dcm < 6.f) ? 4 : ((dcm < 11.f) ? 5 : ((dcm < 21.f) ? 6 : ((dcm < 40.f) ? 7 : 8)))))));
}
__global__ void w5_zero_d(int n, double* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0.0; }
__global__ void w5_zero_f(int n, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0.f; }
__global__ void w5_copy_d(const int* IP, const double* a, double* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < IP[9]*3) b[i] = a[i]; }
__global__ void w5_copy_f(const int* IP, const float* a, float* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < IP[9]*3) b[i] = a[i]; }
// 노드 α (그 걸음 S 표본 평균 구속 질량 몫; ao_pbc_k 와 같은 표본 고리)
__global__ void w5_alpha(const int* IP, const float* mn, const float* acc, int K, int S, int k, float* al) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], a = 0.f; int KS = K*S;
    if (m > 0.f) { float suf = 0.f; for (int j=KS-1; j>=k*S; j--) { suf += acc[(size_t)n*KS + j]; if (j < (k+1)*S) a += fminf(suf/m, 1.f); } a /= (float)S; }
    al[n] = a;
}
// 입자 진단 (걸음 끝): 칸 = 앞면 거리 x 높이
__global__ void w5_pdg(int N, const int* IP, const int* pb, const float* fx, const float* x, const float* m, const float* v, const float* Trel, const float* tbuf, float toff, float vf,
                       const float* kb, const float* al, const float* mn, const float* v0, const float* vk, const double* fe, const double* ft,
                       float dt, float g0, float g1, float g2, float n0, float n1, float n2, float pn, double* D, const int* ctr, int cap) {
    __shared__ double sh[9*2*14];
    for (int i=threadIdx.x;i<9*2*14;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p < N && !IP[13]) {
        float t = tbuf[0] + toff, tr = Trel[p];
        float dcm = ((tr > t) ? -(tr - t) : (t - tr))*vf*100.f;
        float tau = (tr > t) ? 0.f : (t - fmaxf(tr, 0.f));
        float hg = x[p*3]*n0 + x[p*3+1]*n1 + x[p*3+2]*n2 - pn;
        int b = w5_bin(dcm)*2 + ((hg >= 0.1f) ? 1 : 0);
        float w[3][3], dw[3][3];
        for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
        int nb1 = IP[7], nb2 = IP[8];
        float mp = m[p], gn = g0*n0 + g1*n1 + g2*n2;
        float skb = 0.f, sal = 0.f, spin = 0.f, sv0 = 0.f, sv1 = 0.f;
        double fnet = 0.0, fel = 0.0, ftot = 0.0, fnf = 0.0, fef = 0.0;
        for (int o=0;o<27;o++) {
            int i = o/9, j = (o/3)%3, k = o%3;
            int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
            float ww = w[0][i]*w[1][j]*w[2][k];
            float kk = kb ? kb[nid] : 1.f, mm = mn[nid];
            float a0 = v0[nid*3]*n0 + v0[nid*3+1]*n1 + v0[nid*3+2]*n2, a1 = vk[nid*3]*n0 + vk[nid*3+1]*n1 + vk[nid*3+2]*n2;
            skb += ww*kk; sal += ww*(al ? al[nid] : 0.f); spin += ww*(1.f - kk); sv0 += ww*a0; sv1 += ww*a1;
            double fn = (double)ww*mp*((a1 - a0)/dt - gn); fnet += fn; fnf += (double)kk*fn;
            if (mm > 0.f) {
                double sc = (double)ww*mp/mm;
                double e_ = -(fe[nid*3]*n0 + fe[nid*3+1]*n1 + fe[nid*3+2]*n2)*sc;
                fel += e_; fef += (double)kk*e_;
                ftot += -(ft[nid*3]*n0 + ft[nid*3+1]*n1 + ft[nid*3+2]*n2)*sc;
            }
        }
        float vp = v[p*3]*n0 + v[p*3+1]*n1 + v[p*3+2]*n2;
        double* o_ = sh + b*14;
        atomicAdd(o_+0, 1.0); atomicAdd(o_+1, (double)mp); atomicAdd(o_+2, (double)mp*skb); atomicAdd(o_+3, (double)mp*sal); atomicAdd(o_+4, (double)mp*spin);
        atomicAdd(o_+5, (double)mp*sv0); atomicAdd(o_+6, (double)mp*sv1); atomicAdd(o_+7, (double)mp*tau);
        atomicAdd(o_+8, fnet); atomicAdd(o_+9, fel); atomicAdd(o_+10, ftot); atomicAdd(o_+11, fnf); atomicAdd(o_+12, fef); atomicAdd(o_+13, (double)mp*vp);
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<9*2*14;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[(size_t)row*252 + i], sh[i]);
}
// 가우스 점 진단 (걸음 끝 시험 X = (I + ∇U) F_q 의 투영 상태·응력) -- 장벽 판 (표본점 위치 AF_BX) 만
#ifdef AF_BAR
__global__ void __launch_bounds__(128) w5_qdg(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* Vq, const float* U,
                       float mu, float lam, float cdp, const float* kb, const float* al, int ax, float c0, float H0, float vf, const float* tbuf, float toff,
                       float n0, float n1, float n2, float pn, double* D, const int* ctr, int cap) {
    __shared__ double sh[9*2*10];
    for (int i=threadIdx.x;i<9*2*10;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    if (blockIdx.x*blockDim.x < IP[12]) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    float xq[3] = {0.f, 0.f, 0.f};
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[qq*4+a]/bm;
    float kq = 0.f, aq = 0.f;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o]; float ww = BW[oc*27+o];
        kq += ww*(kb ? kb[id] : 1.f); aq += ww*(al ? al[id] : 0.f);
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; xq[a] += ww*ua; } }
    float vq = on ? Vq[qq] : 0.f;
    if (on && bm > 0.f && vq > 0.f) {
        float Uu[9], S[3], V[9]; ssvdf(X, Uu, S, V);
        float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
        int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
        float K3 = lam + 2.f*mu/3.f;
        float pr = (st == 2) ? 0.f : -K3*tr;
        float sd = (st == 2) ? 0.f : 2.f*mu*((st == 1) ? s : 1.f)*rho;
        float dep = (st == 1) ? (1.f - s)*rho : ((st == 2) ? rho : 0.f);
        float t = tbuf[0] + toff;
        float dcm = (fabsf(xq[ax] - c0) - (H0 - vf*t))*100.f;
        float hg = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
        int b = w5_bin(dcm)*2 + ((hg >= 0.1f) ? 1 : 0);
        double* o_ = sh + b*10;
        atomicAdd(o_+0, 1.0); atomicAdd(o_+1, (double)vq);
        if (st == 1) atomicAdd(o_+2, (double)vq);
        if (st == 2) atomicAdd(o_+3, (double)vq);
        atomicAdd(o_+4, (double)vq*pr); atomicAdd(o_+5, (double)vq*sd); atomicAdd(o_+6, (double)vq*dep);
        atomicAdd(o_+7, (double)vq*kq); atomicAdd(o_+8, (double)vq*aq); atomicAdd(o_+9, (double)vq*tr);
    }
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<9*2*10;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[(size_t)row*180 + i], sh[i]);
}
#endif
__global__ void w5_inc(int* ctr, float* meta, const float* tbuf, float toff, float dt, int k, int cap) {
    int r = ctr[0];
    if (r < cap) { meta[r*4] = tbuf[0] + toff; meta[r*4+1] = dt; meta[r*4+2] = (float)k; meta[r*4+3] = 1.f; }
    ctr[0] = r + 1;
}
// ---- S_구속: 정답 격자 (dx, 노드 i dx) 상자 [g0, g0 + G) 의 노드 질량·구속 질량 표본 (ao_pbc_acc 와 같은 표본 시각)
__device__ __forceinline__ void w5_gw(float fx, float* w) { w[0] = 0.5f*(1.5f - fx)*(1.5f - fx); w[1] = 0.75f - (fx - 1.f)*(fx - 1.f); w[2] = 0.5f*(fx - 0.5f)*(fx - 0.5f); }
__device__ __forceinline__ bool w5_gbase(const float* x, float idx, int gi0, int gj0, int gk0, int G0, int G1, int G2, int* bl, float (*w)[3]) {
    int g0_[3] = {gi0, gj0, gk0}, GG[3] = {G0, G1, G2};
    for (int a=0;a<3;a++) { float y = x[a]*idx; float bf = floorf(y - 0.5f); int b = (int)bf - g0_[a]; if (b < 0 || b + 3 > GG[a]) return false; bl[a] = b; w5_gw(y - bf, w[a]); }
    return true;
}
__global__ void w5s_acc(int N, const float* x, const float* m, const float* Trel, const float* tbuf, int K, int S, float dt, float idx,
                        int gi0, int gj0, int gk0, int G0, int G1, int G2, float* gm, float* gacc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int bl[3]; float w[3][3];
    if (!w5_gbase(x + p*3, idx, gi0, gj0, gk0, G0, G1, G2, bl, w)) return;
    float tr = Trel[p], t0 = tbuf[0], mp = m[p];
    int KS = K*S, ncut = 0;
    if (tr > t0) for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) ncut = ks + 1; }
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        size_t gid = ((size_t)(bl[0]+i)*G1 + (bl[1]+j))*G2 + (bl[2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        atomicAdd(&gm[gid], ww);
        if (ncut > 0) atomicAdd(&gacc[gid*KS + ncut - 1], ww);
    }
}
// 정답 노드 k_g (ao_pbc_k / mt_pbc_kphi 와 같은 식) -> 1 − k_g 저장 (받침 밖 정확히 0)
__global__ void w5s_kg(int NG, const float* gm, const float* gacc, int K, int S, int k, float expo, int phi, float* omg) {
    int g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= NG) return;
    float m = gm[g], lk = 0.f; int KS = K*S;
    if (m > 0.f) { float suf = 0.f; for (int j=KS-1; j>=k*S; j--) { suf += gacc[(size_t)g*KS + j]; if (j < (k+1)*S) { float a = fminf(suf/m, 1.f); lk += log1pf(-a); } } }
    if (lk == 0.f) { omg[g] = 0.f; return; }
    float r;
    if (phi) { float xx = -lk*expo; if (xx < 1e-3f) r = 1.f - xx/3.f + xx*xx/12.f; else if (isinf(xx)) r = 0.f; else r = 2.f*(xx - 1.f + expf(-xx))/(xx*xx); }
    else r = expf(lk*expo);
    omg[g] = 1.f - r;
}
// 입자 1 − κ_p = Σ_g w (1 − k_g) (정답 G2P), 우리 노드로 질량 가중 P2G: num = Σ w m κ, nom = Σ w m (1 − κ), den = Σ w m
__global__ void w5s_kn(int N, const int* IP, const int* pb, const float* fx, const float* x, const float* m, float idx,
                       int gi0, int gj0, int gk0, int G0, int G1, int G2, const float* omg, float* num, float* nom, float* den) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int bl[3]; float wg[3][3];
    float om = 0.f, kp = 0.f;                                                   // om = Σ w (1 − k_g) (받침 표지), κ = Σ w k_g (음수 없음)
    if (w5_gbase(x + p*3, idx, gi0, gj0, gk0, G0, G1, G2, bl, wg))
        for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; float og = omg[((size_t)(bl[0]+i)*G1 + (bl[1]+j))*G2 + (bl[2]+k)]; float ww_ = wg[0][i]*wg[1][j]*wg[2][k];
                                 if (og != 0.f) om += ww_*og; kp += ww_*(1.f - og); }
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        atomicAdd(&den[nid], ww);
        if (om != 0.f) { atomicAdd(&nom[nid], ww*om); atomicAdd(&num[nid], ww*kp); } else atomicAdd(&num[nid], ww);
    }
}
__global__ void w5s_kb(const int* IP, const float* num, const float* nom, const float* den, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    kb[n] = (nom[n] > 0.f && den[n] > 0.f) ? num[n]/den[n] : 1.f;
}
// 구속 노드 걸음 끝 가속도: 연속판 a ← k a (mt_akb 의 k < 1 -> 0 과 k ∈ {0,1} 에서 같음)
// B_g: 면 36 노드 중 k_구속 < 1 노드가 있으면 면 표지 비트를 지움 (절 CM63 gq_w2stk 와 같은 배치)
__global__ void w5_gfx(const int* IP, const int* cb, const float* kb, unsigned char* fm, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int nb1 = IP[7], nb2 = IP[8]; int s1 = nb2, s0 = nb1*nb2;
    int base = (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
    int cf = 0;
    for (int o=0;o<27;o++) if (kb[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)] < 1.f) { cf = 1; break; }
    int m2 = m, nr = 0;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        int f = cf;
        for (int i=0;i<3 && !f;i++) for (int j=0;j<3 && !f;j++) {
            int id = (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
            if (kb[id] < 1.f) f = 1;
        }
        if (f) { m2 &= ~(1 << a); nr++; }
    }
    if (m2 != m) { fm[c] = (unsigned char)m2; atomicAdd(&cnt[0], nr); }
}
// B_gk: 면 가중 = 면 36 노드 k_구속 최솟값 (표지 비트가 선 면만)
__global__ void w5_gwk(const int* IP, const int* cb, const float* kb, const unsigned char* fm, float* fw) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int nb1 = IP[7], nb2 = IP[8]; int s1 = nb2, s0 = nb1*nb2;
    int base = (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
    float kc = 1.f;
    for (int o=0;o<27;o++) kc = fminf(kc, kb[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)]);
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        float k = kc;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            int id = (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
            k = fminf(k, kb[id]);
        }
        fw[c*3+a] = k;
    }
}
__global__ void w5s_akb(const int* IP, const float* kb, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = kb[n];
    if (k < 1.f) { ak[n*3] *= k; ak[n*3+1] *= k; ak[n*3+2] *= k; }
}

//@@ LX_DEV 1083

// ---- 절 CM77 (L'): 저질량 고정 노드 자리 = 이웃 26 자유 노드 예측 증분의 질량 가중 평균 (fusedgs/fem_wolfh6.py)
__device__ __forceinline__ float w6_nbr(const int* IP, int n, const float* U, const float* Ut, const float* mn, float mthr, const float* kb, float* ex) {
    ex[0] = 0.f; ex[1] = 0.f; ex[2] = 0.f;
    if (kb && kb[n] < 1.f) return 0.f;                                          // 구속 받침 노드는 절 CM70 C 그대로
    int nb0 = IP[6], nb1 = IP[7], nb2 = IP[8];
    int i0 = n/(nb1*nb2), j0 = (n/nb2)%nb1, k0 = n%nb2; float sw = 0.f;
    for (int di=-1;di<=1;di++) for (int dj=-1;dj<=1;dj++) for (int dk=-1;dk<=1;dk++) {
        if (!di && !dj && !dk) continue;
        int i = i0+di, j = j0+dj, k = k0+dk;
        if (i < 0 || j < 0 || k < 0 || i >= nb0 || j >= nb1 || k >= nb2) continue;
        int q = (i*nb1 + j)*nb2 + k;
        if (q >= IP[9]) continue;
        float m = mn[q];
        if (!(m >= mthr && m > 0.f) || (kb && kb[q] < 1.f)) continue;
        sw += m; for (int a=0;a<3;a++) ex[a] += m*(Ut[q*3+a] - U[q*3+a]);
    }
    if (sw > 0.f) for (int a=0;a<3;a++) ex[a] /= sw;
    return sw;
}

//@@ W8_SRC 7059

// ---- 절 CM81 wolfh8: 확장 B-스플라인 (fusedgs/fem_wolfh8.py)
__device__ __forceinline__ void w8_wts(int x, int deg, float* w) {          // 블록 가운데에서 축 거리 x 인 노드의 블록 노드 (−1, 0, 1) 가중
    w[0] = 0.f; w[1] = 0.f; w[2] = 0.f;
    if (x >= -1 && x <= 1) { w[x+1] = 1.f; return; }                          // (차수 0 은 |x| > 2 에도 끝 노드 하나)
    float t = (float)x;
    if (deg >= 2) { w[0] = 0.5f*t*(t - 1.f); w[1] = (1.f - t)*(1.f + t); w[2] = 0.5f*t*(t + 1.f); }
    else if (deg == 1) { if (x > 0) { w[1] = 1.f - t; w[2] = t; } else { w[0] = -t; w[1] = 1.f + t; } }
    else { if (x > 0) w[2] = 1.f; else w[0] = 1.f; }
}
__global__ void w8_icell(const int* IP, const int* cb, const float* A, int* ic, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int o=0;o<8;o++) if (!(A[(size_t)(c*8 + o)*28 + 27] > 0.f)) return;
    int n1 = IP[7]-2, n2 = IP[8]-2;
    ic[(cb[c*3]*n1 + cb[c*3+1])*n2 + cb[c*3+2]] = 1;
    atomicAdd(&cnt[3], 1);
}
__global__ void w8_icell2(const int* IP, const int* cb, const int* occ, int* ic, int* cnt, int face) {   // 내부 칸 정의 t 1: 점유 칸 중 26 이웃 칸이 모두 점유 (칸 점유 집합의 위상 내부), t 2 (face 1): 면 이웃 6 칸만
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int a0 = cb[c*3], b0 = cb[c*3+1], c0_ = cb[c*3+2];
    for (int a=a0-1;a<=a0+1;a++) for (int b=b0-1;b<=b0+1;b++) for (int d=c0_-1;d<=c0_+1;d++) {
        if (face && (a != a0) + (b != b0) + (d != c0_) > 1) continue;
        if (a < 0 || b < 0 || d < 0 || a >= n0 || b >= n1 || d >= n2) return;
        if (!occ[(a*n1 + b)*n2 + d]) return; }
    ic[(a0*n1 + b0)*n2 + c0_] = 1;
    atomicAdd(&cnt[3], 1);
}
__global__ void w8_class(const int* IP, const float* mn, const int* ic, int deg, int* base, float* w, int* cnt, float* cf, int xiso) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    base[n] = -1;
    float m = mn[n];
    if (!(m > 0.f) || IP[13]) return;
    int nb1 = IP[7], nb2 = IP[8];
    int c0 = IP[6]-2, c1 = nb1-2, c2 = nb2-2;
    int i = n/(nb1*nb2), j = (n/nb2)%nb1, k = n%nb2;
    for (int a=i-2;a<=i;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-2;b<=j;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-2;c<=k;c++) { if (c < 0 || c >= c2) continue;
          if (ic[(a*c1 + b)*c2 + c]) { atomicAdd(&cnt[0], 1); atomicAdd(&cf[0], m); return; } } } }
    int best = -1, bd = 1 << 30, ba = 0, bb = 0, bc = 0;
    for (int a=i-3;a<=i+1;a++) { if (a < 0 || a >= c0) continue;
      for (int b=j-3;b<=j+1;b++) { if (b < 0 || b >= c1) continue;
        for (int c=k-3;c<=k+1;c++) { if (c < 0 || c >= c2) continue;
          int cd = (a*c1 + b)*c2 + c;
          if (!ic[cd]) continue;
          int x = i-a-1, y = j-b-1, z = k-c-1, d = x*x + y*y + z*z;
          if (d < bd || (d == bd && cd < best)) { bd = d; best = cd; ba = a; bb = b; bc = c; } } } }
    int dg = deg;
    if (best < 0 && xiso) {                                                            // 절 CM81 x 1: 고립 노드 = 가장 가까운 내부 칸 (칸 거리 껍질 3, 4, ... 차례) 의 가장 가까운 블록 노드에 0 차 (평행 이동)
        int R = c0 > c1 ? c0 : c1; R = R > c2 ? R : c2;
        for (int d=3; d<=R+1 && best < 0; d++) {
          for (int a=i-1-d;a<=i-1+d;a++) { if (a < 0 || a >= c0) continue;
            for (int b=j-1-d;b<=j-1+d;b++) { if (b < 0 || b >= c1) continue;
              int ea = (a == i-1-d || a == i-1+d || b == j-1-d || b == j-1+d);
              for (int c=k-1-d;c<=k-1+d;c += (ea ? 1 : 2*d)) { if (c < 0 || c >= c2) continue;
                int cd = (a*c1 + b)*c2 + c;
                if (!ic[cd]) continue;
                int x = i-a-1, y = j-b-1, z = k-c-1, dd = x*x + y*y + z*z;
                if (dd < bd || (dd == bd && cd < best)) { bd = dd; best = cd; ba = a; bb = b; bc = c; } } } } }
        if (best >= 0) { dg = 0; atomicAdd(&cnt[8], 1); atomicAdd(&cf[3], m); }
    }
    if (best < 0) { atomicAdd(&cnt[2], 1); atomicAdd(&cf[2], m); return; }           // 고립 (가까운 내부 칸 없음) -- 그대로
    atomicAdd(&cnt[1], 1); atomicAdd(&cf[1], m);
    int xs[3] = {i-ba-1, j-bb-1, k-bc-1}, ne = 0;
    for (int a=0;a<3;a++) if (xs[a] < -1 || xs[a] > 1) ne++;
    atomicAdd(&cnt[4 + ne], 1);                                                       // 외삽 축 수 1·2·3 -> cnt[5..7]
    base[n] = (ba*nb1 + bb)*nb2 + bc;
    w8_wts(xs[0], dg, w + n*9); w8_wts(xs[1], dg, w + n*9 + 3); w8_wts(xs[2], dg, w + n*9 + 6);
}
__global__ void w8_free(const int* IP, const int* base, float* sk, int* pm, const float* yh, const float* y, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    sk[n] = 1.f; pm[n] = 0;
    for (int a=0;a<3;a++) eta[n*3+a] = y[n*3+a] - yh[n*3+a];
}
__global__ void w8_fold(const int* IP, const int* base, const float* w, float* v, float* acc) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    float vv[3] = {v[n*3], v[n*3+1], v[n*3+2]};
    v[n*3] = 0.f; v[n*3+1] = 0.f; v[n*3+2] = 0.f;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) atomicAdd(&acc[q*3+d], ww*vv[d]); } } }
}
__global__ void w8_fold2(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, float* acc, float* v) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float a3[3] = {acc[n*3], acc[n*3+1], acc[n*3+2]};
    if (a3[0] == 0.f && a3[1] == 0.f && a3[2] == 0.f) return;
    acc[n*3] = 0.f; acc[n*3+1] = 0.f; acc[n*3+2] = 0.f;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, a3, t);
    for (int d=0;d<3;d++) v[n*3+d] += t[d];
}
__global__ void w8_gg(const int* IP, const float* r, double* SC) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) for (int d=0;d<3;d++) { double q = r[n*3+d]; s += q*q; }
    blocksum_add(s, &SC[1]);
}
__global__ void w8_expand(const int* IP, const int* base, const float* w, float* p, float* Tp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c;
          for (int d=0;d<3;d++) t[d] += ww*Tp[q*3+d]; } } }
    for (int d=0;d<3;d++) { Tp[n*3+d] = t[d]; p[n*3+d] = t[d]; }
}

//@@ W12_SRC 2005

// ---- 절 CM93 wolfh12: 걸음 끝 (커밋 앞) 가우스 점 체적 진단 (읽기만) ----
__global__ void w12_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* U, float mu, float lam, float cdp, float* o) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o_=0;o_<27;o_++) { float g[3]; gqn(F, BG + (oc*27+o_)*3, g); int id = ids[o_];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float J = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], pe[3], ts = 0.f;
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ts += tau; ss[i] = sgn[i]*sg[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float D[3][3], Dk[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
    float v[3] = {ss[0]/3.f, ss[1]/3.f, ss[2]/3.f};
    float h = 0.f; for (int i=0;i<3;i++) for (int j=0;j<3;j++) h += v[i]*Dk[i][j]*v[j];
    jac3f(D, Ev);
    float hp = 0.f;
    for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f) || !isfinite(dm)) continue;
        float c = Ev[0][m]*v[0] + Ev[1][m]*v[1] + Ev[2][m]*v[2]; hp += dm*c*c; }
    float* r = o + (size_t)q*8;
    r[0] = J; r[1] = tr; r[2] = (float)st; r[3] = ts/3.f; r[4] = 0.5f*K3*tr*tr; r[5] = mu*(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    r[6] = (h != 0.f) ? hp/h : 0.f; r[7] = Vq[q];
}

//@@ X1_LOG 514

// ---- 절 CM83 ext1 기록: 걸음 시작 분류 수·질량을 장치 버퍼에 (그래프 안), 다음 걸음을 위해 0
__global__ void x1_log(const int* IP, int* cnt, float* cf, int* hp, int* H, float* HF, int cap, int ks, int K) {
    int s = hp[0]; hp[0] = s + 1;
    if (s < cap) {
        for (int i=0;i<12;i++) H[s*16 + i] = cnt[i];
        H[s*16 + 12] = ks; H[s*16 + 13] = K; H[s*16 + 14] = IP[11]; H[s*16 + 15] = IP[9];
        for (int i=0;i<4;i++) HF[s*4 + i] = cf[i];
    }
    for (int i=0;i<16;i++) cnt[i] = 0;
    for (int i=0;i<4;i++) cf[i] = 0.f;
}

//@@ X1_SRC 3961

// ---- 절 CM83 ext1: 옛 실행기 자리 (fusedgs/fem_ext1.py)
// 블록 야코비: 접은 r 로 z = D⁻¹ r, p = z, Tp = T z, SC[0] += r·z (rt_grad 와 같은 식; cmp 1 = 압축 판 AO_ACT 목록)
__global__ void x1_jz(const int* IP, const float* Dinv, const float* sk, const int* pm, float n0, float n1, float n2, const float* r, float* z, float* p, float* Tp, double* SC, int cmp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int lim = cmp ? IP[14] : IP[9];
    if (blockIdx.x*blockDim.x >= lim) return;
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0;
    if (ii < lim) {
        int n = cmp ? AO_ACT[ii] : ii;
        const float* D = Dinv + n*9;
        float rr[3] = {r[n*3], r[n*3+1], r[n*3+2]};
        float zz[3];
        for (int a=0;a<3;a++) { zz[a] = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]);
}
// |g|² = |접은 r|² (SC[1]); cmp 1 = AO_ACT 목록 (압축 판은 빈 노드 r 을 안 씀)
__global__ void x1_gg(const int* IP, const float* r, double* SC, int cmp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int lim = cmp ? IP[14] : IP[9];
    if (blockIdx.x*blockDim.x >= lim) return;
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (ii < lim) { int n = cmp ? AO_ACT[ii] : ii; for (int d=0;d<3;d++) { double q = r[n*3+d]; s += q*q; } }
    blocksum_add(s, &SC[1]);
}
// 노드별 보폭 후보: 외부 노드 = 주인 노드 실제 보폭 증분의 외삽 (y1_o = y_o + Σ c 2^-k_j td_j, dd_o 같은 값)
__global__ void x1_wscand(const int* IP, const int* base, const float* w, const int* kn, const float* td, const float* y, float* y1, float* dd) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float sq = ldexpf(1.f, -kn[q]);
          for (int d=0;d<3;d++) t[d] += ww*sq*td[q*3+d]; } } }
    for (int d=0;d<3;d++) { dd[n*3+d] = t[d]; y1[n*3+d] = y[n*3+d] + t[d]; }
}
// 외부 노드 k_o 를 주인 노드에 (k_j = max(k_j, k_o))
__global__ void x1_kprop(const int* IP, const int* base, const float* w, int* kn) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n], k = kn[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { if (W[a]*W[3+b]*W[6+c] == 0.f) continue;
          atomicMax(&kn[b0 + (a*nb1 + b)*nb2 + c], k); } } }
}
// 노드별 보폭 갱신 뒤: 외부 노드 증분 = 주인 노드 실제 증분의 외삽 (T_o = I)
__global__ void x1_wslsup(const int* IP, const int* base, const float* w, const int* kn, const float* td, const float* dl, const float* yh, float* eta, float* y) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float sq = ldexpf(1.f, -kn[q]);
          for (int d=0;d<3;d++) t[d] += ww*sq*td[q*3+d]; } } }
    float so = ldexpf(1.f, -kn[n]);
    for (int d=0;d<3;d++) { float e = eta[n*3+d] + t[d] - so*dl[n*3+d]; eta[n*3+d] = e; y[n*3+d] = yh[n*3+d] + e; }
}

//@@ LG_SRC 3455

// ---- 절 CM73 lightg3: 면 가중 = 1[P1X 면] + r · λ (fusedgs/fem_lightg3.py), λ 정의 fr: 0 = 절 CM71 λ_free (pm ≠ 2) · 1 = (A) 원래 고정 판정에서 자유 (pm ≠ 2, m ≥ m̄) · 2 = (B) (A) + 장벽 받침 표시 노드가 낀 면은 0
__global__ void gq_l3thk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
        int c2 = cidx[(X*n1 + Y)*n2 + Z];
        if (c2 >= 0 && !cut[c2]) k = 1;
    }
    thk[c] = (unsigned char)k;
}
// 면 (칸 c, 축 a) 36 노드: l = mbar/(m + mbar) (m > 0) 의 최댓값 -- l0 자유 (pm ≠ 2), lA 자유이면서 m ≥ mbar (원래 고정 판정 'm < mbar' 에서 자유), la 전체 (진단);
// bt = 36 노드 중 장벽 받침 표시 노드 (절 CM67 (2) mf_mark, bf) 가 있음; mi = 가장 가벼운 노드 질량 (진단 무리). pm 이 null 이면 λ 몫 0 (프레임 첫 바닥 노드 커널 앞).
// fw = ind + (sc && ind ? 0 : r · λ), λ = fr 0: l0 · 1: lA · 2: bt ? 0 : lA
__global__ void gq_l3w(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const unsigned char* thk, const float* mn, const int* pm, const unsigned char* bf,
                       float mbar, float r, int sc, int fr, float* fw, float* wl0, float* wlA, float* wbt, float* wla, float* wmi, float* wid) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    float l0 = 0.f, lA0 = 0.f, a0 = 0.f, mi0 = 3.0e38f; int t0 = 0;
    if (m) {
        for (int o=0;o<27;o++) {
            int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3);
            float q = mn[id];
            if (bf && bf[id]) t0 = 1;
            if (q > 0.f) { float l = mbar/(q + mbar); a0 = fmaxf(a0, l); mi0 = fminf(mi0, q);
                           if (pm && pm[id] != 2) { l0 = fmaxf(l0, l); if (q >= mbar) lA0 = fmaxf(lA0, l); } }
        }
    }
    for (int a=0;a<3;a++) {
        float w = 0.f, lx = 0.f, lA = 0.f, la = 0.f, mi = 0.f, bt = 0.f; int ind = 0;
        if (m & (1 << a)) {
            int x2[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x2[a] += 1;
            int c2 = (x2[a] < n[a]) ? cidx[(x2[0]*n[1] + x2[1])*n[2] + x2[2]] : -1;
            float l1 = l0, A1 = lA0, a1 = a0; int t1 = t0; mi = mi0;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                int id = gq_en(base, a, i, j, s0, s1);
                float q = mn[id];
                if (bf && bf[id]) t1 = 1;
                if (q > 0.f) { float l = mbar/(q + mbar); a1 = fmaxf(a1, l); mi = fminf(mi, q);
                               if (pm && pm[id] != 2) { l1 = fmaxf(l1, l); if (q >= mbar) A1 = fmaxf(A1, l); } }
            }
            lx = l1; lA = A1; la = a1; bt = (float)t1;
            float lam = (fr == 0) ? l1 : ((fr == 1) ? A1 : (t1 ? 0.f : A1));
            ind = (c2 >= 0 && thk[c] && thk[c2]) ? 1 : 0;
            w = (float)ind + ((sc && ind) ? 0.f : r*lam);
        }
        fw[c*3+a] = w; wl0[c*3+a] = lx; wlA[c*3+a] = lA; wbt[c*3+a] = bt; wla[c*3+a] = la; wmi[c*3+a] = mi; wid[c*3+a] = (float)ind;
    }
}

//@@ HC_SRC 7047

// ---- 절 CM58 (fusedgs/fem_hhtc.py): 접촉 노드 판정 (걸음 시작·끝 장벽 받침 안) + 걸음 끝 갱신 바꿈 표시 ----
#ifdef E3W
#define HC_W(q, oc, o) E3W(q, o)
#else
#define HC_W(q, oc, o) BW[(oc)*27+(o)]
#endif
__global__ void hc_zero(const int* IP, int* ms, int* me, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) st[2] = 0;                                                     // 이 걸음 새 접촉 노드 수
    if (n < IP[9]) { ms[n] = 0; me[n] = 0; }
}
__global__ void __launch_bounds__(128) hc_bar(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv,
                                              float n0, float n1, float n2, float pn, float dh, int* ms, int* me) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xs[3], xe[3];
    for (int a=0;a<3;a++) { xs[a] = AF_BX[q*4+a]/bm; xe[a] = xs[a]; }
    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];
        for (int a=0;a<3;a++) { float u = U[id*3+a]; xs[a] += w*u; xe[a] += w*(u + thinv*(y[id*3+a] - u)); } }
    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;
    bool s_ = ds < dh, e_ = de < dh;                                          // 장벽 받침 안 (rt_bar_q 가 힘을 거는 조건 d < d̂, 관통 d ≤ 0 포함)
    if (!s_ && !e_) return;
    for (int o=0;o<27;o++) if (HC_W(q, oc, o) != 0.f) { if (s_) ms[ids[o]] = 1; if (e_) me[ids[o]] = 1; }
}
// st: [0] 새 접촉 노드 합 [1] 걸음 끝 접촉 노드 합 [2] 이 걸음 새 접촉 노드 [3] 충돌 걸음 수 [4] 걸음 수 [5] 바꾼 노드 합
__global__ void hc_fin(const int* IP, const int* ms, const int* me, int mode, int* hm, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int e = me[n], nw = e && !ms[n];
    if (e) atomicAdd(&st[1], 1);
    if (nw) { atomicAdd(&st[0], 1); atomicAdd(&st[2], 1); }
    int h = (mode == 3) ? e : ((mode == 2) ? nw : 0);
    hm[n] = h;
    if (h) atomicAdd(&st[5], 1);
}
__global__ void hc_all(const int* IP, int* st, int* hm) {                     // H1: 충돌 걸음이면 몸 전체
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || st[2] == 0) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
// H4: 충돌 걸음 중 HHT 걸음 끝 속도 v1 = α (y − z0) 가 새 접촉 노드들의 법선 속도 합을 들어가는 쪽 (Σ v0·n < 0) 에서 나오는 쪽 (Σ v1·n > 0) 으로 뒤집는 걸음만 몸 전체
__global__ void hc_fz(double* fa) { fa[0] = 0.0; fa[1] = 0.0; }
__global__ void hc_flip(const int* IP, const int* ms, const int* me, const float* y, const float* z0, const float* vk, float al, float n0, float n1, float n2, double* fa) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !(me[n] && !ms[n])) return;
    float v0 = vk[n*3]*n0 + vk[n*3+1]*n1 + vk[n*3+2]*n2;
    float v1 = al*((y[n*3] - z0[n*3])*n0 + (y[n*3+1] - z0[n*3+1])*n1 + (y[n*3+2] - z0[n*3+2])*n2);
    atomicAdd(&fa[0], (double)v0); atomicAdd(&fa[1], (double)v1);
}
__global__ void hc_all4(const int* IP, int* st, const double* fa, int* hm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0) atomicAdd(&st[6], 1);   // 뒤집힌 충돌 걸음 수
    if (n >= IP[9] || !(st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0)) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
__global__ void hc_cnt(int* st) { st[4] += 1; if (st[2] > 0) st[3] += 1; }
// ---- 절 CM61 (fusedgs/fem_hhtc2.py): 걸음마다 몸 운동량·운동 에너지 (노드 질량 가중, 몸 전체·새 접촉·끝 접촉), 블록 합 뒤 원자 합 ----
#define HC_NV 20
#define HC_MR 32
__global__ void hc_mz(const int* st, double* mo) {
    int i = threadIdx.x; int r = min(st[4], HC_MR - 1);
    if (i < HC_NV) mo[r*HC_NV + i] = 0.0;
}
__global__ void __launch_bounds__(256) hc_mom(const int* IP, const int* ms, const int* me, const float* mn, const float* y, const float* z0, const float* U,
                                              const float* vk, const float* ak, float dt, float gm, float be, float al, float n0, float n1, float n2,
                                              const int* st, double* mo) {
    __shared__ double sh[8][HC_NV];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v[HC_NV];
    for (int i=0;i<HC_NV;i++) v[i] = 0.0;
    if (n < IP[9]) {
        float m = mn[n];
        if (m > 0.f) {
            float nn[3] = {n0, n1, n2}; float p0 = 0.f, pH = 0.f, pB = 0.f, k0 = 0.f, kH = 0.f, kB = 0.f, dH = 0.f, dB = 0.f;
            for (int a=0;a<3;a++) {
                float u = U[n*3+a], vv = vk[n*3+a], ac = ak[n*3+a];
                float v1 = al*(y[n*3+a] - z0[n*3+a]);
                float a1 = (v1 - vv - dt*(1.f - gm)*ac)/(gm*dt);
                float du = dt*vv + dt*dt*((0.5f - be)*ac + be*a1);
                float vb = (y[n*3+a] - u)/dt;
                p0 += vv*nn[a]; pH += v1*nn[a]; pB += vb*nn[a]; k0 += vv*vv; kH += v1*v1; kB += vb*vb; dH += du*nn[a]; dB += (y[n*3+a] - u)*nn[a];
            }
            double md = (double)m;
            v[0] = md; v[1] = md*p0; v[2] = md*pH; v[3] = md*pB; v[4] = 0.5*md*k0; v[5] = 0.5*md*kH; v[6] = 0.5*md*kB; v[7] = md*dH; v[8] = md*dB;
            int e = me[n], nw = e && !ms[n];
            if (nw) { v[9] = md; v[10] = md*p0; v[11] = md*pH; v[12] = md*pB; v[17] = p0; v[18] = pH; v[19] = 1.0; }
            if (e) { v[13] = md; v[14] = md*p0; v[15] = md*pH; v[16] = md*pB; }
        } else if (me[n] && !ms[n]) {                                                // 질량 0 새 접촉 노드도 절 CM58 H4 합에는 듦 (같은 셈)
            float p0 = vk[n*3]*n0 + vk[n*3+1]*n1 + vk[n*3+2]*n2;
            float pH = al*((y[n*3] - z0[n*3])*n0 + (y[n*3+1] - z0[n*3+1])*n1 + (y[n*3+2] - z0[n*3+2])*n2);
            v[17] = p0; v[18] = pH; v[19] = 1.0;
        }
    }
    for (int i=0;i<HC_NV;i++) for (int o=16;o>0;o>>=1) v[i] += __shfl_down_sync(0xffffffff, v[i], o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) for (int i=0;i<HC_NV;i++) sh[w][i] = v[i];
    __syncthreads();
    if (threadIdx.x < HC_NV) {
        double t = 0.0; int nw_ = (blockDim.x + 31) >> 5;
        for (int k=0;k<nw_;k++) t += sh[k][threadIdx.x];
        int r = min(st[4], HC_MR - 1);
        atomicAdd(&mo[r*HC_NV + threadIdx.x], t);
    }
}
// H5 (모드 5): H4 조건 ∧ 몸 질량중심 법선 운동량을 HHT 끝 속도가 뒤집음 (P0 < 0 < PH) ∧ 걸음 평균 운동량 PB < 0; H5a (모드 6): H4 ∧ P0 < 0 < PH. 셈: st[7] H5 · st[8] H5a
__global__ void hc_all5(const int* IP, int* st, const double* fa, const double* mo, int mode, int* hm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int r = min(st[4], HC_MR - 1);
    const double* q = mo + r*HC_NV;
    bool c4 = st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0;
    bool c6 = c4 && q[1] < 0.0 && q[2] > 0.0;                                   // H5a 몸 뒤집힘
    bool c5 = c6 && q[3] < 0.0;                                                  // H5 = H5a ∧ 걸음 평균 들어가는 쪽
    if (n == 0) { if (c4 && mode >= 5) atomicAdd(&st[6], 1); if (c5) atomicAdd(&st[7], 1); if (c6) atomicAdd(&st[8], 1); }   // 모드 0·4 의 st[6] 은 hc_all4 가 셈
    bool on = (mode == 5) ? c5 : ((mode == 6) ? c6 : false);
    if (n >= IP[9] || !on) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}

//@@ MF_SRC 915

// ---- 절 CM67 mfix: 저질량 고정 노드 정의 (연속 덧질량 · 장벽 노드 빼기) ----
__global__ void mf_save(const int* IP, const float* mn, float* mnb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    mnb[n] = mn[n];
}
// c1: 풀이 관성 질량 mn = mnb + add (mnb > 0, 표시 아닌 노드); mnx = 고정 판정 (floor_n 의 'mn < mthr') 이 볼 값: 표시 노드 (m > 0) 는 mthr (고정 아님), 아니면 mn
__global__ void mf_set(const int* IP, const float* mnb, float* mn, float* mnx, const unsigned char* bf, float add, float mthr, int c1, int c2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mnb[n];
    int b = (c2 && bf[n]) ? 1 : 0;
    float mm = (c1 && m > 0.f && !b) ? (m + add) : m;
    if (c1) mn[n] = mm;
    mnx[n] = (b && m > 0.f) ? mthr : mm;
}
__global__ void mf_zero(const int* IP, unsigned char* bf) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bf[n] = 0;
}

//@@ MF_BAR_SRC 1120

#ifdef E3W
#define MFW(q_, oc_, o_) E3W(q_, o_)
#else
#define MFW(q_, oc_, o_) BW[(oc_)*27+(o_)]
#endif
// 장벽 표본점 (x̄_q + Σ W̄ ...) 이 걸음 시작 (U) 또는 관성 예측 (U + thinv (y − U), y = 예측) 에서 d < d̂ 이면 그 27 노드 표시
__global__ void mf_mark(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, unsigned char* bf, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3]; if (!(bm > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xs[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
    float xp[3] = {xs[0], xs[1], xs[2]};
    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];
        for (int a=0;a<3;a++) { xs[a] += w*U[id*3+a]; xp[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); } }
    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;
    if (!(ds < dh) && !(dp < dh)) return;
    atomicAdd(&cnt[0], 1);
    for (int o=0;o<27;o++) if (MFW(q, oc, o) > 0.f) bf[ids[o]] = 1;
}

//@@ startpt_SRC 5007

extern "C" {
__device__ __forceinline__ bool cfs(const int* S) { return S != 0 && *S != 0; }
__global__ void cf_reset(const int* IP, int capN, int nbk, double* prev, double* acc, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= capN || n >= IP[9]) return;
    prev[n*3] = 0.0; prev[n*3+1] = 0.0; prev[n*3+2] = 0.0;
    for (int b=0;b<nbk;b++) acc[(size_t)b*capN + n] = 0.0;
}
__global__ void cf_delta(const int* IP, const double* f, double* prev, double* acc, float n0, float n1, float n2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    double a = f[n*3], b = f[n*3+1], c = f[n*3+2];
    acc[n] += (a - prev[n*3])*(double)n0 + (b - prev[n*3+1])*(double)n1 + (c - prev[n*3+2])*(double)n2;
    prev[n*3] = a; prev[n*3+1] = b; prev[n*3+2] = c;
}
// 창 끝: 작업 통 (acc2) -> 끝 칸 (acc1), n·η, 평가 번호·1/τ²
__global__ void cf_fin(const int* IP, int capN, int nbk, const double* acc2, double* acc1, const float* eta, float n0, float n1, float n2, float* en, int* itb, int code, float* itf, float it2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[1] = code; itf[1] = it2; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) acc1[(size_t)b*capN + n] = acc2[(size_t)b*capN + n];
    en[n] = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
}
__global__ void cf_g0(const int* IP, const float* r0, float n0, float n1, float n2, float* gn, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    gn[n] = -(r0[n*3]*n0 + r0[n*3+1]*n1 + r0[n*3+2]*n2);
}
__global__ void cf_node(const int* IP, const float* sk, const int* pm, const float* me, const int* w8b, float* osk, int* opm, float* ome, int* ow8, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    osk[n] = sk[n]; opm[n] = pm[n]; ome[n] = me[n]; ow8[n] = w8b ? w8b[n] : -1;
}
// 끝 칸 -> 시작 칸 (프레임 첫 걸음 첫 평가 뒤 한 번): acc·en·gn·node
__global__ void cf_cp(const int* IP, int capN, int nbk, const double* acc1, double* acc0, const float* en1, float* en0, const float* gn1, float* gn0,
                      const float* sk1, float* sk0, const int* pm1, int* pm0, const float* me1, float* me0, const int* w81, int* w80,
                      int* itb, float* itf, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[0] = itb[1]; itf[0] = itf[1]; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) acc0[(size_t)b*capN + n] = acc1[(size_t)b*capN + n];
    en0[n] = en1[n]; gn0[n] = gn1[n]; sk0[n] = sk1[n]; pm0[n] = pm1[n]; me0[n] = me1[n]; w80[n] = w81[n];
}
__global__ void cf_frame(int* itb) { itb[0] = -1; itb[1] = -1; }
__global__ void sp_snap(const int* IP, const float* b, float n0, float n1, float n2, float* o, int* flg, int slot, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) flg[slot] = 1;
    if (n >= IP[9]) return;
    o[n] = b[n*3]*n0 + b[n*3+1]*n1 + b[n*3+2]*n2;
}
__global__ void sp_snap1(const int* IP, const float* b, float* o, int* flg, int slot, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) flg[slot] = 1;
    if (n >= IP[9]) return;
    o[n] = b[n];
}
// 절 CM117: 평가별 |r0|² (풀이기 잔차, 안 접은 것) -> gs[idx] (먼저 0 으로; 둘 다 AF_SKIP 지킴)
__global__ void sp_g0z(double* gs, int idx, const int* S) { if (cfs(S)) return; gs[idx] = 0.0; }
__global__ void sp_g2(const int* IP, const float* r0, double* gs, int idx, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v = 0.0;
    if (n < IP[9]) { float a = r0[n*3], b = r0[n*3+1], c = r0[n*3+2]; v = (double)a*a + (double)b*b + (double)c*c; }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0 && v != 0.0) atomicAdd(&gs[idx], v);
}
__global__ void sp_gclr(double* gs, int n, int* gc) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) { gs[i] = -1.0; gc[i] = 0; } }
__global__ void sp_gcnt(int* gc, int idx, const int* S) { if (cfs(S)) return; gc[idx] += 1; }
__global__ void sp_clr(int* flg, int n, int* itb, int m) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) flg[i] = 0; if (i < m) itb[i] = -1; }
// 걸음 x 평가 칸으로: 작업 통 -> acc[slot], n·η, 번호
__global__ void sp_fin(const int* IP, int capN, int nbk, const double* acc2, double* accs, const float* eta, float n0, float n1, float n2, float* en, int* itb, int slot, int code, float* itf, float it2, const int* S) {
    if (cfs(S)) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) { itb[slot] = code; itf[slot] = it2; }
    if (n >= IP[9] || n >= capN) return;
    for (int b=0;b<nbk;b++) accs[(size_t)b*capN + n] = acc2[(size_t)b*capN + n];
    en[n] = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
}
}

//@@ NL_SRC 1095

// ---- 절 CM119 nlwhy: 뉴턴 반복별 가우스 점 소성 상태 (읽기만, 상태 버퍼 따로) ----
__global__ void nl_pst(const int* IP, const int* cb, const float* BG, const float* Fq, const float* y, float cdp, int* stb, int* row) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (q == 0) atomicAdd(&row[7], 1);
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o_=0;o_<27;o_++) { float g[3]; gqn(F, BG + (oc*27+o_)*3, g); int id = ids[o_];
        for (int a=0;a<3;a++) { float ua = y[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    atomicAdd(&row[st], 1);
    int old = stb[q]; stb[q] = st;
    if (old >= 0 && old != st) {
        atomicAdd(&row[3], 1);
        if ((old == 0) != (st == 0)) atomicAdd(&row[4], 1);
        else atomicAdd(&row[5], 1);
    }
}

//@@ w_rotbc_wrap_KERNEL 3649

// [rotbc_wrap] PG enforce_particle_velocity_rotation: 원기둥 안 노드 (기준 위치) -> fixd, vbc = rs (r_⊥ x n) + ts n (r 은 현재 위치 = 기준 + U)
// RP 한 구속 12 칸: 점 3, 단위 법선 3, 반 높이, 반지름, 회전 몫 rs, 이동 몫 ts, (빈 칸), 켜짐
__global__ void rt_bc_rot(const int* IP, const float* Am, const float* U, const float* RP, int ib, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    const float* b = RP + ib*12;
    if (b[11] == 0.f) return;
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float yq[3] = {(float)(IP[0] + i), (float)(IP[1] + j), (float)(IP[2] + k)};
    float xr[3]; for (int a=0;a<3;a++) xr[a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    float o[3]; for (int a=0;a<3;a++) o[a] = xr[a] - b[a];
    float ax = o[0]*b[3] + o[1]*b[4] + o[2]*b[5];
    float r0 = o[0] - ax*b[3], r1 = o[1] - ax*b[4], r2 = o[2] - ax*b[5];
    if (!(fabsf(ax) < b[6] && sqrtf(r0*r0 + r1*r1 + r2*r2) < b[7])) return;
    float c[3]; for (int a=0;a<3;a++) c[a] = xr[a] + U[n*3+a] - b[a];
    float ac = c[0]*b[3] + c[1]*b[4] + c[2]*b[5];
    float q0 = c[0] - ac*b[3], q1 = c[1] - ac*b[4], q2 = c[2] - ac*b[5];
    float v0 = b[8]*(q1*b[5] - q2*b[4]) + b[9]*b[3];
    float v1 = b[8]*(q2*b[3] - q0*b[5]) + b[9]*b[4];
    float v2 = b[8]*(q0*b[4] - q1*b[3]) + b[9]*b[5];
    fixd[n] = 1; vbc[n*3] = v0; vbc[n*3+1] = v1; vbc[n*3+2] = v2;
}
// [rotbc_wrap 11-15] PG enforce_particle_translation (입자 선택 구속) -> 노드: 구속 입자 질량 몫 > 노드 질량 (mn, P2G) 절반 -> fixd, vbc = 구속 입자 질량 가중 속도
// TP 한 구속 4 칸: 속도 3, 켜짐. pid[원래 입자 번호] = 구속 번호 (-1 = 없음, 겹치면 config 뒤쪽). perm[p] = 정렬 입자 p 의 원래 번호.
// PA 노드 4 칸: 구속 질량, 구속 질량 x 속도 3 (속도 합은 0 아닌 구속 속도가 있을 때만, VN). 프레임 시작 위치 x 에서 이차 B-스플라인 (rt_p2g 와 같은 base·가중).
// 빠르게: 구속 입자만, 같은 base 칸 입자끼리 워프 안에서 고정 소수점 합 (__reduce_add_sync) 뒤 원자 합 하나.
__global__ void rt_trp_zero(const int* IP, float* PA) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<4;a++) PA[n*4+a] = 0.f;
}
__global__ void rt_trp_acc(int N, const int* IP, const float* Ai, const float* x, const float* m, const long long* perm, const int* pid, const float* TP, float mref, int VN, float* PA) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    int ib = (p < N && !IP[13]) ? pid[perm[p]] : -1;
    bool on = ib >= 0 && TP[ib*4+3] != 0.f;
    unsigned act = __ballot_sync(0xffffffffu, on);
    if (!on) return;
    int bs[3]; float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) {
        float y = Ai[a*3]*x[p*3] + Ai[a*3+1]*x[p*3+1] + Ai[a*3+2]*x[p*3+2];
        float fb = floorf(y - 0.5f);
        bs[a] = (int)fb - IP[a];
        bwf(y - fb, wx[a], dw[a]);
    }
    int nb1 = IP[7], nb2 = IP[8];
    int key = (bs[0]*nb1 + bs[1])*nb2 + bs[2];
    unsigned grp = __match_any_sync(act, key);
    bool lead = (__ffs(grp) - 1) == (threadIdx.x & 31);
    float mp = m[p], sc = 1048576.f / mref, isc = mref / 1048576.f;
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = key + (i*nb1 + j)*nb2 + k;
        float ww = wx[0][i]*wx[1][j]*wx[2][k]*mp;
        unsigned q = __reduce_add_sync(grp, (unsigned)__float2uint_rn(ww*sc));
        if (lead) atomicAdd(&PA[nid*4], (float)q*isc);
        if (VN) for (int a=0;a<3;a++) atomicAdd(&PA[nid*4+1+a], ww*TP[ib*4+a]);
    }
}
__global__ void rt_trp_fix(const int* IP, const float* mn, const float* PA, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float mc = PA[n*4];
    if (mc > 0.f && mc > 0.5f*mn[n]) { fixd[n] = 1; for (int a=0;a<3;a++) vbc[n*3+a] = PA[n*4+1+a]/mc; }
}

