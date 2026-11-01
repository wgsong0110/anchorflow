"""절 AK.22: 같은 격자 간격 2 차 B-스플라인 요소 판 CUDA 커널 (exe/fem_bsel_sim.py torch 판과 같은 식).
격자: 월드 고정 회전 정육면체 격자 (Kuhn 격자, 간격 dx), 격자 좌표 y = Ai x (간격 1). 프레임마다 점유 입자 둘레 상자 (조밀 노드 배열, 번호 = (i nby + j) nbz + k).
입자 (칸 번호로 정렬): base = floor(y − 0.5) (상자 안 칸 번호 cid), fx = y − base (프레임 시작 고정), 기저 w·∇w 는 커널 안에서 fx 로 다시 계산 (∇_x w = Aiᵀ ∇_y w).
서브스텝: (1) bs_force -- 입자마다 A = I + Σ_27 U ⊗ ∇w, X = A F, 표준 FCR P (전단 몫 가중 r = w_μ/V), G = V P Fᵀ, f_n −= G ∇w_n;
같은 칸 입자는 같은 27 노드라 워프 안 연속 구간 합 뒤 구간 머리 레인만 원자 합. (2) bs_contact -- 바닥 후보 입자 2 차 위치로 판정, 고착 질량 Σ w m_p w(s), 실제 접촉 표시.
(3) bs_node -- v ← (v + h (f/m' + g)) 감쇠, 고정 노드, 고착 (1 − a/m)^kex, 비관통, U += h v, f·acc 비움."""
import cupy as cp
import numpy as np

from . import fem_rt_cuda as RTC

_K = r'''
extern "C" {
__device__ __forceinline__ void bw(float f, float* w, float* d) {
    w[0] = 0.5f*(1.5f-f)*(1.5f-f); w[1] = 0.75f-(f-1.f)*(f-1.f); w[2] = 0.5f*(f-0.5f)*(f-0.5f);
    d[0] = f-1.5f; d[1] = -2.f*(f-1.f); d[2] = f-0.5f;
}
__device__ __forceinline__ void cof3f(const float* A, float* C) {
    C[0]=A[4]*A[8]-A[5]*A[7]; C[1]=A[5]*A[6]-A[3]*A[8]; C[2]=A[3]*A[7]-A[4]*A[6];
    C[3]=A[2]*A[7]-A[1]*A[8]; C[4]=A[0]*A[8]-A[2]*A[6]; C[5]=A[1]*A[6]-A[0]*A[7];
    C[6]=A[1]*A[5]-A[2]*A[4]; C[7]=A[2]*A[3]-A[0]*A[5]; C[8]=A[0]*A[4]-A[1]*A[3];
}
__device__ __forceinline__ float det3f(const float* A) { return A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]); }
__device__ __forceinline__ void polarf(const float* F, float* Q) {
    for (int k=0;k<9;k++) Q[k]=F[k];
    float C[9];
    for (int it=0; it<12; it++) {
        float dt = det3f(Q);
        float ds = (fabsf(dt) > 1e-30f) ? dt : 1e-30f;
        float gm = rcbrtf(fabsf(ds));
        cof3f(Q, C);
        float mx = 0.f;
        for (int k=0;k<9;k++) { float qn = 0.5f*(gm*Q[k] + C[k]/(ds*gm)); mx = fmaxf(mx, fabsf(qn - Q[k])); Q[k] = qn; }
        if (mx < 1e-7f) break;
    }
}
__device__ __forceinline__ void minvec(const float* S0, float* vmin) {
    float a[3][3], v[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) a[i][j] = 0.5f*(S0[i*3+j] + S0[j*3+i]);
    for (int sw=0; sw<8; sw++) for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
        if (fabsf(a[p][q]) < 1e-12f) continue;
        float th = 0.5f*(a[q][q]-a[p][p])/a[p][q];
        float t = (th >= 0.f ? 1.f : -1.f)/(fabsf(th) + sqrtf(th*th + 1.f));
        float c = rsqrtf(t*t + 1.f), sn = t*c;
        for (int k=0;k<3;k++) { float x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - sn*x2; a[k][q] = sn*x1 + c*x2; }
        for (int k=0;k<3;k++) { float x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - sn*x2; a[q][k] = sn*x1 + c*x2; }
        for (int k=0;k<3;k++) { float x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - sn*x2; v[k][q] = sn*x1 + c*x2; }
    }
    int im = 0; if (a[1][1] < a[im][im]) im = 1; if (a[2][2] < a[im][im]) im = 2;
    for (int k=0;k<3;k++) vmin[k] = v[k][im];
}
// 표준 FCR: P = r 2μ (X − R) + λ (J − 1) cof X  (R 진 회전)
__device__ __forceinline__ void fcrP(const float* X, float mu, float lam, float r, float* P) {
    float Q[9], cf[9];
    polarf(X, Q); cof3f(X, cf);
    float dt = det3f(X);
    if (dt < 0.f) {
        float S[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += Q[k*3+a]*X[k*3+b]; S[a*3+b] = t; }
        float v3[3]; minvec(S, v3);
        float u3[3]; for (int a=0;a<3;a++) u3[a] = Q[a*3]*v3[0] + Q[a*3+1]*v3[1] + Q[a*3+2]*v3[2];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) Q[a*3+b] -= 2.f*u3[a]*v3[b];
    }
    for (int k=0;k<9;k++) P[k] = r*2.f*mu*(X[k]-Q[k]) + lam*(dt-1.f)*cf[k];
}
// 칸 cid 의 27 노드 번호 (상자 안): base = (cid / (nby nbz), ...), 노드 (bx+i, by+j, bz+k)
__device__ __forceinline__ int nodeid(int cid, int o, const int* nb) {
    int nby = nb[1], nbz = nb[2];
    int bz = cid % nbz, by = (cid / nbz) % nby, bx = cid / (nbz*nby);
    int i = o / 9, j = (o / 3) % 3, k = o % 3;
    return ((bx+i)*nby + (by+j))*nbz + (bz+k);
}
// (1) 힘: 입자 스레드 (칸 정렬), 노드 오프셋마다 워프 구간 합 -> 머리 레인 원자 합
__global__ void bs_force(int N, const int* cid, const float* fx, const float* F, const float* V, const float* r, const float* Ai, const int* nb,
                         const float* U, float mu, float lam, float* f) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = p < N;
    int c = on ? cid[p] : -1 - lane;
    int cp_ = __shfl_up_sync(0xffffffffu, c, 1);
    bool head = (lane == 0) || (cp_ != c);
    float w[3][3], dw[3][3], G[9] = {0,0,0,0,0,0,0,0,0}, ai[9];
    for (int k=0;k<9;k++) ai[k] = Ai[k];
    if (on) {
        for (int a=0;a<3;a++) bw(fx[p*3+a], w[a], dw[a]);
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        for (int o=0;o<27;o++) {
            int i = o / 9, j = (o / 3) % 3, k = o % 3;
            float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
            float gx[3]; for (int b=0;b<3;b++) gx[b] = ai[0*3+b]*gy[0] + ai[1*3+b]*gy[1] + ai[2*3+b]*gy[2];   // ∇_x = Aiᵀ ∇_y
            int n = nodeid(c, o, nb);
            for (int a=0;a<3;a++) { float u = U[n*3+a]; for (int b=0;b<3;b++) A[a*3+b] += u*gx[b]; }
        }
        float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[p*9+k];
        float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; X[a*3+b] = t; }
        float P[9]; fcrP(X, mu, lam, r[p], P);
        float vp = V[p];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += P[a*3+k]*Fp[b*3+k]; G[a*3+b] = vp*t; }
    } else {
        for (int a=0;a<3;a++) for (int q=0;q<3;q++) { w[a][q] = 0.f; dw[a][q] = 0.f; }
    }
    for (int o=0;o<27;o++) {
        int i = o / 9, j = (o / 3) % 3, k = o % 3;
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = ai[0*3+b]*gy[0] + ai[1*3+b]*gy[1] + ai[2*3+b]*gy[2];
        float s[3]; for (int a=0;a<3;a++) s[a] = -(G[a*3]*gx[0] + G[a*3+1]*gx[1] + G[a*3+2]*gx[2]);
        for (int off=1; off<32; off<<=1) {
            int to = __shfl_down_sync(0xffffffffu, c, off);
            bool same = (lane + off < 32) && (to == c);
            for (int a=0;a<3;a++) { float t = __shfl_down_sync(0xffffffffu, s[a], off); if (same) s[a] += t; }
        }
        if (on && head) { int n = nodeid(c, o, nb); for (int a=0;a<3;a++) atomicAdd(&f[n*3+a], s[a]); }
    }
}
__device__ __forceinline__ float wstk(float s, float pn, float dxg) {
    float u = (s + pn) / dxg;
    float bf = floorf(u - 0.5f);
    float fx = u - bf;
    float w[3] = {0.5f*(1.5f-fx)*(1.5f-fx), 0.75f-(fx-1.f)*(fx-1.f), 0.5f*(fx-0.5f)*(fx-0.5f)};
    float o = 0.f;
    for (int i=0;i<3;i++) { float k = bf + (float)i; if (k*dxg - pn < 0.f) o += w[i]; }
    return o;
}
// (2) 바닥 접촉 (후보 목록, 개수 nc[0])
__global__ void bs_contact(const int* nc, const int* cand, const int* cid, const float* fx, const float* x, const float* m, const int* nb, const float* U,
                           float p0, float p1, float p2, float n0, float n1, float n2, float pn, float dxg, float npd, float* acc, float* accN) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nc[0]) return;
    int p = cand[q];
    int c = cid[p];
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bw(fx[p*3+a], w[a], dw[a]);
    float u[3] = {0,0,0};
    for (int o=0;o<27;o++) { int i = o / 9, j = (o / 3) % 3, k = o % 3; float ww = w[0][i]*w[1][j]*w[2][k]; int n = nodeid(c, o, nb);
        for (int a=0;a<3;a++) u[a] += ww*U[n*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float wt = wstk(s, pn, dxg)*m[p];
    bool real = s < -npd;
    for (int o=0;o<27;o++) { int i = o / 9, j = (o / 3) % 3, k = o % 3; float ww = w[0][i]*w[1][j]*w[2][k]; int n = nodeid(c, o, nb);
        if (wt > 0.f) atomicAdd(&acc[n], ww*wt); if (real) atomicAdd(&accN[n], 1.f); }
}
// (3) 노드 걸음 + 이동
__global__ void bs_node(const int* nbt, const float* mn, const float* mI, float* f, float* vn, float* U, float g0, float g1, float g2, float hs, float damp,
                        const unsigned char* fixd, const float* vbc, float* acc, float* accN, int floor_, float kex, float n0, float n1, float n2, int vout) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= nbt[0]) return;
    float m = mn[n];
    float gg[3] = {g0, g1, g2};
    float v[3];
    for (int a=0;a<3;a++) { v[a] = (m > 0.f) ? (vn[n*3+a] + hs*(f[n*3+a]/mI[n] + gg[a]))*damp : 0.f; f[n*3+a] = 0.f; }
    if (fixd && fixd[n]) for (int a=0;a<3;a++) v[a] = vbc[n*3+a];
    if (floor_) {
        float ac = acc[n]; acc[n] = 0.f; float cn = accN[n]; accN[n] = 0.f;
        if (!(fixd && fixd[n])) {
            if (ac > 0.f && m > 0.f) { float k = powf(fmaxf(1.f - ac/m, 0.f), kex);
                float vv = v[0]*n0 + v[1]*n1 + v[2]*n2;                            // 절 AK.31 vout: 바깥 법선 (이탈) 성분은 고착하지 않음
                if (vout && vv > 0.f) for (int a=0;a<3;a++) { float nn = (a == 0) ? n0 : ((a == 1) ? n1 : n2); v[a] = k*(v[a] - vv*nn) + vv*nn; }
                else for (int a=0;a<3;a++) v[a] *= k; }
            if (cn > 0.f) { float vv = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vv < 0.f) { v[0] -= vv*n0; v[1] -= vv*n1; v[2] -= vv*n2; } }
        }
    }
    for (int a=0;a<3;a++) { vn[n*3+a] = v[a]; U[n*3+a] += hs*v[a]; }
}
// G2P
__global__ void bs_g2p(int N, const int* cid, const float* fx, const float* Ai, const int* nb, const float* U, const float* vn,
                       const float* x, const float* F, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int c = cid[p];
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bw(fx[p*3+a], w[a], dw[a]);
    double u[3] = {0,0,0}, vv[3] = {0,0,0};
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o / 9, j = (o / 3) % 3, k = o % 3;
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        int n = nodeid(c, o, nb);
        for (int a=0;a<3;a++) { float uu = U[n*3+a]; u[a] += ww*uu; vv[a] += ww*vn[n*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    for (int a=0;a<3;a++) { x2[p*3+a] = (float)(x[p*3+a] + u[a]); v2[p*3+a] = (float)vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*F[p*9+k*3+b]; F2[p*9+a*3+b] = t; }
}
}
'''
_MOD = [None]


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def prepare():
    """그래프 캡처 전에 모듈 컴파일·함수 적재 (캡처 안에서 NVRTC 가 돌지 않게)."""
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=_K.encode("ascii", "ignore").decode(), options=("--std=c++14",))
    for nm in ("bs_force", "bs_contact", "bs_node", "bs_g2p"):
        _MOD[0].get_function(nm)


def launch(name, n, args, b=128):
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=_K.encode("ascii", "ignore").decode(), options=("--std=c++14",))
    with RTC._Stream():
        _MOD[0].get_function(name)(((int(n) + b - 1) // b,), (b,), args)
