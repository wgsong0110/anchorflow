"""절 AN 실시간 판: 프레임 전체 (P2G -> 고정 예산 암시적 풀이 -> G2P) 를 손으로 쓴 CUDA 커널로, 고정 용량 버퍼, 프레임마다 CUDA 그래프 재생.
정의는 exe/fem_an_sim.py --fast (블록 야코비 전처리) 와 같다. 정밀도: 곱·누적 f32, 기울기 (잔차)·내적·에너지 f64.
- 상자 (입자 둘레 조밀 노드 상자) 와 칸 압축 (점유 칸 -> 연속 번호, 한 블록 스캔) 을 장치에서 계산 -> 호스트 동기화 없음. 용량을 넘으면 넘침 표시 -> 호스트가 프레임 끝에 보고 다시 할당·캡처·그 프레임 다시.
- 가우스 점 q = 압축 칸 c x 8 + 팔분 o; 기저·기울기는 팔분 8 가지 상수 표 (BW, BG) 에서, G_qn = F_qᵀ ∇w 는 커널 안에서 계산 (저장 안 함).
- 입자 -> 가우스 점 F (절 AI A1 회전·신장 분리 평균), 노드 질량·운동량: 입자마다 원자 합 (f32)."""
import math
import os

import cupy as cp
import numpy as np
import torch

from . import fem_rt_cuda as RTC
from . import fem_an_mg2 as MG2

SRC = r'''
#define NQ8 8
extern "C" {
__device__ __forceinline__ float seg8f(float s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double seg8d(double s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double warpsum(double s) {
    for (int o=16;o>0;o>>=1) s += __shfl_xor_sync(0xffffffffu, s, o); return s;
}
// 블록 합 (f64) -> 원자 합 하나
__device__ void blocksum_add(double v, double* out) {
    __shared__ double sh[32];
    v = warpsum(v);
    int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) { v = (l < (blockDim.x >> 5)) ? sh[l] : 0.0; v = warpsum(v); if (l == 0) atomicAdd(out, v); }
    __syncthreads();
}
__device__ void jac3f(float a[3][3], float v[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) v[i][j] = (i==j) ? 1.f : 0.f;
    for (int sw=0; sw<20; sw++) {
        float off = a[0][1]*a[0][1] + a[0][2]*a[0][2] + a[1][2]*a[1][2];
        float dg = a[0][0]*a[0][0] + a[1][1]*a[1][1] + a[2][2]*a[2][2];
        if (!(off > 1e-14f*dg)) break;
        for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
            if (a[p][q] == 0.f) continue;
            float th = 0.5f*(a[q][q]-a[p][p])/a[p][q];
            float t = (th >= 0.f ? 1.f : -1.f)/(fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), s = t*c;
            for (int k=0;k<3;k++) { float x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - s*x2; a[k][q] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - s*x2; a[q][k] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - s*x2; v[k][q] = s*x1 + c*x2; }
        }
    }
}
// 부호 있는 SVD (f32), fem_an_cuda.ssvd 와 같은 식
__device__ void ssvdf(const float* X, float* U, float* S, float* V) {
    float A[3][3], v[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { float t = 0.f; for (int k=0;k<3;k++) t += X[k*3+i]*X[k*3+j]; A[i][j] = t; }
    jac3f(A, v);
    int o[3] = {0,1,2}; float e[3] = {A[0][0], A[1][1], A[2][2]};
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    if (e[o[1]] < e[o[2]]) { int t = o[1]; o[1] = o[2]; o[2] = t; }
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    for (int i=0;i<3;i++) for (int c=0;c<3;c++) V[i*3+c] = v[i][o[c]];
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) for (int i=0;i<3;i++) V[i*3+2] = -V[i*3+2];
    float b[3][3];
    for (int c=0;c<3;c++) for (int a=0;a<3;a++) { float t = 0.f; for (int k=0;k<3;k++) t += X[a*3+k]*V[k*3+c]; b[c][a] = t; }
    float u1[3], u2[3], u3[3];
    float n1 = sqrtf(b[0][0]*b[0][0] + b[0][1]*b[0][1] + b[0][2]*b[0][2]);
    if (n1 > 0.f) for (int a=0;a<3;a++) u1[a] = b[0][a]/n1; else { u1[0] = 1.f; u1[1] = 0.f; u1[2] = 0.f; }
    float d12 = u1[0]*b[1][0] + u1[1]*b[1][1] + u1[2]*b[1][2];
    for (int a=0;a<3;a++) u2[a] = b[1][a] - d12*u1[a];
    float n2 = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
    if (n2 > 1e-6f*n1 && n2 > 0.f) { for (int a=0;a<3;a++) u2[a] /= n2; }
    else {
        int ax = 0; if (fabsf(u1[1]) < fabsf(u1[ax])) ax = 1; if (fabsf(u1[2]) < fabsf(u1[ax])) ax = 2;
        float ea[3] = {0.f, 0.f, 0.f}; ea[ax] = 1.f;
        u2[0] = u1[1]*ea[2]-u1[2]*ea[1]; u2[1] = u1[2]*ea[0]-u1[0]*ea[2]; u2[2] = u1[0]*ea[1]-u1[1]*ea[0];
        float nn = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]); for (int a=0;a<3;a++) u2[a] /= nn;
    }
    u3[0] = u1[1]*u2[2]-u1[2]*u2[1]; u3[1] = u1[2]*u2[0]-u1[0]*u2[2]; u3[2] = u1[0]*u2[1]-u1[1]*u2[0];
    for (int a=0;a<3;a++) { U[a*3] = u1[a]; U[a*3+1] = u2[a]; U[a*3+2] = u3[a]; }
    S[0] = u1[0]*b[0][0] + u1[1]*b[0][1] + u1[2]*b[0][2];
    S[1] = u2[0]*b[1][0] + u2[1]*b[1][1] + u2[2]*b[1][2];
    S[2] = u3[0]*b[2][0] + u3[1]*b[2][1] + u3[2]*b[2][2];
}
__device__ __forceinline__ void bwf(float f, float* w, float* d) {
    w[0] = 0.5f*(1.5f-f)*(1.5f-f); w[1] = 0.75f-(f-1.f)*(f-1.f); w[2] = 0.5f*(f-0.5f)*(f-0.5f);
    d[0] = f-1.5f; d[1] = -2.f*(f-1.f); d[2] = f-0.5f;
}
// IP (int): 0..2 bmin, 3..5 bmax, 6..8 nb (노드 상자), 9 노드 수, 10 칸 상자 수, 11 압축 칸 수 nc, 12 가우스 점 수 nq, 13 넘침
// ---- P2G ----
__global__ void rt_reset(int* IP) { IP[0] = IP[1] = IP[2] = 2147483647; IP[3] = IP[4] = IP[5] = -2147483647; IP[13] = 0; IP[14] = 0; }
__global__ void rt_coord(int N, const float* x, const float* Ai, int* pb, float* fx, int* IP) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = p < N;
    if (__all_sync(0xffffffffu, !on)) return;
    for (int a=0;a<3;a++) {                                                    // 상자 최소·최대: 워프 안에서 먼저 줄인 뒤 원자 연산 하나 (정수라 결과 같음, 경합만 줄임)
        int b0 = 2147483647, b1 = -2147483647;
        if (on) {
            float y = Ai[a*3]*x[p*3] + Ai[a*3+1]*x[p*3+1] + Ai[a*3+2]*x[p*3+2];
            float fb = floorf(y - 0.5f);
            int b = (int)fb; pb[p*3+a] = b; fx[p*3+a] = y - fb; b0 = b; b1 = b;
        }
        b0 = __reduce_min_sync(0xffffffffu, b0); b1 = __reduce_max_sync(0xffffffffu, b1);
        if ((threadIdx.x & 31) == 0) { atomicMin(&IP[a], b0); atomicMax(&IP[3+a], b1); }
    }
}
__global__ void rt_box(int* IP, int capN, int capC) {
    int nb0 = IP[3]-IP[0]+3, nb1 = IP[4]-IP[1]+3, nb2 = IP[5]-IP[2]+3;
    IP[6] = nb0; IP[7] = nb1; IP[8] = nb2;
    long long nn = (long long)nb0*nb1*nb2, nc = (long long)(nb0-2)*(nb1-2)*(nb2-2);
    IP[9] = (int)nn; IP[10] = (int)nc;
    if (nn > capN || nc > capC) { IP[13] = 1; IP[9] = 0; IP[10] = 0; }
}
__global__ void rt_zero_f(int n, const int* cnt, int ci, int mul, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.f; }
__global__ void rt_zero_d(int n, const int* cnt, int ci, int mul, double* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.0; }
__global__ void rt_zero_i(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
// 입자: 상자 안 base, 칸 상자 번호 표시
__global__ void rt_mark(int N, int* pb, const float* fx, const int* IP, int* pcd, int* occ) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int b0 = pb[p*3]-IP[0], b1 = pb[p*3+1]-IP[1], b2 = pb[p*3+2]-IP[2];
    pb[p*3] = b0; pb[p*3+1] = b1; pb[p*3+2] = b2;
    int cd = (b0*(IP[7]-2) + b1)*(IP[8]-2) + b2;
    pcd[p] = cd; occ[cd] = 1;
}
// 한 블록 배타 스캔: occ -> cidx (압축 번호, 비점유 -1), 칸 base 좌표 cb, nc·nq
__global__ void rt_scan(int* IP, const int* occ, int* cidx, int* cb, int capQc) {
    __shared__ int sh[1024];
    int n = IP[13] ? 0 : IP[10];
    int carry = 0;
    int n1 = IP[7]-2, n2 = IP[8]-2;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n) ? occ[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        int inc = sh[threadIdx.x];
        if (i < n) {
            int id = carry + inc - v;
            if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
            else cidx[i] = -1;
        }
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
// 입자 -> 가우스 점 누적 (회전 R, log S, F, 가중) + 칸 누적 + 칸 부피·질량 + 노드 질량·운동량
// A: [q*28]: 0..8 R, 9..17 LS, 18..26 F, 27 w;  C: [c*30]: 0..27 같은 것, 28 V, 29 M;  bad: q 와 c (int, 뒤집힌 입자)
__global__ void rt_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const int* pb, const float* fx, const int* pcd,
                       const int* cidx, const int* IP, float* A, float* C, int* badq, int* badc, float* mn, float* mv, const float* pa, float* ma) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float U[9], S[3], V[9];
    ssvdf(Fp, U, S, V);
    float w = vol[p];
    bool ng = !(S[2] > 0.f);
    float R[9], L[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
    float* aq = A + (size_t)q*28; float* ac = C + (size_t)c*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*L[k]); atomicAdd(&ac[18+k], w*Fp[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w); atomicAdd(&ac[29], m[p]);
#ifdef AF_BAR
    for (int a=0;a<3;a++) atomicAdd(&AF_BX[q*4+a], m[p]*x[p*3+a]); atomicAdd(&AF_BX[q*4+3], m[p]);   // 장벽 판: 팔분 점 입자 질량 중심 (표본점)
#endif
#ifdef AF_OCT
    atomicAdd(&AF_MO[q], m[p]);                                                     // 팔분 칸 실제 입자 질량 (점유 가중, 절 AN.err4 정의)
#endif
    if (ng) { atomicOr(&badq[q], 1); atomicOr(&badc[c], 1); }
    float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], wx[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = wx[0][i]*wx[1][j]*wx[2][k]*mp;
        atomicAdd(&mn[nid], ww);
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], ww*v[p*3+a]);
        if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[nid*3+a], ww*pa[p*3+a]);   // generalized-α: 입자 가속도 -> 노드 (질량 가중)
    }
}
__device__ void rs_fin(const float* sR, const float* sL, float w, float* out, bool* bad) {
    float U[9], S[3], V[9];
    ssvdf(sR, U, S, V);
    float Rb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Rb[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float Lm[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Lm[i][j] = 0.5f*(sL[i*3+j] + sL[j*3+i])/w;
    jac3f(Lm, Ev);
    float ex[3]; for (int i=0;i<3;i++) ex[i] = expf(Lm[i][i]);
    float Sb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Sb[a*3+b] = Ev[a][0]*ex[0]*Ev[b][0] + Ev[a][1]*ex[1]*Ev[b][1] + Ev[a][2]*ex[2]*Ev[b][2];
    *bad = false;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = Rb[a*3]*Sb[b] + Rb[a*3+1]*Sb[3+b] + Rb[a*3+2]*Sb[6+b]; out[a*3+b] = t; if (!isfinite(t)) *bad = true; }
}
// 가우스 점 마무리: F_q, V_q, m_q, 접촉용 시작 위치 xq0 (월드), 노드 몫 질량 mqn (원자 합)
__global__ void rt_gauss(const int* IP, const float* A, const float* C, const int* badq, const int* badc, const int* cb, const float* Am, const float* BW,
                         float* Fq, float* Vq, float* mq, float* xq0, float* mqn, int* nbad) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (aq[27] > 0.f) {
        rs_fin(aq, aq + 9, aq[27], out, &bad);
        if (bad || badq[q]) { for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; atomicAdd(nbad, 1); }
    } else {
        rs_fin(ac, ac + 9, ac[27], out, &bad);
        if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27];
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f;
#ifdef AF_OCT
    Vq[q] = (1.f - AF_GOCT)*aq[27] + AF_GOCT*Vq[q]; mm = (1.f - AF_GOCT)*AF_MO[q] + AF_GOCT*mm;   // 점유 가중: (1 − κ) 팔분 칸 실제 부피·질량 + κ 칸 합 / 8
#endif
    mq[q] = mm;
    const float GG = 0.28867513459481287f;
    float yq[3] = {(float)(IP[0] + cb[c*3]) + 1.f + ((oc & 4) ? GG : -GG), (float)(IP[1] + cb[c*3+1]) + 1.f + ((oc & 2) ? GG : -GG), (float)(IP[2] + cb[c*3+2]) + 1.f + ((oc & 1) ? GG : -GG)};
    for (int a=0;a<3;a++) xq0[q*3+a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        atomicAdd(&mqn[nid], BW[oc*27+o]*mm);
    }
}
__global__ void rt_vn(const int* IP, const float* mn, const float* mv, float* vk, float* U) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) { vk[n*3+a] = (m > 0.f) ? mv[n*3+a]/m : 0.f; U[n*3+a] = 0.f; }
}
// ---- 풀이 ----
// 가우스 점 공통: 27 노드 번호, G_qn = F_qᵀ ∇w (BG 월드 기울기 표)
__device__ __forceinline__ void qnodes(int q, const int* IP, const int* cb, int* ids) {
    int c = q >> 3; int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k); }
}
__device__ __forceinline__ void gqn(const float* Fq, const float* bg, float* g) {
    g[0] = Fq[0]*bg[0] + Fq[3]*bg[1] + Fq[6]*bg[2];
    g[1] = Fq[1]*bg[0] + Fq[4]*bg[1] + Fq[7]*bg[2];
    g[2] = Fq[2]*bg[0] + Fq[5]*bg[1] + Fq[8]*bg[2];
}
// 걸음 시작: Ut = U + τ v + τ² g, z0 = U + β τ v, y = Ut (고정 노드는 U + τ vbc)
__global__ void rt_step0(const int* IP, const float* U, const float* vk, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                         float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], vv = vk[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? u + tau*vbc[n*3+a] : ut;
    }
}
// 바닥: 가우스 점 걸음 끝 위치 -> 정답 sticky 가중 (s < 0 만) -> 노드 몫 an (원자 합)
__global__ void rt_floor_q(const int* IP, const int* cb, const float* BW, const float* xq0, const float* mq, const float* y, const float* U, float thinv,
                           float p0, float p1, float p2, float n0, float n1, float n2, float pn, float dxg, float* an) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
#ifdef AF_BAR
    float bm_ = AF_BX[q*4+3]; if (bm_ > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[q*4+a]/bm_;   // 장벽 판: 고착 판정도 표본점 (팔분 입자 질량 중심) 에서
#endif
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }
    float s = (xq[0]-p0)*n0 + (xq[1]-p1)*n1 + (xq[2]-p2)*n2;
#ifdef AF_BAR
    float sw_ = s; s = (bm_ > 0.f) ? -1.f : 1.f;                                 // 장벽 판: 정답 sticky 가중 전체 (평면 위 띠 포함), 빈 팔분 점 없음
#else
    float sw_ = s;
#endif
    if (!(s < 0.f)) return;
    float u = (sw_ + pn)/dxg; float bf = floorf(u - 0.5f); float f = u - bf;
    float w3[3] = {0.5f*(1.5f-f)*(1.5f-f), 0.75f-(f-1.f)*(f-1.f), 0.5f*(f-0.5f)*(f-0.5f)};
    float wst = 0.f; for (int i=0;i<3;i++) if ((bf + (float)i)*dxg - pn < 0.f) wst += w3[i];
    float mw = mq[q]*wst;
#ifdef AF_FLJ
    mw = mq[q]*(1.f - expf(AF_KEXQ*log1pf(-fminf(wst, 1.f))));                            // 점별 고착 (--fl_pt, 절 AN.17): 점 질량 x (1 − (1 − w_st)^(Δt/dt_정답)) -- 거듭제곱을 점에서 한 뒤 노드로 평균
#endif
    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);
}
// 노드: 고착 인자 k, 변수 바꿈 (sk, pm), ŷ, η, y = ŷ + T η. pm 0 = P I, 1 = P (I − nnᵀ) (바깥 법선 이탈), 2 = 고정 (T 0)
__device__ __forceinline__ void Tapply(float sk, int pm, float n0, float n1, float n2, const float* v, float* o) {
    if (pm == 2) { o[0] = o[1] = o[2] = 0.f; return; }
    if (pm == 0) { o[0] = sk*v[0]; o[1] = sk*v[1]; o[2] = sk*v[2]; return; }
    float vn = v[0]*n0 + v[1]*n1 + v[2]*n2;                                   // P v = v − (v·n) n (접선), (I − P) v = (v·n) n
    o[0] = sk*(v[0]-vn*n0) + vn*n0; o[1] = sk*(v[1]-vn*n1) + vn*n1; o[2] = sk*(v[2]-vn*n2) + vn*n2;
}
__global__ void rt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
#ifdef AF_FLJ
        k = 1.f - r;                                                             // 점별 고착: an 이 이미 거듭제곱한 몫의 질량 합
#endif
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
#ifdef AF_BAR
        p = 1;                                                                   // 장벽 판: 법선은 장벽, 고착은 접선에만
#endif
    }
    if (fixd && fixd[n]) p = 2;
    float s = sqrtf(k);
    float d[3], yy[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] - Ut[n*3+a];
    if (p == 2) { for (int a=0;a<3;a++) { yy[a] = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy[a]; eta[n*3+a] = 0.f; y[n*3+a] = yy[a]; } sk[n] = 0.f; pm[n] = 2; return; }
    float Pd[3];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 요소: mode 1 기울기 (f64 원자, 8 레인 구간 합), 2 헤시안 (f32 저장), 4 에너지 (f64 블록 합)
__global__ void __launch_bounds__(128) rt_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, int mode,
                        double* E, double* f, float* H, int capQ, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float J = S[0]*S[1]*S[2];
    float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    float vq = on ? Vq[qq] : 0.f;
    if (mode & 4) {
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*((double)J-1.0)*((double)J-1.0)) : 0.0;
        blocksum_add(e, E);
    }
    if (mode & 1) {
        float ph[3]; for (int i=0;i<3;i++) ph[i] = 2.f*mu*(S[i]-1.f) + lam*(J-1.f)*gJ[i];
        float P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*ph[0]*V[b*3] + U[a*3+1]*ph[1]*V[b*3+1] + U[a*3+2]*ph[2]*V[b*3+2];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
            for (int a=0;a<3;a++) { double s = seg8d((double)(vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2])));
                if (on && head) atomicAdd(&f[ids[o]*3+a], s); } }
    }
    if (mode & 2) {                                                             // 헤시안 (모든 레인이 들어옴: 대각 구간 합 때문에; 밖 레인은 0)
        float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
        if (on) {
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
        for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];                     // 성분 우선 배치 (곱 커널 읽기가 합쳐지게)
        }
        if (mode & 8) {                                                           // 노드 블록 대각 (융합): m_ac = V Σ_b g_b Σ_d H[(ab),(cd)] g_d
            for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
                float m[9];
                for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                    for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
                    m[a*3+c] = vq*t; }
                for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
                if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
        }
    }
}
// 노드 블록 대각 Σ V Σ_bd H[(ab),(cd)] g_b g_d (f32, 8 레인 구간 합)
__global__ void __launch_bounds__(128) rt_bd(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    const float* h = H + (size_t)qq*81;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 헤시안 곱 out += Σ V (H δX) g, δX = Σ dy ⊗ g (f32)
__global__ void __launch_bounds__(128) rt_mv(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*g[0]; dX[a*3+1] += ya*g[1]; dX[a*3+2] += ya*g[2]; } }
    float vq = on ? Vq[qq] : 0.f;
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); } }
}
// 노드: 기울기 g = Mi η + T f (f64), r = −g (f32), 블록 야코비 Dinv = (Mi I + T B T)⁻¹, z = D r, p = z; 스칼라 SC: 0 rz, 1 |g|², ...
__device__ __forceinline__ void inv3(const float* M, float* Mi) {
    float c0 = M[4]*M[8]-M[5]*M[7], c1 = M[5]*M[6]-M[3]*M[8], c2 = M[3]*M[7]-M[4]*M[6];
    float d = M[0]*c0 + M[1]*c1 + M[2]*c2;
    if (d == 0.f || !isfinite(d)) { for (int k=0;k<9;k++) Mi[k] = 0.f; return; }
    float id = 1.f/d;
    Mi[0] = c0*id; Mi[1] = (M[2]*M[7]-M[1]*M[8])*id; Mi[2] = (M[1]*M[5]-M[2]*M[4])*id;
    Mi[3] = c1*id; Mi[4] = (M[0]*M[8]-M[2]*M[6])*id; Mi[5] = (M[2]*M[3]-M[0]*M[5])*id;
    Mi[6] = c2*id; Mi[7] = (M[1]*M[6]-M[0]*M[7])*id; Mi[8] = (M[0]*M[4]-M[1]*M[3])*id;
}
__device__ __forceinline__ void Tmat(float sk, int pm, float n0, float n1, float n2, float* T) {
    float nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        float I = (a==b) ? 1.f : 0.f;
        if (pm == 2) T[a*3+b] = 0.f; else if (pm == 0) T[a*3+b] = sk*I; else { float Pab = I - nn[a]*nn[b]; T[a*3+b] = sk*Pab + (I - Pab); }
    }
}
__global__ void rt_grad(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* eta, const double* f,
                        const float* B, float* Dinv, float* r, float* z, float* p, float* dl, float* Tp, double* SC, float* r0) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0, gg = 0.0, ff = 0.0;
    if (n < IP[9]) {
        float T[9]; Tmat(sk[n], pm[n], n0, n1, n2, T);
        float Mi = mn[n]*itau2;
        double g[3];
        for (int a=0;a<3;a++) g[a] = (double)Mi*eta[n*3+a] + (double)T[a*3]*f[n*3] + (double)T[a*3+1]*f[n*3+1] + (double)T[a*3+2]*f[n*3+2];
#ifdef AF_ADAPT
        for (int a=0;a<3;a++) { double tf = g[a] - (double)Mi*eta[n*3+a]; ff += tf*tf; }      // 탄성·접촉 힘 몫 |T f|² (절 AN.14 기준 척도)
#endif
        float M[9], TB[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += T[a*3+k]*B[n*9+k*3+b]; TB[a*3+b] = t; }
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = (a==b) ? Mi : 0.f; for (int k=0;k<3;k++) t += TB[a*3+k]*T[k*3+b]; M[a*3+b] = t; }
        float Di[9]; inv3(M, Di);
        for (int k=0;k<9;k++) Dinv[n*9+k] = Di[k];
        float rr[3]; for (int a=0;a<3;a++) { rr[a] = (float)(-g[a]); r[n*3+a] = rr[a]; r0[n*3+a] = rr[a]; dl[n*3+a] = 0.f; gg += g[a]*g[a]; }
        float zz[3]; for (int a=0;a<3;a++) { zz[a] = Di[a*3]*rr[0] + Di[a*3+1]*rr[1] + Di[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]); blocksum_add(gg, &SC[1]);
#ifdef AF_ADAPT
    blocksum_add(ff, &SC[56]);
#endif
}
// Ap = Mi p + T (K T p) 의 노드 마무리 + p·Ap (SC[2])
__global__ void rt_ap(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* p, const float* KTp, float* Ap, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) {
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, KTp + n*3, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float v = Mi*p[n*3+a] + t[a]; Ap[n*3+a] = v; s += (double)p[n*3+a]*v; }
    }
    blocksum_add(s, &SC[2]);
}
// α = rz/pAp; dl += α p; r −= α Ap; zo = z; z = D r; SC[3] += r·z, SC[4] += r·(z − zo)
__global__ void rt_cg1(const int* IP, const double* SC, const float* Dinv, const float* p, const float* Ap, float* dl, float* r, float* z, double* SC2) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0, s5 = 0.0;
    if (n < IP[9]) {
        double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
        float rr[3];
        for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; rr[a] = r[n*3+a] - al*Ap[n*3+a]; r[n*3+a] = rr[a]; }
        const float* D = Dinv + n*9;
        for (int a=0;a<3;a++) { float zn = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; float zo = z[n*3+a]; z[n*3+a] = zn; s3 += (double)rr[a]*zn; s4 += (double)rr[a]*(zn - zo); s5 += (double)rr[a]*rr[a]; }
    }
    blocksum_add(s3, &SC2[3]); blocksum_add(s4, &SC2[4]);
#ifdef AF_ADAPT
    blocksum_add(s5, &SC2[57]);
#endif
}
// β = SC4/SC0 (폴라크-리비에르); p = z + β p; Tp = T p; 다음을 위해 SC0 ← SC3 (단일 스레드 쪽에서)
__global__ void rt_cg2(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* z, float* p, float* Tp, float* KTz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (KTz) { KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f; }                // 다음 헤시안 곱의 원자 합 바탕 (0 채우기 커널 융합)
    float be = (SC[0] > 0.0) ? (float)(SC[4]/SC[0]) : 0.f;
    float pp[3]; for (int a=0;a<3;a++) { pp[a] = z[n*3+a] + be*p[n*3+a]; p[n*3+a] = pp[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, pp, t); for (int a=0;a<3;a++) Tp[n*3+a] = t[a];
}
__global__ void rt_sc_roll(double* SC) { SC[0] = SC[3]; SC[2] = 0.0; SC[3] = 0.0; SC[4] = 0.0; }
__global__ void rt_sc_clear(double* SC, int a, int b) { for (int i=a;i<b;i++) SC[i] = 0.0; }
// 선탐색: 후보 c 의 y_c = ŷ + T (η + c δ) 와 관성 몫 ½ Σ Mi |η + c δ|² (SC[slot]), 그리고 g·δ (SC[9])
__global__ void rt_lsy(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl,
                       float c, float* yc, double* SC, int slot, const float* r) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0, gd = 0.0;
    if (n < IP[9]) {
        float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + c*dl[n*3+a];
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { yc[n*3+a] = yh[n*3+a] + t[a]; s += 0.5*(double)Mi*e[a]*e[a]; if (r) gd += -(double)r[n*3+a]*dl[n*3+a]; }
    }
    blocksum_add(s, &SC[slot]);
    if (r) blocksum_add(gd, &SC[9]);
}
// 보폭 고르기: SC[10..13] = E 탄성 (c = 0, ¼, ½, 1), SC[5..8] = 관성 몫; 받아들여지는 가장 큰 c -> SC[14]
__global__ void rt_lspick(double* SC) {
    double E0 = SC[5] + SC[10]; double gd = SC[9];
    const double cs[3] = {0.25, 0.5, 1.0};
    double st = 0.0;
    for (int i=0;i<3;i++) { double E1 = SC[6+i] + SC[11+i]; if (isfinite(E1) && E1 <= E0 + 1e-4*cs[i]*gd) st = cs[i]; }
    SC[14] = st;
}
__global__ void rt_lsup(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, int full) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float st = full ? 1.f : (float)SC[14];
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 걸음 끝: U, v (중점·후퇴 오일러), 고정 노드 v = vbc, 감쇠
__global__ void rt_stepend(const int* IP, int mid, float tau, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = mid ? al*(yy - z0[n*3+a]) : (yy - u)/tau;
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = mid ? 2.f*yy - u : yy;
        vk[n*3+a] = v1*damp;
    }
}
// G2P (제자리): x += Σ w U, v = Σ w v, F ← (I + Σ U ⊗ ∇w) F
__global__ void rt_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* U, const float* vk, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 고정 노드 (ficus 등 cuboid 구속): 상자 안 노드 -> fixd, vbc (프레임 시각 기준, 이동 상자)
__global__ void rt_bc_box(const int* IP, const float* Am, const float* BP, int ib, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    const float* b = BP + ib*10;                                              // 중심 3, 반 크기 3, 속도 3, 켜짐 1 (호스트가 프레임마다 채움)
    if (b[9] == 0.f) return;
    float c0 = b[0], c1 = b[1], c2 = b[2], s0 = b[3], s1 = b[4], s2 = b[5], v0 = b[6], v1 = b[7], v2 = b[8];
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float yq[3] = {(float)(IP[0] + i), (float)(IP[1] + j), (float)(IP[2] + k)};
    float xp[3]; for (int a=0;a<3;a++) xp[a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    bool in = fabsf(xp[0]-c0) < s0 && fabsf(xp[1]-c1) < s1 && fabsf(xp[2]-c2) < s2;
    if (in) { fixd[n] = 1; vbc[n*3] = v0; vbc[n*3+1] = v1; vbc[n*3+2] = v2; }
}
// 선탐색 (융합): 후보 c = ¼, ½, 1 의 탄성 에너지를 한 순회로 (y_c = ŷ + T (η + c δ) 를 모으며 계산) -> SC[11..13]
__global__ void __launch_bounds__(128) rt_elem_ls(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    const float cs[3] = {0.25f, 0.5f, 1.f};
    float X[3][9]; for (int c=0;c<3;c++) for (int k=0;k<9;k++) X[c][k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int c=0;c<3;c++) {
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + cs[c]*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[c][a*3] += ya*g[0]; X[c][a*3+1] += ya*g[1]; X[c][a*3+2] += ya*g[2]; } } }
    float vq = on ? Vq[qq] : 0.f;
    for (int c=0;c<3;c++) {
        float U[9], S[3], V[9]; ssvdf(X[c], U, S, V);
        double J = (double)S[0]*S[1]*S[2];
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
        blocksum_add(e, &SC[11+c]);
    }
}
__global__ void rt_lsy4(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s[4] = {0,0,0,0}, gd = 0.0;
    const float cs[4] = {0.f, 0.25f, 0.5f, 1.f};
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; for (int c=0;c<4;c++) { float e = e0 + cs[c]*d; s[c] += 0.5*(double)Mi*e*e; } gd += -(double)r0[n*3+a]*d; } }
    for (int c=0;c<4;c++) blocksum_add(s[c], &SC[5+c]);
    blocksum_add(gd, &SC[9]);
}
// G2P + 마지막 걸음 노드 갱신 (융합): 노드의 U_1 = 2y − U (중점) 또는 y, v_1 = α (y − z0) 또는 (y − U)/τ, 고정 노드 v = vbc, 감쇠
__global__ void rt_g2p_end(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, int mid, float tau, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = mid ? 2.f*yy - uk : yy;
            float v1 = fx_ ? vbc[nid*3+a] : (mid ? al*(yy - z0[nid*3+a]) : (yy - uk)/tau);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// ================= 멀티그리드 (거친 층 = 조밀 상자, 프레임당 한 번 구성) =================
// LM[l*8 + ..]: 0..2 노드 상자 차원, 3 노드 수, 4..6 칸 상자 차원, 7 가우스 점 수 (= 칸 x 8). 층 0 = 고운 층 (IP).
__global__ void rt_mg_dims(int* IP, int* LM, int nl, const int* capn, const int* capq) {
    int n0 = IP[6], n1 = IP[7], n2 = IP[8];
    LM[0] = n0; LM[1] = n1; LM[2] = n2; LM[3] = IP[9];
    for (int l=1; l<nl; l++) {
        n0 = n0/2 + 3; n1 = n1/2 + 3; n2 = n2/2 + 3;
        int* o = LM + l*8; o[0] = n0; o[1] = n1; o[2] = n2; o[3] = n0*n1*n2; o[4] = n0-2; o[5] = n1-2; o[6] = n2-2; o[7] = (n0-2)*(n1-2)*(n2-2)*8;
        if (o[3] > capn[l] || o[7] > capq[l]) { o[3] = 0; o[7] = 0; atomicOr(&IP[13], 2); }
    }
}
// 거친 층 l 가우스 점 누적: 출처 = 층 l−1 의 가우스 점 (위치 tq = 칸 좌표 + 팔분 위치, X = F, 가중 V)
__global__ void rt_mg_acc(int l, const int* IP, const int* LM, const int* cbF, const float* Fsrc, const float* Vsrc, float* A, float* C, int* badq, int* badc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nsrc = (l == 1) ? IP[12] : LM[(l-1)*8+7];
    if (q >= nsrc) return;
    float w = Vsrc[q]; if (!(w > 0.f)) return;
    int c = q >> 3, oc = q & 7;
    const float GG = 0.28867513459481287f;
    int cx, cy, cz;
    if (l == 1) { cx = cbF[c*3]; cy = cbF[c*3+1]; cz = cbF[c*3+2]; }
    else { const int* pl = LM + (l-1)*8; cz = c % pl[6]; cy = (c / pl[6]) % pl[5]; cx = c / (pl[5]*pl[6]); }
    float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};
    const int* L = LM + l*8;
    int b[3]; float fx[3];
    for (int a=0;a<3;a++) { float t = 0.5f*(tq[a] + 1.5f); float fb = floorf(t - 0.5f); b[a] = (int)fb; fx[a] = t - fb; }
    int cc = (b[0]*L[5] + b[1])*L[6] + b[2];
    int qc = cc*8 + (fx[0] >= 1.f ? 4 : 0) + (fx[1] >= 1.f ? 2 : 0) + (fx[2] >= 1.f ? 1 : 0);
    float X[9]; for (int k=0;k<9;k++) X[k] = Fsrc[(size_t)q*9+k];
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    bool ng = !(S[2] > 0.f);
    float R[9], Lg[9]; float lg[3]; for (int i=0;i<3;i++) lg[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int bb=0;bb<3;bb++) { R[a*3+bb] = U[a*3]*V[bb*3] + U[a*3+1]*V[bb*3+1] + U[a*3+2]*V[bb*3+2];
        Lg[a*3+bb] = V[a*3]*lg[0]*V[bb*3] + V[a*3+1]*lg[1]*V[bb*3+1] + V[a*3+2]*lg[2]*V[bb*3+2]; }
    float* aq = A + (size_t)qc*28; float* ac = C + (size_t)cc*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*Lg[k]); atomicAdd(&aq[18+k], w*X[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*Lg[k]); atomicAdd(&ac[18+k], w*X[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w);
    if (ng) { atomicOr(&badq[qc], 1); atomicOr(&badc[cc], 1); }
}
__global__ void rt_mg_fin(int l, const int* LM, const float* A, const float* C, const int* badq, const int* badc, float* Fq, float* Vq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= LM[l*8+7]) return;
    int c = q >> 3;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (!(ac[27] > 0.f)) { for (int k=0;k<9;k++) out[k] = (k % 4 == 0) ? 1.f : 0.f; Vq[q] = 0.f; }
    else {
        if (aq[27] > 0.f) { rs_fin(aq, aq + 9, aq[27], out, &bad); if (bad || badq[q]) for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; }
        else { rs_fin(ac, ac + 9, ac[27], out, &bad); if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27]; }
        Vq[q] = ac[28]*0.125f;
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
}
__device__ __forceinline__ void qnodes_d(int q, const int* L, int* ids) {
    int c = q >> 3; int cz = c % L[6], cy = (c / L[6]) % L[5], cx = c / (L[5]*L[6]);
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cx+i)*L[1] + (cy+j))*L[2] + (cz+k); }
}
// 거친 층 헤시안 (y = 0 -> X = F_q) 성분 우선 저장 + 노드 대각 융합. BG 기울기에 sc = 2^-l
__global__ void __launch_bounds__(128) rt_mg_elem(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, float mu, float lam, float* H, int capQ, float* B) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;
    float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
    if (on && vq > 0.f) {
        float U[9], S[3], V[9]; ssvdf(F, U, S, V);
        float J = S[0]*S[1]*S[2]; float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
    }
    if (on) for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; float g[3]; gqn(F, bs, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 거친 층 헤시안 곱 (조밀)
__global__ void __launch_bounds__(128) rt_mg_mv(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ, const float* dy, float* out) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;                           // 빈 칸 (부피 0) 만 든 워프는 기여 0
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    float gs[27][3];
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; gqn(F, bs, gs[o]); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*gs[o][0]; dX[a*3+1] += ya*gs[o][1]; dX[a*3+2] += ya*gs[o][2]; } }
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; if (vq > 0.f) for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*gs[o][0] + dP[a*3+1]*gs[o][1] + dP[a*3+2]*gs[o][2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); }
}
// 제한 out[J] = Σ c_k c_l c_m in[2J − 3 + (k,l,m)] (c = 1,3,3,1 / 4), 고운 층 차원 nf (LM 층 lf), 거친 층 차원 (LM 층 lf+1). ch = 1 또는 3
__global__ void rt_mg_restrict(int lf, const int* LM, const float* in, float* out, int ch) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int J = blockIdx.x*blockDim.x + threadIdx.x;
    if (J >= B[3]) return;
    int jz = J % B[2], jy = (J / B[2]) % B[1], jx = J / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    float acc[3] = {0.f, 0.f, 0.f};
    for (int k=0;k<4;k++) { int ix = 2*jx - 3 + k; if (ix < 0 || ix >= A[0]) continue;
      for (int m=0;m<4;m++) { int iy = 2*jy - 3 + m; if (iy < 0 || iy >= A[1]) continue;
        for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; if (iz < 0 || iz >= A[2]) continue;
          float w = cw[k]*cw[m]*cw[n]; int i = (ix*A[1] + iy)*A[2] + iz;
          for (int c=0;c<ch;c++) acc[c] += w*in[i*ch+c]; } } }
    for (int c=0;c<ch;c++) out[J*ch+c] = acc[c];
}
// 연장 out[i] (+)= Σ_J c_{i−2J+3} in[J]
__global__ void rt_mg_prolong(int lf, const int* LM, const float* in, float* out, int add) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int I = blockIdx.x*blockDim.x + threadIdx.x;
    if (I >= A[3]) return;
    int iz = I % A[2], iy = (I / A[2]) % A[1], ix = I / (A[1]*A[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    int ii[3] = {ix, iy, iz};
    int Jc[3][2]; float Wc[3][2];
    for (int a=0;a<3;a++) { int j0 = (ii[a] + 3) >> 1; int k0 = ii[a] + 3 - 2*j0; Jc[a][0] = j0; Wc[a][0] = cw[k0]; Jc[a][1] = j0 - 1; Wc[a][1] = cw[k0 + 2]; }
    float acc[3] = {0.f, 0.f, 0.f};
    for (int p=0;p<2;p++) { int jx = Jc[0][p]; if (jx < 0 || jx >= B[0]) continue;
      for (int q=0;q<2;q++) { int jy = Jc[1][q]; if (jy < 0 || jy >= B[1]) continue;
        for (int r=0;r<2;r++) { int jz = Jc[2][r]; if (jz < 0 || jz >= B[2]) continue;
          float w = Wc[0][p]*Wc[1][q]*Wc[2][r]; int J = (jx*B[1] + jy)*B[2] + jz;
          for (int c=0;c<3;c++) acc[c] += w*in[J*3+c]; } } }
    for (int c=0;c<3;c++) out[I*3+c] = (add ? out[I*3+c] : 0.f) + acc[c];
}
// 층 l 배열 0 채우기 (실제 크기만): which 0 = 노드 수, 1 = 가우스 점 수, 2 = 칸 수
__global__ void rt_zero_lm(int l, const int* LM, int which, int mul, float* a, int* ai) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int n = (which == 0) ? LM[l*8+3] : ((which == 1) ? LM[l*8+7] : LM[l*8+7]/8);
    if (i >= n*mul) return;
    if (a) a[i] = 0.f; else ai[i] = 0;
}
// 거친 층 노드: D = (Mi I + B)⁻¹ (질량·대각 0 이면 0)
__global__ void rt_mg_D(int l, const int* LM, const float* mc, float itau2, const float* B, float* D) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float M[9]; float Mi = mc[n]*itau2;
    for (int k=0;k<9;k++) M[k] = B[n*9+k] + ((k % 4 == 0) ? Mi : 0.f);
    float Di[9]; inv3(M, Di); for (int k=0;k<9;k++) D[n*9+k] = Di[k];
}
// 층 노드 연산: (0) z = w D r; (1) z += w D (r − Az); (2) res = r − Az;  Az = Mi z + KTz 마무리 (층 0 은 T 끼움)
// 층 0: Az = Mi z + T (K (T z)); 여기서 KTz 는 이미 계산된 K(Tz) 또는 K z. 모드 3: Tz = T z (층 0), 모드 4: out = T v
__global__ void rt_mg_node(int l, const int* LM, int mode, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2,
                           const float* D, const double* W, const float* r, float* KTz, float* z, float* res) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float w = (float)W[l];
    const float* d = D + n*9;
    float rr[3], az[3];
    for (int a=0;a<3;a++) rr[a] = r ? r[n*3+a] : 0.f;
    if (mode == 0 || mode == 6) { float zz[3]; for (int a=0;a<3;a++) { zz[a] = w*(d[a*3]*rr[0] + d[a*3+1]*rr[1] + d[a*3+2]*rr[2]); z[n*3+a] = zz[a]; }
        if (mode == 6) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; }       // 층 0: 다음 Az 의 T z 를 함께 (융합)
        return; }
    if (mode == 3 || mode == 4) { const float* v = (mode == 3) ? z : r; float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }
    if (mode == 5 || mode == 9) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, r + n*3, t); for (int a=0;a<3;a++) z[n*3+a] += t[a];
        if (mode == 9) { float u[3] = {z[n*3], z[n*3+1], z[n*3+2]}; Tapply(sk[n], pm[n], n0, n1, n2, u, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; }   // 층 0: 다음 Az 의 T z (융합)
        return; }
    float Mi = mn[n]*itau2;
    float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
    KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;                         // 읽은 뒤 비움 (다음 곱의 원자 합 바탕; 0 채우기 커널 없앰)
    if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
    for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
    float e[3]; for (int a=0;a<3;a++) e[a] = rr[a] - az[a];
    if (mode == 2) { for (int a=0;a<3;a++) res[n*3+a] = e[a]; return; }
    if (mode == 7) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }   // 층 0: T (r − Az) (제한 입력, 융합)
    for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]);
}
// 거듭제곱법 한 걸음: z ← D A z / |D A z|, λ = (z·DAz)/(z·z) -> SCM[l]; 마지막에 W[l] = ω/λ
__global__ void rt_mg_pw1(int l, const int* LM, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* D, const float* z, float* KTz,
                          float* y, double* SCM) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int nn = LM[l*8+3];
    if (blockIdx.x*blockDim.x >= nn) return;
    if (n < nn) {
        float Mi = mn[n]*itau2;
        float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
        KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;
        if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
        float az[3]; for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
        const float* d = D + n*9;
        for (int a=0;a<3;a++) { float v = d[a*3]*az[0] + d[a*3+1]*az[1] + d[a*3+2]*az[2]; y[n*3+a] = v; s1 += (double)z[n*3+a]*v; s2 += (double)z[n*3+a]*z[n*3+a]; s3 += (double)v*v; }
    }
    blocksum_add(s1, &SCM[l*4]); blocksum_add(s2, &SCM[l*4+1]); blocksum_add(s3, &SCM[l*4+2]);
}
__global__ void rt_mg_pw2(int l, const int* LM, const float* y, float* z, double* SCM, double* W, float omega, int last) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && last) { double lm = SCM[l*4] / fmax(SCM[l*4+1], 1e-300); W[l] = omega / fmax(lm, 1e-30); }
    if (n >= LM[l*8+3]) return;
    double nr = sqrt(fmax(SCM[l*4+2], 1e-300));
    for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a] / nr);
}
__global__ void rt_mg_pw0(int l, double* SCM) { SCM[l*4] = 0.0; SCM[l*4+1] = 0.0; SCM[l*4+2] = 0.0; }
__global__ void rt_fill_rand(int l, const int* LM, float* z) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    unsigned int h = (unsigned int)n*2654435761u + 12345u*(l+1);
    for (int a=0;a<3;a++) { h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; z[n*3+a] = ((float)(h & 0xffffff) / 16777216.f) - 0.5f; }
}
// PCG + MG: α 갱신만 (z 는 V 순환이 따로); 그다음 내적 r·z, r·(z − zo) 와 zo ← z
__global__ void rt_cg1m(const int* IP, const double* SC, const float* p, const float* Ap, float* dl, float* r) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
    for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; r[n*3+a] -= al*Ap[n*3+a]; }
}
__global__ void rt_dotz(const int* IP, const float* r, const float* z, float* zo, double* SC, int first) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0, s5 = 0.0;
    if (n < IP[9]) for (int a=0;a<3;a++) { float zn = z[n*3+a]; s3 += (double)r[n*3+a]*zn; s4 += (double)r[n*3+a]*(zn - zo[n*3+a]); s5 += (double)r[n*3+a]*r[n*3+a]; zo[n*3+a] = zn; }
    blocksum_add(s3, first ? &SC[0] : &SC[3]); if (!first) blocksum_add(s4, &SC[4]);
#ifdef AF_ADAPT
    if (!first) blocksum_add(s5, &SC[57]);                                         // PCG 잔차 |r|² (절 AN.14 판정)
#endif
}
__global__ void rt_pinit(const int* IP, const float* z, const float* sk, const int* pm, float n0, float n1, float n2, float* p, float* Tp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, z + n*3, t);
    for (int a=0;a<3;a++) { p[n*3+a] = z[n*3+a]; Tp[n*3+a] = t[a]; }
}
// 하강 방향 보장 (--descent): PCG 결과 δ 가 g·δ ≥ 0 이면 δ ← α0 p0 (p0 = 첫 방향 = M r0, α0 = r0·M r0 / p0·A p0). SC[16] = r0·Mr0, SC[17] = p0·Ap0, SC[18] 바꾼 횟수
__global__ void rt_fixdir(const int* IP, double* SC, const float* p0, float* dl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool bad = !(SC[9] < 0.0);
    if (n == 0 && bad) SC[62] += 1.0;
    if (n >= IP[9] || !bad) return;
    float a0 = (SC[61] > 0.0) ? (float)(SC[60]/SC[61]) : 0.f;
    for (int a=0;a<3;a++) dl[n*3+a] = a0*p0[n*3+a];
}
__global__ void rt_sc_copy(double* SC, int dst, int src) { SC[dst] = SC[src]; }
// 선탐색 넓힘 (--ls_n NC): 후보 c_i = 2^-i (i = 0..NC−1). 관성 몫 SC[20] (c = 0), SC[21+i]; 탄성 SC[40+i]; g·δ SC[9]; 보폭 SC[14]
__global__ void rt_lsyN(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, int NC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, sc[12];
    for (int i=0;i<12;i++) sc[i] = 0.0;
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*(double)Mi*e0*e0; gd += -(double)r0[n*3+a]*d;
            float c = 1.f; for (int i=0;i<NC;i++) { float e = e0 + c*d; sc[i] += 0.5*(double)Mi*e*e; c *= 0.5f; } } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]);
    for (int i=0;i<NC;i++) blocksum_add(sc[i], &SC[21+i]);
}
__global__ void __launch_bounds__(128) rt_elem_lsN(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC, int NC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    float c = 1.f;
    for (int i=0;i<NC;i++) {
        float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + c*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
        float U[9], S[3], V[9]; ssvdf(X, U, S, V);
        double J = (double)S[0]*S[1]*S[2];
        double en = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
        blocksum_add(en, &SC[40+i]);
        c *= 0.5f;
    }
}
__global__ void rt_lspickN(double* SC, int NC) {
    double E0 = SC[20] + SC[10]; double gd = SC[9];
    double st = 0.0, c = 1.0;
    for (int i=0;i<NC;i++) { double E1 = SC[21+i] + SC[40+i]; if (st == 0.0 && isfinite(E1) && E1 <= E0 + 1e-4*c*gd) st = c; c *= 0.5; }
    SC[14] = st;
}
// ---- 단계 일반화 (θ 법·TR-BDF2) ----
// 단계 시작: Ut = Ub + τ vb + τ² g, z0 = Ub + β τ vb, y = Ut (고정 노드 y = Uf + dtf vbc). θ 단계: Ub = U, vb = v, β = 1 − θ; BDF2 단계: Ub = w, vb = ŵ, β = 0.
__global__ void rt_step0g(const int* IP, const float* Ub, const float* vb, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                          const float* Uf, float dtf, float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = Ub[n*3+a], vv = vb[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? Uf[n*3+a] + dtf*vbc[n*3+a] : ut;
    }
}
// 단계 끝: U ← U + thinv (y − U) (θ 단계 thinv = 1/θ, BDF2 1), v ← α (y − z0) (고정 노드 vbc), 감쇠
__global__ void rt_stepend_g(const int* IP, float thinv, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = al*(yy - z0[n*3+a]);
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = u + thinv*(yy - u);
        vk[n*3+a] = v1*damp;
    }
}
// BDF2 기저: w = c1 Uγ − c0 U0, ŵ = c1 vγ − c0 v0
__global__ void rt_bdf_base(const int* IP, float c1, float c0, const float* Ug, const float* vg, const float* U0, const float* v0, float* w, float* wv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) { w[n*3+a] = c1*Ug[n*3+a] - c0*U0[n*3+a]; wv[n*3+a] = c1*vg[n*3+a] - c0*v0[n*3+a]; }
}
// 관성 질량 하한: m' = max(m, cs tr(B)/3) (cs = c τ², B = 걸음 첫 뉴턴의 노드 탄성 대각 블록), cs = 0 이면 m 그대로
__global__ void rt_meff(const int* IP, const float* mn, const float* B, float cs, float* meff) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    meff[n] = (cs > 0.f) ? fmaxf(m, cs*(B[n*9] + B[n*9+4] + B[n*9+8])*(1.f/3.f)) : m;
}
// G2P + 마지막 단계 끝 (일반): uu = U + thinv (y − U), v1 = α (y − z0) 또는 vbc
__global__ void rt_g2p_endg(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, float thinv, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = uk + thinv*(yy - uk);
            float v1 = fx_ ? vbc[nid*3+a] : al*(yy - z0[nid*3+a]);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 접촉 적응: 고착 노드 수 -> IP[14] (rt_reset 이 비움)
__global__ void rt_count_stuck(const int* IP, const float* an, int* IPw) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (an[n] > 0.f) atomicAdd(&IPw[14], 1);
}
// 가장 거친 층 평활 (융합, 한 블록): z = ω D r 뒤 nc−1 번 z += ω D (r − (Mi z + K z)). z·곱 누적 KT 는 전역 (한 블록이라 블록 동기화로 차례; 크기 제한 없음), KT 는 끝에 비움.
__global__ void __launch_bounds__(1024) rt_mg_csolve(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ,
                                                     const float* mc, float itau2, const float* D, const double* W, const float* r, float* z, float* ks, int nc) {
    const int* L = LM + l*8;
    int nn = L[3], nq = L[7];
    float w = (float)W[l];
    for (int n = threadIdx.x; n < nn; n += blockDim.x) {
        const float* d = D + n*9; float r0 = r[n*3], r1 = r[n*3+1], r2 = r[n*3+2];
        for (int a=0;a<3;a++) { z[n*3+a] = w*(d[a*3]*r0 + d[a*3+1]*r1 + d[a*3+2]*r2); ks[n*3+a] = 0.f; }
    }
    __syncthreads();
    for (int it = 1; it < nc; it++) {
        for (int q = threadIdx.x; q < nq; q += blockDim.x) {
            float vq = Vq[q]; if (!(vq > 0.f)) continue;
            int oc = q & 7; int ids[27]; qnodes_d(q, L, ids);
            float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
            float gs[27][3]; float dX[9] = {0,0,0,0,0,0,0,0,0};
            for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; gqn(F, bs, gs[o]); int id = ids[o];
                for (int a=0;a<3;a++) { float ya = z[id*3+a]; dX[a*3] += ya*gs[o][0]; dX[a*3+1] += ya*gs[o][1]; dX[a*3+2] += ya*gs[o][2]; } }
            float dP[9]; for (int rr=0;rr<9;rr++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(rr*9+c)*capQ + q]*dX[c]; dP[rr] = vq*t; }
            for (int o=0;o<27;o++) for (int a=0;a<3;a++) atomicAdd(&ks[ids[o]*3+a], dP[a*3]*gs[o][0] + dP[a*3+1]*gs[o][1] + dP[a*3+2]*gs[o][2]);
        }
        __syncthreads();
        for (int n = threadIdx.x; n < nn; n += blockDim.x) {
            float Mi = mc[n]*itau2; const float* d = D + n*9;
            float e[3]; for (int a=0;a<3;a++) { e[a] = r[n*3+a] - (Mi*z[n*3+a] + ks[n*3+a]); ks[n*3+a] = 0.f; }
            for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]);
        }
        __syncthreads();
    }
}
// ---- generalized-α / HHT (implicit-err 절 AN.err3 증분 포텐셜 꼴) ----
// 노드 가속도 a = ma / m (질량 0 이면 0, 고정 노드 0)
__global__ void rt_ga_anode(const int* IP, const float* mn, const float* ma, const unsigned char* fixd, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) ak[n*3+a] = (m > 0.f && !(fixd && fixd[n])) ? ma[n*3+a]/m : 0.f;
}
// 걸음 시작: xp = U + dt v + dt²(½−β) a, ys = αf U + (1−αf) xp, Ũ = ys + τ² (g − αm a), z0 = ys − (v + dt (1−γ) a)/α, y = U (시작점 = 걸음 시작 위치; 고정 노드 U + (1−αf) dt vbc)
__global__ void rt_ga_step0(const int* IP, const float* U, const float* vk, const float* ak, float dt, float am, float af, float gm, float be, float tau, float al,
                            float g0, float g1, float g2, const unsigned char* fixd, const float* vbc, float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], ac = ak[n*3+a];
        float xp = u + dt*v + dt*dt*(0.5f - be)*ac;
        float ys = af*u + (1.f - af)*xp;
        Ut[n*3+a] = ys + tau*tau*(gg[a] - am*ac);
        z0[n*3+a] = ys - (v + dt*(1.f - gm)*ac)/al;
        y[n*3+a] = (fixd && fixd[n]) ? u + (1.f - af)*dt*vbc[n*3+a] : Ut[n*3+a];    // 고착 판정은 관성 예측 Ũ 에서 (implicit-err 와 같게); 시작점은 rt_ga_y0 가 U 로
    }
}
// 뉴턴 시작점 = 걸음 시작 위치 U (고착 판정 뒤, implicit-err floor_state --y0 uk 와 같은 식): dd = U − ŷ, η = (I − P) dd + (1/√k) P dd (√k ≤ 1e-6 이면 그 성분 0), y = ŷ + T η
__global__ void rt_ga_y0(const int* IP, const float* U, const float* yh, const float* sk, const int* pm, float n0, float n1, float n2, float* eta, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int p = pm[n];
    if (p == 2) return;                                                                // 고정 노드: rt_floor_n 이 정한 그대로
    float s = sk[n];
    float isk = (s > 1e-6f) ? 1.f/s : 0.f;
    float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - yh[n*3+a];
    float e[3];
    if (p == 0) { for (int a=0;a<3;a++) e[a] = isk*dd[a]; }
    else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = dn*nn[a] + isk*(dd[a] - dn*nn[a]); }
    float t[3]; Tapply(s, p, n0, n1, n2, e, t);
    for (int a=0;a<3;a++) { eta[n*3+a] = e[a]; y[n*3+a] = yh[n*3+a] + t[a]; }
}
// 걸음 끝: v1 = α (y − z0) (고정 노드 vbc), a1 = (v1 − v − dt (1−γ) a)/(γ dt), U1 = U + dt v + dt² ((½−β) a + β a1) (Newmark 위치 식), 고착 노드 (an > 0) 의 a1 = 0 (areset), 고정 노드 a1 = 0, 감쇠
__global__ void rt_ga_end(const int* IP, float dt, float gm, float be, float al, const float* y, const float* z0, float* U, float* vk, float* ak,
                          const float* an, int areset, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool fx_ = fixd && fixd[n];
    bool stk = areset && an && an[n] > 0.f;
#ifdef AF_BAR
    bool stkb = stk; stk = false;                                               // 장벽 판: 고착은 접선뿐이라 가속도 0 도 접선 성분만 (법선 반동은 장벽 몫)
#endif
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], ac = ak[n*3+a];
        float v1 = fx_ ? vbc[n*3+a] : al*(y[n*3+a] - z0[n*3+a]);
        float a1 = (v1 - v - dt*(1.f - gm)*ac)/(gm*dt);
        U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*ac + be*a1);
        vk[n*3+a] = v1*damp;
        ak[n*3+a] = (fx_ || stk) ? 0.f : a1;
    }
#ifdef AF_BAR
    if (stkb && !fx_) { float an_ = ak[n*3]*AF_BNRM[0] + ak[n*3+1]*AF_BNRM[1] + ak[n*3+2]*AF_BNRM[2]; for (int a=0;a<3;a++) ak[n*3+a] = an_*AF_BNRM[a]; }
#endif
}
// 첫 프레임 가속도 a0 = g − f(0)/m', m' = max(m, mlow) (고정 노드 0)
__global__ void rt_ga_a0(const int* IP, const float* mn, const double* f, float mlow, float g0, float g1, float g2, const unsigned char* fixd, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float me = fmaxf(mn[n], mlow); float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) ak[n*3+a] = (fixd && fixd[n]) ? 0.f : (gg[a] - (float)f[n*3+a]/me);
}
// G2P (generalized-α): x += Σ w U, v = Σ w v, a_p = Σ w a, F ← (I + Σ U ⊗ ∇w) F
__global__ void rt_g2p_ga(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* U, const float* vk, const float* ak, float* x, float* v, float* pa, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, aa[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; aa[a] += ww*ak[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; pa[p*3+a] = aa[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 허용치 조기 종료 제어 (절 AN.14, 단일 스레드). skipify() 가 AF_SKIP 선언과 다른 모든 커널 머리의 "AF_SKIP 이면 바로 돌아감" 을 넣는다 (그 판에서만 컴파일).
// CT (int): 0 ND (이 걸음 뉴턴 끝), 1 PD (이 풀이 PCG 끝), 2 뉴턴 반복 수, 3 PCG 반복 수, 4 뉴턴 상한까지 간 걸음 수, 5 PCG 상한까지 간 풀이 수, 6 걸음 수, 7 풀이 수 (프레임 합)
// CD (double): 0 걸음 첫 |g|², 1 이 풀이의 PCG 목표 |r|², 2 기준 척도 F² = |T f(y0)|² (걸음 시작점의 탄성·접촉 힘 몫, rt_grad 의 SC[56]). eN2 = εN², eP2 = εP² (음수면 그 판정 없음)
// 규칙 (두 적분기 공통): 뉴턴은 |g_k| <= εN F 이면 끝 (k >= 1), PCG 는 |r_i| <= εP min(|g_k|, F) 이면 끝 (|r|² = rt_dotz·rt_cg1 의 SC[57]).
// 후퇴 오일러는 시작점 = 관성 예측 Ũ 라 |g0| = F (바닥 변수 바꿈 몫 빼고); HHT 는 시작점 = 걸음 시작 위치 U 라 |g0| 가 관성 불일치로 부풀어 있어 F 를 따로 둔다.
#ifdef AF_ADAPT
__global__ void rt_ctl(int op, int it, double* SC, int* CT, double* CD, double eN2, double eP2) {
    if (op == 0) { for (int i=0;i<16;i++) CT[i] = 0; AF_SKIP = 0; return; }                         // 프레임 시작
    if (op == 1) { CT[0] = 0; CT[1] = 0; CT[6]++; AF_SKIP = 0; return; }                              // 걸음 시작
    if (op == 2) { CT[1] = 0; AF_SKIP = CT[0]; return; }                                              // 뉴턴 반복 시작
    if (op == 3) {                                                                                     // rt_grad 뒤: |g|² = SC[1], |T f|² = SC[56]
        if (CT[0]) return;
        double g2 = SC[1];
        if (it == 0) { CD[0] = g2; CD[2] = SC[56]; if (!(g2 > 0.0)) { CT[0] = 1; AF_SKIP = 1; return; } }
        else if (eN2 >= 0.0 && g2 <= eN2*CD[2]) { CT[0] = 1; AF_SKIP = 1; return; }
        CT[2]++; return;
    }
    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); CD[1] = (eP2 >= 0.0) ? eP2*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }   // PCG 시작: 목표 |r|²
    if (op == 5) { AF_SKIP = CT[0] | CT[1]; return; }                                                 // PCG 반복 시작
    if (op == 6) { double r2 = SC[57]; SC[57] = 0.0; if (CT[0] | CT[1]) return; CT[3]++; if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }   // PCG 반복 끝
    if (op == 7) { if (!CT[0] && !CT[1]) CT[5]++; AF_SKIP = CT[0]; return; }                          // PCG 끝 -> 선탐색·갱신
    if (op == 8) { if (!CT[0]) CT[4]++; AF_SKIP = 0; return; }                                        // 걸음 끝 (뉴턴 고리 밖)
}
#endif
}
'''

SKIP_EXEMPT = ("rt_reset", "rt_ctl")


def skipify(src):
    """허용치 조기 종료 판 (절 AN.14): AF_SKIP 선언·rt_ctl 을 켜고, 제어 커널 (rt_reset·rt_ctl) 을 뺀 모든 커널 몸 첫머리에 AF_SKIP 이면 바로 돌아가는 줄 (블록 전체가 같은 값을 읽으므로 __syncthreads 와 충돌 없음)."""
    import re
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ int AF_SKIP;\n#define AF_ADAPT 1', 1)
    out, pos = [], 0
    for m in re.finditer(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?(\w+)\s*\(", src):
        b = src.index("{", m.end())
        out.append(src[pos:b + 1])
        if m.group(1) not in SKIP_EXEMPT:
            out.append(" if (AF_SKIP) return;")
        pos = b + 1
    out.append(src[pos:])
    return "".join(out)


def octify(src):
    """점유 가중 가우스 점 판 (절 AN.err4 의 --gq_oct κ 정의를 실시간 판에): AF_OCT 를 켜고 팔분 질량 버퍼 AF_MO·κ AF_GOCT 장치 전역을 선언."""
    return src.replace('extern "C" {', 'extern "C" {\n#define AF_OCT 1\n__device__ float* AF_MO;\n__device__ float AF_GOCT;', 1)


BAR_SRC = r'''
// ---- 장벽 판 (--fl_bar, 절 AN.19): 바닥 법선 = 표본점 IPC 거리 장벽, 고착 = 접선만 ----
// 표본점 q = 팔분 칸 입자 질량 중심 (프레임 시작, P2G), 걸음 안 이동 = Σ_o BW (U + thinv (y − U)) (가우스 점 기저); d = (x − p)·n.
// b(d) = −κ (d − d̂)² ln(d/d̂) (0 < d < d̂), κ = ks V_q (ks = (λ + 2μ)/d̂², d̂ = ½ dx_정답). 걸음 시작 d ≤ 0 인 점은 그 걸음 장벽에서 뺌 (AF_BACT 0).
// mode 0: 걸음 시작 (y = 시작점) 판정; mode 1: 기울기 스칼라 c = b' thinv, 덩어리 헤시안 스칼라 h = b'' thinv², 에너지 -> E.
__global__ void __launch_bounds__(128) rt_bar_q(const int* IP, const int* cb, const float* BW, const float* Vq, const float* y, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, int mode, float* Gn, float* Kn, double* E) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    double e = 0.0; float c = 0.f, h = 0.f;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) {
        int ids[27]; qnodes(qq, IP, cb, ids);
        float xq[3] = {AF_BX[qq*4]/bm, AF_BX[qq*4+1]/bm, AF_BX[qq*4+2]/bm};
        for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
            for (int a=0;a<3;a++) xq[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
        if (mode == 0) AF_BACT[qq] = (d > 0.f) ? 1 : 0;
        else if (AF_BACT[qq] && d < dh) {
            double kap = (double)ks*(double)Vq[qq];
            if (d > 0.f) {
                double dd = (double)d - dh, L = log((double)d/dh);
                e = -kap*dd*dd*L;
                c = (float)(-kap*(2.0*dd*L + dd*dd/d))*thinv;
                h = (float)(-kap*(2.0*L + 4.0*dd/d - dd*dd/((double)d*d)))*thinv*thinv;
            } else e = 1e30;
        }
        if (mode == 1 && !AF_BDET && (c != 0.f || h != 0.f)) for (int o=0;o<27;o++) { float w = BW[oc*27+o]; atomicAdd(&Gn[ids[o]], w*c); atomicAdd(&Kn[ids[o]], w*h); }
    }
    if (mode == 1) {
        if (on && AF_BDET) { AF_BPG[qq] = c; AF_BPH[qq] = h; }
        blocksum_add(e, E);
    }
}
// 노드: f += G n, B += K n nᵀ (덩어리 헤시안: 행 합이라 참 헤시안보다 크다 -> 하강 방향 유지)
__global__ void rt_bar_n(const int* IP, const float* Gn, const float* Kn, float n0, float n1, float n2, double* f, float* B) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float G = Gn[n], K = Kn[n]; if (G == 0.f && K == 0.f) return;
    float nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) { f[n*3+a] += (double)(G*nn[a]); for (int b=0;b<3;b++) B[n*9+a*3+b] += K*nn[a]*nn[b]; }
}
// 헤시안 곱 몫: KTp += K n (n·Tp)
__global__ void rt_bar_mv(const int* IP, const float* Kn, float n0, float n1, float n2, const float* Tp, float* KTp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float K = Kn[n]; if (K == 0.f) return;
    float t = K*(Tp[n*3]*n0 + Tp[n*3+1]*n1 + Tp[n*3+2]*n2);
    KTp[n*3] += t*n0; KTp[n*3+1] += t*n1; KTp[n*3+2] += t*n2;
}
// 선탐색 후보 NC 개의 장벽 에너지 -> SC[40+i] (rt_elem_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i)
__global__ void __launch_bounds__(128) rt_bar_lsN(const int* IP, const int* cb, const float* BW, const float* Vq, const float* yh, const float* eta, const float* dl,
                         const float* sk, const int* pm, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, double* SC, int NC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    int ids[27]; qnodes(qq, IP, cb, ids);
    float x0[3] = {0.f, 0.f, 0.f};
    if (act) for (int a=0;a<3;a++) x0[a] = AF_BX[qq*4+a]/bm;
    double kap = act ? (double)ks*(double)Vq[qq] : 0.0;
    float c = 1.f;
    for (int i=0;i<NC;i++) {
        double e = 0.0;
        if (act) {
            float xq[3] = {x0[0], x0[1], x0[2]};
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
                float ee[3]; for (int a=0;a<3;a++) ee[a] = eta[id*3+a] + c*dl[id*3+a];
                float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, ee, t);
                for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; xq[a] += w*(U[id*3+a] + thinv*(ya - U[id*3+a])); } }
            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; }
        }
        blocksum_add(e, &SC[40+i]);
        c *= 0.5f;
    }
}
'''


def flptify(src, kex):
    """점별 고착 판 (--fl_pt, 절 AN.17): 노드 몫 평균 뒤 거듭제곱 (1 − r̄)^κ 대신 점마다 1 − (1 − w_st)^κ 를 질량 가중 평균 (κ = Δt/dt_정답, 단계 frac 1 판만)."""
    return src.replace('extern "C" {', 'extern "C" {\n#define AF_FLJ 1\n#define AF_KEXQ %.9ef' % float(kex), 1)


def barify(src):
    """장벽 판 (--fl_bar, 절 AN.19): AF_BAR 를 켜고 표본점 버퍼 전역을 선언, 장벽 커널을 모듈 끝 (extern 닫기 앞) 에."""
    src = src.replace('extern "C" {', 'extern "C" {\n#define AF_BAR 1\n__device__ float* AF_BX;\n__device__ float* AF_BPG;\n__device__ float* AF_BPH;\n__device__ int* AF_BACT;\n__device__ int AF_BDET;\n__device__ const float* AF_XP;\n__device__ float AF_BNRM[3];', 1)
    i = src.rstrip().rfind("}")
    return src[:i] + BAR_SRC + "\n}\n"


def build_src(src, adapt, goct, floor=None):
    if goct is not None and goct >= 0:
        src = octify(src)
    if floor and floor.get("pt"):
        src = flptify(src, floor["KEX"])
    if floor and floor.get("bar"):
        src = barify(src)
    return skipify(src) if adapt else src


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


class RTSolver(MG2.MG2Mixin):
    """고정 용량 버퍼 + 프레임 CUDA 그래프. step(): 그래프 재생 한 번 = 한 프레임 (P2G -> K 걸음 x 뉴턴 n x PCG c -> G2P), 입자 상태 x·v·F 제자리 갱신."""

    def __init__(s, x, v, F, m, vol, A, Ai, mu, lam, g, frame_dt, K, integ, newton, cg, ls, floor=None, damp=1.0, bc=None, cap_mul=2.0, dev="cuda", mg_levels=1, mg_nc=8, mg_pow=6, mg_omega=1.0, descent=False, ls_n=0, scheme=None, theta=0.5, mfloor=0.0, K_contact=None, gfin=False, fuse2=True, mg_wreuse=False, mg_lreuse=False, mg_csolve=False, ga=None, rho=0.7, areset=True, tolN=-1.0, tolP=-1.0, goct=-1.0, pk=False, pk_loc=2, pkb=256, pkminb=1, mv2=False, mg_occ=False, mg_fuse=0):
        s.dev = dev
        # 절 AN.20 (lib/anchorflow/fem_an_mg2.py): 칸 협동 곱 (mv2), 거친 층 점유 칸 목록 (mg_occ, mv2 포함), 층 mg_fuse 이상 V 순환 한 블록 (mg_fuse >= 1)
        s.mg_occ, s.mg_fuse = bool(mg_occ), int(mg_fuse or 0)
        s.mv2 = bool(mv2) or s.mg_occ
        s.mg2 = s.mv2 or s.mg_fuse > 0
        assert not (s.mg2 and pk), "절 AN.20 판은 그래프 판만"
        assert s.mg_fuse == 0 or 1 <= s.mg_fuse <= int(mg_levels) - 1, "--mg_fuse 는 1 .. 층 수 − 1"
        # 지속형 커널 판 (절 AN.18, lib/anchorflow/fem_an_pk.py): 프레임 하나 = 협동 실행 커널 한 번 (명령 목록을 격자 장벽으로 차례로). pk_loc = 블록 0 하나가 도는 MG 층 (이 층 이상)
        s.pk, s.pk_loc_l, s.pkb, s.pkminb = bool(pk), int(pk_loc), int(pkb), int(pkminb)
        s._pkc, s._pk_loc, s.pkmod = None, 0, None
        s._src0 = MG2.mg2_src(SRC) if s.mg2 else SRC
        # 허용치 조기 종료 (절 AN.14): 뉴턴 |g|/|g0| <= tolN, PCG sqrt(r·z / r0·z0) <= tolP (음수 = 그 판정 없음). 뉴턴·PCG 수는 상한 (그래프에 펼친 수)
        s.tolN, s.tolP = float(tolN), float(tolP)
        s.adapt = s.tolN >= 0 or s.tolP >= 0
        s.goct = float(goct)                                                           # >= 0 이면 점유 가중 가우스 점 (정의 변경, 절 AN.err4)
        f32 = torch.float32
        s.N = int(x.shape[0])
        # 입자를 첫 프레임 칸 순서로 정렬 (P2G·G2P 모으기·원자 합 지역성; 결과 정의는 같음). 원래 순서 = s.perm 의 역
        yy = x.double() @ Ai.double().T
        bb = torch.floor(yy - 0.5).long(); bb = bb - bb.min(0).values; mx = bb.max(0).values + 1
        s.perm = torch.argsort((bb[:, 0] * mx[1] + bb[:, 1]) * mx[2] + bb[:, 2])
        pr = s.perm
        s.x = x[pr].float().contiguous().clone(); s.v = v[pr].float().contiguous().clone(); s.F = F.reshape(-1, 9)[pr].float().contiguous().clone()
        s.m = m[pr].float().contiguous(); s.vol = vol[pr].float().contiguous()
        s.A = A.float().contiguous(); s.Ai = Ai.float().contiguous()
        s.mu, s.lam, s.g = float(mu), float(lam), [float(q) for q in g]
        s.h = float(frame_dt); s.K = int(K); s.mid = integ == "mid"
        s.dt = s.h / s.K; s.tau = s.dt / 2 if s.mid else s.dt
        s.al = (2.0 / s.tau) if s.mid else (1.0 / s.tau)
        s.beta = 0.5 if s.mid else 0.0
        s.nt, s.cg, s.ls = int(newton), int(cg), int(ls)
        s.floor = floor                                                                # dict(PT, NR, PN, DXG, KEX_per_step, vout) 또는 None
        s.damp = float(damp)
        s.nl, s.mg_nc, s.mg_pow, s.mg_omega = int(mg_levels), int(mg_nc), int(mg_pow), float(mg_omega)
        s.descent = bool(descent)
        s.ls_n = int(ls_n)                                                             # 0 = 옛 후보 1·½·¼, 그 밖 = 후보 2^0 .. 2^-(ls_n−1)
        # 단계 방식: "theta" (θ; ½ = 중점, 1 = 후퇴 오일러) 또는 "trbdf2" (γ = 2 − √2). 기본은 integ 에서 (mid -> θ ½, be -> θ 1) -- 옛 판과 같은 식
        s.scheme = scheme or "theta"
        s.theta = float(theta) if scheme else (0.5 if s.mid else 1.0)
        s.mfloor = float(mfloor)
        s.gfin = bool(gfin)
        s.fuse2 = bool(fuse2)                                                          # 결과 불변 융합 2 판 (가장 거친 층 한 블록, 층 0 연장 보정 + T z, KTp 0 채우기를 cg2 에)
        s.mg_wreuse = bool(mg_wreuse)
        s.mg_lreuse = bool(mg_lreuse)
        s.mg_csolve = bool(mg_csolve)
        # generalized-α / HHT (ga = "ga" 또는 "hht"): 증분 포텐셜 꼴, 노드 가속도 상태 (입자 가속도로 프레임 사이 전달)
        s.ga = ga
        if ga:
            r = float(rho)
            am, af = ((0.0, (1.0 - r) / (1.0 + r)) if ga == "hht" else ((2 * r - 1) / (r + 1), r / (r + 1)))
            s.ga_c = dict(am=am, af=af, gm=0.5 - am + af, be=0.25 * (1 - am + af) ** 2)
            s.areset = bool(areset)
            s.ga_first = True                                                  # 가장 거친 층 한 블록 커널 (결과 불변이나 이 층 크기에서 더 느림 -- 기본 끔)                                                  # 정의 변경: 거친 층 ω 를 첫 프레임 (또는 다시 할당 뒤 첫) 값으로 재사용 (거친 층 거듭제곱법 생략)                                                  # 정의 변경: 한 프레임 안 단계 τ 가 같으면 층 0 ω 를 다시 재지 않음                                                            # 걸음 끝 |g| 를 한 번 더 잼 (진단; 시간 측정 판에서는 끔)                                                       # 관성 질량 하한 계수 c (0 = 없음)
        s.Kc = int(K_contact) if K_contact else None                                   # 접촉 적응: 앞 프레임에 고착 노드가 있으면 이 걸음 수
        s.Kcur = s.K
        s.graphs = {}                                                                  # 걸음 수별 프레임 그래프 (접촉 적응)
        s.bc = bc                                                                      # [(center(3), size(3), vel(3), t0, t1)] cuboid 구속 또는 None
        s.mod = cp.RawModule(code=build_src(s._src0, s.adapt, s.goct, s.floor), options=("--std=c++14",))
        s.CT = torch.zeros(16, device=dev, dtype=torch.int32); s.CD = torch.zeros(4, device=dev, dtype=torch.float64)
        s.fn = {}
        # 팔분 8 가지 기저 표 (BW: 값, BG: 월드 기울기)
        GG = 0.5 / math.sqrt(3.0)
        BW = np.zeros((8, 27)); BG = np.zeros((8, 27, 3))
        Ai64 = Ai.double().cpu().numpy()
        for oc in range(8):
            fxv = [1.0 + (GG if (oc >> (2 - a)) & 1 else -GG) for a in range(3)]
            w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
            dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
            for o in range(27):
                i, j, k = o // 9, (o // 3) % 3, o % 3
                BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
                gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
                BG[oc, o] = gy @ Ai64
        s.BW = torch.tensor(BW, device=dev, dtype=f32).reshape(-1).contiguous()
        s.BG = torch.tensor(BG, device=dev, dtype=f32).reshape(-1).contiguous()
        s.IP = torch.zeros(16, device=dev, dtype=torch.int32)
        s.SC = torch.zeros(64, device=dev, dtype=torch.float64)
        s.Np = torch.zeros(s.N, 3, device=dev, dtype=torch.int32)
        s.fx = torch.zeros(s.N, 3, device=dev, dtype=f32)
        s.pcd = torch.zeros(s.N, device=dev, dtype=torch.int32)
        s.nbad = torch.zeros(1, device=dev, dtype=torch.int32)
        s.BP = torch.zeros(max(1, len(bc or [])) * 10, device=dev, dtype=f32)
        s.cap_mul = cap_mul
        s.graph = None
        s.alloc(None)

    def k(s, name, n, args, b=128):
        if s._pkc is not None:                                                        # 지속형 판: 명령 목록에 모으기 (실행 안 함)
            i32 = np.int32
            if name == "rt_scan":                                                     # 한 블록 1024 스캔 -> 같은 출력의 격자 3 단계 스캔
                s._pkc.append(("pk_scan_a", s.capC, 256, (args[0], args[1], P(s.pk_bs)), s._pk_loc > 0))
                s._pkc.append(("pk_scan_b", 256, 256, (args[0], P(s.pk_bs), args[4], i32(s.pk_capB)), True))
                s._pkc.append(("pk_scan_c", s.capC, 256, (args[0], args[1], P(s.pk_bs), args[2], args[3], args[4], i32(s.pk_capB)), s._pk_loc > 0))
                return
            assert name not in ("rt_mg_csolve", "ao_act_scan"), name
            s._pkc.append((name, int(n), int(b), tuple(args), s._pk_loc > 0))
            return
        if name not in s.fn:
            s.fn[name] = s.mod.get_function(name)
        if getattr(s, "prof", None) is not None:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        with RTC._Stream():
            s.fn[name](((int(n) + b - 1) // b,), (b,), args)
        if getattr(s, "prof", None) is not None:
            e1.record()
            s.prof.append((name, e0, e1))

    def profile(s, t):
        """커널별 GPU 시간 (그래프 밖에서 같은 순서로 한 프레임; 상태는 되돌림). -> {이름: [ms 합, 호출 수]}"""
        bk = s._bk()
        s.set_bc(t)
        s.prof = []
        s.record(t)
        torch.cuda.synchronize()
        o = {}
        for nm, e0, e1 in s.prof:
            a_, c_ = o.get(nm, (0.0, 0))
            o[nm] = (a_ + e0.elapsed_time(e1), c_ + 1)
        s.prof = None
        s._rs(bk)
        return {k_: [round(a_, 4), c_] for k_, (a_, c_) in o.items()}

    def alloc(s, need):
        """용량: 첫 프레임 상자에서 cap_mul 배 (need 가 주어지면 그 cap_mul 배)."""
        dev, f32 = s.dev, torch.float32
        if need is None:
            y = s.x.double() @ s.Ai.double().T
            b = torch.floor(y - 0.5)
            nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
            nn = nb[0] * nb[1] * nb[2]
            ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
            uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=dev)).sum(1)).numel()
            need = (nn, ncb, uc)
            s.nb0 = nb
        if not hasattr(s, "nb0"):
            s.nb0 = [round(need[0] ** (1 / 3))] * 3
        s.capN, s.capC, s.capQc = [int(q * s.cap_mul) + 1024 for q in need]
        capQ = s.capQc * 8
        Z = lambda *sh: torch.zeros(*sh, device=dev, dtype=f32)
        s.occ = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cidx = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cb = torch.zeros(s.capQc, 3, device=dev, dtype=torch.int32)
        s.Aacc = Z(capQ, 28); s.Cacc = Z(s.capQc, 30)
        s.badq = torch.zeros(capQ, device=dev, dtype=torch.int32); s.badc = torch.zeros(s.capQc, device=dev, dtype=torch.int32)
        s.Mo = Z(capQ) if s.goct >= 0 else None
        s.bar = bool(s.floor and s.floor.get("bar"))                                  # 장벽 판 (절 AN.19): 표본점 누적·점별 스칼라·노드 스칼라
        if s.bar:
            s.bx = Z(capQ * 4); s.bpg = Z(capQ); s.bph = Z(capQ); s.bact = torch.zeros(capQ, device=dev, dtype=torch.int32); s.bGn = Z(s.capN); s.bKn = Z(s.capN)
        s.Fq = Z(capQ, 9); s.Vq = Z(capQ); s.mq = Z(capQ); s.xq0 = Z(capQ, 3); s.H = Z(capQ, 81)
        s.mn = Z(s.capN); s.mv = Z(s.capN, 3); s.mqn = Z(s.capN); s.an = Z(s.capN)
        s.U = Z(s.capN, 3); s.vk = Z(s.capN, 3); s.Ut = Z(s.capN, 3); s.z0 = Z(s.capN, 3); s.y = Z(s.capN, 3); s.yh = Z(s.capN, 3); s.eta = Z(s.capN, 3)
        s.sk = Z(s.capN); s.pm = torch.zeros(s.capN, device=dev, dtype=torch.int32)
        s.f = torch.zeros(s.capN, 3, device=dev, dtype=torch.float64)
        s.B = Z(s.capN, 9); s.Dinv = Z(s.capN, 9)
        s.r = Z(s.capN, 3); s.z = Z(s.capN, 3); s.p = Z(s.capN, 3); s.dl = Z(s.capN, 3); s.Tp = Z(s.capN, 3); s.KTp = Z(s.capN, 3); s.Ap = Z(s.capN, 3); s.yc = Z(s.capN, 3)
        s.fixd = torch.zeros(s.capN, device=dev, dtype=torch.uint8) if s.bc else None
        s.vbc = Z(s.capN, 3) if s.bc else None
        s.g0buf = Z(s.capN, 3)
        s.graphs = {}
        s._wdone = False
        s.an2 = Z(s.capN)
        s.p0 = Z(s.capN, 3)
        s.ma = Z(s.capN, 3); s.ak = Z(s.capN, 3)
        if not hasattr(s, "pa"):
            s.pa = torch.zeros(s.N, 3, device=dev, dtype=torch.float32)
        s.meff = Z(s.capN); s.U0 = Z(s.capN, 3); s.v0 = Z(s.capN, 3); s.wb = Z(s.capN, 3); s.wvb = Z(s.capN, 3)
        s.DG = torch.zeros(4 * max(s.K, s.Kc or 0) * s.nt, 64, device=dev, dtype=torch.float64)
        s.DGF = torch.zeros(4 * max(s.K, s.Kc or 0), 64, device=dev, dtype=torch.float64)    # 걸음 끝 SC (gfin: SC1 = |g_끝|²)          # 진단: 뉴턴 반복마다 [PCG 시작 직후 SC, 선탐색 뒤 SC]
        # 멀티그리드 층 (1 .. nl−1): 첫 프레임 상자에서 차원을 내려 잡고 cap_mul 배
        s.ML = []
        if s.nl > 1:
            nbh = list(s.nb0)
            capn, capq = [0], [0]
            for l in range(1, s.nl):
                nbh = [q // 2 + 3 for q in nbh]
                cn = int(nbh[0] * nbh[1] * nbh[2] * s.cap_mul) + 1024
                cq = int((nbh[0] - 2) * (nbh[1] - 2) * (nbh[2] - 2) * 8 * s.cap_mul) + 1024
                cq = (cq + 7) // 8 * 8
                capn.append(cn); capq.append(cq)
                L = dict(capn=cn, capq=cq, Fq=Z(cq, 9), Vq=Z(cq), H=Z(81, cq), B=Z(cn, 9), D=Z(cn, 9), mc=Z(cn), A=Z(cq, 28), C=Z(cq // 8, 30),
                         badq=torch.zeros(cq, device=dev, dtype=torch.int32), badc=torch.zeros(cq // 8, device=dev, dtype=torch.int32),
                         r=Z(cn, 3), z=Z(cn, 3), t=Z(cn, 3), KT=Z(cn, 3), y=Z(cn, 3))
                s.ML.append(L)
            s.capn_d = torch.tensor(capn, device=dev, dtype=torch.int32); s.capq_d = torch.tensor(capq, device=dev, dtype=torch.int32)
            s.LM = torch.zeros(s.nl * 8, device=dev, dtype=torch.int32)
            s.W = torch.zeros(s.nl, device=dev, dtype=torch.float64); s.SCM = torch.zeros(s.nl * 4, device=dev, dtype=torch.float64)
            s.m0t = Z(s.capN, 3); s.m0u = Z(s.capN, 3); s.m0KT = Z(s.capN, 3); s.zo = Z(s.capN, 3); s.mnd = s.mn
            if s.mg2:                                                                  # 절 AN.20: 점유 칸 목록 용량 (지금 입자 상자에서 센 점유 칸 x cap_mul)
                yb = torch.floor(s.x.double() @ s.Ai.double().T - 0.5)
                s.mg2_alloc(torch.unique((yb - yb.min(0).values).long(), dim=0))
        if s.pk:                                                                       # 지속형 판 보조 버퍼: 칸 스캔 덩어리 합, 결정적 판 정렬 (칸 개수·시작·자리, 순번, 모으기 임시)
            s.pk_capB = (s.capC + 255) // 256
            s.pk_bs = torch.zeros(s.pk_capB + 1, device=dev, dtype=torch.int32)
            s.pk_cnt = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32); s.pk_st = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32)
            s.pk_cur = torch.zeros(s.capQc + 1, device=dev, dtype=torch.int32); s.pk_idx = torch.zeros(s.N, device=dev, dtype=torch.int32)
            s.pk_scr = torch.zeros(s.N * 9, device=dev, dtype=torch.float32)
        s.graph = None
        s.oct_bind()

    def _mods(s):
        return [s.mod] + ([s.pkmod] if s.pkmod is not None else [])

    def oct_bind(s):
        """점유 가중 판: 장치 전역 AF_MO (팔분 질량 버퍼)·AF_GOCT (κ) 를 지금 모듈에 (다시 할당·모듈 바꿈 뒤마다). 장벽 판 전역도 여기서."""
        if getattr(s, "bar", False):
            for mod in s._mods():
                for nm, t in (("AF_BX", s.bx), ("AF_BPG", s.bpg), ("AF_BPH", s.bph), ("AF_BACT", s.bact), ("AF_XP", s.x)):
                    g = mod.get_global(nm); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(t.data_ptr())
                g = mod.get_global("AF_BDET"); cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = 1 if getattr(s, "det", False) else 0
                g = mod.get_global("AF_BNRM"); cp.ndarray((3,), dtype=cp.float32, memptr=g)[:] = cp.asarray(np.array(s.floor["NR"], dtype=np.float32))
        if s.goct < 0:
            return
        for mod in s._mods():
            g = mod.get_global("AF_MO"); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.Mo.data_ptr())
            g = mod.get_global("AF_GOCT"); cp.ndarray((1,), dtype=cp.float32, memptr=g)[0] = s.goct


    # ---------- 멀티그리드 (그래프 안) ----------
    def mg_build(s):
        i32, f = np.int32, np.float32
        it2 = f(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        s.k("rt_mg_dims", 1, (P(s.IP), P(s.LM), i32(s.nl), P(s.capn_d), P(s.capq_d)), b=1)
        for l in range(1, s.nl):
            L = s.ML[l - 1]
            for t_, wh, mul in ((L["A"], 1, 28), (L["C"], 2, 30), (L["B"], 0, 9), (L["mc"], 0, 1)):
                s.k("rt_zero_lm", t_.numel(), (i32(l), P(s.LM), i32(wh), i32(mul), P(t_), P(None)), b=256)
            s.k("rt_zero_lm", L["badq"].numel(), (i32(l), P(s.LM), i32(1), i32(1), P(None), P(L["badq"])), b=256)
            s.k("rt_zero_lm", L["badc"].numel(), (i32(l), P(s.LM), i32(2), i32(1), P(None), P(L["badc"])), b=256)
            src_cap = (s.capQc * 8) if l == 1 else s.ML[l - 2]["capq"]
            Fs = s.Fq if l == 1 else s.ML[l - 2]["Fq"]
            Vs = s.Vq if l == 1 else s.ML[l - 2]["Vq"]
            s.k("rt_mg_acc", src_cap, (i32(l), P(s.IP), P(s.LM), P(s.cb), P(Fs), P(Vs), P(L["A"]), P(L["C"]), P(L["badq"]), P(L["badc"])))
            s.k("rt_mg_fin", L["capq"], (i32(l), P(s.LM), P(L["A"]), P(L["C"]), P(L["badq"]), P(L["badc"]), P(L["Fq"]), P(L["Vq"])))
            if s.mg2:
                s.mg2_occ(l)
            mprev = s.mn if l == 1 else s.ML[l - 2]["mc"]
            s.k("rt_mg_restrict", L["capn"], (i32(l - 1), P(s.LM), P(mprev), P(L["mc"]), i32(1)), b=256)
            s.k("rt_mg_elem", L["capq"], (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), f(s.mu), f(s.lam), P(L["H"]), i32(L["capq"]), P(L["B"])))
            s.k("rt_mg_D", L["capn"], (i32(l), P(s.LM), P(L["mc"]), it2, P(L["B"]), P(L["D"])), b=256)
        for l in range(1, s.nl):
            if not (s.mg_lreuse and getattr(s, "_wdone", False)):
                s.mg_power(l)
        s._wdone = True

    def mg_Az(s, l, z, KT, tz_ready=False):
        """KT += K (T z) (층 0) 또는 K z (거친 층). KT 는 앞 소비 커널이 비워 둔다. tz_ready: 층 0 의 T z 가 이미 m0u 에 (융합 노드 연산)."""
        i32, f = np.int32, np.float32
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        if l == 0:
            if not tz_ready:
                s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(3), P(s.mn), f(0), P(s.sk), P(s.pm), *[f(q) for q in nr], P(None), P(s.W), P(None), P(None), P(z), P(s.m0u)), b=256)
            if s.mv2:
                s.mg2_mv(0, s.m0u, KT)
                return
            s.k("rt_mv", s.capQc * 8, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.m0u), P(KT), i32(s.capQc * 8)))
        else:
            if s.mv2:
                s.mg2_mv(l, z, KT)
                return
            L = s.ML[l - 1]
            s.k("rt_mg_mv", L["capq"], (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(z), P(KT)))

    def mg_lv(s, l):
        if l == 0:
            return dict(mn=s.meff, D=s.Dinv, cap=s.capN)
        L = s.ML[l - 1]
        return dict(mn=L["mc"], D=L["D"], cap=L["capn"])

    def mg_power(s, l):
        i32, f = np.int32, np.float32
        it2 = f(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        V = s.mg_lv(l)
        z = s.m0t if l == 0 else s.ML[l - 1]["z"]
        y = s.zo if l == 0 else s.ML[l - 1]["y"]
        KT = s.m0KT if l == 0 else s.ML[l - 1]["KT"]
        s.k("rt_fill_rand", V["cap"], (i32(l), P(s.LM), P(z)), b=256)
        for it in range(s.mg_pow):
            s.k("rt_mg_pw0", 1, (i32(l), P(s.SCM)), b=1)
            s.mg_Az(l, z, KT)
            s.k("rt_mg_pw1", V["cap"], (i32(l), P(s.LM), P(V["mn"]), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(V["D"]), P(z), P(KT), P(y), P(s.SCM)), b=256)
            s.k("rt_mg_pw2", V["cap"], (i32(l), P(s.LM), P(y), P(z), P(s.SCM), P(s.W), f(s.mg_omega), i32(1 if it == s.mg_pow - 1 else 0)), b=256)

    def _pkl(s, on):
        """지속형 판: on 이면 그 안에서 모은 명령을 블록 0 전용 묶음으로 (MG 작은 층)."""
        import contextlib

        @contextlib.contextmanager
        def cm():
            if on:
                s._pk_loc += 1
            try:
                yield
            finally:
                if on:
                    s._pk_loc -= 1
        return cm()

    def mg_vc(s, l, r, z):
        with s._pkl(s._pkc is not None and l >= 1 and l >= s.pk_loc_l):
            s._mg_vc(l, r, z)

    def _mg_vc(s, l, r, z):
        """z = V 순환 (r). 층 0 의 r·z 는 고운 노드 배열 (조밀 상자)."""
        i32, f = np.int32, np.float32
        it2 = f(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        V = s.mg_lv(l)
        KT = s.m0KT if l == 0 else s.ML[l - 1]["KT"]
        res = s.m0t if l == 0 else s.ML[l - 1]["t"]
        node = lambda mode, rr, kt, zz, rs: s.k("rt_mg_node", V["cap"], (i32(l), P(s.LM), i32(mode), P(V["mn"]), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(V["D"]), P(s.W),
                                                                         P(rr), P(kt), P(zz), P(rs)), b=256)
        csolve = (l == s.nl - 1 and s.mg_csolve and l > 0 and not getattr(s, "det", False))
        if l == 0:
            node(6, r, None, z, s.m0u)                                                 # z = ω D r, m0u = T z (융합)
        elif not csolve:
            node(0, r, None, z, None)
        if l == s.nl - 1:
            if csolve:
                L = s.ML[l - 1]                                                          # 가장 거친 층 평활을 한 블록 커널로 (z = ω D r 부터)
                s.k("rt_mg_csolve", 1024, (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(V["mn"]), it2, P(V["D"]), P(s.W),
                                           P(r), P(z), P(L["KT"]), i32(s.mg_nc)), b=1024)
                return
            for _ in range(s.mg_nc - 1):
                s.mg_Az(l, z, KT); node(1, r, KT, z, None)
            return
        Lc = s.ML[l]
        if l == 0:
            s.mg_Az(0, z, KT, tz_ready=True); node(7, r, KT, z, s.m0u)                 # m0u = T (r − A z) (융합)
            with s._pkl(s._pkc is not None and 1 >= s.pk_loc_l):
                s.k("rt_mg_restrict2" if s.mv2 else "rt_mg_restrict", Lc["capn"], (i32(0), P(s.LM), P(s.m0u), P(Lc["r"]), i32(3)), b=256)
            if not (1 >= s.pk_loc_l) and not (s.nl - 1 == 1 and s.mg_csolve):
                s.pk_chain("rt_mg_restrict", "rt_mg_node")                             # 다음 = 거친 층 첫 노드 연산 (같은 거친 노드)
        else:
            s.mg_Az(l, z, KT); node(2, r, KT, z, res)
            with s._pkl(s._pkc is not None and l + 1 >= s.pk_loc_l):
                s.k("rt_mg_restrict2" if s.mv2 else "rt_mg_restrict", Lc["capn"], (i32(l), P(s.LM), P(res), P(Lc["r"]), i32(3)), b=256)
            if not (l + 1 >= s.pk_loc_l) and not (s.nl - 1 == l + 1 and s.mg_csolve):
                s.pk_chain("rt_mg_restrict", "rt_mg_node")
        if s.mg_fuse and l + 1 == s.mg_fuse:                                           # 절 AN.20: 층 mg_fuse 이상 V 순환 한 블록
            s.mg2_cvc(Lc["r"], Lc["z"])
        else:
            s.mg_vc(l + 1, Lc["r"], Lc["z"])
        if l == 0:
            s.k("rt_mg_prolong", V["cap"], (i32(0), P(s.LM), P(Lc["z"]), P(s.m0u), i32(0)), b=256)
            s.pk_chain("rt_mg_prolong", "rt_mg_node")                                  # 다음 = z += T P z_c (같은 노드)
            if s.fuse2:
                node(9, s.m0u, None, z, s.m0u)                                         # z += T P z_c, m0u = T z (융합; 다음 Az 의 T z)
            else:
                node(5, s.m0u, None, z, None)                                          # z += T P z_c
        else:
            s.k("rt_mg_prolong", V["cap"], (i32(l), P(s.LM), P(Lc["z"]), P(z), i32(1)), b=256)
        s.mg_Az(l, z, KT, tz_ready=(l == 0 and s.fuse2)); node(1, r, KT, z, None)

    def zf(s, t, cnt_idx=None, mul=1):
        n = t.numel()
        if t.dtype == torch.float64:
            s.k("rt_zero_d", n, (np.int32(n), P(s.IP if cnt_idx is not None else None), np.int32(cnt_idx or 0), np.int32(mul), P(t)), b=256)
        elif t.dtype == torch.int32:
            s.k("rt_zero_i", n, (np.int32(n), P(t)), b=256)
        else:
            s.k("rt_zero_f", n, (np.int32(n), P(s.IP if cnt_idx is not None else None), np.int32(cnt_idx or 0), np.int32(mul), P(t)), b=256)

    def stages(s, K):
        """한 프레임 단계 목록. 단계 = dict(tau, al, beta, thinv, kind (theta|bdf2), dtf (고정 노드 변위 시간))."""
        dt = s.h / K
        out = []
        for _ in range(K):
            if s.ga:
                c = s.ga_c
                tau = math.sqrt((1 - c["af"]) * c["be"] * dt * dt / (1 - c["am"]))
                al = c["gm"] * dt / ((1 - c["af"]) * c["be"] * dt * dt)
                out.append(dict(kind="ga", tau=tau, al=al, beta=0.0, thinv=1.0 / (1 - c["af"]), dtf=(1 - c["af"]) * dt, frac=1.0, last=True, dt=dt, **c))
                continue
            if s.scheme == "trbdf2":
                ga = 2.0 - math.sqrt(2.0)
                t1 = 0.5 * ga * dt
                out.append(dict(kind="theta", tau=t1, al=1.0 / (0.25 * ga * dt), beta=0.5, thinv=2.0, dtf=ga * dt, save0=True, frac=ga, last=False))
                c2 = (1.0 - ga) / (2.0 - ga); c1 = 1.0 / (ga * (2.0 - ga)); c0 = (1.0 - ga) ** 2 / (ga * (2.0 - ga))
                out.append(dict(kind="bdf2", tau=c2 * dt, al=1.0 / (c2 * dt), beta=0.0, thinv=1.0, dtf=(1.0 - ga) * dt, c1=c1, c0=c0, frac=1.0 - ga, last=True))
            else:
                th = s.theta
                out.append(dict(kind="theta", tau=th * dt, al=1.0 / (th * th * dt), beta=1.0 - th, thinv=1.0 / th, dtf=th * dt, frac=1.0, last=True))
        return out

    def record(s, t_frame, K=None):
        """한 프레임 커널 순서 (그래프 캡처 대상). t_frame = 구속 시각 (고정값; 시각이 바뀌는 구속은 프레임마다 다시 캡처)."""
        N, IP = s.N, s.IP
        i32, f = np.int32, np.float32
        capN, capQ = s.capN, s.capQc * 8
        s.k("rt_reset", 1, (P(IP),), b=1)
        s.ctl(0)
        s.k("rt_coord", N, (i32(N), P(s.x), P(s.Ai), P(s.Np), P(s.fx), P(IP)))
        s.k("rt_box", 1, (P(IP), i32(capN), i32(s.capC)), b=1)
        s.zf(s.occ)
        s.k("rt_mark", N, (i32(N), P(s.Np), P(s.fx), P(IP), P(s.pcd), P(s.occ)))
        s.k("rt_scan", 1024, (P(IP), P(s.occ), P(s.cidx), P(s.cb), i32(s.capQc)), b=1024)
        for t_, ci, mul in ((s.Aacc, 12, 28), (s.Cacc, 11, 30), (s.mn, 9, 1), (s.mv, 9, 3), (s.mqn, 9, 1)) + (((s.ma, 9, 3),) if s.ga else ()) + (((s.Mo, 12, 1),) if s.goct >= 0 else ()):
            s.zf(t_, ci, mul)
        s.zf(s.badq); s.zf(s.badc); s.zf(s.nbad)
        if getattr(s, "bar", False):
            s.zf(s.bx)
        s.k("rt_p2g", N, (i32(N), P(s.x), P(s.v), P(s.F), P(s.m), P(s.vol), P(s.Np), P(s.fx), P(s.pcd), P(s.cidx), P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.mn), P(s.mv),
                              P(s.pa if s.ga else None), P(s.ma if s.ga else None)))
        s.k("rt_gauss", capQ, (P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.cb), P(s.A), P(s.BW), P(s.Fq), P(s.Vq), P(s.mq), P(s.xq0), P(s.mqn), P(s.nbad)))
        s.k("rt_vn", capN, (P(IP), P(s.mn), P(s.mv), P(s.vk), P(s.U)), b=256)
        if s.nl > 1:
            s.mg_build()
        if s.bc:
            s.bzero(s.fixd)
            for ib in range(len(s.bc)):
                s.k("rt_bc_box", capN, (P(IP), P(s.A), P(s.BP), i32(ib), P(s.fixd), P(s.vbc)), b=256)
        FL = s.floor
        nr = FL["NR"] if FL else [0.0, 0.0, 1.0]
        ST = s.stages(K or s.K)
        s.nstage = len(ST)
        for kstep, sg in enumerate(ST):
            it2 = f(1.0 / (sg["tau"] * sg["tau"]))
            if sg.get("save0"):
                s.dcopy(s.U0, s.U); s.dcopy(s.v0, s.vk)
            if sg["kind"] == "ga":
                if kstep == 0:
                    if s.ga_first:                                                     # 첫 프레임 (그래프 밖에서만): a0 = g − ∇Ψ(0)/m', m' = max(m, 1e-3 x 질량 중앙값)
                        s.zf(s.f, 9, 3); s.zf(s.y, 9, 3)
                        s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1), P(s.SC[10:]), P(s.f), P(s.H), i32(capQ), P(s.B)))
                        mm = s.mn[:int(s.IP[9])]; mlow = 1e-3 * float(mm[mm > 0].median())
                        s.k("rt_ga_a0", capN, (P(IP), P(s.mn), P(s.f), f(mlow), *[f(q) for q in s.g], P(s.fixd), P(s.ak)), b=256)
                    else:
                        s.k("rt_ga_anode", capN, (P(IP), P(s.mn), P(s.ma), P(s.fixd), P(s.ak)), b=256)
                s.k("rt_ga_step0", capN, (P(IP), P(s.U), P(s.vk), P(s.ak), f(sg["dt"]), f(sg["am"]), f(sg["af"]), f(sg["gm"]), f(sg["be"]), f(sg["tau"]), f(sg["al"]),
                                          *[f(q) for q in s.g], P(s.fixd), P(s.vbc), P(s.Ut), P(s.z0), P(s.y)), b=256)
                Ub = vb = None
            elif sg["kind"] == "bdf2":
                s.k("rt_bdf_base", capN, (P(IP), f(sg["c1"]), f(sg["c0"]), P(s.U), P(s.vk), P(s.U0), P(s.v0), P(s.wb), P(s.wvb)), b=256)
                Ub, vb = s.wb, s.wvb
            else:
                Ub, vb = s.U, s.vk
            if sg["kind"] != "ga":
                s.k("rt_step0g", capN, (P(IP), P(Ub), P(vb), f(sg["tau"]), f(sg["beta"]), *[f(q) for q in s.g], P(s.fixd), P(s.vbc), P(s.U), f(sg["dtf"]), P(s.Ut), P(s.z0), P(s.y)), b=256)
            if FL:
                s.zf(s.an, 9, 1)
                s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), f(sg["thinv"]), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                         f(FL["PN"]), f(FL["DXG"]), P(s.an)))
            s.k("rt_floor_n", capN, (P(IP), P(s.an), P(s.mqn), f(FL["KEX"] * (s.K / (K or s.K)) * sg["frac"] if FL else 1.0), i32(1 if FL else 0), i32(1 if (FL and FL["vout"]) else 0), f(sg["al"]), *[f(q) for q in nr],
                                     P(s.fixd), P(s.vbc), P(s.U), f(sg["dtf"]), P(s.Ut), P(s.z0), P(s.y), P(s.sk), P(s.pm), P(s.yh), P(s.eta)), b=256)
            if sg["kind"] == "ga" or s.bar:                                            # 장벽 판: 모든 적분기 시작점 = 걸음 시작 위치 (장벽 실현 가능)
                s.k("rt_ga_y0", capN, (P(IP), P(s.U), P(s.yh), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.y)), b=256)
            if s.bar:
                BA = (*[f(q) for q in nr], f(FL["PN"]), f(FL["DH"]), f(FL["KS"]))
                s.k("rt_bar_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg["thinv"]), *BA, i32(0), P(s.bGn), P(s.bKn), P(s.SC[10:])))
            s.ctl(1)
            for it in range(s.nt):
                s.ctl(2, it)
                s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
                s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8 | (4 if s.ls else 0)), P(s.SC[10:]), P(s.f), P(s.H),
                                      i32(capQ), P(s.B)))
                if s.bar:
                    s.zf(s.bGn); s.zf(s.bKn)
                    s.k("rt_bar_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg["thinv"]), *BA, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])))
                    if getattr(s, "det", False):
                        s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bpg), P(s.bGn)), b=256)
                        s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bph), P(s.bKn)), b=256)
                    s.k("rt_bar_n", capN, (P(IP), P(s.bGn), P(s.bKn), *[f(q) for q in nr], P(s.f), P(s.B)), b=256)
                if it == 0:
                    s.k("rt_meff", capN, (P(IP), P(s.mn), P(s.B), f(s.mfloor * sg["tau"] ** 2), P(s.meff)), b=256)
                s.k("rt_grad", capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
                s.ctl(3, it)
                if s.nl > 1:                                                           # MG 전처리: z = V(r) 로 바꿔 시작
                    if it == 0 and not (s.mg_wreuse and kstep > 0 and abs(getattr(s, "mg_tau", -1.0) - sg["tau"]) == 0.0):
                        s.mg_tau = sg["tau"]
                        s.mg_power(0)
                    s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(1)), b=1)
                    s.mg_vc(0, s.r, s.z)
                    s.zf(s.zo, 9, 3)
                    s.k("rt_dotz", capN, (P(IP), P(s.r), P(s.z), P(s.zo), P(s.SC), i32(1)), b=256)
                    s.k("rt_pinit", capN, (P(IP), P(s.z), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.Tp)), b=256)
                s.dcopy(s.DG[2 * (kstep * s.nt + it)], s.SC)                                  # 진단: SC0 = r·z 처음, SC1 = |g|²
                s.ctl(4, it)
                if s.descent:
                    s.dcopy(s.p0, s.p); s.k("rt_sc_copy", 1, (P(s.SC), i32(60), i32(0)), b=1)
                for ci in range(s.cg):
                    s.ctl(5, ci)
                    if ci == 0 or not s.fuse2:
                        s.zf(s.KTp, 9, 3)
                    s.k("rt_mv2" if s.mv2 else "rt_mv", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.Tp), P(s.KTp), i32(capQ)))
                    if s.bar:
                        s.k("rt_bar_mv", capN, (P(IP), P(s.bKn), *[f(q) for q in nr], P(s.Tp), P(s.KTp)), b=256)
                    s.k("rt_ap", capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.KTp), P(s.Ap), P(s.SC)), b=256)
                    if s.descent and ci == 0:
                        s.k("rt_sc_copy", 1, (P(s.SC), i32(61), i32(2)), b=1)
                    if s.nl > 1:
                        s.k("rt_cg1m", capN, (P(IP), P(s.SC), P(s.p), P(s.Ap), P(s.dl), P(s.r)), b=256)
                        s.pk_chain("rt_cg1m", "rt_mg_node")                            # 다음 = V 순환 첫 노드 연산 (z = ω D r, 같은 노드)
                        s.mg_vc(0, s.r, s.z)
                        s.pk_chain("rt_mg_node", "rt_dotz")                            # V 순환 끝 노드 연산 -> 내적 (같은 노드)
                        s.k("rt_dotz", capN, (P(IP), P(s.r), P(s.z), P(s.zo), P(s.SC), i32(0)), b=256)
                    else:
                        s.k("rt_cg1", capN, (P(IP), P(s.SC), P(s.Dinv), P(s.p), P(s.Ap), P(s.dl), P(s.r), P(s.z), P(s.SC)), b=256)
                    s.k("rt_cg2", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.z), P(s.p), P(s.Tp), P(s.KTp if s.fuse2 else None)), b=256)
                    s.k("rt_sc_roll", 1, (P(s.SC),), b=1)
                    s.ctl(6, ci)
                s.ctl(7, it)
                # 선탐색: 기울기 곱 g·δ = −r0·δ (r0 = rt_grad 가 남긴 처음 잔차)
                if s.ls and s.ls_n:
                    NC = i32(s.ls_n)
                    s.k("rt_lsyN", capN, (P(IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), NC), b=256)
                    if s.descent:
                        s.k("rt_fixdir", capN, (P(IP), P(s.SC), P(s.p0), P(s.dl)), b=256)
                        s.k("rt_sc_clear", 1, (P(s.SC), i32(9), i32(10)), b=1); s.k("rt_sc_clear", 1, (P(s.SC), i32(20), i32(40)), b=1)
                        s.k("rt_lsyN", capN, (P(IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), NC), b=256)
                    s.k("rt_elem_lsN", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), P(s.SC), NC))
                    if s.bar:
                        s.k("rt_bar_lsN", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), P(s.U), f(sg["thinv"]), *BA[:4], BA[4], BA[5], P(s.SC), NC))
                    s.k("rt_lspickN", 1, (P(s.SC), NC), b=1)
                elif s.ls:
                    assert not s.bar, "장벽 판은 --ls_n 선탐색만"
                    s.k("rt_lsy4", capN, (P(IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC)), b=256)
                    if s.descent:                                                      # g·δ ≥ 0 이면 전처리 기울기 방향으로 바꾸고 관성 몫·g·δ 를 다시
                        s.k("rt_fixdir", capN, (P(IP), P(s.SC), P(s.p0), P(s.dl)), b=256)
                        s.k("rt_sc_clear", 1, (P(s.SC), i32(5), i32(10)), b=1)
                        s.k("rt_lsy4", capN, (P(IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC)), b=256)
                    s.k("rt_elem_ls", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), P(s.SC)))
                    s.k("rt_lspick", 1, (P(s.SC),), b=1)
                s.dcopy(s.DG[2 * (kstep * s.nt + it) + 1], s.SC)                              # 진단: SC0 = r·z 끝, 5..8 관성, 9 g·δ, 10..13 탄성 E, 14 보폭
                s.k("rt_lsup", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(0 if s.ls else 1)), b=256)
            s.ctl(8)
            if s.gfin:                                                                 # 걸음 끝 y 에서 기울기 (같은 변수 바꿈) -> |g_끝|²
                s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
                s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8), P(s.SC[10:]), P(s.f), P(s.H), i32(capQ), P(s.B)))
                s.k("rt_grad", capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
                s.dcopy(s.DGF[kstep], s.SC)
            if FL:                                                                     # 진단: 걸음 끝 위치의 고착 몫 an2 (걸음 시작 an 과 비교)
                s.zf(s.an2, 9, 1)
                s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), f(sg["thinv"]), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                         f(FL["PN"]), f(FL["DXG"]), P(s.an2)))
            if FL and kstep == 0:
                s.k("rt_count_stuck", capN, (P(IP), P(s.an), P(IP)), b=256)
            dmp = f(s.damp ** ((s.h / (K or s.K)) / (s.h / s.K)) if (s.damp != 1.0 and sg["last"]) else 1.0)       # 감쇠는 걸음 끝 단계에서만 (걸음 길이 비례)
            if sg["kind"] == "ga":
                s.k("rt_ga_end", capN, (P(IP), f(sg["dt"]), f(sg["gm"]), f(sg["be"]), f(sg["al"]), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.ak),
                                        P(s.an if FL else None), i32(1 if s.areset else 0), P(s.fixd), P(s.vbc), dmp), b=256)
            elif kstep < len(ST) - 1:
                s.k("rt_stepend_g", capN, (P(IP), f(sg["thinv"]), f(sg["al"]), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.fixd), P(s.vbc), dmp), b=256)
        sl = ST[-1]
        if sl["kind"] == "ga":
            s.k("rt_g2p_ga", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.U), P(s.vk), P(s.ak), P(s.x), P(s.v), P(s.pa), P(s.F)))
            return
        s.k("rt_g2p_endg", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.y), P(s.z0), P(s.U), f(sl["thinv"]), f(sl["al"]), P(s.fixd), P(s.vbc), dmp,
                               P(s.x), P(s.v), P(s.F)))


    def dcopy(s, dst, src):
        """그래프 판의 torch 복사 (지속형 판에서는 같은 일의 장치 명령)."""
        if s._pkc is None:
            dst.copy_(src)
            return
        nw = dst.numel() * dst.element_size() // 4
        assert src.numel() * src.element_size() // 4 == nw and dst.is_contiguous() and src.is_contiguous()
        s.k("pk_copy32", nw, (np.int32(nw), P(dst), P(src)), b=256)

    def bzero(s, t):
        if s._pkc is None:
            t.zero_()
            return
        s.k("pk_zero_b", t.numel(), (np.int32(t.numel()), P(t)), b=256)

    def pk_sort(s):
        """결정적 판 입자 정렬 (DetMixin.det_sort 의 torch.sort(pcd, stable=True) + 재배열과 같은 결과) 을 장치 명령으로."""
        i32, IP = np.int32, s.IP
        s.zf(s.pk_cnt)
        s.k("pk_sort_cnt", s.N, (i32(s.N), P(IP), P(s.pcd), P(s.cidx), P(s.pk_cnt)), b=256)
        s.k("pk_sort_scan", 256, (i32(s.N), P(IP), P(s.pk_cnt), P(s.pk_st), P(s.pk_cur)), b=256)
        s.k("pk_sort_put", s.N, (i32(s.N), P(IP), P(s.pcd), P(s.cidx), P(s.pk_st), P(s.pk_cur), P(s.pk_idx)), b=256)
        s.k("pk_sort_cell", s.capQc, (P(IP), P(s.pk_st), P(s.pk_idx)), b=128)
        for nm in ("x", "v", "F", "m", "vol", "fx", "Np", "pcd", "perm") + (("trel",) if getattr(s, "trel", None) is not None else ()) + (("pa",) if getattr(s, "ga", None) else ()):
            t = getattr(s, nm)
            assert t.is_contiguous() and t.shape[0] == s.N
            nw = t.numel() * t.element_size() // 4 // s.N
            assert nw <= 9
            s.k("pk_gather", s.N * nw, (i32(s.N), P(IP), P(s.pk_idx), i32(nw), P(t), P(s.pk_scr)), b=256)
            s.k("pk_copy32", s.N * nw, (i32(s.N * nw), P(t), P(s.pk_scr)), b=256)

    def pk_chain(s, prod, cons):
        """지속형 판: 방금 모은 명령 (prod) 과 다음 명령 (cons) 이 같은 자리 (스레드 n 이 노드 n) 만 주고받으면 사이 격자 장벽을 뺀다 (AF_PK_CHAIN=0 이면 안 함)."""
        if s._pkc is None or os.environ.get("AF_PK_CHAIN", "1") != "1":
            return
        assert s._pkc[-1][0] == prod, (s._pkc[-1][0], prod)
        s._pk_chain_set.add(len(s._pkc) - 1)
        s._pk_chain_next = cons

    def pk_prepare(s):
        """지속형 판 모듈 (한 번): 소스 변환·컴파일, 블록 수 = 동시 상주 수 x SM 수, 장치 전역 묶기."""
        if s.pkmod is not None:
            return
        from . import fem_an_pk as PKM
        src = PKM.pk_src(s._src0, s.adapt, s.goct, "AO_DET" in s._src0)
        s.pk_meta = src.meta
        s.pkmod = cp.RawModule(code=src.code, options=("--std=c++14", "-DPKB=%d" % s.pkb, "-DPKMINB=%d" % s.pkminb) + tuple(os.environ.get("AF_PK_OPT", "").split()), enable_cooperative_groups=True)
        s.pkfn = s.pkmod.get_function("pk_run")
        nsm = torch.cuda.get_device_properties(torch.device(s.dev)).multi_processor_count
        fptr = s.pkfn.kernel.ptr if hasattr(s.pkfn, "kernel") else s.pkfn.ptr            # RawKernel (cupy 새 판) 또는 Function
        fa = lambda at: int(cp.cuda.driver.funcGetAttribute(at, fptr))                   # CU_FUNC_ATTRIBUTE_: 1 공유, 3 지역, 4 레지스터
        regs, smem, lmem = fa(4), fa(1), fa(3)
        try:
            occ = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(fptr, s.pkb, 0))
            how = "api"
        except Exception as ex:                                                         # 없으면 레지스터·스레드 상한으로 (A6000: SM 당 65536 레지스터, 1536 스레드)
            occ = min(65536 // (((regs + 7) // 8 * 8) * s.pkb), 1536 // s.pkb)
            how = "regs (%s)" % type(ex).__name__
        if os.environ.get("AF_PK_OCC"):
            occ = min(occ, int(os.environ["AF_PK_OCC"]))
        assert occ >= 1, "지속형 커널이 SM 에 한 블록도 못 올라감"
        s.pk_grid = occ * nsm
        s.pk_info = dict(grid=s.pk_grid, occ=occ, occ_how=how, nsm=nsm, pkb=s.pkb, regs=regs, lmem=lmem, smem=smem, loc=s.pk_loc_l)
        print("[지속형 판]", s.pk_info, flush=True)
        s.oct_bind()
        if hasattr(s, "d_part"):
            s.det_bind()

    def pk_capture(s, t_frame, K):
        """record() 를 모으기 모드로 돌려 명령 목록 -> 장치 배열. 돌려주는 것: replay() 가 지속형 커널 한 번을 띄우는 객체."""
        from . import fem_an_pk as PKM
        s.pk_prepare()
        s._pkc, s._pk_loc, s._pk_chain_set = [], 0, set()
        try:
            s.record(t_frame, K)
            cmds = s._pkc
        finally:
            s._pkc = None
        Cn, An, st = PKM.build_cmds(cmds, s.pk_meta, s._pk_chain_set)
        CMD = torch.from_numpy(Cn.copy()).to(s.dev)
        ARG = torch.from_numpy(An.view(np.int64).copy()).to(s.dev)
        s.pk_info.update(st)
        fn, grid, blk = s.pkfn, s.pk_grid, s.pkb

        class _PK:
            def replay(self_):
                with RTC._Stream():
                    fn((grid,), (blk,), (P(CMD), np.int32(Cn.shape[0]), P(ARG)))                 # 협동 실행 (모듈을 enable_cooperative_groups 로 만듦)
        g = _PK()
        g.CMD, g.ARG = CMD, ARG
        g.names = [c[0] + (":%d" % int(c[3][0]) if (c[0].startswith("rt_mg_") and isinstance(c[3][0], np.int32)) else "") + ("@L" if (Cn[i, 3] & 1) else "") for i, c in enumerate(cmds)]
        g.ncmd = Cn.shape[0]
        return g

    def pk_profile(s, t):
        """지속형 판 한 프레임을 장치 시계로 (블록 0 이 장벽마다 시각 기록; 상태는 되돌림). -> {명령 이름 (블록 0 묶음은 첫 명령@L): [ms 합, 장벽 수]}, 합계."""
        g = s.graph
        if g is None or not hasattr(g, "names"):
            return None
        bk = s._bk()
        s.set_bc(t)
        buf = torch.zeros(2 * (g.ncmd + 2), device=s.dev, dtype=torch.int64)
        gp = s.pkmod.get_global("PK_TP")
        cp.ndarray((1,), dtype=cp.uint64, memptr=gp)[0] = int(buf.data_ptr())
        g.replay(); torch.cuda.synchronize()
        cp.ndarray((1,), dtype=cp.uint64, memptr=gp)[0] = 0
        s._rs(bk)
        b = buf.view(-1, 2).cpu().numpy()
        o, tot, ns = {}, 0.0, 0
        for e in range(1, b.shape[0]):
            if b[e, 0] == 0:
                break
            dtm = (b[e, 0] - b[e - 1, 0]) * 1e-6
            nm = g.names[int(b[e, 1])]
            a_, c_ = o.get(nm, (0.0, 0))
            o[nm] = (a_ + dtm, c_ + 1)
            tot += dtm; ns += 1
        o = {k_: [round(a_, 4), c_] for k_, (a_, c_) in sorted(o.items(), key=lambda kv: -kv[1][0])}
        o["_total_ms"] = round(tot, 4); o["_syncs"] = ns
        return o

    def ctl(s, op, it=0):
        """허용치 조기 종료 제어 커널 (adapt 판만; 아니면 아무것도 안 넣음 -- 원래 그래프 그대로)."""
        if not s.adapt:
            return
        eN2 = s.tolN * s.tolN if s.tolN >= 0 else -1.0
        eP2 = s.tolP * s.tolP if s.tolP >= 0 else -1.0
        s.k("rt_ctl", 1, (np.int32(op), np.int32(it), P(s.SC), P(s.CT), P(s.CD), np.float64(eN2), np.float64(eP2)), b=1)

    def _bk(s):
        """프레임 시작 입자 상태 백업: x·v·F 와 함께 m·vol·perm (결정적 판은 P2G 에서 입자를 정렬하며 이 배열들도 재배열한다 -- 데움 실행·넘침 다시 하기 뒤 같이 되돌려야 질량·부피가 제 입자에 붙는다)."""
        return tuple(getattr(s, nm).clone() for nm in ("x", "v", "F", "m", "vol", "perm", "pa"))

    def _rs(s, bk):
        for nm, t in zip(("x", "v", "F", "m", "vol", "perm", "pa"), bk):
            getattr(s, nm).copy_(t)

    def set_bc(s, t):
        """구속 매개변수 (중심 = point + v (t − t0), 반 크기, 속도, 켜짐) 를 장치 버퍼에 (그래프 밖, 재생 전)."""
        if not s.bc:
            return
        v = []
        for (c_, sz_, vel_, t0, t1) in s.bc:
            on = 1.0 if (t0 <= t < t1) else 0.0
            v += [c_[a] + vel_[a] * (t - t0) for a in range(3)] + list(sz_) + list(vel_) + [on]
        s.BP.copy_(torch.tensor(v, dtype=torch.float32))

    def capture(s):
        s.mod.get_function("rt_reset")
        K = s.Kcur
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            s.record(0.0, K)                                                           # 데움 (함수 적재)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        if s.pk:                                                                       # 지속형 판: 그래프 대신 명령 목록 (같은 순서)
            g = s.pk_capture(0.0, K)
        else:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                s.record(0.0, K)
        s.graph = g
        s.graphs[K] = g

    def step(s, t):
        """한 프레임. 넘치면 상태를 되돌리고 용량 2 배로 다시 할당·캡처해 그 프레임을 다시 한다. 돌려주는 것: 다시 했는지. s.last_ms = 그래프 재생 GPU 시간 (CUDA 이벤트)."""
        bk = s._bk()
        if s.ga and s.ga_first:                                                        # generalized-α 첫 프레임 (a0 에 질량 중앙값이 필요해 그래프 밖에서 한 번)
            for tries in range(4):
                s.set_bc(t)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); s.record(t, s.Kcur); e1.record()
                torch.cuda.synchronize()
                if int(s.IP[13]) == 0:
                    break
                s._rs(bk); s.cap_mul = 1.5 + 0.5 * (tries + 1); s.alloc(None); s._rs(bk)
            s.ga_first = False
            s.last_ms = e0.elapsed_time(e1); s.contact_prev = int(s.IP[14]) > 0; s.last_K = s.Kcur
            return False
        Kw = s.Kc if (s.Kc and getattr(s, "contact_prev", False)) else s.K               # 접촉 적응: 앞 프레임 고착 노드가 있으면 K_c
        if Kw != s.Kcur:
            s.Kcur = Kw
            s.graph = s.graphs.get(Kw)
        if s.graph is None:
            s.capture()
            s._rs(bk)                      # 데움 실행이 상태를 바꿨으므로 되돌림
        s.set_bc(t)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        s.graph.replay()
        e1.record()
        redo = False
        tries = 0
        while int(s.IP[13]) != 0:
            if not bool(torch.isfinite(bk[0]).all()) or tries >= 3:                    # 비유한 상태 (발산) 면 상자가 정의되지 않아 매번 넘친다 -> 늘리지 않고 멈춤
                s.x.fill_(float("nan"))
                s.last_ms = float("nan")
                return False
            tries += 1
            s._rs(bk)
            s.cap_mul = 1.5 + 0.5 * tries                                              # 필요량은 지금 상태에서 다시 재므로 배수는 작게 (두 배씩 키우면 메모리가 넘친다)
            s.alloc(None)
            s.capture()
            s._rs(bk)
            s.set_bc(t)
            e0.record()
            s.graph.replay()
            e1.record()
            redo = True
        torch.cuda.synchronize()
        s.last_ms = e0.elapsed_time(e1)
        s.contact_prev = int(s.IP[14]) > 0
        s.last_K = s.Kcur
        return redo

    def counts(s):
        ip = s.IP.cpu().tolist()
        o = dict(nn=ip[9], nc=ip[11], nq=ip[12], nb=ip[6:9], nbad=int(s.nbad))
        if s.adapt:                                                                    # 마지막 프레임 반복 수 (뉴턴 합, PCG 합, 뉴턴 상한 걸음, PCG 상한 풀이, 걸음, 풀이)
            ct = s.CT.cpu().tolist()
            o["it"] = dict(nN=ct[2], nP=ct[3], capN=ct[4], capP=ct[5], steps=ct[6], solves=ct[7])
        return o

    def x_orig(s):
        o = torch.empty_like(s.x)
        o[s.perm] = s.x
        return o

    def diag(s):
        """마지막 프레임 진단: 뉴턴 반복별 [|g|², r·z 처음, r·z 끝, g·δ, E0, E(¼), E(½), E(1), 보폭] + 고착 불일치 (걸음 끝 몫 > 0 과 시작 몫 > 0 이 다른 노드 수, 몫 차 합)."""
        D = s.DG.cpu().numpy()
        out = []
        for k in range(s.K * s.nt):
            a, b = D[2 * k], D[2 * k + 1]
            if s.ls_n:
                out.append(dict(g2=a[1], rz0=a[0], rz1=b[0], gd=b[9], E0=b[20] + b[10], Ec=[b[21 + i] + b[40 + i] for i in range(s.ls_n)], st=b[14], fix=b[62]))
            else:
                out.append(dict(g2=a[1], rz0=a[0], rz1=b[0], gd=b[9], E0=b[5] + b[10], Ec=[b[6] + b[11], b[7] + b[12], b[8] + b[13]], st=b[14], fix=b[62]))
        nn = int(s.IP[9])
        r0 = torch.where(s.mqn[:nn] > 0, s.an[:nn] / s.mqn[:nn].clamp_min(1e-30), torch.zeros_like(s.an[:nn]))
        r1 = torch.where(s.mqn[:nn] > 0, s.an2[:nn] / s.mqn[:nn].clamp_min(1e-30), torch.zeros_like(s.an2[:nn]))
        return dict(it=out, stick_flip=int(((r0 > 0) != (r1 > 0)).sum()), stick_n0=int((r0 > 0).sum()), stick_n1=int((r1 > 0).sum()), stick_dr=float((r1 - r0).abs().sum()))

    def grel(s):
        """마지막 프레임 걸음별 |g_끝| / |g_처음| (gfin 일 때)."""
        D, F = s.DG.cpu().numpy(), s.DGF.cpu().numpy()
        return [float(np.sqrt(max(F[k][1], 0.0) / max(D[2 * (k * s.nt)][1], 1e-300))) for k in range(getattr(s, "nstage", s.K))]
