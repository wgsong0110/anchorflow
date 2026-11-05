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
// 절 CM3 (--ncomp): rt_scan 과 같은 출력 (정수, 비트 같음) 을 격자 3 단계로 -- 칸 상자 수십만 칸을 한 블록이 1024 씩 도는 rt_scan (0.7 ms) 대신.
__global__ void rt_scan_a(const int* IP, const int* occ, int* bc) {                 // 블록 256 고정: 블록별 점유 칸 수
    int n = IP[13] ? 0 : IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int c = __syncthreads_count(i < n && occ[i]);
    if (threadIdx.x == 0) bc[blockIdx.x] = c;
}
__global__ void rt_scan_b(int* IP, const int* bc, int* bo, int capQc) {               // 한 블록 1024: 블록 수 배타 스캔 -> nc·nq (넘치면 표시)
    __shared__ int sh[1024];
    int nb = ((IP[13] ? 0 : IP[10]) + 255) / 256, carry = 0;
    for (int base = 0; base < nb; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < nb) ? bc[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (i < nb) bo[i] = carry + sh[threadIdx.x] - v;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
__global__ void rt_scan_c(const int* IP, const int* occ, const int* bo, int* cidx, int* cb, int capQc) {   // 블록 256 고정: 자리 = 블록 시작 + 블록 안 순번
    __shared__ int ws[8];
    int n = IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int v = (i < n) ? occ[i] : 0;
    unsigned int m = __ballot_sync(0xffffffffu, v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) ws[w] = __popc(m);
    __syncthreads();
    if (i >= n) return;
    int id = bo[blockIdx.x];
    for (int k=0;k<w;k++) id += ws[k];
    id += __popc(m & ((1u << lane) - 1u));
    int n1 = IP[7]-2, n2 = IP[8]-2;
    if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
    else cidx[i] = -1;
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
#ifndef AF_GQM
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
#endif
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
#ifdef AF_FSTK
    bool qa_ = dxg < 0.f; dxg = fabsf(dxg);                                     // 절 CM12 (--fl_stk): dxg < 0 으로 부르면 (A) 가우스 점 고착 판정 (CM8 a 판 정의 그대로: 가우스 점 xq0, s < 0, 정답 sticky 가중)
#endif
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
#ifdef AF_BAR
#ifdef AF_FSTK
    float bm_ = qa_ ? 0.f : AF_BX[q*4+3]; if (bm_ > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[q*4+a]/bm_;
#else
    float bm_ = AF_BX[q*4+3]; if (bm_ > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[q*4+a]/bm_;   // 장벽 판: 고착 판정도 표본점 (팔분 입자 질량 중심) 에서
#endif
#endif
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }
    float s = (xq[0]-p0)*n0 + (xq[1]-p1)*n1 + (xq[2]-p2)*n2;
#ifdef AF_BAR
#ifdef AF_FSTK
    float sw_ = s; if (!qa_) s = (bm_ > 0.f) ? -1.f : 1.f;
#else
    float sw_ = s; s = (bm_ > 0.f) ? -1.f : 1.f;                                 // 장벽 판: 정답 sticky 가중 전체 (평면 위 띠 포함), 빈 팔분 점 없음
#endif
#else
    float sw_ = s;
#endif
    if (!(s < 0.f)) return;
    float u = (sw_ + pn)/dxg; float bf = floorf(u - 0.5f); float f = u - bf;
    float w3[3] = {0.5f*(1.5f-f)*(1.5f-f), 0.75f-(f-1.f)*(f-1.f), 0.5f*(f-0.5f)*(f-0.5f)};
    float wst = 0.f; for (int i=0;i<3;i++) if ((bf + (float)i)*dxg - pn < 0.f) wst += w3[i];
#ifdef AF_BST
#ifdef AF_FSTK
    if (qa_) ; else
#endif
    if (sw_ < AF_BDHC) wst = 1.f;                                               // 띠 고착 (--fl_bst, 절 AN.21): 장벽 띠 안 (d < d̂, 평면 아래 포함) 표본점은 완전 고착 (정답 sticky 노드 v = 0 과 같은 몫 1)
    else if (AF_BST == 1) wst = 0.f;                                            //   1 = 띠 밖은 고착 없음, 2 = 띠 밖은 정답 sticky 가중 그대로
#endif
    float mw = mq[q]*wst;
#ifdef AF_FLJ
    mw = mq[q]*(1.f - expf(AF_KEXQ*log1pf(-fminf(wst, 1.f))));                            // 점별 고착 (--fl_pt, 절 AN.17): 점 질량 x (1 − (1 − w_st)^(Δt/dt_정답)) -- 거듭제곱을 점에서 한 뒤 노드로 평균
#endif
    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);
}
// 노드: 고착 인자 k, 변수 바꿈 (sk, pm), ŷ, η, y = ŷ + T η. pm 0 = P I, 1 = P (I − nnᵀ) (바깥 법선 이탈), 2 = 고정 (T 0)
__device__ __forceinline__ void Tapply(float sk, int pm, float n0, float n1, float n2, const float* v, float* o) {
#ifdef AF_FSTK
    if (pm < 0) { float sn_ = __int_as_float(-pm - 1); float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2;          // 절 CM12: 접선 √k_t·법선 √k_n 따로 (pm = −1 − 비트(√k_n))
        o[0] = sk*(v[0]-vn_*n0) + sn_*vn_*n0; o[1] = sk*(v[1]-vn_*n1) + sn_*vn_*n1; o[2] = sk*(v[2]-vn_*n2) + sn_*vn_*n2; return; }
#endif
    if (pm == 2) { o[0] = o[1] = o[2] = 0.f; return; }
    if (pm == 0) { o[0] = sk*v[0]; o[1] = sk*v[1]; o[2] = sk*v[2]; return; }
    float vn = v[0]*n0 + v[1]*n1 + v[2]*n2;                                   // P v = v − (v·n) n (접선), (I − P) v = (v·n) n
    o[0] = sk*(v[0]-vn*n0) + vn*n0; o[1] = sk*(v[1]-vn*n1) + vn*n1; o[2] = sk*(v[2]-vn*n2) + vn*n2;
}
#ifdef AF_FSTK
// 절 CM12 노드 변수 바꿈 (접선 k_t, 법선 k_n 따로): ŷ = Ũ + (1 − k_t) P d + (1 − k_n) (I − P) d (d = z0' − Ũ), η = √k_t P (y − ŷ) + √k_n (I − P)(y − ŷ), y = ŷ + T η.
// k_n = 1 이면 pm 1 경로와 같은 식, k_n = k_t 이면 pm 0 경로와 같은 식 (이 함수는 k_n ≠ 1 일 때만 부름).
__device__ void fstk_node(int n, float kt, float kn, float n0, float n1, float n2, const float* d, const float* Ut, float* y, float* sk, int* pm, float* yh, float* eta) {
    float st = sqrtf(kt), sn = sqrtf(kn);
    float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; float nn[3] = {n0, n1, n2};
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-kt)*(d[a]-dn*nn[a]) + (1.f-kn)*dn*nn[a];
    float e2[3]; for (int a=0;a<3;a++) e2[a] = y[n*3+a] - h[a];
    float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
    int pmv = -1 - __float_as_int(sn);
    float et[3]; for (int a=0;a<3;a++) et[a] = st*(e2[a] - c2*nn[a]) + sn*c2*nn[a];
    float ty[3]; Tapply(st, pmv, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = st; pm[n] = pmv;
}
#endif
__global__ void rt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
#ifdef AF_FSTK
    float kfn_ = 1.f;
#endif
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
#ifdef AF_FSTK
        float ra_ = (mqn[n] > 0.f) ? fminf(AF_ANA[n]/mqn[n], 1.f) : 0.f;       // 절 CM12: (A) 가우스 점 고착 몫 -> k_A = (1 − r̄_A)^κ, 법선은 바깥 이탈 (vout, vv > 0) 이 아니면 k_A
        float ka_ = expf(kex*log1pf(-ra_));
        kfn_ = (vout && vv > 0.f) ? 1.f : ka_; k *= ka_;
#endif
    }
    if (fixd && fixd[n]) p = 2;
#ifdef AF_FSTK
    if (p != 2 && kfn_ != 1.f) { float dd_[3]; for (int a=0;a<3;a++) dd_[a] = z0[n*3+a] - Ut[n*3+a]; fstk_node(n, k, kfn_, n0, n1, n2, dd_, Ut, y, sk, pm, yh, eta); return; }
#endif
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
#ifdef AF_HSPEC
        {                                                                         // 절 CM3 --hspec: 같은 헤시안의 주값 분해 30 성분 (U 9, M = F V 9, C 12 x V_q) -- rt_mv2s 가 읽음
            float vq_ = vq, Cs[12];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) { float t = 0.f; for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (dm > 0.f) t += dm*Ev[RI[k]][m]*Ev[RJ[k]][m]; } Cs[k] = vq_*t; }
            for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
                float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
                lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
                Cs[6+pp] = vq_*0.5f*(lt + lf); Cs[9+pp] = vq_*0.5f*(lf - lt); }
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)(9 + a*3+b)*capQ + qq] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
            for (int k=0;k<9;k++) H[(size_t)k*capQ + qq] = U[k];
            for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = Cs[k];
        }
#else
        for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];                     // 성분 우선 배치 (곱 커널 읽기가 합쳐지게)
#endif
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
#ifdef AF_FSTK
        if (pm < 0) { float Pab = I - nn[a]*nn[b]; T[a*3+b] = sk*Pab + __int_as_float(-pm - 1)*(I - Pab); continue; }
#endif
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
__global__ void rt_ap(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* p, const float* KTp, float* Ap, double* SC, const float* Kb) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) {
        float kt[3] = {KTp[n*3], KTp[n*3+1], KTp[n*3+2]};
        if (Kb) { float K = Kb[n];                                                // 절 AN.22 --mgbar: 장벽 덩어리 곱 K n (n·T p) 융합 (rt_bar_mv 와 같은 식)
            if (K != 0.f) { float pp[3] = {p[n*3], p[n*3+1], p[n*3+2]}, tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, pp, tp); float c = K*(tp[0]*n0 + tp[1]*n1 + tp[2]*n2); kt[0] += c*n0; kt[1] += c*n1; kt[2] += c*n2; } }
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t);
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
    int ovf = 0;
    for (int l=1; l<nl; l++) {
        n0 = n0/2 + 3; n1 = n1/2 + 3; n2 = n2/2 + 3;
        int* o = LM + l*8; o[0] = n0; o[1] = n1; o[2] = n2; o[3] = n0*n1*n2; o[4] = n0-2; o[5] = n1-2; o[6] = n2-2; o[7] = (n0-2)*(n1-2)*(n2-2)*8;
        if (o[3] > capn[l] || o[7] > capq[l]) ovf = 1;
    }
    // 절 CM7: 넘친 프레임 (앞 단계 IP[13] 또는 거친 층) 이면 모든 층 차원·수를 0 -- 다시 할 프레임이라 값은 버림.
    // 전에는 넘친 층의 수만 0 이라 (층 0 은 IP[9] = 0 이어도 차원 IP[6..8] 은 큰 상자 그대로) 제한 rt_mg_restrict*·ao_mg_acc_det 가
    // 차원만 보고 용량 밖 칸 번호 (cidx·mn·r·bKn) 를 읽어 불법 접근 (wolf K 8 프레임 40 등).
#ifndef AF_MGDIMS_OLD
    if (ovf || IP[13]) { for (int l=0; l<nl; l++) for (int k=0;k<8;k++) LM[l*8+k] = 0; if (ovf) atomicOr(&IP[13], 2); }
#else
    for (int l=1; l<nl; l++) { int* o = LM + l*8; if (o[3] > capn[l] || o[7] > capq[l]) { o[3] = 0; o[7] = 0; atomicOr(&IP[13], 2); } }   // 옛 판 (진단 대조 전용)
#endif
}
// 거친 층 l 가우스 점 누적: 출처 = 층 l−1 의 가우스 점 (위치 tq = 칸 좌표 + 팔분 위치, X = F, 가중 V)
__global__ void rt_mg_acc(int l, const int* IP, const int* LM, const int* cbF, const float* Fsrc, const float* Vsrc, float* A, float* C, int* badq, int* badc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nsrc = (l == 1) ? IP[12] : LM[(l-1)*8+7];
#ifndef AF_MGDIMS_OLD
    if (q >= nsrc || LM[l*8+7] == 0) return;                                          // 절 CM7: 넘친 프레임 (층 차원 0) 이면 쓰지 않음
#else
    if (q >= nsrc) return;
#endif
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
#ifdef AF_HSPEC
        {                                                                         // 절 CM3 --hspec: 거친 층도 주값 분해 30 성분 (U, M = F V, C x V_q) -- rt_mg_mv2 (mv2_core) 가 읽음
            float Cs[12];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) { float t = 0.f; for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (dm > 0.f) t += dm*Ev[RI[k]][m]*Ev[RJ[k]][m]; } Cs[k] = vq*t; }
            for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
                float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
                lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
                Cs[6+pp] = vq*0.5f*(lt + lf); Cs[9+pp] = vq*0.5f*(lf - lt); }
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)(9 + a*3+b)*capQ + qq] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
            for (int k=0;k<9;k++) H[(size_t)k*capQ + qq] = U[k];
            for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = Cs[k];
        }
#endif
    }
#ifndef AF_HSPEC
    if (on) for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
#endif
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
                           const float* D, const double* W, const float* r, float* KTz, float* z, float* res, const float* Kb) {
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
    if (Kb) { float K = Kb[n];                                                   // 절 AN.22 --mgbar: 장벽 덩어리 헤시안을 노드 연산에 융합 (층 0: T (K n nᵀ) T z, 거친 층: K n nᵀ z)
        if (K != 0.f) { float zz[3] = {z[n*3], z[n*3+1], z[n*3+2]}, tz[3]; if (l == 0) Tapply(sk[n], pm[n], n0, n1, n2, zz, tz); else { tz[0] = zz[0]; tz[1] = zz[1]; tz[2] = zz[2]; }
            float c = K*(tz[0]*n0 + tz[1]*n1 + tz[2]*n2); float bv[3] = {c*n0, c*n1, c*n2}, tb[3];
            if (l == 0) Tapply(sk[n], pm[n], n0, n1, n2, bv, tb); else { tb[0] = bv[0]; tb[1] = bv[1]; tb[2] = bv[2]; }
            for (int a=0;a<3;a++) az[a] += tb[a]; } }
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
#ifdef AF_FK
    double inr = 1.0/nr; for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a]*inr);    // 절 AN.22 --fk: f64 나눗셈 셋 -> 하나
#else
    for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a] / nr);
#endif
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
    if (n == 0) SC[59] = bad ? 0.0 : 1.0;                                       // 절 AN.22: 방향을 안 바꾸면 1 (두 번째 관성 몫 계산 생략 문)
    if (n >= IP[9] || !bad) return;
    float a0 = (SC[61] > 0.0) ? (float)(SC[60]/SC[61]) : 0.f;
    for (int a=0;a<3;a++) dl[n*3+a] = a0*p0[n*3+a];
}
__global__ void rt_sc_copy(double* SC, int dst, int src) { SC[dst] = SC[src]; }
// 절 CM3 노드 압축 (--ncomp): 점유 칸 27 개 중 하나라도 이웃인 노드 (가우스 점 곱·질량·대각이 닿는 노드 전부) 목록 ACT[1..ACT[0]] (노드 번호 오름차순, 결정적).
// 그 밖 노드는 질량·대각·힘·곱이 모두 0 이라 PCG·V 순환 값이 목록 노드에 주는 몫이 0 -- 목록만 돌아도 목록 노드 값은 같다 (블록 합 순서만 다름).
__device__ __forceinline__ int rt_act_on(const int* IP, const int* occ, int n) {
    if (n >= IP[9]) return 0;
    int nb1 = IP[7], nb2 = IP[8];
    int ni = n / (nb1*nb2), nj = (n / nb2) % nb1, nk = n % nb2;
    int c0 = IP[6]-2, c1 = nb1-2, c2 = nb2-2;
    for (int i=0;i<3;i++) { int ci = ni - i; if (ci < 0 || ci >= c0) continue;
      for (int j=0;j<3;j++) { int cj = nj - j; if (cj < 0 || cj >= c1) continue;
        for (int k=0;k<3;k++) { int ck = nk - k; if (ck < 0 || ck >= c2) continue;
          if (occ[(ci*c1 + cj)*c2 + ck]) return 1; } } }
    return 0;
}
__global__ void rt_act_cnt(const int* IP, const int* occ, int* bc) {                // 블록 256 고정
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int c = __syncthreads_count(rt_act_on(IP, occ, n));
    if (threadIdx.x == 0) bc[blockIdx.x] = c;
}
__global__ void rt_act_scan(const int* IP, const int* bc, int* bo, int* ACT) {       // 한 블록 1024: 블록 수 배타 스캔
    __shared__ int sh[1024];
    int nb = (IP[9] + 255) / 256, carry = 0;
    for (int base = 0; base < nb; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < nb) ? bc[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (i < nb) bo[i] = carry + sh[threadIdx.x] - v;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) ACT[0] = carry;
}
__global__ void rt_act_wr(const int* IP, const int* occ, const int* bo, int* ACT) {  // 블록 256 고정
    __shared__ int ws[8];
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int v = rt_act_on(IP, occ, n);
    unsigned int m = __ballot_sync(0xffffffffu, v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) ws[w] = __popc(m);
    __syncthreads();
    int off = bo[blockIdx.x];
    for (int k=0;k<w;k++) off += ws[k];
    if (v) ACT[1 + off + __popc(m & ((1u << lane) - 1u))] = n;
}
// 선탐색 넓힘 (--ls_n NC): 후보 c_i = 2^-i (i = 0..NC−1). 관성 몫 SC[20] (c = 0), SC[21+i]; 탄성 SC[40+i]; g·δ SC[9]; 보폭 SC[14]
// 절 AN.22: 후보 고리를 컴파일 시 12 로 펼쳐 sc[] 를 레지스터에 (원래는 실행 시 NC 고리라 지역 메모리 -- 같은 연산·같은 순서, 결과 비트 같음). gate != 0 이고 *gate != 0 이면 아무것도 안 함 (하강 방향을 안 바꾼 반복의 두 번째 계산 생략)
__global__ void rt_lsyN(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, int NC, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, sc[12];
#pragma unroll
    for (int i=0;i<12;i++) sc[i] = 0.0;
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*(double)Mi*e0*e0; gd += -(double)r0[n*3+a]*d;
            float c = 1.f;
#pragma unroll
            for (int i=0;i<12;i++) { if (i < NC) { float e = e0 + c*d; sc[i] += 0.5*(double)Mi*e*e; } c *= 0.5f; } } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]);
#pragma unroll
    for (int i=0;i<12;i++) if (i < NC) blocksum_add(sc[i], &SC[21+i]);
}
// 절 AN.22 --fk (같은 식, 반올림만 다름): 관성 몫 후보를 이차식으로 -- ½ Σ Mi |η + c δ|² = S0 + c S1 + c² S2 (S0 = SC20, S1 = Σ Mi η·δ -> SC33, S2 = ½ Σ Mi |δ|² -> SC34),
// 후보마다 블록 합 12 번 대신 4 번 (f64 처리량이 낮은 A6000 에서 rt_lsyN 이 노드 커널 중 가장 비쌌음). rt_lsyfin 이 SC[21+i] 를 채움.
__global__ void rt_lsyQ(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, s1 = 0.0, s2 = 0.0;
    if (n < IP[9]) { double Mi = (double)(mn[n]*itau2);
        for (int a=0;a<3;a++) { double e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*Mi*e0*e0; gd += -(double)r0[n*3+a]*d; s1 += Mi*e0*d; s2 += 0.5*Mi*d*d; } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]); blocksum_add(s1, &SC[33]); blocksum_add(s2, &SC[34]);
}
__global__ void rt_lsyfin(double* SC, int NC, const double* gate) {
    if (gate && *gate != 0.0) return;
    double c = 1.0; for (int i=0;i<NC;i++) { SC[21+i] = SC[20] + c*SC[33] + c*c*SC[34]; c *= 0.5; }
}
// 절 AN.22 선탐색 조기 판정 (결과 같음): 후보 c = 1 이 아르미호를 만족하면 SC[63] = 1 -> 나머지 후보 커널 (gate = SC+63) 생략. rt_lspickN 은 첫 만족 후보를 고르므로 결과가 같다.
__global__ void rt_lsgate(double* SC) {
    double E0 = SC[20] + SC[10]; double gd = SC[9]; double E1 = SC[21] + SC[40];
    SC[63] = (isfinite(E1) && E1 <= E0 + 1e-4*gd) ? 1.0 : 0.0;
}
__global__ void rt_sc_clear_g(double* SC, int a, int b, const double* gate) { if (*gate != 0.0) return; for (int i=a;i<b;i++) SC[i] = 0.0; }
__global__ void __launch_bounds__(128) rt_elem_lsN(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;                                           // 절 AN.22: 후보 c = 1 이 받아들여졌으면 나머지 후보 생략
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
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
// 절 AN.22 시작점 탐색 (--y0ls, 결과 정의 같음 -- 풀리는 증분 포텐셜·장벽 활성 판정 (걸음 시작 U) 그대로, 뉴턴 시작점만):
// rt_floor_n 뒤 η = η(Ũ) (관성 예측) 를 dl 에 둔 뒤 rt_ga_y0 가 η = η(U) 로 바꾸면 dl ← η(Ũ) − η(U). 선탐색과 같은 후보 커널로 c = 2^-i (i = 0..NC−1) 의 전체 에너지 (관성 + 탄성 + 장벽) -> 유한한 것 중 가장 작은 c -> SC[14].
__global__ void rt_y0dir(const int* IP, const float* eta, float* dl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) dl[n*3+a] -= eta[n*3+a];
}
__global__ void rt_y0pick(double* SC, int NC) {
    double best = 0.0, st = 0.0, c = 1.0; bool have = false;
    for (int i=0;i<NC;i++) { double E1 = SC[21+i] + SC[40+i]; if (isfinite(E1) && E1 < 1e29 && (!have || E1 < best)) { best = E1; st = c; have = true; } c *= 0.5; }
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
#ifdef AF_FSTK
    if (p < 0) { float sn_ = __int_as_float(-p - 1); float isn_ = (sn_ > 1e-6f) ? 1.f/sn_ : 0.f;               // 절 CM12: 법선 성분은 1/√k_n
        float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = isn_*dn*nn[a] + isk*(dd[a] - dn*nn[a]); } else
#endif
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
#ifdef AF_FSTK
    stk = areset && AF_ANA[n] > 0.f;                                            // 절 CM12: (A) 가우스 점 고착 노드는 a 판 그대로 세 성분 가속도 0
#endif
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
// 절 CM2 시작점 섞기: η = (1 − β) η_U + β η_ŷ (β = 2^-kk[0]), y = ŷ + T η (kk[0] = 0 이면 η_ŷ 그대로 -> y = 관성 예측 ŷ 와 비트 같음)
__global__ void rt_y0mix(const int* IP, const int* kk, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eS, float* eta, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (pm[n] == 2) return;
    float be = ldexpf(1.f, -kk[0]);
    float e[3]; for (int a=0;a<3;a++) { e[a] = (1.f - be)*eta[n*3+a] + be*eS[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 절 CM2 재질 판 (선탐색 없음) 장벽 걸음: 온 걸음 후보 y1 = ŷ + T (η + δ)
__global__ void rt_ytry(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl, float* y1) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + dl[n*3+a];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y1[n*3+a] = yh[n*3+a] + t[a];
}
// 보폭 SC[14] = 2^-kk[0] (rt_lsup full = 0 이 씀)
__global__ void rt_kstep(const int* kk, double* SC) { SC[14] = (double)ldexpf(1.f, -kk[0]); }
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

# 절 CM9 (--gq_mass, 사용자 승인 11-05 정의 변경): 질량·운동량 (HHT 판은 입자 가속도의 질량 가중 몫도) 을 힘과 같은 가우스 점·같은 가중·같은 기저로 노드에.
# 입자 -> (자기 팔분) 가우스 점: P_q = Σ_{p∈q} m_p (v_p, a_p), 칸 합 P_c = Σ_{p∈c} m_p (v_p, a_p);
# 점 몫 p_q = 점유 가중이면 (1 − κ) P_q + κ P_c / 8, 아니면 P_c / 8 -- rt_gauss 의 m_q·V_q 와 같은 가중 (질량 m_q 는 rt_gauss 값 그대로);
# 노드: m_n = Σ_q m_q w_n(x_q), (mv)_n = Σ_q p_q w_n(x_q) (w_n(x_q) = BW, 힘 ∇w_n(x_q) 와 같은 점 x_q). Σ_n w_n(x_q) = 1 이라 질량·운동량 합 보존.
# 끄면 이 커널들은 불리지 않고 rt_p2g 노드 몫 (AF_GQM 정의 안 됨) 도 그대로 -- 예전과 같은 기계어.
GQ_SRC = r'''
__global__ void gq_p2q(int N, const float* v, const float* m, const float* pa, const float* fx, const int* pcd, const int* cidx, const int* IP, float* PQ, float* PC) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    float mp = m[p];
    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); atomicAdd(&PC[(size_t)c*6+a], t); }
    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); atomicAdd(&PC[(size_t)c*6+3+a], t); }
}
// 결정적 판: 정렬된 입자의 칸 구간 (st) 을 칸 하나가 차례로
__global__ void gq_p2q_d(const int* IP, const int* st, const float* v, const float* m, const float* pa, const float* fx, float* PQ, float* PC) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int p = st[c]; p < st[c+1]; p++) {
        int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
        int q = c*8 + oc;
        float mp = m[p];
        for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; PQ[(size_t)q*6+a] += t; PC[(size_t)c*6+a] += t; }
        if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; PQ[(size_t)q*6+3+a] += t; PC[(size_t)c*6+3+a] += t; }
    }
}
__device__ __forceinline__ void gq_pq(int q, const float* PQ, const float* PC, float kap, int oct, int na, float* pq) {
    int c = q >> 3;
    for (int a=0;a<na;a++) { float pc = PC[(size_t)c*6+a]*0.125f; pq[a] = oct ? ((1.f - kap)*PQ[(size_t)q*6+a] + kap*pc) : pc; }
}
// 점 -> 노드 (원자 합)
__global__ void gq_q2n(const int* IP, const int* cb, const float* BW, const float* mq, const float* PQ, const float* PC, float kap, int oct, float* mn, float* mv, float* ma) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7;
    float pq[6]; gq_pq(q, PQ, PC, kap, oct, ma ? 6 : 3, pq);
    float mm = mq[q];
    int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        float w = BW[oc*27+o];
        atomicAdd(&mn[nid], w*mm);
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], w*pq[a]);
        if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[nid*3+a], w*pq[3+a]);
    }
}
'''
SRC = SRC.rstrip()[:-1].rstrip() + "\n" + GQ_SRC + "}\n"

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
#if defined(AF_BCROSS) || defined(AF_BCNT)
// 절 CM10 (--bar_cross, 사용자 승인 정의 변경; --bar_cnt 진단도 같은 판정): 표본점이 속한 B-스플라인 칸이 바닥 평면을 관통하는가.
// 칸 범위 = 코드의 칸 정의 그대로 (rt_coord: 격자 좌표 L = Ai x 에서 base b = floor(L − ½), 칸 = L ∈ [b+½, b+1½)³ -- 입자 배정 범위, 노드 b..b+2 의 27 개).
// 판정 위치 = 걸음 시작 위치 (rt_bar_q mode 0 과 같은 y): 칸 꼭짓점 8 개 (L = b + ½ + {0,1}³) 의 기준 위치 A L 에 같은 이차 B-스플라인 기저로 노드 이동 U + thinv (y − U) 를 더한 점의
// 평면 부호 거리 d 의 최소 < 0 < 최대 이면 관통 칸 (꼭짓점 축별 기저: f = ½ -> (½, ½, 0), f = 1½ -> (0, ½, ½)).
__device__ __forceinline__ bool bar_cross(int qq, const int* IP, const int* cb, const int* ids, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn) {
    int c = qq >> 3; float dmin = 3.0e38f, dmax = -3.0e38f;
    for (int cn=0; cn<8; cn++) {
        int ci[3] = {(cn>>2)&1, (cn>>1)&1, cn&1};
        float L[3], w[3][3];
        for (int a=0;a<3;a++) { L[a] = (float)(IP[a] + cb[c*3+a]) + 0.5f + (float)ci[a];
            w[a][0] = ci[a] ? 0.f : 0.5f; w[a][1] = 0.5f; w[a][2] = ci[a] ? 0.5f : 0.f; }
        float x[3]; for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2];
        for (int o=0;o<27;o++) { float ww = w[0][o/9]*w[1][(o/3)%3]*w[2][o%3]; if (ww == 0.f) continue; int id = ids[o];
            for (int a=0;a<3;a++) x[a] += ww*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
        float d = x[0]*n0 + x[1]*n1 + x[2]*n2 - pn;
        dmin = fminf(dmin, d); dmax = fmaxf(dmax, d);
    }
    return dmin < 0.f && dmax > 0.f;
}
#endif
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
#if defined(AF_BCROSS) || defined(AF_BCNT)
        if (mode == 0) { bool cr_ = bar_cross(qq, IP, cb, ids, y, U, thinv, n0, n1, n2, pn);
#ifdef AF_BCNT
            if (cr_) AF_BCF[qq] |= 2;                                           // 진단: 이 프레임 어느 걸음 시작에서든 관통 칸
#endif
#ifdef AF_BCROSS
            if (!cr_) AF_BACT[qq] = 0;                                          // 절 CM10: 관통 칸이 아닌 칸의 표본점은 이 걸음 장벽에서 뺌 (걸음 시작 d ≤ 0 인 점과 같은 처리)
#endif
        }
#endif
        else if (AF_BACT[qq] && d < dh) {
            double kap = (double)ks*(double)Vq[qq];
#ifdef AF_BCNT
            AF_BCF[qq] |= 1;                                                    // 진단: 이 프레임 어느 뉴턴 반복에서든 장벽이 걸림 (활성이고 d < d̂)
#endif
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
#ifdef AF_BFULL
        if (on) AF_BPH[qq] = h;                                                 // 비덩어리 헤시안 곱 (rt_bar_mvq) 이 쓰는 점 스칼라
#endif
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
// 비덩어리 헤시안 곱 (--bar_full, 절 AN.21): (H Tp)_i += n Σ_q w_qi h_q Σ_j w_qj (n·Tp_j) -- 점 스칼라 t_q = h_q Σ_j w_qj (n·Tp_j) 를 노드 스칼라 bS 로 모은 뒤 rt_bar_mvn 이 n 방향으로 더함
// (덩어리 판 rt_bar_mv 는 diag(Σ_q h_q w_q) 라 상대 변위 몫까지 강성을 줌; 블록 대각 전처리 B 에는 덩어리 값을 그대로 씀 -- 전처리만)
__global__ void __launch_bounds__(128) rt_bar_mvq(const int* IP, const int* cb, const float* BW, float n0, float n1, float n2, const float* Tp, float* bS, float* tq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float h = AF_BPH[q];
    if (h == 0.f) { if (AF_BDET) tq[q] = 0.f; return; }
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float t = 0.f;
    for (int o=0;o<27;o++) { int id = ids[o]; t += BW[oc*27+o]*(Tp[id*3]*n0 + Tp[id*3+1]*n1 + Tp[id*3+2]*n2); }
    t *= h;
    if (AF_BDET) { tq[q] = t; return; }
    for (int o=0;o<27;o++) atomicAdd(&bS[ids[o]], BW[oc*27+o]*t);
}
__global__ void rt_bar_mvn(const int* IP, const float* bS, float n0, float n1, float n2, float* KTp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t = bS[n]; if (t == 0.f) return;
    KTp[n*3] += t*n0; KTp[n*3+1] += t*n1; KTp[n*3+2] += t*n2;
}
// 선탐색 후보 NC 개의 장벽 에너지 -> SC[40+i] (rt_elem_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i)
__global__ void __launch_bounds__(128) rt_bar_lsN(const int* IP, const int* cb, const float* BW, const float* Vq, const float* yh, const float* eta, const float* dl,
                         const float* sk, const int* pm, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;                                           // 절 AN.22: 후보 c = 1 이 받아들여졌으면 나머지 후보 생략
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    int ids[27]; qnodes(qq, IP, cb, ids);
    float x0[3] = {0.f, 0.f, 0.f};
    if (act) for (int a=0;a<3;a++) x0[a] = AF_BX[qq*4+a]/bm;
    double kap = act ? (double)ks*(double)Vq[qq] : 0.0;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
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
// 절 AN.22 (--mgbar 2, 전처리만): 거친 층 장벽 덩어리 헤시안. Kc_l = R Kc_{l−1} (제한 = 연장의 전치, 1 의 분할이라 Pᵀ diag(K) P 의 행 합 덩어리와 같음 -- 거친 질량 mc = R m 과 같은 꼴),
// 곱 KT += Kc n (n·z), 블록 대각 역 D_b = (D⁻¹ + Kc n nᵀ)⁻¹ = D − Kc (D n)(D n)ᵀ / (1 + Kc nᵀ D n) (셔먼-모리슨, D 대칭)
__global__ void rt_bar_mvl(int l, const int* LM, const float* Kc, float n0, float n1, float n2, const float* z, float* KT) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float K = Kc[n]; if (K == 0.f) return;
    float t = K*(z[n*3]*n0 + z[n*3+1]*n1 + z[n*3+2]*n2);
    KT[n*3] += t*n0; KT[n*3+1] += t*n1; KT[n*3+2] += t*n2;
}
// 절 CM2 (정의 변경 셋째 --y0ut, 둘째 재질 판 장벽): 장벽 실현 가능 보폭. 활성 표본점 (AF_BACT) 의 d 는 y 에 선형이라 y0 -> y1 의 β = 2^-k 에서 d(β) = d0 + β (d1 − d0);
// d(β) > 0 인 가장 작은 k 를 점마다 구해 최대를 kk 에 (정수 원자 최대 -- 순서 무관). k 상한 없음 (β 가 f32 에서 0 이 되면 멈춤 -- 표현 한계).
__global__ void __launch_bounds__(128) rt_bar_kfe(const int* IP, const int* cb, const float* BW, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, int* kk) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3], xb[3];
    for (int a=0;a<3;a++) { xa[a] = AF_BX[q*4+a]/bm; xb[a] = xa[a]; }
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a])); xb[a] += w*(U[id*3+a] + thinv*(y1[id*3+a] - U[id*3+a])); } }
    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
    if (d1 > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; }
    atomicMax(kk, k);
}
// 절 CM8 (B, --y0eb, 정의 변경): 시작점 섞기 β 를 장벽 에너지 기준으로 -- "장벽 에너지가 걸음 시작 위치 (β = 0, 시작점 U) 의 장벽 에너지보다 크지 않은 가장 큰 2^-k".
// 후보 β_i = 2^-(k0+i) (k0 = rt_bar_kfe 의 실현 가능 k, i = 0..NE−1) 의 장벽 에너지 -> SC[1+i], β = 0 -> SC[0]. 활성 표본점 (U 에서 판정, AF_BACT) 의 d 는 y 에 선형
// (y(β) = (1−β) y_U + β y_ŷ) 이라 두 끝 d0 (U)·d1 (ŷ) 만 모으고 후보마다 d = d0 + β (d1 − d0) (rt_bar_kfe 와 같은 식). 장벽 에너지는 β 에 볼록이라 조건을 만족하는 β 는 [0, β*] 구간.
__global__ void __launch_bounds__(128) rt_bar_y0e(const int* IP, const int* cb, const float* BW, const float* Vq, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, const int* kk, double* SC, int NE) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    float d0 = 0.f, d1 = 0.f; double kap = 0.0;
    if (act) {
        int ids[27]; qnodes(qq, IP, cb, ids);
        float xa[3], xb[3];
        for (int a=0;a<3;a++) { xa[a] = AF_BX[qq*4+a]/bm; xb[a] = xa[a]; }
        for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
            for (int a=0;a<3;a++) { xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a])); xb[a] += w*(U[id*3+a] + thinv*(y1[id*3+a] - U[id*3+a])); } }
        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
        kap = (double)ks*(double)Vq[qq];
    }
    double e = 0.0;
    if (act && d0 < dh) { if (d0 > 0.f) { double dd = (double)d0 - dh; e = -kap*dd*dd*log((double)d0/dh); } else e = 1e30; }
    blocksum_add(e, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int i=0;i<NE;i++) {
        e = 0.0;
        if (act) { float d = d0 + c*(d1 - d0);
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; } }
        blocksum_add(e, &SC[1+i]);
        c *= 0.5f;
    }
}
// 후보 중 E(β_i) ≤ E(0) 인 첫 i (가장 큰 β) -> kk = k0 + i; 없으면 β = 0 (kk 1000: ldexpf(1, −1000) = 0 -> rt_y0mix 가 η_U 그대로 = 시작점 U)
__global__ void rt_y0epick(double* SC, int* kk, int NE) {
    double E0 = SC[0]; int k0 = kk[0];
    for (int i=0;i<NE;i++) { double E1 = SC[1+i]; if (isfinite(E1) && E1 <= E0) { kk[0] = k0 + i; return; } }
    kk[0] = 1000;
}
__global__ void rt_mg_dbar(int l, const int* LM, const float* D, const float* Kc, float n0, float n1, float n2, float* Db) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    const float* d = D + n*9; float K = Kc[n];
    float u[3]; for (int a=0;a<3;a++) u[a] = d[a*3]*n0 + d[a*3+1]*n1 + d[a*3+2]*n2;
    float den = 1.f + K*(u[0]*n0 + u[1]*n1 + u[2]*n2);
    float c = (K != 0.f && den > 0.f) ? K/den : 0.f;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Db[n*9+a*3+b] = d[a*3+b] - c*u[a]*u[b];
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


def bstify(src, mode, dh, full):
    """절 AN.21 (장벽 판 위): mode 1·2 = 띠 고착 (AF_BST; d < d̂ 표본점 고착 몫 1, 띠 밖 1 = 0 / 2 = 정답 sticky 가중), full = 비덩어리 헤시안 곱 (AF_BFULL)."""
    add = ""
    if mode:
        add += "\n#define AF_BST %d\n#define AF_BDHC %.9ef" % (int(mode), float(dh))
    if full:
        add += "\n#define AF_BFULL 1"
    return src.replace('extern "C" {', 'extern "C" {' + add, 1) if add else src


BARN_SRC = r'''
// ---- 절 CM14 노드 장벽 (--bar_node add|only, 사용자 승인 11-05 정의 변경): B-스플라인 노드 위치의 바닥 위 높이 d_n 에 같은 IPC 거리 장벽 ----
// 노드 위치 x_n = A L_n + U_n + thinv (y_n − U_n) (L_n = 노드 격자 좌표 IP[0..2] + 상자 번호, 표본점 이동과 같은 걸음 끝 식), d_n = (x_n − p)·n.
// b(d) = −κ_n (d − d̂_n)² ln(d/d̂_n) (0 < d < d̂_n), κ_n = ks_n V_n, ks_n = (λ + 2μ)/d̂_n², V_n = Σ_q w_qn V_q (노드에 모이는 부피, Σ_n V_n = Σ_q V_q).
// 노드 하나의 변위에만 걸리므로 기울기 c_n n (c_n = b' thinv)·헤시안 h_n n nᵀ (h_n = b'' thinv²) 은 노드 대각 (정확) -> 표본점 장벽의 노드 스칼라 bGn·bKn 에 더함.
// 걸음 시작 d_n ≤ 0 (또는 V_n = 0) 인 노드는 그 걸음 장벽에서 뺌 (AF_BNACT 0; 표본점 규칙과 같음). AF_BNF: 진단 비트 1 = 이 프레임 장벽 걸림 (활성이고 d < d̂_n), 2 = 어느 걸음 시작에서 d ≤ 0.
__device__ __forceinline__ float barn_d(int n, const int* IP, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn) {
    int nb1 = IP[7], nb2 = IP[8];
    float L[3] = {(float)(IP[0] + n/(nb1*nb2)), (float)(IP[1] + (n/nb2)%nb1), (float)(IP[2] + n%nb2)};
    float x[3];
    for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2] + U[n*3+a] + thinv*(y[n*3+a] - U[n*3+a]);
    return x[0]*n0 + x[1]*n1 + x[2]*n2 - pn;
}
__device__ __forceinline__ double barn_e(float d, float dh, double kap) {
    if (!(d < dh)) return 0.0;
    if (!(d > 0.f)) return 1e30;
    double dd = (double)d - dh; return -kap*dd*dd*log((double)d/dh);
}
// 노드 부피 (비결정 판; 결정적 판은 ao_gath_q 고정 순서)
__global__ void __launch_bounds__(128) rt_barn_vq(const int* IP, const int* cb, const float* BW, const float* Vq, float* Vn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float v = Vq[q]; if (!(v > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    for (int o=0;o<27;o++) atomicAdd(&Vn[ids[o]], BW[oc*27+o]*v);
}
// mode 0: 걸음 시작 판정; mode 1: Gn += c, Kn += h, 에너지 -> E
__global__ void rt_barn_q(const int* IP, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, int mode, float* Gn, float* Kn, double* E) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9];
    float V = on ? AF_BNV[n] : 0.f;
    double e = 0.0;
    if (V > 0.f) {
        float d = barn_d(n, IP, y, U, thinv, n0, n1, n2, pn);
        if (mode == 0) { AF_BNACT[n] = (d > 0.f) ? 1 : 0; if (!(d > 0.f)) AF_BNF[n] |= 2; }
        else if (AF_BNACT[n] && d < dh) {
            AF_BNF[n] |= 1;
            double kap = (double)ks*(double)V;
            if (d > 0.f) {
                double dd = (double)d - dh, L = log((double)d/dh);
                e = -kap*dd*dd*L;
                Gn[n] += (float)(-kap*(2.0*dd*L + dd*dd/d))*thinv;
                Kn[n] += (float)(-kap*(2.0*L + 4.0*dd/d - dd*dd/((double)d*d)))*thinv*thinv;
            } else e = 1e30;
        }
    } else if (on && mode == 0) AF_BNACT[n] = 0;
    if (mode == 1) blocksum_add(e, E);
}
// 선탐색 후보 NC 개 (rt_bar_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i) 의 노드 장벽 에너지 -> SC[40+i]
__global__ void rt_barn_lsN(const int* IP, const float* yh, const float* eta, const float* dl, const float* sk, const int* pm, const float* U, float thinv,
                            float n0, float n1, float n2, float pn, float dh, float ks, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9]; int nn = on ? n : 0;
    float V = on ? AF_BNV[nn] : 0.f;
    bool act = V > 0.f && AF_BNACT[nn];
    double kap = act ? (double)ks*(double)V : 0.0;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
        double e = 0.0;
        if (act) {
            float ee[3]; for (int a=0;a<3;a++) ee[a] = eta[nn*3+a] + c*dl[nn*3+a];
            float t[3]; Tapply(sk[nn], pm[nn], n0, n1, n2, ee, t);
            float yy[3]; for (int a=0;a<3;a++) yy[a] = yh[nn*3+a] + t[a];
            int nb1 = IP[7], nb2 = IP[8];
            float L[3] = {(float)(IP[0] + nn/(nb1*nb2)), (float)(IP[1] + (nn/nb2)%nb1), (float)(IP[2] + nn%nb2)};
            float x[3]; for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2] + U[nn*3+a] + thinv*(yy[a] - U[nn*3+a]);
            e = barn_e(x[0]*n0 + x[1]*n1 + x[2]*n2 - pn, dh, kap);
        }
        blocksum_add(e, &SC[40+i]);
        c *= 0.5f;
    }
}
// 실현 가능 보폭 (rt_bar_kfe 와 같은 식, 노드): y0 -> y1 의 β = 2^-k 에서 활성 노드 모두 d > 0 인 가장 작은 k -> kk (정수 원자 최대)
__global__ void rt_barn_kfe(const int* IP, const float* y0, const float* y1, const float* U, float thinv, float n0, float n1, float n2, float pn, int* kk) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (!(AF_BNV[n] > 0.f) || !AF_BNACT[n]) return;
    float d0 = barn_d(n, IP, y0, U, thinv, n0, n1, n2, pn), d1 = barn_d(n, IP, y1, U, thinv, n0, n1, n2, pn);
    if (d1 > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; }
    atomicMax(kk, k);
}
// 시작점 섞기 β (--y0eb, rt_bar_y0e 와 같은 식, 노드): β = 0 -> SC[0], 후보 β_i = 2^-(k0+i) -> SC[1+i]
__global__ void rt_barn_y0e(const int* IP, const float* y0, const float* y1, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, const int* kk, double* SC, int NE) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9]; int nn = on ? n : 0;
    float V = on ? AF_BNV[nn] : 0.f;
    bool act = V > 0.f && AF_BNACT[nn];
    float d0 = 0.f, d1 = 0.f; double kap = 0.0;
    if (act) { d0 = barn_d(nn, IP, y0, U, thinv, n0, n1, n2, pn); d1 = barn_d(nn, IP, y1, U, thinv, n0, n1, n2, pn); kap = (double)ks*(double)V; }
    blocksum_add(act ? barn_e(d0, dh, kap) : 0.0, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int i=0;i<NE;i++) {
        blocksum_add(act ? barn_e(d0 + c*(d1 - d0), dh, kap) : 0.0, &SC[1+i]);
        c *= 0.5f;
    }
}
'''


def bnodeify(src, mode):
    """절 CM14 (장벽 판 위, --bar_node add|only): 노드 위치 장벽 (AF_BNODE; 노드 부피·활성·진단 버퍼). only = 표본점 장벽을 끔 (걸음 시작 판정에서 모든 표본점 비활성 -> 기울기·헤시안·선탐색·실현 가능 보폭·시작점 섞기에서 빠짐; 띠 고착은 그대로). 끄면 소스 그대로."""
    if not mode:
        return src
    add = "\n#define AF_BNODE 1\n__device__ float* AF_BNV;\n__device__ int* AF_BNACT;\n__device__ int* AF_BNF;"
    if "__device__ const float* AF_BAM;" not in src:
        add += "\n__device__ const float* AF_BAM;"
    src = src.replace('extern "C" {', 'extern "C" {' + add, 1)
    if mode == "only":
        old = "if (mode == 0) AF_BACT[qq] = (d > 0.f) ? 1 : 0;"
        assert src.count(old) == 1
        src = src.replace(old, "if (mode == 0) AF_BACT[qq] = 0;                                             // 절 CM14 (나) 노드 장벽만: 표본점 장벽 끔", 1)
    i = src.rstrip().rfind("}")
    return src[:i] + BARN_SRC + "\n}\n"


def fstkify(src, on):
    """절 CM12 (장벽 판 위, --fl_stk): (A) 가우스 점 고착 (CM8 a 판 정의) 을 장벽 바닥과 함께 (AF_FSTK; (A) 노드 몫 버퍼 AF_ANA). 끄면 소스 그대로."""
    return src.replace('extern "C" {', 'extern "C" {\n#define AF_FSTK 1\n__device__ float* AF_ANA;', 1) if on else src


def bcrossify(src, cross, cnt):
    """절 CM10 (장벽 판 위): cross = 관통 칸 표본점에만 장벽 (AF_BCROSS, --bar_cross), cnt = 진단 표본점 플래그 (AF_BCNT, --bar_cnt; 결과 불변). 둘 다 끄면 소스 그대로."""
    add = ""
    if cross:
        add += "\n#define AF_BCROSS 1"
    if cnt:
        add += "\n#define AF_BCNT 1\n__device__ int* AF_BCF;"
    if cross or cnt:
        add += "\n__device__ const float* AF_BAM;"
    return src.replace('extern "C" {', 'extern "C" {' + add, 1) if add else src


# 절 CM3 노드 압축 판 커널 (--ncomp): 원본 본문에서 머리만 바꿔 이름 뒤 _c, 마지막 인자 ACT (목록). LM 커널은 층 0 에서만 _c 를 부른다 (RTSolver.k).
HB_ = "    if (blockIdx.x*blockDim.x >= IP[9]) return;\n    int n = blockIdx.x*blockDim.x + threadIdx.x;\n"
HS_ = "    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= IP[9]) return;\n"
CN_SUB = (("rt_ap", HB_, "    if (blockIdx.x*blockDim.x >= ACT[0]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < ACT[0]) ? ACT[1+ii_] : IP[9];\n"),
          ("rt_dotz", HB_, "    if (blockIdx.x*blockDim.x >= ACT[0]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < ACT[0]) ? ACT[1+ii_] : IP[9];\n"),
          ("rt_cg1m", HS_, "    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n"),
          ("rt_cg2", HS_, "    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n"),
          ("rt_pinit", HS_, "    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n"),
          ("rt_mg_node", "    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= LM[l*8+3]) return;\n", "    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= ACT[0]) return;\n    int n = ACT[1+ii_];\n"),
          ("rt_mg_pw1", "    int nn = LM[l*8+3];\n    if (blockIdx.x*blockDim.x >= nn) return;\n    if (n < nn) {\n", "    int nn = ACT[0];\n    if (blockIdx.x*blockDim.x >= nn) return;\n    if (n < nn) { n = ACT[1+n];\n"),
          ("rt_mg_pw2", "    if (n >= LM[l*8+3]) return;\n", "    if (n >= ACT[0]) return;\n    n = ACT[1+n];\n"),
          ("rt_mg_prolong", "    if (I >= A[3]) return;\n", "    if (I >= ACT[0]) return;\n    I = ACT[1+I];\n"))
CN_ALL = frozenset(q[0] for q in CN_SUB)
CN_LM = frozenset(("rt_mg_node", "rt_mg_pw1", "rt_mg_pw2", "rt_mg_prolong"))


def compactify(src):
    out = []
    for nm, old, new in CN_SUB:
        i = src.index("__global__ void " + nm + "(")
        j = src.index("\n}\n", i) + 3
        body = src[i:j].replace("void " + nm + "(", "void " + nm + "_c(", 1)
        k = body.index(") {")
        body = body[:k] + ", const int* ACT" + body[k:]
        assert body.count(old) == 1, nm
        out.append(body.replace(old, new, 1))
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n".join(out) + "\n}\n"


def build_src(src, adapt, goct, floor=None, fk=False, hspec=False, ncomp=False, gqm=False):
    if gqm:                                                                            # 절 CM9 --gq_mass: rt_p2g 노드 몫 빼기 (노드 질량·운동량은 gq_q2n 이 가우스 점에서)
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_GQM 1', 1)
    if ncomp:
        src = compactify(src)
    if os.environ.get("AF_MGDIMS_OLD", "0") == "1":                                   # 절 CM7 진단 대조 전용: 넘친 프레임 MG 층 차원 옛 판 (불법 접근 재현)
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_MGDIMS_OLD 1', 1)
    if hspec:
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_HSPEC 1', 1)
    if fk:
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_FK 1', 1)
    if goct is not None and goct >= 0:
        src = octify(src)
    if floor and floor.get("pt"):
        src = flptify(src, floor["KEX"])
    if floor and floor.get("bar"):
        src = barify(src)
        src = bstify(src, floor.get("bst", 0), floor["DH"], floor.get("bfull", False))
        src = bcrossify(src, floor.get("bcross", False), floor.get("bcnt", False))
        src = fstkify(src, floor.get("stk", False))
        src = bnodeify(src, floor.get("bnode", ""))
    return skipify(src) if adapt else src


def _force_redo_frames():
    """절 CM7 진단: AF_FORCE_REDO="5,20,35" -> {5, 20, 35} (프레임 번호 0 부터; 그 프레임에서 넘치지 않아도 다시 할당 경로)."""
    v = os.environ.get("AF_FORCE_REDO", "")
    return {int(q) for q in v.split(",") if q.strip()}


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


class RTSolver(MG2.MG2Mixin):
    """고정 용량 버퍼 + 프레임 CUDA 그래프. step(): 그래프 재생 한 번 = 한 프레임 (P2G -> K 걸음 x 뉴턴 n x PCG c -> G2P), 입자 상태 x·v·F 제자리 갱신."""

    def __init__(s, x, v, F, m, vol, A, Ai, mu, lam, g, frame_dt, K, integ, newton, cg, ls, floor=None, damp=1.0, bc=None, cap_mul=2.0, dev="cuda", mg_levels=1, mg_nc=8, mg_pow=6, mg_omega=1.0, descent=False, ls_n=0, scheme=None, theta=0.5, mfloor=0.0, K_contact=None, gfin=False, fuse2=True, mg_wreuse=False, mg_lreuse=False, mg_csolve=False, ga=None, rho=0.7, areset=True, tolN=-1.0, tolP=-1.0, goct=-1.0, pk=False, pk_loc=2, pkb=256, pkminb=1, mv2=False, mg_occ=False, mg_fuse=0, y0ls=False, lsfast=False, mgbar=False, cg_list=None, fk=False, y0ut=False, hspec=False, ncomp=False, y0eb=False, gqm=False):
        s.dev = dev
        s.gqm = bool(gqm)                                                              # 절 CM9 (정의 변경): 노드 질량·운동량을 가우스 점에서 (힘과 같은 점·가중·기저)
        assert not (s.gqm and pk), "--gq_mass 는 그래프 판만"
        s.hspec = bool(hspec)                                                          # 절 CM3 (결과 정의 같음): 고운 층 헤시안 주값 분해 30 성분 저장·곱 (rt_mv2s; --mv2 필요)
        s.ncomp = bool(ncomp)                                                          # 절 CM3 (결과 정의 같음): PCG·층 0 V 순환 노드 커널을 점유 칸 이웃 노드 목록만
        assert not s.hspec or bool(mv2) or bool(mg_occ), "--hspec 은 --mv2 판만"
        assert not (s.ncomp and pk), "--ncomp 는 그래프 판만"
        assert not (s.hspec and (mg_fuse or mg_csolve)), "--hspec 은 거친 층 한 블록 판 (mg_fuse·mg_csolve) 과 함께 못 씀 (81 성분 읽음)"
        s.y0eb = bool(y0eb)                                                            # 절 CM8 (B): 장벽 판 시작점 섞기 β = 장벽 에너지 ≤ 걸음 시작의 장벽 에너지인 가장 큰 2^-k
        s.y0ut = bool(y0ut)                                                            # 절 CM2 정의 변경 셋째: 뉴턴 시작점 = 관성 예측 ŷ (장벽 판은 걸음 시작 U 에서 ŷ 쪽 실현 가능한 최대 2^-k)
        s.matk = None                                                                  # 절 CM2: 재질 판 (lib/anchorflow/fem_mat_rt.py MatRT 가 채움)
        s.lsfast = bool(lsfast)                                                        # 절 AN.22: 선탐색 조기 판정 (결과 같음)
        s.cg_list = [int(q) for q in cg_list.split(",")] if cg_list else None          # 절 AN.22: 뉴턴 반복별 PCG 상한 (길이 = newton)
        assert s.cg_list is None or len(s.cg_list) == int(newton), "--cg_list 길이는 --newton"
        s.fk = bool(fk)                                                                # 절 AN.22: 같은 식·반올림만 다른 커널 가속 (관성 후보 이차식, 거듭제곱법 나눗셈)
        s.mgbar = int(mgbar)                                                          # 절 AN.22: MG 전처리 고운 층 연산자에 장벽 헤시안
        s.y0ls = bool(y0ls)                                                            # 절 AN.22: 뉴턴 시작점을 U 와 관성 예측 Ũ 사이 후보 중 전체 에너지 최소로 (ga·장벽 판, --ls_n 필요)
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
        s.mod = cp.RawModule(code=build_src(s._src0, s.adapt, s.goct, s.floor, fk=s.fk, hspec=s.hspec, ncomp=s.ncomp, gqm=s.gqm), options=("--std=c++14",))
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
        if s.ncomp and name in CN_ALL and (name not in CN_LM or int(args[0]) == 0):  # 절 CM3: 노드 압축 판 (LM 커널은 층 0 만)
            name = name + "_c"
            args = tuple(args) + (P(s.actl),)
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
        # 절 CM7: --mg_lreuse 거친 층 ω (W[1..]) 는 첫 프레임 값 -- 다시 할당이 지우고 그 프레임에서 다시 재면 용량에 따라 결과가 달라진다 -> 옛 값 유지
        keepW = (s.mg_lreuse and getattr(s, "_wdone", False) and getattr(s, "W", None) is not None and os.environ.get("AF_REDO_KEEPW", "1") == "1")
        oldW = s.W.clone() if keepW else None
        s.capN, s.capC, s.capQc = [int(q * s.cap_mul) + 1024 for q in need]
        capQ = s.capQc * 8
        Z = lambda *sh: torch.zeros(*sh, device=dev, dtype=f32)
        s.occ = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cidx = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.cb = torch.zeros(s.capQc, 3, device=dev, dtype=torch.int32)
        s.Aacc = Z(capQ, 28); s.Cacc = Z(s.capQc, 30)
        s.badq = torch.zeros(capQ, device=dev, dtype=torch.int32); s.badc = torch.zeros(s.capQc, device=dev, dtype=torch.int32)
        s.Mo = Z(capQ) if s.goct >= 0 else None
        s.gqP = Z(capQ * 6) if getattr(s, "gqm", False) else None                     # 절 CM9: 팔분 점·칸 질량 가중 속도·가속도 합
        s.gqC = Z(s.capQc * 6) if getattr(s, "gqm", False) else None
        s.bar = bool(s.floor and s.floor.get("bar"))                                  # 장벽 판 (절 AN.19): 표본점 누적·점별 스칼라·노드 스칼라
        if s.bar:
            s.bx = Z(capQ * 4); s.bpg = Z(capQ); s.bph = Z(capQ); s.bact = torch.zeros(capQ, device=dev, dtype=torch.int32); s.bGn = Z(s.capN); s.bKn = Z(s.capN)
            s.bfull = bool(s.floor.get("bfull", False))                               # 비덩어리 헤시안 곱 (절 AN.21)
            s.bcross = bool(s.floor.get("bcross", False)); s.bcnt = bool(s.floor.get("bcnt", False))   # 절 CM10: 관통 칸 장벽 / 진단 표본점 플래그
            if s.bcnt:
                s.bcf = torch.zeros(capQ, device=dev, dtype=torch.int32)
            s.fstk = bool(s.floor.get("stk", False))                                   # 절 CM12: (A) 가우스 점 고착 함께 -- (A) 노드 몫 버퍼
            if s.fstk:
                s.ana = Z(s.capN)
            if s.bfull:
                s.bS = Z(s.capN); s.btq = Z(capQ)
            s.bnode = s.floor.get("bnode", "")                                       # 절 CM14: 노드 위치 장벽 (add = 표본점 장벽에 더함, only = 노드 장벽만)
            if s.bnode:
                assert not s.bfull, "--bar_node 는 덩어리 장벽 헤시안 판만 (노드 몫은 bKn 에 더함)"
                s.bnv = Z(s.capN); s.bnact = torch.zeros(s.capN, device=dev, dtype=torch.int32); s.bnf = torch.zeros(s.capN, device=dev, dtype=torch.int32)
        s.Fq = Z(capQ, 9); s.Vq = Z(capQ); s.mq = Z(capQ); s.xq0 = Z(capQ, 3); s.H = Z(capQ, 81)
        if s.ncomp:                                                                    # 절 CM3: 노드 목록 (ACT[0] = 수), 블록별 수·자리
            nbk = (s.capN + 255) // 256 + 1
            s.actl = torch.zeros(s.capN + 1, device=dev, dtype=torch.int32)
            s.actb = torch.zeros(nbk, device=dev, dtype=torch.int32); s.acto = torch.zeros(nbk, device=dev, dtype=torch.int32)
            nbc = (s.capC + 255) // 256 + 1
            s.scb = torch.zeros(nbc, device=dev, dtype=torch.int32); s.sco = torch.zeros(nbc, device=dev, dtype=torch.int32)
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
        s.eS = Z(s.capN, 3); s.ySv = Z(s.capN, 3); s.kk = torch.zeros(4, device=dev, dtype=torch.int32)        # 절 CM2 시작점·장벽 걸음
        if not hasattr(s, "pa"):
            s.pa = torch.zeros(s.N, 3, device=dev, dtype=torch.float32)
        s.meff = Z(s.capN); s.U0 = Z(s.capN, 3); s.v0 = Z(s.capN, 3); s.wb = Z(s.capN, 3); s.wvb = Z(s.capN, 3)
        s.DG = torch.zeros(4 * max(s.K, s.Kc or 0) * s.nt, 64, device=dev, dtype=torch.float64)
        s.Y0K = torch.zeros(4 * max(s.K, s.Kc or 0), 4, device=dev, dtype=torch.int32)          # 절 CM2 진단: 걸음별 시작점 k (β = 2^-k)
        s.BSK = torch.zeros(4 * max(s.K, s.Kc or 0) * s.nt, 4, device=dev, dtype=torch.int32)   # 절 CM6 진단: 선탐색 없는 장벽 판 뉴턴 반복별 보폭 k (보폭 2^-k; 결과 불변)
        s.Y0S = torch.zeros(4 * max(s.K, s.Kc or 0), 64, device=dev, dtype=torch.float64)       # 절 AN.22 시작점 탐색 진단 (걸음마다 SC: 14 = 고른 c, 21+i·40+i = 후보 에너지)
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
                         r=Z(cn, 3), z=Z(cn, 3), t=Z(cn, 3), KT=Z(cn, 3), y=Z(cn, 3), Kb=Z(cn), Db=Z(cn, 9))
                s.ML.append(L)
            s.capn_d = torch.tensor(capn, device=dev, dtype=torch.int32); s.capq_d = torch.tensor(capq, device=dev, dtype=torch.int32)
            s.LM = torch.zeros(s.nl * 8, device=dev, dtype=torch.int32)
            s.W = torch.zeros(s.nl, device=dev, dtype=torch.float64); s.SCM = torch.zeros(s.nl * 4, device=dev, dtype=torch.float64)
            if oldW is not None and oldW.numel() == s.W.numel():
                s.W.copy_(oldW)
                s._wdone = True
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
                if s.bcross or s.bcnt:                                                 # 절 CM10: 격자 행렬 A (x = A L, 칸 꼭짓점 기준 위치), 진단 플래그 버퍼
                    g = mod.get_global("AF_BAM"); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.A.data_ptr())
                if s.bcnt:
                    g = mod.get_global("AF_BCF"); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.bcf.data_ptr())
                if getattr(s, "fstk", False):                                           # 절 CM12: (A) 가우스 점 고착 노드 몫
                    g = mod.get_global("AF_ANA"); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s.ana.data_ptr())
                if getattr(s, "bnode", ""):                                             # 절 CM14: 노드 장벽 버퍼·격자 행렬 A
                    for nm, t in (("AF_BNV", s.bnv), ("AF_BNACT", s.bnact), ("AF_BNF", s.bnf), ("AF_BAM", s.A)):
                        g = mod.get_global(nm); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(t.data_ptr())
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

    def mg_Az(s, l, z, KT, tz_ready=False, bar=False):
        """KT += K (T z) (층 0) 또는 K z (거친 층). KT 는 앞 소비 커널이 비워 둔다. tz_ready: 층 0 의 T z 가 이미 m0u 에 (융합 노드 연산)."""
        i32, f = np.int32, np.float32
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        if l == 0:
            if not tz_ready:
                s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(3), P(s.mn), f(0), P(s.sk), P(s.pm), *[f(q) for q in nr], P(None), P(s.W), P(None), P(None), P(z), P(s.m0u), P(None)), b=256)
            if s.mv2:
                s.mg2_mv(0, s.m0u, KT)
            else:
                s.k("rt_mv", s.capQc * 8, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.m0u), P(KT), i32(s.capQc * 8)))
            if bar and s.mgbar and getattr(s, "bar", False):                           # 절 AN.22 (거듭제곱법만; V 순환은 노드 연산에 융합) (전처리만, 풀리는 방정식 같음): 고운 층 V 순환 연산자에 장벽 덩어리 헤시안 K n nᵀ (블록 대각 D 에는 원래 들어 있음)
                s.k("rt_bar_mv", s.capN, (P(s.IP), P(s.bKn), *[f(q) for q in nr], P(s.m0u), P(KT)), b=256)
            return
        else:
            if s.mv2:
                s.mg2_mv(l, z, KT)
                if bar and getattr(s, "_mgb_on", False):                               # 절 AN.22 --mgbar 2: 거친 층 장벽 덩어리 헤시안 (거듭제곱법만; V 순환은 노드 연산에 융합)
                    s.k("rt_bar_mvl", s.ML[l - 1]["capn"], (i32(l), P(s.LM), P(s.ML[l - 1]["Kb"]), *[f(q) for q in nr], P(z), P(KT)), b=256)
                return
            L = s.ML[l - 1]
            s.k("rt_mg_mv", L["capq"], (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(z), P(KT)))

    def fmv_name(s):
        return "rt_mv2s" if s.hspec else ("rt_mv2" if s.mv2 else "rt_mv")

    def fmv_H(s):
        """고운 층 곱이 읽는 헤시안 (--hspec: 주값 분해 30 성분 -- 재질 plasticine 판은 H30, 그 밖은 H 앞 30 x capQ)."""
        h = getattr(s, "Hs", None)
        return s.H if h is None else h

    def mg_lv(s, l):
        if l == 0:
            return dict(mn=s.meff, D=s.Dinv, cap=s.capN)
        L = s.ML[l - 1]
        return dict(mn=L["mc"], D=(L["Db"] if getattr(s, "_mgb_on", False) else L["D"]), cap=L["capn"])

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
            s.mg_Az(l, z, KT, bar=True)
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
        KB = None                                                                      # 절 AN.22 --mgbar: 이 층 장벽 덩어리 강성 (노드 연산 융합)
        if getattr(s, "bar", False) and s.mgbar and not getattr(s, "_pkc", None):
            KB = s.bKn if l == 0 else (s.ML[l - 1]["Kb"] if getattr(s, "_mgb_on", False) else None)
        node = lambda mode, rr, kt, zz, rs: s.k("rt_mg_node", V["cap"], (i32(l), P(s.LM), i32(mode), P(V["mn"]), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(V["D"]), P(s.W),
                                                                         P(rr), P(kt), P(zz), P(rs), P(KB)), b=256)
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
                s.k(*(("rt_mg_restrict3", Lc["capn"] * 16) if s.mv2 else ("rt_mg_restrict", Lc["capn"])), (i32(0), P(s.LM), P(s.m0u), P(Lc["r"]), i32(3)), b=256)
            if not (1 >= s.pk_loc_l) and not (s.nl - 1 == 1 and s.mg_csolve):
                s.pk_chain("rt_mg_restrict", "rt_mg_node")                             # 다음 = 거친 층 첫 노드 연산 (같은 거친 노드)
        else:
            s.mg_Az(l, z, KT); node(2, r, KT, z, res)
            with s._pkl(s._pkc is not None and l + 1 >= s.pk_loc_l):
                s.k(*(("rt_mg_restrict3", Lc["capn"] * 16) if s.mv2 else ("rt_mg_restrict", Lc["capn"])), (i32(l), P(s.LM), P(res), P(Lc["r"]), i32(3)), b=256)
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
        s._mgb_on = False
        s.k("rt_reset", 1, (P(IP),), b=1)
        s.ctl(0)
        s.k("rt_coord", N, (i32(N), P(s.x), P(s.Ai), P(s.Np), P(s.fx), P(IP)))
        s.k("rt_box", 1, (P(IP), i32(capN), i32(s.capC)), b=1)
        s.zf(s.occ)
        s.k("rt_mark", N, (i32(N), P(s.Np), P(s.fx), P(IP), P(s.pcd), P(s.occ)))
        if s.ncomp:                                                                    # 절 CM3: 같은 스캔을 격자 3 단계로 (정수, 같은 출력)
            s.k("rt_scan_a", s.capC, (P(IP), P(s.occ), P(s.scb)), b=256)
            s.k("rt_scan_b", 1024, (P(IP), P(s.scb), P(s.sco), i32(s.capQc)), b=1024)
            s.k("rt_scan_c", s.capC, (P(IP), P(s.occ), P(s.sco), P(s.cidx), P(s.cb), i32(s.capQc)), b=256)
        else:
            s.k("rt_scan", 1024, (P(IP), P(s.occ), P(s.cidx), P(s.cb), i32(s.capQc)), b=1024)
        if s.ncomp:                                                                    # 절 CM3: 점유 칸 이웃 노드 목록 (결정적 스캔), 층 0 제한 입력 m0u 의 목록 밖 0
            s.k("rt_act_cnt", capN, (P(IP), P(s.occ), P(s.actb)), b=256)
            s.k("rt_act_scan", 1024, (P(IP), P(s.actb), P(s.acto), P(s.actl)), b=1024)
            s.k("rt_act_wr", capN, (P(IP), P(s.occ), P(s.acto), P(s.actl)), b=256)
            if s.nl > 1:
                s.zf(s.m0u, 9, 3)
        for t_, ci, mul in ((s.Aacc, 12, 28), (s.Cacc, 11, 30), (s.mn, 9, 1), (s.mv, 9, 3), (s.mqn, 9, 1)) + (((s.ma, 9, 3),) if s.ga else ()) + (((s.Mo, 12, 1),) if s.goct >= 0 else ()) + (((s.gqP, 12, 6), (s.gqC, 11, 6)) if s.gqm else ()):
            s.zf(t_, ci, mul)
        s.zf(s.badq); s.zf(s.badc); s.zf(s.nbad)
        if getattr(s, "bar", False):
            s.zf(s.bx)
            if s.bcnt:
                s.zf(s.bcf)                                                            # 절 CM10 진단: 프레임마다 표본점 플래그 비움
            if s.bnode:
                s.zf(s.bnf)                                                            # 절 CM14 진단: 프레임마다 노드 플래그 비움
        if s.matk:
            s.mat_pre_p2g()                                                            # 절 CM2 재질 판: 구속 입자 v·a (순차 방출)
        s.k("rt_p2g", N, (i32(N), P(s.x), P(s.v), P(s.F), P(s.m), P(s.vol), P(s.Np), P(s.fx), P(s.pcd), P(s.cidx), P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.mn), P(s.mv),
                              P(s.pa if s.ga else None), P(s.ma if s.ga else None)))
        if s.gqm:                                                                      # 절 CM9: 입자 -> 팔분 점·칸 운동량 (가속도) 합
            s.k("gq_p2q", N, (i32(N), P(s.v), P(s.m), P(s.pa if s.ga else None), P(s.fx), P(s.pcd), P(s.cidx), P(IP), P(s.gqP), P(s.gqC)))
        s.k("rt_gauss", capQ, (P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.cb), P(s.A), P(s.BW), P(s.Fq), P(s.Vq), P(s.mq), P(s.xq0), P(s.mqn), P(s.nbad)))
        if s.gqm:                                                                      # 절 CM9: 노드 질량·운동량 (가속도) = Σ_q (m_q, p_q) w_n(x_q)
            s.k("gq_q2n", capQ, (P(IP), P(s.cb), P(s.BW), P(s.mq), P(s.gqP), P(s.gqC), f(max(s.goct, 0.0)), i32(1 if s.goct >= 0 else 0), P(s.mn), P(s.mv), P(s.ma if s.ga else None)))
        s.k("rt_vn", capN, (P(IP), P(s.mn), P(s.mv), P(s.vk), P(s.U)), b=256)
        if s.matk:
            s.mat_post_vn()                                                            # 절 CM2 재질 판: 가우스 점 소성 상태, 노드 구속 목표, 구속 질량
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
                    if s.ga_first and not s.matk:                                      # 첫 프레임 (그래프 밖에서만): a0 = g − ∇Ψ(0)/m', m' = max(m, 1e-3 x 질량 중앙값)
                        s.zf(s.f, 9, 3); s.zf(s.y, 9, 3)
                        s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1), P(s.SC[10:]), P(s.f), P(s.H), i32(capQ), P(s.B)))
                        mm = s.mn[:int(s.IP[9])]; mlow = 1e-3 * float(mm[mm > 0].median())
                        s.k("rt_ga_a0", capN, (P(IP), P(s.mn), P(s.f), f(mlow), *[f(q) for q in s.g], P(s.fixd), P(s.ak)), b=256)
                    else:
                        s.k("rt_ga_anode", capN, (P(IP), P(s.mn), P(s.ma), P(s.fixd), P(s.ak)), b=256)
                    if s.matk:
                        s.mat_post_anode()                                             # 절 CM2 재질 판: 질량 0 노드 속도·가속도 이웃 평균
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
            if FL and FL["KEX"] > 0:                                                   # 절 CM6 진단 --fl_nost: KEX 0 이면 고착 몫 0 (노드 고착 인자 1)
                s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), f(sg["thinv"]), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                         f(FL["PN"]), f(FL["DXG"]), P(s.an)))
                if getattr(s, "fstk", False):                                          # 절 CM12: (A) 가우스 점 고착 몫 (dxg < 0 = a 판 판정) -> ana
                    s.zf(s.ana, 9, 1)
                    s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), f(sg["thinv"]), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                             f(FL["PN"]), f(-FL["DXG"]), P(s.ana)))
            elif getattr(s, "fstk", False):
                s.zf(s.ana, 9, 1)
            FA = (P(IP), P(s.an), P(s.mqn), f(FL["KEX"] * (s.K / (K or s.K)) * sg["frac"] if FL else 1.0), i32(1 if FL else 0), i32(1 if (FL and FL["vout"]) else 0), f(sg["al"]), *[f(q) for q in nr],
                  P(s.fixd), P(s.vbc), P(s.U), f(sg["dtf"]), P(s.Ut), P(s.z0), P(s.y), P(s.sk), P(s.pm), P(s.yh), P(s.eta))
            if s.matk:
                s.mat_floor_n(FA, kstep, capN)                                         # 절 CM2 재질 판: 구속 인자 k_구속 곱, 거의 빈 노드 관성 예측 고정
            else:
                s.k("rt_floor_n", capN, FA, b=256)
            y0s = s.y0ls and (sg["kind"] == "ga" or s.bar)
            if y0s:
                s.dcopy(s.dl, s.eta)                                                   # 절 AN.22: η(Ũ) 보관
            y0u_ = (sg["kind"] == "ga" or s.bar) and not (s.y0ut and not s.bar)       # 절 CM2 --y0ut: 장벽 없는 판은 시작점 = ŷ 그대로 (rt_floor_n 결과)
            if y0u_ and s.y0ut:                                                        # 장벽 판: ŷ·η_ŷ 보관 -> U 에서 장벽 활성 판정 -> U 에서 ŷ 쪽 실현 가능한 최대 2^-k
                s.dcopy(s.ySv, s.y); s.dcopy(s.eS, s.eta)
            if y0u_:                                                                   # 장벽 판: 모든 적분기 시작점 = 걸음 시작 위치 (장벽 실현 가능)
                s.k("rt_ga_y0", capN, (P(IP), P(s.U), P(s.yh), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.y)), b=256)
            if s.bar:
                BA = (*[f(q) for q in nr], f(FL["PN"]), f(FL["DH"]), f(FL["KS"]))
                s.k("rt_bar_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg["thinv"]), *BA, i32(0), P(s.bGn), P(s.bKn), P(s.SC[10:])))
                if s.bnode:                                                            # 절 CM14: 노드 부피 V_n = Σ_q w V_q -> 걸음 시작 노드 판정 (같은 y)
                    BN = (*[f(q) for q in nr], f(FL["PN"]), f(FL["DHN"]), f(FL["KSN"]))
                    s.zf(s.bnv, 9, 1)
                    if getattr(s, "det", False):
                        s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.Vq), P(s.bnv)), b=256)
                    else:
                        s.k("rt_barn_vq", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.bnv)))
                    s.k("rt_barn_q", capN, (P(IP), P(s.y), P(s.U), f(sg["thinv"]), *BN, i32(0), P(s.bGn), P(s.bKn), P(s.SC[10:])), b=256)
                if s.y0ut:
                    s.zf(s.kk)
                    s.k("rt_bar_kfe", capQ, (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.ySv), P(s.U), f(sg["thinv"]), *[f(q) for q in nr], f(FL["PN"]), P(s.kk)))
                    if s.bnode:
                        s.k("rt_barn_kfe", capN, (P(IP), P(s.y), P(s.ySv), P(s.U), f(sg["thinv"]), *[f(q) for q in nr], f(FL["PN"]), P(s.kk)), b=256)
                    if s.y0eb:                                                         # 절 CM8 (B): 실현 가능 k 에서 더 줄여 장벽 에너지 ≤ 걸음 시작 (U) 의 장벽 에너지인 가장 큰 2^-k
                        s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
                        s.k("rt_bar_y0e", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.ySv), P(s.U), f(sg["thinv"]), *BA, P(s.kk), P(s.SC), i32(59)))
                        if s.bnode:
                            s.k("rt_barn_y0e", capN, (P(IP), P(s.y), P(s.ySv), P(s.U), f(sg["thinv"]), *BN, P(s.kk), P(s.SC), i32(59)), b=256)
                        s.k("rt_y0epick", 1, (P(s.SC), P(s.kk), i32(59)), b=1)
                    s.dcopy(s.Y0K[kstep], s.kk)                                        # 진단: 걸음별 시작점 k
                    s.k("rt_y0mix", capN, (P(IP), P(s.kk), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eS), P(s.eta), P(s.y)), b=256)
            if y0s:                                                                    # 절 AN.22 시작점 탐색: U 에서 Ũ 쪽 후보 2^-i 중 전체 에너지 최소 (장벽 활성 판정은 위 U 에서 그대로)
                NC0 = i32(s.ls_n)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
                s.k("rt_y0dir", capN, (P(IP), P(s.eta), P(s.dl)), b=256)
                s.k("rt_lsyN", capN, (P(IP), P(s.mn), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), NC0, P(None)), b=256)
                s.k("rt_elem_lsN", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), P(s.SC), NC0, i32(0), P(None)))
                if s.bar:
                    s.k("rt_bar_lsN", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), P(s.U), f(sg["thinv"]), *BA[:4], BA[4], BA[5], P(s.SC), NC0, i32(0), P(None)))
                    if s.bnode:
                        s.k("rt_barn_lsN", capN, (P(IP), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), P(s.U), f(sg["thinv"]), *BN, P(s.SC), NC0, i32(0), P(None)), b=256)
                s.k("rt_y0pick", 1, (P(s.SC), NC0), b=1)
                s.dcopy(s.Y0S[kstep], s.SC)                                            # 진단: 고른 c = SC14
                s.k("rt_lsup", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(0)), b=256)
            s.ctl(1)
            for it in range(s.nt):
                s.ctl(2, it)
                s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
                if s.matk:
                    s.mat_elem(kstep, it, len(ST), capQ)                               # 절 CM2 재질 판 요소 (같은 81 성분 헤시안 꼴 -> rt_mv·rt_mv2·MG 공유)
                else:
                    s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8 | (4 if s.ls else 0)), P(s.SC[10:]), P(s.f), P(s.H),
                                          i32(capQ), P(s.B)))
                if s.bar:
                    s.zf(s.bGn); s.zf(s.bKn)
                    s.k("rt_bar_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg["thinv"]), *BA, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])))
                    if getattr(s, "det", False):
                        s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bpg), P(s.bGn)), b=256)
                        s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bph), P(s.bKn)), b=256)
                    if s.bnode:                                                        # 절 CM14: 노드 장벽 기울기·헤시안 (노드 대각) 을 같은 노드 스칼라에
                        s.k("rt_barn_q", capN, (P(IP), P(s.y), P(s.U), f(sg["thinv"]), *BN, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])), b=256)
                    s.k("rt_bar_n", capN, (P(IP), P(s.bGn), P(s.bKn), *[f(q) for q in nr], P(s.f), P(s.B)), b=256)
                    if s.mgbar >= 2 and s.nl > 1 and s.mv2:                            # 절 AN.22: 거친 층 장벽 (Kc = R Kc_{l−1}, D_b 셔먼-모리슨)
                        for l_ in range(1, s.nl):
                            Lc_ = s.ML[l_ - 1]
                            s.k("rt_mg_restrict3", Lc_["capn"] * 16, (i32(l_ - 1), P(s.LM), P(s.bKn if l_ == 1 else s.ML[l_ - 2]["Kb"]), P(Lc_["Kb"]), i32(1)), b=256)
                            s.k("rt_mg_dbar", Lc_["capn"], (i32(l_), P(s.LM), P(Lc_["D"]), P(Lc_["Kb"]), *[f(q) for q in nr], P(Lc_["Db"])), b=256)
                        s._mgb_on = True
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
                for ci in range(s.cg_list[it] if s.cg_list else s.cg):                   # 절 AN.22 --cg_list: 뉴턴 반복별 PCG 상한 (풀이기 설정만)
                    s.ctl(5, ci)
                    if ci == 0 or not s.fuse2:
                        s.zf(s.KTp, 9, 3)
                    s.k(s.fmv_name(), capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.fmv_H()), P(s.Tp), P(s.KTp), i32(capQ)))
                    if s.bar and s.bfull:
                        s.zf(s.bS, 9, 1)
                        s.k("rt_bar_mvq", capQ, (P(IP), P(s.cb), P(s.BW), *[f(q) for q in nr], P(s.Tp), P(s.bS), P(s.btq)))
                        if getattr(s, "det", False):
                            s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.btq), P(s.bS)), b=256)
                        s.k("rt_bar_mvn", capN, (P(IP), P(s.bS), *[f(q) for q in nr], P(s.KTp)), b=256)
                    elif s.bar and not s.mgbar:
                        s.k("rt_bar_mv", capN, (P(IP), P(s.bKn), *[f(q) for q in nr], P(s.Tp), P(s.KTp)), b=256)
                    s.k("rt_ap", capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.KTp), P(s.Ap), P(s.SC), P(s.bKn if (s.bar and s.mgbar and not s.bfull) else None)), b=256)
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
                    s.lsy(capN, it2, NC, None)
                    if s.descent:
                        s.k("rt_fixdir", capN, (P(IP), P(s.SC), P(s.p0), P(s.dl)), b=256)
                        if s.lsfast:                                                   # 절 AN.22: 방향을 안 바꿨으면 (SC59 = 1) 지우기·다시 계산 생략 (값 같음)
                            G59 = P(s.SC[59:])
                            s.k("rt_sc_clear_g", 1, (P(s.SC), i32(9), i32(10), G59), b=1); s.k("rt_sc_clear_g", 1, (P(s.SC), i32(20), i32(40), G59), b=1)
                            s.lsy(capN, it2, NC, s.SC[59:])
                        else:
                            s.k("rt_sc_clear", 1, (P(s.SC), i32(9), i32(10)), b=1); s.k("rt_sc_clear", 1, (P(s.SC), i32(20), i32(40)), b=1)
                            s.lsy(capN, it2, NC, None)
                    EA = (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), *[f(q) for q in nr], f(s.mu), f(s.lam), P(s.SC), NC)
                    BB = (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), P(s.U), f(sg["thinv"]), *BA[:4], BA[4], BA[5], P(s.SC), NC) if s.bar else None
                    BBN = (P(IP), P(s.yh), P(s.eta), P(s.dl), P(s.sk), P(s.pm), P(s.U), f(sg["thinv"]), *BN, P(s.SC), NC) if (s.bar and s.bnode) else None
                    if s.lsfast:                                                       # 절 AN.22: 후보 c = 1 먼저, 받아들여지면 (SC63 = 1) 나머지 후보 커널 생략 -- 고르는 보폭 같음
                        G63 = P(s.SC[63:])
                        s.k("rt_elem_lsN", capQ, EA[:-1] + (i32(1), i32(0), P(None)))
                        if s.bar:
                            s.k("rt_bar_lsN", capQ, BB[:-1] + (i32(1), i32(0), P(None)))
                        if BBN:
                            s.k("rt_barn_lsN", capN, BBN[:-1] + (i32(1), i32(0), P(None)), b=256)
                        s.k("rt_lsgate", 1, (P(s.SC),), b=1)
                        s.k("rt_elem_lsN", capQ, EA + (i32(1), G63))
                        if s.bar:
                            s.k("rt_bar_lsN", capQ, BB + (i32(1), G63))
                        if BBN:
                            s.k("rt_barn_lsN", capN, BBN + (i32(1), G63), b=256)
                    else:
                        s.k("rt_elem_lsN", capQ, EA + (i32(0), P(None)))
                        if s.bar:
                            s.k("rt_bar_lsN", capQ, BB + (i32(0), P(None)))
                        if BBN:
                            s.k("rt_barn_lsN", capN, BBN + (i32(0), P(None)), b=256)
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
                fbar = (not s.ls) and s.bar                                            # 절 CM2: 선탐색 없는 판 (재질) 의 장벽 -- 보폭 = 활성 표본점 d > 0 인 최대 2^-k
                if fbar:
                    s.k("rt_ytry", capN, (P(IP), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.yc)), b=256)
                    s.zf(s.kk)
                    s.k("rt_bar_kfe", capQ, (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.yc), P(s.U), f(sg["thinv"]), *[f(q) for q in nr], f(FL["PN"]), P(s.kk)))
                    if s.bnode:                                                        # 절 CM14: 노드도 d > 0 유지
                        s.k("rt_barn_kfe", capN, (P(IP), P(s.y), P(s.yc), P(s.U), f(sg["thinv"]), *[f(q) for q in nr], f(FL["PN"]), P(s.kk)), b=256)
                    s.k("rt_kstep", 1, (P(s.kk), P(s.SC)), b=1)
                    s.dcopy(s.BSK[kstep * s.nt + it], s.kk)                           # 절 CM6 진단: 반복별 보폭 k
                s.k("rt_lsup", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(0 if (s.ls or fbar) else 1)), b=256)
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
                assert not s.matk, "재질 판은 ga 만"
                s.k("rt_stepend_g", capN, (P(IP), f(sg["thinv"]), f(sg["al"]), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.fixd), P(s.vbc), dmp), b=256)
            if s.matk:
                s.mat_step_end(kstep, len(ST), capN, capQ)                             # 절 CM2 재질 판: 구속 노드 가속도 0, 걸음 변위 보관, 걸음 사이 소성 커밋
        sl = ST[-1]
        if s.matk:
            s.mat_g2p()                                                                # 절 CM2 재질 판 G2P (입자 F 걸음별 RM, a_p)
            return
        if sl["kind"] == "ga":
            s.k("rt_g2p_ga", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.U), P(s.vk), P(s.ak), P(s.x), P(s.v), P(s.pa), P(s.F)))
            return
        s.k("rt_g2p_endg", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.y), P(s.z0), P(s.U), f(sl["thinv"]), f(sl["al"]), P(s.fixd), P(s.vbc), dmp,
                               P(s.x), P(s.v), P(s.F)))


    def lsy(s, capN, it2, NC, gate):
        """선탐색 관성 몫 후보 (SC20·SC9·SC[21+i]). --fk 면 이차식 판 (rt_lsyQ + rt_lsyfin)."""
        G = P(gate) if gate is not None else P(None)
        if s.fk:
            s.k("rt_lsyQ", capN, (P(s.IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), G), b=256)
            s.k("rt_lsyfin", 1, (P(s.SC), NC, G), b=1)
        else:
            s.k("rt_lsyN", capN, (P(s.IP), P(s.meff), it2, P(s.eta), P(s.dl), P(s.g0buf), P(s.SC), NC, G), b=256)

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
        fr = int(round(t / s.h))
        force = fr in _force_redo_frames()                                             # 절 CM7 진단: AF_FORCE_REDO=프레임,.. (0 부터) 이면 넘치지 않아도 다시 할당 경로를 그대로 탐
        while int(s.IP[13]) != 0 or force:
            force = False
            if not bool(torch.isfinite(bk[0]).all()) or tries >= 3:                    # 비유한 상태 (발산) 면 상자가 정의되지 않아 매번 넘친다 -> 늘리지 않고 멈춤
                s.x.fill_(float("nan"))
                s.last_ms = float("nan")
                return False
            tries += 1
            s._rs(bk)
            snap = s._cap_snap() if os.environ.get("AF_REDO_COPY", "0") == "1" else None
            s.cap_mul = 1.5 + 0.5 * tries                                              # 필요량은 지금 상태에서 다시 재므로 배수는 작게 (두 배씩 키우면 메모리가 넘친다)
            s.alloc(None)
            s.capture()
            s._rs(bk)
            if snap is not None:                                                       # 절 CM7 진단: 옛 버퍼 내용을 새 버퍼 앞자리에 (데움 실행이 쓴 것은 지움) -- 큰 용량 판과 같은 버퍼 상태
                s._cap_put(snap)
                del snap
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

    _CAP_SKIP = ("capn_d", "capq_d", "mg_MP", "mg_MI", "IP", "SC", "CT", "CD", "BW", "BG", "BP", "A", "Ai")

    def _cap_snap(s):
        """절 CM7 진단 (AF_REDO_COPY=1): 다시 할당 전 버퍼 (이름 -> 텐서) 와 용량."""
        d = {k: v for k, v in vars(s).items() if torch.is_tensor(v) and k not in s._CAP_SKIP}
        ml = [{k: v for k, v in L.items() if torch.is_tensor(v)} for L in getattr(s, "ML", [])]
        return d, ml, s.capN, s.capQc * 8, [L["capq"] for L in getattr(s, "ML", [])]

    def _cap_put(s, snap):
        """옛 버퍼 내용을 새 버퍼 앞자리에 (노드·칸·점 번호는 상자로만 정해져 용량과 무관). 용량 보폭 배열: H [81][capQ], H30·Hs [30][capQ], Ust [K][capN][3], 층 H [81][cq]."""
        d, ml, N0, Q0, cq0 = snap
        N1, Q1 = s.capN, s.capQc * 8

        def put(new, old, rows=None, c0=None, c1=None):
            if new is None or old is None or new is old or new.dtype != old.dtype:
                return
            new.zero_()
            if rows is not None:
                a, b = new.reshape(rows, c1), old.reshape(rows, c0)
                m = min(c0, c1)
                a[:, :m].copy_(b[:, :m])
                return
            m = min(new.numel(), old.numel())
            new.view(-1)[:m].copy_(old.view(-1)[:m])
        for k, old in d.items():
            new = getattr(s, k, None)
            if not torch.is_tensor(new):
                continue
            if k == "H":
                put(new, old, 81, Q0, Q1)
            elif k in ("H30", "Hs"):
                put(new, old, 30, Q0, Q1)
            elif k == "Ust":
                put(new, old, s.K, N0 * 3, N1 * 3)
            else:
                put(new, old)
        for l, (L, Lo) in enumerate(zip(getattr(s, "ML", []), ml)):
            for k, old in Lo.items():
                new = L.get(k)
                if k == "H":
                    put(new, old, 81, cq0[l], L["capq"])
                else:
                    put(new, old)

    def counts(s):
        ip = s.IP.cpu().tolist()
        o = dict(nn=ip[9], nc=ip[11], nq=ip[12], nb=ip[6:9], nbad=int(s.nbad))
        if s.adapt:                                                                    # 마지막 프레임 반복 수 (뉴턴 합, PCG 합, 뉴턴 상한 걸음, PCG 상한 풀이, 걸음, 풀이)
            ct = s.CT.cpu().tolist()
            o["it"] = dict(nN=ct[2], nP=ct[3], capN=ct[4], capP=ct[5], steps=ct[6], solves=ct[7])
        return o

    def bar_counts(s):
        """절 CM10 진단 (--bar_cnt, 결과 불변): 이 프레임 장벽이 걸린 표본점 (어느 뉴턴 반복에서든 활성이고 d < d̂) 수·전체 표본점 (입자 있는 팔분 점) 대비·그 점들이 속한 칸 수,
        관통 칸 (어느 걸음 시작에서든) 수와 그 칸의 표본점 수, 장벽 걸린 점 중 관통 칸에 속한 점 수."""
        nq = int(s.IP[12])
        fl = s.bcf[:nq]; occ = s.bx[:nq * 4].view(-1, 4)[:, 3] > 0
        eng = (fl & 1) > 0; cr = ((fl & 2) > 0) & occ
        cid = torch.arange(nq, device=fl.device) >> 3
        return dict(neng=int(eng.sum()), nocc=int(occ.sum()), ceng=int(torch.unique(cid[eng]).numel()), nc=nq // 8,
                    ccr=int(torch.unique(cid[cr]).numel()), ncr=int(cr.sum()), engcr=int((eng & cr).sum()))

    def barn_counts(s):
        """절 CM14 진단 (결과 불변): 이 프레임 노드 장벽이 걸린 노드 (어느 뉴턴 반복에서든 활성이고 d < d̂_n) 수, 부피 > 0 노드 수 (마지막 걸음 V_n), 어느 걸음 시작에서 d ≤ 0 이라 빠진 노드 수."""
        nn = int(s.IP[9]); fl = s.bnf[:nn]
        return dict(nneng=int(((fl & 1) > 0).sum()), nnv=int((s.bnv[:nn] > 0).sum()), nnbelow=int(((fl & 2) > 0).sum()))

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
        y0 = None
        if getattr(s, "y0ls", False):
            Y = s.Y0S.cpu().numpy()
            y0 = [dict(c=float(Y[k][14]), E=[float(Y[k][21 + i] + Y[k][40 + i]) for i in range(s.ls_n)]) for k in range(getattr(s, "nstage", s.K))]
        return dict(it=out, y0=y0, stick_flip=int(((r0 > 0) != (r1 > 0)).sum()), stick_n0=int((r0 > 0).sum()), stick_n1=int((r1 > 0).sum()), stick_dr=float((r1 - r0).abs().sum()))

    def grel(s):
        """마지막 프레임 걸음별 |g_끝| / |g_처음| (gfin 일 때)."""
        D, F = s.DG.cpu().numpy(), s.DGF.cpu().numpy()
        return [float(np.sqrt(max(F[k][1], 0.0) / max(D[2 * (k * s.nt)][1], 1e-300))) for k in range(getattr(s, "nstage", s.K))]
