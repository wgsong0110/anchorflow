"""FEM 실시간 판 4 차 커널 (doc/fem_realtime.md 절 J, 사용자 10-28 "할 수 있는 거 다 해보고 오차랑 영상이랑 시간 측정").

fem_rt_cuda (절 H·I) 와 따로 둔 모듈 (같은 파일을 다른 작업이 고치고 있어 충돌을 피한다). 같은 블록 ELL 배치 (PC.B [15,9,Ms] float32, PC.nb [15,Ms],
대칭 반쪽 = 칸 0..7 직접, 8..14 는 이웃 칸 전치) 위에:

  pcg_th<MK>   타일 곱 지속 PCG, 안쪽 반복은 야코비 대칭 척도 Â = S A S (S = diag(1/√a_ii), 대각 1) 를 MK 형식으로 저장한 행렬
               (0 = float32, 1 = float16, 2 = bfloat16) 과 float32 벡터로, 신뢰 갱신은 원래 float32 저장 행렬 A 와 float64 (r = b − A x_hi) --
               **멈춤 판정은 float64 실제 잔차 |b − A x|/|b| < rtol 로만** (절 I 혼합 정밀도와 같다). 행렬 바이트가 반 (16 비트) 이 된다.
  bell_mv_th   같은 곱만 (마이크로벤치).
"""
from __future__ import annotations

import numpy as np
import torch

_SRC2 = r'''
__device__ __forceinline__ double warp_sum(double v) {
    for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ double block_sum(double v) {
    __shared__ double sh2[32];
    __syncthreads();
    v = warp_sum(v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane==0) sh2[w]=v;
    __syncthreads();
    double t = 0;
    if (w==0) { t = (lane < (blockDim.x>>5)) ? sh2[lane] : 0.0; t = warp_sum(t); }
    return t;  // valid in thread 0
}
__device__ void gbar(unsigned int* cnt, volatile unsigned int* gen, unsigned int nblk) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(cnt, 1u) == nblk - 1) { *cnt = 0; __threadfence(); atomicAdd((unsigned int*)gen, 1u); }
        else { while (*gen == g) { } }
        __threadfence();
    }
    __syncthreads();
}
// float64 row of the sym-half block ELL (float32 storage) -- same as fem_rt_cuda ell_row_sym<true>
__device__ __forceinline__ double ell_row_sym_d(int m, int c, int Ms, const float* B, const int* nb, const double* v) {
    double y = 0;
    for (int k=0;k<8;k++) {
        int j = nb[k*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
        y += bb[0]*__ldcg(&v[j*3]) + bb[Ms]*__ldcg(&v[j*3+1]) + bb[2*Ms]*__ldcg(&v[j*3+2]);
    }
    for (int k=1;k<8;k++) {
        int j = nb[(7+k)*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c)*Ms + j;
        y += bb[0]*__ldcg(&v[j*3]) + bb[3*(long long)Ms]*__ldcg(&v[j*3+1]) + bb[6*(long long)Ms]*__ldcg(&v[j*3+2]);
    }
    return y;
}
// matrix element loaders: MK 0 float32, 1 float16 (PTX cvt, no header), 2 bfloat16
template <int MK> __device__ __forceinline__ float ldm(const void* B, long long i) {
    if (MK == 0) return __ldg(((const float*)B) + i);
    unsigned short u = __ldg(((const unsigned short*)B) + i);
    if (MK == 2) return __uint_as_float(((unsigned int)u) << 16);
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(u)); return f;
}
// tile part: warp k (<15) of a 32-node tile computes slot k's 3-vector for node m (sym half: k >= 8 = transpose of neighbor's slot k-7).
// Bd (optional, float32 [9,Ms]) replaces slot 0 (diagonal block kept in float32).
template <int MK> __device__ __forceinline__ void tile_part_h(int m, int k, int M, int Ms, const void* B, const float* Bd, const int* nb, const float* v,
                                                            float& a0, float& a1, float& a2) {
    a0 = 0.f; a1 = 0.f; a2 = 0.f;
    if (m >= M || k >= 15) return;
    int j = nb[k*Ms+m];
    if (j < 0) return;
    float x0 = __ldcg(&v[j*3]), x1 = __ldcg(&v[j*3+1]), x2 = __ldcg(&v[j*3+2]);
    if (k == 0 && Bd) {
        const float* bb = Bd + m;
        a0 = bb[0]*x0 + bb[Ms]*x1 + bb[2*(long long)Ms]*x2;
        a1 = bb[3*(long long)Ms]*x0 + bb[4*(long long)Ms]*x1 + bb[5*(long long)Ms]*x2;
        a2 = bb[6*(long long)Ms]*x0 + bb[7*(long long)Ms]*x1 + bb[8*(long long)Ms]*x2;
    } else if (k < 8) {
        long long o = (long long)k*9*Ms + m;
        a0 = ldm<MK>(B, o)*x0 + ldm<MK>(B, o+Ms)*x1 + ldm<MK>(B, o+2*(long long)Ms)*x2;
        a1 = ldm<MK>(B, o+3*(long long)Ms)*x0 + ldm<MK>(B, o+4*(long long)Ms)*x1 + ldm<MK>(B, o+5*(long long)Ms)*x2;
        a2 = ldm<MK>(B, o+6*(long long)Ms)*x0 + ldm<MK>(B, o+7*(long long)Ms)*x1 + ldm<MK>(B, o+8*(long long)Ms)*x2;
    } else {
        long long o = (long long)(k-7)*9*Ms + j;
        a0 = ldm<MK>(B, o)*x0 + ldm<MK>(B, o+3*(long long)Ms)*x1 + ldm<MK>(B, o+6*(long long)Ms)*x2;
        a1 = ldm<MK>(B, o+Ms)*x0 + ldm<MK>(B, o+4*(long long)Ms)*x1 + ldm<MK>(B, o+7*(long long)Ms)*x2;
        a2 = ldm<MK>(B, o+2*(long long)Ms)*x0 + ldm<MK>(B, o+5*(long long)Ms)*x1 + ldm<MK>(B, o+8*(long long)Ms)*x2;
    }
}
template <int MK> __device__ void bell_mv_th_body(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) {
    __shared__ float sy[16][3][33];
    int lane = threadIdx.x & 31, k = threadIdx.x >> 5;
    int m = blockIdx.x*32 + lane;
    float a0, a1, a2; tile_part_h<MK>(m, k, M, Ms, B, Bd, nb, v, a0, a1, a2);
    sy[k][0][lane]=a0; sy[k][1][lane]=a1; sy[k][2][lane]=a2;
    __syncthreads();
    if (k < 3 && m < M) { float s=0.f; for (int q=0;q<15;q++) s += sy[q][k][lane]; y[m*3+k] = s; }
}
// persistent mixed PCG: inner on the scaled 16-bit (or float32) matrix Â with float32 vectors in scaled coordinates
// (x̂ = S⁻¹x, r̂ = S r, ẑ = D̂⁻¹ r̂ with D̂⁻¹ = S⁻¹ D⁻¹ S⁻¹, so r̂·ẑ = r·z), reliable update with the float32-stored A and float64:
// xh += S x̂, r = b − A xh, r̂ = S r. The recursion residual estimate |S⁻¹ r̂|² and the stop test |r|² < thr are in original units.
// sc in: [0] rz0 [1] dAd0 ; out: [2] nit [3] rr (true, last update) [4] neg [5] n_updates.  acc: 3 x 2 parities + 2 x 2 (updates)
template <int MK> __device__ void pcg_th_body(int M, int Ms, const float* B32, const void* B, const float* Bd, const int* nb, const float* Dh,
                        const float* sv, const double* bs, double* xh, float* x, float* r, float* z, float* d, float* Ad,
                        double* acc, double* sc, unsigned int* bar, double thr, int maxit, double delta2) {
    __shared__ float sy[16][3][33];
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    int lane = threadIdx.x & 31, kw = threadIdx.x >> 5;
    int ntile = (M + 31) / 32;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0, nup = 0, q = 0;
    double rr = 1.0, rmax = 1.0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        float al = (float)(rz/dAd);
        float prr=0.f, prz=0.f;
        for (int m=tid; m<M; m+=nth) {
            float rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; float rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; float ro = rc/sv[m*3+c]; prr += ro*ro; }
            const float* D = Dh + m*9;
            for (int c=0;c<3;c++){ float zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; }
        }
        double b1 = block_sum((double)prr); double b2 = block_sum((double)prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < delta2*rmax || rr < thr) {
            for (int m=tid; m<M; m+=nth) for (int c=0;c<3;c++){ xh[m*3+c] += (double)x[m*3+c]*(double)sv[m*3+c]; x[m*3+c] = 0.f; }
            if (tid==0) { acc[6+(1-q)*2]=0; acc[6+(1-q)*2+1]=0; }
            gbar(bar, bar+1, nblk);
            double trr=0, trz=0;
            for (int m=tid; m<M; m+=nth) {
                double rv[3]; float rh[3];
                for (int c=0;c<3;c++) {
                    double ax = ell_row_sym_d(m, c, Ms, B32, nb, xh);
                    rv[c] = bs[m*3+c] - ax; trr += rv[c]*rv[c];
                    rh[c] = (float)(rv[c]*(double)sv[m*3+c]); r[m*3+c] = rh[c];
                }
                const float* D = Dh + m*9;
                for (int c=0;c<3;c++){ float zz = D[c*3]*rh[0]+D[c*3+1]*rh[1]+D[c*3+2]*rh[2]; z[m*3+c]=zz; trz += (double)rh[c]*(double)zz; }
            }
            double c1 = block_sum(trr); double c2 = block_sum(trz);
            if (threadIdx.x==0) { atomicAdd(&acc[6+q*2], c1); atomicAdd(&acc[6+q*2+1], c2); }
            gbar(bar, bar+1, nblk);
            rr = va[6+q*2]; rz2 = va[6+q*2+1];
            q = 1-q; nup++;
            rmax = rr;
            if (rr < thr) break;
        } else if (rr > rmax) rmax = rr;
        double be = rz2/rz; rz = rz2;
        float bev = (float)be;
        float pd = 0.f;
        for (int tile = blockIdx.x; tile < ntile; tile += gridDim.x) {
            int m = tile*32 + lane;
            float a0, a1, a2; tile_part_h<MK>(m, kw, M, Ms, B, Bd, nb, z, a0, a1, a2);
            sy[kw][0][lane]=a0; sy[kw][1][lane]=a1; sy[kw][2][lane]=a2;
            __syncthreads();
            if (kw < 3 && m < M) {
                float s=0.f; for (int qq=0;qq<15;qq++) s += sy[qq][kw][lane];
                int e = m*3+kw;
                float dd = z[e] + bev*d[e]; float aa = s + bev*Ad[e]; d[e]=dd; Ad[e]=aa; pd += dd*aa;
            }
            __syncthreads();
        }
        double b3 = block_sum((double)pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; sc[5]=nup; }
}

// ---------------- pcg_t2 (절 J 항목 2, 반복 비용): fem_rt_cuda pcg_t32 mode 1 (혼합 정밀도, float32 저장 행렬 그대로) 과 같은 식,
// 곱에서 한 블록이 타일 R 개를 동시에 (메모리 병렬도 R 배), 행렬·이웃 표는 읽기 전용 경로 (__ldg).
template <int R> __device__ void pcg_t2_body(int M, int Ms, const float* __restrict__ B, const int* __restrict__ nb, const float* __restrict__ Dinv,
                        const double* bs, double* xh, float* x, float* r, float* z, float* d, float* Ad,
                        double* acc, double* sc, unsigned int* bar, double thr, int maxit, double delta2) {
    __shared__ float sy[R][16][3][33];
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    int lane = threadIdx.x & 31, kw = threadIdx.x >> 5;
    int ntile = (M + 31) / 32;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0, nup = 0, q = 0;
    double rr = 1.0, rmax = 1.0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        float al = (float)(rz/dAd);
        float prr=0.f, prz=0.f;
        for (int m=tid; m<M; m+=nth) {
            float rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; float rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const float* D = Dinv + m*9;
            for (int c=0;c<3;c++){ float zz = __ldg(&D[c*3])*rv[0]+__ldg(&D[c*3+1])*rv[1]+__ldg(&D[c*3+2])*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum((double)prr); double b2 = block_sum((double)prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < delta2*rmax || rr < thr) {
            for (int m=tid; m<M; m+=nth) for (int c=0;c<3;c++){ xh[m*3+c] += (double)x[m*3+c]; x[m*3+c] = 0.f; }
            if (tid==0) { acc[6+(1-q)*2]=0; acc[6+(1-q)*2+1]=0; }
            gbar(bar, bar+1, nblk);
            double trr=0, trz=0;
            for (int m=tid; m<M; m+=nth) {
                double rv[3];
                for (int c=0;c<3;c++) { double ax = ell_row_sym_d(m, c, Ms, B, nb, xh); rv[c] = bs[m*3+c] - ax; r[m*3+c] = (float)rv[c]; }
                const float* D = Dinv + m*9;
                for (int c=0;c<3;c++){ float zz = D[c*3]*(float)rv[0]+D[c*3+1]*(float)rv[1]+D[c*3+2]*(float)rv[2]; z[m*3+c]=zz; trz += rv[c]*(double)zz; trr += rv[c]*rv[c]; }
            }
            double c1 = block_sum(trr); double c2 = block_sum(trz);
            if (threadIdx.x==0) { atomicAdd(&acc[6+q*2], c1); atomicAdd(&acc[6+q*2+1], c2); }
            gbar(bar, bar+1, nblk);
            rr = va[6+q*2]; rz2 = va[6+q*2+1];
            q = 1-q; nup++;
            rmax = rr;
            if (rr < thr) break;
        } else if (rr > rmax) rmax = rr;
        double be = rz2/rz; rz = rz2;
        float bev = (float)be;
        float pd = 0.f;
        for (int t0 = blockIdx.x*R; t0 < ntile; t0 += gridDim.x*R) {
            float a[R][3];
            #pragma unroll
            for (int u=0; u<R; u++) { int m = (t0+u)*32 + lane; tile_part_h<0>(m, kw, (t0+u < ntile) ? M : 0, Ms, B, (const float*)0, nb, z, a[u][0], a[u][1], a[u][2]); }
            #pragma unroll
            for (int u=0; u<R; u++) { sy[u][kw][0][lane]=a[u][0]; sy[u][kw][1][lane]=a[u][1]; sy[u][kw][2][lane]=a[u][2]; }
            __syncthreads();
            #pragma unroll
            for (int u=0; u<R; u++) {
                int m = (t0+u)*32 + lane;
                if (kw < 3 && m < M && t0+u < ntile) {
                    float s=0.f; for (int qq=0;qq<15;qq++) s += sy[u][qq][kw][lane];
                    int e = m*3+kw;
                    float dd = z[e] + bev*d[e]; float aa = s + bev*Ad[e]; d[e]=dd; Ad[e]=aa; pd += dd*aa;
                }
            }
            __syncthreads();
        }
        double b3 = block_sum((double)pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; sc[5]=nup; }
}
extern "C" {
__global__ void bell_mv_th0(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<0>(M, B, Bd, nb, v, y, Ms); }
__global__ void bell_mv_th1(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<1>(M, B, Bd, nb, v, y, Ms); }
__global__ void bell_mv_th2(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<2>(M, B, Bd, nb, v, y, Ms); }

// ---------------- P2G (절 J 항목 4): CudaTetqK.__init__ 의 torch 연산 (바닥 후보 분할, 사면체 관성 M4·B4, 부피·부피 가중 F, C0) 을 입자당 스레드 하나로 ----------------
__global__ void p2g_tq(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const float* U,
                       const long long* rows, const float* lamb, const long long* pinv, double h, double g0, double g1, double g2,
                       float p0, float p1, float p2, float n0, float n1, float n2, float eps_sp, float eps_st, int floor,
                       double* M4, double* B4, double* VT, double* FTs, double* C0, long long* nlist, int* ncnt, float* kk, float* cc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    double ce = 0.0;
    if (p < N) {
        long long r[4]; float l[4];
        for (int i=0;i<4;i++){ r[i]=rows[p*4+i]; l[i]=lamb[p*4+i]; }
        bool near = false;
        if (floor) {
            float u0=0.f, u1=0.f, u2=0.f;
            for (int i=0;i<4;i++){ u0 += l[i]*U[r[i]*3]; u1 += l[i]*U[r[i]*3+1]; u2 += l[i]*U[r[i]*3+2]; }
            float s = (x[p*3]+u0-p0)*n0 + (x[p*3+1]+u1-p1)*n1 + (x[p*3+2]+u2-p2)*n2;
            if (eps_sp > 0.f && s/eps_sp <= 20.f) near = true;
            if (eps_st > 0.f && s/eps_st <= 20.f) near = true;
        }
        double k = (double)m[p]/(h*h);
        double c[3] = {h*(double)v[p*3] + h*h*g0, h*(double)v[p*3+1] + h*h*g1, h*(double)v[p*3+2] + h*h*g2};
        kk[p] = (float)k; cc[p*3] = (float)c[0]; cc[p*3+1] = (float)c[1]; cc[p*3+2] = (float)c[2];
        if (near) { int q = atomicAdd(ncnt, 1); nlist[q] = p; }
        else {
            long long t = pinv[p];
            double lm[4] = {(double)l[0], (double)l[1], (double)l[2], (double)l[3]};
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) atomicAdd(&M4[t*16+i*4+j], k*lm[i]*lm[j]);
            for (int i=0;i<4;i++) for (int a=0;a<3;a++) atomicAdd(&B4[t*12+i*3+a], k*lm[i]*c[a]);
            double V = (double)vol[p];
            atomicAdd(&VT[t], V);
            for (int e=0;e<9;e++) atomicAdd(&FTs[t*9+e], V*(double)F[p*9+e]);
            ce = 0.5*k*(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
        }
    }
    ce = warp_sum(ce);
    if ((threadIdx.x & 31) == 0 && ce != 0.0) atomicAdd(C0, ce);
}
// per tet: F̄ = Σ V F / Σ V (or I), float32 copies of gl, F̄, V, M4
__global__ void tq_fin(int T, const double* gl8, double* FTs, const double* VT, const double* M4, float* gl4, float* FT4, float* VT4, float* M44) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    double V = VT[t];
    for (int e=0;e<9;e++) { double f = (V > 0) ? FTs[t*9+e]/fmax(V, 1e-300) : ((e%4==0) ? 1.0 : 0.0); FTs[t*9+e] = f; FT4[t*9+e] = (float)f; }
    VT4[t] = (float)V;
    for (int e=0;e<12;e++) gl4[t*12+e] = (float)gl8[t*12+e];
    for (int e=0;e<16;e++) M44[t*16+e] = (float)M4[t*16+e];
}
// glp[p] = gl4[pinv[p]] (float32 [N,12])
__global__ void gather12(int N, const long long* pinv, const float* gl4, float* glp) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N*12) return;
    int p = i / 12, e = i - p*12;
    glp[i] = gl4[pinv[p]*12 + e];
}
#define PCG_T2_ARGS int M, int Ms, const float* B, const int* nb, const float* Dinv, const double* bs, double* xh, float* x, float* r, float* z, \
    float* d, float* Ad, double* acc, double* sc, unsigned int* bar, double thr, int maxit, double delta2
#define PCG_T2_CALL(R) pcg_t2_body<R>(M, Ms, B, nb, Dinv, bs, xh, x, r, z, d, Ad, acc, sc, bar, thr, maxit, delta2)
__global__ void __launch_bounds__(512, 1) pcg_t2_r1(PCG_T2_ARGS) { PCG_T2_CALL(1); }
__global__ void __launch_bounds__(512, 1) pcg_t2_r2(PCG_T2_ARGS) { PCG_T2_CALL(2); }
__global__ void __launch_bounds__(512, 1) pcg_t2_r4(PCG_T2_ARGS) { PCG_T2_CALL(4); }
__global__ void __launch_bounds__(512, 2) pcg_t2_r1b2(PCG_T2_ARGS) { PCG_T2_CALL(1); }
__global__ void __launch_bounds__(512, 2) pcg_t2_r2b2(PCG_T2_ARGS) { PCG_T2_CALL(2); }
__global__ void __launch_bounds__(512, 2) pcg_t2_r4b2(PCG_T2_ARGS) { PCG_T2_CALL(4); }
#define PCG_TH_ARGS int M, int Ms, const float* B32, const void* B, const float* Bd, const int* nb, const float* Dh, const float* sv, const double* bs, \
    double* xh, float* x, float* r, float* z, float* d, float* Ad, double* acc, double* sc, unsigned int* bar, double thr, int maxit, double delta2
#define PCG_TH_CALL(K) pcg_th_body<K>(M, Ms, B32, B, Bd, nb, Dh, sv, bs, xh, x, r, z, d, Ad, acc, sc, bar, thr, maxit, delta2)
__global__ void pcg_th0(PCG_TH_ARGS) { PCG_TH_CALL(0); }
__global__ void pcg_th1(PCG_TH_ARGS) { PCG_TH_CALL(1); }
__global__ void pcg_th2(PCG_TH_ARGS) { PCG_TH_CALL(2); }
}
'''

_MOD2 = [None]


def _mod2():
    if _MOD2[0] is None:
        import cupy as cp
        _MOD2[0] = cp.RawModule(code=_SRC2.encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD2[0]


def _p(t):
    """커널 포인터 인자 (CuPy 배열 변환 없이 장치 주소를 값으로)."""
    return np.uint64(t.data_ptr()) if t is not None else np.uint64(0)


def coop_blocks2(name, bsz=512, cap=3):
    import cupy as cp
    f = _mod2().get_function(name)
    dev = cp.cuda.Device()
    sm = int(dev.attributes["MultiProcessorCount"])
    regs_sm = int(dev.attributes["MaxRegistersPerMultiprocessor"])
    thr_sm = int(dev.attributes["MaxThreadsPerMultiProcessor"])
    nreg = max(int(f.num_regs), 1)
    per = min(cap, thr_sm // bsz, regs_sm // (((nreg + 7) // 8 * 8) * bsz))
    return sm * max(per, 1), dict(sm=sm, regs=nreg, per_sm=per)


class _Stream:
    def __enter__(self):
        import cupy as cp
        self.s = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        self.s.__enter__()
        return self

    def __exit__(self, *e):
        self.s.__exit__(*e)


class HalfMat:
    """PC.B (float32 대칭 반쪽 블록 ELL) 에서 야코비 대칭 척도 Â = S A S 를 MK 형식 (0 f32 / 1 f16 / 2 bf16) 으로 만든다.
    dg32 = 대각 블록 (칸 0) 은 float32 로 따로 (Bd), comp = 평행이동 모드 보정 (반올림 오차의 같은 성분 행 합을 대각에서 뺀다)."""

    def __init__(self, Mmax, dev):
        self.Mmax, self.dev = int(Mmax), dev
        self.B16 = torch.zeros(8 * 9 * self.Mmax, dtype=torch.int16, device=dev)
        self.Bf = torch.zeros(8 * 9 * self.Mmax, dtype=torch.float32, device=dev)
        self.Bd = torch.zeros(9 * self.Mmax, dtype=torch.float32, device=dev)
        self.sv = torch.ones(3 * self.Mmax, dtype=torch.float32, device=dev)
        self.Dh = torch.zeros(9 * self.Mmax, dtype=torch.float32, device=dev)

    def build(self, PC, M, mk, dg32=False, comp=False):
        Ms = PC.Mmax
        B = PC.B.view(15, 9, Ms)[:8, :, :M]
        nb = PC.nb.view(15, Ms)[:8, :M].long()
        dia = torch.stack([B[0, 0], B[0, 4], B[0, 8]], 1)                      # [M,3]
        s = torch.where(dia > 0, dia.clamp_min(1e-30).rsqrt(), torch.ones_like(dia))
        e = torch.arange(9, device=B.device)
        rs = s.t()[e // 3]                                                      # [9,M]
        cs = s[nb.clamp_min(0)].permute(0, 2, 1)[:, e % 3]                      # [8,9,M]
        Bh = B * rs.unsqueeze(0) * cs                                           # Â
        self.sv[:3 * M].copy_(s.reshape(-1))
        Di = PC.Dinv[:M * 9].view(M, 3, 3)
        sig = 1.0 / s.double()
        self.Dh[:M * 9].copy_((sig.unsqueeze(2) * Di * sig.unsqueeze(1)).reshape(-1).float())  # D̂⁻¹ = S⁻¹ D⁻¹ S⁻¹
        Bv = self.Bf.view(8, 9, Ms) if mk == 0 else self.B16.view(8, 9, Ms)
        if mk == 0:
            Bv[:, :, :M] = Bh
            q = Bh
        else:
            dt = torch.float16 if mk == 1 else torch.bfloat16
            qd = Bh.to(dt)
            Bv[:, :, :M] = qd.view(torch.int16)
            q = qd.float()
        self.dg32 = bool(dg32 or comp)
        if self.dg32:
            Bd = Bh[0].clone()                                                  # [9,M] float32 대각 블록
            if comp and mk != 0:
                # 같은 성분 (a == c) 반올림 오차 (원래 척도) 의 행 합: 칸 1..7 (행 = 노드 m) 과 그 전치 (행 = 이웃 j)
                err = (q[1:] - Bh[1:]).double() / (rs.unsqueeze(0) * cs[1:]).double()          # [7,9,M]
                ok = (nb[1:] >= 0)
                fix = torch.zeros(M, 3, device=B.device, dtype=torch.float64)
                for a_ in range(3):
                    ea = err[:, a_ * 4] * ok                                        # 칸 (a,a)
                    fix[:, a_] += ea.sum(0)
                    fix[:, a_].index_add_(0, nb[1:].clamp_min(0).reshape(-1), (ea * ok).reshape(-1))
                for a_ in range(3):
                    Bd[a_ * 4] = (Bd[a_ * 4].double() - fix[:, a_] * s[:, a_].double() ** 2).float()
            self.Bd.view(9, Ms)[:, :M] = Bd
        return s

    def ptrs(self, mk):
        return (self.Bf if mk == 0 else self.B16), (self.Bd if self.dg32 else None)


def solve_th(PC, HM, M, b, rtol, ncg, mk=1, delta=0.1):
    """pcg_th<mk>: HM.build 뒤에. 반환 (x float64, 반복, 실제 잔차 (마지막 갱신, 원래 단위), 음곡률)."""
    n = 3 * M
    dev = PC.dev
    key = f"th{mk}"
    if not hasattr(PC, key):
        bufs = dict(acc=torch.zeros(12, dtype=torch.float64, device=dev), sc=torch.zeros(8, dtype=torch.float64, device=dev),
                    bar=torch.zeros(2, dtype=torch.int32, device=dev),
                    bs=torch.zeros(PC.Mmax * 3, dtype=torch.float64, device=dev), xh=torch.zeros(PC.Mmax * 3, dtype=torch.float64, device=dev))
        for nm in ("x", "r", "z", "d", "Ad"):
            bufs[nm] = torch.zeros(PC.Mmax * 3, dtype=torch.float32, device=dev)
        bufs["nblk"], bufs["info"] = coop_blocks2(f"pcg_th{mk}", 512, cap=getattr(PC, "tl_cap", 3))
        setattr(PC, key, bufs)
    T_ = getattr(PC, key)
    PC.th_info = (T_["nblk"], T_["info"])
    sc_ = (b * b).sum().clamp_min(1e-300).sqrt()
    bs = b / sc_
    T_["bs"][:n].copy_(bs)
    T_["xh"].zero_()
    T_["x"].zero_()
    rh = (bs * HM.sv[:n].double()).float()
    T_["r"][:n].copy_(rh)
    zh = (HM.Dh[:M * 9].view(M, 3, 3) @ rh.view(M, 3, 1)).view(-1)
    T_["z"][:n].copy_(zh)
    T_["d"][:n].copy_(zh)
    Bp, Bd = HM.ptrs(mk)
    nt = (M + 31) // 32
    md = _mod2()
    with _Stream():
        md.get_function(f"bell_mv_th{mk}")((nt,), (512,), (np.int32(M), _p(Bp), _p(Bd), _p(PC.nb), _p(T_["d"]), _p(T_["Ad"]), np.int32(PC.Mmax)))
    T_["acc"].zero_()
    T_["bar"].zero_()
    T_["sc"][0] = (rh.double() * zh.double()).sum()
    T_["sc"][1] = (T_["d"][:n].double() * T_["Ad"][:n].double()).sum()
    with _Stream():
        md.get_function(f"pcg_th{mk}")((T_["nblk"],), (512,), (np.int32(M), np.int32(PC.Mmax), _p(PC.B), _p(Bp), _p(Bd), _p(PC.nb), _p(HM.Dh), _p(HM.sv),
                                                              _p(T_["bs"]), _p(T_["xh"]), _p(T_["x"]), _p(T_["r"]), _p(T_["z"]), _p(T_["d"]),
                                                              _p(T_["Ad"]), _p(T_["acc"]), _p(T_["sc"]), _p(T_["bar"]), np.float64(rtol * rtol),
                                                              np.int32(ncg), np.float64(delta * delta)))
    out = T_["sc"][2:6].tolist()
    PC.last_nup = int(out[3])
    return T_["xh"][:n] * sc_, int(out[0]), float(out[1]) ** 0.5, bool(out[2] > 0.5)


def morton_perm(gi):
    """격자 좌표 gi [M,3] 의 Morton (Z 곡선) 순서 -> perm (새 i 번째 = 옛 perm[i])."""
    g = (gi - gi.min(0).values).long()

    def spread(v):
        v = v & 0x3ff
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v
    key = spread(g[:, 0]) | (spread(g[:, 1]) << 1) | (spread(g[:, 2]) << 2)
    return torch.argsort(key)


class GridK2:
    """fem_rt_cuda.GridK 와 같은 격자·같은 값 (같은 GPU 연산), 호스트 동기화만 줄인 판 (절 J 항목 4):
    격자 상수 (lattice A·Ai) 는 node_h 별로 한 번, 정수 범위 imin·imax 는 동기화 1 번, 노드·사면체 수는 cumsum 끝 두 값을 동기화 1 번으로
    (nonzero 대신 scatter 로 uniq). 이전: nn.tolist·nonzero·int(tcm[-1])·격자 상수 H2D·역행렬 매 프레임."""
    _LAT = {}

    def __init__(self, x, node_h):
        from . import simplex as SX
        from . import fem_rt_cuda as RTC
        dev = x.device
        key = (float(node_h), str(dev))
        if key not in GridK2._LAT:
            GridK2._LAT[key] = SX.lattice(float(node_h), SX.HZ_CUBE * float(node_h), device=dev, dtype=x.dtype)
        lat = GridK2._LAT[key]
        xd = x.detach()
        lo_ = xd.min(0).values
        hi_ = xd.max(0).values
        ext = (hi_ - lo_).max().clamp_min(1e-12)
        pad = 0.05 * ext
        b0, b1 = lo_ - pad, hi_ + pad
        cor = torch.stack([torch.stack([b0[0] if (mm & 1) else b1[0], b0[1] if (mm & 2) else b1[1], b0[2] if (mm & 4) else b1[2]]) for mm in range(8)], 0)
        y = cor @ lat.Ai.T
        imin = torch.floor(y.min(0).values).long() - 1
        imax = torch.ceil(y.max(0).values).long() + 1
        ii = torch.cat([imin, imax]).tolist()                                     # 동기화 1
        imn = ii[:3]
        nnl = [max(ii[3 + k] - ii[k] + 1, 2) for k in range(3)]
        lo = (lat.A @ imin.to(lat.A.dtype))
        n0, n1, n2 = nnl
        P = n0 * n1 * n2
        N = x.shape[0]
        idx = torch.empty(N, 4, dtype=torch.long, device=dev)
        lamb = torch.empty(N, 4, dtype=torch.float32, device=dev)
        tid = torch.empty(N, dtype=torch.long, device=dev)
        nmark = torch.zeros(P, dtype=torch.uint8, device=dev)
        tmark = torch.zeros(P * 9, dtype=torch.uint8, device=dev)
        xs = x.float().contiguous()
        lo_f = lo.float().contiguous()
        Ai = lat.Ai.float().contiguous()
        gr, bl = RTC._grid(N, 256)
        with RTC._Stream():
            RTC._mod().get_function("locate_k")(gr, bl, (np.int32(N), RTC._cp(xs), RTC._cp(lo_f), RTC._cp(Ai), np.int32(n0), np.int32(n1), np.int32(n2),
                                                         RTC._cp(idx), RTC._cp(lamb), RTC._cp(tid), RTC._cp(nmark), RTC._cp(tmark)))
        ncm = torch.cumsum(nmark, 0, dtype=torch.int64) - 1
        tcm = torch.cumsum(tmark, 0, dtype=torch.int64) - 1
        MT = torch.stack([ncm[-1], tcm[-1]]).tolist()                              # 동기화 2
        M, T = int(MT[0]) + 1, int(MT[1]) + 1
        ar = torch.arange(P, device=dev)
        uniq = torch.empty(M + 1, dtype=torch.long, device=dev).scatter_(0, torch.where(nmark.bool(), ncm, torch.full_like(ncm, M)), ar)[:M]
        rows = ncm[idx]
        pinv = tcm[tid]
        rep = torch.empty(T, dtype=torch.long, device=dev).scatter_(0, pinv, torch.arange(N, device=dev))
        tets = rows[rep].contiguous()
        k2 = uniq % n2
        k1 = (uniq // n2) % n1
        k0 = uniq // (n1 * n2)
        npos = (torch.stack([k0, k1, k2], -1).to(lat.A.dtype) @ lat.A.T + lo).float().contiguous()
        gl8 = torch.empty(T, 12, dtype=torch.float64, device=dev)
        gr, bl = RTC._grid(T, 128)
        with RTC._Stream():
            RTC._mod().get_function("tet_geo")(gr, bl, (np.int32(T), RTC._cp(tets), RTC._cp(npos), RTC._cp(gl8)))
        self.gi = torch.stack([k0, k1, k2], -1) + torch.tensor(imn, device=dev, dtype=torch.long)
        self.rows, self.lam, self.uniq, self.npos, self.tets, self.pinv = rows, lamb, uniq, npos, tets, pinv
        self.nmark, self.ncm = nmark, ncm
        self.T, self.M, self.N, self.nn = T, M, N, (n0, n1, n2)
        self.gl8 = gl8.view(T, 4, 3)


def _plane(cfg):
    """바닥 평면 (CPU float32 로 정규화 -- CudaTetqK 의 pt.tolist()/nr.tolist() 와 같은 값, 장치 동기화 없음)."""
    bc = [b for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider"][0]
    pt = torch.as_tensor(bc["point"], dtype=torch.float32)
    nr = torch.as_tensor(bc["normal"], dtype=torch.float32)
    nr = nr / nr.norm()
    return [float(q) for q in pt.tolist()] + [float(q) for q in nr.tolist()]


def make_tq(G, x, v, F, m, vol, cfg, h, U, eps_sp=0.01, eps_st=0.01, stick_xn=True):
    """fem_rt_cuda.CudaTetqK 와 같은 필드를 P2G 커널 (p2g_tq·tq_fin·gather12) 로 채운 객체 (입자 바닥, cfg_bc 없음).
    바닥 후보 목록은 커널 원자 순서로 모은 뒤 정렬 (기존 nonzero 와 같은 오름차순)."""
    from . import fem_rt_cuda as RTC
    from . import phys_resid as PR
    CF = RTC.CudaTetqK.__new__(RTC.CudaTetqK)
    dev = x.device
    f8 = torch.float64
    T, N = G.T, G.N
    h = float(h)
    CF.T, CF.dev, CF.tets, CF.h = T, dev, G.tets, h
    CF.floor = True
    pl = _plane(cfg)
    if not hasattr(make_tq, "buf") or make_tq.buf["N"] != N:
        make_tq.buf = dict(N=N, nlist=torch.empty(N, dtype=torch.long, device=dev), kk=torch.empty(N, dtype=torch.float32, device=dev),
                           cc=torch.empty(N, 3, dtype=torch.float32, device=dev), glp=torch.empty(N, 12, dtype=torch.float32, device=dev))
    bf = make_tq.buf
    sc = torch.zeros(2, dtype=f8, device=dev)                              # [0] C0, [1] (int) count
    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    M48 = torch.zeros(T, 16, dtype=f8, device=dev)
    B48 = torch.zeros(T, 12, dtype=f8, device=dev)
    VT8 = torch.zeros(T, dtype=f8, device=dev)
    FT8 = torch.zeros(T, 9, dtype=f8, device=dev)
    g = [float(q) for q in cfg["g"]]
    xs, vs, Fs = x.float().contiguous(), v.float().contiguous(), F.float().reshape(-1, 9).contiguous()
    Uf = U.float().contiguous()
    gr, bl = RTC._grid(N, 256)
    md = _mod2()
    with RTC._Stream():
        md.get_function("p2g_tq")(gr, bl, (np.int32(N), RTC._cp(xs), RTC._cp(vs), RTC._cp(Fs), RTC._cp(m), RTC._cp(vol), RTC._cp(Uf),
                                           RTC._cp(G.rows), RTC._cp(G.lam), RTC._cp(G.pinv), np.float64(h), np.float64(g[0]), np.float64(g[1]), np.float64(g[2]),
                                           np.float32(pl[0]), np.float32(pl[1]), np.float32(pl[2]), np.float32(pl[3]), np.float32(pl[4]), np.float32(pl[5]),
                                           np.float32(eps_sp), np.float32(eps_st), np.int32(1), RTC._cp(M48), RTC._cp(B48), RTC._cp(VT8), RTC._cp(FT8),
                                           RTC._cp(sc), RTC._cp(bf["nlist"]), RTC._cp(cnt), RTC._cp(bf["kk"]), RTC._cp(bf["cc"])))
    gl8 = G.gl8.reshape(T, 12).contiguous()
    gl4 = torch.empty(T, 12, dtype=torch.float32, device=dev)
    FT4 = torch.empty(T, 9, dtype=torch.float32, device=dev)
    VT4 = torch.empty(T, dtype=torch.float32, device=dev)
    M44 = torch.empty(T, 16, dtype=torch.float32, device=dev)
    gr, bl = RTC._grid(T, 128)
    with RTC._Stream():
        md.get_function("tq_fin")(gr, bl, (np.int32(T), RTC._cp(gl8), RTC._cp(FT8), RTC._cp(VT8), RTC._cp(M48), RTC._cp(gl4), RTC._cp(FT4), RTC._cp(VT4), RTC._cp(M44)))
        gr, bl = RTC._grid(N * 12, 256)
        md.get_function("gather12")(gr, bl, (np.int32(N), RTC._cp(G.pinv), RTC._cp(gl4), RTC._cp(bf["glp"])))
    nn = int(cnt.item())                                                      # 동기화 1 (바닥 후보 수 -- 커널 격자 크기)
    CF.ni = torch.sort(bf["nlist"][:nn])[0]
    CF.nn = nn
    CF.M48, CF.B48, CF.VT8, CF.FT8 = M48.view(T, 4, 4), B48.view(T, 4, 3), VT8, FT8.view(T, 3, 3)
    CF.C0 = sc[0]
    CF.gl8, CF.gl4, CF.FT4, CF.VT4, CF.M44 = gl8, gl4, FT4.view(T, 3, 3), VT4, M44.view(T, 4, 4)
    CF.mu, CF.lam = PR.lame(cfg["E"], cfg["nu"])
    CF.x = xs
    CF.rows = G.rows
    CF.lamb = G.lam.contiguous()
    CF.glp = bf["glp"]
    CF.Fp = Fs
    CF.vol = vol.float().contiguous()
    CF.kk = bf["kk"]
    CF.cc = bf["cc"]
    CF.tinv = G.pinv.contiguous()
    CF.pl = pl
    CF.eps_st, CF.eps_sp = float(eps_st), float(eps_sp)
    CF.xn = 1.0 if (stick_xn and eps_st > 0) else 0.0
    CF.cnt = torch.zeros(2, dtype=torch.int32, device=dev)
    CF.Ht = None
    return CF


def solve_t2(PC, M, b, rtol, ncg, kern="pcg_t2_r2", delta=0.1, cap=2):
    """pcg_t2_*: fem_rt_cuda.CudaPcg.solve_tile (prec 32, mode 1) 과 같은 식, 곱의 메모리 병렬도만 높인 판. PC.Dinv32 (asm_k) 또는 PC.Dinv 의 float32."""
    n = 3 * M
    dev = PC.dev
    key = "t2_" + kern
    if not hasattr(PC, "t2buf"):
        bufs = dict(acc=torch.zeros(12, dtype=torch.float64, device=dev), sc=torch.zeros(8, dtype=torch.float64, device=dev),
                    bar=torch.zeros(2, dtype=torch.int32, device=dev),
                    bs=torch.zeros(PC.Mmax * 3, dtype=torch.float64, device=dev), xh=torch.zeros(PC.Mmax * 3, dtype=torch.float64, device=dev),
                    Di=torch.zeros(PC.Mmax * 9, dtype=torch.float32, device=dev))
        for nm in ("x", "r", "z", "d", "Ad"):
            bufs[nm] = torch.zeros(PC.Mmax * 3, dtype=torch.float32, device=dev)
        PC.t2buf = bufs
    T_ = PC.t2buf
    if not hasattr(PC, key):
        setattr(PC, key, coop_blocks2(kern, 512, cap=cap))
    nblk, info = getattr(PC, key)
    PC.t2_info = (nblk, info)
    sc_ = (b * b).sum().clamp_min(1e-300).sqrt()
    bs = b / sc_
    T_["bs"][:n].copy_(bs)
    T_["Di"][:M * 9].copy_(PC.Dinv[:M * 9])
    T_["xh"].zero_()
    T_["x"].zero_()
    rV = bs.float()
    T_["r"][:n].copy_(rV)
    zV = (T_["Di"][:M * 9].view(M, 3, 3) @ rV.view(M, 3, 1)).view(-1)
    T_["z"][:n].copy_(zV)
    T_["d"][:n].copy_(zV)
    nt = (M + 31) // 32
    md = _mod2()
    with _Stream():
        md.get_function("bell_mv_th0")((nt,), (512,), (np.int32(M), _p(PC.B), _p(None), _p(PC.nb), _p(T_["d"]), _p(T_["Ad"]), np.int32(PC.Mmax)))
    T_["acc"].zero_()
    T_["bar"].zero_()
    T_["sc"][0] = (rV.double() * zV.double()).sum()
    T_["sc"][1] = (T_["d"][:n].double() * T_["Ad"][:n].double()).sum()
    with _Stream():
        md.get_function(kern)((nblk,), (512,), (np.int32(M), np.int32(PC.Mmax), _p(PC.B), _p(PC.nb), _p(T_["Di"]), _p(T_["bs"]), _p(T_["xh"]), _p(T_["x"]),
                                                _p(T_["r"]), _p(T_["z"]), _p(T_["d"]), _p(T_["Ad"]), _p(T_["acc"]), _p(T_["sc"]), _p(T_["bar"]),
                                                np.float64(rtol * rtol), np.int32(ncg), np.float64(delta * delta)))
    out = T_["sc"][2:6].tolist()
    PC.last_nup = int(out[3])
    return (T_["xh"][:n] + T_["x"][:n].double()) * sc_, int(out[0]), float(out[1]) ** 0.5, bool(out[2] > 0.5)
