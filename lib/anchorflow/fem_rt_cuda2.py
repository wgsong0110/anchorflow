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
extern "C" {
__global__ void bell_mv_th0(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<0>(M, B, Bd, nb, v, y, Ms); }
__global__ void bell_mv_th1(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<1>(M, B, Bd, nb, v, y, Ms); }
__global__ void bell_mv_th2(int M, const void* B, const float* Bd, const int* nb, const float* v, float* y, int Ms) { bell_mv_th_body<2>(M, B, Bd, nb, v, y, Ms); }
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
