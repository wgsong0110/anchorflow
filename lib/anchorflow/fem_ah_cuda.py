"""절 AH (10-30 사용자 자율 지시 "물어보지 말고 해볼 수 있는 거 다 해봐"): 한 설정으로 모든 씬 + 서브스텝 로버스트를 위한 바닥 정의 정리.

확정 판 (절 AD/AF) 의 속도 바닥 (--floor vel, 절 O) 은 서브스텝마다
    접촉 = 지금 보간 위치가 평면 아래인 입자 (s < 0), 노드 접촉 질량 a_n = Σ 0.25 m_p (사면체 등분), v_n <- (1 - a_n/m_n) v_n, 그 뒤 비관통.
여기서 두 가지를 바꾼다 (입자 판정은 그대로, 정의 변경 -- 보고서에 명시):
 (1) --vfl_w gtk: 접촉 가중 = 정답 MPM 의 sticky 격자 노드가 그 입자에 갖는 2 차 B-스플라인 가중 합 w(s) (정답 격자 dx_g, 평면 아래 격자 노드 = 속도 0).
     정답 격자 노드 k 의 법선 좌표 k dx_g, 고착 = k dx_g - p·n < 0 (정답 collide 커널과 같은 float32 비교). 입자 법선 좌표 u = (s + p·n)/dx_g,
     base = floor(u - 0.5), fx = u - base, w = Σ_{i: base+i 고착} {0.5(1.5-fx)², 0.75-(fx-1)², 0.5(fx-0.5)²}_i.
     -> 평면 위 0.5 dx_g 안의 입자도 일부 고착 (정답처럼), 평면 아래 1.5 dx_g 보다 깊으면 1. a_n = Σ 0.25 m_p w(s_p).
     비관통은 실제로 평면 아래 (s < -npd, --vfl_npd: 정답처럼 얕은 관통 허용) 입자가 걸린 노드에만 (정답 비고착 노드는 법선 구속이 없지만, 우리 노드는 정답 격자 노드가 아니라 남긴다).
 (2) --vfl_rate: 서브스텝 고착 몫을 시간 단위로 -- v_n <- (1 - a_n/m_n)^(h_s/h_ref) v_n, h_ref = 정답 substep_dt (정답 sticky 가 서브스텝마다 지우는 속도와 같은 단위 시간당 감쇠;
     기존은 서브스텝마다 (1 - a/m) 라 N 을 바꾸면 접촉 감쇠율이 N 에 비례해 바뀐다).
커널: ah_fc (c_fc 와 같은 힘 + 위 접촉), ah_node_step (ex_node_step_lag 와 같은 걸음 + 위 고착). 기존 커널은 손대지 않는다.
"""
from __future__ import annotations

import numpy as np

from . import fem_rt_cuda as RTC
from . import fem_ex_cuda as EX
from . import fem_c_cuda as CCU

_AH = r'''
extern "C" {
__device__ __forceinline__ float ah_wstk(float s, float pn, float dxg) {
    float u = (s + pn) / dxg;
    float bf = floorf(u - 0.5f);
    float fx = u - bf;
    float w[3] = {0.5f*(1.5f-fx)*(1.5f-fx), 0.75f-(fx-1.f)*(fx-1.f), 0.5f*(fx-0.5f)*(fx-0.5f)};
    float o = 0.f;
    for (int i=0;i<3;i++) { float k = bf + (float)i; if (k*dxg - pn < 0.f) o += w[i]; }
    return o;
}
__global__ void ah_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                      float mu, float lam, float* f, float* rst, int fpol, float* eF,
                      const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                      float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon, int gtk, float band, float pn, float dxg, float* accN, float npd) {
    if ((int)blockIdx.x < nbT) {
        int i0 = blockIdx.x*blockDim.x + threadIdx.x;
        int t;
        if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
        else { if (i0 >= szp[1]) return; t = i0; }
        c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, 0.f, 0.f, 0, rst, fpol, eF);
        return;
    }
    int q = (blockIdx.x - nbT)*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f + band) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    float w = 1.f;
    if (gtk) { if (!(s < band)) return; w = ah_wstk(s, pn, dxg); if (!(w > 0.f) && !(s < 0.f)) return; }
    else if (!(s < 0.f)) return;
    float mp = m[p];
    bool real = s < -npd;
    for (int i=0;i<4;i++) { if (w > 0.f) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp*w); if (real && accN) atomicAdd(&accN[nd[i]], 1.f); }
    if (ncon && s < 0.f) atomicAdd(ncon, 1);
}
// ex_node_step_lag 와 같은 걸음 (nodisp 1 판 = 속도만, U 는 c_lim_apply 가) + 고착 몫 (1 - a/m)^kex, 비관통은 accN > 0 (실제 접촉) 노드만 (accN null 이면 a > 0 노드 -- 기존과 같음)
__global__ void ah_node_step(const int* szp, const float* mn, float* f, float* vn, float g0, float g1, float g2, float hs, float damp,
                             const unsigned char* fixd, const float* vbc, float* acc, float* accN, int mode, int np_, float n0, float n1, float n2, float kex) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    float v[3];
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f) v[c] = (vn[n*3+c] + hs*(f[n*3+c]/m + gg[c]))*damp;
        else v[c] = 0.f;
        f[n*3+c] = 0.f;
    }
    if (acc) {
        float a = acc[n];
        acc[n] = 0.f;
        float cn = a;
        if (accN) { cn = accN[n]; accN[n] = 0.f; }
        if (!(fixd && fixd[n])) {
            if (a > 0.f) {
                float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : powf(fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f), kex));
                for (int c=0;c<3;c++) v[c] *= k;
            }
            if (np_ && cn > 0.f) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
        }
    }
    for (int c=0;c<3;c++) vn[n*3+c] = v[c];
}
}
'''
_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + _AH).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def launch(name, n, args, b=128, grid=None):
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        mod().get_function(name)((g,), (b,), args)


def gt_floor_k0(dxg, pn):
    """검산용 (호스트): 정답 collide 와 같은 float32 비교로 고착 격자 노드 최대 지표."""
    k = np.arange(0, 400, dtype=np.float32)
    return int(k[(k * np.float32(dxg) - np.float32(pn)) < 0].max())
