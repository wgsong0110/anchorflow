"""절 AK: 입자 적분점 셀 탄성 (--ak_pq) -- 셀 에너지 E_T = Σ_{p∈T} w_p ψ(A_T F_p) (F_p = 프레임 시작 입자 F, w_p = V_p · V_T'/V_T, 상한 뒤 부피 비 그대로).
힘 f_i = -Σ_p w_p P(A_T F_p) F_pᵀ ∇λ_i (= -∂E/∂x_i, 에너지의 기울기 -> 운동량·각운동량 보존). 셀 평균 F̄ 의 ψ(A F̄) 대신 입자마다 ψ 를 더하므로
프레임 끝 셀 에너지 = 다음 프레임 시작 셀 에너지 (B0 갱신이면 정확히 같다) -- 절 AK.2 의 '재평균 때 사라지는 탄성 에너지' (Jensen 몫) 가 없다.
사면체당 스레드, 입자는 프레임마다 사면체 번호로 정렬한 목록 (offs/ord)."""
import cupy as cp
import numpy as np  # noqa: F401

from . import fem_c_cuda as CCU
from . import fem_ex_cuda as EX
from . import fem_rt_cuda as RTC

_AK = r'''
#include <cuda_fp16.h>
extern "C" {
// 빠른 FCR P (입자 스레드 판): Q = 빠른 극분해 (절 C c_polar_fast), J = |det X| (sp_fcr_P 의 Π max(σ, 0.01) 와 σ ≥ 0.01 에서 같다)
__device__ int AKPIT;                                                                // 극분해 반복 상한 (0 = c_polar_fast 그대로 -- 수렴 1e-7 까지 최대 10 회)
__device__ __forceinline__ void ak_polar_n(const float* F, float* Q, int nit) {
    for (int k=0;k<9;k++) Q[k]=F[k];
    float C[9];
    for (int it=0; it<nit; it++) {
        float dt = det3(Q);
        float ds = (fabsf(dt) > 1e-30f) ? dt : 1e-30f;
        float gm = rcbrtf(fabsf(ds));
        cof3(Q, C);
        for (int k=0;k<9;k++) Q[k] = 0.5f*(gm*Q[k] + C[k]/(ds*gm));
    }
}
__device__ __forceinline__ void ak_fcr_P(const float* X, float mu, float lam, float* P) {
    float Q[9], cf[9];
    if (AKPIT > 0) ak_polar_n(X, Q, AKPIT); else c_polar_fast(X, Q);
    cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    float J = fabsf(dt);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
}
__global__ void ak_pq_force(const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const int* offs, const int* ord,
                            const float* Fp, const float* wp, float mu, float lam, float* f) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0];
    int a = offs[t], b = offs[t+1];
    if (a >= b) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i] = tets[(long long)t*4+i];
    for (int k=0;k<12;k++) G[k] = gl[(long long)t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u = U[n[i]*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*G[i*3+q]; }
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    for (int k=a;k<b;k++) {
        float F[9]; for (int j=0;j<9;j++) F[j] = Fp[(long long)k*9+j];       // Fp·wp 는 사면체 정렬 순서로 담았다
        float X[9]; mm3(A, F, X);
        float P[9]; sp_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] += w*s; }
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*G[i*3+q];
        atomicAdd(&f[n[i]*3+c], -s);
    }
}
// 입자 스레드 판 (같은 식, 합 순서만 다름): 입자를 사면체 번호로 정렬 (ord, tt = 정렬된 사면체 번호) -> 워프 안 연속 구간 합 -> 구간 머리 레인이 노드 원자 합 12 개
__global__ void ak_pq_force_p(int N, const int* ord, const int* tt, const long long* tets, const float* U, const float* gl,
                              const float* Fp, const float* wp, float mu, float lam, float* f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    float G[12];
    if (on) {
        long long n[4];
        for (int i=0;i<4;i++) n[i] = tets[(long long)t*4+i];
        for (int q=0;q<12;q++) G[q] = gl[(long long)t*12+q];
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u = U[n[i]*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*G[i*3+q]; }
        float F[9]; for (int j=0;j<9;j++) F[j] = Fp[(long long)k*9+j];
        float X[9]; mm3(A, F, X);
        float P[9]; ak_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    if (!on || (lane > 0 && tp == t)) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*G[i*3+q]; atomicAdd(&f[nd*3+c], -s); } }
}
// 2 단 판 (같은 식): (1) 사면체 스레드가 A_T 를 버퍼에 (점유 목록 위), (2) 입자 스레드가 A 를 읽어 P(A F_p) F_pᵀ -> 워프 구간 합 -> 머리 레인이 노드 원자 합
__global__ void ak_tetA(const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, float* AT) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    for (int k=0;k<9;k++) AT[(long long)t*9+k] = A[k];
}
__global__ void __launch_bounds__(128) ak_pq_force_p2(int N, const int* tt, const long long* tets, const float* gl, const float* AT,
                               const float* Fp, const float* wp, float mu, float lam, float* f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    if (on) {
        float A[9], F[9];
        for (int j=0;j<9;j++) { A[j] = AT[(long long)t*9+j]; F[j] = Fp[(long long)k*9+j]; }
        float X[9]; mm3(A, F, X);
        float P[9]; ak_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    if (!on || (lane > 0 && tp == t)) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*gl[(long long)t*12+i*3+q]; atomicAdd(&f[nd*3+c], -s); } }
}
// 3 판 (같은 식): 워프 안 사면체 구간마다 머리 레인 하나만 노드 U·∇λ 를 읽어 A_T 를 만들고 셔플로 나눠 준다 (임의 접근 읽기 1/구간),
// 입자 레인은 F_p·w_p 만 읽는다 -> P(A F_p) F_pᵀ -> 구간 합 -> 머리 레인이 노드 원자 합
__global__ void __launch_bounds__(128) ak_pq_force_p3(int N, const int* tt, const long long* tets, const float* U, const float* gl,
                               const float* Fp, const float* wp, float mu, float lam, float* f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    bool head = (lane == 0) || (tp != t);
    unsigned hm = __ballot_sync(0xffffffffu, head);
    int hl = 31 - __clz(hm & (0xffffffffu >> (31 - lane)));                    // 내 구간 머리 레인
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    if (head && on) {
        for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
            for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    }
    for (int j=0;j<9;j++) A[j] = __shfl_sync(0xffffffffu, A[j], hl);
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    if (on) {
        float F[9]; for (int j=0;j<9;j++) F[j] = Fp[(long long)k*9+j];
        float X[9]; mm3(A, F, X);
        float P[9]; ak_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    if (!on || !head) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*gl[(long long)t*12+i*3+q]; atomicAdd(&f[nd*3+c], -s); } }
}
// 4 판 (3 판 + 입자 F 를 셀 F̄ 와의 차 δ = F_p − F̄_T 로 반정밀도 저장 -- 읽는 양 36 -> 18 B; F = F̄ + δ, δ 반올림 오차 ~|δ| 5e-4)
__global__ void __launch_bounds__(128) ak_pq_force_p4(int N, const int* tt, const long long* tets, const float* U, const float* gl, const float* FT,
                               const __half* dF, const float* wp, float mu, float lam, float* f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    bool head = (lane == 0) || (tp != t);
    unsigned hm = __ballot_sync(0xffffffffu, head);
    int hl = 31 - __clz(hm & (0xffffffffu >> (31 - lane)));
    float A[9] = {1,0,0, 0,1,0, 0,0,1}, Fb[9] = {1,0,0, 0,1,0, 0,0,1};
    if (head && on) {
        for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
            for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
        for (int j=0;j<9;j++) Fb[j] = FT[(long long)t*9+j];
    }
    for (int j=0;j<9;j++) { A[j] = __shfl_sync(0xffffffffu, A[j], hl); Fb[j] = __shfl_sync(0xffffffffu, Fb[j], hl); }
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    if (on) {
        float F[9]; for (int j=0;j<9;j++) F[j] = Fb[j] + __half2float(dF[(long long)k*9+j]);
        float X[9]; mm3(A, F, X);
        float P[9]; ak_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    if (!on || !head) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*gl[(long long)t*12+i*3+q]; atomicAdd(&f[nd*3+c], -s); } }
}
// 5 판 (혼합 적분점): 4 판과 같은 식을 '거친 사면체' (입자 2 개 이상 + 셀 안 max‖F_p − F̄‖ > ε) 의 입자 목록만 (개수는 장치에서), 나머지 사면체는 한 점 (F̄, Σw) 사면체 힘
__global__ void __launch_bounds__(128) ak_pq_force_p5(const int* Nd, const int* tt, const long long* tets, const float* U, const float* gl, const float* FT,
                               const __half* dF, const float* wp, float mu, float lam, float* f) {
    int N = Nd[0];
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (blockIdx.x*blockDim.x >= N) return;                                    // 블록 전체가 목록 밖 (워프 단위 셔플 전에 블록째 끝냄)
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    bool head = (lane == 0) || (tp != t);
    unsigned hm = __ballot_sync(0xffffffffu, head);
    int hl = 31 - __clz(hm & (0xffffffffu >> (31 - lane)));
    float A[9] = {1,0,0, 0,1,0, 0,0,1}, Fb[9] = {1,0,0, 0,1,0, 0,0,1};
    if (head && on) {
        for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
            for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
        for (int j=0;j<9;j++) Fb[j] = FT[(long long)t*9+j];
    }
    for (int j=0;j<9;j++) { A[j] = __shfl_sync(0xffffffffu, A[j], hl); Fb[j] = __shfl_sync(0xffffffffu, Fb[j], hl); }
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    if (on) {
        float F[9]; for (int j=0;j<9;j++) F[j] = Fb[j] + __half2float(dF[(long long)k*9+j]);
        float X[9]; mm3(A, F, X);
        float P[9]; ak_fcr_P(X, mu, lam, P);
        float w = wp[k];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    if (!on || !head) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*gl[(long long)t*12+i*3+q]; atomicAdd(&f[nd*3+c], -s); } }
}
// 클램프 끈 판 노드 걸음 + 이동 한 커널 (ah_node_step 과 c_lim_apply (α = 1) 를 이어 붙인 같은 식): 속도 갱신 (힘·중력·감쇠, 고착·비관통) -> U += h v -> 바닥 컬링 D 최대
__global__ void ak_node_step_mv(const int* szp, const float* mn, float* f, float* vn, float* Ucum, float g0, float g1, float g2, float hs, float damp,
                                const unsigned char* fixd, const float* vbc, float* acc, float* accN, int mode, int np_, float n0, float n1, float n2, float kex, float mfl,
                                unsigned int* alph, int* lst, unsigned int* dn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float gg[3] = {g0, g1, g2};
        float m = mn[n];
        float mi = fmaxf(m, mfl);
        float v[3];
        for (int c=0;c<3;c++) {
            if (fixd && fixd[n]) v[c] = vbc[n*3+c];
            else if (m > 0.f) v[c] = (vn[n*3+c] + hs*(f[n*3+c]/mi + gg[c]))*damp;
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
        float U[3];
        if (alph) alph[n] = 0x3f800000u;
        for (int c=0;c<3;c++) { vn[n*3+c] = v[c]; Ucum[n*3+c] += hs*v[c]; U[c] = Ucum[n*3+c]; }
        dl = fmaxf(-(U[0]*n0 + U[1]*n1 + U[2]*n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(0xffffffffu, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
}
'''
_MOD = [None]


def set_pit(n):
    """극분해 반복 상한 (장치 전역 AKPIT)."""
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + _AK).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    cp.ndarray((1,), cp.int32, _MOD[0].get_global("AKPIT"))[:] = int(n)


def launch(name, n, args, b=128, grid=None):
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + _AK).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        _MOD[0].get_function(name)((g,), (b,), args)
