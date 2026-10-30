"""절 AK: 입자 적분점 셀 탄성 (--ak_pq) -- 셀 에너지 E_T = Σ_{p∈T} w_p ψ(A_T F_p) (F_p = 프레임 시작 입자 F, w_p = V_p · V_T'/V_T, 상한 뒤 부피 비 그대로).
힘 f_i = -Σ_p w_p P(A_T F_p) F_pᵀ ∇λ_i (= -∂E/∂x_i, 에너지의 기울기 -> 운동량·각운동량 보존). 셀 평균 F̄ 의 ψ(A F̄) 대신 입자마다 ψ 를 더하므로
프레임 끝 셀 에너지 = 다음 프레임 시작 셀 에너지 (B0 갱신이면 정확히 같다) -- 절 AK.2 의 '재평균 때 사라지는 탄성 에너지' (Jensen 몫) 가 없다.
사면체당 스레드, 입자는 프레임마다 사면체 번호로 정렬한 목록 (offs/ord)."""
import cupy as cp
import numpy as np  # noqa: F401

from . import fem_ex_cuda as EX
from . import fem_rt_cuda as RTC

_AK = r'''
extern "C" {
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
        int p = ord[k];
        float F[9]; for (int j=0;j<9;j++) F[j] = Fp[(long long)p*9+j];
        float X[9]; mm3(A, F, X);
        float P[9]; sp_fcr_P(X, mu, lam, P);
        float w = wp[p];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s = 0.f; for (int j=0;j<3;j++) s += P[r*3+j]*F[c*3+j]; Gm[r*3+c] += w*s; }
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int q=0;q<3;q++) s += Gm[c*3+q]*G[i*3+q];
        atomicAdd(&f[n[i]*3+c], -s);
    }
}
}
'''
_MOD = [None]


def launch(name, n, args, b=128, grid=None):
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + _AK).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        _MOD[0].get_function(name)((g,), (b,), args)
