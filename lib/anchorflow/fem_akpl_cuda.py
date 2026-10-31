"""절 AG.8 (조정자 10-31): 절 AK 입자 적분점 판 (--ak_pq) 에 재질 return mapping -- 절 AG 에서 옮긴 PG/GF 재질 식 (fem_pl_cuda 의 pl_svd·pl_rm·pl_pi·pl_nhb_tau) 그대로.
셀 에너지 E_T = Σ_{p∈T} w_p ψ(F_e,p) 꼴로, 입자마다 시험 F = A_T F_p (F_p = 프레임 시작 입자 F, 사면체 정렬 순서 akF·가중 akw).
  --ak_pq_plast inc   : 서브스텝 증분 RM -- 입자 상태 (F_e, X_prev, ys, logJp, 손상) 를 사면체 정렬 순서로 들고, F_trial = X X_prev⁻¹ F_e -> RM -> 응력 τ(F_e), 힘 = -w τ X⁻ᵀ F_pᵀ ∇λ.
                        프레임 끝 akpl_fin 이 마지막 U 로 한 번 더 증분 RM 해 입자 F·상태를 쓴다 (프레임 끝 ag_rm 대신).
  --ak_pq_plast frame : 서브스텝 힘은 투영 없이 재질 응력 P(X) (절 AG pl_cell none 과 같은 뜻), 프레임 끝 ag_rm (입자 RM) 그대로.
힘 합은 절 AK p 판처럼 워프 안 같은 사면체 구간 합 -> 구간 머리 레인이 노드 원자 합. 프레임당 P2G/G2P 왕복은 늘리지 않는다."""
from __future__ import annotations

import cupy as cp

from . import fem_c_cuda as CCU
from . import fem_ex_cuda as EX
from . import fem_pl_cuda as PLC
from . import fem_rt_cuda as RTC

_AKPL = r'''
extern "C" {
__device__ __forceinline__ void akpl_inv(const float* X, float* Xi) {          // X⁻¹ = cof(X)ᵀ / det
    float c[9]; cof3(X, c); float d = det3(X); float id = (fabsf(d) > 1e-20f) ? 1.f/d : 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Xi[i*3+j] = c[j*3+i]*id;
}
// GF neo-Hookean Boarden τ (pl_nhb_tau 와 같은 식) + J 하한 jfl (> 0 이면 J = max(det F, jfl): 뒤집힌 셀에서 J^(-2/3) 가 비유한이 되는 것을 막는 정의 변경)
__device__ __forceinline__ void akpl_nhb_tau(const float* Fe, float mu, float kap, float jfl, float* tau) {
    float B[9]; mmt3(Fe, Fe, B);
    float J = det3(Fe); if (jfl > 0.f) J = fmaxf(J, jfl);
    float tb = (B[0] + B[4] + B[8])/3.f;
    float sd = mu*powf(J, -2.f/3.f), sv = 0.5f*kap*(J*J - 1.f);
    for (int k=0;k<9;k++) tau[k] = sd*(B[k] - ((k%4==0) ? tb : 0.f)) + ((k%4==0) ? sv : 0.f);
}
// 그래프 첫 커널: 정렬 순서 입자 상태 = 프레임 시작 (F_e = X_prev = F_p, 상태 = 입자 것)
__global__ void akpl_init(int N, const int* ord, const float* akF, const float* ys, const float* lj, const float* al,
                          float* Fe, float* Xp, float* ysP, float* ljP, float* alP) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N) return;
    int p = ord[k];
    for (int j=0;j<9;j++) { float v = akF[(long long)k*9+j]; Fe[(long long)k*9+j] = v; Xp[(long long)k*9+j] = v; }
    ysP[k] = ys ? ys[p] : 0.f; ljP[k] = lj ? lj[p] : 0.f; alP[k] = al ? al[p] : 1.f;
}
__global__ void __launch_bounds__(128) akpl_force(int N, int mode, const int* ord, const int* tt, const long long* tets, const float* U, const float* gl,
                               const float* akF, const float* wp, int mat, const float* prm, float dt, const float* alPart,
                               float* Fe, float* Xp, float* ysP, float* ljP, float* alP, float* f, float jfl) {
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
        float F[9]; for (int j=0;j<9;j++) F[j] = akF[(long long)k*9+j];
        float X[9]; mm3(A, F, X);
        float w = wp[k];
        float tau[9];
        if (mode == 1) {                                                        // 증분 RM
            float Xpi[9], Fpe[9], Xin[9];
            for (int j=0;j<9;j++) { Xpi[j] = Xp[(long long)k*9+j]; Fpe[j] = Fe[(long long)k*9+j]; }
            akpl_inv(Xpi, Xin);
            float D[9]; mm3(X, Xin, D);
            float Ft[9]; mm3(D, Fpe, Ft);
            float ys = ysP[k], lj = ljP[k], al = alP[k];
            float Uu[9], s[3], Vv[9]; pl_svd(Ft, Uu, s, Vv);
            bool ch = pl_rm(mat, s, prm, ys, lj, al, dt);
            float Fn[9]; if (ch) pl_usv(Uu, s, Vv, Fn); else for (int j=0;j<9;j++) Fn[j] = Ft[j];
            for (int j=0;j<9;j++) { Fe[(long long)k*9+j] = Fn[j]; Xp[(long long)k*9+j] = X[j]; }
            ysP[k] = ys; ljP[k] = lj; alP[k] = al;
            if (mat == 7) akpl_nhb_tau(Fn, prm[0], prm[2], jfl, tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(Fn, U2, s2, V2); float Pi[3]; pl_pi(mat, s2, prm, al, Pi); float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, Fn, tau); }
        } else {                                                                // 투영 없는 재질 응력 (프레임 끝 입자 RM 만)
            float al = alPart ? alPart[ord[k]] : 1.f;
            if (mat == 7) akpl_nhb_tau(X, prm[0], prm[2], jfl, tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(X, U2, s2, V2); float Pi[3]; pl_pi(mat, s2, prm, al, Pi); float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, X, tau); }
        }
        float XiT[9]; { float Xi[9]; akpl_inv(X, Xi); for (int i=0;i<3;i++) for (int j=0;j<3;j++) XiT[i*3+j] = Xi[j*3+i]; }
        float PX[9]; mm3(tau, XiT, PX);                                          // P_X = τ X⁻ᵀ
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s_ = 0.f; for (int j=0;j<3;j++) s_ += PX[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s_; }
    }
    for (int off=1; off<32; off<<=1) {
        int to = __shfl_down_sync(0xffffffffu, t, off);
        bool same = (lane + off < 32) && (to == t);
        for (int j=0;j<9;j++) { float o = __shfl_down_sync(0xffffffffu, Gm[j], off); if (same) Gm[j] += o; }
    }
    int tp = __shfl_up_sync(0xffffffffu, t, 1);
    if (!on || (lane > 0 && tp == t)) return;
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int c=0;c<3;c++) { float s_ = 0.f; for (int q=0;q<3;q++) s_ += Gm[c*3+q]*G[i*3+q]; atomicAdd(&f[nd*3+c], -s_); } }
}
// 프레임 끝 (G2P 뒤, inc 판): 마지막 U 로 증분 RM 한 번 더 -> 입자 F·ys·logJp·손상 (원래 입자 번호로)
__global__ void akpl_fin(int N, const int* ord, const int* tt, const long long* tets, const float* U, const float* gl, const float* akF,
                         const float* Fe, const float* Xp, const float* ysP, const float* ljP, const float* alP, int mat, const float* prm, float dt,
                         float* Fout, float* ys, float* lj, float* al, float* rmc) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N) return;
    int t = tt[k], p = ord[k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    float F[9]; for (int j=0;j<9;j++) F[j] = akF[(long long)k*9+j];
    float X[9]; mm3(A, F, X);
    float Xpi[9], Fpe[9], Xin[9];
    for (int j=0;j<9;j++) { Xpi[j] = Xp[(long long)k*9+j]; Fpe[j] = Fe[(long long)k*9+j]; }
    akpl_inv(Xpi, Xin);
    float D[9]; mm3(X, Xin, D);
    float Ft[9]; mm3(D, Fpe, Ft);
    float y_ = ysP[k], l_ = ljP[k], a_ = alP[k], a0 = al ? al[p] : 1.f;
    float Uu[9], s[3], Vv[9]; pl_svd(Ft, Uu, s, Vv);
    bool ch = pl_rm(mat, s, prm, y_, l_, a_, dt);
    float Fn[9]; if (ch) pl_usv(Uu, s, Vv, Fn); else for (int j=0;j<9;j++) Fn[j] = Ft[j];
    for (int j=0;j<9;j++) Fout[(long long)p*9+j] = Fn[j];
    if (ys) ys[p] = y_;
    if (lj) lj[p] = l_;
    if (al) al[p] = a_;
    if (rmc && ch && mat != 7) atomicAdd(&rmc[0], 1.f);
    if (rmc && a0 > 0.f && !(a_ > 0.f)) atomicAdd(&rmc[1], 1.f);
}
}
'''
_MOD = [None]


def launch(name, n, args, b=128, grid=None):
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + PLC._PL + _AKPL).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        _MOD[0].get_function(name)((g,), (b,), args)
