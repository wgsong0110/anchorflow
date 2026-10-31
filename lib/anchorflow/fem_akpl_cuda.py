"""절 AG.8 (조정자 10-31): 절 AK 입자 적분점 판 (--ak_pq) 에 재질 return mapping -- 절 AG 에서 옮긴 PG/GF 재질 식 (fem_pl_cuda 의 pl_svd·pl_rm·pl_pi·pl_nhb_tau) 그대로.
셀 에너지 E_T = Σ_{p∈T} w_p ψ(F_e,p) 꼴로, 입자마다 시험 F = A_T F_p (F_p = 프레임 시작 입자 F, 사면체 정렬 순서 akF·가중 akw).
  --ak_pq_plast inc   : 서브스텝 증분 RM -- 입자 상태 (F_e, X_prev, ys, logJp, 손상) 를 사면체 정렬 순서로 들고, F_trial = X X_prev⁻¹ F_e -> RM -> 응력 τ(F_e), 힘 = -w τ X⁻ᵀ F_pᵀ ∇λ.
                        프레임 끝 akpl_fin 이 마지막 U 로 한 번 더 증분 RM 해 입자 F·상태를 쓴다 (프레임 끝 ag_rm 대신).
  --ak_pq_plast frame : 서브스텝 힘은 투영 없이 재질 응력 P(X) (절 AG pl_cell none 과 같은 뜻), 프레임 끝 ag_rm (입자 RM) 그대로.
힘 합은 절 AK p 판처럼 워프 안 같은 사면체 구간 합 -> 구간 머리 레인이 노드 원자 합. 프레임당 P2G/G2P 왕복은 늘리지 않는다."""
from __future__ import annotations

import cupy as cp

from . import fem_ah_cuda as AHC
from . import fem_ak_cuda as AKC
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
__device__ __forceinline__ bool akpl_isf(const float* a, int n) { for (int i=0;i<n;i++) if (!isfinite(a[i])) return false; return true; }
// 절 AG.12 진단: 단계별 첫 비유한 개수 nanc[stage] (0 X, 1 X_prev 역, 2 F_trial, 3 SVD, 4 RM 결과, 5 응력 τ, 6 X 역 (나눗셈), 7 힘 몫, 8 det X ≤ 0 (비유한 아님, 개수))
#define AKNC(st, cond) do { if (nanc && !(cond)) { atomicAdd(&nanc[st], 1); bad = true; } } while (0)
// 절 AG.13 뒤집힘 장벽 (정의 변경): 입자 적분점마다 E_b = κ μ w b(J), J = det X (X = A F_p), Ĵ 아래에서만
//   b(J) = −(J − Ĵ)² log(J/Ĵ) (J_e ≤ J < Ĵ), J < J_e 는 값·기울기 연속 선형 연장 (기울기 b'(J_e) 상수 -- 강성 상한, J ≤ 0 에서도 계속 밀어냄)
//   ∂E_b/∂A = κ μ w b'(J) cof(X) F_pᵀ  (dJ/dX = cof X) -> Gm 에 더한다
__device__ __forceinline__ void akpl_bar(const float* X, const float* F, float w, float kmu, float jh, float je, float* Gm, int* nbar) {
    if (!(kmu > 0.f)) return;
    float J = det3(X);
    if (!(J < jh)) return;
    float Jc = (J > je) ? J : je;
    float d = Jc - jh;
    float db = -2.f*d*logf(Jc/jh) - d*d/Jc;                                     // b'(max(J, J_e))
    float cf[9]; cof3(X, cf);
    float s = kmu*w*db;
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float a_ = 0.f; for (int j=0;j<3;j++) a_ += cf[r*3+j]*F[c*3+j]; Gm[r*3+c] += s*a_; }
    if (nbar) atomicAdd(nbar, 1);
}
// 절 AG.16 (b) GF neo-Hookean Boarden 주값 응력 P_i (P = U diag(P_i) Vᵀ, τ = P Fᵀ 와 같은 식) + 작은 특이값 선형 연장:
//   P_i(σ) = (μ J^(-2/3) (σ_i² − Σσ²/3) + κ/2 (J² − 1)) / σ_i,  J = σ0 σ1 σ2.  σ_i < σ_e 이면 P_i(σ_i=σ_e) + ∂P_i/∂σ_i(σ_e) (σ_i − σ_e) (값·기울기 연속, 뒤집힘 (σ ≤ 0) 에도 정의, 접선 강성 상한)
__device__ __forceinline__ float akpl_nhbPi(const float* s, int i, float mu, float kap) {
    float J = s[0]*s[1]*s[2];
    float Jp = (J > 0.f) ? powf(J, -2.f/3.f) : 0.f;
    float b = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])/3.f;
    return (mu*Jp*(s[i]*s[i] - b) + 0.5f*kap*(J*J - 1.f))/s[i];
}
__device__ __forceinline__ void akpl_nhbP_ext(const float* s0, float mu, float kap, float se, float* Pi) {
    float s[3] = {s0[0], s0[1], s0[2]};
    bool ext[3];
    for (int i=0;i<3;i++) { ext[i] = s[i] < se; if (ext[i]) s[i] = se; }               // 연장점: 작은 특이값을 σ_e 로
    for (int i=0;i<3;i++) {
        Pi[i] = akpl_nhbPi(s, i, mu, kap);
        if (ext[i]) {
            float d = 1e-3f*se, sv = s[i];
            s[i] = sv + d; float a_ = akpl_nhbPi(s, i, mu, kap);
            s[i] = sv - d; float b_ = akpl_nhbPi(s, i, mu, kap);
            s[i] = sv;
            Pi[i] += (a_ - b_)/(2.f*d)*(s0[i] - se);
        }
    }
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
                               float* Fe, float* Xp, float* ysP, float* ljP, float* alP, float* f, float jfl,
                               const int* gp, const float* AG, const float* VG, float* GmG, float mu0, float lam0, int invm, int* nanc,
                               float bk, float bjh, float bje, int* nbar, float nse) {
    // 절 AG.16: nse > 0 이면 수박 (mat 7) 응력을 주값 + 작은 특이값 선형 연장 (akpl_nhbP_ext), 힘은 P 에서 바로 (X⁻ᵀ 안 씀)
    // 절 AG.12 진단: invm 1 = det X ≤ 0 인 입자의 응력 몫을 0 으로 (개입 시험), nanc != null 이면 단계별 비유한 개수
    // 절 AG.11 묶음 (gp != null): 희소 사면체 입자 (gp[k] ≥ 0) 는 묶음 평균 사상 Ā_g = AG[g]/VG[g] 로 X = Ā F, 응력 몫 w P Fᵀ 는 GmG[g] 에 (akg_dist 가 묶음 사면체로 나눔)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = k < N;
    int t = on ? tt[k] : -1 - lane;
    int g = (on && gp) ? gp[k] : -1;
    float Gm[9] = {0,0,0,0,0,0,0,0,0};
    float G[12];
    if (on) {
        long long n[4];
        for (int i=0;i<4;i++) n[i] = tets[(long long)t*4+i];
        for (int q=0;q<12;q++) G[q] = gl[(long long)t*12+q];
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        if (g >= 0) { float iv = 1.f/VG[g]; for (int j=0;j<9;j++) A[j] = AG[(long long)g*9+j]*iv; }
        else for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u = U[n[i]*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*G[i*3+q]; }
        float F[9]; for (int j=0;j<9;j++) F[j] = akF[(long long)k*9+j];
        float X[9]; mm3(A, F, X);
        float w = wp[k];
        float tau[9];
        bool bad = false;
        float dX = det3(X);
        if (nanc && !(dX > 0.f)) atomicAdd(&nanc[8], 1);
        AKNC(0, akpl_isf(X, 9));
        if (invm == 1 && !(dX > 0.f)) goto akpl_red;
        if (mode == 1) {                                                        // 증분 RM
            float Xpi[9], Fpe[9], Xin[9];
            for (int j=0;j<9;j++) { Xpi[j] = Xp[(long long)k*9+j]; Fpe[j] = Fe[(long long)k*9+j]; }
            akpl_inv(Xpi, Xin);
            if (!bad) AKNC(1, akpl_isf(Xin, 9));
            float D[9]; mm3(X, Xin, D);
            float Ft[9]; mm3(D, Fpe, Ft);
            if (!bad) AKNC(2, akpl_isf(Ft, 9));
            float ys = ysP[k], lj = ljP[k], al = alP[k];
            float Uu[9], s[3], Vv[9]; pl_svd(Ft, Uu, s, Vv);
            if (!bad) AKNC(3, akpl_isf(s, 3) && akpl_isf(Uu, 9) && akpl_isf(Vv, 9));
            bool ch = pl_rm(mat, s, prm, ys, lj, al, dt);
            float Fn[9]; if (ch) pl_usv(Uu, s, Vv, Fn); else for (int j=0;j<9;j++) Fn[j] = Ft[j];
            if (!bad) AKNC(4, akpl_isf(Fn, 9) && isfinite(lj) && isfinite(ys));
            for (int j=0;j<9;j++) { Fe[(long long)k*9+j] = Fn[j]; Xp[(long long)k*9+j] = X[j]; }
            ysP[k] = ys; ljP[k] = lj; alP[k] = al;
            if (mat == 7 && nse > 0.f) {                                        // P(F_e) Gᵀ Fᵀ, G = X_prev⁻¹ F_e,prev (F_trial = X G)
                float U2[9], s2[3], V2[9]; pl_svd(Fn, U2, s2, V2); float Pi[3]; akpl_nhbP_ext(s2, prm[0], prm[2], nse, Pi);
                float P_[9]; pl_usv(U2, Pi, V2, P_);
                float Gt[9]; mm3(Xin, Fpe, Gt);
                float PG[9]; mmt3(P_, Gt, PG);
                for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s_ = 0.f; for (int j=0;j<3;j++) s_ += PG[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s_; }
                akpl_bar(X, F, w, bk*prm[0], bjh, bje, Gm, nbar);
                goto akpl_red;
            }
            if (mat == 7) akpl_nhb_tau(Fn, prm[0], prm[2], jfl, tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(Fn, U2, s2, V2); float Pi[3]; pl_pi(mat, s2, prm, al, Pi); float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, Fn, tau); }
        } else if (mat == 0) {                                                 // 탄성 (절 AK 와 같은 FCR, 빠른 극분해): P 를 바로
            float P[9]; ak_fcr_P(X, mu0, lam0, P);
            for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s_ = 0.f; for (int j=0;j<3;j++) s_ += P[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s_; }
            akpl_bar(X, F, w, bk*mu0, bjh, bje, Gm, nbar);
            goto akpl_red;
        } else {                                                                // 투영 없는 재질 응력 (프레임 끝 입자 RM 만)
            float al = alPart ? alPart[ord[k]] : 1.f;
            if (mat == 7 && nse > 0.f) {                                        // P(X) Fᵀ 바로
                float U2[9], s2[3], V2[9]; pl_svd(X, U2, s2, V2); float Pi[3]; akpl_nhbP_ext(s2, prm[0], prm[2], nse, Pi);
                float P_[9]; pl_usv(U2, Pi, V2, P_);
                for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s_ = 0.f; for (int j=0;j<3;j++) s_ += P_[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s_; }
                akpl_bar(X, F, w, bk*prm[0], bjh, bje, Gm, nbar);
                goto akpl_red;
            }
            if (mat == 7) akpl_nhb_tau(X, prm[0], prm[2], jfl, tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(X, U2, s2, V2); float Pi[3]; pl_pi(mat, s2, prm, al, Pi); float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, X, tau); }
        }
        if (!bad) AKNC(5, akpl_isf(tau, 9));
        float XiT[9]; { float Xi[9]; akpl_inv(X, Xi); for (int i=0;i<3;i++) for (int j=0;j<3;j++) XiT[i*3+j] = Xi[j*3+i]; }
        if (!bad) AKNC(6, akpl_isf(XiT, 9));
        float PX[9]; mm3(tau, XiT, PX);                                          // P_X = τ X⁻ᵀ
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { float s_ = 0.f; for (int j=0;j<3;j++) s_ += PX[r*3+j]*F[c*3+j]; Gm[r*3+c] = w*s_; }
        if (!bad) AKNC(7, akpl_isf(Gm, 9));
        akpl_bar(X, F, w, bk*prm[0], bjh, bje, Gm, nbar);
    }
akpl_red:
    if (g >= 0) {                                                               // 묶음 몫: 원자 합으로 GmG, 사면체 구간 합에서는 뺀다
        for (int j=0;j<9;j++) { atomicAdd(&GmG[(long long)g*9+j], Gm[j]); Gm[j] = 0.f; }
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
// ---- 절 AG.11 묶음 (희소 사면체를 같은 칸의 점유 사면체들과 하나의 적분 단위로) ----
// 묶음 사상 합: AG[g] += V_t A_t (묶음 g = tg[t] ≥ 0 인 점유 사면체)
__global__ void akg_A(const int* tl, const int* tlc, const int* tg, const long long* tets, const float* U, const float* gl, const float* VT, float* AG) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0]; int g = tg[t];
    if (g < 0) return;
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    float V = VT[t];
    for (int j=0;j<9;j++) atomicAdd(&AG[(long long)g*9+j], V*A[j]);
}
// 묶음 몫 나누기: 사면체 t ∈ g 에 (V_t/V_g) GmG[g] + 모래시계 억제 2 γ μ V_t (A_t − Ā_g) -> 노드 힘 −Gm ∇λ_i
__global__ void akg_dist(const int* tl, const int* tlc, const int* tg, const long long* tets, const float* U, const float* gl, const float* VT,
                         const float* AG, const float* VG, const float* GmG, float hg, float* f) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0]; int g = tg[t];
    if (g < 0) return;
    float V = VT[t], iv = 1.f/VG[g], c = V*iv;
    float Gm[9]; for (int j=0;j<9;j++) Gm[j] = c*GmG[(long long)g*9+j];
    if (hg > 0.f) {
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int cc=0;cc<3;cc++) { float u = U[nd*3+cc]; for (int q=0;q<3;q++) A[cc*3+q] += u*gl[(long long)t*12+i*3+q]; } }
        for (int j=0;j<9;j++) Gm[j] += 2.f*hg*V*(A[j] - AG[(long long)g*9+j]*iv);
    }
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i];
        for (int cc=0;cc<3;cc++) { float s_ = 0.f; for (int q=0;q<3;q++) s_ += Gm[cc*3+q]*gl[(long long)t*12+i*3+q]; atomicAdd(&f[nd*3+cc], -s_); } }
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
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + AHC._AH + AKC._AK + PLC._PL + _AKPL).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        _MOD[0].get_function(name)((g,), (b,), args)
