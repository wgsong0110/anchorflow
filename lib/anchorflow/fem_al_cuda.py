"""절 AL (사용자 11-01): wolf (PG 모래, 재질 판) 명시적 FEM 서브스텝 가속 -- 결과 불변 (같은 식, 부동소수 반올림·원자 합 순서만 다름).

기준 서브스텝 (절 AG.19 공통 설정, 재질 씬): 커널 5 개
  ah_fc (접촉 몫만) -> akpl_force (입자 적분점 증분 RM, SVD 두 번) -> ah_node_step (국소 CFL 질량) -> ag_pbc (입자 구속 섞기) -> c_lim_apply (α = 1 이동, 컬링 D)
이 판 (--al_fuse): 커널 2 개
  al_fc   : 블록 [0, nbQ) = 입자 적분점 증분 RM + 힘 (모래 mat 2 전용 식), [nbQ, nbQ + nbP) = ah_fc 접촉 몫 (같은 식)
  al_step : ah_node_step (AHKD 없는 갈래) + ag_pbc + c_lim_apply (클램프 끔 = α 1) 를 이어 붙인 같은 식
모래 갈래에서 바꾼 계산 (수학적으로 같은 값):
  (1) 두 번째 SVD 생략: RM 뒤 F_e = U diag(σ') Vᵀ 이므로 응력 SVD 는 (U, σ', V) 그대로. τ = P F_eᵀ = U diag(P_i σ'_i) Uᵀ.
      PG kirchoff_stress_drucker_prager P_i = (2μ ε_i + λ tr ε)/|σ_i| 이므로 P_i σ_i = sign(σ_i)(2μ ε_i + λ tr ε), ε_i = ln max(|σ_i|, 1e-14).
      RM 이 바꾸지 않았으면 ε 는 RM 이 이미 계산한 값, 바꿨으면 ε'_i = ε_i − ê_i δγ/|ê| (= ln σ'_i), tr > 0 투영 (σ' = 1) 이면 τ = 0.
  (2) 힘 몫 w τ X⁻ᵀ F_pᵀ = w τ A⁻ᵀ (X = A F_p): X 역 대신 A 역 (cof A / det A).
  (3) 모래는 ys·logJp·손상 상태를 쓰지 않는다 (pl_rm mat 2, pl_pi mat 2) -> 정렬 순서 상태 읽기·쓰기 생략 (값은 프레임 시작 그대로라 akpl_fin 결과도 같다).
"""
from __future__ import annotations

import cupy as cp

from . import fem_ah_cuda as AHC
from . import fem_ak_cuda as AKC
from . import fem_akpl_cuda as AKPLC
from . import fem_c_cuda as CCU
from . import fem_ex_cuda as EX
from . import fem_pl_cuda as PLC
from . import fem_rt_cuda as RTC

_AL = r'''
extern "C" {
__device__ int ALNSW = 10;                                                         // --al_svd: Jacobi 순회 상한 (pl_svd 는 10; 수렴 판정 off ≤ 1e-14 dg 는 그대로)
__device__ __forceinline__ void al_svd(const float* X, float* U, float* s, float* V) {
    float A[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) A[i*3+k] = X[i]*X[k] + X[3+i]*X[3+k] + X[6+i]*X[6+k];
    for (int k=0;k<9;k++) V[k] = (k%4==0) ? 1.f : 0.f;
    for (int sw=0; sw<ALNSW; sw++) {
        float off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
        float dg = A[0]*A[0] + A[4]*A[4] + A[8]*A[8];
        if (off <= 1e-14f*dg || off < 1e-36f) break;
        for (int r=0;r<3;r++) {
            int p = (r==2) ? 1 : 0, q = (r==0) ? 1 : 2;
            float apq = A[p*3+q];
            if (fabsf(apq) < 1e-30f) continue;
            float th = (A[q*3+q] - A[p*3+p]) / (2.f*apq);
            float t = (th >= 0.f ? 1.f : -1.f) / (fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), sn = t*c;
            for (int k=0;k<3;k++) { float akp=A[k*3+p], akq=A[k*3+q]; A[k*3+p]=c*akp - sn*akq; A[k*3+q]=sn*akp + c*akq; }
            for (int k=0;k<3;k++) { float apk=A[p*3+k], aqk=A[q*3+k]; A[p*3+k]=c*apk - sn*aqk; A[q*3+k]=sn*apk + c*aqk; }
            for (int k=0;k<3;k++) { float vkp=V[k*3+p], vkq=V[k*3+q]; V[k*3+p]=c*vkp - sn*vkq; V[k*3+q]=sn*vkp + c*vkq; }
        }
    }
    // 내림차순 정렬 (열 교환)
    float l[3] = {A[0], A[4], A[8]};
    for (int i=0;i<2;i++) for (int j=0;j<2-i;j++) if (l[j] < l[j+1]) {
        float tl=l[j]; l[j]=l[j+1]; l[j+1]=tl;
        for (int k=0;k<3;k++) { float tv=V[k*3+j]; V[k*3+j]=V[k*3+j+1]; V[k*3+j+1]=tv; }
    }
    // V 를 회전으로 (det +1)
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) { V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8]; }
    float xv[3][3];
    for (int j=0;j<3;j++) for (int r=0;r<3;r++) xv[j][r] = X[r*3]*V[j] + X[r*3+1]*V[3+j] + X[r*3+2]*V[6+j];
    float s0 = sqrtf(fmaxf(l[0], 0.f));
    float u0[3], u1[3], u2[3];
    float n0 = sqrtf(xv[0][0]*xv[0][0] + xv[0][1]*xv[0][1] + xv[0][2]*xv[0][2]);
    if (n0 > 1e-20f) { for (int r=0;r<3;r++) u0[r] = xv[0][r]/n0; } else { u0[0]=1.f; u0[1]=0.f; u0[2]=0.f; }
    float d01 = u0[0]*xv[1][0] + u0[1]*xv[1][1] + u0[2]*xv[1][2];
    for (int r=0;r<3;r++) u1[r] = xv[1][r] - d01*u0[r];
    float n1 = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]);
    if (n1 > 1e-20f) { for (int r=0;r<3;r++) u1[r] /= n1; }
    else {   // u0 에 수직인 아무 단위 벡터
        float e[3] = {0.f, 0.f, 0.f}; int k = (fabsf(u0[0]) < 0.6f) ? 0 : ((fabsf(u0[1]) < 0.6f) ? 1 : 2); e[k] = 1.f;
        float d = u0[k]; for (int r=0;r<3;r++) u1[r] = e[r] - d*u0[r];
        float nn = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]); for (int r=0;r<3;r++) u1[r] /= nn;
    }
    u2[0] = u0[1]*u1[2] - u0[2]*u1[1]; u2[1] = u0[2]*u1[0] - u0[0]*u1[2]; u2[2] = u0[0]*u1[1] - u0[1]*u1[0];
    s[0] = n0 > 1e-20f ? n0 : s0;
    s[1] = u1[0]*xv[1][0] + u1[1]*xv[1][1] + u1[2]*xv[1][2];
    s[2] = u2[0]*xv[2][0] + u2[1]*xv[2][1] + u2[2]*xv[2][2];
    for (int r=0;r<3;r++) { U[r*3] = u0[r]; U[r*3+1] = u1[r]; U[r*3+2] = u2[r]; }
}
// 입자 적분점 증분 RM + 힘 (모래): akpl_force (mode 1, mat 2, 묶음·장벽·진단·시험 버퍼 없음) 과 같은 식
__device__ __forceinline__ void al_pq_sand(int k, int N, const int* tt, const long long* tets, const float* U, const float* gl,
                                           const float* akF, const float* wp, const float* prm, float* Fe, float* Xp, float* f) {
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
        float Xpi[9], Fpe[9], Xin[9];
        for (int j=0;j<9;j++) { Xpi[j] = Xp[(long long)k*9+j]; Fpe[j] = Fe[(long long)k*9+j]; }
        akpl_inv(Xpi, Xin);
        float D[9]; mm3(X, Xin, D);
        float Ft[9]; mm3(D, Fpe, Ft);
        float Uu[9], s[3], Vv[9]; al_svd(Ft, Uu, s, Vv);
        float mu = prm[0], lam = prm[1];
        // pl_rm mat 2 (PG sand_return_mapping) 그대로 + 응력 주값 τ_i (위 머리말 (1))
        float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fmaxf(fabsf(s[i]), 1e-14f));
        float tr = ep[0] + ep[1] + ep[2];
        float eh[3]; for (int i=0;i<3;i++) eh[i] = ep[i] - tr/3.f;
        float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
        float dg = ehn + (3.f*lam + 2.f*mu)/(2.f*mu)*tr*prm[3];
        float ti[3];
        float Fn[9];
        if (dg <= 0.f) {                                                        // 바뀌지 않음: F_e = F_trial
            for (int j=0;j<9;j++) Fn[j] = Ft[j];
            for (int i=0;i<3;i++) ti[i] = (2.f*mu*ep[i] + lam*tr)*(s[i]/fmaxf(fabsf(s[i]), 1e-14f));   // P_i σ_i (pl_pi mat 2 와 같은 |σ| 하한)
        } else if (tr > 0.f) {                                                  // 정점 투영 σ' = 1: 응력 0
            float o[3] = {1.f, 1.f, 1.f}; pl_usv(Uu, o, Vv, Fn);
            ti[0] = ti[1] = ti[2] = 0.f;
        } else {
            float sn[3], e2[3];
            for (int i=0;i<3;i++) { e2[i] = ep[i] - eh[i]*(dg/ehn); sn[i] = expf(e2[i]); }
            pl_usv(Uu, sn, Vv, Fn);
            float tr2 = e2[0] + e2[1] + e2[2];
            for (int i=0;i<3;i++) ti[i] = 2.f*mu*e2[i] + lam*tr2;
        }
        for (int j=0;j<9;j++) { Fe[(long long)k*9+j] = Fn[j]; Xp[(long long)k*9+j] = X[j]; }
        float tau[9]; pl_usv(Uu, ti, Uu, tau);                                  // τ = U diag(τ_i) Uᵀ
        float cA[9]; cof3(A, cA);
        float dA = A[0]*cA[0] + A[1]*cA[1] + A[2]*cA[2];
        float id = (fabsf(dA) > 1e-20f) ? w/dA : 0.f;                           // A⁻ᵀ = cof(A)/det A
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) Gm[r*3+c] = id*(tau[r*3]*cA[c] + tau[r*3+1]*cA[3+c] + tau[r*3+2]*cA[6+c]);
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
// ah_fc 의 접촉 몫 (같은 식)
__device__ __forceinline__ void al_contact(int q, const int* szp, const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const float* U,
                                           const long long* rows, const float* lamb, const float* m, int tetmass,
                                           float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon, int gtk, float band, float pn, float dxg,
                                           float* accN, float npd) {
    if (q >= szp[2]) return;
    if (!AFVN && s0c[q] > __uint_as_float(*dn) + 1e-5f + band) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (AFVN) { float vp = 0.f; for (int i=0;i<4;i++) vp += lm[i]*(AFVN[nd[i]*3]*n0 + AFVN[nd[i]*3+1]*n1 + AFVN[nd[i]*3+2]*n2); s += AFHS*vp; }
    if (ncon) {
        unsigned am = __activemask();
        unsigned bb = __ballot_sync(am, s < 0.f);
        if ((threadIdx.x & 31) == __ffs(am) - 1 && bb) atomicAdd(ncon, __popc(bb));
    }
    float w = 1.f;
    if (gtk) { if (!(s < band)) return; w = ah_wstk(s, pn, dxg); if (!(w > 0.f) && !(s < 0.f)) return; }
    else if (!(s < 0.f)) return;
    float mp = m[p];
    bool real = s < -npd;
    for (int i=0;i<4;i++) { if (w > 0.f) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp*w); if (real && accN) atomicAdd(&accN[nd[i]], 1.f); }
}
__global__ void __launch_bounds__(128) al_fc(int nbQ, int N, const int* tt, const long long* tets, const float* U, const float* gl,
                               const float* akF, const float* wp, const float* prm, float* Fe, float* Xp, float* f,
                               const int* szp, const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb,
                               const float* m, int tetmass, float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon,
                               int gtk, float band, float pn, float dxg, float* accN, float npd) {
    if ((int)blockIdx.x < nbQ) { al_pq_sand(blockIdx.x*blockDim.x + threadIdx.x, N, tt, tets, U, gl, akF, wp, prm, Fe, Xp, f); return; }
    al_contact((blockIdx.x - nbQ)*blockDim.x + threadIdx.x, szp, s0c, dn, nidx, x, U, rows, lamb, m, tetmass, p0, p1, p2, n0, n1, n2, acc, ncon, gtk, band, pn, dxg, accN, npd);
}
// 노드 걸음 (ah_node_step, AHKD 없는 갈래) + 입자 구속 섞기 (ag_pbc) + 이동 (c_lim_apply, α = 1) + 컬링 D -- 같은 식, 한 커널
__global__ void al_step(const int* szp, const float* mn, float* f, float* vn, float* Ucum, float g0, float g1, float g2, float hs, float damp,
                        const unsigned char* fixd, const float* vbc, float* acc, float* accN, int mode, int np_, float n0, float n1, float n2, float kex, float mfl,
                        const float* mfa, const float* pa, const float* pv, unsigned int* alph, int* lst, unsigned int* dn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float gg[3] = {g0, g1, g2};
        float m = mn[n];
        float mi = fmaxf(m, mfa ? mfa[n] : mfl);
        float v[3];
        for (int c=0;c<3;c++) {
            if (fixd && fixd[n]) v[c] = vbc[n*3+c];
            else if (m > 0.f || (mfa && mi > 0.f)) v[c] = (vn[n*3+c] + hs*(f[n*3+c]/mi + gg[c]))*damp;
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
                    ah_stick(v, k, n0, n1, n2);
                }
                if (np_ == 2) {
                    float fr = (m > 0.f) ? fminf(a/m, 1.f) : 1.f;
                    float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2;
                    if (a > 0.f && vn_ < 0.f) { v[0] -= fr*vn_*n0; v[1] -= fr*vn_*n1; v[2] -= fr*vn_*n2; }
                }
                else if (np_ && cn > 0.f) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
            }
        }
        if (pa) { float a = pa[n]; if (a > 0.f) for (int c=0;c<3;c++) v[c] = (1.f - a)*v[c] + pv[n*3+c]; }
        if (alph) alph[n] = 0x3f800000u;
        float U[3];
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


def _mod():
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + AHC._AH + AKC._AK + PLC._PL + AKPLC._AKPL + _AL).encode("ascii", "ignore").decode(),
                               options=("--std=c++14",))
    return _MOD[0]


def set_global(name, val, dtype):
    cp.ndarray((1,), dtype, _mod().get_global(name))[:] = val


def launch(name, n, args, b=128, grid=None):
    m_ = _mod()
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        m_.get_function(name)((g,), (b,), args)
