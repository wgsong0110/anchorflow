"""절 C (갈래 C, 10-30 사용자 "물어보지 말고 해볼 수 있는 거 다 해봐"): 특이값 비율 클램프 판 (절 Z) 의 속도 최적화 커널 -- 결과 불변 우선.

기존 lag 판 서브스텝 (절 R·Z: ex_tet_force_R -> ex_vfl_contact -> ex_node_step_lag -> ex_lim_sr_tet -> ex_lim_box_apply) 과 같은 식을
다음처럼 줄인다 (fem_ex_cuda 의 커널은 그대로 두고 이 모듈은 별도 컴파일 -- 같은 장치 함수 RTC._SRC + EX._EX 뒤에 붙인다):

(1) 바닥 접촉 판정 걸러 내기 (--vfl_cull): 바닥 후보 (프레임 시작에 관성 예측 끝 높이 ≤ 0.2 m, 프레임당 6~17 만 입자) 중
    서브스텝마다 실제 접촉은 수십 개뿐이다. 접촉 판정 s = (x0 + Σλ_i U_i - p)·n 에서 Σλ_i = 1, λ_i ≥ 0 이므로
    s ≥ s0 - D,  s0 = (x0 - p)·n,  D = max_n max(0, -U_n·n)  (프레임 안 노드 아래쪽 누적 변위의 최댓값, 단조 증가로 유지)
    -> s0 > D + 1e-5 m 인 후보는 접촉일 수 없다 (판정 생략). 남은 후보는 기존과 같은 식·같은 자료로 판정 (같은 접촉 집합).
    s0 는 프레임마다 한 번 (c_s0), D 는 노드 이동 커널 (c_lim_apply) 이 워프 최댓값 -> 원자 max 로.
(2) 클램프 사면체 판정 둘로 (--sr_fast):
    c_sr_pre (사면체당 스레드): 상자 [0,1]⁴ 의 모든 배치 B = I + E + P (E = A0 - I, ‖P‖₂ ≤ ρ = Σ|d_i||∇λ_i|) 에서
      δ = ‖E‖_F + ρ < 1 이면 σ1(B) ≤ 1 + δ, σ3(B) ≥ 1 - δ, det B ≥ (1 - δ)³ > 0  ->  δ < (c-1)/(c+1) 이면 비율 ≤ c ≤ max(c, r0) 보장
      (+ det 하한 켬이면 (1-δ)³ ≥ Ĵ 도) -> 기존 커널도 제한하지 않는 사면체 (꼭짓점 판정이 통과) 라 건너뛴다. 아닌 사면체만 목록에.
    c_sr_warp (목록 사면체당 워프 하나): 기존 ex_lim_sr_tet 과 **같은 순서·같은 식** (det 꼭짓점 -> Weyl -> s = 1 꼭짓점 -> 이분 14 번) 을
      꼭짓점 15 개를 레인 15 개에 나눠 (__all_sync) 계산 -> 같은 s_T (비트 단위), 원자 min 은 순서 무관.
(3) 점유 사면체 목록 (--tlist): 프레임마다 V_T > 0 사면체를 압축 (c_tet_list) -> 힘·클램프 판정 커널이 빈 사면체 레인을 돌지 않는다.
(4) 극분해 (--fpolar): 힘 커널의 뉴턴 극분해 10 회 (pow) 를 rcbrtf + 수렴 (max|ΔQ| < 1e-7) 조기 종료로 -- 값은 float 반올림 수준만 다르다.
(5) 융합 (--cfuse): 서브스텝 커널 6 개 (힘·접촉·노드 걸음·판정 둘·적용) -> 4 개 (힘+접촉, 판정 둘 (속도 그 자리 계산), 걸음+적용). 같은 식.
"""
from __future__ import annotations

import numpy as np

from . import fem_rt_cuda as RTC
from . import fem_ex_cuda as EX

_C = r'''
#define CFULL 0xffffffffu
extern "C" {
// ---- (3) 점유 사면체 목록 (프레임마다, tlc 는 미리 0) ----
__global__ void c_tet_list(const int* szp, const float* VT, int* tl, int* tlc) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool o = (t < T) && (VT[t] > 0.f);
    unsigned b = __ballot_sync(CFULL, o);
    int base = 0;
    if (lane == 0 && b) base = atomicAdd(tlc, __popc(b));
    base = __shfl_sync(CFULL, base, 0);
    if (o) tl[base + __popc(b & ((1u << lane) - 1u))] = t;
}
// ---- (1) 바닥 후보 시작 높이 s0 (프레임마다) ----
__global__ void c_s0(const int* szp, const long long* nidx, const float* x, float p0, float p1, float p2, float n0, float n1, float n2, float* s0c) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    long long p = nidx[q];
    s0c[q] = (x[p*3]-p0)*n0 + (x[p*3+1]-p1)*n1 + (x[p*3+2]-p2)*n2;
}
// (1) 걸러 낸 접촉 판정: 통과한 후보는 ex_vfl_contact 와 같은 식
__global__ void c_vfl_contact(const int* szp, const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const float* Ucum,
                              const long long* rows, const float* lamb, const float* m, int tetmass, float p0, float p1, float p2, float n0, float n1, float n2,
                              float* acc, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*Ucum[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// ---- 상자 판 적용 (ex_lim_box_apply 그대로) + 목록 개수 비움 + (1) 의 D 갱신 (dn != null) ----
__global__ void c_lim_apply(const int* szp, const float* mn, float* vn, float* Ucum, float hs, unsigned int* alph, int vmode, const unsigned char* fixd, float* rst,
                            int* lst, unsigned int* dn, float n0, float n1, float n2) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float al = __uint_as_float(alph[n]);
        alph[n] = 0x3f800000u;
        if (fixd && fixd[n]) al = 1.f;
        float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
        float U[3];
        for (int c=0;c<3;c++) { Ucum[n*3+c] += al*hs*v[c]; U[c] = Ucum[n*3+c]; }
        if (al < 1.f) {
            float m = mn[n], w = 1.f - al;
            atomicAdd(&rst[1], 1.f);
            if (vmode == 0) {
                for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
                float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
                atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
                atomicAdd(&rst[6], w*m*sqrtf(v2));
            }
        }
        dl = fmaxf(-(U[0]*n0 + U[1]*n1 + U[2]*n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(CFULL, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
// ---- (2) 클램프 판정 ----
__device__ __forceinline__ void c_sr_load(int t, const long long* tets, const float* gl, const float* vn, const float* Ucum, float hs,
                                          long long* n, float* A0, float* Mo, float& rho) {
    float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    for (int k=0;k<9;k++) A0[k] = (k % 4 == 0) ? 1.f : 0.f;
    rho = 0.f;
    for (int i=0;i<4;i++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
        for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
        rho += sqrtf(dd*gg);
    }
}
// 빠른 보장 판정 (사면체당 스레드): 보장되면 아무것도 안 함, 아니면 lst 에 넣는다. tl != null 이면 점유 목록 (i < tlc[0]) 위로.
__global__ void c_sr_pre(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum,
                         float hs, float csr, int jon, float jmin, int* lst, const float* eF) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i >= tlc[0]) return; t = tl[i]; }
    else { if (i >= szp[1] || !(VT[i] > 0.f)) return; t = i; }
    if (eF) {   // (6) ‖A0 - I‖_F² 는 같은 서브스텝 힘 커널이 같은 U 로 남긴 값 -- U 를 다시 읽지 않는다 (판정 보수성은 같다)
        float rho = 0.f;
        for (int a=0;a<4;a++) {
            long long na = tets[t*4+a];
            float dd = 0.f, gg = 0.f;
            for (int c=0;c<3;c++) { float d=hs*vn[na*3+c]; dd += d*d; float g_=gl[t*12+a*3+c]; gg += g_*g_; }
            rho += sqrtf(dd*gg);
        }
        float del = sqrtf(eF[t]) + rho;
        bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
        if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
        if (fast) return;
        int slot = atomicAdd(&lst[0], 1);
        lst[1 + slot] = t;
        return;
    }
    long long n[4]; float G[12];
    for (int k=0;k<4;k++) n[k]=tets[t*4+k];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float rho = 0.f;
    for (int a=0;a<4;a++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[a]*3+c], d=hs*vn[n[a]*3+c]; dd += d*d; for (int b=0;b<3;b++) A0[c*3+b] += u*G[a*3+b]; }
        for (int b=0;b<3;b++) gg += G[a*3+b]*G[a*3+b];
        rho += sqrtf(dd*gg);
    }
    float e2 = 0.f;
    for (int k=0;k<9;k++) { float d = A0[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; }
    float del = sqrtf(e2) + rho;
    bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
    if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
    if (fast) return;
    int slot = atomicAdd(&lst[0], 1);
    lst[1 + slot] = t;
}
__device__ __forceinline__ bool c_feasw(int lane, const float* A0, const float* Mo, float s, float thrR, int jon, float thrJ) {
    bool o = true;
    if (lane < 15) {
        int S = lane + 1;
        float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
        if (jon && det3(B) < thrJ) o = false;
        else if (sr_ratio(B) > thrR) o = false;
    }
    return __all_sync(CFULL, o);
}
// 목록 사면체당 워프 하나 -- ex_lim_sr_tet 과 같은 순서·식 (꼭짓점 15 개를 레인에), 격자 크기 고정 + 워프 단위 반복
__global__ void c_sr_warp(const int* lst, const long long* tets, const float* gl, const float* vn, const float* Ucum, float hs,
                          float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int cnt = lst[0];
    int lane = threadIdx.x & 31;
    int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int nw = (gridDim.x*blockDim.x) >> 5;
    for (int q = gw; q < cnt; q += nw) {
        int t = lst[1 + q];
        long long n[4]; float A0[9], Mo[36], rho;
        c_sr_load(t, tets, gl, vn, Ucum, hs, n, A0, Mo, rho);
        float D0 = det3(A0);
        float r0 = sr_ratio(A0);
        float thrR = fmaxf(csr, r0);
        float thrJ = fminf(jmin, D0);
        bool ok = true;
        if (jon) { bool o = (lane >= 15) || !(det_sub(A0, Mo, lane + 1, 1.f) < thrJ); ok = __all_sync(CFULL, o); }
        if (ok && D0 > 0.f) {
            float C[9];
            for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
            float l[3]; symeig3(C, l);
            float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
            if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) continue;
        }
        if (c_feasw(lane, A0, Mo, 1.f, thrR, jon, thrJ)) continue;
        float lo = 0.f, hi = 1.f;
        for (int it=0; it<14; it++) {
            float mid = 0.5f*(lo+hi);
            if (c_feasw(lane, A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
        }
        if (lane == 0) {
            unsigned int bits = __float_as_uint(lo);
            for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
            atomicAdd(&rst[7], 1.f);
        }
    }
}
// ---- 힘: ex_tet_force_R 과 같은 식 (+ 점유 목록, (4) 빠른 극분해 선택) ----
__device__ __forceinline__ void c_polar_fast(const float* F, float* Q) {
    for (int k=0;k<9;k++) Q[k]=F[k];
    float C[9];
    for (int it=0; it<10; it++) {
        float dt = det3(Q);
        float ds = (fabsf(dt) > 1e-30f) ? dt : 1e-30f;
        float gm = rcbrtf(fabsf(ds));
        cof3(Q, C);
        float mx = 0.f;
        for (int k=0;k<9;k++) { float qn = 0.5f*(gm*Q[k] + C[k]/(ds*gm)); mx = fmaxf(mx, fabsf(qn - Q[k])); Q[k] = qn; }
        if (mx < 1e-7f) break;
    }
}
__device__ __forceinline__ void c_force_body(int t, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                             float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, int fpol, float* eF) {
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (eF) { float e2 = 0.f; for (int k=0;k<9;k++) { float d = A[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; } eF[t] = e2; }   // (6) 클램프 빠른 판정용 ‖A - I‖_F²
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    if (fpol) c_polar_fast(X, Q); else polar(X, Q);
    cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
__global__ void c_tet_force_R(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                              float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, int fpol, float* eF) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
    else { if (i0 >= szp[1]) return; t = i0; }
    c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, fbk, fbj, fbon, rst, fpol, eF);
}

// ======== (5) 융합 판 (--cfuse): 서브스텝 커널 6 개 -> 4 개, 같은 식 ========
//   K1 c_fc      : [사면체 블록] c_tet_force_R 와 같은 힘 + [바닥 후보 블록] c_vfl_contact 와 같은 접촉 판정 (둘 다 노드 배열에 원자 합만 -- 서로 독립)
//   K2 c_sr_pre2 : c_sr_pre 와 같은 판정, 노드 속도는 ex_node_step_lag 와 같은 식으로 그 자리에서 (c_vnew)
//   K3 c_sr_warp2: c_sr_warp 와 같은 판정, 속도는 c_vnew
//   K4 c_apply2  : ex_node_step_lag (속도 갱신·바닥 고착·비관통, f·gn·acc 비움) + c_lim_apply (α, U, 기록, 목록 비움, D) 를 한 번에
// ex_node_step_lag 와 같은 식 (nodisp = 1 판) -- 쓰지 않고 값만
__device__ __forceinline__ void c_vnew(long long n, const float* mn, const float* f, const float* gn, const float* vn, float g0, float g1, float g2, float hs, float damp,
                                       const unsigned char* fixd, const float* vbc, const float* acc, int mode, int np_, float n0, float n1, float n2, float* v) {
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f) v[c] = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v[c] = 0.f;
    }
    if (acc) {
        float a = acc[n];
        if (a > 0.f && !(fixd && fixd[n])) {
            float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f));
            for (int c=0;c<3;c++) v[c] *= k;
            if (np_) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
        }
    }
}
struct CVS { const float* mn; const float* f; const float* gn; const float* vn; float g0, g1, g2, hs, damp; const unsigned char* fixd; const float* vbc; const float* acc; int mode, np_; float n0, n1, n2; };
__device__ __forceinline__ void c_vget(const CVS& S, long long n, float* v) {
    c_vnew(n, S.mn, S.f, S.gn, S.vn, S.g0, S.g1, S.g2, S.hs, S.damp, S.fixd, S.vbc, S.acc, S.mode, S.np_, S.n0, S.n1, S.n2, v);
}
__global__ void c_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                     float mu, float lam, float* f, float* rst, int fpol, float* eF,
                     const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                     float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
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
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
__global__ void c_sr_pre2(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, const float* Ucum, CVS S,
                          float csr, int jon, float jmin, int* lst) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i >= tlc[0]) return; t = tl[i]; }
    else { if (i >= szp[1] || !(VT[i] > 0.f)) return; t = i; }
    long long n[4]; float G[12];
    for (int k=0;k<4;k++) n[k]=tets[t*4+k];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float rho = 0.f;
    for (int a=0;a<4;a++) {
        float vv[3]; c_vget(S, n[a], vv);
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[a]*3+c], d=S.hs*vv[c]; dd += d*d; for (int b=0;b<3;b++) A0[c*3+b] += u*G[a*3+b]; }
        for (int b=0;b<3;b++) gg += G[a*3+b]*G[a*3+b];
        rho += sqrtf(dd*gg);
    }
    float e2 = 0.f;
    for (int k=0;k<9;k++) { float d = A0[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; }
    float del = sqrtf(e2) + rho;
    bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
    if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
    if (fast) return;
    int slot = atomicAdd(&lst[0], 1);
    lst[1 + slot] = t;
}
__global__ void c_sr_warp2(const int* lst, const long long* tets, const float* gl, const float* Ucum, CVS S,
                           float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int cnt = lst[0];
    int lane = threadIdx.x & 31;
    int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int nw = (gridDim.x*blockDim.x) >> 5;
    float hs = S.hs;
    for (int q = gw; q < cnt; q += nw) {
        int t = lst[1 + q];
        long long n[4]; float A0[9], Mo[36], rho, G[12];
        for (int i=0;i<4;i++) n[i]=tets[t*4+i];
        for (int k=0;k<12;k++) G[k]=gl[t*12+k];
        for (int k=0;k<9;k++) A0[k] = (k % 4 == 0) ? 1.f : 0.f;
        rho = 0.f;
        for (int i=0;i<4;i++) {
            float vv[3]; c_vget(S, n[i], vv);
            float dd = 0.f, gg = 0.f;
            for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vv[c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
            for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
            rho += sqrtf(dd*gg);
        }
        float D0 = det3(A0);
        float r0 = sr_ratio(A0);
        float thrR = fmaxf(csr, r0);
        float thrJ = fminf(jmin, D0);
        bool ok = true;
        if (jon) { bool o = (lane >= 15) || !(det_sub(A0, Mo, lane + 1, 1.f) < thrJ); ok = __all_sync(CFULL, o); }
        if (ok && D0 > 0.f) {
            float C[9];
            for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
            float l[3]; symeig3(C, l);
            float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
            if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) continue;
        }
        if (c_feasw(lane, A0, Mo, 1.f, thrR, jon, thrJ)) continue;
        float lo = 0.f, hi = 1.f;
        for (int it=0; it<14; it++) {
            float mid = 0.5f*(lo+hi);
            if (c_feasw(lane, A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
        }
        if (lane == 0) {
            unsigned int bits = __float_as_uint(lo);
            for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
            atomicAdd(&rst[7], 1.f);
        }
    }
}
__global__ void c_apply2(const int* szp, CVS S, float* vn, float* f, float* gn, float* acc, float* Ucum, unsigned int* alph, int vmode, float* rst,
                         int* lst, unsigned int* dn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float v[3]; c_vget(S, n, v);
        for (int c=0;c<3;c++) { f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f; }
        if (acc) acc[n] = 0.f;
        float al = __uint_as_float(alph[n]);
        alph[n] = 0x3f800000u;
        if (S.fixd && S.fixd[n]) al = 1.f;
        float U[3];
        for (int c=0;c<3;c++) { vn[n*3+c] = v[c]; Ucum[n*3+c] += al*S.hs*v[c]; U[c] = Ucum[n*3+c]; }
        if (al < 1.f) {
            float m = S.mn[n], w = 1.f - al;
            atomicAdd(&rst[1], 1.f);
            if (vmode == 0) {
                for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
                float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
                atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
                atomicAdd(&rst[6], w*m*sqrtf(v2));
            }
        }
        dl = fmaxf(-(U[0]*S.n0 + U[1]*S.n1 + U[2]*S.n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(CFULL, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
}  // extern "C" (절 C)
'''

_MOD = [None]
NBW = 336                                                                            # c_sr_warp 블록 수 (128 스레드 = 4 워프; A6000 84 SM x 4)


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + _C).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def launch(name, n, args, b=128, grid=None):
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        mod().get_function(name)((g,), (b,), args)


P = EX.P


CVS_DT = np.dtype([("mn", np.uint64), ("f", np.uint64), ("gn", np.uint64), ("vn", np.uint64),
                   ("g0", np.float32), ("g1", np.float32), ("g2", np.float32), ("hs", np.float32), ("damp", np.float32),
                   ("fixd", np.uint64), ("vbc", np.uint64), ("acc", np.uint64),
                   ("mode", np.int32), ("np_", np.int32), ("n0", np.float32), ("n1", np.float32), ("n2", np.float32)], align=True)


def cvs(mn, f, gn, vn, g, hs, damp, fixd, vbc, acc, mode, np_, nrm):
    """장치 구조체 CVS (c_vnew 인자 묶음) -- numpy 정렬 구조 스칼라 (C 배치와 같다: 포인터 8 바이트 정렬)."""
    r = np.zeros(1, dtype=CVS_DT)
    r["mn"], r["f"], r["gn"], r["vn"] = P(mn), P(f), P(gn), P(vn)
    r["g0"], r["g1"], r["g2"], r["hs"], r["damp"] = float(g[0]), float(g[1]), float(g[2]), float(hs), float(damp)
    r["fixd"], r["vbc"], r["acc"] = P(fixd), P(vbc), P(acc)
    r["mode"], r["np_"] = int(mode), int(np_)
    r["n0"], r["n1"], r["n2"] = float(nrm[0]), float(nrm[1]), float(nrm[2])
    return r[0]
