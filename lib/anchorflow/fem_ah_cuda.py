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
    if (ncon) {                                                                 // 절 AH.11: 접촉 수 (통계) 워프 집계 -- 전역 카운터 하나에 서브스텝마다 수천 원자 합이 몰리던 것 (같은 정수)
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
// ex_node_step_lag 와 같은 걸음 (nodisp 1 판 = 속도만, U 는 c_lim_apply 가) + 고착 몫 (1 - a/m)^kex, 비관통은 accN > 0 (실제 접촉) 노드만 (accN null 이면 a > 0 노드 -- 기존과 같음)
__global__ void ah_node_step(const int* szp, const float* mn, float* f, float* vn, float g0, float g1, float g2, float hs, float damp,
                             const unsigned char* fixd, const float* vbc, float* acc, float* accN, int mode, int np_, float n0, float n1, float n2, float kex, float mfl,
                             const float* mfa) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    float mi = fmaxf(m, mfa ? mfa[n] : mfl);                                    // 절 AG.15: 노드별 국소 CFL 관성 질량 하한 (mfa != null)                                                   // 절 AH.18 --ah_mlow: 가속도에만 쓰는 관성 질량 하한 (0 이면 기존)
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
    for (int c=0;c<3;c++) vn[n*3+c] = v[c];
}
// ---- 운동량 보존 클램프 (--lim_v mom): c_lim_apply 와 같은 α 이동 + 속도 α 배, 줄인 운동량 Σ (1-α) m v 를 dP 에 모아 다음 커널이 몸 전체 (고정 아닌 질량 노드) 에 균일 속도로 되돌린다.
//      선운동량 보존 (클램프는 내부 구속이라 외력 아님). 줄인 KE 중 강체 병진 몫만 돌아오고 상대 운동 몫은 그대로 지워진다.
__global__ void ah_lim_apply(const int* szp, const float* mn, float* vn, float* Ucum, float hs, unsigned int* alph, const unsigned char* fixd, float* rst,
                             int* lst, unsigned int* dn, float n0, float n1, float n2, float* dP) {
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
            for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); atomicAdd(&dP[c], w*m*v[c]); }
            float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
            atomicAdd(&rst[6], w*m*sqrtf(v2));
        }
        dl = fmaxf(-(U[0]*n0 + U[1]*n1 + U[2]*n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(0xffffffffu, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
// 프레임마다 한 번: 고정 아닌 노드 질량 합 -> mt[0] (dP 도 비움)
__global__ void ah_mtot(const int* szp, const float* mn, const unsigned char* fixd, float* mt, float* dP) {
    __shared__ float sh[1024];
    int M = szp[0];
    float s = 0.f;
    for (int n = threadIdx.x; n < M; n += blockDim.x) if (!(fixd && fixd[n])) s += mn[n];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) { mt[0] = sh[0]; dP[0] = dP[1] = dP[2] = 0.f; }
}
// 서브스텝 끝: v += dP / M (고정 아닌 질량 노드). dP 는 다음 서브스텝 ah_fc 첫 스레드가 비운다.
__global__ void ah_mom(const int* szp, const float* mn, const unsigned char* fixd, float* vn, const float* mt, const float* dP) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= szp[0] || !(mn[n] > 0.f) || (fixd && fixd[n])) return;
    float im = 1.f / mt[0];
    for (int c=0;c<3;c++) vn[n*3+c] += dP[c]*im;
}
// ---- 절 AH.7 결과 불변 가속 (--vfl_agg): ah_fc 와 같은 식, 접촉 블록만 워프 집계 -- 후보를 프레임마다 입자 사면체 첫 노드 번호로 정렬해 두면
//      한 워프 안 입자들이 노드를 공유하므로, 노드 슬롯 i 마다 __match_any_sync 로 같은 노드 레인을 묶어 대표 레인 하나만 원자 합 (합 순서만 다름).
__global__ void ah_fc_agg(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
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
    int lane = threadIdx.x & 31;
    bool ok = q < szp[2];
    if (ok) ok = !(s0c[q] > __uint_as_float(*dn) + 1e-5f + band);
    if (!__any_sync(0xffffffffu, ok)) return;                                    // 워프 전체가 후보 아님 (블록은 32 의 배수, 레인 모두 여기 도달)
    long long nd[4] = {-1, -1, -1, -1}; float lm[4] = {0, 0, 0, 0};
    float w = 0.f, mp = 0.f; bool real = false;
    if (ok) {
        long long p = nidx[q];
        float u[3] = {0,0,0};
        for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
        float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
        if (gtk) { if (s < band) { w = ah_wstk(s, pn, dxg); if (!(w > 0.f) && !(s < 0.f)) w = 0.f; } }
        else if (s < 0.f) w = 1.f;
        if (gtk ? (s < band && (w > 0.f || s < 0.f)) : (s < 0.f)) { mp = m[p]; real = s < -npd; if (ncon && s < 0.f) atomicAdd(ncon, 1); }
        else ok = false;
    }
    if (!__any_sync(0xffffffffu, ok)) return;
    for (int i=0;i<4;i++) {
        long long key = ok ? nd[i] : -(long long)(lane + 1);
        unsigned msk = __match_any_sync(0xffffffffu, key);
        float va = ok && w > 0.f ? (tetmass ? 0.25f : lm[i])*mp*w : 0.f;
        float vc = (ok && real) ? 1.f : 0.f;
        float sa = 0.f, sc = 0.f;
        for (int j=0;j<32;j++) { float oa = __shfl_sync(0xffffffffu, va, j), oc = __shfl_sync(0xffffffffu, vc, j); if ((msk >> j) & 1u) { sa += oa; sc += oc; } }
        if (ok && lane == __ffs(msk) - 1) { if (sa > 0.f) atomicAdd(&acc[nd[i]], sa); if (accN && sc > 0.f) atomicAdd(&accN[nd[i]], sc); }
    }
}
// ---- 절 AH.11 결과 불변 가속 (--vfl_tacc, 사면체 등분 질량만): 띠 입자의 접촉 질량을 먼저 입자 사면체 accT 에 (원자 1 개), 처음 받은 사면체를 목록에;
//      ah_tdist 가 목록 사면체마다 0.25 accT 를 네 노드 acc 에 (+ 실제 접촉 표시). a_n = Σ_{T∋n} 0.25 Σ_{p∈T} m_p w -- ah_fc 와 같은 식, 합 순서만 다름.
__global__ void ah_fc_tacc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                           float mu, float lam, float* f, float* rst, int fpol, float* eF,
                           const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const long long* pinv,
                           const float* m, float p0, float p1, float p2, float n0, float n1, float n2, int* ncon, float band, float pn, float dxg, float npd,
                           float* accT, int* rT, int* clT) {
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
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm*U[nd*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < band)) return;
    float w = ah_wstk(s, pn, dxg);
    if (!(w > 0.f)) return;                                                     // s < 0 이면 w ≥ 0.125 > 0
    long long t = pinv[p];
    float old = atomicAdd(&accT[t], m[p]*w);
    if (old == 0.f) { int sl = atomicAdd(&clT[0], 1); clT[1 + sl] = (int)t; }
    if (s < -npd) rT[t] = 1;
    if (ncon && s < 0.f) atomicAdd(ncon, 1);
}
// 목록 사면체 -> 노드 (격자 크기 고정, 반복). 목록 개수는 ah_node_step2 가 비운다.
__global__ void ah_tdist(const int* clT, const long long* tets, float* accT, int* rT, float* acc, float* accN) {
    int cnt = clT[0];
    for (int k = blockIdx.x*blockDim.x + threadIdx.x; k < cnt; k += gridDim.x*blockDim.x) {
        int t = clT[1 + k];
        float a = 0.25f*accT[t]; int r = rT[t];
        accT[t] = 0.f; rT[t] = 0;
        for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; atomicAdd(&acc[nd], a); if (r) atomicAdd(&accN[nd], 1.f); }
    }
}
__global__ void ah_clr1(int* c) { if (threadIdx.x == 0) c[0] = 0; }
// ---- 절 AH.12 정의 변경 (--vfl_bf: 띠 가중을 프레임당 한 번) -- 프레임 시작 높이 s0 로 띠 입자의 고착 질량 accB_n = Σ 0.25 m_p w(s0_p) (s0_p < 띠) 를 한 번 모으고,
//      서브스텝은 그 값을 고착 몫으로 쓴다. 서브스텝마다 다시 보는 것은 평면 아래로 들어갔을 수 있는 입자 (s0 ≤ D + 1e-5, 기존 판과 같은 컬링) 뿐 -- 비관통 (s < -npd) 표시와 접촉 수.
__global__ void ah_band0(const int* szp, const float* s0c, const long long* nidx, const long long* rows, const float* lamb, const float* m, int tetmass,
                         float band, float pn, float dxg, float* accB) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    float s = s0c[q];
    if (!(s < band)) return;
    float w = ah_wstk(s, pn, dxg);
    if (!(w > 0.f)) return;
    long long p = nidx[q];
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&accB[rows[p*4+i]], (tetmass ? 0.25f : lamb[p*4+i])*mp*w);
}
__global__ void ah_fc_bf(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                         float mu, float lam, float* f, float* rst, int fpol, float* eF,
                         const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb,
                         float p0, float p1, float p2, float n0, float n1, float n2, int* ncon, float npd, float* accN) {
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
    long long nd[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    if (ncon) atomicAdd(ncon, 1);
    if (s < -npd) for (int i=0;i<4;i++) atomicAdd(&accN[nd[i]], 1.f);
}
// ah_node_step 과 같은 걸음, 고착 질량은 프레임 상수 accB (비우지 않음), 비관통은 accN (비움)
__global__ void ah_node_step_bf(const int* szp, const float* mn, float* f, float* vn, float g0, float g1, float g2, float hs, float damp,
                                const unsigned char* fixd, const float* vbc, const float* accB, float* accN, int mode, int np_, float n0, float n1, float n2, float kex) {
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
    float a = accB[n];
    float cn = accN[n]; accN[n] = 0.f;
    if (!(fixd && fixd[n])) {
        if (a > 0.f) {
            float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : powf(fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f), kex));
            for (int c=0;c<3;c++) v[c] *= k;
        }
        if (np_ && cn > 0.f) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
    }
    for (int c=0;c<3;c++) vn[n*3+c] = v[c];
}
// ---- 절 AH.14 결과 불변 가속 (--ah_fast): _sparse 의 vcap 처리 (torch 연산 여럿) 를 커널 하나로 -- 같은 식
//      Vg = float(1 / (6 |det G3|)) (G3 = ∇λ 의 1~3 행, double 삼중곱), VF = VT > 0 ? min(VT, θ Vg) : VT, VTs 도 VF 로.
__global__ void ah_vcap(const int* szp, const float* gl, float th, float* VTs, float* VF) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    float V = VTs[t];
    float o = V;
    if (V > 0.f) {
        double q[9];
        for (int k=0;k<9;k++) q[k] = (double)gl[t*12+3+k];
        double c0 = q[4]*q[8] - q[5]*q[7], c1 = q[5]*q[6] - q[3]*q[8], c2 = q[3]*q[7] - q[4]*q[6];
        double d = q[0]*c0 + q[1]*c1 + q[2]*c2;
        float Vg = (float)(1.0 / (6.0 * fabs(d)));
        o = fminf(V, th*Vg);
    }
    VF[t] = o; VTs[t] = o;
}
// ---- 절 AH.14 --ah_fast: 격자 노드 위치 (uniq scatter·나눗셈·stack·matmul 등 torch 연산 10 여 개) 를 커널 하나로 -- 같은 식 (float32: Σ_j k_j A_ij + lo_i)
__global__ void ah_npos(int Pn, int n1, int n2, const unsigned char* nm, const long long* ncm, const float* A, const float* lo, float* npos) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= Pn || !nm[c]) return;
    long long q = ncm[c];
    float k[3] = {(float)(c / (n1*n2)), (float)((c / n2) % n1), (float)(c % n2)};
    for (int i=0;i<3;i++) npos[q*3+i] = (k[0]*A[i*3] + k[1]*A[i*3+1] + k[2]*A[i*3+2]) + lo[i];
}
// 절 AG.15 국소 CFL 관성 질량: 노드 강성 상계 (Gershgorin) k_n = Σ_{T∋n} V_T M_w |∇λ_i| Σ_j |∇λ_j| (M_w = λ + 2μ 파동 계수, V_T = 셀 탄성 부피),
// 명시적 중앙차분 안정 조건 h_s ≤ 2/ω, ω² ≤ k_n/m_n -> m_n ≥ (h_s/2)² k_n.  mfa[n] = (h_s/2)² k_n (사면체 스레드가 원자 합)
__global__ void ah_kn(const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, float cmw, float* mfa) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0];
    float V = VT[t];
    if (!(V > 0.f)) return;
    float gn[4], sg = 0.f;
    for (int i=0;i<4;i++) { float a_=0.f; for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; } gn[i] = sqrtf(a_); sg += gn[i]; }
    for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], cmw*V*gn[i]*sg);
}
__global__ void ah_zero3(float* dP) { if (threadIdx.x < 3) dP[threadIdx.x] = 0.f; }
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
