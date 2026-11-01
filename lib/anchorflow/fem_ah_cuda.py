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
#ifndef AF_KD_DEF
#define AF_KD_DEF
__device__ const float* AHCD;                                                      // 절 AG.19: 노드 점성 대각 블록 C_nn (6 성분, null = 없음) -- IMEX 에 점성 감쇠를 후퇴 오일러로
__device__ const float* AHDV;                                                      // 절 AG.18 --ah_imex_cl: 대상 무리 가우스-자이델 (야코비) 으로 푼 Δv (null = 노드 걸음에서 직접 풂)
__device__ float* AHNB;                                                             // 절 AG.18 --ah_imex_rel: 노드별 [Σc v_j (3), Σc Δv_j (3), Σc] (7 성분) -- 이웃 결합 상대 꼴 (null = 이웃 고정 판)
__device__ int AHKF;                                                                // 1 = 대상 판정을 노드 대각 블록 최대 고윳값으로: (h_s/2)² λ_max(K_nn) > m (그 노드 자기 진동수 -- 이웃 고정)
__device__ int AHKG;                                                                // 1 = K_nn 대신 Gershgorin 상계 k_n I (k_n = 4 mfa / h_s², 대각 우세라 이웃 고정 한 걸음이 안정)
__device__ const float* AHKD;                                                       // 절 AG.18 --ah_imex: 노드 3x3 접선 강성 대각 블록 (6 성분, null = 끔). 켜면 국소 CFL 비 > 1 노드만 국소 후퇴 오일러
#endif
#ifndef AF_VN_DEF
#define AF_VN_DEF
__device__ const float* AFVN;                                                       // 절 AG.18 --vfl_pred: 노드 속도 (null = 끔) -- 비관통 판정을 서브스텝 끝 예측 입자 위치 s + h_s (Σλ v)·n 로
__device__ float AFHS;
#endif
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
    if (!AFVN && s0c[q] > __uint_as_float(*dn) + 1e-5f + band) return;          // --vfl_pred 판은 후보 생략 없이 (예측 이동만큼 앞서 닿을 수 있다)
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (AFVN) { float vp = 0.f; for (int i=0;i<4;i++) vp += lm[i]*(AFVN[nd[i]*3]*n0 + AFVN[nd[i]*3+1]*n1 + AFVN[nd[i]*3+2]*n2); s += AFHS*vp; }
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
    float mi = AHKD ? fmaxf(m, mfl) : fmaxf(m, mfa ? mfa[n] : mfl);            // --ah_imex: 질량 하한 대신 (mfa = 국소 CFL 질량 (h_s/2)² k_n 은 판정에만)
    bool imp = AHKD && mfa && !(fixd && fixd[n]) && (mfa[n] > m);             // 국소 CFL 비 (h_s/2)² k_n / m_n > 1
    if (AHKD && AHKF && !(fixd && fixd[n])) {
        const float* K = AHKD + (long long)n*6;
        float Cm[9] = {K[0], K[3], K[5], K[3], K[1], K[4], K[5], K[4], K[2]}; float l_[3]; symeig3(Cm, l_);
        imp = (AHKF == 2) ? true : (0.25f*hs*hs*l_[2] > m);                       // AHKF 2 = 모든 노드 (단위 시험 전용)
    }                                    // 절 AG.15: 노드별 국소 CFL 관성 질량 하한 (mfa != null)                                                   // 절 AH.18 --ah_mlow: 가속도에만 쓰는 관성 질량 하한 (0 이면 기존)
    float v[3];
    if (imp) {                                                                  // 절 AG.18: (m I + h² K_nn) Δv = h (f + m g − h K_nn v), 이웃은 명시적 값 고정 (VBD 노드 블록과 같은 꼴, 선형화 후퇴 오일러 한 걸음)
        const float* K = AHKD + (long long)n*6;                                  // xx yy zz xy yz xz
        float Km[9] = {K[0], K[3], K[5], K[3], K[1], K[4], K[5], K[4], K[2]};
        if (AHKG) { float kn = 4.f*mfa[n]/(hs*hs); for (int k_=0;k_<9;k_++) Km[k_] = 0.f; Km[0] = Km[4] = Km[8] = kn; }
        float v0[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
        if (AHDV) {                                                             // 무리 풀이 결과 그대로
            if (AHNB) { float* B = AHNB + (long long)n*7; for (int q_=0;q_<7;q_++) B[q_] = 0.f; }
            for (int a_=0;a_<3;a_++) v[a_] = (v0[a_] + AHDV[n*3+a_])*damp;
            for (int c=0;c<3;c++) f[n*3+c] = 0.f;
        } else {
        float r[3], Mx[9];
        float vb[3] = {v0[0], v0[1], v0[2]}, dvb[3] = {hs*gg[0], hs*gg[1], hs*gg[2]};
        if (AHNB) {                                                             // 상대 꼴: K_nn (v_n − v̄) − K_nn Δv̄ -- 병진에 영 강성 (Σ c = K 행 합의 대각 몫)
            float* B = AHNB + (long long)n*7;
            if (B[6] > 1e-30f) for (int a_=0;a_<3;a_++) { vb[a_] = B[a_]/B[6]; dvb[a_] = B[3+a_]/B[6]; }
            for (int q_=0;q_<7;q_++) B[q_] = 0.f;
        }
        for (int a_=0;a_<3;a_++) {
            float d0 = AHNB ? (v0[0]-vb[0]) : v0[0], d1 = AHNB ? (v0[1]-vb[1]) : v0[1], d2 = AHNB ? (v0[2]-vb[2]) : v0[2];
            float kv = Km[a_*3]*d0 + Km[a_*3+1]*d1 + Km[a_*3+2]*d2;
            float kd = AHNB ? (Km[a_*3]*dvb[0] + Km[a_*3+1]*dvb[1] + Km[a_*3+2]*dvb[2]) : 0.f;
            r[a_] = hs*(f[n*3+a_] + m*gg[a_] - hs*kv + hs*kd);
        }
        for (int k_=0;k_<9;k_++) Mx[k_] = hs*hs*Km[k_]; Mx[0] += m; Mx[4] += m; Mx[8] += m;
        float cM[9]; cof3(Mx, cM); float dM = Mx[0]*cM[0]+Mx[1]*cM[1]+Mx[2]*cM[2];
        if (dM > 1e-30f) { for (int a_=0;a_<3;a_++) { float dv = (cM[a_]*r[0] + cM[3+a_]*r[1] + cM[6+a_]*r[2])/dM; v[a_] = (v0[a_] + dv)*damp; } }   // 대칭이라 cofᵀ = cof
        else for (int a_=0;a_<3;a_++) v[a_] = (m > 0.f) ? (v0[a_] + hs*(f[n*3+a_]/m + gg[a_]))*damp : 0.f;
        for (int c=0;c<3;c++) f[n*3+c] = 0.f;
        }
    } else
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f || (mfa && mi > 0.f)) v[c] = (vn[n*3+c] + hs*(f[n*3+c]/mi + gg[c]))*damp;   // 절 AG.18: 질량 0 노드 (빈 사면체만 붙음) 도 국소 CFL 관성 질량이 있으면 움직인다
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
            if (np_ == 2) {                                                     // 절 AG.18 --vfl_np 2: 관통 입자가 가진 안쪽 법선 운동량만 지운다 -- 노드가 진 관통 입자 질량 몫 a/m (P2G 와 같은 가중) 만큼
                float fr = (m > 0.f) ? fminf(a/m, 1.f) : 1.f;
                float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2;
                if (a > 0.f && vn_ < 0.f) { v[0] -= fr*vn_*n0; v[1] -= fr*vn_*n1; v[2] -= fr*vn_*n2; }
            }
            else if (np_ && cn > 0.f) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
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
// 절 AG.16 (a) 사면체별 계수 MT[t] (현재 접선 강성 기반) 로 국소 CFL 질량: mfa[n] += c V_T MT[t] |∇λ_i| Σ|∇λ_j|
__global__ void ah_kn2(const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, const float* MT, float c, float* mfa) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0];
    float V = VT[t];
    if (!(V > 0.f)) return;
    float gn[4], sg = 0.f;
    for (int i=0;i<4;i++) { float a_=0.f; for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; } gn[i] = sqrtf(a_); sg += gn[i]; }
    for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], c*V*MT[t]*gn[i]*sg);
}
// 절 AG.17 서브스텝마다 국소 CFL 질량: 현재 셀 기하 (∇ₓλ_i = A⁻ᵀ ∇λ_i) 와 현재 변형 (σ_min(A F̄)) 의 접선 계수로
//   k_n = Σ V_T M (σ) |A⁻ᵀg_i| Σ_j |A⁻ᵀg_j|,  M(σ) = (λ+2μ) max(1, (1 + σ⁻²)/2)  (neo-Hookean 단축 압축 접선의 σ 의존 꼴 κ/2 (1 + σ⁻²) 을 λ+2μ 로 맞춘 것)
//   뒤집힌 (det ≤ 0) 셀은 σ_min 을 0.01 로 (강성 상한). mfa[n] += c k_n (c = s (h_s/2)²)
__global__ void ah_kn_sub(const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* VT, const float* FT,
                          float mw, float c, float* mfa) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= tlc[0]) return;
    int t = tl[i0];
    float V = VT[t];
    if (!(V > 0.f)) return;
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int cc=0;cc<3;cc++) { float u = U[nd*3+cc]; for (int q=0;q<3;q++) A[cc*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    float cA[9]; cof3(A, cA); float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    float Fb[9]; for (int k=0;k<9;k++) Fb[k] = FT[(long long)t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float sm = sqrtf(fmaxf(l[0], 0.f));
    if (!(dA > 0.f) || !(det3(X) > 0.f)) sm = 0.01f;
    sm = fmaxf(sm, 0.01f);
    float M = mw*fmaxf(1.f, 0.5f*(1.f + 1.f/(sm*sm)));
    float id = (fabsf(dA) > 1e-12f) ? 1.f/dA : 0.f;
    float gn[4], sg = 0.f;
    for (int i=0;i<4;i++) {                                                     // A⁻ᵀ g = cof(A) g / det
        float a_ = 0.f;
        for (int r=0;r<3;r++) { float v_ = 0.f; for (int q=0;q<3;q++) v_ += cA[r*3+q]*gl[(long long)t*12+i*3+q]; v_ *= id; a_ += v_*v_; }
        gn[i] = (dA > 0.f) ? sqrtf(a_) : 0.f; sg += gn[i];
    }
    if (!(dA > 0.f)) { for (int i=0;i<4;i++) { float a_=0.f; for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; } gn[i] = sqrtf(a_)/0.01f; } sg = gn[0]+gn[1]+gn[2]+gn[3]; }   // 뒤집힌 셀: 현재 기울기 정의 안 됨 -> σ_min 하한 0.01 과 같은 1/σ 배
    for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], c*V*M*gn[i]*sg);
}
// 절 AG.18: 빈 사면체 포함 칸 묶음의 서브스텝 국소 CFL 상계 (ah_kn_sub 와 같은 현재 기하·변형 계수). 빈 사면체는 F̄ = I 로
__device__ __forceinline__ void akg_cur(int t, const long long* tets, const float* U, const float* gl, const float* VT, const float* FT, float* gx, float* Mf) {
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int cc=0;cc<3;cc++) { float u = U[nd*3+cc]; for (int q=0;q<3;q++) A[cc*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    float cA[9]; cof3(A, cA); float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    float Fb[9] = {1,0,0, 0,1,0, 0,0,1};
    if (VT[t] > 0.f) for (int k=0;k<9;k++) Fb[k] = FT[(long long)t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float sm = sqrtf(fmaxf(l[0], 0.f));
    if (!(dA > 0.f) || !(det3(X) > 0.f)) sm = 0.01f;
    sm = fmaxf(sm, 0.01f);
    *Mf = fmaxf(1.f, 0.5f*(1.f + 1.f/(sm*sm)));
    float id = (dA > 0.f) ? 1.f/dA : 0.f;
    for (int i=0;i<4;i++) {
        float a_ = 0.f;
        if (dA > 0.f) { for (int r=0;r<3;r++) { float v_ = 0.f; for (int q=0;q<3;q++) v_ += cA[r*3+q]*gl[(long long)t*12+i*3+q]; v_ *= id; a_ += v_*v_; } gx[i] = sqrtf(a_); }
        else { for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; } gx[i] = sqrtf(a_)/0.01f; }
    }
}
__global__ void akg_kns1(const int* szp, const int* tg, const long long* tets, const float* U, const float* gl, const float* VT, const float* FT,
                         const float* W, const float* VG, float* Sg, float* Fm) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    int g = tg[t]; if (g < 0) return;
    float gx[4], Mf; akg_cur(t, tets, U, gl, VT, FT, gx, &Mf);
    atomicAdd(&Sg[g], W[t]/fmaxf(VG[g], 1e-30f)*(gx[0]+gx[1]+gx[2]+gx[3]));
    if (VT[t] > 0.f) atomicMax((int*)&Fm[g], __float_as_int(Mf));            // 양수 float 는 int 비교 순서가 같다
}
__global__ void akg_kns2(const int* szp, const int* tg, const long long* tets, const float* U, const float* gl, const float* VT, const float* FT,
                         const float* W, const float* VG, const float* MG, const float* Sg, const float* Fm, float c, float hg2mu, float* mfa) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    int g = tg[t]; if (g < 0) return;
    float gx[4], Mf; akg_cur(t, tets, U, gl, VT, FT, gx, &Mf);
    float S = Sg[g], wr = W[t]/fmaxf(VG[g], 1e-30f), sx = gx[0]+gx[1]+gx[2]+gx[3];
    float fm = fmaxf(Fm[g], 1.f);
    float k = MG[g]*fm*wr*S + hg2mu*W[t]*(sx + S);                            // Ā 결합 (입자 실제 계수 x 압축 계수) + 모래시계 (자기 + Ā 결합)
    for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], c*k*gx[i]);
}
// 절 AG.18 --ah_imex_rel: 이웃 결합 (서브스텝마다, 힘 뒤·노드 걸음 앞). c_ij = −w_t ∇λ_i·∇λ_j (Σ_j≠i c_ij = w_t |∇λ_i|², 병진 정확), 이웃 Δv = 명시적 h(f/m + g) (대상·질량 0 이웃은 h g)
__global__ void ah_imx_nb(const int* szp, const long long* tets, const float* gl, const float* VT, const float* W, const float* vn, const float* f, const float* mn,
                          const float* mfa, float hs, float g0, float g1, float g2, float* B, int clus) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    float w = VT[t] + (W ? W[t] : 0.f);
    if (!(w > 0.f)) return;
    long long nd[4]; float G[12];
    for (int i=0;i<4;i++) nd[i] = tets[(long long)t*4+i];
    for (int q=0;q<12;q++) G[q] = gl[(long long)t*12+q];
    float gg[3] = {g0, g1, g2};
    float dv[4][3];
    for (int j=0;j<4;j++) { long long m_ = nd[j]; float mm = mn[m_]; bool fl = mfa && (mfa[m_] > mm);
        for (int c=0;c<3;c++) dv[j][c] = (!fl && mm > 0.f) ? hs*(f[m_*3+c]/mm + gg[c]) : ((fl && clus) ? 0.f : hs*gg[c]); }   // 무리 판: 대상 이웃 몫은 반복에서
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) { if (i == j) continue;
        float c_ = -w*(G[i*3]*G[j*3] + G[i*3+1]*G[j*3+1] + G[i*3+2]*G[j*3+2]);
        float* Bn = B + nd[i]*7;
        for (int c=0;c<3;c++) { atomicAdd(&Bn[c], c_*vn[nd[j]*3+c]); atomicAdd(&Bn[3+c], c_*dv[j][c]); }
        atomicAdd(&Bn[6], c_); }
}
// 절 AG.18 --ah_imex_cl: 무리 반복 -- (1) 사면체: 대상 이웃 Δv 의 c 가중 합, (2) 대상 노드: 같은 3x3 풀이로 새 Δv, 잔차 (최대 변화·최대 크기)
__global__ void ah_imx_cl_t(const int* szp, const long long* tets, const float* gl, const float* VT, const float* W, const float* mn, const float* mfa,
                            const float* dvc, float* S2) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    float w = VT[t] + (W ? W[t] : 0.f);
    if (!(w > 0.f)) return;
    long long nd[4]; float G[12];
    for (int i=0;i<4;i++) nd[i] = tets[(long long)t*4+i];
    for (int q=0;q<12;q++) G[q] = gl[(long long)t*12+q];
    bool fl[4]; for (int j=0;j<4;j++) fl[j] = mfa[nd[j]] > mn[nd[j]];
    for (int i=0;i<4;i++) { if (!fl[i]) continue; for (int j=0;j<4;j++) { if (i == j || !fl[j]) continue;
        float c_ = -w*(G[i*3]*G[j*3] + G[i*3+1]*G[j*3+1] + G[i*3+2]*G[j*3+2]);
        for (int c=0;c<3;c++) atomicAdd(&S2[nd[i]*3+c], c_*dvc[nd[j]*3+c]); } }
}
__global__ void ah_imx_cl_n(const int* szp, const float* mn, const float* mfa, const unsigned char* fixd, const float* f, const float* vn, float g0, float g1, float g2,
                            float hs, const float* Kd, const float* B, float* S2, const float* dvc, float* dvn, unsigned int* res) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= szp[0]) return;
    float m = mn[n];
    if (!(mfa[n] > m) || (fixd && fixd[n])) { S2[n*3] = S2[n*3+1] = S2[n*3+2] = 0.f; return; }
    float gg[3] = {g0, g1, g2};
    const float* K = Kd + (long long)n*6;
    float Km[9] = {K[0], K[3], K[5], K[3], K[1], K[4], K[5], K[4], K[2]};
    if (AHKG) { float kn = 4.f*mfa[n]/(hs*hs); for (int k_=0;k_<9;k_++) Km[k_] = 0.f; Km[0] = Km[4] = Km[8] = kn; }
    const float* Bn = B + (long long)n*7;
    float v0[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]}, vb[3] = {v0[0], v0[1], v0[2]}, dvb[3] = {0.f, 0.f, 0.f};
    if (Bn[6] > 1e-30f) for (int a_=0;a_<3;a_++) { vb[a_] = Bn[a_]/Bn[6]; dvb[a_] = (Bn[3+a_] + S2[n*3+a_])/Bn[6]; }
    else for (int a_=0;a_<3;a_++) dvb[a_] = hs*gg[a_];
    S2[n*3] = S2[n*3+1] = S2[n*3+2] = 0.f;
    float Cm[9] = {0,0,0,0,0,0,0,0,0};
    if (AHCD) { const float* Cc = AHCD + (long long)n*6; Cm[0]=Cc[0]; Cm[4]=Cc[1]; Cm[8]=Cc[2]; Cm[1]=Cm[3]=Cc[3]; Cm[5]=Cm[7]=Cc[4]; Cm[2]=Cm[6]=Cc[5]; }
    float r[3], Mx[9];
    for (int a_=0;a_<3;a_++) {
        float kv = Km[a_*3]*(v0[0]-vb[0]) + Km[a_*3+1]*(v0[1]-vb[1]) + Km[a_*3+2]*(v0[2]-vb[2]);
        float kd = Km[a_*3]*dvb[0] + Km[a_*3+1]*dvb[1] + Km[a_*3+2]*dvb[2];
        float cd = Cm[a_*3]*dvb[0] + Cm[a_*3+1]*dvb[1] + Cm[a_*3+2]*dvb[2];      // 점성 (힘 f 에 −C v 가 이미 있다): 후퇴 오일러 몫 h C (Δv_n − Δv̄)
        r[a_] = hs*(f[n*3+a_] + m*gg[a_] - hs*kv + hs*kd + cd);
    }
    for (int k_=0;k_<9;k_++) Mx[k_] = hs*hs*Km[k_] + hs*Cm[k_]; Mx[0] += m; Mx[4] += m; Mx[8] += m;
    float cM[9]; cof3(Mx, cM); float dM = Mx[0]*cM[0]+Mx[1]*cM[1]+Mx[2]*cM[2];
    float d[3] = {hs*gg[0], hs*gg[1], hs*gg[2]};
    if (dM > 1e-30f) for (int a_=0;a_<3;a_++) d[a_] = (cM[a_]*r[0] + cM[3+a_]*r[1] + cM[6+a_]*r[2])/dM;
    float ch = 0.f, mg = 0.f;
    for (int a_=0;a_<3;a_++) { dvn[n*3+a_] = d[a_]; ch = fmaxf(ch, fabsf(d[a_] - dvc[n*3+a_])); mg = fmaxf(mg, fabsf(d[a_])); }
    if (isfinite(ch)) { atomicMax(&res[0], __float_as_uint(ch)); atomicMax(&res[1], __float_as_uint(mg)); } else atomicMax(&res[0], 0x7f7fffffu);
}
// 절 AG.19 --ah_mcfl_Fs: FCR 접선 강성 해석 상계 (주 늘림 공간 헤세 행 합, 비틀림 몫 2μ) / (λ+2μ) -- |σ| 라 σ -> 0 에서도 유한
__device__ __forceinline__ float fcr_kfac(const float* X, float mu, float lam) {
    float C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float s[3] = {sqrtf(fmaxf(l[0],0.f)), sqrtf(fmaxf(l[1],0.f)), sqrtf(fmaxf(l[2],0.f))};
    float J = s[0]*s[1]*s[2];
    float P[3] = {s[1]*s[2], s[0]*s[2], s[0]*s[1]};
    float km = 2.f*mu;
    for (int i=0;i<3;i++) { float r = 2.f*mu + lam*P[i]*P[i];
        for (int j=0;j<3;j++) { if (j == i) continue; int k = 3 - i - j; r += fabsf(lam*(P[i]*P[j] + (J - 1.f)*s[k])); }
        km = fmaxf(km, r); }
    return fmaxf(1.f, km/(lam + 2.f*mu));
}
__global__ void akfs_p(int N, const int* tt, const long long* tets, const float* U, const float* gl, const float* akF, const int* gp, const float* AG, const float* VG,
                       float mu, float lam, unsigned int* fTn, unsigned int* Fgn) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N) return;
    int t = tt[k]; int g = gp ? gp[k] : -1;
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    if (g >= 0) { float iv = 1.f/fmaxf(VG[g], 1e-30f); for (int j=0;j<9;j++) A[j] = AG[(long long)g*9+j]*iv; }
    else for (int i=0;i<4;i++) { long long nd = tets[(long long)t*4+i]; for (int c=0;c<3;c++) { float u = U[nd*3+c]; for (int q=0;q<3;q++) A[c*3+q] += u*gl[(long long)t*12+i*3+q]; } }
    float F[9]; for (int j=0;j<9;j++) F[j] = akF[(long long)k*9+j];
    float X[9]; mm3(A, F, X);
    float f = fcr_kfac(X, mu, lam);
    if (!isfinite(f)) return;
    if (g >= 0) atomicMax(&Fgn[g], __float_as_uint(f)); else atomicMax(&fTn[t], __float_as_uint(f));
}
__global__ void akfs_t(const int* szp, const long long* tets, const float* gl, const int* tg, const float* vb, float* fTu, const float* fTn,
                       const float* W, const float* VG, const float* MG0, const float* Sg, const float* Fgu, const float* Fgn, float clp, float c, float* mfa) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    float gn[4], sg = 0.f;
    for (int i=0;i<4;i++) { float a_=0.f; for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; } gn[i] = sqrtf(a_); sg += gn[i]; }
    if (fTn[t] > fTu[t]) { float d = (fTn[t] - fTu[t])*clp*vb[t]; for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], d*gn[i]*sg); fTu[t] = fTn[t]; }
    int g = tg ? tg[t] : -1;
    if (g >= 0 && Fgn[g] > Fgu[g]) { float d = c*MG0[g]*(Fgn[g] - Fgu[g])*(W[t]/fmaxf(VG[g], 1e-30f))*Sg[g]; for (int i=0;i<4;i++) atomicAdd(&mfa[tets[(long long)t*4+i]], d*gn[i]); }
}
__global__ void akfs_g(int G, float* Fgu, const float* Fgn) { int g = blockIdx.x*blockDim.x + threadIdx.x; if (g < G) Fgu[g] = fmaxf(Fgu[g], Fgn[g]); }
// 절 AG.18 --ah_imex: 노드 대각 블록 K_nn (프레임 시작 기준 선형 탄성 접선) -- 입자 적분점 w (μ|g|² I + (μ + λ r) g gᵀ), g = F_pᵀ ∇λ_n (r = --ak_capmu λ 배율)
__global__ void ah_kd_pq(int N, const int* tt, const long long* tets, const float* gl, const float* akF, const float* wp, const float* lr, const int* gp,
                         float mu, float lam, float* Kd) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N) return;
    if (gp && gp[k] >= 0) return;                                               // 묶음 Ā 를 쓰는 입자는 akg_kd 에서
    int t = tt[k];
    float F[9]; for (int j=0;j<9;j++) F[j] = akF[(long long)k*9+j];
    float w = wp[k], la = lam*(lr ? lr[k] : 1.f);
    for (int i=0;i<4;i++) {
        float G[3] = {gl[(long long)t*12+i*3], gl[(long long)t*12+i*3+1], gl[(long long)t*12+i*3+2]};
        float g[3]; for (int b=0;b<3;b++) g[b] = G[0]*F[b] + G[1]*F[3+b] + G[2]*F[6+b];   // g_b = Σ_q G_q F_qb
        float gg_ = g[0]*g[0]+g[1]*g[1]+g[2]*g[2], c2 = mu + la;
        long long nd = tets[(long long)t*4+i];
        atomicAdd(&Kd[nd*6+0], w*(mu*gg_ + c2*g[0]*g[0])); atomicAdd(&Kd[nd*6+1], w*(mu*gg_ + c2*g[1]*g[1])); atomicAdd(&Kd[nd*6+2], w*(mu*gg_ + c2*g[2]*g[2]));
        atomicAdd(&Kd[nd*6+3], w*c2*g[0]*g[1]); atomicAdd(&Kd[nd*6+4], w*c2*g[1]*g[2]); atomicAdd(&Kd[nd*6+5], w*c2*g[0]*g[2]);
    }
}
// 묶음 몫 (빈 사면체 포함): Ā 를 쓰는 입자 계수 합 MG (W_t/V_g)² |∇λ_n|² + 모래시계 2γμ W_t |∇λ_n|² (등방 근사, 같은 칸 다른 사면체와의 교차 항 생략)
__global__ void akg_kd(const int* szp, const int* tg, const long long* tets, const float* gl, const float* W, const float* VG, const float* MG, float hg2mu, float* Kd) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    int g = tg[t]; if (g < 0) return;
    float wr = W[t]/fmaxf(VG[g], 1e-30f);
    float c = MG[g]*wr*wr + hg2mu*W[t];
    for (int i=0;i<4;i++) { float a_=0.f; for (int q=0;q<3;q++) { float g_ = gl[(long long)t*12+i*3+q]; a_ += g_*g_; }
        long long nd = tets[(long long)t*4+i]; atomicAdd(&Kd[nd*6+0], c*a_); atomicAdd(&Kd[nd*6+1], c*a_); atomicAdd(&Kd[nd*6+2], c*a_); }
}
// 절 AG.19 --visc_rpic: 사면체 점성 응력 τ_v = 2 η sym(L), L = Σ v_i ⊗ ∇λ_i (프레임 시작 기하), 힘 f_i -= V_p τ_v ∇λ_i (V_p = 사면체 안 입자 부피 합)
__global__ void ah_visc(const int* szp, const long long* tets, const float* gl, const float* VP, const float* vn, float eta, float* f) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    float V = VP[t];
    if (!(V > 0.f)) return;
    long long nd[4]; float G[12];
    for (int i=0;i<4;i++) nd[i] = tets[(long long)t*4+i];
    for (int q=0;q<12;q++) G[q] = gl[(long long)t*12+q];
    float L[9] = {0,0,0,0,0,0,0,0,0};
    for (int i=0;i<4;i++) for (int a_=0;a_<3;a_++) for (int b=0;b<3;b++) L[a_*3+b] += vn[nd[i]*3+a_]*G[i*3+b];
    float tv[9]; for (int a_=0;a_<3;a_++) for (int b=0;b<3;b++) tv[a_*3+b] = eta*(L[a_*3+b] + L[b*3+a_]);
    for (int i=0;i<4;i++) for (int a_=0;a_<3;a_++) { float s_ = 0.f; for (int b=0;b<3;b++) s_ += tv[a_*3+b]*G[i*3+b]; atomicAdd(&f[nd[i]*3+a_], -V*s_); }
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


def set_kd(t):
    """절 AG.18 --ah_imex: AHKD 장치 포인터 (torch float32 [Mcap*6] 또는 None)."""
    import cupy as cp
    cp.ndarray((1,), cp.uint64, mod().get_global("AHKD"))[:] = 0 if t is None else t.data_ptr()


def set_kg(on):
    """절 AG.18 --ah_imex_k gersh."""
    import cupy as cp
    cp.ndarray((1,), cp.int32, mod().get_global("AHKG"))[:] = 1 if on else 0


def set_cd(t):
    """절 AG.19: AHCD 포인터."""
    import cupy as cp
    cp.ndarray((1,), cp.uint64, mod().get_global("AHCD"))[:] = 0 if t is None else t.data_ptr()


def set_dv(t):
    """절 AG.18 --ah_imex_cl: AHDV 포인터."""
    import cupy as cp
    cp.ndarray((1,), cp.uint64, mod().get_global("AHDV"))[:] = 0 if t is None else t.data_ptr()


def set_nb(t):
    """절 AG.18 --ah_imex_rel: AHNB 포인터."""
    import cupy as cp
    cp.ndarray((1,), cp.uint64, mod().get_global("AHNB"))[:] = 0 if t is None else t.data_ptr()


def set_kf(on):
    """절 AG.18 --ah_imex_flag diag."""
    import cupy as cp
    cp.ndarray((1,), cp.int32, mod().get_global("AHKF"))[:] = int(on)


def set_vpred(vn, hs):
    """절 AG.18 --vfl_pred: 비관통 판정 예측용 노드 속도 주소·h_s (vn None = 끔)."""
    import cupy as cp
    cp.ndarray((1,), cp.uint64, mod().get_global("AFVN"))[:] = 0 if vn is None else vn.data_ptr()
    cp.ndarray((1,), cp.float32, mod().get_global("AFHS"))[:] = float(hs)


def launch(name, n, args, b=128, grid=None):
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        mod().get_function(name)((g,), (b,), args)


def gt_floor_k0(dxg, pn):
    """검산용 (호스트): 정답 collide 와 같은 float32 비교로 고착 격자 노드 최대 지표."""
    k = np.arange(0, 400, dtype=np.float32)
    return int(k[(k * np.float32(dxg) - np.float32(pn)) < 0].max())
