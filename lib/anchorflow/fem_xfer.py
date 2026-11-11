"""절 CM109 (사용자 11-11 지시): 입자 <-> 격자 전달 (P2G·G2P) 정의로 wolf 무너짐·높이 쌓임 막기 -- 정규화 벌점 없이.

바탕 = 육면체 H9 사슬에서 B-스플라인 전용 정규화 (P1X 강성·질량 · 가벼움 벌점 · (K) · 확장 T1E2) 를 모두 뺀 판 ('H9nb', 절 CM105 의 N 과 같은 계산).
새 실행기 (lib/anchorflow/fem_an_rt.py RTSolver 와 재질 판 fem_mat_rt.MatRT) 만 -- 레고·ficus (탄성), wolf (모래), bread (plasticine). 옛 실행기 (수박) 는 안 고침.

(A) P2G 받침 판정 ('--xa 1'):
  - 받침 덮임 몫 φ_i = Σ_p V_p w_i(x_p) / |det A|  (V_p = 입자 부피 = 실행기 vol (처음 부피), w = 이차 B-스플라인 가중, |det A| = 칸 부피).
    노드 받침이 물질로 다 차면 Σ_p V_p w_i(x_p) -> ∫ N_i dx = |det A| (이차 B-스플라인 기저 적분 = 칸 부피) 라 φ = 1 이 '완전히 찬 경우'.
  - '완전히 찬 경우의 값' 은 장면 입자 부피 정의 (wolf 는 PG 가우스 부피라 칸을 다 채워도 Σ V w / |det A| ≈ 0.6) 에 따라 달라서,
    프레임마다 받침 27 칸이 모두 점유된 활성 노드의 Σ V w / |det A| 평균 φ̄_full 로 나눔: φ_i = Σ_p V_p w_i(x_p) / (|det A| φ̄_full) (연기 s0k 첫 판에서 rest 부피 정의는 wolf 묶는 질량 57 % -> 고침).
  - 판정 (새 임의 문턱 없음): φ_i < ½ 인 활성 노드 (질량 > 0) 를 '묶는 노드' 로 -- 근거: 기저가 노드 중심에 대칭이라 평평한 물질 경계 위에 놓인 노드는 φ = ½ 이고,
    노드 중심이 물질 밖이면 φ < ½, 안이면 > ½. 곧 '받침의 빈 몫이 찬 몫보다 큰 노드' = 중심이 물질 밖인 노드 = 독립 미지수로 두지 않음.
  - 가중 이웃 = 그 노드와 입자를 함께 덮는 (일관 질량 결합 M_oj = Σ_p m_p w_o(x_p) w_j(x_p) > 0, 오프셋 |Δ| ≤ 2) 독립 노드 (φ_j ≥ ½) 전부,
    가중 c_oj = M_oj / Σ_j M_oj (≥ 0, 합 1 = 볼록 결합; 2 차 외삽 (1, −3, 3) 같은 음의 가중 없음). 그런 이웃이 없으면 '고립' -- 그대로 (원래 자유도·질량).
  - P2G: 묶는 노드 o 의 질량·운동량 (+ HHT 가속도 몫 ma) 을 이웃에 c 로 넘김: m_j += c_oj m_o, (mv)_j += c_oj (mv)_o, (ma)_j += c_oj (ma)_o, 그리고 o 는 0.
    Σ_j c_oj = 1 이라 질량·운동량 합이 그대로 (= 일관 질량 E^T M E 의 행 합 lumping).
  - 위치·속도 = 이웃 값: 걸음 시작 (rt_ga_step0 앞) 과 걸음 끝 (rt_ga_end* 뒤, G2P·걸음 변위 보관 앞) 에 U_o, v_o, a_o = Σ c_oj (·)_j.
    풀이 안: 바닥 노드 커널 (rt_floor_n·mt_floor_n) 뒤 o 를 자유 노드 (T = I, ŷ = Ũ) 로 바꿔 시작점 규칙 (장벽 실현 가능 시작점·섞기) 이 이웃과 같이 정하게 하고,
    뉴턴 걸음 증분을 y_o − y0_o = Σ c_oj (y_j − y0_j) 로 묶음 (절 CM81 확장과 같은 줄어든 문제 꼴: 잔차·r0 접기 r_j += T_j c_oj r_o, r_o = 0, |g|² 다시,
    PCG 방향 펼치기 p_o = Σ c (T p)_j, 헤시안 곱 접기). 줄어든 에너지 = 원래 에너지 (탄성·장벽) 에 y = E u 를 넣은 것 + 합친 질량의 관성.
  - 고정 구속 노드 (fixd, ficus 화분) 는 묶지 않음 (질량 합침은 P2G 순간이라 그 뒤 고정 표시와 무관하게 일어남 -- ficus 고정 상자 안 가벼운 노드만 해당).
(B) G2P APIC ('--xb 1', Jiang et al. 2015 "The Affine Particle-In-Cell Method", SIGGRAPH, 식 (APIC 이차 B-스플라인)):
  - G2P: v_p = Σ_i w_ip v_i (그대로) + C_p = B_p D_p⁻¹, B_p = Σ_i w_ip v_i (x_i − x_p)ᵀ, D_p = Σ_i w_ip (x_i − x_p)(x_i − x_p)ᵀ = ¼ A Aᵀ (이차 B-스플라인, 격자 축 A).
    x_i − x_p = A (i − f_p) (격자 단위 거리 d_i), 그래서 C_p = 4 Σ_i w_ip v_i d_iᵀ A⁻¹. v_i = 프레임 끝 노드 속도 (G2P 가 쓰는 것과 같은 vk), 가중·거리 = 프레임 시작 입자 위치 (G2P 와 같음).
  - P2G: (mv)_i += w_ip m_p (v_p + C_p (x_i − x_p)) -- 입자 속도에 더한 아핀 몫만 이 모듈 (xb_p2g) 이 rt_p2g 뒤에 더함. Σ_i w_ip (x_i − x_p) = 0 이라 운동량 합은 그대로.
  - 입자 가속도 (HHT) 전달·질량은 그대로. 구속 입자 (순차 방출 전, v = 0) 는 C_p = 0.
기록 (결과 불변): xf.jsonl 프레임마다 -- 입자 Σm·Σm v 대 노드 Σm·Σ(mv) (P2G 뒤, (A) 합침·(B) 아핀 몫 뒤), (A) 활성·묶는·고립 노드 수와 질량 몫·넘침, φ 분위.
검사 (--xck 1, 셋째 프레임 뒤 그래프 밖 한 번): (A) 가중 행 합 1·음수 없음, (B) 아핀 속도장 v_i = c + G x_i 를 xb_g2p 에 넣어 C_p = G 재현.
"""
import json
import os

import numpy as np
import torch

CFG = dict(a=0, b=0, dg=1, ck=0, out=".")
INFO = dict(hooks={}, err=None, nsrc=0, nalloc=0, solvers=[], check=None)
ENDP = "rt_ga_end"
FLOORN = ("rt_floor_n", "mt_floor_n")
G2PN = ("rt_g2p_ga", "ao_g2p", "aq_g2p")
MARK = "// ---- 절 CM109 xfer"

SRC = r'''
// ---- 절 CM109 xfer: P2G 받침 판정 (A) · G2P APIC (B) (lib/anchorflow/fem_xfer.py) ----
#ifndef AF_ADAPT
#define XF_SKIP 0
#else
#define XF_SKIP AF_SKIP
#endif
__device__ __forceinline__ void xf_wsum(double v, double* dst) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v != 0.0) atomicAdd(dst, v);
}
// (A) φ_i += V_p w_i / |det A|
__global__ void xa_phi(int N, const int* IP, const int* pb, const float* fx, const float* vol, float idet, float* phi) {
    if (XF_SKIP) return;
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float vp = vol[p]*idet;
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        atomicAdd(&phi[nid], vp*w[0][i]*w[1][j]*w[2][k]);
    }
}
// '완전히 찬 경우의 값' = 받침 27 칸 (매듭 칸 [n−2, n]³) 이 모두 점유된 활성 노드의 Σ V w / |det A| 평균 (프레임마다) -> φ 를 그 값으로 나눔
__global__ void xa_full(const int* IP, const float* mn, const int* occ, const float* phi, double* R) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double a = 0.0, b = 0.0;
    if (n < IP[9] && !IP[13] && mn[n] > 0.f) {
        int nb1 = IP[7], nb2 = IP[8], c0 = IP[6]-2, c1 = nb1-2, c2 = nb2-2;
        int i = n/(nb1*nb2), j = (n/nb2)%nb1, k = n%nb2;
        bool full = (i >= 2 && j >= 2 && k >= 2 && i < c0+2 && j < c1+2 && k < c2+2);
        for (int x=i-2; full && x<=i; x++) for (int y=j-2; full && y<=j; y++) for (int z=k-2; full && z<=k; z++)
            if (x >= c0 || y >= c1 || z >= c2 || !occ[(x*c1 + y)*c2 + z]) full = false;
        if (full) { a = phi[n]; b = 1.0; }
    }
    xf_wsum(a, &R[0]); xf_wsum(b, &R[1]);
}
__global__ void xa_scale(const int* IP, const double* R, float* phi) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13] || !(R[1] > 0.0)) return;
    phi[n] = (float)((double)phi[n]*R[1]/R[0]);
}
// 묶는 노드 후보 (활성 · φ < ½) 에 자리 (slot) 와 가중 행 (125) 0
__global__ void xa_slot(const int* IP, const float* mn, const float* phi, int* slot, float* M, int* cnt, float* cf, int capT) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    slot[n] = -1;
    float m = mn[n];
    if (!(m > 0.f)) return;
    atomicAdd(&cnt[0], 1); atomicAdd(&cf[0], m);
    if (!(phi[n] < 0.5f)) return;
    int s = atomicAdd(&cnt[1], 1);
    if (s >= capT) { atomicAdd(&cnt[2], 1); return; }
    slot[n] = s;
    for (int t=0;t<125;t++) M[(size_t)s*125 + t] = 0.f;
}
// 일관 질량 결합 M_oj = Σ_p m_p w_o w_j (o = 묶는 후보, j = 독립 활성 노드), 오프셋 j − o ∈ [−2, 2]³
__global__ void xa_pair(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* mn, const float* phi, const int* slot, float* M) {
    if (XF_SKIP) return;
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    int b0 = pb[p*3], b1 = pb[p*3+1], b2 = pb[p*3+2];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int no = ((b0+i)*nb1 + (b1+j))*nb2 + (b2+k);
        int so = slot[no];
        if (so < 0) continue;
        float wo = w[0][i]*w[1][j]*w[2][k]*mp;
        for (int q=0;q<27;q++) {
            if (q == o) continue;
            int i2 = q/9, j2 = (q/3)%3, k2 = q%3;
            int nj = ((b0+i2)*nb1 + (b1+j2))*nb2 + (b2+k2);
            if (!(mn[nj] > 0.f) || !(phi[nj] >= 0.5f)) continue;
            int t = ((i2-i+2)*5 + (j2-j+2))*5 + (k2-k+2);
            atomicAdd(&M[(size_t)so*125 + t], wo*w[0][i2]*w[1][j2]*w[2][k2]);
        }
    }
}
// 행 정규화 c = M / Σ M; Σ = 0 이면 고립 (자리 지움 -> 원래 자유도)
__global__ void xa_norm(const int* IP, const float* mn, int* slot, float* M, int* cnt, float* cf) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int s = slot[n];
    if (s < 0) return;
    float* r = M + (size_t)s*125;
    double S = 0.0;
    for (int t=0;t<125;t++) S += (double)r[t];
    if (!(S > 0.0)) { slot[n] = -1; atomicAdd(&cnt[3], 1); atomicAdd(&cf[2], mn[n]); return; }
    float iS = (float)(1.0/S);
    for (int t=0;t<125;t++) r[t] *= iS;
    atomicAdd(&cnt[4], 1); atomicAdd(&cf[1], mn[n]);
}
__device__ __forceinline__ int xa_nb(int n, int t, int nb1, int nb2) {
    int dx = t/25 - 2, dy = (t/5)%5 - 2, dz = t%5 - 2;
    return n + (dx*nb1 + dy)*nb2 + dz;
}
// P2G 합침: 묶는 노드 질량·운동량·가속도 몫을 이웃에 c 로, 자기는 0
__global__ void xa_merge(const int* IP, const int* slot, const float* M, float* mn, float* mv, float* ma) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int s = slot[n];
    if (s < 0) return;
    const float* c = M + (size_t)s*125;
    int nb1 = IP[7], nb2 = IP[8];
    float m = mn[n], p[3] = {mv[n*3], mv[n*3+1], mv[n*3+2]}, q[3] = {0.f, 0.f, 0.f};
    if (ma) for (int a=0;a<3;a++) q[a] = ma[n*3+a];
    for (int t=0;t<125;t++) {
        float w = c[t];
        if (!(w > 0.f)) continue;
        int j = xa_nb(n, t, nb1, nb2);
        atomicAdd(&mn[j], w*m);
        for (int a=0;a<3;a++) atomicAdd(&mv[j*3+a], w*p[a]);
        if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[j*3+a], w*q[a]);
    }
    mn[n] = 0.f; for (int a=0;a<3;a++) mv[n*3+a] = 0.f;
    if (ma) for (int a=0;a<3;a++) ma[n*3+a] = 0.f;
}
// 위치·속도·가속도 = 이웃 값 (고정 노드 빼고): U_o, v_o, a_o = Σ c (·)_j
__global__ void xa_proj(const int* IP, const int* slot, const float* M, const unsigned char* fixd, float* U, float* vk, float* ak) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int s = slot[n];
    if (s < 0 || (fixd && fixd[n])) return;
    const float* c = M + (size_t)s*125;
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0.f,0.f,0.f}, v[3] = {0.f,0.f,0.f}, a_[3] = {0.f,0.f,0.f};
    for (int t=0;t<125;t++) {
        float w = c[t];
        if (!(w > 0.f)) continue;
        int j = xa_nb(n, t, nb1, nb2);
        for (int a=0;a<3;a++) { u[a] += w*U[j*3+a]; v[a] += w*vk[j*3+a]; if (ak) a_[a] += w*ak[j*3+a]; }
    }
    for (int a=0;a<3;a++) { U[n*3+a] = u[a]; vk[n*3+a] = v[a]; if (ak) ak[n*3+a] = a_[a]; }
}
// 바닥 노드 커널 뒤: 묶는 노드 = 자유 노드 (T = I, ŷ = Ũ, η = 0, y = Ũ) -- 시작점 규칙이 이웃과 같이 정함
__global__ void xa_free(const int* IP, const int* slot, const unsigned char* fixd, const float* Ut, float* y, float* sk, int* pm, float* yh, float* eta) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    if (slot[n] < 0 || (fixd && fixd[n])) return;
    for (int a=0;a<3;a++) { float u = Ut[n*3+a]; yh[n*3+a] = u; eta[n*3+a] = 0.f; y[n*3+a] = u; }
    sk[n] = 1.f; pm[n] = 0;
}
// 걸음 시작 (ctl 1, 시작점 다 정한 뒤): 변수 바꿈 항등 다시 (다른 훅이 바꿨어도), η = y − ŷ
__global__ void xa_free2(const int* IP, const int* slot, const unsigned char* fixd, float* sk, int* pm, const float* yh, const float* y, float* eta) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    if (slot[n] < 0 || (fixd && fixd[n])) return;
    sk[n] = 1.f; pm[n] = 0;
    for (int a=0;a<3;a++) eta[n*3+a] = y[n*3+a] - yh[n*3+a];
}
// 잔차·헤시안 곱 접기: acc_j += c v_o, v_o = 0  ->  v_j += T_j acc_j
__global__ void xa_fold(const int* IP, const int* slot, const float* M, const unsigned char* fixd, float* v, float* acc) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int s = slot[n];
    if (s < 0 || (fixd && fixd[n])) return;
    const float* c = M + (size_t)s*125;
    int nb1 = IP[7], nb2 = IP[8];
    float vv[3] = {v[n*3], v[n*3+1], v[n*3+2]};
    v[n*3] = 0.f; v[n*3+1] = 0.f; v[n*3+2] = 0.f;
    for (int t=0;t<125;t++) {
        float w = c[t];
        if (!(w > 0.f)) continue;
        int j = xa_nb(n, t, nb1, nb2);
        for (int a=0;a<3;a++) atomicAdd(&acc[j*3+a], w*vv[a]);
    }
}
__global__ void xa_fold2(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, float* acc, float* v) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float a3[3] = {acc[n*3], acc[n*3+1], acc[n*3+2]};
    if (a3[0] == 0.f && a3[1] == 0.f && a3[2] == 0.f) return;
    acc[n*3] = 0.f; acc[n*3+1] = 0.f; acc[n*3+2] = 0.f;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, a3, t);
    for (int d=0;d<3;d++) v[n*3+d] += t[d];
}
__global__ void xa_gg(const int* IP, const float* r, double* SC) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) for (int d=0;d<3;d++) { double q = r[n*3+d]; s += q*q; }
    xf_wsum(s, &SC[1]);
}
// PCG 방향 펼치기: Tp_o = p_o = Σ c (T p)_j
__global__ void xa_expand(const int* IP, const int* slot, const float* M, const unsigned char* fixd, float* p, float* Tp) {
    if (XF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int s = slot[n];
    if (s < 0 || (fixd && fixd[n])) return;
    const float* c = M + (size_t)s*125;
    int nb1 = IP[7], nb2 = IP[8];
    float t3[3] = {0.f, 0.f, 0.f};
    for (int t=0;t<125;t++) {
        float w = c[t];
        if (!(w > 0.f)) continue;
        int j = xa_nb(n, t, nb1, nb2);
        for (int a=0;a<3;a++) t3[a] += w*Tp[j*3+a];
    }
    for (int a=0;a<3;a++) { Tp[n*3+a] = t3[a]; p[n*3+a] = t3[a]; }
}
// (B) APIC P2G 아핀 몫: (mv)_i += w m_p C_p (x_i − x_p), x_i − x_p = A (i − f)
__global__ void xb_p2g(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* Am, const float* C, float* mv) {
    if (XF_SKIP) return;
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float Cp[9]; for (int k=0;k<9;k++) Cp[k] = C[(size_t)p*9+k];
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float d[3] = {(float)i - fx[p*3], (float)j - fx[p*3+1], (float)k - fx[p*3+2]};
        float xd[3]; for (int a=0;a<3;a++) xd[a] = Am[a*3]*d[0] + Am[a*3+1]*d[1] + Am[a*3+2]*d[2];
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], ww*(Cp[a*3]*xd[0] + Cp[a*3+1]*xd[1] + Cp[a*3+2]*xd[2]));
    }
}
// (B) APIC G2P: C_p = 4 Σ_i w v_i d_iᵀ A⁻¹ (D_p⁻¹ = 4 A⁻ᵀ A⁻¹, 이차 B-스플라인)
__global__ void xb_g2p(int N, const int* IP, const int* pb, const float* fx, const float* Ai, const float* vk, float* C) {
    if (XF_SKIP) return;
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float Bg[9] = {0,0,0, 0,0,0, 0,0,0};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float d[3] = {(float)i - fx[p*3], (float)j - fx[p*3+1], (float)k - fx[p*3+2]};
        float ww = w[0][i]*w[1][j]*w[2][k];
        for (int a=0;a<3;a++) { float v = ww*vk[nid*3+a]; for (int c=0;c<3;c++) Bg[a*3+c] += v*d[c]; }
    }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) C[(size_t)p*9+a*3+b] = 4.f*(Bg[a*3]*Ai[b] + Bg[a*3+1]*Ai[3+b] + Bg[a*3+2]*Ai[6+b]);
}
// 구속 입자 (방출 전) C_p = 0 (ao_vzero·aq_vset 가 v 를 정한 입자)
__global__ void xb_czero(int N, const float* Trel, const float* tbuf, float* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (Trel[p] > tbuf[0]) for (int k=0;k<9;k++) C[(size_t)p*9+k] = 0.f;
}
// 기록: 입자 Σm·Σm v (S[0..3]), 노드 Σm·Σmv (S[4..7])
__global__ void xf_psum(int N, const int* IP, const float* m, const float* v, double* S) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0;
    if (p < N && !IP[13]) { double mp = m[p]; a = mp; b = mp*v[p*3]; c = mp*v[p*3+1]; d = mp*v[p*3+2]; e = mp*(fabs((double)v[p*3]) + fabs((double)v[p*3+1]) + fabs((double)v[p*3+2])); }
    xf_wsum(a, &S[0]); xf_wsum(b, &S[1]); xf_wsum(c, &S[2]); xf_wsum(d, &S[3]); xf_wsum(e, &S[8]);
}
__global__ void xf_nsum(const int* IP, const float* mn, const float* mv, double* S) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    if (n < IP[9] && !IP[13]) { a = mn[n]; b = mv[n*3]; c = mv[n*3+1]; d = mv[n*3+2]; }
    xf_wsum(a, &S[4]); xf_wsum(b, &S[5]); xf_wsum(c, &S[6]); xf_wsum(d, &S[7]);
}
'''


def add_src(src):
    if MARK in src or not ("Tapply(" in src and "blocksum_add(" in src and "void bwf(" in src):
        return src
    INFO["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + SRC + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _cnt(name):
    INFO["hooks"][name] = INFO["hooks"].get(name, 0) + 1


def _dump():
    try:
        with open(os.path.join(CFG["out"], "xfinfo.json"), "w") as f_:
            json.dump(dict(CFG=CFG, **INFO), f_, ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _on(s):
    return getattr(s, "_xf", None) is not None


class St:
    def __init__(h, s):
        h.s = s
        dev = s.A.device
        A = s.A.detach().double().reshape(3, 3).cpu()
        h.idet = float(1.0 / abs(float(torch.linalg.det(A))))
        h.Am = s.A.detach().float().reshape(9).contiguous().clone()
        h.Aim = s.Ai.detach().float().reshape(9).contiguous().clone()
        h.S = torch.zeros(16, dtype=torch.float64, device=dev)
        h.R = torch.zeros(4, dtype=torch.float64, device=dev)
        h.cnt = torch.zeros(8, dtype=torch.int32, device=dev)
        h.cf = torch.zeros(4, dtype=torch.float32, device=dev)
        h.C = torch.zeros(int(s.N) * 9, dtype=torch.float32, device=dev) if CFG["b"] else None
        h.capN = 0
        h.nstep = 0
        h.f = open(os.path.join(CFG["out"], "xf.jsonl"), "a") if CFG["dg"] else None
        INFO["solvers"].append(dict(cls=type(s).__name__, N=int(s.N), A=[float(v) for v in A.reshape(-1)], cell_vol=1.0 / h.idet, a=CFG["a"], b=CFG["b"]))
        print(f"[절 CM109 xfer] {type(s).__name__}: (A) 받침 판정 {'켬' if CFG['a'] else '끔'}, (B) APIC {'켬' if CFG['b'] else '끔'}, 입자 {int(s.N)}, 칸 부피 {1.0 / h.idet:.6g} m³", flush=True)
        _dump()

    def alloc(h):
        s = h.s
        if not CFG["a"] or (h.capN == s.capN and getattr(h, "phi", None) is not None):
            return
        dev = s.A.device
        h.capN = int(s.capN)
        h.capT = h.capN // 4 + 1024
        h.phi = torch.zeros(h.capN, dtype=torch.float32, device=dev)
        h.slot = torch.full((h.capN,), -1, dtype=torch.int32, device=dev)
        h.M = torch.zeros(h.capT * 125, dtype=torch.float32, device=dev)
        h.acc = torch.zeros(h.capN * 3, dtype=torch.float32, device=dev)
        INFO["nalloc"] += 1

    def read(h, t):
        if h.f is None:
            return
        S = h.S.cpu().numpy()
        c = h.cnt.cpu().tolist()
        cf = h.cf.cpu().tolist()
        mp, mn = float(S[0]), float(S[4])
        pref = max(float(S[8]), 1e-300)
        row = dict(f=h.nstep, t=float(t), mp=mp, mn=mn, dm=abs(mn - mp) / max(mp, 1e-300),
                   dP=max(abs(float(S[5 + a]) - float(S[1 + a])) for a in range(3)) / pref, Pp=[float(S[1 + a]) for a in range(3)], Pn=[float(S[5 + a]) for a in range(3)])
        if CFG["a"]:
            nn = int(h.s.IP[9])
            ph = h.phi[:nn]
            act = h.s.mn[:nn] > 0                                                      # 합친 뒤 (묶는 노드는 0) -- φ 분위는 φ > 0 노드로
            pp = ph[ph > 0]
            q = torch.quantile(pp[torch.randperm(pp.numel(), device=pp.device)[:200000]], torch.tensor([0.1, 0.5, 0.9, 0.99], device=pp.device)).tolist() if pp.numel() else []
            row.update(nact=c[0], ncand=c[1], over=c[2], iso=c[3], tied=c[4], m_act=cf[0], mf_tied=(cf[1] / cf[0] if cf[0] > 0 else None), mf_iso=(cf[2] / cf[0] if cf[0] > 0 else None),
                       phi_q=q, full_ref=(float(h.R[0] / h.R[1]) if float(h.R[1]) > 0 else None), n_full=int(h.R[1]), n_phi_ge1=int((ph >= 1.0).sum()), nact_after=int(act.sum()))
        h.f.write(json.dumps(row) + "\n"); h.f.flush()
        h.nstep += 1


def _post(s, ok, name, args):
    h = s._xf
    i32 = np.int32
    if name == "rt_p2g":                                                               # N x v F m vol pb fx pcd cidx IP A C badq badc mn mv pa ma
        N = int(args[0])
        IP, pb, fx, m, vol, mn, mv, ma = args[10], args[6], args[7], args[4], args[5], args[15], args[16], args[18]
        if CFG["b"]:
            ok(s, "xb_p2g", N, (args[0], IP, pb, fx, m, P(h.Am), P(h.C), mv), 128); _cnt("xb_p2g")
        if CFG["a"]:
            ok(s, "rt_zero_f", h.capN, (i32(h.capN), P(None), i32(0), i32(1), P(h.phi)), 256); ok(s, "rt_zero_i", 8, (i32(8), P(h.cnt)), 256); ok(s, "rt_zero_f", 4, (i32(4), P(None), i32(0), i32(1), P(h.cf)), 256)
            ok(s, "xa_phi", N, (args[0], IP, pb, fx, vol, np.float32(h.idet), P(h.phi)), 128)
            ok(s, "rt_zero_d", 4, (i32(4), P(None), i32(0), i32(1), P(h.R)), 256)
            ok(s, "xa_full", h.capN, (IP, mn, P(s.occ), P(h.phi), P(h.R)), 256)
            ok(s, "xa_scale", h.capN, (IP, P(h.R), P(h.phi)), 256)
            ok(s, "xa_slot", h.capN, (IP, mn, P(h.phi), P(h.slot), P(h.M), P(h.cnt), P(h.cf), i32(h.capT)), 256)
            ok(s, "xa_pair", N, (args[0], IP, pb, fx, m, mn, P(h.phi), P(h.slot), P(h.M)), 128)
            ok(s, "xa_norm", h.capN, (IP, mn, P(h.slot), P(h.M), P(h.cnt), P(h.cf)), 256)
            ok(s, "xa_merge", h.capN, (IP, P(h.slot), P(h.M), mn, mv, ma), 256); _cnt("xa_merge")
        ok(s, "rt_zero_d", 16, (i32(16), P(None), i32(0), i32(1), P(h.S)), 256)
        ok(s, "xf_psum", N, (args[0], IP, m, args[2], P(h.S)), 256)
        ok(s, "xf_nsum", s.capN, (IP, mn, mv, P(h.S)), 256)
        return
    if CFG["b"]:
        if name in G2PN:
            vk = args[6] if name == "rt_g2p_ga" else args[8]
            ok(s, "xb_g2p", int(args[0]), (args[0], args[1], args[3], args[4], P(h.Aim), vk, P(h.C)), 128); _cnt("xb_g2p:" + name)
        elif name == "ao_vzero":                                                        # N IP Trel tbuf v
            ok(s, "xb_czero", int(args[0]), (args[0], args[2], args[3], P(h.C)), 256); _cnt("xb_czero")
        elif name == "aq_vset":                                                         # N Trel tbuf vbc v pvc pmk
            ok(s, "xb_czero", int(args[0]), (args[0], args[1], args[2], P(h.C)), 256); _cnt("xb_czero")
    if not CFG["a"] or getattr(h, "phi", None) is None:
        return
    fixd = P(getattr(s, "fixd", None))
    if name.startswith(ENDP):
        ok(s, "xa_proj", s.capN, (P(s.IP), P(h.slot), P(h.M), fixd, P(s.U), P(s.vk), P(s.ak if getattr(s, "ga", None) else None)), 256); _cnt("xa_proj:end")
    elif name.startswith(FLOORN) and not name.startswith("rt_floor_q"):
        ok(s, "xa_free", s.capN, (P(s.IP), P(h.slot), fixd, P(s.Ut), P(s.y), P(s.sk), P(s.pm), P(s.yh), P(s.eta)), 256); _cnt("xa_free:" + name)
    elif name == "rt_grad":                                                             # IP mn it2 sk pm n0 n1 n2 eta f B Dinv r z p dl Tp SC r0
        nr = tuple(args[5:8])
        for v in (args[12], args[18]):
            ok(s, "xa_fold", s.capN, (P(s.IP), P(h.slot), P(h.M), fixd, v, P(h.acc)), 256)
            ok(s, "xa_fold2", s.capN, (P(s.IP), args[3], args[4], *nr, P(h.acc), v), 256)
        ok(s, "rt_sc_clear", 1, (args[17], i32(1), i32(2)), 1)
        ok(s, "xa_gg", s.capN, (P(s.IP), args[12], args[17]), 256); _cnt("xa_fold:rt_grad")
    elif name == "rt_ap":                                                               # IP mn it2 sk pm n0 n1 n2 p KTp Ap SC Kb
        ok(s, "xa_fold", s.capN, (P(s.IP), P(h.slot), P(h.M), fixd, args[10], P(h.acc)), 256)
        ok(s, "xa_fold2", s.capN, (P(s.IP), args[3], args[4], *tuple(args[5:8]), P(h.acc), args[10]), 256); _cnt("xa_fold:rt_ap")
    elif name == "rt_pinit":                                                            # IP z sk pm n0 n1 n2 p Tp
        ok(s, "xa_expand", s.capN, (P(s.IP), P(h.slot), P(h.M), fixd, args[7], args[8]), 256); _cnt("xa_expand:rt_pinit")
    elif name == "rt_cg2":                                                              # IP SC sk pm n0 n1 n2 z p Tp KTz
        ok(s, "xa_expand", s.capN, (P(s.IP), P(h.slot), P(h.M), fixd, args[8], args[9]), 256); _cnt("xa_expand:rt_cg2")


def _pre(s, ok, name, args):
    h = s._xf
    if CFG["a"] and getattr(h, "phi", None) is not None and name == "rt_ga_step0":
        ok(s, "xa_proj", s.capN, (P(s.IP), P(h.slot), P(h.M), P(getattr(s, "fixd", None)), P(s.U), P(s.vk), P(s.ak)), 256); _cnt("xa_proj:step0")


def selfcheck(s):
    """그래프 밖 (프레임 사이) 한 번: (A) 가중 행 합·음수, (B) 아핀 속도장 재현."""
    h, out = s._xf, {}
    ok = s._xf_ok
    if CFG["a"]:
        c = h.cnt.cpu().tolist()
        nt = min(c[1], h.capT)
        sl = h.slot[:int(s.IP[9])]
        used = sl[sl >= 0].long()
        if used.numel():
            R = h.M.view(-1, 125)[used]
            out["A_rows"] = int(used.numel())
            out["A_rowsum_maxerr"] = float((R.double().sum(1) - 1).abs().max())
            out["A_min_w"] = float(R.min())
            out["A_nnz_mean"] = float((R > 0).sum(1).float().mean())
            out["A_ok"] = bool(out["A_rowsum_maxerr"] < 1e-5 and out["A_min_w"] >= 0)
        else:
            out["A_rows"] = 0; out["A_ok"] = nt == 0
    if CFG["b"]:
        nn = int(s.IP[9]); nb = [int(s.IP[6]), int(s.IP[7]), int(s.IP[8])]
        dev = s.A.device
        idx = torch.arange(nn, device=dev)
        g = torch.stack([idx // (nb[1] * nb[2]), (idx // nb[2]) % nb[1], idx % nb[2]], 1).double()
        gen = torch.Generator(device=dev); gen.manual_seed(7)
        G = torch.randn(3, 3, generator=gen, device=dev, dtype=torch.float64)
        c0 = torch.randn(3, generator=gen, device=dev, dtype=torch.float64)
        xw = g @ s.A.double().T                                                      # 노드 위치 (격자 원점 차이는 상수 -> c0 에 흡수)
        vt = torch.zeros(s.vk.numel(), dtype=torch.float32, device=dev)
        vt[:nn * 3] = (c0[None] + xw @ G.T).reshape(-1).float()
        Ct = torch.zeros_like(h.C)
        ok(s, "xb_g2p", int(s.N), (np.int32(s.N), P(s.IP), P(s.Np), P(s.fx), P(h.Aim), P(vt), P(Ct)), 128)
        torch.cuda.synchronize()
        Cm = Ct.view(-1, 3, 3).double()
        out["B_affine_relerr"] = float((Cm - G[None]).abs().max() / G.abs().max())
        out["B_ok"] = out["B_affine_relerr"] < 1e-3
        out["B_C_absmax"] = float(h.C.abs().max())
    out["ok"] = all(out.get(k, True) for k in ("A_ok", "B_ok"))
    return out


def install(out, a=0, b=0, dg=1, ck=0):
    from . import fem_an_rt as ART
    CFG.update(a=int(a), b=int(b), dg=int(dg), ck=int(ck), out=out)
    for fn in ("xf.jsonl", "xfck.json"):
        try:
            os.remove(os.path.join(out, fn))
        except OSError:
            pass
    ob = ART.build_src
    ART.build_src = lambda *a_, **kw: add_src(ob(*a_, **kw))
    R = ART.RTSolver
    oinit, oalloc, ok, oc = R.__init__, R.alloc, R.k, R.ctl

    def init(s, *a_, **kw):
        oinit(s, *a_, **kw)
        if getattr(s, "_xf", None) is not None:
            return
        assert not getattr(s, "det", False), "절 CM109 xfer: 결정적 판 (--det) 은 입자 정렬이 C 를 안 옮김 -- 안 씀"
        s._xf_ok = ok
        s._xf = St(s)
        s._xf.alloc()
        if CFG["b"]:                                                                   # 입자 상태 C 를 프레임 백업·되돌림에 (데움 실행·넘침 다시 하기·걸음 쪼개기)
            ob_, or_ = s._bk, s._rs

            def _bk():
                return tuple(ob_()) + (s._xf.C.clone(),)

            def _rs(bk):
                or_(bk[:-1])
                s._xf.C.copy_(bk[-1])
            s._bk, s._rs = _bk, _rs
        st0 = s.step

        def step(t, *a2, **kw2):
            s._xf.alloc()
            r = st0(t, *a2, **kw2)
            h = s._xf
            try:
                h.read(t)
                if CFG["ck"] and not getattr(h, "_ckd", False) and h.nstep >= 3:
                    h._ckd = True
                    o = selfcheck(s)
                    INFO["check"] = o
                    json.dump(o, open(os.path.join(CFG["out"], "xfck.json"), "w"), indent=1, default=str)
                    print("[절 CM109 xfer] 검사 " + ("통과 " if o.get("ok") else "실패 ") + json.dumps(o, default=str), flush=True)
            except Exception as e_:
                INFO["err"] = f"read: {type(e_).__name__}: {e_}"[:600]
            _dump()
            return r
        s.step = step

    def alloc(s, need):
        r = oalloc(s, need)
        if getattr(s, "_xf", None) is not None:
            s._xf.alloc()
        return r

    def k(s, name, n, args, b=128):
        if _on(s) and not getattr(s, "_xfin", False) and not name.startswith(("xa_", "xb_", "xf_")):
            s._xfin = True
            try:
                _pre(s, ok, name, args)
            finally:
                s._xfin = False
        r = ok(s, name, n, args, b)
        if not _on(s) or getattr(s, "_xfin", False) or name.startswith(("xa_", "xb_", "xf_")):
            return r
        s._xfin = True
        try:
            _post(s, ok, name, args)
        except Exception as e_:
            INFO["err"] = f"{name}: {type(e_).__name__}: {e_}"[:600]; _dump(); raise
        finally:
            s._xfin = False
        return r

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 1 and _on(s) and CFG["a"] and getattr(s._xf, "phi", None) is not None:
            h = s._xf
            s._xfin = True
            try:
                ok(s, "xa_free2", s.capN, (P(s.IP), P(h.slot), P(getattr(s, "fixd", None)), P(s.sk), P(s.pm), P(s.yh), P(s.y), P(s.eta)), 256); _cnt("xa_free2:ctl1")
            finally:
                s._xfin = False
        return r
    R.__init__, R.alloc, R.k, R.ctl = init, alloc, k, ctl
    import atexit
    atexit.register(_dump)
    print(f"[절 CM109] xfer 켬 -- (A) P2G 받침 판정 {CFG['a']} (φ < ½ 노드 질량·운동량을 일관 질량 결합 가중 이웃에 합치고 위치·속도·걸음 증분을 이웃 볼록 결합으로), "
          f"(B) G2P APIC {CFG['b']} (Jiang 2015), 기록 {CFG['dg']}, 검사 {CFG['ck']}", flush=True)
    _dump()
