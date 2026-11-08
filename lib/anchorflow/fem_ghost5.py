"""절 CM55 묶음 ghost5 (fem_ghost4 복사 + 면 가중 AF_G5_W): 면 벌점 (강성·고스트 질량 모두) 에 면마다 가중 w_F (양쪽 칸 점유 φ = min(1, 칸 입자 부피 합 / h³) 의 연속 함수, 문턱 없음) 를 곱함.
  w 방식 (AF_G5_W): 0 = 1 (ghost4 그대로) · 1 = φ_max (1 − φ_min) · 2 = Frei, Knoke, Steinbach, Wenske, Wick 2024 (arXiv:2402.00209 2.3 절) Σ_{K ∋ F} ½ w_max^{1−2κ_K} (w_max = AF_G5_WMAX, 그 논문 수치 예 3) · 3 = 1 − ½(φ_a + φ_b) · 4 = φ_max.
  진단 (결과 불변): ediag = 걸음 끝 y 방향 헤시안 몫 (요소 탄성 yᵀK y · 벌점 k yᵀG_w y · 고스트 질량 · 집중 질량), 칸 밀도로 나눈 면 갈래별.
원래 절 CM53 묶음 ghost4 (fem_ghost3 복사 + 면 한정 방식 AF_G4_THIN 0~3: 2 = 5x5x5 열림 (두꺼운 칸 = 체비셰프 반경 2 안 124 칸이 모두 활성인 칸의 반경 2 안), 3 = 내부 칸-경계 칸 면만; 환경 AF_G4_*).
원래 절 CM51 묶음 ghost3: 작은 받침 안정화 ghost penalty 3 -- (1) 고스트 질량 (Stoter 외 2023) + (2) 얇은 구조 면 빼기 + (3) 칸별 커널 (가속). 감싸개 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로.

강성 벌점 (절 CM47·CM48 그대로): E_K = ½ k Σ_F Σ_{t,t'} (M⊗M)_{tt'} D_t(y)·D_t'(y),  k = γ̄ (2μ+λ) h  (= γ_a h^{-2} h^{2p+1}, p = 2; 칸 단위 3 차 차분 D, 면 위 이차 B-스플라인 조각 질량 M).
(1) 고스트 질량 (Stoter, Divi, van Brummelen, Larson, de Prenter, Verhoosel 2023, CMAME 412:116074, arXiv:2302.07019, 식 (27)~(29)):
      M_γ(ü, v) = Σ_F ∫_F ρ γ_M [[∂_n^{k+1} ü]]·[[∂_n^{k+1} v]] dS,  γ_M = γ̄_M h^{2p+1}   (최대 연속 B-스플라인 k = p − 1 -> 면 위 p 차 법선 미분 점프, 강성 식 (25)·(26a) γ_K = γ̄_K h^{2p−1} 와 같은 면·같은 점프).
    이 판 (집중 질량 노드별 갱신 위의 증분 포텐셜): 걸음 풀이 관성 몫 ½ Σ m/τ² |y − ŷ|² 에 ½ (k_M/τ²) (y − ŷ)ᵀ G (y − ŷ) 를 더함 (G = 강성과 같은 면 연산자 Σ_F C_Fᵀ (M⊗M) C_F),
      k_M = γ̄_M ρ̄ h^{2p+1} h^{-2} = γ̄_M ρ̄ h³ (kg),  ρ̄ = Σ 입자 질량 / Σ 입자 부피 (씬 평균 밀도), τ² = 그 걸음 관성 계수 (새 실행기 걸음별 sg tau, 옛 실행기 s.tau),
      γ̄_M = γ̄_K = γ̄ ('같은 크기', Stoter 외 결론). 균일·선형 장에는 G = 0 이라 중력·강체 운동에 힘 없음. 걸음 끝 속도·가속도 갱신 (운동학) 과 P2G·G2P 는 그대로.
    걸음 밖 요소 커널 (새 실행기 첫 프레임 a0 = g − f(0)/m') 에는 질량 몫 없음 (걸음 시작 커널 rt_ga_step0·rt_step0g·ao_floor_n 뒤부터 rt_gauss 까지만).
(2) 얇은 구조 빼기 (위상·기하 정의, 조절 값 없음): 내부 칸 = 26 이웃이 모두 활성인 활성 칸 (= 경계 칸 아님). 두꺼운 칸 = 닫힌 27 이웃 안에 내부 칸이 있는 활성 칸
      = 활성 칸이 3x3x3 덩어리로 모두 찬 곳에 속하는 칸 (활성 칸 집합의 3x3x3 형태학적 열림; 3 칸 = 이차 B-스플라인 노드 하나의 받침 폭).
      벌점 면 = 절 CM47 면 (축 이웃 두 칸 모두 활성, 하나 이상 경계 칸) 중 **두 칸 모두 두꺼운 칸** 인 면. 어느 방향으로든 3 칸보다 얇은 곳 (잎·다리) 과 그 이음 면은 뺌.
(3) 칸별 커널: 면 목록 (원자 추가) 대신 칸마다 3 비트 면 표지 (+x·+y·+z 면). 칸 하나가 자기 면들을 한 스레드에서 -- 칸 27 노드를 한 번 읽고 축마다 바깥 층 9 노드만 더 읽음,
      칸 27 노드 몫은 스레드 안에서 모아 한 번만 원자 합 (면 3 개 칸: 원자 합 324 -> 162, 읽기 324 -> 162), 강성·질량 한 커널 (힘·곱은 D(k y + k_M/τ² (y − ŷ)) 한 번),
      M⊗M 곱은 분리 (M D Mᵀ, 243 -> 54 곱/성분), 성분별 차례 (레지스터). 정의·식은 절 CM47 커널 (fem_ghost1.GP_SRC) 과 같음 -- 검사·비교용으로 옛 커널도 같이 컴파일.
"""
import json
import math
import os

import numpy as np
import torch

from .fem_ghost1 import GP_SRC
from . import fem_ghost2 as G2

GAMMA = float(os.environ.get("AF_G4_GAMMA", "1e-1"))                                      # γ̄ (강성 γ̄_K = 질량 γ̄_M)
MASS = int(os.environ.get("AF_G4_MASS", "1"))                                             # (1) 고스트 질량 켬
FIXX = int(os.environ.get("AF_G4_FIXX", "0"))                                            # 절 CM53: 구속 (cuboid 고정) 노드에 닿는 면 빼기 (면 36 노드 중 고정 노드가 있으면)
WMODE = int(os.environ.get("AF_G5_W", "0"))                                              # 절 CM55 면 가중 방식 (0 = 없음)
WMAX = float(os.environ.get("AF_G5_WMAX", "3"))                                           # 방식 2 의 w_max (Frei 외 2024 수치 예 값)
THIN = int(os.environ.get("AF_G4_THIN", "1"))                                             # 면 한정 방식 (절 CM53): 0 없음 (절 CM47 면) · 1 3x3x3 열림 (절 CM51) · 2 5x5x5 열림 · 3 내부 칸-경계 칸 면
ST = dict(on=False, said=False, nk={}, nmass=0, nstep=0)
ELEM = ("rt_elem", "aq_elem", "ao_elem", "ao_elem_s")
MV = ("rt_mv", "rt_mv2", "rt_mv2s", "ao_mv_q", "ao_mv_h", "ao_mv_c", "ao_mv_s")
STEP0 = {"rt_ga_step0": 9, "rt_step0g": 3}                                                # 새 실행기 걸음 시작 (τ 인자 자리)
REC = ELEM + MV + ("rt_elem_lsN", "en_phi_n")

GQ_SRC = r'''
// ---- 절 CM53 ghost4 칸별 커널 (lib/anchorflow/fem_ghost4.py; 절 CM51 ghost3 커널 + 면 한정 방식 2·3) ----
__constant__ float GQ_M[9] = {6.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 54.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 6.f/120.f};
// 축 a, 층 l (0..2 = 칸 노드), 가로 (i, j) -> 칸 노드 번호 (칸 기준 오프셋 (o0, o1, o2) -> o0*9 + o1*3 + o2); t1 = (a==0)?1:0, t2 = (a==2)?1:2 (gp_ids 와 같은 배치)
__device__ __forceinline__ int gq_cn(int a, int l, int i, int j) { return (a == 0) ? (l*9 + i*3 + j) : ((a == 1) ? (i*9 + l*3 + j) : (i*9 + j*3 + l)); }
// 바깥 층 (l = 3) 노드 id: base + 3 s_a + i s_t1 + j s_t2
__device__ __forceinline__ int gq_en(int base, int a, int i, int j, int s0, int s1) {
    return (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
}
// MD = M D Mᵀ (3x3, M 대칭)
__device__ __forceinline__ void gq_MDM(const float D[9], float MD[9]) {
    float T[9];
#pragma unroll
    for (int i=0;i<3;i++)
#pragma unroll
        for (int j=0;j<3;j++) T[i*3+j] = GQ_M[i*3]*D[j] + GQ_M[i*3+1]*D[3+j] + GQ_M[i*3+2]*D[6+j];
#pragma unroll
    for (int i=0;i<3;i++)
#pragma unroll
        for (int j=0;j<3;j++) MD[i*3+j] = T[i*3]*GQ_M[j*3] + T[i*3+1]*GQ_M[j*3+1] + T[i*3+2]*GQ_M[j*3+2];
}
__device__ __forceinline__ int gq_base(const int* IP, const int* cb, int c, int* s0, int* s1) {
    int nb1 = IP[7], nb2 = IP[8]; *s1 = nb2; *s0 = nb1*nb2;
    return (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
}
// ---- 절 CM55 ghost5: 칸 점유 φ_c = min(1, 칸 입자 부피 합 / h³) (rt_gauss 가 읽는 칸 누적 C[c*30+28] = 그 칸 가우스 점 부피 Vq 합, h³ = |det A|), 칸 밀도 ρ_c = C[29]/C[28]
__global__ void gq5_cell(const int* IP, const float* C, const float* Am, float* phi, float* rhoc) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    float det = Am[0]*(Am[4]*Am[8] - Am[5]*Am[7]) - Am[1]*(Am[3]*Am[8] - Am[5]*Am[6]) + Am[2]*(Am[3]*Am[7] - Am[4]*Am[6]);
    float V = C[(size_t)c*30 + 28];
    phi[c] = fminf(V / fabsf(det), 1.f);
    rhoc[c] = (V > 0.f) ? C[(size_t)c*30 + 29] / V : 0.f;
}
// 면 가중 fw[c*3+a] (면 = 칸 c 와 +a 이웃 c2, 표지 비트가 선 면만): 1 = φ_max (1 − φ_min), 2 = Frei 외 2024 Σ_K ½ w_max^{1−2κ_K}, 3 = 1 − ½(φ_a + φ_b), 4 = φ_max, 그 밖 = 1
__global__ void gq5_w(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const float* phi, int mode, float wmax, float* fw) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int m = fm[c];
    for (int a=0;a<3;a++) {
        float w = 0.f;
        if (m & (1 << a)) {
            int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
            int c2 = (x[a] < n[a]) ? cidx[(x[0]*n[1] + x[1])*n[2] + x[2]] : -1;
            float pa = phi[c], pb = (c2 >= 0) ? phi[c2] : 0.f, lo = fminf(pa, pb), hi = fmaxf(pa, pb);
            if (mode == 1) w = hi*(1.f - lo);
            else if (mode == 2) w = 0.5f*powf(wmax, 1.f - 2.f*pa) + 0.5f*powf(wmax, 1.f - 2.f*pb);
            else if (mode == 3) w = 1.f - 0.5f*(pa + pb);
            else if (mode == 4) w = hi;
            else w = 1.f;
        }
        fw[c*3+a] = w;
    }
}
// ---- 면 표지 (프레임마다 rt_gauss 뒤): 경계 칸 -> 두꺼운 칸 -> 칸별 3 비트 면 표지; cnt = [경계 칸, 면 (빼기 뒤), 면 (빼기 앞), 두꺼운 칸]
__global__ void gq_zero(int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    cnt[0] = 0; cnt[1] = 0; cnt[2] = 0; cnt[3] = 0;
}
__global__ void gq_cut(const int* IP, const int* cb, const int* cidx, unsigned char* cut, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) k = 1;
    }
    cut[c] = (unsigned char)k;
    if (k) atomicAdd(&cnt[0], 1);
}
// 두꺼운 칸 (비트 0): thin 0·3 = 모두; thin 1 = 닫힌 27 이웃에 내부 칸 (활성·경계 아님) 이 있음 (3x3x3 열림);
// thin 2 = 이 커널은 비트 1 (5x5x5 내부 칸 = 체비셰프 반경 2 안 124 칸이 모두 활성) 만 쓰고, gq_thk2 가 비트 0 = 반경 2 안에 5x5x5 내부 칸이 있음 (5x5x5 열림)
__global__ void gq_thk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk, int thin, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int k = 0;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2];
    if (thin == 0 || thin == 3) k = 1;
    else if (thin == 1) {
        for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
            int X = x+dx, Y = y+dy, Z = z+dz;
            if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
            int c2 = cidx[(X*n1 + Y)*n2 + Z];
            if (c2 >= 0 && !cut[c2]) k = 1;
        }
    } else {
        int in = 1;
        for (int dx=-2;dx<=2 && in;dx++) for (int dy=-2;dy<=2 && in;dy++) for (int dz=-2;dz<=2 && in;dz++) {
            int X = x+dx, Y = y+dy, Z = z+dz;
            if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) in = 0;
        }
        thk[c] = (unsigned char)(in << 1);
        return;
    }
    thk[c] = (unsigned char)k;
    if (k && cnt) atomicAdd(&cnt[3], 1);
}
__global__ void gq_thk2(const int* IP, const int* cb, const int* cidx, unsigned char* thk, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-2;dx<=2 && !k;dx++) for (int dy=-2;dy<=2 && !k;dy++) for (int dz=-2;dz<=2 && !k;dz++) {
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
        int c2 = cidx[(X*n1 + Y)*n2 + Z];
        if (c2 >= 0 && (thk[c2] & 2)) k = 1;
    }
    thk[c] = (unsigned char)((thk[c] & 2) | k);
    if (k && cnt) atomicAdd(&cnt[3], 1);
}
// 벌점 면: 절 CM47 면 (축 이웃 두 칸 활성, 하나 이상 경계 칸) 중 thin 3 = 한쪽 내부 칸·한쪽 경계 칸 (cut 다름), 그 밖 = 두 칸 모두 두꺼움 (비트 0)
__global__ void gq_face(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, const unsigned char* thk, unsigned char* fm, int thin, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int m = 0, na = 0, nk = 0;
    for (int a=0;a<3;a++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
        if (x[a] >= n[a]) continue;
        int c2 = cidx[(x[0]*n[1] + x[1])*n[2] + x[2]];
        if (c2 < 0 || !(cut[c] || cut[c2])) continue;
        na++;
        if ((thin == 3) ? (cut[c] != cut[c2]) : ((thk[c] & 1) && (thk[c2] & 1))) { m |= (1 << a); nk++; }
    }
    fm[c] = (unsigned char)m;
    if (cnt) { if (nk) atomicAdd(&cnt[1], nk); if (na) atomicAdd(&cnt[2], na); }
}
// ---- 구속 노드에 닿는 면 빼기 (FIXX; rt_bc_box 뒤): 면 표지 비트 a 의 면 36 노드 (칸 27 노드 중 그 축 층 0..2 + 바깥 층 9) 중 fixd 가 있으면 비트를 지움, cnt[1] 에서 뺌
__global__ void gq_fix(const int* IP, const int* cb, const unsigned char* fixd, unsigned char* fm, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    int cf = 0;
    for (int o=0;o<27;o++) if (fixd[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)]) { cf = 1; break; }
    int m2 = m, nr = 0;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        int f = cf;
        for (int i=0;i<3 && !f;i++) for (int j=0;j<3 && !f;j++) if (fixd[gq_en(base, a, i, j, s0, s1)]) f = 1;
        if (f) { m2 &= ~(1 << a); nr++; }
    }
    if (m2 != m) { fm[c] = (unsigned char)m2; if (cnt) atomicSub(&cnt[1], nr); }
}
// ---- 곱: out += kk G dy (f32), kk = k + k_M/τ² (걸음 안) 또는 k
__global__ void __launch_bounds__(128) gq_mv(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float kk, const float* dy, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
#pragma unroll 1
    for (int q=0;q<3;q++) {
        float u[27], acc[27];
#pragma unroll
        for (int o=0;o<27;o++) { u[o] = dy[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q]; acc[o] = 0.f; }
#pragma unroll
        for (int a=0;a<3;a++) {
            if (!(m & (1 << a))) continue;
            float wf = fw ? fw[c*3+a] : 1.f;                                   // 절 CM55 면 가중
            float D[9], MD[9];
#pragma unroll
            for (int i=0;i<3;i++)
#pragma unroll
                for (int j=0;j<3;j++) {
                    float ue = dy[gq_en(base, a, i, j, s0, s1)*3 + q];
                    D[i*3+j] = -u[gq_cn(a,0,i,j)] + 3.f*u[gq_cn(a,1,i,j)] - 3.f*u[gq_cn(a,2,i,j)] + ue;
                }
            gq_MDM(D, MD);
#pragma unroll
            for (int i=0;i<3;i++)
#pragma unroll
                for (int j=0;j<3;j++) {
                    float v = kk*wf*MD[i*3+j];
                    acc[gq_cn(a,0,i,j)] -= v; acc[gq_cn(a,1,i,j)] += 3.f*v; acc[gq_cn(a,2,i,j)] -= 3.f*v;
                    atomicAdd(&out[gq_en(base, a, i, j, s0, s1)*3 + q], v);
                }
        }
#pragma unroll
        for (int o=0;o<27;o++) if (acc[o] != 0.f) atomicAdd(&out[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q], acc[o]);
    }
}
// ---- 힘·블록 대각·에너지: f += G (k y + cm (y − ŷ)) (f64), B 대각 += (k + cm) c_l² M_ii M_jj, E += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ)  (yh 없으면 cm 몫 없음)
__global__ void __launch_bounds__(128) gq_force(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[11]) return;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    int m = (!IP[13] && c < IP[11]) ? fm[c] : 0;
    if (m) {
        int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
        bool hm = (yh != 0) && (cm != 0.f);
        float kk = k + (hm ? cm : 0.f);
        if (f || E) {
#pragma unroll 1
        for (int q=0;q<3;q++) {
            float u[27], w[27], acc[27];
#pragma unroll
            for (int o=0;o<27;o++) { int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3); u[o] = y[id*3+q]; w[o] = hm ? (u[o] - yh[id*3+q]) : 0.f; acc[o] = 0.f; }
#pragma unroll
            for (int a=0;a<3;a++) {
                if (!(m & (1 << a))) continue;
                float wf = fw ? fw[c*3+a] : 1.f;                               // 절 CM55 면 가중
                float Dy[9], Dd[9], Dz[9], MD[9];
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        int ie = gq_en(base, a, i, j, s0, s1);
                        float ue = y[ie*3 + q], we = hm ? (ue - yh[ie*3 + q]) : 0.f;
                        Dy[i*3+j] = -u[gq_cn(a,0,i,j)] + 3.f*u[gq_cn(a,1,i,j)] - 3.f*u[gq_cn(a,2,i,j)] + ue;
                        Dd[i*3+j] = -w[gq_cn(a,0,i,j)] + 3.f*w[gq_cn(a,1,i,j)] - 3.f*w[gq_cn(a,2,i,j)] + we;
                        Dz[i*3+j] = wf*(k*Dy[i*3+j] + (hm ? cm*Dd[i*3+j] : 0.f));
                    }
                if (E) {
                    gq_MDM(Dy, MD); float s = 0.f;
#pragma unroll
                    for (int t=0;t<9;t++) s += Dy[t]*MD[t];
                    e += 0.5*(double)k*(double)wf*(double)s;
                    if (hm) { gq_MDM(Dd, MD); s = 0.f;
#pragma unroll
                        for (int t=0;t<9;t++) s += Dd[t]*MD[t];
                        e += 0.5*(double)cm*(double)wf*(double)s; }
                }
                if (f) {
                    gq_MDM(Dz, MD);
#pragma unroll
                    for (int i=0;i<3;i++)
#pragma unroll
                        for (int j=0;j<3;j++) {
                            float v = MD[i*3+j];
                            acc[gq_cn(a,0,i,j)] -= v; acc[gq_cn(a,1,i,j)] += 3.f*v; acc[gq_cn(a,2,i,j)] -= 3.f*v;
                            atomicAdd(&f[gq_en(base, a, i, j, s0, s1)*3 + q], (double)v);
                        }
                }
            }
            if (f) {
#pragma unroll
                for (int o=0;o<27;o++) if (acc[o] != 0.f) atomicAdd(&f[(base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3))*3 + q], (double)acc[o]);
            }
        }
        }
        if (B) {
            float bd[27];
#pragma unroll
            for (int o=0;o<27;o++) bd[o] = 0.f;
#pragma unroll
            for (int a=0;a<3;a++) {
                if (!(m & (1 << a))) continue;
                float wf = fw ? fw[c*3+a] : 1.f;                               // 절 CM55 면 가중
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        float mm = kk*wf*GQ_M[i*4]*GQ_M[j*4];
                        bd[gq_cn(a,0,i,j)] += mm; bd[gq_cn(a,1,i,j)] += 9.f*mm; bd[gq_cn(a,2,i,j)] += 9.f*mm;
                        int id = gq_en(base, a, i, j, s0, s1);
                        atomicAdd(&B[id*9], mm); atomicAdd(&B[id*9+4], mm); atomicAdd(&B[id*9+8], mm);
                    }
            }
#pragma unroll
            for (int o=0;o<27;o++) if (bd[o] != 0.f) { int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3); atomicAdd(&B[id*9], bd[o]); atomicAdd(&B[id*9+4], bd[o]); atomicAdd(&B[id*9+8], bd[o]); }
        }
    }
    if (E) blocksum_add(e, E);
}
// ---- 선탐색 후보: y_c = ŷ + T(η + c δ) = a + c b, y_c − ŷ = e + c b (e = Tη) -> GQ[0..4] = Q_aa, Q_ab, Q_bb, Q_ee, Q_eb (면마다 D M Dᵀ, 칸별 스레드)
__global__ void __launch_bounds__(128) gq_ls(const int* IP, const int* cb, const unsigned char* fm, const float* fw, const float* yh, const float* eta, const float* dl, const float* sk, const int* pm,
                      float n0, float n1, float n2, double* GQ, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[11]) return;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    double qa = 0.0, qab = 0.0, qb = 0.0, qe = 0.0, qeb = 0.0;
    int m = (!IP[13] && c < IP[11]) ? fm[c] : 0;
    if (m) {
        int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
#pragma unroll 1
        for (int a=0;a<3;a++) {
            if (!(m & (1 << a))) continue;
            double wf = fw ? (double)fw[c*3+a] : 1.0;                          // 절 CM55 면 가중
            float Da[27], Db[27], De[27];
#pragma unroll
            for (int t=0;t<27;t++) { Da[t] = 0.f; Db[t] = 0.f; De[t] = 0.f; }
#pragma unroll 1
            for (int l=0;l<4;l++) {
                float cl = (l == 0) ? -1.f : ((l == 1) ? 3.f : ((l == 2) ? -3.f : 1.f));
#pragma unroll
                for (int i=0;i<3;i++)
#pragma unroll
                    for (int j=0;j<3;j++) {
                        int id = (l < 3) ? (base + ((a == 0) ? l : i)*s0 + ((a == 0) ? i : ((a == 1) ? l : j))*s1 + ((a == 2) ? l : j)) : gq_en(base, a, i, j, s0, s1);
                        float ev[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, dv[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]}, te[3], td[3];
                        Tapply(sk[id], pm[id], n0, n1, n2, ev, te); Tapply(sk[id], pm[id], n0, n1, n2, dv, td);
#pragma unroll
                        for (int q=0;q<3;q++) { int t = (i*3+j)*3 + q; Da[t] += cl*(yh[id*3+q] + te[q]); Db[t] += cl*td[q]; De[t] += cl*te[q]; }
                    }
            }
#pragma unroll
            for (int q=0;q<3;q++) {
                float A[9], Bv[9], Ev[9], MB[9], MA[9];
#pragma unroll
                for (int t=0;t<9;t++) { A[t] = Da[t*3+q]; Bv[t] = Db[t*3+q]; Ev[t] = De[t*3+q]; }
                gq_MDM(Bv, MB); gq_MDM(A, MA);
                float sa = 0.f, sab = 0.f, sb = 0.f, se = 0.f, seb = 0.f;
#pragma unroll
                for (int t=0;t<9;t++) { sa += A[t]*MA[t]; sab += A[t]*MB[t]; sb += Bv[t]*MB[t]; seb += Ev[t]*MB[t]; }
                gq_MDM(Ev, MA);
#pragma unroll
                for (int t=0;t<9;t++) se += Ev[t]*MA[t];
                qa += wf*sa; qab += wf*sab; qb += wf*sb; qe += wf*se; qeb += wf*seb;
            }
        }
    }
    blocksum_add(qa, &GQ[0]); blocksum_add(qab, &GQ[1]); blocksum_add(qb, &GQ[2]); blocksum_add(qe, &GQ[3]); blocksum_add(qeb, &GQ[4]);
}
__global__ void gq_lsfin(double* SC, int NC, int i0, const double* gate, double* GQ, float k, float cm) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) { double cd = (double)c; SC[40+i] += 0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[2]) + 0.5*(double)cm*(GQ[3] + 2.0*cd*GQ[4] + cd*cd*GQ[2]); c *= 0.5f; }
    GQ[0] = 0.0; GQ[1] = 0.0; GQ[2] = 0.0; GQ[3] = 0.0; GQ[4] = 0.0;
}
// ---- 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ) (en_phi_n 과 같은 MO 문)
__global__ void __launch_bounds__(128) gq_phi(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const int* MO, double* PH) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (!MO[0]) return;
    gq_force(IP, cb, fm, fw, k, cm, y, yh, (double*)0, (float*)0, PH + 1);
}
'''
# gq_phi 는 __global__ 안에서 __global__ 을 부를 수 없어 아래에서 본문을 복제해 만든다 (_phi_src).


def _phi_src():
    s = GQ_SRC
    i = s.index("__global__ void __launch_bounds__(128) gq_force(")
    j = s.index("// ---- 선탐색 후보")
    body = s[i:j]
    body = body.replace("gq_force(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, double* f, float* B, double* E) {",
                        "gq_phi(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const int* MO, double* PH) {\n"
                        "    if (!MO[0]) return;\n    double* f = 0; float* B = 0; double* E = PH + 1;", 1)
    k0 = s.index("// ---- 옛 실행기 --en_mo Φ")
    return s[:k0] + "// ---- 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀGy + ½ cm (y−ŷ)ᵀG(y−ŷ) (en_phi_n 과 같은 MO 문; gq_force 본문 복제, f·B 없음)\n" + body


SRC = None


def _append(src):
    global SRC
    if "gq_force(" in src:
        return src
    assert "blocksum_add(" in src and "Tapply(" in src, "ghost3: 바탕 소스에 blocksum_add·Tapply 없음"
    if SRC is None:
        SRC = _phi_src()
    i = src.rstrip().rfind("}")
    add = (GP_SRC if "gp_force(" not in src else "") + SRC
    return src[:i] + add + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def FWP(s):
    """면 가중 버퍼 포인터 (WMODE 0 이면 없음 = 가중 1)."""
    return P(s.g5w) if (WMODE and getattr(s, "g5w", None) is not None) else P(None)


def wfun(pa, pb, mode=None):
    """torch 참값: 면 가중 (gq5_w 와 같은 식)."""
    mode = WMODE if mode is None else mode
    lo, hi = torch.minimum(pa, pb), torch.maximum(pa, pb)
    if mode == 1:
        return hi * (1 - lo)
    if mode == 2:
        return 0.5 * WMAX ** (1 - 2 * pa) + 0.5 * WMAX ** (1 - 2 * pb)
    if mode == 3:
        return 1 - 0.5 * (pa + pb)
    if mode == 4:
        return hi
    return torch.ones_like(pa)


def ref_phi(s):
    """torch 참값: 칸 점유 φ (f64) · 칸 밀도."""
    nc = int(s.IP[11])
    C = s.Cacc.reshape(-1, 30)[:nc].double()
    h3 = abs(float(torch.linalg.det(s.A.double().reshape(3, 3).cpu())))
    return torch.clamp(C[:, 28] / h3, max=1.0), torch.where(C[:, 28] > 0, C[:, 29] / C[:, 28].clamp_min(1e-30), torch.zeros_like(C[:, 28]))


def face_c2(s, R):
    nb = R["IP"][6:9]
    n = [nb[0] - 2, nb[1] - 2, nb[2] - 2]
    cidx = s.cidx.reshape(-1)[:n[0] * n[1] * n[2]].long().view(*n)
    cb, fc, fa = R["cb"], R["fc"], R["fa"]
    x2 = cb[fc].clone()
    x2[torch.arange(fc.numel(), device=s.dev), fa] += 1
    return cidx[x2[:, 0], x2[:, 1], x2[:, 2]]


def ref_Ew(R, y, k, MM):
    D = G2.ref_D(R, y)
    if "w" not in R:
        return 0.5 * k * torch.einsum("fta,ts,fsa->", D, MM, D)
    return 0.5 * k * torch.einsum("f,fta,ts,fsa->", R["w"], D, MM, D)


def ref_Gw(R, v, k, MM, nnode):
    D = G2.ref_D(R, v)
    MD = torch.einsum("ts,fsa->fta", MM, D)
    if "w" in R:
        MD = MD * R["w"][:, None, None]
    out = torch.zeros(nnode, 3, dtype=torch.float64, device=v.device)
    for l in range(4):
        out.index_add_(0, R["ids"][:, l, :].reshape(-1), (k * G2.CL[l] * MD).reshape(-1, 3))
    return out


def _k_of(s):
    if getattr(s, "_gpk", None) is None:
        A = s.A.double().reshape(3, 3).cpu()
        h = abs(float(torch.linalg.det(A))) ** (1.0 / 3.0)
        kap = 2.0 * float(s.mu) + float(s.lam)
        m_, v_ = float(s.m.double().sum()), float(s.vol.double().sum())
        rho = m_ / v_ if v_ > 0 else 0.0
        s._gpk = GAMMA * kap * h
        s._g3kM = (GAMMA * rho * h ** 3) if MASS else 0.0
        s._gph, s._gpkap, s._g3rho = h, kap, rho
        print(f"[절 CM53 ghost4] γ̄ {GAMMA:g} (강성 = 질량), 질량 {MASS} · 면 한정 방식 {THIN}; μ {float(s.mu):.6g}, λ {float(s.lam):.6g}, 2μ+λ {kap:.6g}, h {h:.6g} m, "
              f"ρ̄ {rho:.6g} kg/m³ -> k = γ̄ (2μ+λ) h = {s._gpk:.6g}, k_M = γ̄ ρ̄ h³ = {s._g3kM:.6g} kg", flush=True)
    return s._gpk


def _alloc(s):
    cap = int(s.capQc)
    if getattr(s, "g3cut", None) is None or s.g3cut.numel() < cap:
        s.g3cut = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.g3thk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.g3fm = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
    if getattr(s, "g5w", None) is None or s.g5w.numel() < cap * 3:
        s.g5phi = torch.zeros(cap, device=s.dev, dtype=torch.float32)
        s.g5rho = torch.zeros(cap, device=s.dev, dtype=torch.float32)
        s.g5w = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
    if getattr(s, "gpcnt", None) is None:
        s.gpcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.gpGQ = torch.zeros(8, device=s.dev, dtype=torch.float64)


def _cnt(name):
    ST["nk"][name] = ST["nk"].get(name, 0) + 1


def cm_of(s):
    """걸음 안이면 k_M/τ², 아니면 0."""
    if not getattr(s, "_g3in", False) or not MASS:
        return 0.0
    return float(s._g3kM) * float(getattr(s, "_g3it2", 0.0))


def build_mask(s, _ok, thin, fm, cnt):
    cq = int(s.capQc)
    if cnt is not None:
        _ok(s, "gq_zero", 1, (P(cnt),), 1)
    _ok(s, "gq_cut", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(cnt if cnt is not None else s.g3dummy)), 128)
    _ok(s, "gq_thk", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.g3thk), np.int32(thin), P(cnt)), 128)
    if thin == 2:
        _ok(s, "gq_thk2", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3thk), P(cnt)), 128)
    _ok(s, "gq_face", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.g3thk), P(fm), np.int32(thin), P(cnt)), 128)


def install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_an_rt_v0 as V0
    ST["on"] = True
    _obs, _oas = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(_obs(*a, **kw))
    AORT._src = lambda *a, **kw: _append(_oas(*a, **kw))
    f32 = np.float32

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name in REC and not getattr(s, "_g3chk", False):
                if getattr(s, "_g3args", None) is None:
                    s._g3args = {}
                s._g3args[name] = tuple(args)
            r = _ok(s, name, n, args, b)
            if getattr(s, "_gpin", False) or name.startswith("gp_") or name.startswith("gq_"):
                return r
            s._gpin = True
            try:
                _hook(s, _ok, name, args)
            finally:
                s._gpin = False
            return r
        cls.k = k

    def _go(s, _ok, nm, n, args, b):
        _cnt(nm)
        _ok(s, nm, n, args, b)

    def _hook(s, _ok, name, args):
        if name == "rt_gauss":
            _k_of(s)
            if getattr(s, "det", False):
                assert s._gpk == 0.0 and s._g3kM == 0.0, "ghost3: 결정적 판 (det) 은 γ̄ = 0 대조만 (원자 합 순서)"
            assert getattr(s, "pkmod", None) is None, "ghost3: 지속형 판은 안 고침"
            _alloc(s)
            s._g3in = False                                                                     # 프레임 시작: 걸음 밖 (질량 몫 없음)
            cq = int(s.capQc)
            _go(s, _ok, "gq_zero", 1, (P(s.gpcnt),), 1)
            _go(s, _ok, "gq_cut", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.gpcnt)), 128)
            _go(s, _ok, "gq_thk", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.g3thk), np.int32(THIN), P(s.gpcnt)), 128)
            if THIN == 2:
                _go(s, _ok, "gq_thk2", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3thk), P(s.gpcnt)), 128)
            _go(s, _ok, "gq_face", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.g3thk), P(s.g3fm), np.int32(THIN), P(s.gpcnt)), 128)
            s._g5C, s._g5A = args[2], args[6]                                                   # 절 CM55: 칸 누적 C (부피·질량 합) · 격자 A
            _go(s, _ok, "gq5_cell", cq, (P(s.IP), args[2], args[6], P(s.g5phi), P(s.g5rho)), 128)
            if WMODE:
                _go(s, _ok, "gq5_w", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.g5phi), np.int32(WMODE), np.float32(WMAX), P(s.g5w)), 128)
            if not ST["said"]:
                ST["said"] = True
                print(f"[절 CM55 ghost5] 면 가중 방식 {WMODE} (w_max {WMAX:g}): 칸 점유 φ·가중 (rt_gauss 뒤, 면 표지 다음)", flush=True)
                print("[절 CM53 ghost4] 켬: 경계 칸·두꺼운 칸·칸별 면 표지 (rt_gauss 뒤), 요소·곱·에너지 커널 뒤 칸별 G 몫 (강성 + 걸음 안 질량)", flush=True)
            return
        if name == "rt_bc_box":                                                                 # 절 CM53 FIXX: 고정 노드 표시 뒤 면 빼기 (상자마다 한 번, 멱등)
            if FIXX and getattr(s, "g3fm", None) is not None:
                _go(s, _ok, "gq_fix", int(s.capQc), (P(s.IP), P(s.cb), args[4], P(s.g3fm), P(s.gpcnt)), 128)
            return
        if name in STEP0:                                                                       # 새 실행기 걸음 시작: τ
            tau = float(args[STEP0[name]])
            s._g3it2 = 1.0 / (tau * tau) if tau > 0 else 0.0
            s._g3in = True
            ST["nstep"] += 1
            return
        if name == "ao_floor_n":                                                                # 옛 실행기 걸음 시작 (ŷ 를 이 커널이 씀)
            s._g3it2 = 1.0 / (float(s.tau) ** 2)
            s._g3in = True
            ST["nstep"] += 1
            return
        if getattr(s, "gpcnt", None) is None:
            return
        k = float(s._gpk)
        cm = cm_of(s)
        if cm != 0.0:
            ST["nmass"] += 1
        nq = int(s.capQc)
        G = (P(s.IP), P(s.cb), P(s.g3fm), FWP(s))
        YH = P(s.yh) if cm != 0.0 else P(None)
        DT = bool(getattr(s, "det", False))
        if name in ELEM:
            if DT:
                args = list(args)
                if name == "rt_elem":
                    args[8] = np.int32(int(args[8]) & ~4)
                args = tuple(args)
            if name == "rt_elem":
                mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
                _go(s, _ok, "gq_force", nq, G + (f32(k), f32(cm), y, YH, f_ if mode & 1 else P(None), B if mode & 8 else P(None), E if mode & 4 else P(None)), 128)
            elif name == "aq_elem":
                _go(s, _ok, "gq_force", nq, G + (f32(k), f32(cm), args[5], YH, args[10], args[13], P(None)), 128)
            else:
                mode = int(args[9])
                _go(s, _ok, "gq_force", nq, G + (f32(k), f32(cm), args[5], YH, args[10] if mode & 1 else P(None), args[13] if mode & 8 else P(None), P(None)), 128)
        elif name in MV:
            _go(s, _ok, "gq_mv", nq, G + (f32(k + cm), args[6], args[7]), 128)
        elif name == "rt_bd":
            _go(s, _ok, "gq_force", nq, G + (f32(k), f32(cm), P(s.y), YH, P(None), args[6], P(None)), 128)
        elif name in ("rt_elem_lsN", "en_phi_n") and DT:
            return
        elif name == "rt_elem_lsN":
            assert len(args) >= 19, "ghost3: rt_elem_lsN (i0·gate 없는 옛 판) 은 안 고침"
            yh, eta, dl, sk, pm, n0, n1, n2, SC, NC, i0, gate = args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], args[16], args[17], args[18]
            _go(s, _ok, "gq_ls", nq, G + (yh, eta, dl, sk, pm, n0, n1, n2, P(s.gpGQ), gate), 128)
            _go(s, _ok, "gq_lsfin", 1, (SC, NC, i0, gate, P(s.gpGQ), f32(k), f32(cm)), 1)
        elif name == "en_phi_n":                                                                # Φ: 강성 + 질량 (it2 = 이 커널 인자)
            cmp = float(s._g3kM) * float(args[2]) if MASS else 0.0
            _go(s, _ok, "gq_phi", nq, G + (f32(k), f32(cmp), args[3], args[4] if cmp != 0.0 else P(None), args[5], args[6]), 128)
        elif name == "rt_elem_ls":
            raise RuntimeError(f"ghost3: {name} 경로는 안 고침")

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)


# ------------------------------------------------------------------------------------------------ 검사·비교 (그래프 밖, 결과 불변)
def ref_struct3(s, thin):
    """torch 로 따로 만든 면 목록 (얇은 구조 빼기 thin) + 면 36 노드 (fem_ghost2.ref_struct 의 ids 배치)."""
    R = G2.ref_struct(s)
    if not thin:
        return R
    IP = R["IP"]
    nb = IP[6:9]
    n = [nb[0] - 2, nb[1] - 2, nb[2] - 2]
    cidx = s.cidx.reshape(-1)[:n[0] * n[1] * n[2]].long().view(*n)
    act = cidx >= 0
    pad = torch.nn.functional.pad((~act).double()[None, None], (1, 1, 1, 1, 1, 1), value=1.0)
    cutg = act & (torch.nn.functional.max_pool3d(pad, 3, stride=1)[0, 0] > 0)
    inner = (act & ~cutg).double()
    mp = torch.nn.functional.max_pool3d
    if thin == 1:
        thkg = act & (mp(torch.nn.functional.pad(inner[None, None], (1, 1, 1, 1, 1, 1), value=0.0), 3, stride=1)[0, 0] > 0)
    elif thin == 2:
        pad2 = torch.nn.functional.pad((~act).double()[None, None], (2, 2, 2, 2, 2, 2), value=1.0)
        in2 = (act & ~(mp(pad2, 5, stride=1)[0, 0] > 0)).double()
        thkg = act & (mp(torch.nn.functional.pad(in2[None, None], (2, 2, 2, 2, 2, 2), value=0.0), 5, stride=1)[0, 0] > 0)
    else:
        thkg = act.clone()
    cb = R["cb"]
    fc, fa = R["fc"], R["fa"]
    x2 = cb[fc].clone()
    x2[torch.arange(fc.numel(), device=s.dev), fa] += 1
    if thin == 3:
        keep = cutg[cb[fc, 0], cb[fc, 1], cb[fc, 2]] != cutg[x2[:, 0], x2[:, 1], x2[:, 2]]
    else:
        keep = thkg[cb[fc, 0], cb[fc, 1], cb[fc, 2]] & thkg[x2[:, 0], x2[:, 1], x2[:, 2]]
    R2 = dict(R)
    R2.update(fc=fc[keep], fa=fa[keep], ids=R["ids"][keep], nf=int(keep.sum()), fkey=torch.sort(fc[keep] * 3 + fa[keep]).values, nthick=int(thkg.sum()), nact=int(act.sum()))
    return R2


def ref_struct4(s, thin):
    """ref_struct3 + FIXX (면 36 노드 중 고정 노드 있는 면 빼기)."""
    R = ref_struct3(s, thin)
    if WMODE and R["nf"] > 0:                                                                   # 절 CM55: torch 참값 면 가중 (실행이 쓴 φ = s.g5phi 로; φ 자체는 check 에서 torch 와 따로 대조)
        ph = s.g5phi[:int(s.IP[11])].double()
        R = dict(R)
        R["w"] = wfun(ph[R["fc"]], ph[face_c2(s, R)])
    if not FIXX or getattr(s, "fixd", None) is None or R["nf"] == 0:
        return R
    fx = s.fixd.bool()
    keep = ~fx[R["ids"].reshape(R["ids"].shape[0], -1)].any(1)
    R2 = dict(R)
    R2.update(fc=R["fc"][keep], fa=R["fa"][keep], ids=R["ids"][keep], nf=int(keep.sum()), fkey=torch.sort(R["fc"][keep] * 3 + R["fa"][keep]).values)
    if "w" in R:
        R2["w"] = R["w"][keep]
    return R2


def apply_fix(s, _ok, fm, cnt):
    if FIXX and getattr(s, "fixd", None) is not None:
        _ok(s, "gq_fix", int(s.capQc), (P(s.IP), P(s.cb), P(s.fixd), P(fm), P(cnt)), 128)


def mask_keys(fm, nc):
    m = fm[:nc].long()
    ks = []
    for a in range(3):
        c = torch.nonzero((m >> a) & 1).squeeze(1)
        ks.append(c * 3 + a)
    return torch.sort(torch.cat(ks)).values


def _ev(fn, reps):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000.0                                                    # µs


def check(s, fi, out, reps=200):
    """프레임 fi 걸음 뒤 (그래프 밖): 면 집합 (커널 대 torch, 빼기 끔·켬), 옛 면 목록 커널 (절 CM47) 대 칸별 커널 (같은 면 집합) 값·시간, 질량 몫 참값, 선탐색 후보·Φ."""
    torch.cuda.synchronize()
    G2._skip0(s)
    s._g3chk = True
    dev = s.dev
    rs = dict(f=int(fi))
    k0, kM = float(s._gpk), float(s._g3kM)
    kt = k0 if k0 > 0 else 1.0
    it2 = float(getattr(s, "_g3it2", 0.0)) or 1.0
    cmt = (kM if kM > 0 else 1.0) * it2
    capN, capQc = int(s.capN), int(s.capQc)
    MM = G2._MM(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(4321 + int(fi))
    f32 = np.float32
    IP = [int(q) for q in s.IP.cpu().tolist()]
    nnode, nc = IP[9], IP[11]
    _ok = lambda s_, nm, n, a, b: s_.k(nm, n, a, b)
    # A. 면 집합: 빼기 끔 (절 CM47 집합) · 켬, 커널 대 torch; 옛 면 목록 커널 (gp_cut·gp_face) 도
    fm0 = torch.zeros(capQc, device=dev, dtype=torch.uint8)
    cnt0 = torch.zeros(4, device=dev, dtype=torch.int32)
    s.g3dummy = cnt0
    s._gpin = True
    try:
        build_mask(s, _ok, 0, fm0, cnt0)
        fm1 = s.g3fm
        cut_old = torch.zeros(capQc, device=dev, dtype=torch.int32)
        fl_old = torch.zeros(3 * capQc, device=dev, dtype=torch.int32)
        cnt_old = torch.zeros(4, device=dev, dtype=torch.int32)
        s.k("gp_zero", 1, (P(cnt_old),), 1)
        s.k("gp_cut", capQc, (P(s.IP), P(s.cb), P(s.cidx), P(cut_old), P(cnt_old)), 128)
        s.k("gp_face", 3 * capQc, (P(s.IP), P(s.cb), P(s.cidx), P(cut_old), P(cnt_old), P(fl_old)), 128)
        torch.cuda.synchronize()
        R0 = ref_struct3(s, 0)
        R1 = ref_struct4(s, THIN)
        k0s, k1s = mask_keys(fm0, nc), mask_keys(fm1, nc)
        nfo = int(cnt_old[1])
        kold = torch.sort(fl_old[:nfo].long()).values
        same = lambda a, b: bool(a.numel() == b.numel() and bool((a == b).all()))
        rs.update(nact=nc, nnode=nnode, ncut=int(cnt0[0]), nface_all=int(k0s.numel()), nface_thin=int(k1s.numel()), nthick=R1.get("nthick"), nface_old=nfo,
                  set_all_eq_torch=same(k0s, R0["fkey"]), set_thin_eq_torch=same(k1s, R1["fkey"]), set_all_eq_old=same(k0s, kold),
                  run_cnt=[int(q) for q in s.gpcnt.cpu().tolist()])
        # 절 CM55: φ 커널 대 torch (지금 C·A), 실행 φ (rt_gauss 때) 대 지금 다시 돈 커널, 면 가중 커널 대 torch (실행 φ 로)
        phr, rhr = ref_phi(s)
        ph2 = torch.zeros_like(s.g5phi); rh2 = torch.zeros_like(s.g5rho)
        s.k("gq5_cell", capQc, (P(s.IP), P(s.Cacc), P(s.A), P(ph2), P(rh2)), 128)
        torch.cuda.synchronize()
        rs.update(phi_ker_ref=float((ph2[:nc].double() - phr).abs().max()), rho_ker_ref=G2._rel(rh2[:nc].double(), rhr), phi_run_eq_now=float((s.g5phi[:nc] - ph2[:nc]).abs().max()),
                  phi_q=[float(q) for q in torch.quantile(phr.float(), torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()])
        if WMODE and R1["nf"] > 0:
            wk = s.g5w[R1["fc"] * 3 + R1["fa"]].double()
            rs.update(w_ker_ref=float((wk - R1["w"]).abs().max()), w_mean=float(R1["w"].mean()), w_q=[float(q) for q in torch.quantile(R1["w"].float(), torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()])
        # B. 같은 면 집합 (빼기 끔) 에서 옛 커널 대 칸별 커널: 곱·힘·대각·에너지 값과 시간
        y = s.y.reshape(-1).contiguous()
        sc = max(float(y[:nnode * 3].abs().max()) if nnode else 0.0, 1e-4)
        v = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
        v[nnode * 3:] = 0
        Gold = (P(s.IP), P(s.cb), P(cnt_old), P(fl_old))
        o1 = torch.zeros(capN * 3, device=dev); o2 = torch.zeros(capN * 3, device=dev); o3 = torch.zeros(capN * 3, device=dev)
        s.k("gp_mv", 3 * capQc, Gold + (f32(kt), P(v), P(o1)), 128)
        s.k("gp_mv", 3 * capQc, Gold + (f32(kt), P(v), P(o2)), 128)
        s.k("gq_mv", capQc, (P(s.IP), P(s.cb), P(fm0), P(None), f32(kt), P(v), P(o3)), 128)
        torch.cuda.synchronize()
        mr = ref_Gw(R0, v[:nnode * 3], kt, MM, nnode)
        rs.update(mv_old_old=G2._rel(o2[:nnode * 3], o1[:nnode * 3].double()), mv_new_old=G2._rel(o3[:nnode * 3], o1[:nnode * 3].double()),
                  mv_new_ref=G2._rel(o3[:nnode * 3], mr), mv_old_ref=G2._rel(o1[:nnode * 3], mr))

        def fo(y_, yh_, k_, cm_, new, fmx=fm0):
            f_ = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
            B_ = torch.zeros(capN * 9, device=dev)
            E_ = torch.zeros(2, device=dev, dtype=torch.float64)
            if new:
                s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(fmx), (FWP(s) if fmx is fm1 else P(None)), f32(k_), f32(cm_), P(y_), P(yh_), P(f_), P(B_), P(E_)), 128)
            else:
                s.k("gp_force", 3 * capQc, Gold + (f32(k_), P(y_), P(f_), P(B_), P(E_)), 128)
            torch.cuda.synchronize()
            return f_, B_, float(E_[0])
        fa_, Ba_, Ea_ = fo(y, None, kt, 0.0, False)
        fb_, Bb_, Eb_ = fo(y, None, kt, 0.0, True)
        rs.update(f_new_old=G2._rel(fb_[:nnode * 3], fa_[:nnode * 3]), B_new_old=G2._rel(Bb_[:nnode * 9].double(), Ba_[:nnode * 9].double()),
                  E_new_old=abs(Eb_ - Ea_) / max(abs(Ea_), 1e-300))
        # 질량 몫 (빼기 켬 집합, 걸음 ŷ): f = G (k y + cm (y − ŷ)), E = ½ k yGy + ½ cm (y−ŷ)G(y−ŷ) 참값 (torch f64)
        yh = s.yh.reshape(-1).contiguous()
        fc_, Bc_, Ec_ = fo(y, yh, kt, cmt, True, fm1)
        d = (y - yh)[:nnode * 3]
        fr = ref_Gw(R1, y[:nnode * 3], kt, MM, nnode) + ref_Gw(R1, d, cmt, MM, nnode)
        Er = float(ref_Ew(R1, y[:nnode * 3], kt, MM)) + float(ref_Ew(R1, d, cmt, MM))
        rs.update(fm_rel=G2._rel(fc_[:nnode * 3], fr), Em_rel=abs(Ec_ - Er) / max(abs(Er), 1e-300), Em=Ec_, Ek_part=float(ref_Ew(R1, y[:nnode * 3], kt, MM)),
                  kt=kt, cmt=cmt, k=k0, kM=kM, it2=it2)
        # 대각: 단위 벡터 곱 (kk = k + cm) 대 B
        o = torch.zeros(capN * 3, device=dev)
        nodes = torch.unique(R1["ids"].reshape(-1)) if R1["nf"] > 0 else torch.zeros(0, device=dev, dtype=torch.long)
        if nodes.numel():
            dd = []
            for nd in nodes[torch.randint(0, nodes.numel(), (32,), device=dev, generator=gen)].tolist():
                e_ = torch.zeros(capN * 3, device=dev); e_[nd * 3] = 1.0; o.zero_()
                s.k("gq_mv", capQc, (P(s.IP), P(s.cb), P(fm1), FWP(s), f32(kt + cmt), P(e_), P(o)), 128)
                torch.cuda.synchronize()
                dd.append((float(o[nd * 3]), float(Bc_[nd * 9])))
            dd = np.array(dd)
            rs["diag_rel"] = float(np.abs(dd[:, 0] - dd[:, 1]).max() / max(np.abs(dd[:, 0]).max(), 1e-300))
        # 선탐색 후보 (합성 η·δ·sk·pm): SC[40+i] = ½k E(ŷ + T(η + cδ)) + ½cm E(T(η + cδ))
        nrm = torch.tensor([0.6, 0.0, 0.8], device=dev, dtype=torch.float64)
        eta = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
        dl = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
        sk = (torch.rand(capN, device=dev, generator=gen) * 0.7 + 0.3).float()
        pm = torch.randint(0, 3, (capN,), device=dev, generator=gen).int()
        GQ = torch.zeros(8, device=dev, dtype=torch.float64)
        errs = []
        for i0 in (0, 2):
            SCb = torch.zeros(64, device=dev, dtype=torch.float64)
            s.k("gq_ls", capQc, (P(s.IP), P(s.cb), P(fm1), FWP(s), P(yh), P(eta), P(dl), P(sk), P(pm), f32(0.6), f32(0.0), f32(0.8), P(GQ), P(None)), 128)
            s.k("gq_lsfin", 1, (P(SCb), np.int32(10), np.int32(i0), P(None), P(GQ), f32(kt), f32(cmt)), 1)
            torch.cuda.synchronize()
            got = SCb[40:50].cpu().numpy()
            for i in range(i0, 10):
                c = 0.5 ** i
                e3 = eta[:nnode * 3].double().reshape(-1, 3) + c * dl[:nnode * 3].double().reshape(-1, 3)
                te = G2._T(e3, sk[:nnode].double(), pm[:nnode], nrm)
                yc = yh[:nnode * 3].double().reshape(-1, 3) + te
                ex = float(ref_Ew(R1, yc.reshape(-1), kt, MM)) + float(ref_Ew(R1, te.reshape(-1), cmt, MM))
                errs.append(abs(got[i] - ex) / max(abs(ex), 1e-300))
        rs["ls_rel_max"] = float(max(errs)) if errs else None
        # Φ (옛 실행기): gq_phi = gq_force E
        MO = torch.ones(1, device=dev, dtype=torch.int32)
        PH = torch.zeros(2, device=dev, dtype=torch.float64)
        s.k("gq_phi", capQc, (P(s.IP), P(s.cb), P(fm1), FWP(s), f32(kt), f32(cmt), P(y), P(yh), P(MO), P(PH)), 128)
        torch.cuda.synchronize()
        rs["phi_rel"] = abs(float(PH[1]) - Ec_) / max(abs(Ec_), 1e-300)
        # C. 시간 (µs, CUDA 이벤트 reps 회 평균): 옛 면 목록 커널 대 칸별 (같은 면 집합) · 칸별 (빼기 켬)
        ob = torch.zeros(capN * 3, device=dev)
        fb = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
        Bb = torch.zeros(capN * 9, device=dev)
        rs["us"] = dict(
            mv_old=_ev(lambda: s.k("gp_mv", 3 * capQc, Gold + (f32(kt), P(v), P(ob)), 128), reps),
            mv_new=_ev(lambda: s.k("gq_mv", capQc, (P(s.IP), P(s.cb), P(fm0), P(None), f32(kt), P(v), P(ob)), 128), reps),
            mv_new_thin=_ev(lambda: s.k("gq_mv", capQc, (P(s.IP), P(s.cb), P(fm1), FWP(s), f32(kt), P(v), P(ob)), 128), reps),
            force_old=_ev(lambda: s.k("gp_force", 3 * capQc, Gold + (f32(kt), P(y), P(fb), P(Bb), P(None)), 128), reps),
            force_new=_ev(lambda: s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(fm0), P(None), f32(kt), f32(0.0), P(y), P(None), P(fb), P(Bb), P(None)), 128), reps),
            force_new_mass_thin=_ev(lambda: s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(fm1), FWP(s), f32(kt), f32(cmt), P(y), P(yh), P(fb), P(Bb), P(None)), 128), reps),
            mask_new=_ev(lambda: build_mask(s, _ok, THIN, fm0, None), reps // 4),
        )
        build_mask(s, _ok, THIN, fm1, None)                                                          # 실행 표지 되돌림 (같은 정의 -> 같은 값)
        apply_fix(s, _ok, fm1, None)
        torch.cuda.synchronize()
        rs["mask_restored"] = same(mask_keys(s.g3fm, nc), k1s)
        # H. 훅 자리 ('훅 켬 − 훅 끔' = 칸별 커널, k·k_M 1e3 배)
        if not getattr(s, "det", False) and getattr(s, "_g3args", None):
            rs["hook"] = _hook_check(s, R1, MM, nnode, gen, sc)
    finally:
        s._gpin = False
        s._g3chk = False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if out:
        with open(os.path.join(out, "g4fd.jsonl"), "a") as fo_:
            fo_.write(json.dumps(rs, ensure_ascii=False) + "\n")
    return rs


def _hook_check(s, R, MM, nnode, gen, sc):
    dev = s.dev
    capN, capQ, capQc = int(s.capN), int(s.capQc) * 8, int(s.capQc)
    f32 = np.float32
    k_old, kM_old = s._gpk, s._g3kM
    o = {}
    try:
        s._gpk = (k_old if k_old > 0 else 1.0) * 1e3
        s._g3kM = (kM_old if kM_old > 0 else 1.0) * 1e3
        kb = float(s._gpk)
        cb_ = cm_of(s)
        yh = s.yh.reshape(-1).contiguous()
        o["cm"] = cb_
        for name, args in list(s._g3args.items()):
            try:
                if name in ELEM:
                    res = []
                    for raw in (True, False):
                        a = list(args)
                        f_ = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
                        B_ = torch.zeros(capN * 9, device=dev)
                        E_ = torch.zeros(64, device=dev, dtype=torch.float64)
                        Ht = G2._tensor_by_ptr(s, a[11])
                        if Ht is None and int(a[11]) != 0:
                            raise RuntimeError("H 버퍼 못 찾음")
                        H_ = torch.empty_like(Ht) if Ht is not None else None
                        a[10], a[11], a[13] = P(f_), P(H_), P(B_)
                        if name == "rt_elem":
                            a[9] = P(E_)
                        if name in ("aq_elem", "ao_elem", "ao_elem_s") and len(a) > 14:
                            a[14] = P(None)
                        if name == "aq_elem" and len(a) > 18 and int(a[18]) != 0:
                            qt = G2._tensor_by_ptr(s, a[18])
                            a[18] = P(torch.zeros_like(qt)) if qt is not None else P(None)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        res.append((f_, B_, float(E_[0])))
                        del H_
                    mode = int(args[8]) if name == "rt_elem" else (11 if name == "aq_elem" else int(args[9]))
                    yv = G2._tensor_by_ptr(s, args[5])
                    yv = yv.reshape(-1) if yv is not None else s.y.reshape(-1)
                    d_ = (yv - yh)[:nnode * 3] if cb_ != 0.0 else torch.zeros(nnode * 3, device=dev)
                    fr = ref_Gw(R, yv[:nnode * 3], kb, MM, nnode) + (ref_Gw(R, d_, cb_, MM, nnode) if cb_ else 0)
                    d = dict(mode=mode)
                    if mode & 1:
                        d["f_rel"] = G2._rel(res[1][0][:nnode * 3] - res[0][0][:nnode * 3], fr)
                    if mode & 8:
                        Bg = torch.zeros(capN * 9, device=dev)
                        s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(s.g3fm), FWP(s), f32(kb), f32(cb_), P(yv), P(yh), P(None), P(Bg), P(None)), 128)
                        torch.cuda.synchronize()
                        d["B_rel"] = G2._rel((res[1][1] - res[0][1]).double(), Bg.double())
                    if name == "rt_elem" and mode & 4:
                        Er = float(ref_Ew(R, yv[:nnode * 3], kb, MM)) + (float(ref_Ew(R, d_, cb_, MM)) if cb_ else 0.0)
                        d["E_rel"] = abs((res[1][2] - res[0][2]) - Er) / max(abs(Er), 1e-300)
                    o[name] = d
                elif name in MV:
                    vv = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
                    vv[nnode * 3:] = 0
                    outs = []
                    for raw in (True, False):
                        a = list(args)
                        oo = torch.zeros(capN * 3, device=dev)
                        a[6], a[7] = P(vv), P(oo)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        outs.append(oo)
                    o[name] = dict(mv_rel=G2._rel((outs[1] - outs[0])[:nnode * 3].double(), ref_Gw(R, vv[:nnode * 3], kb + cb_, MM, nnode)))
                elif name == "rt_elem_lsN":
                    yh2 = s.y.reshape(-1).clone()
                    eta = (torch.randn(capN * 3, device=dev, generator=gen) * sc * 1e-2).float()
                    dl = (torch.randn(capN * 3, device=dev, generator=gen) * sc * 1e-2).float()
                    sk = (torch.rand(capN, device=dev, generator=gen) * 0.7 + 0.3).float()
                    pm = torch.randint(0, 3, (capN,), device=dev, generator=gen).int()
                    scs = []
                    for raw in (True, False):
                        a = list(args)
                        SCb = torch.zeros(64, device=dev, dtype=torch.float64)
                        a[5], a[6], a[7], a[8], a[9] = P(yh2), P(eta), P(dl), P(sk), P(pm)
                        a[10], a[11], a[12] = f32(0.6), f32(0.0), f32(0.8)
                        a[15], a[16], a[17], a[18] = P(SCb), np.int32(10), np.int32(0), P(None)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        scs.append(SCb[40:50].cpu().numpy())
                    nrm = torch.tensor([0.6, 0.0, 0.8], device=dev, dtype=torch.float64)
                    errs = []
                    for i in range(10):
                        c = 0.5 ** i
                        e3 = eta[:nnode * 3].double().reshape(-1, 3) + c * dl[:nnode * 3].double().reshape(-1, 3)
                        te = G2._T(e3, sk[:nnode].double(), pm[:nnode], nrm)
                        yc = yh2[:nnode * 3].double().reshape(-1, 3) + te
                        ex = float(ref_Ew(R, yc.reshape(-1), kb, MM)) + (float(ref_Ew(R, te.reshape(-1), cb_, MM)) if cb_ else 0.0)
                        errs.append(abs((scs[1][i] - scs[0][i]) - ex) / max(abs(ex), 1e-300))
                    o[name] = dict(ls_rel_max=float(max(errs)), ls_fin=bool(np.isfinite(scs[0]).all()))
                elif name == "en_phi_n":
                    phs = []
                    for raw in (True, False):
                        a = list(args)
                        MO = torch.ones(1, device=dev, dtype=torch.int32)
                        PH = torch.zeros(2, device=dev, dtype=torch.float64)
                        a[5], a[6] = P(MO), P(PH)
                        s._gpin = raw
                        s.k(name, int(s.capN), tuple(a), 256)
                        s._gpin = False
                        torch.cuda.synchronize()
                        phs.append(float(PH[1]))
                    it2 = float(args[2])
                    cmp = float(s._g3kM) * it2
                    yv = G2._tensor_by_ptr(s, args[3]).reshape(-1)
                    yhv = G2._tensor_by_ptr(s, args[4]).reshape(-1)
                    Er = float(ref_Ew(R, yv[:nnode * 3], kb, MM)) + float(ref_Ew(R, (yv - yhv)[:nnode * 3], cmp, MM))
                    o[name] = dict(phi_rel=abs((phs[1] - phs[0]) - Er) / max(abs(Er), 1e-300))
            except Exception as ex:
                o[name] = dict(err=repr(ex)[:300])
            finally:
                s._gpin = False
    finally:
        s._gpk, s._g3kM = k_old, kM_old
    return o


# ------------------------------------------------------------------------------------------------ 절 CM55 진단 (그래프 밖, 결과 불변)
def ediag(s, fi, out):
    """걸음 끝 y (걸음 풀이 변위) 방향 헤시안 몫: q_e = yᵀ K_e y (마지막 요소 헤시안 곱 커널 그대로, 훅 없이), q_M = Σ m_i |y_i|² / τ²,
    q_g = k yᵀ G_w y (면 가중 방식 0~4 각각, 실행 면 표지), q_gm = (k_M/τ²) yᵀ G_w y. 면 갈래 = 소유 칸 밀도 ρ_c 가 씬 평균 ρ̄ 보다 작음/큼 (진단 분류만)."""
    torch.cuda.synchronize()
    dev = s.dev
    capN, capQc = int(s.capN), int(s.capQc)
    nc, nnode = int(s.IP[11]), int(s.IP[9])
    f32 = np.float32
    rs = dict(f=int(fi))
    s._g3chk = True
    s._gpin = True
    try:
        y = s.y.reshape(-1).contiguous().float()
        names = [n for n in MV if getattr(s, "_g3args", None) and n in s._g3args]
        if names:
            a = list(s._g3args[names[-1]])
            o = torch.zeros(capN * 3, device=dev)
            a[6], a[7] = P(y), P(o)
            s.k(names[-1], capQc * 8, tuple(a))
            torch.cuda.synchronize()
            rs.update(mv=names[-1], q_e=float(torch.dot(y[:nnode * 3].double(), o[:nnode * 3].double())))
        it2 = float(getattr(s, "_g3it2", 0.0))
        y3 = y[:nnode * 3].double().reshape(-1, 3)
        rs.update(q_M=float((s.mqn[:nnode].double() * (y3 * y3).sum(1)).sum() * it2), it2=it2, k=float(s._gpk), kM=float(s._g3kM))
        lowc = (s.g5rho[:nc] < float(s._g3rho)).to(torch.uint8)
        fmr = s.g3fm[:capQc].clone()
        fl = torch.zeros_like(fmr); fh = torch.zeros_like(fmr)
        fl[:nc] = fmr[:nc] * lowc; fh[:nc] = fmr[:nc] * (1 - lowc)
        nf = lambda f_: int(sum(int(((f_[:nc].long() >> b) & 1).sum()) for b in range(3)))
        rs.update(nface=dict(all=nf(fmr), low=nf(fl), high=nf(fh)), ncell_low=int(lowc.sum()), rho_bar=float(s._g3rho))
        fwt = torch.zeros(capQc * 3, device=dev)
        qg = {}
        for md in (0, 1, 2, 3, 4):
            s.k("gq5_w", capQc, (P(s.IP), P(s.cb), P(s.cidx), P(fmr), P(s.g5phi), np.int32(md), np.float32(WMAX), P(fwt)), 128)
            d = {}
            for nm, fm_ in (("all", fmr), ("low", fl), ("high", fh)):
                E_ = torch.zeros(2, device=dev, dtype=torch.float64)
                s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(fm_), P(fwt), f32(float(s._gpk)), f32(0.0), P(y), P(None), P(None), P(None), P(E_)), 128)
                torch.cuda.synchronize()
                d[nm] = 2.0 * float(E_[0])
            sel = fmr[:nc].long()
            ws = torch.cat([fwt[:nc * 3].view(nc, 3)[:, b][((sel >> b) & 1).bool()] for b in range(3)])
            d["w_mean"] = float(ws.mean()) if ws.numel() else None
            qg[md] = d
        rs["q_g"] = qg
        php = s.g5phi[:nc]
        rs["phi_low_q"] = [float(q) for q in torch.quantile(php[lowc.bool()], torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()] if int(lowc.sum()) else None
        rs["phi_high_q"] = [float(q) for q in torch.quantile(php[~lowc.bool()], torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()] if int((1 - lowc).sum()) else None
    finally:
        s._gpin = False
        s._g3chk = False
    torch.cuda.synchronize()
    if out:
        with open(os.path.join(out, "g5ed.jsonl"), "a") as fo_:
            fo_.write(json.dumps(rs, ensure_ascii=False) + "\n")
    return rs
