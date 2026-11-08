"""절 CM61 묶음 hhtc2 (11-08 위임): 절 CM58 H4 가 레고 오차를 올리는 문제 -- 충돌 걸음 몸 운동량 진단 + H4 를 '몸 전체가 뒤집힌 걸음' 으로 좁힌 H5.

lib/anchorflow/fem_hhtc.py (절 CM58) 복사 + 아래. 모드 0~4 는 절 CM58 글자 그대로 (같은 커널·같은 순서 + 진단 커널 hc_mom 하나 더, 결과 불변).
진단 (늘, 결과 불변): 장벽 씬 걸음마다 (걸음 끝 갱신 바로 앞) 노드 질량 mn 가중 합을 한 걸음 한 줄 (hc.jsonl 'mo', 프레임 안 걸음 순서대로, 최대 32 걸음) 로:
  몸 전체 [0] Σm [1] P0 = Σ m v0·n [2] PH = Σ m vH·n [3] PB = Σ m vB·n [4] K0 = Σ ½ m |v0|² [5] KH [6] KB [7] Σ m ΔU_H·n [8] Σ m ΔU_B·n
  새 접촉 노드 [9] Σm [10] P0 [11] PH [12] PB · 걸음 끝 접촉 노드 [13] Σm [14] P0 [15] PH [16] PB · [17] Σ v0·n [18] Σ vH·n (절 CM58 H4 의 무게 없는 합) [19] 새 접촉 노드 수.
  v0 = 걸음 시작 속도, vH = α (y − z0) (HHT 걸음 끝), vB = (y − U)/h (같은 y 의 후퇴 오일러 걸음 끝 = 걸음 평균 속도), ΔU_H = h v0 + h² ((½−β) a0 + β a1), ΔU_B = y − U.
  노드 질량 가중 합 = 입자 질량 가중 합 (Σ_n m_n v_n = Σ_p m_p Σ_n w_pn v_n) 이라 P/Σm 이 입자 몸 질량중심 속도와 같음.
정의 (문턱·클램프 없음):
  5 (H5, s2 부터; s1 진단 뒤 확정): H4 의 걸음 (충돌 걸음 ∧ 새 접촉 노드 법선 속도 합을 HHT 가 들어가는 쪽 -> 나오는 쪽으로 뒤집음) 중
     **몸 질량중심 법선 운동량을 HHT 끝 속도가 뒤집는데 (P0 < 0 < PH) 같은 y 의 걸음 평균 (후퇴 오일러) 운동량은 아직 들어가는 쪽 (PB < 0)** 인 걸음만 몸 전체 후퇴 오일러.
     = 반발 계수로: HHT 몸 반발 e_H = −PH/P0 > 0 인데 풀이가 실제로 옮긴 몸 반발 e_B = −PB/P0 < 0 -- 정답 바닥 (MPM sticky collider, 들어오는 노드 속도 0 =
     완전 비탄성, 반발 0) 아래에서 위치로 실현되지 않은 반동 (v1 = 2 v̄ − v0 꼴 사다리꼴 식이 만든 몫). 위치로 실현된 반동 (PB ≥ 0, 레고가 실제로 구르며 튀는 걸음) 과
     국소 노드만 뒤집힌 걸음 (몸은 계속 들어감, H4 가 레고에서 켜지던 걸음 대부분) 은 그대로 HHT.
  6 (H5a, 비교용): H4 ∧ P0 < 0 < PH (몸 뒤집힘만, PB 조건 없음).
  모드 0·4 는 H4·H5·H5a 걸음 수만 셈 (nflip·nf5·nf6). 주의: 커밋 fc33f1c (s0k·s1) 는 이름이 반대 (모드·셈 5 = 몸 뒤집힘만, 6 = + PB < 0).

--- 아래는 절 CM58 fem_hhtc.py 원문 ---
절 CM58 묶음 hhtc (조정자 11-08 위임): 수박 첫 접촉 뒤 몸 전체 튀어오름 (절 CM57.4) 을 접촉 걸음의 HHT 걸음 끝 갱신에서 끊는 정의.

원인 (절 CM57.4, 확인됨): 첫 접촉 프레임의 K 1 걸음이 (원자 합 순서 잡음으로) 수렴 판정되면 쪼개지 않고 받는다. 그 풀이는 몸을 바닥에서 한 걸음 안에 멈춰 세운
y 이고, HHT (Newmark γ 0.676, β 0.34) 의 걸음 끝 속도 v1 = α (y − z0) 는 '자리는 멈췄는데 걸음 평균 속도 ≈ 0' 인 걸음에서 들어온 속도를 거의 그대로 뒤집는다
(사다리꼴 규칙의 성질: v1 ≈ −(γ/β − 1) v0; 실측 −7.35 -> +5.39 m/s). 그 걸음의 가속도 a1 = (v1 − v0 − dt (1−γ) a0)/(γ dt) 가 기억으로 넘어가
접촉이 끝난 다음 걸음에 (1−γ) dt a1 = +6 m/s 를 한 번 더 더한다 (에너지 5.0 -> 18.5 J). 보통 판은 첫 시도가 무너져 쪼개기 (K 2) 로 넘어가 이 일이 없다.

정의 (모두 위상·기하 판정, 문턱·클램프 없음; --hc M, 0 이면 진단만 = 결과 불변):
  접촉 노드 판정 (걸음 끝 갱신 바로 앞, 그래프 안): 장벽 표본점 q (활성 AF_BACT, 질량 > 0) 의 부호 거리 d 를 걸음 시작 위치 (U) 와 걸음 끝 위치 (U + thinv (y − U),
  풀이 장벽 rt_bar_q 와 같은 식·같은 기저 -- W̄ (--e3w) 판이면 W̄, R2·A2 의 옮긴 기준 AF_BX 그대로) 에서 재고, d < d̂ (장벽 받침 안 = 장벽 힘이 걸리는 자리)
  인 표본점의 기저가 0 아닌 노드를 '걸음 시작 접촉'·'걸음 끝 접촉' 노드로 표시. 새 접촉 노드 = 걸음 끝 접촉 ∧ ¬ 걸음 시작 접촉 (이 걸음에 접촉 활성 집합에 든 노드).
  충돌 걸음 = 새 접촉 노드가 하나라도 있는 걸음.
  1 (H1, 충돌 걸음 몸 전체 후퇴 오일러): 충돌 걸음은 모든 노드의 걸음 끝 갱신을 후퇴 오일러로 (U1 = y, v1 = (y − U)/h, 가속도 기억 0 -- 절 CM24 --ue be 와 같은 식).
     충돌 아닌 걸음은 그대로 (HHT, 덜 풀린 걸음은 --ue_loc 집합 S 후퇴 오일러 그대로).
  2 (H2, 새 접촉 노드만 후퇴 오일러): 새 접촉 노드만 위 후퇴 오일러, 나머지 그대로.
  3 (H3, 걸음 끝 접촉 노드 가속도 기억 0): 걸음 끝 접촉 노드 (장벽 힘이 걸린 노드) 는 위치·속도 HHT 그대로, 가속도 기억만 0 (절 CM24 --ue a0 와 같은 식).
  4 (H4, H1 을 '적분이 만든 되튐' 걸음으로 좁힘): 충돌 걸음 중 HHT 걸음 끝 속도 v1 = α (y − z0) 가 새 접촉 노드들의 법선 속도 합을 걸음 시작의 들어가는 쪽
     (Σ v0·n < 0) 에서 나오는 쪽 (Σ v1·n > 0) 으로 뒤집는 걸음만 몸 전체 후퇴 오일러 (H1 은 쉬거나 미끄러지며 받침에 새로 드는 노드가 있는 걸음마다 켜져
     레고·wolf 를 몸 전체 감쇠로 깸 -- s2). 모드 0 은 뒤집힘 걸음 수만 셈 (nflip).
  덜 풀린 걸음 (판정 커널 uf/cj = 1) 의 집합 S 노드는 원래 갱신 (--ue be) 이 먼저 (같은 후퇴 오일러).
근거: Newmark 계열 적분은 접촉 힘처럼 한 걸음 안에서 켜지고 꺼지는 힘을 걸음 양 끝 평균 (사다리꼴) 으로 넣어 충돌에서 에너지를 만들거나 되튐을 과장한다 --
  Deuflhard, Krause, Ertel 2008 (IJNME 73:1274, 'A contact-stabilized Newmark method for dynamical contact problems') 은 접촉 힘을 걸음 끝 시점에만 (암시적, 평균 없이)
  넣어 에너지가 늘지 않게 하고, Kane, Repetto, Ortiz, Marsden 1999 (CMAME 180:1) 도 충돌 걸음을 따로 다룸. H1·H2 는 '충돌 걸음 (노드) 은 1 차 암시 (후퇴 오일러)
  갱신 = 접촉 힘을 걸음 끝에만, 기억 없이', H3 은 '접촉 노드의 접촉 몫이 든 가속도를 다음 걸음 평균에 넣지 않음'.

훅 (끄면 = --hc 없으면 이 모듈을 안 부름): 실행기 runpy 순간 (다른 감싸개 설치 뒤, 가장 바깥) 에 ART.build_src·AORT._src 를 감싸 커널을 덧붙이고
  (hc_* + --ue_loc 갱신 커널 rt_ga_end_ufl·ao_stepend_ufl 의 복사본 rt_ga_end_hc·ao_stepend_hc = 노드 표시 hcm 이 서면 갱신 방식 hk), s.k 를 감싸
  걸음 끝 갱신 커널을 띄울 때 표시 커널을 먼저 띄우고 복사본으로 바꿔 띄움. 장벽 없는 씬 (ficus·bread) 은 rt_bar_q 를 안 불러 아무것도 안 바뀜 (원래 커널 그대로).
진단 (늘, 결과 불변): <out>/hc.jsonl -- step() 시도마다 걸음 수·충돌 걸음 수·새 접촉 노드 수·걸음 끝 접촉 노드 수·바꾼 노드 수 (합), K, t.
"""
import json
import os

import numpy as np
import torch

MODE = 0
ST = dict(on=False, said=False, out=".", log=None, nsrc=0, nhc=0, nbar=0)
NV, MR = 20, 32                                                                   # hc_mom 줄 길이·걸음 줄 수 (커널 HC_NV·HC_MR)

HC_SRC = r'''
// ---- 절 CM58 (lib/anchorflow/fem_hhtc.py): 접촉 노드 판정 (걸음 시작·끝 장벽 받침 안) + 걸음 끝 갱신 바꿈 표시 ----
#ifdef E3W
#define HC_W(q, oc, o) E3W(q, o)
#else
#define HC_W(q, oc, o) BW[(oc)*27+(o)]
#endif
__global__ void hc_zero(const int* IP, int* ms, int* me, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) st[2] = 0;                                                     // 이 걸음 새 접촉 노드 수
    if (n < IP[9]) { ms[n] = 0; me[n] = 0; }
}
__global__ void __launch_bounds__(128) hc_bar(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv,
                                              float n0, float n1, float n2, float pn, float dh, int* ms, int* me) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xs[3], xe[3];
    for (int a=0;a<3;a++) { xs[a] = AF_BX[q*4+a]/bm; xe[a] = xs[a]; }
    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];
        for (int a=0;a<3;a++) { float u = U[id*3+a]; xs[a] += w*u; xe[a] += w*(u + thinv*(y[id*3+a] - u)); } }
    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;
    bool s_ = ds < dh, e_ = de < dh;                                          // 장벽 받침 안 (rt_bar_q 가 힘을 거는 조건 d < d̂, 관통 d ≤ 0 포함)
    if (!s_ && !e_) return;
    for (int o=0;o<27;o++) if (HC_W(q, oc, o) != 0.f) { if (s_) ms[ids[o]] = 1; if (e_) me[ids[o]] = 1; }
}
// st: [0] 새 접촉 노드 합 [1] 걸음 끝 접촉 노드 합 [2] 이 걸음 새 접촉 노드 [3] 충돌 걸음 수 [4] 걸음 수 [5] 바꾼 노드 합
__global__ void hc_fin(const int* IP, const int* ms, const int* me, int mode, int* hm, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int e = me[n], nw = e && !ms[n];
    if (e) atomicAdd(&st[1], 1);
    if (nw) { atomicAdd(&st[0], 1); atomicAdd(&st[2], 1); }
    int h = (mode == 3) ? e : ((mode == 2) ? nw : 0);
    hm[n] = h;
    if (h) atomicAdd(&st[5], 1);
}
__global__ void hc_all(const int* IP, int* st, int* hm) {                     // H1: 충돌 걸음이면 몸 전체
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || st[2] == 0) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
// H4: 충돌 걸음 중 HHT 걸음 끝 속도 v1 = α (y − z0) 가 새 접촉 노드들의 법선 속도 합을 들어가는 쪽 (Σ v0·n < 0) 에서 나오는 쪽 (Σ v1·n > 0) 으로 뒤집는 걸음만 몸 전체
__global__ void hc_fz(double* fa) { fa[0] = 0.0; fa[1] = 0.0; }
__global__ void hc_flip(const int* IP, const int* ms, const int* me, const float* y, const float* z0, const float* vk, float al, float n0, float n1, float n2, double* fa) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !(me[n] && !ms[n])) return;
    float v0 = vk[n*3]*n0 + vk[n*3+1]*n1 + vk[n*3+2]*n2;
    float v1 = al*((y[n*3] - z0[n*3])*n0 + (y[n*3+1] - z0[n*3+1])*n1 + (y[n*3+2] - z0[n*3+2])*n2);
    atomicAdd(&fa[0], (double)v0); atomicAdd(&fa[1], (double)v1);
}
__global__ void hc_all4(const int* IP, int* st, const double* fa, int* hm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0) atomicAdd(&st[6], 1);   // 뒤집힌 충돌 걸음 수
    if (n >= IP[9] || !(st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0)) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
__global__ void hc_cnt(int* st) { st[4] += 1; if (st[2] > 0) st[3] += 1; }
// ---- 절 CM61 (lib/anchorflow/fem_hhtc2.py): 걸음마다 몸 운동량·운동 에너지 (노드 질량 가중, 몸 전체·새 접촉·끝 접촉), 블록 합 뒤 원자 합 ----
#define HC_NV 20
#define HC_MR 32
__global__ void hc_mz(const int* st, double* mo) {
    int i = threadIdx.x; int r = min(st[4], HC_MR - 1);
    if (i < HC_NV) mo[r*HC_NV + i] = 0.0;
}
__global__ void __launch_bounds__(256) hc_mom(const int* IP, const int* ms, const int* me, const float* mn, const float* y, const float* z0, const float* U,
                                              const float* vk, const float* ak, float dt, float gm, float be, float al, float n0, float n1, float n2,
                                              const int* st, double* mo) {
    __shared__ double sh[8][HC_NV];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v[HC_NV];
    for (int i=0;i<HC_NV;i++) v[i] = 0.0;
    if (n < IP[9]) {
        float m = mn[n];
        if (m > 0.f) {
            float nn[3] = {n0, n1, n2}; float p0 = 0.f, pH = 0.f, pB = 0.f, k0 = 0.f, kH = 0.f, kB = 0.f, dH = 0.f, dB = 0.f;
            for (int a=0;a<3;a++) {
                float u = U[n*3+a], vv = vk[n*3+a], ac = ak[n*3+a];
                float v1 = al*(y[n*3+a] - z0[n*3+a]);
                float a1 = (v1 - vv - dt*(1.f - gm)*ac)/(gm*dt);
                float du = dt*vv + dt*dt*((0.5f - be)*ac + be*a1);
                float vb = (y[n*3+a] - u)/dt;
                p0 += vv*nn[a]; pH += v1*nn[a]; pB += vb*nn[a]; k0 += vv*vv; kH += v1*v1; kB += vb*vb; dH += du*nn[a]; dB += (y[n*3+a] - u)*nn[a];
            }
            double md = (double)m;
            v[0] = md; v[1] = md*p0; v[2] = md*pH; v[3] = md*pB; v[4] = 0.5*md*k0; v[5] = 0.5*md*kH; v[6] = 0.5*md*kB; v[7] = md*dH; v[8] = md*dB;
            int e = me[n], nw = e && !ms[n];
            if (nw) { v[9] = md; v[10] = md*p0; v[11] = md*pH; v[12] = md*pB; v[17] = p0; v[18] = pH; v[19] = 1.0; }
            if (e) { v[13] = md; v[14] = md*p0; v[15] = md*pH; v[16] = md*pB; }
        } else if (me[n] && !ms[n]) {                                                // 질량 0 새 접촉 노드도 절 CM58 H4 합에는 듦 (같은 셈)
            float p0 = vk[n*3]*n0 + vk[n*3+1]*n1 + vk[n*3+2]*n2;
            float pH = al*((y[n*3] - z0[n*3])*n0 + (y[n*3+1] - z0[n*3+1])*n1 + (y[n*3+2] - z0[n*3+2])*n2);
            v[17] = p0; v[18] = pH; v[19] = 1.0;
        }
    }
    for (int i=0;i<HC_NV;i++) for (int o=16;o>0;o>>=1) v[i] += __shfl_down_sync(0xffffffff, v[i], o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) for (int i=0;i<HC_NV;i++) sh[w][i] = v[i];
    __syncthreads();
    if (threadIdx.x < HC_NV) {
        double t = 0.0; int nw_ = (blockDim.x + 31) >> 5;
        for (int k=0;k<nw_;k++) t += sh[k][threadIdx.x];
        int r = min(st[4], HC_MR - 1);
        atomicAdd(&mo[r*HC_NV + threadIdx.x], t);
    }
}
// H5 (모드 5): H4 조건 ∧ 몸 질량중심 법선 운동량을 HHT 끝 속도가 뒤집음 (P0 < 0 < PH) ∧ 걸음 평균 운동량 PB < 0; H5a (모드 6): H4 ∧ P0 < 0 < PH. 셈: st[7] H5 · st[8] H5a
__global__ void hc_all5(const int* IP, int* st, const double* fa, const double* mo, int mode, int* hm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int r = min(st[4], HC_MR - 1);
    const double* q = mo + r*HC_NV;
    bool c4 = st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0;
    bool c6 = c4 && q[1] < 0.0 && q[2] > 0.0;                                   // H5a 몸 뒤집힘
    bool c5 = c6 && q[3] < 0.0;                                                  // H5 = H5a ∧ 걸음 평균 들어가는 쪽
    if (n == 0) { if (c4 && mode >= 5) atomicAdd(&st[6], 1); if (c5) atomicAdd(&st[7], 1); if (c6) atomicAdd(&st[8], 1); }   // 모드 0·4 의 st[6] 은 hc_all4 가 셈
    bool on = (mode == 5) ? c5 : ((mode == 6) ? c6 : false);
    if (n >= IP[9] || !on) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
'''

SIG_OLD = "int um, const int* ulm) {"
SIG_NEW = "int um0, const int* ulm, const int* hcm, int hk) {\n    int um = um0;                                                              // 절 CM58: 노드마다 갱신 방식"
UL_LINE = "    uf_ = uf_ && ulm[n];"
HC_LINE = "    if (hcm[n] && !uf_) { uf_ = 1; um = hk; }                                    // 절 CM58: 표시 노드 = 후퇴 오일러 (hk 1) 또는 가속도 기억 0 (hk 2)\n"
PAIRS = (("rt_ga_end_ufl", "rt_ga_end_hc"), ("ao_stepend_ufl", "ao_stepend_hc"))


def hcify(src):
    """--ue_loc 갱신 커널 복사본 + 표시 커널을 extern "C" 끝에 (장벽 판 소스에만 표시 커널)."""
    add = []
    for nm, nh in PAIRS:
        key = "__global__ void " + nm + "("
        if key not in src:
            continue
        i = src.index(key)
        j = src.index("\n}\n", i) + 3
        k = src[i:j].replace(nm + "(", nh + "(", 1)
        assert k.count(SIG_OLD) == 1 and k.count(UL_LINE) == 1, ("절 CM58 hcify", nm)
        k = k.replace(SIG_OLD, SIG_NEW, 1)
        e = k.index("\n", k.index(UL_LINE)) + 1
        k = k[:e] + HC_LINE + k[e:]
        add.append(k)
        ST["nhc"] += 1
    if not add:
        return src
    if "__device__ float* AF_BX;" in src:
        add.append(HC_SRC)
        ST["nbar"] += 1
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n".join(add) + "\n}\n"


P = lambda t: np.uint64(0 if t is None else t.data_ptr())


def _buf(s, n):
    if getattr(s, "_hcst", None) is None:
        assert not torch.cuda.is_current_stream_capturing(), "절 CM58 버퍼는 데움 record 에서"
        s._hcst = torch.zeros(12, device=s.dev, dtype=torch.int32)
        s._hcfa = torch.zeros(2, device=s.dev, dtype=torch.float64)
        s._hcmo = torch.zeros(MR * NV, device=s.dev, dtype=torch.float64)            # 절 CM61 걸음마다 몸 운동량 줄
    if getattr(s, "_hcm", None) is None or s._hcm.numel() < n:
        assert not torch.cuda.is_current_stream_capturing(), "절 CM58 노드 버퍼는 데움 record 에서"
        if getattr(s, "_hcm", None) is not None:                                         # 앞 그래프가 옛 포인터를 들고 있을 수 있어 버리지 않음
            s._hckeep = getattr(s, "_hckeep", []) + [s._hcm, s._hcms, s._hcme]
        m = int(n * 1.25) + 1024
        s._hcm = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcms = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcme = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcdum = torch.zeros(m, device=s.dev, dtype=torch.int32)                     # 모드 0 의 H4 셈용 버림 표시


def _wrap_k(cls, nm, nh):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name == "rt_bar_q" and len(args) >= 12:
            s._hcbq = (int(n), tuple(args))                                              # 마지막 장벽 호출의 표본점 수·기저·thinv·법선·평면·d̂
        if name == nm and getattr(s, "_hcbq", None) is not None:
            capN = int(n)
            _buf(s, capN)
            nq, ba = s._hcbq
            IP = args[0]
            y, U = args[5], args[7]                                                      # 두 실행기 모두 (IP, dt, gm, be, al, y, z0, U, ...)
            i32 = np.int32
            ok(s, "hc_zero", capN, (IP, P(s._hcms), P(s._hcme), P(s._hcst)), 256)
            ok(s, "hc_bar", nq, (IP, ba[1], ba[2], y, U, ba[6], ba[7], ba[8], ba[9], ba[10], ba[11], P(s._hcms), P(s._hcme)), 128)
            ok(s, "hc_fin", capN, (IP, P(s._hcms), P(s._hcme), i32(MODE), P(s._hcm), P(s._hcst)), 256)
            if MODE == 1:
                ok(s, "hc_all", capN, (IP, P(s._hcst), P(s._hcm)), 256)
            if MODE in (0, 4, 5, 6):                                                     # H4 (모드 0 은 셈만): HHT 가 새 접촉 노드 법선 속도를 뒤집는지
                ok(s, "hc_fz", 1, (P(s._hcfa),), 1)
                ok(s, "hc_flip", capN, (IP, P(s._hcms), P(s._hcme), y, args[6], args[8], args[4], ba[7], ba[8], ba[9], P(s._hcfa)), 256)
            ok(s, "hc_mz", 1, (P(s._hcst), P(s._hcmo)), 32)                              # 절 CM61 진단: 이 걸음 줄 비우고 몸 운동량 합 (결과 불변)
            ok(s, "hc_mom", capN, (IP, P(s._hcms), P(s._hcme), P(s.mn), y, args[6], U, args[8], args[9], args[1], args[2], args[3], args[4], ba[7], ba[8], ba[9],
                                   P(s._hcst), P(s._hcmo)), 256)
            if MODE in (0, 4):
                ok(s, "hc_all4", capN, (IP, P(s._hcst), P(s._hcfa), P(s._hcm) if MODE == 4 else P(s._hcdum)), 256)
            if MODE in (0, 4, 5, 6):                                                     # 절 CM61 H5·H6 (모드 0·4 는 셈만, hc_all4 가 센 st[6] 은 hc_all5 에서 다시 세지 않게 따로)
                ok(s, "hc_all5", capN, (IP, P(s._hcst), P(s._hcfa), P(s._hcmo), i32(MODE), P(s._hcm) if MODE in (5, 6) else P(s._hcdum)), 256)
            ok(s, "hc_cnt", 1, (P(s._hcst),), 1)
            s._hcn = getattr(s, "_hcn", 0) + 1
            if MODE in (1, 2, 3, 4, 5, 6):
                return ok(s, nh, n, tuple(args) + (P(s._hcm), i32(2 if MODE == 3 else 1)), b)
        return ok(s, name, n, args, b)
    cls.k = k


def _wrap_step(cls):
    o_s = cls.step

    def step(s, t, *a, **kw):
        d = getattr(s, "_hcdep", 0)
        s._hcdep = d + 1
        try:
            r = o_s(s, t, *a, **kw)
        finally:
            s._hcdep = d
        if d == 0 and getattr(s, "_hcst", None) is not None and ST["log"] is not None:
            try:
                c = [int(q) for q in s._hcst.cpu().tolist()]
                mo = s._hcmo.view(MR, NV)[:min(c[4], MR)].cpu().tolist()
                s._hcst.zero_()
                ST["log"].write(json.dumps(dict(t=float(t), h=float(getattr(s, "h", 0.0) or 0.0), K=int(getattr(s, "Kcur", getattr(s, "K", 1)) or 1),
                                                new=c[0], eng=c[1], ncol=c[3], nstep=c[4], napp=c[5], nflip=c[6], nf5=c[7], nf6=c[8], mo=mo, nlaunch=int(getattr(s, "_hcn", 0)),
                                                ms=float(getattr(s, "last_ms", float("nan")) or float("nan")))) + "\n")
                ST["log"].flush()
            except Exception as e_:
                with open(os.path.join(ST["out"], "hc_err.txt"), "a") as f_:
                    f_.write(f"step 기록: {e_!r}\n")
        return r
    cls.step = step


def install(mode, out):
    """실행기 runpy 순간 (다른 감싸개 뒤) 에 부름. mode 0 = 진단만 (결과 불변), 1 H1 · 2 H2 · 3 H3."""
    global MODE
    MODE = int(mode)
    assert MODE in (0, 1, 2, 3, 4, 5, 6), MODE
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    ST.update(on=True, out=out)
    ST["log"] = open(os.path.join(out, "hc.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: hcify(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: hcify(o_s(*a, **kw))
    _wrap_k(ART.RTSolver, "rt_ga_end_ufl", "rt_ga_end_hc")
    _wrap_k(AORT.AORTSolver, "ao_stepend_ufl", "ao_stepend_hc")
    for cls in (ART.RTSolver, V0.RTSolver):
        _wrap_step(cls)
    name = {0: "진단만 (결과 불변)", 1: "H1 충돌 걸음 몸 전체 후퇴 오일러", 2: "H2 새 접촉 노드 후퇴 오일러", 3: "H3 걸음 끝 접촉 노드 가속도 기억 0", 4: "H4 HHT 가 새 접촉 노드 법선 속도를 뒤집는 충돌 걸음만 몸 전체 후퇴 오일러", 5: "H5 H4 걸음 중 HHT 끝 속도가 몸 질량중심 법선 운동량을 뒤집고 걸음 평균 운동량은 아직 들어가는 쪽인 걸음만 몸 전체 후퇴 오일러", 6: "H5a H4 걸음 중 몸 질량중심 법선 운동량 뒤집힘 걸음만"}[MODE]
    print(f"[절 CM61] hhtc2 켬 -- 모드 {MODE}: {name}", flush=True)

    import atexit

    @atexit.register
    def _info():
        try:
            json.dump(dict(mode=MODE, nsrc=ST["nsrc"], nhc=ST["nhc"], nbar=ST["nbar"]), open(os.path.join(out, "hcinfo.json"), "w"))
        except Exception:
            pass
