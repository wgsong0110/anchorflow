"""절 CM70 묶음 wolfh5: wolf 높게 쌓임 5 -- 방출 직후 낙하가 무엇에 매달리는지 (노드 구속 받침 · 재질 응력 · ghost penalty · 덜 풀림) 가르기 + 원인에 맞춘 정의.
감싸개 exe/wolfh5_wrap.py 가 실행기 (fem_an_rt_sim.py) 를 띄우는 순간 install() (다른 감싸개가 모두 설치된 뒤 = 가장 바깥). 인자 없으면 이 모듈을 안 부름.

dg (--w5dg 1) 진단 (결과 불변 -- 따로 버퍼에만 쓰고 커널 시간만 보탬). 걸음마다 (쪼갠 걸음 포함) 걸음 끝 (커밋 자리) 에 한 행:
    방출 앞면 거리 d (cm; 방출된 쪽 +) 칸 (-∞,-6) [-6,-3) [-3,0) [0,3) [3,6) [6,11) [11,21) [21,40) [40,∞) x 바닥 위 높이 칸 (<10 cm, ≥10 cm) 별 합.
    입자 (w5_pdg; d = (t − t_방출) · v_앞면, t = 프레임 시작 + 걸음 가운데 시각, 높이·무게 = 프레임 시작 위치):
        M, Σm k̄ (k̄ = Σ_n w k_구속,n), Σm ᾱ (ᾱ = Σ_n w α_n, α_n = 그 걸음 표본 평균 구속 질량 몫), Σm (1 − k̄) (구속 노드에 걸린 무게),
        Σm v0·n̂, Σm v1·n̂ (노드 속도 걸음 시작·끝을 입자로 보간; n̂ = 바닥 법선 = 위), Σm τ (τ = 방출 뒤 시간),
        F_net = Σ_n w m_p ((v1 − v0)·n̂/Δt − g·n̂) (중력 말고 걸린 힘 = 노드 운동에서 거꾸로; 구속 노드는 구속 반력),
        F_el = Σ_n (w m_p/m_n) (−∇Ψ_el,n·n̂) (재질 응력 힘, 마지막 뉴턴 반복 ao_elem_s 몫만 -- W5_FE),
        F_tot = 같은 꼴 (−f·n̂), f = 요소 커널 사슬 뒤 기울기 (재질 + ghost penalty 강성·고스트 질량; 장벽은 그 뒤라 빠짐),
        F_net·F_el 의 풀린 노드 몫 (k_구속 가중), Σm v_p·n̂ (입자 프레임 시작 속도).
    가우스 점 (w5_qdg; d = |x_축 − 끝 위치| − (반폭 − v_앞면 t), 위치 = 표본점 질량 중심 + 걸음 변위): 걸음 끝 시험 X 의 PG Drucker-Prager 투영 --
        V, 원뿔·꼭짓점 V, V p (키르히호프 압력, 압축 +), V |dev τ|, V Δε_p, V k_q (Σ BW k_구속), V α_q, V tr ε (Hencky 체적).
s (--w5s 1) 정의 S_구속 (정의 변경, 순차 방출 구속 노드 몫의 받침만; 시간 법칙 e^−x (또는 --pbc_phi 면 φ) 는 그대로):
    지금 k_구속,n = exp(expo Σ_s log(1 − α_n(t_s))) 는 우리 노드 (간격 2.83 cm, 받침 반경 4.2 cm) 의 구속 질량 몫 α_n 에서 -- 구속 입자가 노드 받침에 1 % 만 있어도 k ≈ 0
    (expo = Δt/(S dt_gt) = 500) 이라, 방출 앞면 뒤 받침 폭만큼의 풀린 질량이 걸음 내내 묶임. 정답은 dx 1 cm 격자 노드 g 에서 같은 법칙 (서브스텝마다 v *= 1 − α_g).
    S_구속: (1) 정답 격자 (dx = grid_lim/n_grid, 노드 i dx) 노드 g 에서 같은 표본·같은 식으로 k_g = exp(expo Σ_s log(1 − α_g(t_s))) (α_g = 정답 이차 B-스플라인 무게로 모은 구속 질량 / 전체 질량),
            (2) 입자 몫 κ_p = Σ_g w_pg k_g (정답 G2P = 정답 입자가 그 걸음에 잃지 않는 속도 몫),
            (3) 우리 노드 k_구속,n = Σ_p w_pn m_p κ_p / Σ_p w_pn m_p (우리 P2G 질량 가중 -- 노드가 대표하는 입자들의 정답 몫 평균).
        = 절 CM63 S (바닥 고착 몫을 정답 격자에서 우리 표본점으로) 와 같은 꼴: '정답 받침의 구속 몫' 을 우리 노드로 옮김. 구속 입자 질량 몫이 아니라 정답 노드가 묶는 몫이라,
        앞면 뒤 정답 받침 (1.5 cm) 밖 입자는 κ = 1. 구속 노드 걸음 끝 가속도 처치 (mt_akb: k < 1 이면 a = 0) 는 연속판 a ← k a (k ∈ {0, 1} 이면 같음).
        문턱·클램프 없음 (κ = Σ w k_g ≥ 0, 받침 표지 Σ w (1 − k_g) 가 0 인 노드만 정확히 1). 정답 격자 버퍼는 첫 프레임 입자 상자 ± 3 칸 (구속 입자는 거의 안 움직임; 상자 밖 입자는 몫 없음 = κ 1).
    순차 방출 구속이 없는 씬 (구속 입자 없음) 과 구속 속도 (pvb, bread 의 enforce_particle_translation) 씬은 아무것도 안 함.
c (--w5c 1) 정의 C (정의 변경, 구속 노드의 저질량 고정 자리): 지금 mt_floor_n 은 질량 < m̄ (mlow) 인 노드를 구속 몫과 상관없이 관성 예측 Ũ (자유 낙하) 에 고정 --
    구속 받침 안 (k_구속 < 1) 가벼운 노드도 한 프레임 ½ g Δt² (7.8 mm) 씩 떨어져, 움직이지 않는 구속 입자의 F 에 그 노드 ∇w 만큼 수직 압축·전단이 매 프레임 쌓임
    (진단 s1: 구속 입자 변위 0 인데 F_zz −5e-5/프레임, 30 프레임 뒤 구속 몸 압력 150 kPa -- 정답 0.0~0.3 kPa; 방출된 입자가 이 압축을 들고 나가 마찰 (DP) 로 매달림).
    C: 가벼운 고정 노드도 같은 구속 법칙의 예측 ŷ_구속 = Ũ + (1 − k_구속)(z0 + v_목표 − Ũ) 에 고정 (풀린 노드의 h 와 같은 식; k_구속 = 1 이면 Ũ 그대로, 0 이면 걸음 시작 자리).
    문턱·클램프 없음 (k_구속 은 그 씬의 구속 정의 그대로 -- 기본 e^−x 또는 S_구속).
g (--w5g 1) 정의 B_g (정의 변경, ghost penalty 면): P1X 는 고정 (cuboid) 구속 노드에 닿는 면을 뺌 (@g4x) -- 순차 방출 구속 노드 (k_구속 < 1) 는 고정 노드와 같은
    구속인데 면이 남아, 걸음 변위 법선 미분 점프 벌점 (강성 + 고스트 질량) 이 구속 받침 옆 풀린 칸의 상대 운동을 막음. B_g: 걸음마다 (mt_floor_n 뒤, 프레임 면 표지 위 멱등)
    면 36 노드 (칸 27 + 그 축 바깥 층 9) 중 k_구속 < 1 노드가 있는 면을 뺌 (절 CM63 gs·fem_ghost4 gq_fix 와 같은 배치). 문턱 없음 (k_구속 < 1 = 구속 노드 정의 그대로, mt_akb 와 같은 판정).
"""
import atexit
import json
import os

import numpy as np

ST = dict(dg=0, s=0, c=0, g=0, out=".", marks=[], sol=None, said=False, info={})
NB, NH, PF, QF = 9, 2, 14, 10
CAPR = 4096
EDGES = [-6, -3, 0, 3, 6, 11, 21, 40]
PFIELDS = ["n", "M", "Mk", "Ma", "Mpin", "Mv0", "Mv1", "Mtau", "Fnet", "Fel", "Ftot", "Fnet_free", "Fel_free", "Mvp"]
QFIELDS = ["n", "V", "V_cone", "V_apex", "V_p", "V_sdev", "V_dep", "V_k", "V_a", "V_tr"]

W5_SRC = r'''
// ---- 절 CM70 wolfh5 (lib/anchorflow/fem_wolfh5.py): 방출 뒤 낙하 매달림 진단 + 정답 받침 구속 몫 S_구속 ----
__device__ __forceinline__ int w5_bin(float dcm) {
    return (dcm < -6.f) ? 0 : ((dcm < -3.f) ? 1 : ((dcm < 0.f) ? 2 : ((dcm < 3.f) ? 3 : ((dcm < 6.f) ? 4 : ((dcm < 11.f) ? 5 : ((dcm < 21.f) ? 6 : ((dcm < 40.f) ? 7 : 8)))))));
}
__global__ void w5_zero_d(int n, double* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0.0; }
__global__ void w5_zero_f(int n, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0.f; }
__global__ void w5_copy_d(const int* IP, const double* a, double* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < IP[9]*3) b[i] = a[i]; }
__global__ void w5_copy_f(const int* IP, const float* a, float* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < IP[9]*3) b[i] = a[i]; }
// 노드 α (그 걸음 S 표본 평균 구속 질량 몫; ao_pbc_k 와 같은 표본 고리)
__global__ void w5_alpha(const int* IP, const float* mn, const float* acc, int K, int S, int k, float* al) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], a = 0.f; int KS = K*S;
    if (m > 0.f) { float suf = 0.f; for (int j=KS-1; j>=k*S; j--) { suf += acc[(size_t)n*KS + j]; if (j < (k+1)*S) a += fminf(suf/m, 1.f); } a /= (float)S; }
    al[n] = a;
}
// 입자 진단 (걸음 끝): 칸 = 앞면 거리 x 높이
__global__ void w5_pdg(int N, const int* IP, const int* pb, const float* fx, const float* x, const float* m, const float* v, const float* Trel, const float* tbuf, float toff, float vf,
                       const float* kb, const float* al, const float* mn, const float* v0, const float* vk, const double* fe, const double* ft,
                       float dt, float g0, float g1, float g2, float n0, float n1, float n2, float pn, double* D, const int* ctr, int cap) {
    __shared__ double sh[9*2*14];
    for (int i=threadIdx.x;i<9*2*14;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p < N && !IP[13]) {
        float t = tbuf[0] + toff, tr = Trel[p];
        float dcm = ((tr > t) ? -(tr - t) : (t - tr))*vf*100.f;
        float tau = (tr > t) ? 0.f : (t - fmaxf(tr, 0.f));
        float hg = x[p*3]*n0 + x[p*3+1]*n1 + x[p*3+2]*n2 - pn;
        int b = w5_bin(dcm)*2 + ((hg >= 0.1f) ? 1 : 0);
        float w[3][3], dw[3][3];
        for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
        int nb1 = IP[7], nb2 = IP[8];
        float mp = m[p], gn = g0*n0 + g1*n1 + g2*n2;
        float skb = 0.f, sal = 0.f, spin = 0.f, sv0 = 0.f, sv1 = 0.f;
        double fnet = 0.0, fel = 0.0, ftot = 0.0, fnf = 0.0, fef = 0.0;
        for (int o=0;o<27;o++) {
            int i = o/9, j = (o/3)%3, k = o%3;
            int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
            float ww = w[0][i]*w[1][j]*w[2][k];
            float kk = kb ? kb[nid] : 1.f, mm = mn[nid];
            float a0 = v0[nid*3]*n0 + v0[nid*3+1]*n1 + v0[nid*3+2]*n2, a1 = vk[nid*3]*n0 + vk[nid*3+1]*n1 + vk[nid*3+2]*n2;
            skb += ww*kk; sal += ww*(al ? al[nid] : 0.f); spin += ww*(1.f - kk); sv0 += ww*a0; sv1 += ww*a1;
            double fn = (double)ww*mp*((a1 - a0)/dt - gn); fnet += fn; fnf += (double)kk*fn;
            if (mm > 0.f) {
                double sc = (double)ww*mp/mm;
                double e_ = -(fe[nid*3]*n0 + fe[nid*3+1]*n1 + fe[nid*3+2]*n2)*sc;
                fel += e_; fef += (double)kk*e_;
                ftot += -(ft[nid*3]*n0 + ft[nid*3+1]*n1 + ft[nid*3+2]*n2)*sc;
            }
        }
        float vp = v[p*3]*n0 + v[p*3+1]*n1 + v[p*3+2]*n2;
        double* o_ = sh + b*14;
        atomicAdd(o_+0, 1.0); atomicAdd(o_+1, (double)mp); atomicAdd(o_+2, (double)mp*skb); atomicAdd(o_+3, (double)mp*sal); atomicAdd(o_+4, (double)mp*spin);
        atomicAdd(o_+5, (double)mp*sv0); atomicAdd(o_+6, (double)mp*sv1); atomicAdd(o_+7, (double)mp*tau);
        atomicAdd(o_+8, fnet); atomicAdd(o_+9, fel); atomicAdd(o_+10, ftot); atomicAdd(o_+11, fnf); atomicAdd(o_+12, fef); atomicAdd(o_+13, (double)mp*vp);
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<9*2*14;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[(size_t)row*252 + i], sh[i]);
}
// 가우스 점 진단 (걸음 끝 시험 X = (I + ∇U) F_q 의 투영 상태·응력) -- 장벽 판 (표본점 위치 AF_BX) 만
#ifdef AF_BAR
__global__ void __launch_bounds__(128) w5_qdg(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* Vq, const float* U,
                       float mu, float lam, float cdp, const float* kb, const float* al, int ax, float c0, float H0, float vf, const float* tbuf, float toff,
                       float n0, float n1, float n2, float pn, double* D, const int* ctr, int cap) {
    __shared__ double sh[9*2*10];
    for (int i=threadIdx.x;i<9*2*10;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    if (blockIdx.x*blockDim.x < IP[12]) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    float xq[3] = {0.f, 0.f, 0.f};
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[qq*4+a]/bm;
    float kq = 0.f, aq = 0.f;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o]; float ww = BW[oc*27+o];
        kq += ww*(kb ? kb[id] : 1.f); aq += ww*(al ? al[id] : 0.f);
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; xq[a] += ww*ua; } }
    float vq = on ? Vq[qq] : 0.f;
    if (on && bm > 0.f && vq > 0.f) {
        float Uu[9], S[3], V[9]; ssvdf(X, Uu, S, V);
        float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
        int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
        float K3 = lam + 2.f*mu/3.f;
        float pr = (st == 2) ? 0.f : -K3*tr;
        float sd = (st == 2) ? 0.f : 2.f*mu*((st == 1) ? s : 1.f)*rho;
        float dep = (st == 1) ? (1.f - s)*rho : ((st == 2) ? rho : 0.f);
        float t = tbuf[0] + toff;
        float dcm = (fabsf(xq[ax] - c0) - (H0 - vf*t))*100.f;
        float hg = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
        int b = w5_bin(dcm)*2 + ((hg >= 0.1f) ? 1 : 0);
        double* o_ = sh + b*10;
        atomicAdd(o_+0, 1.0); atomicAdd(o_+1, (double)vq);
        if (st == 1) atomicAdd(o_+2, (double)vq);
        if (st == 2) atomicAdd(o_+3, (double)vq);
        atomicAdd(o_+4, (double)vq*pr); atomicAdd(o_+5, (double)vq*sd); atomicAdd(o_+6, (double)vq*dep);
        atomicAdd(o_+7, (double)vq*kq); atomicAdd(o_+8, (double)vq*aq); atomicAdd(o_+9, (double)vq*tr);
    }
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<9*2*10;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[(size_t)row*180 + i], sh[i]);
}
#endif
__global__ void w5_inc(int* ctr, float* meta, const float* tbuf, float toff, float dt, int k, int cap) {
    int r = ctr[0];
    if (r < cap) { meta[r*4] = tbuf[0] + toff; meta[r*4+1] = dt; meta[r*4+2] = (float)k; meta[r*4+3] = 1.f; }
    ctr[0] = r + 1;
}
// ---- S_구속: 정답 격자 (dx, 노드 i dx) 상자 [g0, g0 + G) 의 노드 질량·구속 질량 표본 (ao_pbc_acc 와 같은 표본 시각)
__device__ __forceinline__ void w5_gw(float fx, float* w) { w[0] = 0.5f*(1.5f - fx)*(1.5f - fx); w[1] = 0.75f - (fx - 1.f)*(fx - 1.f); w[2] = 0.5f*(fx - 0.5f)*(fx - 0.5f); }
__device__ __forceinline__ bool w5_gbase(const float* x, float idx, int gi0, int gj0, int gk0, int G0, int G1, int G2, int* bl, float (*w)[3]) {
    int g0_[3] = {gi0, gj0, gk0}, GG[3] = {G0, G1, G2};
    for (int a=0;a<3;a++) { float y = x[a]*idx; float bf = floorf(y - 0.5f); int b = (int)bf - g0_[a]; if (b < 0 || b + 3 > GG[a]) return false; bl[a] = b; w5_gw(y - bf, w[a]); }
    return true;
}
__global__ void w5s_acc(int N, const float* x, const float* m, const float* Trel, const float* tbuf, int K, int S, float dt, float idx,
                        int gi0, int gj0, int gk0, int G0, int G1, int G2, float* gm, float* gacc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int bl[3]; float w[3][3];
    if (!w5_gbase(x + p*3, idx, gi0, gj0, gk0, G0, G1, G2, bl, w)) return;
    float tr = Trel[p], t0 = tbuf[0], mp = m[p];
    int KS = K*S, ncut = 0;
    if (tr > t0) for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) ncut = ks + 1; }
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        size_t gid = ((size_t)(bl[0]+i)*G1 + (bl[1]+j))*G2 + (bl[2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        atomicAdd(&gm[gid], ww);
        if (ncut > 0) atomicAdd(&gacc[gid*KS + ncut - 1], ww);
    }
}
// 정답 노드 k_g (ao_pbc_k / mt_pbc_kphi 와 같은 식) -> 1 − k_g 저장 (받침 밖 정확히 0)
__global__ void w5s_kg(int NG, const float* gm, const float* gacc, int K, int S, int k, float expo, int phi, float* omg) {
    int g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= NG) return;
    float m = gm[g], lk = 0.f; int KS = K*S;
    if (m > 0.f) { float suf = 0.f; for (int j=KS-1; j>=k*S; j--) { suf += gacc[(size_t)g*KS + j]; if (j < (k+1)*S) { float a = fminf(suf/m, 1.f); lk += log1pf(-a); } } }
    if (lk == 0.f) { omg[g] = 0.f; return; }
    float r;
    if (phi) { float xx = -lk*expo; if (xx < 1e-3f) r = 1.f - xx/3.f + xx*xx/12.f; else if (isinf(xx)) r = 0.f; else r = 2.f*(xx - 1.f + expf(-xx))/(xx*xx); }
    else r = expf(lk*expo);
    omg[g] = 1.f - r;
}
// 입자 1 − κ_p = Σ_g w (1 − k_g) (정답 G2P), 우리 노드로 질량 가중 P2G: num = Σ w m κ, nom = Σ w m (1 − κ), den = Σ w m
__global__ void w5s_kn(int N, const int* IP, const int* pb, const float* fx, const float* x, const float* m, float idx,
                       int gi0, int gj0, int gk0, int G0, int G1, int G2, const float* omg, float* num, float* nom, float* den) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int bl[3]; float wg[3][3];
    float om = 0.f, kp = 0.f;                                                   // om = Σ w (1 − k_g) (받침 표지), κ = Σ w k_g (음수 없음)
    if (w5_gbase(x + p*3, idx, gi0, gj0, gk0, G0, G1, G2, bl, wg))
        for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; float og = omg[((size_t)(bl[0]+i)*G1 + (bl[1]+j))*G2 + (bl[2]+k)]; float ww_ = wg[0][i]*wg[1][j]*wg[2][k];
                                 if (og != 0.f) om += ww_*og; kp += ww_*(1.f - og); }
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        atomicAdd(&den[nid], ww);
        if (om != 0.f) { atomicAdd(&nom[nid], ww*om); atomicAdd(&num[nid], ww*kp); } else atomicAdd(&num[nid], ww);
    }
}
__global__ void w5s_kb(const int* IP, const float* num, const float* nom, const float* den, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    kb[n] = (nom[n] > 0.f && den[n] > 0.f) ? num[n]/den[n] : 1.f;
}
// 구속 노드 걸음 끝 가속도: 연속판 a ← k a (mt_akb 의 k < 1 -> 0 과 k ∈ {0,1} 에서 같음)
// B_g: 면 36 노드 중 k_구속 < 1 노드가 있으면 면 표지 비트를 지움 (절 CM63 gq_w2stk 와 같은 배치)
__global__ void w5_gfx(const int* IP, const int* cb, const float* kb, unsigned char* fm, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int nb1 = IP[7], nb2 = IP[8]; int s1 = nb2, s0 = nb1*nb2;
    int base = (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
    int cf = 0;
    for (int o=0;o<27;o++) if (kb[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)] < 1.f) { cf = 1; break; }
    int m2 = m, nr = 0;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        int f = cf;
        for (int i=0;i<3 && !f;i++) for (int j=0;j<3 && !f;j++) {
            int id = (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
            if (kb[id] < 1.f) f = 1;
        }
        if (f) { m2 &= ~(1 << a); nr++; }
    }
    if (m2 != m) { fm[c] = (unsigned char)m2; atomicAdd(&cnt[0], nr); }
}
__global__ void w5s_akb(const int* IP, const float* kb, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = kb[n];
    if (k < 1.f) { ak[n*3] *= k; ak[n*3+1] *= k; ak[n*3+2] *= k; }
}
'''

FLOOR_OLD = "    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy;"
FLOOR_NEW = ("    if (mn[n] < mthr) { float kc_ = (kb && W5_CF) ? kb[n] : 1.f;   // 절 CM70 C: 구속 노드의 저질량 고정 자리 = 구속 예측\n"
             "        for (int a=0;a<3;a++) { float yy = Ut[n*3+a] + ((kc_ < 1.f) ? (1.f - kc_)*(z0[n*3+a] + vt[a] - Ut[n*3+a]) : 0.f); yh[n*3+a] = yy;")
ELEM_OLD = "else atomicAdd(&f[ids[o]*3+a], fs[a]); }"
ELEM_NEW = "else atomicAdd(&f[ids[o]*3+a], fs[a]); if (W5_FE) atomicAdd(&W5_FE[ids[o]*3+a], fs[a]); }"


def patch_src(src):
    if "w5_pdg(" in src or "ao_elem_s(" not in src:
        return src
    i = src.index("__global__ void __launch_bounds__(128) ao_elem_s(")
    j = src.find("__global__", i + 10)
    body = src[i:j]
    assert body.count(ELEM_OLD) == 1, ("wolfh5: ao_elem_s 힘 줄", body.count(ELEM_OLD))
    src = src[:i] + body.replace(ELEM_OLD, ELEM_NEW) + src[j:]
    i = src.index("__global__ void __launch_bounds__(128) ao_elem_s(")
    i = src.rfind("\n", 0, src.rfind("__device__ __forceinline__ void ao_pairs", 0, i))
    src = src[:i] + "\n__device__ double* W5_FE;\n" + src[i:]
    i = src.index("__global__ void mt_floor_n(")
    j = src.find("__global__", i + 10)
    body = src[i:j]
    assert body.count(FLOOR_OLD) == 1, ("wolfh5: mt_floor_n 저질량 줄", body.count(FLOOR_OLD))
    src = src[:i] + "__device__ int W5_CF;\n" + body.replace(FLOOR_OLD, FLOOR_NEW) + src[j:]
    body = W5_SRC
    k = src.rstrip().rfind("}")
    return src[:k] + body + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _rel(s_):
    """순차 방출 구속 씬 (구속 입자 있고 구속 속도 없음) 만."""
    return bool(getattr(s_, "has_pbc", False)) and getattr(s_, "pvb", None) is None and getattr(s_, "trel", None) is not None


def _alloc(s_):
    import torch
    dev = s_.dev
    capN = int(s_.capN)
    if getattr(s_, "_w5capN", -1) != capN:
        s_._w5capN = capN
        Z = lambda n, dt=torch.float32: torch.zeros(n, device=dev, dtype=dt)
        s_._w5fe = Z(capN * 3, torch.float64); s_._w5ft = Z(capN * 3, torch.float64)
        s_._w5v0 = Z(capN * 3); s_._w5al = Z(capN)
        s_._w5num = Z(capN); s_._w5nom = Z(capN); s_._w5den = Z(capN)
    if getattr(s_, "_w5ctr", None) is None:
        s_._w5ctr = torch.zeros(1, device=dev, dtype=torch.int32)
        s_._w5D = torch.zeros(CAPR * NB * NH * PF, device=dev, dtype=torch.float64)
        s_._w5Q = torch.zeros(CAPR * NB * NH * QF, device=dev, dtype=torch.float64)
        s_._w5meta = torch.zeros(CAPR * 4, device=dev, dtype=torch.float32)
        s_._w5gc = torch.zeros(2, device=dev, dtype=torch.int32)


def _gtbox(s_):
    """S_구속 정답 격자 상자 (첫 부름의 입자 위치 ± 3 칸)."""
    import torch
    if getattr(s_, "_w5g", None) is not None:
        return s_._w5g
    dx = float(ST["dxg"])
    x = s_.x.reshape(-1, 3)[:s_.N].double()
    lo = torch.floor(x.min(0).values / dx).long() - 3
    hi = torch.floor(x.max(0).values / dx).long() + 4
    G = (hi - lo + 1).tolist()
    K = max(int(s_.K), int(getattr(s_, "kcap", 0) or 0)); S = int(s_.pbc_ns)
    ng = G[0] * G[1] * G[2]
    s_._w5g = dict(lo=[int(q) for q in lo.tolist()], G=[int(q) for q in G], ng=ng, KS=K * S,
                   gm=torch.zeros(ng, device=s_.dev, dtype=torch.float32), gacc=torch.zeros(ng * K * S, device=s_.dev, dtype=torch.float32),
                   omg=torch.zeros(ng, device=s_.dev, dtype=torch.float32))
    ST["info"].update(gt_box_lo=s_._w5g["lo"], gt_box_G=s_._w5g["G"], gt_nodes=ng, gt_KS=K * S, gt_dx=dx)
    return s_._w5g


def _bindfe(s_):
    mod = getattr(s_, "mod", None)
    if mod is None:
        return
    from . import fem_an_rt as ART
    try:
        g = mod.get_global("W5_FE")
    except Exception:                                                             # noqa: BLE001
        return
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s_._w5fe.data_ptr()) if ST["dg"] else 0
    ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=mod.get_global("W5_CF"))[0] = 1 if (ST["c"] and _rel(s_)) else 0


def install(dg=0, s=0, out=".", rel=None, c=0, g=0):
    from . import fem_an_rt as ART
    from . import fem_mat_rt as MRT
    ST.update(dg=int(dg), s=int(s), c=int(c), g=int(g), out=out, rel=rel or {}, dxg=float((rel or {}).get("dxg", 0.01)))
    _ob = ART.build_src
    ART.build_src = lambda *a, **kw: patch_src(_ob(*a, **kw))
    i32, f32 = np.int32, np.float32
    _mb = MRT.MatRT.mat_bind

    def mat_bind(s_):
        r = _mb(s_)
        _alloc(s_)
        ST["sol"] = s_
        _bindfe(s_)
        if ST["s"] and _rel(s_):                                                          # 상자·버퍼는 그래프 캡처 밖에서
            _gtbox(s_)
        return r
    MRT.MatRT.mat_bind = mat_bind
    _k = ART.RTSolver.k

    def k(s_, name, nn, args, b=128):
        if getattr(s_, "_w5in", False) or name.startswith("w5") or not hasattr(s_, "matk") or not s_.matk:
            return _k(s_, name, nn, args, b)
        s_._w5in = True
        try:
            _alloc(s_)
            on_rel = _rel(s_)
            if name == "rt_reset":
                s_._w5ks, s_._w5toff = 0, 0.0
            if ST["dg"]:
                if name == "rt_ga_step0":                                                    # 걸음 시작 노드 속도 (rt_vn·ao 처치 뒤)
                    _k(s_, "w5_copy_f", int(s_.capN) * 3, (P(s_.IP), P(s_.vk), P(s_._w5v0)), 256)
                    s_._w5dt = float(args[4]); s_._w5toff_c = getattr(s_, "_w5toff", 0.0) + 0.5 * float(args[4])
                elif name == "ao_elem_s":
                    _k(s_, "w5_zero_d", int(s_.capN) * 3, (i32(int(s_.capN) * 3), P(s_._w5fe)), 256)
            if ST["s"] and on_rel and name == "mt_akb":
                return _k(s_, "w5s_akb", nn, (args[0], args[1], args[2]), 256)
            r = _k(s_, name, nn, args, b)
            if ST["g"] and on_rel and name == "mt_floor_n" and getattr(s_, "g3fm", None) is not None:   # B_g: 구속 노드 면 빼기 (걸음마다, k_구속 계산 뒤)
                _k(s_, "w5_gfx", int(s_.capQc), (args[0], P(s_.cb), P(s_.kb), P(s_.g3fm), P(s_._w5gc)), 128)
            if ST["dg"] and name == "ao_elem_s":
                _k(s_, "w5_copy_d", int(s_.capN) * 3, (P(s_.IP), P(s_.f), P(s_._w5ft)), 256)
            if name == "ao_pbc_acc" and ST["s"] and on_rel:                                     # 프레임마다: 정답 격자 노드 질량·구속 질량 표본
                G = _gtbox(s_)
                _k(s_, "w5_zero_f", G["ng"], (i32(G["ng"]), P(G["gm"])), 256)
                _k(s_, "w5_zero_f", G["ng"] * G["KS"], (i32(G["ng"] * G["KS"]), P(G["gacc"])), 256)
                _k(s_, "w5s_acc", s_.N, (i32(s_.N), P(s_.x), P(s_.m), P(s_.trel), P(s_.tbuf), args[7], args[8], args[9], f32(1.0 / float(ST["dxg"])),
                                         *[i32(q) for q in G["lo"]], *[i32(q) for q in G["G"]], P(G["gm"]), P(G["gacc"])), 256)
            if name in ("ao_pbc_k", "mt_pbc_kphi"):                                                # (IP, mn, acc, K, S, k, expo, kb)
                if ST["s"] and on_rel:
                    G = _gtbox(s_)
                    _k(s_, "w5s_kg", G["ng"], (i32(G["ng"]), P(G["gm"]), P(G["gacc"]), args[3], args[4], args[5], args[6], i32(1 if name == "mt_pbc_kphi" else 0), P(G["omg"])), 256)
                    for t_ in (s_._w5num, s_._w5nom, s_._w5den):
                        _k(s_, "w5_zero_f", int(s_.capN), (i32(int(s_.capN)), P(t_)), 256)
                    _k(s_, "w5s_kn", s_.N, (i32(s_.N), P(s_.IP), P(s_.Np), P(s_.fx), P(s_.x), P(s_.m), f32(1.0 / float(ST["dxg"])), *[i32(q) for q in G["lo"]], *[i32(q) for q in G["G"]],
                                            P(G["omg"]), P(s_._w5num), P(s_._w5nom), P(s_._w5den)), 256)
                    _k(s_, "w5s_kb", int(s_.capN), (P(s_.IP), P(s_._w5num), P(s_._w5nom), P(s_._w5den), args[7]), 256)
                if ST["dg"]:
                    _k(s_, "w5_alpha", int(s_.capN), (args[0], args[1], args[2], args[3], args[4], args[5], P(s_._w5al)), 256)
            if ST["dg"] and name == "ao_commit":                                                   # 걸음 끝 (커밋 앞: U = 걸음 변위, vk = 걸음 끝 속도)
                FL = s_.floor or dict(NR=[0.0, 0.0, 1.0], PN=-1e30); nr = FL["NR"]; toff = getattr(s_, "_w5toff_c", 0.5 * float(s_.dt)); dt = getattr(s_, "_w5dt", float(s_.dt))
                rl = ST["rel"]; kbp = P(s_.kb) if getattr(s_, "has_pbc", False) else P(None); alp = P(s_._w5al) if getattr(s_, "has_pbc", False) else P(None)
                trel = s_.trel if getattr(s_, "trel", None) is not None else None
                if trel is not None:
                    _k(s_, "w5_pdg", s_.N, (i32(s_.N), P(s_.IP), P(s_.Np), P(s_.fx), P(s_.x), P(s_.m), P(s_.v), P(trel), P(s_.tbuf), f32(toff), f32(rl.get("vf", 0.0)),
                                            kbp, alp, P(s_.mn), P(s_._w5v0), P(s_.vk), P(s_._w5fe), P(s_._w5ft), f32(dt), *[f32(q) for q in s_.g], *[f32(q) for q in nr], f32(FL["PN"]),
                                            P(s_._w5D), P(s_._w5ctr), i32(CAPR)), 256)
                if getattr(s_, "bar", False):
                    _k(s_, "w5_qdg", nn, (args[0], args[1], args[2], P(s_.BW), args[3], P(s_.Vq), args[4], f32(s_.mu), f32(s_.lam), args[5], kbp, alp,
                                          i32(rl.get("ax", 0)), f32(rl.get("c0", 0.0)), f32(rl.get("H0", 0.0)), f32(rl.get("vf", 0.0)), P(s_.tbuf), f32(toff),
                                          *[f32(q) for q in nr], f32(FL["PN"]), P(s_._w5Q), P(s_._w5ctr), i32(CAPR)), 128)
                _k(s_, "w5_inc", 1, (P(s_._w5ctr), P(s_._w5meta), P(s_.tbuf), f32(toff), f32(dt), i32(getattr(s_, "_w5ks", 0)), i32(CAPR)), 1)
                s_._w5ks = getattr(s_, "_w5ks", 0) + 1; s_._w5toff = getattr(s_, "_w5toff", 0.0) + dt
            elif name == "ao_commit":
                s_._w5ks = getattr(s_, "_w5ks", 0) + 1
            return r
        finally:
            s_._w5in = False
    ART.RTSolver.k = k
    _os = ART.RTSolver.step

    def step(s_, t, *a, **kw):
        d = getattr(s_, "_w5dep", 0)
        s_._w5dep = d + 1
        try:
            r = _os(s_, t, *a, **kw)
        finally:
            s_._w5dep = d
        if d == 0 and ST["dg"]:
            c = getattr(s_, "_w5ctr", None)
            ST["marks"].append((int(round(t / s_.h)), int(c.item()) if c is not None else -1))
        if d == 0 and ST["s"] and not ST["said"] and getattr(s_, "_w5g", None) is not None:
            ST["said"] = True
            print(f"[절 CM70] S_구속 정답 격자 상자 {s_._w5g['G']} 노드 {s_._w5g['ng']} 표본 {s_._w5g['KS']}", flush=True)
        return r
    ART.RTSolver.step = step
    print(f"[절 CM70] wolfh5 켬: 진단 {ST['dg']} · S_구속 {ST['s']} · C {ST['c']} · B_g {ST['g']} · 방출 {ST['rel']}", flush=True)

    def _fin():
        info = dict(dg=ST["dg"], s=ST["s"], c=ST["c"], g=ST["g"], rel=ST["rel"], **ST["info"])
        sol = ST.get("sol")
        if sol is not None:
            try:
                import torch
                torch.cuda.synchronize()
                info["rel_scene"] = _rel(sol)
                if ST["g"] and getattr(sol, "_w5gc", None) is not None:
                    info["g_removed_faces_total"] = int(sol._w5gc[0].item())
                if ST["s"] and getattr(sol, "has_pbc", False):
                    kb = sol.kb[:int(sol.IP[9].item())]
                    info.update(kb_lt1=int((kb < 1).sum()), kb_lt05=int((kb < 0.5).sum()), kb_min=float(kb.min()) if kb.numel() else None)
                if ST["dg"]:
                    nr = int(sol._w5ctr.item())
                    D = sol._w5D.view(CAPR, NB, NH, PF)[:min(nr, CAPR)].cpu().numpy()
                    Q = sol._w5Q.view(CAPR, NB, NH, QF)[:min(nr, CAPR)].cpu().numpy()
                    M = sol._w5meta.view(CAPR, 4)[:min(nr, CAPR)].cpu().numpy()
                    np.savez_compressed(os.path.join(out, "w5dg.npz"), D=D, Q=Q, meta=M, marks=np.array(ST["marks"]))
                    fr, lo = [], 0
                    for f_, hi in ST["marks"]:
                        hi = min(hi, D.shape[0])
                        if hi > lo:
                            rows = list(range(lo, hi))
                            k0 = [r_ for r_ in rows if int(M[r_, 2]) == 0]                       # 쪼개기로 다시 푼 걸음: 마지막 시도 (걸음 0 부터) 만
                            use = rows[rows.index(k0[-1]):] if k0 else rows
                            fr.append(dict(f=f_, rows=[lo, hi], use=use, dt=[float(M[r_, 1]) for r_ in use],
                                           P=np.sum([D[r_] * M[r_, 1] for r_ in use], 0).tolist(), Q=np.sum([Q[r_] * M[r_, 1] for r_ in use], 0).tolist()))
                        lo = hi
                    json.dump(dict(edges_cm=EDGES, pfields=PFIELDS, qfields=QFIELDS, nrows=nr, frames=fr,
                                   note="P·Q = 그 프레임 걸음들의 Δt 가중 합 (나누기는 표에서 Σ Δt)"), open(os.path.join(out, "w5dg.json"), "w"))
                    info["dg_rows"] = nr
            except Exception as e:                                                       # noqa: BLE001
                info["err"] = repr(e)
        json.dump(info, open(os.path.join(out, "w5info.json"), "w"))
    atexit.register(_fin)
