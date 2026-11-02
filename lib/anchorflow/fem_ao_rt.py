"""절 AO 실시간 판: 절 AN 실시간 판 (lib/anchorflow/fem_an_rt.py RTSolver: 손으로 쓴 CUDA 커널 + 고정 용량 버퍼 + 프레임 CUDA 그래프) 에 모래 (PG Drucker-Prager) 를 얹은 융합 판.
정의는 exe/fem_ao_sim.py --fast --pc jacobi --tan el --sand_ls 0 (+ --pbc --mlow) 와 같다 (정밀도는 절 AN 실시간 판과 같은 분리: 곱·누적 f32, 기울기·내적 f64).
바뀐 커널 (절 AN 파일은 고치지 않고 SRC 에 덧붙여 따로 컴파일):
- ao_elem   : 가우스 점 시험 X -> 부호 있는 SVD -> 정답 RM (ε = log|σ|, 꼬리 투영·꼭짓점) -> 응력 P = U diag(τ̂/σ) Vᵀ (기울기 f64 원자), 헤시안 = 시험 X 의 Hencky 탄성 (투영, 초기 강성법), 노드 대각 융합.
- ao_commit : 걸음 끝 소성 상태 커밋 F_q ← F_q X⁻¹ RM(X) (G_qn = F_qᵀ ∇w 는 커널 안에서 F_q 로 계산하므로 같이 바뀜).
- ao_g2p    : 걸음별 노드 변위 (걸음마다 복사해 둔 U_k) 로 입자 F ← RM(A_k A_{k−1}⁻¹ F) 걸음 순서대로, x·v 는 마지막 걸음.
- ao_vzero / ao_pbc_acc / ao_pbc_k : 입자 속도 구속 (순차 방출) -- 프레임 시작 구속 입자 v = 0, 걸음마다 노드 몫 α_n(t_s) 의 S 표본으로 k_구속 = exp(Σ log(1 − α) Δt/(S dt_gt)).
- ao_floor_n: rt_floor_n + 구속 인자 곱 + 거의 빈 노드 (질량 < 문턱) 는 관성 예측 고정.
  근사 하나: 바닥 이탈 (바깥 법선) 노드가 구속도 받으면 (k_구속 < 1) 세 성분 모두 k_바닥 k_구속 으로 (torch 판은 법선 성분에 k_구속 만) -- 구속 노드는 k_구속 이 사실상 0 이라 차이가 없다.
시각은 장치 버퍼 tbuf[0] (프레임 시작 시각) 로 그래프 밖에서 매 프레임 채운다 (그래프는 한 번만 캡처)."""
import numpy as np
import torch

from . import fem_an_rt as ART
from . import fem_det_rt as DET

P = ART.P

AO_SRC = r'''
__device__ __forceinline__ int ao_dpf(const float* S, float cdp, float* sgn, float* sg, float* ep, float* e, float& tr, float& rho, float& s) {
    for (int i=0;i<3;i++) { sgn[i] = (S[i] < 0.f) ? -1.f : 1.f; sg[i] = fmaxf(fabsf(S[i]), 1e-14f); ep[i] = logf(sg[i]); }
    tr = ep[0] + ep[1] + ep[2];
    for (int i=0;i<3;i++) e[i] = ep[i] - tr/3.f;
    rho = sqrtf(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    float dg = rho + cdp*tr;
    s = 1.f;
    if (!(dg > 0.f)) return 0;
    if (tr > 0.f) { s = 0.f; return 2; }
    s = -cdp*tr/rho;
    return 1;
}
__global__ void ao_vzero(int N, const int* IP, const float* Trel, const float* tbuf, float* v) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (Trel[p] > tbuf[0]) { v[p*3] = 0.f; v[p*3+1] = 0.f; v[p*3+2] = 0.f; }
}
// 구속 입자 질량을 걸음 k x 표본 s 시각마다 노드에 (acc[n*KS + k*S + s])
__global__ void ao_pbc_acc(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* Trel, const float* tbuf, int K, int S, float dt, float* acc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float tr = Trel[p], t0 = tbuf[0];
    if (!(tr > t0)) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8], KS = K*S;
    float mp = m[p];
    int ncut = 0;
    for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) ncut = ks + 1; }
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        if (ncut > 0) atomicAdd(&acc[(size_t)nid*KS + ncut - 1], ww);       // 표본 시각은 커지는 순서 -> 구속된 표본은 앞쪽 ncut 개 (끝 자리에만 더하고 노드에서 뒤쪽 누적합)
    }
}
__global__ void ao_pbc_k(const int* IP, const float* mn, const float* acc, int K, int S, int k, float expo, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], lk = 0.f; int KS = K*S;
    if (m > 0.f) {
        float suf = 0.f;                                                        // 표본 ks 의 구속 질량 = Σ_{j ≥ ks} acc[j]
        for (int j=KS-1; j>=k*S; j--) { suf += acc[(size_t)n*KS + j]; if (j < (k+1)*S) { float al = fminf(suf/m, 1.f); lk += log1pf(-al); } }
    }
    kb[n] = expf(lk*expo);
}
__global__ void ao_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta,
                           const float* kb, const float* mn, float mthr) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
    }
    if (kb) { float b = kb[n]; if (b < 1.f) { k *= b; p = 0; } }
    if (fixd && fixd[n]) p = 2;
    if (p == 2) { for (int a=0;a<3;a++) { float yy = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    float s = sqrtf(k);
    float d[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] - Ut[n*3+a];
    float Pd[3];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 모래 요소: mode 1 기울기 (RM 응력), 2 헤시안 (시험 X Hencky 탄성, 투영), 8 노드 대각 융합. st 수 IPq[0..1] (꼬리, 꼭짓점) 원자 합 (진단)
__global__ void __launch_bounds__(128) ao_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, float cdp, int mode,
                        double* f, float* H, int capQ, float* B, int* cnt) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (cnt && on && st > 0 && (mode & 1)) atomicAdd(&cnt[st-1], 1);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float vq = on ? Vq[qq] : 0.f;
    if (mode & 1) {
        float Pm[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
            for (int a=0;a<3;a++) { double sv = seg8d((double)(vq*(Pm[a*3]*g[0] + Pm[a*3+1]*g[1] + Pm[a*3+2]*g[2])));
                if (on && head) atomicAdd(&f[ids[o]*3+a], sv); } }
    }
    if (mode & 2) {
        float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
        if (on) {
        float D[3][3], Ev[3][3], Dk[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f) || !isfinite(dm)) continue;
            float ee[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) ee[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*ee[r]*ee[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp];
            float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;
            float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
        if (mode & 512) { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) ((__half*)H)[(size_t)(k++)*capQ + qq] = __float2half(h[r*9+c]/(2.f*mu + lam)); }   // f16 위 삼각
        else if (mode & 128) { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) H[(size_t)(k++)*capQ + qq] = h[r*9+c]; }   // 위 삼각 45 개 (h 는 비트 대칭)
        else for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
        }
        if ((mode & 8) && (mode & 256)) {                                        // 노드 대각 (공유 메모리 판): 칸 8 점의 헤시안·F·V 를 공유 메모리에, 레인마다 노드 o = 레인, 레인+8, ... 을 맡아 8 점을 직접 합 (셔플 없음)
            __shared__ float shH[128][45]; __shared__ float shF[128][9]; __shared__ float shV[128];
            int tl = threadIdx.x;
            { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) shH[tl][k++] = h[r*9+c]; }
            for (int k=0;k<9;k++) shF[tl][k] = F[k];
            shV[tl] = vq;
            __syncwarp();
            int l8 = tl & 7, b8 = tl & ~7;
            for (int o = l8; o < 27; o += 8) {
                float mm[9] = {0,0,0,0,0,0,0,0,0};
                for (int j=0;j<8;j++) {
                    const float* hj = shH[b8+j]; float g[3]; gqn(shF[b8+j], BG + (j*27+o)*3, g); float vj = shV[b8+j];
                    for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                        for (int b=0;b<3;b++) for (int dd=0;dd<3;dd++) { int r_ = a*3+b, c_ = c*3+dd; int lo = r_ < c_ ? r_ : c_, hi = r_ < c_ ? c_ : r_;
                            t += hj[lo*9 - lo*(lo-1)/2 + (hi - lo)]*g[b]*g[dd]; }
                        mm[a*3+c] += vj*t; }
                }
                if (on) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); }
            }
        } else if (mode & 8) {
            for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
                float mm[9];
                for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                    for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
                    mm[a*3+c] = vq*t; }
                for (int k=0;k<9;k++) mm[k] = seg8f(mm[k]);
                if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], mm[k]); }
        }
    }
}
__device__ __forceinline__ bool ao_rm3(const float* X, float cdp, float* Y) {
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (st == 0) { for (int k=0;k<9;k++) Y[k] = X[k]; return false; }
    float sn[3]; for (int i=0;i<3;i++) sn[i] = (st == 2) ? sgn[i] : sgn[i]*expf(tr/3.f + s*e[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Y[a*3+b] = U[a*3]*sn[0]*V[b*3] + U[a*3+1]*sn[1]*V[b*3+1] + U[a*3+2]*sn[2]*V[b*3+2];
    return true;
}
__device__ __forceinline__ void ao_mm3(const float* A, const float* B, float* C) {
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) C[a*3+b] = A[a*3]*B[b] + A[a*3+1]*B[3+b] + A[a*3+2]*B[6+b];
}
// 걸음 끝 커밋: X = F_q + Σ U ⊗ g (U = 걸음 끝 변위), F_q ← F_q X⁻¹ RM(X)
__global__ void ao_commit(const int* IP, const int* cb, const float* BG, float* Fq, const float* U, float cdp) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Y[9];
    if (!ao_rm3(X, cdp, Y)) return;
    float Xi[9]; inv3(X, Xi);
    float T1[9], T2[9]; ao_mm3(Xi, Y, T1); ao_mm3(F, T1, T2);
    bool ok = true; for (int k=0;k<9;k++) if (!isfinite(T2[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = T2[k];
}
// G2P (모래): 걸음별 U_k (Ust[k*capN*3 + ...]) 로 F ← RM(A_k A_{k−1}⁻¹ F), x += Σ w U_K, v = Σ w v_K. 입자 RM 수 cnt[2..3]
__global__ void ao_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* Ust, int K, int capN, const float* vk, float cdp,
                       float* x, float* v, float* F, int* cnt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float Ap[9] = {1,0,0, 0,1,0, 0,0,1};
    float u[3] = {0,0,0}, vv[3] = {0,0,0};
    int nrm = 0;
    for (int kk=0; kk<K; kk++) {
        const float* Uk = Ust + (size_t)kk*capN*3;
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        bool last = kk == K-1;
        for (int o=0;o<27;o++) {
            int i = o/9, j = (o/3)%3, k = o%3;
            int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
            float ww = w[0][i]*w[1][j]*w[2][k];
            float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
            float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
            for (int a=0;a<3;a++) { float uu = Uk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; if (last) { u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; } }
        }
        float Api[9]; inv3(Ap, Api);
        float Ainc[9], Ft[9]; ao_mm3(A, Api, Ainc); ao_mm3(Ainc, Fp, Ft);
        float Y[9]; if (ao_rm3(Ft, cdp, Y)) nrm++;
        for (int k=0;k<9;k++) { Fp[k] = Y[k]; Ap[k] = A[k]; }
    }
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int k=0;k<9;k++) F[(size_t)p*9+k] = Fp[k];
    if (cnt && nrm) atomicAdd(&cnt[2], 1);
}
// θ 방식 걸음 끝 (절 AN.err2 정의): y = U_k + θ (U_{k+1} − U_k), v_{k+1} = (y − z0)/(θτ), U_{k+1} = U_k + (y − U_k)/θ
__global__ void ao_stepend_th(const int* IP, float ith, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = al*(yy - z0[n*3+a]);
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = u + (yy - u)*ith;
        vk[n*3+a] = v1*damp;
    }
}
__global__ void ao_act_reset(int* IP) { IP[14] = 0; }
__global__ void ao_act(int* IP, const float* mn) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (mn[n] > 0.f) { int k = atomicAdd(&IP[14], 1); AO_ACT[k] = n; }
}
// PCG 첫 추정 (앞 걸음 첫 뉴턴의 δ): rt_grad 뒤에 dl = d0, r = −g − A d0, z = D r, p = z, Tp = T p, SC0 = r·z
__global__ void ao_tvec(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* v, float* o) {
    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii_ >= IP[14]) return;
    int n = AO_ACT[ii_];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) o[n*3+a] = t[a];
}
__global__ void ao_warm(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* d0, const float* KTd,
                        const float* Dinv, float* dl, float* r, float* z, float* p, float* Tp, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[14]) return;
    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0;
    if (ii_ < IP[14]) {
        int n = AO_ACT[ii_];
        float Mi = mn[n]*itau2;
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, KTd + n*3, t);
        float rr[3];
        for (int a=0;a<3;a++) { float d = d0[n*3+a]; dl[n*3+a] = d; rr[a] = r[n*3+a] - (Mi*d + t[a]); r[n*3+a] = rr[a]; }
        const float* D = Dinv + n*9;
        float zz[3]; for (int a=0;a<3;a++) { zz[a] = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]);
}

// 헤시안 곱 (칸 공유 메모리 판, 위 삼각 45 저장): 칸 27 노드 입력을 공유 메모리에 한 번, 레인마다 노드 o = 레인 + 8k 를 맡아 8 점의 V (H δX)·g 를 직접 합
__global__ void __launch_bounds__(128) ao_mv_c(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sP[128][9]; __shared__ float sF[128][9];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = dy[id*3+a]; }
    float F[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)qq*9+k]; sF[tl][k] = F[k]; }
    __syncwarp();
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; dX[a*3] += ya*g[0]; dX[a*3+1] += ya*g[1]; dX[a*3+2] += ya*g[2]; } }
    float vq = on ? Vq[qq] : 0.f;
    for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) { int a_ = r < c ? r : c, b_ = r < c ? c : r; t += H[(size_t)(a_*9 - a_*(a_-1)/2 + (b_ - a_))*capQ + qq]*dX[c]; } sP[tl][r] = vq*t; }
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        float sv[3] = {0.f, 0.f, 0.f};
        for (int j=0;j<8;j++) { float g[3]; gqn(sF[b8+j], BG + (j*27+o)*3, g); const float* dP = sP[b8+j];
            for (int a=0;a<3;a++) sv[a] += dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2]; }
        if (on) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = sv[a]; else atomicAdd(&out[ids[o]*3+a], sv[a]); }
    }
}
// ---- 주값 분해 저장 판 (--hspec): 탄성 접선 헤시안 = Σ_m dm e_m e_mᵀ + Σ_쌍 (lt e_t e_tᵀ + lf e_f e_fᵀ), e = U Ê Vᵀ (순위 1 항 9 개).
// 점마다 H[k*capQ + q] (k = 0..29): U (9), M = F V (9), D⁺ = Ev diag(dm) Evᵀ 위 삼각 (00 11 22 01 02 12) x V_q, c1 = (lt+lf)/2 x V_q (쌍 01 02 12), c2 = (lf−lt)/2 x V_q.
// 곱: dX = dA F (dA = Σ_o y_o ⊗ ∇w_o), X̂ = Uᵀ dA M, P̂_ii = (D⁺ diag X̂)_i, P̂_ij = c1 X̂_ij + c2 X̂_ji, 노드 o 몫 = (U P̂ Mᵀ) ∇w_o.
// 노드 대각: h = Mᵀ ∇w_o, S = diag(h) D⁺ diag(h) + 쌍 ((i,i) += c1 h_j², (j,j) += c1 h_i², (i,j) += c2 h_i h_j), 블록 = U S Uᵀ.
__device__ __forceinline__ void ao_pairs(int pp, int& i, int& j) { i = (pp == 2) ? 1 : 0; j = (pp == 0) ? 1 : 2; }
__global__ void __launch_bounds__(128) ao_elem_s(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, float cdp, int mode,
                        double* f, float* H, int capQ, float* B, int* cnt) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sQ[128][9]; __shared__ float sU[128][9]; __shared__ float sM[128][9]; __shared__ float sC[128][12];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3, oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = y[id*3+a]; }
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (cnt && on && st > 0) atomicAdd(&cnt[st-1], 1);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float vq = on ? Vq[qq] : 0.f;
    // 힘: Qf = V_q P Fᵀ (P = U diag(p) Vᵀ), 노드 o 몫 = Qf ∇w_o
    { float Pm[9];
      for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = vq*(U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2]);
      for (int a=0;a<3;a++) for (int c=0;c<3;c++) sQ[tl][a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2]; }
    // 헤시안 주값 계수
    float M[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) M[a*3+b] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
    float C[12];
    { float D[3][3], Ev[3][3], Dk[3][3];
      for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
      jac3f(D, Ev);
      float dm[3]; for (int m=0;m<3;m++) { float d_ = fmaxf(D[m][m], 0.f); dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }
      const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
      for (int k=0;k<6;k++) C[k] = vq*(dm[0]*Ev[RI[k]][0]*Ev[RJ[k]][0] + dm[1]*Ev[RI[k]][1]*Ev[RJ[k]][1] + dm[2]*Ev[RI[k]][2]*Ev[RJ[k]][2]);
      for (int pp=0; pp<3; pp++) { int i, j; ao_pairs(pp, i, j);
          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;
          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);
          lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;
          C[6+pp] = vq*0.5f*(lt + lf); C[9+pp] = vq*0.5f*(lf - lt); } }
    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; } for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = C[k]; }
    for (int k=0;k<9;k++) { sU[tl][k] = U[k]; sM[tl][k] = M[k]; }
    for (int k=0;k<12;k++) sC[tl][k] = C[k];
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        double fs[3] = {0.0, 0.0, 0.0};
        float bb[6] = {0,0,0,0,0,0};
        for (int j=0;j<8;j++) {
            const float* bg = BG + (j*27+o)*3;
            const float* Qj = sQ[b8+j];
            for (int a=0;a<3;a++) fs[a] += (double)(Qj[a*3]*bg[0] + Qj[a*3+1]*bg[1] + Qj[a*3+2]*bg[2]);
            const float* Mj = sM[b8+j]; const float* Cj = sC[b8+j]; const float* Uj = sU[b8+j];
            float h[3]; for (int b=0;b<3;b++) h[b] = Mj[b]*bg[0] + Mj[3+b]*bg[1] + Mj[6+b]*bg[2];
            float S6[6] = {Cj[0]*h[0]*h[0] + Cj[6]*h[1]*h[1] + Cj[7]*h[2]*h[2],
                           Cj[1]*h[1]*h[1] + Cj[6]*h[0]*h[0] + Cj[8]*h[2]*h[2],
                           Cj[2]*h[2]*h[2] + Cj[7]*h[0]*h[0] + Cj[8]*h[1]*h[1],
                           (Cj[3] + Cj[9])*h[0]*h[1], (Cj[4] + Cj[10])*h[0]*h[2], (Cj[5] + Cj[11])*h[1]*h[2]};
            float Sm[9] = {S6[0], S6[3], S6[4], S6[3], S6[1], S6[5], S6[4], S6[5], S6[2]};
            float T[9]; for (int a=0;a<3;a++) for (int c=0;c<3;c++) T[a*3+c] = Uj[a*3]*Sm[c] + Uj[a*3+1]*Sm[3+c] + Uj[a*3+2]*Sm[6+c];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) bb[k] += T[RI[k]*3]*Uj[RJ[k]*3] + T[RI[k]*3+1]*Uj[RJ[k]*3+1] + T[RI[k]*3+2]*Uj[RJ[k]*3+2];
        }
        float mm[9] = {bb[0], bb[3], bb[4], bb[3], bb[1], bb[5], bb[4], bb[5], bb[2]};
        if (on) {
            for (int a=0;a<3;a++) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = fs[a]; else atomicAdd(&f[ids[o]*3+a], fs[a]); }
            for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); }
        }
    }
}
__global__ void __launch_bounds__(128) ao_mv_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sP[128][9];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = dy[id*3+a]; }
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float U[9], M[9], C[12];
    for (int k=0;k<9;k++) { U[k] = H[(size_t)k*capQ + qq]; M[k] = H[(size_t)(9+k)*capQ + qq]; }
    for (int k=0;k<12;k++) C[k] = on ? H[(size_t)(18+k)*capQ + qq] : 0.f;
    float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = A[a*3]*M[b] + A[a*3+1]*M[3+b] + A[a*3+2]*M[6+b];
    float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[a]*T[b] + U[3+a]*T[3+b] + U[6+a]*T[6+b];
    float Ph[9];
    Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
    Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
    Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
    Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
    Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
    Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
    float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) sP[tl][a*3+c] = W[a*3]*M[c*3] + W[a*3+1]*M[c*3+1] + W[a*3+2]*M[c*3+2];
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        float sv[3] = {0.f, 0.f, 0.f};
        for (int j=0;j<8;j++) { const float* bg = BG + (j*27+o)*3; const float* Q = sP[b8+j];
            for (int a=0;a<3;a++) sv[a] += Q[a*3]*bg[0] + Q[a*3+1]*bg[1] + Q[a*3+2]*bg[2]; }
        if (on) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = sv[a]; else atomicAdd(&out[ids[o]*3+a], sv[a]); }
    }
}
'''


HB = "    if (blockIdx.x*blockDim.x >= IP[9]) return;\n    int n = blockIdx.x*blockDim.x + threadIdx.x;\n"
HB2 = "    if (blockIdx.x*blockDim.x >= IP[14]) return;\n    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    int n = (ii_ < IP[14]) ? AO_ACT[ii_] : IP[9];\n"
HS = "    int n = blockIdx.x*blockDim.x + threadIdx.x;\n    if (n >= IP[9]) return;\n"
HS2 = "    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;\n    if (ii_ >= IP[14]) return;\n    int n = AO_ACT[ii_];\n"


def _compact(src):
    """노드 압축 판: rt_grad·rt_ap·rt_cg1 (블록 합) 과 rt_cg2·rt_lsup 를 질량 있는 노드 목록 AO_ACT[0..IP[14]) 만 돌게 (이름 뒤 _c). 원본 커널은 그대로 둔다."""
    out = []
    for nm, h0, h1 in (("rt_grad", HB, HB2), ("rt_ap", HB, HB2), ("rt_cg1", HB, HB2), ("rt_cg2", HS, HS2), ("rt_lsup", HS, HS2)):
        i = src.index("__global__ void " + nm + "(")
        j = src.index("\n}\n", i) + 3
        body = src[i:j].replace("void " + nm + "(", "void " + nm + "_c(", 1)
        assert h0 in body, nm
        out.append(body.replace(h0, h1, 1))
    return "\n".join(out)


def _mv_packed(head):
    """절 AN rt_mv 본문에서 헤시안 읽기만 위 삼각 45 개 판으로 바꾼 ao_mv_s (곱·합 순서 같음)."""
    i = head.index("__global__ void __launch_bounds__(128) rt_mv(")
    j = head.index("\n}\n", i) + 3
    body = head[i:j].replace("rt_mv(", "ao_mv_s(", 1)
    old = "for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c];"
    assert old in body
    new = "for (int c=0;c<9;c++) { int a_ = r < c ? r : c, b_ = r < c ? c : r; t += H[(size_t)(a_*9 - a_*(a_-1)/2 + (b_ - a_))*capQ + qq]*dX[c]; }"
    return body.replace(old, new, 1)


def _mv_half(ao):
    """ao_mv_c 에서 헤시안 읽기만 f16 으로 바꾼 ao_mv_h."""
    i = ao.index("__global__ void __launch_bounds__(128) ao_mv_c(")
    j = ao.index("\n}\n", i) + 3
    body = ao[i:j].replace("ao_mv_c(", "ao_mv_h(", 1)
    old = "t += H[(size_t)(a_*9 - a_*(a_-1)/2 + (b_ - a_))*capQ + qq]*dX[c];"
    assert old in body
    body = body.replace(old, "t += __half2float(((const __half*)H)[(size_t)(a_*9 - a_*(a_-1)/2 + (b_ - a_))*capQ + qq])*dX[c];", 1)
    body = body.replace("float* out, int capQ) {", "float* out, int capQ, float hsc) {", 1)
    old2 = "sP[tl][r] = vq*t; }"
    assert old2 in body
    return body.replace(old2, "sP[tl][r] = vq*hsc*t; }", 1)                  # f16 은 2μ+λ 로 나눠 저장 (범위), 곱에서 되돌림


def _src():
    s = ART.SRC.rstrip()
    i = s.rfind("}")
    head = DET.det_src(s[:i])
    head = "#include <cuda_fp16.h>\n" + head
    return head.replace("extern \"C\" {", "extern \"C\" {\n__device__ int* AO_ACT;", 1) + _mv_half(DET.det_ao(AO_SRC)) + DET.det_ao(AO_SRC) + _compact(head) + _mv_packed(head) + DET.DET_SRC + "\n}\n"


class AORTSolver(DET.DetMixin):
    """모래 융합 판. 추가 인자: cdp (PG DP c), trel (입자별 구속 끝 시각 [N] 또는 None), pbc_ns (표본 수), dt_gt (정답 substep_dt), mthr (거의 빈 노드 질량 문턱, 절대값)."""

    def __init__(s, *args, cdp=1.0, trel=None, pbc_ns=4, dt_gt=2e-5, mthr=0.0, compact=True, warm=True, det=False, hpack=False, bdsh=False, mvc=False, h16=False, theta=None, hspec=False, **kw):
        s.cdp, s.pbc_ns, s.dt_gt, s.mthr = float(cdp), int(pbc_ns), float(dt_gt), float(mthr)
        s.det = bool(det)
        s.hpack = bool(hpack)
        s.bdsh = bool(bdsh)
        s.mvc = bool(mvc)
        s.h16 = bool(h16)
        s.hspec = bool(hspec)
        s.has_pbc = trel is not None
        s.compact, s.warm = bool(compact), bool(warm)
        super().__init__(*args, **kw)
        s.th = None if theta is None else float(theta)                                  # θ 방식 (None = integ 의 중점/후퇴 오일러 그대로)
        if s.th is not None:
            s.set_th()
        s.mod = ART.cp.RawModule(code=_src(), options=("--std=c++14",))
        s.fn = {}
        s.ao_ready = True
        s.bind_act()
        s.tbuf = torch.zeros(1, device=s.dev, dtype=torch.float32)
        s.cnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.trel = (trel[s.perm].float().contiguous() if s.has_pbc else None)
        s.graph = None

    def mvname(s):
        if s.hspec:
            return "ao_mv_q"
        return ("ao_mv_h" if s.h16 else ("ao_mv_c" if s.mvc else "ao_mv_s")) if s.hpack else "rt_mv"

    def set_th(s):
        s.tau = s.th * s.dt; s.beta = 1.0 - s.th; s.al = 1.0 / (s.th * s.tau)

    def alloc(s, need):
        K0 = s.K
        s.K = max(K0, getattr(s, "kcap", K0))                                           # 적응 걸음 수: 버퍼는 걸음 수 상한으로
        super().alloc(need)
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.Ust = Z(s.K, s.capN, 3)
        if getattr(s, "has_pbc", False):
            s.pacc = Z(s.capN * s.K * s.pbc_ns)
            s.kb = Z(s.capN)
        s.act = torch.zeros(s.capN, device=s.dev, dtype=torch.int32)
        s.det_alloc()
        s.d0 = Z(s.capN, 3); s.td = Z(s.capN, 3); s.KTd = Z(s.capN, 3)
        s.kalloc = s.K
        s.K = K0
        s.graph = None
        if getattr(s, "ao_ready", False):
            s.bind_act()

    def bind_act(s):
        g = s.mod.get_global("AO_ACT")
        ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.act.data_ptr())
        s.det_bind()

    def set_bc(s, t):
        super().set_bc(t)
        s.tbuf.fill_(float(t))

    def record(s, t_frame, K=None):
        N, IP = s.N, s.IP
        i32, f = np.int32, np.float32
        capN, capQ = s.capN, s.capQc * 8
        s.k("rt_reset", 1, (P(IP),), b=1)
        s.zf(s.cnt)
        s.k("rt_coord", N, (i32(N), P(s.x), P(s.Ai), P(s.Np), P(s.fx), P(IP)))
        s.k("rt_box", 1, (P(IP), i32(capN), i32(s.capC)), b=1)
        s.zf(s.occ)
        s.k("rt_mark", N, (i32(N), P(s.Np), P(s.fx), P(IP), P(s.pcd), P(s.occ)))
        s.k("rt_scan", 1024, (P(IP), P(s.occ), P(s.cidx), P(s.cb), i32(s.capQc)), b=1024)
        for t_, ci, mul in ((s.Aacc, 12, 28), (s.Cacc, 11, 30), (s.mn, 9, 1), (s.mv, 9, 3), (s.mqn, 9, 1)):
            s.zf(t_, ci, mul)
        s.zf(s.badq); s.zf(s.badc); s.zf(s.nbad)
        if s.has_pbc:
            s.k("ao_vzero", N, (i32(N), P(IP), P(s.trel), P(s.tbuf), P(s.v)))
        s.k("rt_p2g", N, (i32(N), P(s.x), P(s.v), P(s.F), P(s.m), P(s.vol), P(s.Np), P(s.fx), P(s.pcd), P(s.cidx), P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.mn), P(s.mv)))
        s.k("rt_gauss", capQ, (P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.cb), P(s.A), P(s.BW), P(s.Fq), P(s.Vq), P(s.mq), P(s.xq0), P(s.mqn), P(s.nbad)))
        s.k("rt_vn", capN, (P(IP), P(s.mn), P(s.mv), P(s.vk), P(s.U)), b=256)
        s.k("ao_act_reset", 1, (P(IP),), b=1)
        s.k("ao_act", capN, (P(IP), P(s.mn)), b=256)
        if s.warm:
            s.zf(s.d0, 9, 3)
        if s.nl > 1:                                                                   # 절 AN 실시간 판 MG (거친 층 = 프레임 시작 F 의 FCR 연산자 -- 전처리로만)
            s.mg_build()
        KS = s.K * s.pbc_ns
        if s.has_pbc:
            s.zf(s.pacc, 9, KS)
            s.k("ao_pbc_acc", N, (i32(N), P(IP), P(s.Np), P(s.fx), P(s.m), P(s.trel), P(s.tbuf), i32(s.K), i32(s.pbc_ns), f(s.dt), P(s.pacc)))
        if s.bc:
            s.fixd.zero_()
            for ib in range(len(s.bc)):
                s.k("rt_bc_box", capN, (P(IP), P(s.A), P(s.BP), i32(ib), P(s.fixd), P(s.vbc)), b=256)
        FL = s.floor
        nr = FL["NR"] if FL else [0.0, 0.0, 1.0]
        it2 = f(1.0 / (s.tau * s.tau))
        expo = f(s.dt / s.pbc_ns / s.dt_gt)
        for kstep in range(s.K):
            s.k("rt_step0", capN, (P(IP), P(s.U), P(s.vk), f(s.tau), f(s.beta), *[f(q) for q in s.g], P(s.fixd), P(s.vbc), P(s.Ut), P(s.z0), P(s.y)), b=256)
            if FL:
                s.zf(s.an, 9, 1)
                s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), f((1.0 / s.th) if s.th is not None else (2.0 if s.mid else 1.0)), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                         f(FL["PN"]), f(FL["DXG"]), P(s.an)))
            if s.has_pbc:
                s.k("ao_pbc_k", capN, (P(IP), P(s.mn), P(s.pacc), i32(s.K), i32(s.pbc_ns), i32(kstep), expo, P(s.kb)), b=256)
            s.k("ao_floor_n", capN, (P(IP), P(s.an), P(s.mqn), f(FL["KEX"] if FL else 1.0), i32(1 if FL else 0), i32(1 if (FL and FL["vout"]) else 0), f(s.al), *[f(q) for q in nr],
                                     P(s.fixd), P(s.vbc), P(s.U), f(s.tau), P(s.Ut), P(s.z0), P(s.y), P(s.sk), P(s.pm), P(s.yh), P(s.eta),
                                     P(s.kb if s.has_pbc else None), P(s.mn), f(s.mthr)), b=256)
            for it in range(s.nt):
                s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(16)), b=1)
                s.k("ao_elem_s" if s.hspec else "ao_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), f(s.cdp), i32(1 | 2 | 8 | (128 if s.hpack else 0) | (256 if s.bdsh else 0) | (512 if s.h16 else 0)), P(s.f), P(s.H), i32(capQ), P(s.B),
                                      P(s.cnt if (kstep == s.K - 1 and it == s.nt - 1) else None)))
                s.k("rt_grad" + ("_c" if s.compact else ""), capN, (P(IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
                if s.warm and it == 0 and kstep > 0 and s.nl <= 1:                                      # PCG 첫 추정 = 앞 걸음 첫 뉴턴의 δ (프레임 첫 걸음은 0)
                    s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(1)), b=1)
                    s.k("ao_tvec", capN, (P(IP), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.d0), P(s.td)), b=256)
                    s.zf(s.KTd, 9, 3)
                    s.k(s.mvname(), capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.td), P(s.KTd), i32(capQ)) + ((f(2.0 * s.mu + s.lam),) if s.h16 else ()))
                    s.k("ao_warm", capN, (P(IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.d0), P(s.KTd), P(s.Dinv), P(s.dl), P(s.r), P(s.z), P(s.p), P(s.Tp), P(s.SC)), b=256)
                if s.nl > 1:
                    if kstep == 0 and it == 0:
                        s.mg_power(0)
                    s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(1)), b=1)
                    s.mg_vc(0, s.r, s.z)
                    s.zf(s.zo, 9, 3)
                    s.k("rt_dotz", capN, (P(IP), P(s.r), P(s.z), P(s.zo), P(s.SC), i32(1)), b=256)
                    s.k("rt_pinit", capN, (P(IP), P(s.z), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.Tp)), b=256)
                for ci in range(s.cg):
                    s.zf(s.KTp, 9, 3)
                    s.k(s.mvname(), capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.Tp), P(s.KTp), i32(capQ)) + ((f(2.0 * s.mu + s.lam),) if s.h16 else ()))
                    s.k("rt_ap" + ("_c" if s.compact else ""), capN, (P(IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.KTp), P(s.Ap), P(s.SC)), b=256)
                    if s.nl > 1:
                        s.k("rt_cg1m", capN, (P(IP), P(s.SC), P(s.p), P(s.Ap), P(s.dl), P(s.r)), b=256)
                        s.mg_vc(0, s.r, s.z)
                        s.k("rt_dotz", capN, (P(IP), P(s.r), P(s.z), P(s.zo), P(s.SC), i32(0)), b=256)
                    else:
                        s.k("rt_cg1" + ("_c" if s.compact else ""), capN, (P(IP), P(s.SC), P(s.Dinv), P(s.p), P(s.Ap), P(s.dl), P(s.r), P(s.z), P(s.SC)), b=256)
                    s.k("rt_cg2" + ("_c" if s.compact else ""), capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.z), P(s.p), P(s.Tp)), b=256)
                    s.k("rt_sc_roll", 1, (P(s.SC),), b=1)
                if s.warm and it == 0:
                    s.d0.copy_(s.dl)
                s.k("rt_lsup" + ("_c" if s.compact else ""), capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(1)), b=256)
            if s.th is not None:
                s.k("ao_stepend_th", capN, (P(IP), f(1.0 / s.th), f(s.al), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.fixd), P(s.vbc), f(s.damp)), b=256)
            else:
                s.k("rt_stepend", capN, (P(IP), i32(1 if s.mid else 0), f(s.tau), f(s.al), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.fixd), P(s.vbc), f(s.damp)), b=256)
            s.Ust[kstep].copy_(s.U)
            if kstep < s.K - 1:
                s.k("ao_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.U), f(s.cdp)))
        s.k("ao_g2p", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.Ust), i32(s.K), i32(capN), P(s.vk), f(s.cdp), P(s.x), P(s.v), P(s.F), P(s.cnt)))

    def check_hspec(s, seed=0):
        """같은 상태 (프레임 끝 y, Fq) 에서 기존 판 (ao_elem 위 삼각 + ao_mv_s) 과 주값 분해 판 (ao_elem_s + ao_mv_q) 의 힘·노드 대각·무작위 곱 상대 차 (비결정 모드에서만)."""
        i32, f = np.int32, np.float32
        IP, capQ = s.IP, s.capQc * 8
        nn = int(s.IP[9]); nq = int(s.IP[12])
        g = torch.Generator(device=s.dev).manual_seed(seed)
        dv = (torch.randn(s.capN, 3, device=s.dev, generator=g) * 1e-3).contiguous()
        Hs = torch.zeros(capQ * 81, device=s.dev, dtype=torch.float32)
        out = {}
        for tag, en, mn, md in (("old", "ao_elem", "ao_mv_s", 1 | 2 | 8 | 128), ("spec", "ao_elem_s", "ao_mv_q", 1 | 2 | 8)):
            s.f.zero_(); s.B.zero_(); Hs.zero_()
            s.k(en, capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), f(s.cdp), i32(md), P(s.f), P(Hs), i32(capQ), P(s.B), P(None)))
            o = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
            s.k(mn, capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(Hs), P(dv), P(o), i32(capQ)))
            torch.cuda.synchronize()
            out[tag] = (s.f[:nn].clone(), s.B[:nn].clone(), o[:nn].clone())
        rel = lambda a, b: float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-300))
        r = dict(nq=nq, nn=nn, f=rel(out["spec"][0], out["old"][0]), B=rel(out["spec"][1], out["old"][1]), mv=rel(out["spec"][2], out["old"][2]),
                 mv_dot=float((dv[:nn].double() * out["spec"][2].double()).sum() / (dv[:nn].double() * out["old"][2].double()).sum()))
        return r

    def counts(s):
        c = super().counts()
        cc = s.cnt.cpu().tolist()
        c.update(q_tail=cc[0], q_apex=cc[1], p_rm=cc[2])
        return c


# ---------------- 적응 걸음 수 (조정자 11-02 규칙): K = ceil(v_max · frame_dt / (c · 칸)), 최소 1, v_max = 그 프레임 시작 노드 속도 최대 (실행 중 값) ----------------
def frame_vmax(s, t):
    """그래프 밖 앞 계산: 상자·칸 압축·P2G (구속 v = 0 포함) -> 노드 속도 -> 최대 |v| (호스트로 하나 읽음). 넘침이면 입자 속도 최대로 (노드 속도 ≤ 입자 속도 최대)."""
    N, IP = s.N, s.IP
    i32 = np.int32
    capN = s.capN
    if hasattr(s, "tbuf"):
        s.tbuf.fill_(float(t))
    s.k("rt_reset", 1, (P(IP),), b=1)
    s.k("rt_coord", N, (i32(N), P(s.x), P(s.Ai), P(s.Np), P(s.fx), P(IP)))
    s.k("rt_box", 1, (P(IP), i32(capN), i32(s.capC)), b=1)
    s.zf(s.occ)
    s.k("rt_mark", N, (i32(N), P(s.Np), P(s.fx), P(IP), P(s.pcd), P(s.occ)))
    s.k("rt_scan", 1024, (P(IP), P(s.occ), P(s.cidx), P(s.cb), i32(s.capQc)), b=1024)
    for t_, ci, mul in ((s.Aacc, 12, 28), (s.Cacc, 11, 30), (s.mn, 9, 1), (s.mv, 9, 3)):
        s.zf(t_, ci, mul)
    s.zf(s.badq); s.zf(s.badc)
    if getattr(s, "has_pbc", False):
        s.k("ao_vzero", N, (i32(N), P(IP), P(s.trel), P(s.tbuf), P(s.v)))
    s.k("rt_p2g", N, (i32(N), P(s.x), P(s.v), P(s.F), P(s.m), P(s.vol), P(s.Np), P(s.fx), P(s.pcd), P(s.cidx), P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc), P(s.mn), P(s.mv)))
    s.k("rt_vn", capN, (P(IP), P(s.mn), P(s.mv), P(s.vk), P(s.U)), b=256)
    ip = s.IP[[9, 13]].tolist()
    if ip[1] != 0 or ip[0] <= 0:
        return float(s.v.norm(dim=-1).max())
    return float(s.vk[:ip[0]].norm(dim=-1).max())


def set_K(s, K, h, sub_dt):
    s.K = int(K)
    s.dt = h / s.K
    s.tau = s.dt / 2 if s.mid else s.dt
    s.al = (2.0 / s.tau) if s.mid else (1.0 / s.tau)
    if getattr(s, "th", None) is not None:
        s.set_th()
    if s.floor:
        s.floor["KEX"] = s.dt / sub_dt


def adapt_step(s, t, h, cell, sub_dt, kmax, c=1.0):
    """한 프레임 (적응 걸음 수). 돌려줌: (다시 했는지, K, v_max). 그래프는 K 마다 캐시 (용량이 바뀌면 비움)."""
    import math
    if not hasattr(s, "gK"):
        s.gK, s.gcap = {}, s.capN
    vmax = frame_vmax(s, t)
    K = max(1, min(int(kmax), int(math.ceil(vmax * h / (c * cell) - 1e-9))))
    if K > getattr(s, "kalloc", s.K):                                                  # 버퍼는 실제 필요한 걸음 수까지만 키움
        s.kcap = K
        s.alloc(None)
        s.gK, s.gcap = {}, s.capN
    set_K(s, K, h, sub_dt)
    redo = s.step(t)                                                                   # 그래프는 절 AN step 의 걸음 수별 캐시 (s.graphs) 를 씀
    return redo, K, vmax


class RTAdapt(DET.DetMixin):
    """탄성 씬 (절 AN RTSolver 의 record 그대로) + 적응 걸음 수 버퍼 + 노드 압축·PCG 첫 추정 (절 AO.11 과 같은 커널).
    절 AN 코드는 고치지 않고 커널 실행 함수 k() 에서: rt_grad·rt_ap·rt_cg1·rt_cg2·rt_lsup -> 압축 판 (_c), rt_vn 뒤 노드 목록, 걸음 첫 뉴턴 rt_grad 뒤 첫 추정, rt_lsup 앞 δ 저장."""
    CN = ("rt_grad", "rt_ap", "rt_cg1", "rt_cg2", "rt_lsup")

    def __init__(s, *args, compact=False, warm=False, det=False, **kw):
        s.compact, s.warm, s.det = bool(compact), bool(warm), bool(det)
        super().__init__(*args, **kw)
        if s.compact or s.warm or s.det:
            s.mod = ART.cp.RawModule(code=_src(), options=("--std=c++14",))
            s.fn = {}
            s.ao_ready = True
            s.bind_act()
            s.graph = None

    def alloc(s, need):
        K0 = s.K
        s.K = max(K0, getattr(s, "kcap", K0))
        super().alloc(need)
        s.kalloc = s.K
        s.K = K0
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.act = torch.zeros(s.capN, device=s.dev, dtype=torch.int32)
        s.det_alloc()
        s.d0 = Z(s.capN, 3); s.td = Z(s.capN, 3); s.KTd = Z(s.capN, 3)
        if getattr(s, "ao_ready", False):
            s.bind_act()

    def bind_act(s):
        g = s.mod.get_global("AO_ACT")
        ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.act.data_ptr())
        s.det_bind()

    def k(s, name, n, args, b=128):
        if not getattr(s, "ao_ready", False):
            return super().k(name, n, args, b)
        i32, f = np.int32, np.float32
        if name == "rt_reset":
            s._ks = -1; s._it = -1
        elif name in ("rt_step0", "rt_step0g"):
            s._ks += 1; s._it = -1
        elif name == "rt_grad":
            s._it += 1
        warm = s.warm and s.nl <= 1
        if name == "rt_lsup" and warm and s._it == 0:
            s.d0.copy_(s.dl)
        nm = name + "_c" if (s.compact and name in s.CN) else name
        super().k(nm, n, args, b)
        if name == "rt_vn":
            if s.compact:                                                              # 압축 판은 빈 노드의 δ·처음 잔차를 0 으로 두지 않으므로 (상자 전체를 도는 선탐색 rt_lsy4 가 읽음) 프레임마다 비움
                s.zf(s.dl); s.zf(s.g0buf)
            super().k("ao_act_reset", 1, (P(s.IP),), 1)
            super().k("ao_act", s.capN, (P(s.IP), P(s.mn)), 256)
            if warm:
                s.zf(s.d0, 9, 3)
        elif name == "rt_grad" and warm and s._it == 0 and s._ks > 0:
            nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
            it2 = f(1.0 / (s.tau * s.tau))
            capQ = s.capQc * 8
            super().k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(1)), 1)
            super().k("ao_tvec", s.capN, (P(s.IP), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.d0), P(s.td)), 256)
            s.zf(s.KTd, 9, 3)
            super().k("rt_mv", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.td), P(s.KTd), i32(capQ)), 128)
            super().k("ao_warm", s.capN, (P(s.IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.d0), P(s.KTd), P(s.Dinv), P(s.dl), P(s.r), P(s.z), P(s.p), P(s.Tp), P(s.SC)), 256)
