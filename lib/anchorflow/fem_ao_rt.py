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
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) atomicAdd(&acc[(size_t)nid*KS + ks], ww); }
    }
}
__global__ void ao_pbc_k(const int* IP, const float* mn, const float* acc, int K, int S, int k, float expo, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], lk = 0.f; int KS = K*S;
    if (m > 0.f) for (int ss=0; ss<S; ss++) { float al = fminf(acc[(size_t)n*KS + k*S + ss]/m, 1.f); lk += log1pf(-al); }
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
        for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
        }
        if (mode & 8) {
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
'''


def _src():
    s = ART.SRC.rstrip()
    i = s.rfind("}")
    return s[:i] + AO_SRC + "\n}\n"


class AORTSolver(ART.RTSolver):
    """모래 융합 판. 추가 인자: cdp (PG DP c), trel (입자별 구속 끝 시각 [N] 또는 None), pbc_ns (표본 수), dt_gt (정답 substep_dt), mthr (거의 빈 노드 질량 문턱, 절대값)."""

    def __init__(s, *args, cdp=1.0, trel=None, pbc_ns=4, dt_gt=2e-5, mthr=0.0, **kw):
        s.cdp, s.pbc_ns, s.dt_gt, s.mthr = float(cdp), int(pbc_ns), float(dt_gt), float(mthr)
        s.has_pbc = trel is not None
        super().__init__(*args, **kw)
        s.mod = ART.cp.RawModule(code=_src(), options=("--std=c++14",))
        s.fn = {}
        s.tbuf = torch.zeros(1, device=s.dev, dtype=torch.float32)
        s.cnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.trel = (trel[s.perm].float().contiguous() if s.has_pbc else None)
        s.graph = None

    def alloc(s, need):
        super().alloc(need)
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.Ust = Z(s.K, s.capN, 3)
        if getattr(s, "has_pbc", False):
            s.pacc = Z(s.capN * s.K * s.pbc_ns)
            s.kb = Z(s.capN)
        s.graph = None

    def set_bc(s, t):
        super().set_bc(t)
        s.tbuf.fill_(float(t))

    def record(s, t_frame):
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
                s.k("rt_floor_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.xq0), P(s.mq), P(s.y), P(s.U), i32(1 if s.mid else 0), *[f(q) for q in FL["PT"]], *[f(q) for q in nr],
                                         f(FL["PN"]), f(FL["DXG"]), P(s.an)))
            if s.has_pbc:
                s.k("ao_pbc_k", capN, (P(IP), P(s.mn), P(s.pacc), i32(s.K), i32(s.pbc_ns), i32(kstep), expo, P(s.kb)), b=256)
            s.k("ao_floor_n", capN, (P(IP), P(s.an), P(s.mqn), f(FL["KEX"] if FL else 1.0), i32(1 if FL else 0), i32(1 if (FL and FL["vout"]) else 0), f(s.al), *[f(q) for q in nr],
                                     P(s.fixd), P(s.vbc), P(s.U), f(s.tau), P(s.Ut), P(s.z0), P(s.y), P(s.sk), P(s.pm), P(s.yh), P(s.eta),
                                     P(s.kb if s.has_pbc else None), P(s.mn), f(s.mthr)), b=256)
            for it in range(s.nt):
                s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
                s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(16)), b=1)
                s.k("ao_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), f(s.cdp), i32(1 | 2 | 8), P(s.f), P(s.H), i32(capQ), P(s.B),
                                      P(s.cnt if (kstep == s.K - 1 and it == s.nt - 1) else None)))
                s.k("rt_grad", capN, (P(IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
                for ci in range(s.cg):
                    s.zf(s.KTp, 9, 3)
                    s.k("rt_mv", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.Tp), P(s.KTp), i32(capQ)))
                    s.k("rt_ap", capN, (P(IP), P(s.mn), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.p), P(s.KTp), P(s.Ap), P(s.SC)), b=256)
                    s.k("rt_cg1", capN, (P(IP), P(s.SC), P(s.Dinv), P(s.p), P(s.Ap), P(s.dl), P(s.r), P(s.z), P(s.SC)), b=256)
                    s.k("rt_cg2", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.z), P(s.p), P(s.Tp)), b=256)
                    s.k("rt_sc_roll", 1, (P(s.SC),), b=1)
                s.k("rt_lsup", capN, (P(IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(1)), b=256)
            s.k("rt_stepend", capN, (P(IP), i32(1 if s.mid else 0), f(s.tau), f(s.al), P(s.y), P(s.z0), P(s.U), P(s.vk), P(s.fixd), P(s.vbc), f(s.damp)), b=256)
            s.Ust[kstep].copy_(s.U)
            if kstep < s.K - 1:
                s.k("ao_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.U), f(s.cdp)))
        s.k("ao_g2p", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.Ust), i32(s.K), i32(capN), P(s.vk), f(s.cdp), P(s.x), P(s.v), P(s.F), P(s.cnt)))

    def counts(s):
        c = super().counts()
        cc = s.cnt.cpu().tolist()
        c.update(q_tail=cc[0], q_apex=cc[1], p_rm=cc[2])
        return c
