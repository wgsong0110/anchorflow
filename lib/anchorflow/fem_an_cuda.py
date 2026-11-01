"""절 AN: 암시적 B-스플라인 하이브리드 판의 CUDA 커널 (cupy RawModule, float64).
적분점 q = 점유 칸 (B-스플라인 칸 [base+½, base+3/2]³) 마다 2x2x2 가우스 점 8 개, 번호 q = 칸 순번 x 8 + 팔분 번호 (같은 칸 8 점이 연속 -> 8 레인 구간 합 뒤 머리 레인만 원자 합).
점마다 27 노드 (압축 번호 qn[q, 27]), G_qn = F_qᵀ ∇w_qn (프레임 시작 고정) 이라 X_q(y) = F_q + Σ_n y_n ⊗ G_qn.
- an_elem: X -> 부호 있는 SVD -> 표준 FCR ψ = μ Σ(σ−1)² + λ/2 (J−1)², 에너지 (mode&4), 기울기 f_n += V_q P_q G_qn (mode&1), 투영 헤시안 H_q 9x9 (mode&2; 해석 고유계 -- 대각 3x3 블록 고유분해 + 비틀림·뒤집기 쌍, 음의 고유값은 0 으로 = 투영 뉴턴).
- an_mv: 헤시안 곱 out_n += V_q (H_q δX_q) G_qn, δX_q = Σ δy_n ⊗ G_qn (행렬 없는).
- an_bd: 노드 3x3 대각 블록 Σ_q V_q Σ_bd H[(ab),(cd)] G_b G_d (블록 야코비 전처리).
- an_rsp / an_rsf: 절 AI A1 회전·신장 분리 평균의 입자 준비 (R_p = U Vᵀ, log S_p) 와 점 마무리 (R̄ = polar(Σ w R), S̄ = exp(Σ w log S / Σ w))."""
import cupy as cp
import numpy as np

from . import fem_rt_cuda as RTC

_K = r'''
extern "C" {
__device__ void jac3(double a[3][3], double v[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) v[i][j] = (i==j) ? 1.0 : 0.0;
    for (int sw=0; sw<40; sw++) {
        double off = a[0][1]*a[0][1] + a[0][2]*a[0][2] + a[1][2]*a[1][2];
        double dg = a[0][0]*a[0][0] + a[1][1]*a[1][1] + a[2][2]*a[2][2];
        if (!(off > 1e-34*dg)) break;
        for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
            if (a[p][q] == 0.0) continue;
            double th = 0.5*(a[q][q]-a[p][p])/a[p][q];
            double t = (th >= 0.0 ? 1.0 : -1.0)/(fabs(th) + sqrt(th*th + 1.0));
            double c = 1.0/sqrt(t*t + 1.0), s = t*c;
            for (int k=0;k<3;k++) { double x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - s*x2; a[k][q] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { double x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - s*x2; a[q][k] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { double x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - s*x2; v[k][q] = s*x1 + c*x2; }
        }
    }
}
// 부호 있는 SVD: X = U diag(S) Vᵀ, U·V 회전 (det +1), S0 >= S1 >= |S2|, S2 부호 = det X 부호. 행 우선 [a*3+b].
__device__ void ssvd(const double* X, double* U, double* S, double* V) {
    double A[3][3], v[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { double t = 0.0; for (int k=0;k<3;k++) t += X[k*3+i]*X[k*3+j]; A[i][j] = t; }
    jac3(A, v);
    int o[3] = {0,1,2};
    double e[3] = {A[0][0], A[1][1], A[2][2]};
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    if (e[o[1]] < e[o[2]]) { int t = o[1]; o[1] = o[2]; o[2] = t; }
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    for (int i=0;i<3;i++) for (int c=0;c<3;c++) V[i*3+c] = v[i][o[c]];
    double dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.0) for (int i=0;i<3;i++) V[i*3+2] = -V[i*3+2];
    double b[3][3];
    for (int c=0;c<3;c++) for (int a=0;a<3;a++) { double t = 0.0; for (int k=0;k<3;k++) t += X[a*3+k]*V[k*3+c]; b[c][a] = t; }
    double u1[3], u2[3], u3[3];
    double n1 = sqrt(b[0][0]*b[0][0] + b[0][1]*b[0][1] + b[0][2]*b[0][2]);
    if (n1 > 0.0) for (int a=0;a<3;a++) u1[a] = b[0][a]/n1; else { u1[0] = 1.0; u1[1] = 0.0; u1[2] = 0.0; }
    double d12 = u1[0]*b[1][0] + u1[1]*b[1][1] + u1[2]*b[1][2];
    for (int a=0;a<3;a++) u2[a] = b[1][a] - d12*u1[a];
    double n2 = sqrt(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
    if (n2 > 1e-14*n1 && n2 > 0.0) { for (int a=0;a<3;a++) u2[a] /= n2; }
    else {                                                                      // 퇴화 (σ2 ≈ 0): u1 에 수직인 아무 단위 벡터
        int ax = 0; if (fabs(u1[1]) < fabs(u1[ax])) ax = 1; if (fabs(u1[2]) < fabs(u1[ax])) ax = 2;
        double ea[3] = {0.0, 0.0, 0.0}; ea[ax] = 1.0;
        u2[0] = u1[1]*ea[2]-u1[2]*ea[1]; u2[1] = u1[2]*ea[0]-u1[0]*ea[2]; u2[2] = u1[0]*ea[1]-u1[1]*ea[0];
        double nn = sqrt(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]); for (int a=0;a<3;a++) u2[a] /= nn;
    }
    u3[0] = u1[1]*u2[2]-u1[2]*u2[1]; u3[1] = u1[2]*u2[0]-u1[0]*u2[2]; u3[2] = u1[0]*u2[1]-u1[1]*u2[0];
    for (int a=0;a<3;a++) { U[a*3] = u1[a]; U[a*3+1] = u2[a]; U[a*3+2] = u3[a]; }
    S[0] = u1[0]*b[0][0] + u1[1]*b[0][1] + u1[2]*b[0][2];
    S[1] = u2[0]*b[1][0] + u2[1]*b[1][1] + u2[2]*b[1][2];
    S[2] = u3[0]*b[2][0] + u3[1]*b[2][1] + u3[2]*b[2][2];
}
__device__ __forceinline__ double seg8(double s) {                              // 같은 칸 8 레인 합 (모든 레인이 불러야 함)
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    return s;
}
// mode: 1 기울기, 2 헤시안, 4 에너지
__global__ void an_elem(int nq, const int* qn, const double* G, const double* Fq, const double* Vq, const double* y,
                        double mu, double lam, int mode, double* E, double* f, double* H) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double X[9];
    for (int k=0;k<9;k++) X[k] = Fq[(size_t)qq*9+k];
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n];
        const double* g = G + ((size_t)qq*27+n)*3;
        for (int a=0;a<3;a++) { double ya = y[id*3+a]; for (int b=0;b<3;b++) X[a*3+b] += ya*g[b]; }
    }
    double U[9], S[3], V[9];
    ssvd(X, U, S, V);
    double J = S[0]*S[1]*S[2];
    double gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    double vq = on ? Vq[qq] : 0.0;
    if (mode & 4) { if (on) E[qq] = vq*(mu*((S[0]-1.0)*(S[0]-1.0) + (S[1]-1.0)*(S[1]-1.0) + (S[2]-1.0)*(S[2]-1.0)) + 0.5*lam*(J-1.0)*(J-1.0)); }
    if (mode & 1) {
        double ph[3]; for (int i=0;i<3;i++) ph[i] = 2.0*mu*(S[i]-1.0) + lam*(J-1.0)*gJ[i];
        double P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*ph[0]*V[b*3] + U[a*3+1]*ph[1]*V[b*3+1] + U[a*3+2]*ph[2]*V[b*3+2];
        for (int n=0;n<27;n++) {
            const double* g = G + ((size_t)qq*27+n)*3;
            double s[3];
            for (int a=0;a<3;a++) s[a] = vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2]);
            for (int a=0;a<3;a++) s[a] = seg8(s[a]);
            if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&f[id*3+a], s[a]); }
        }
    }
    if ((mode & 2) && on) {
        double D[3][3], Ev[3][3];
        double Sg[3][3] = {{0.0, S[2], S[1]}, {S[2], 0.0, S[0]}, {S[1], S[0], 0.0}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.0*mu : 0.0) + lam*gJ[i]*gJ[j] + lam*(J-1.0)*Sg[i][j];
        jac3(D, Ev);
        double h[81];
        for (int k=0;k<81;k++) h[k] = 0.0;
        for (int m=0;m<3;m++) {
            double dm = fmax(D[m][m], 0.0);
            if (!(dm > 0.0)) continue;
            double e[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c];
        }
        const int PI[3] = {0, 0, 1}, PJ[3] = {1, 2, 2}, PK[3] = {2, 1, 0};
        const double r2 = 0.70710678118654752440;
        for (int pp=0; pp<3; pp++) {
            int i = PI[pp], j = PJ[pp], k = PK[pp];
            double lt = 2.0*mu*(1.0 - 2.0/(S[i]+S[j])) + lam*(J-1.0)*S[k];
            double lf = 2.0*mu - lam*(J-1.0)*S[k];
            lt = isfinite(lt) ? fmax(lt, 0.0) : 0.0;
            lf = fmax(lf, 0.0);
            double et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                double x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i];
                et[a*3+b] = r2*(x1 - x2); ef[a*3+b] = r2*(x1 + x2);
            }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c];
        }
        for (int k=0;k<81;k++) H[(size_t)qq*81+k] = h[k];
    }
}
__global__ void an_mv(int nq, const int* qn, const double* G, const double* H, const double* Vq, const double* dy, double* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n];
        const double* g = G + ((size_t)qq*27+n)*3;
        for (int a=0;a<3;a++) { double ya = dy[id*3+a]; for (int b=0;b<3;b++) dX[a*3+b] += ya*g[b]; }
    }
    double vq = on ? Vq[qq] : 0.0;
    double dP[9];
    const double* h = H + (size_t)qq*81;
    for (int r=0;r<9;r++) { double t = 0.0; for (int c=0;c<9;c++) t += h[r*9+c]*dX[c]; dP[r] = vq*t; }
    for (int n=0;n<27;n++) {
        const double* g = G + ((size_t)qq*27+n)*3;
        double s[3];
        for (int a=0;a<3;a++) s[a] = dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2];
        for (int a=0;a<3;a++) s[a] = seg8(s[a]);
        if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&out[id*3+a], s[a]); }
    }
}
__global__ void an_bd(int nq, const int* qn, const double* G, const double* H, const double* Vq, double* B) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double vq = on ? Vq[qq] : 0.0;
    const double* h = H + (size_t)qq*81;
    for (int n=0;n<27;n++) {
        const double* g = G + ((size_t)qq*27+n)*3;
        double m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) {
            double t = 0.0;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t;
        }
        for (int k=0;k<9;k++) m[k] = seg8(m[k]);
        if (on && head) { int id = qn[(size_t)qq*27+n]; for (int k=0;k<9;k++) atomicAdd(&B[id*9+k], m[k]); }
    }
}
// 입자 준비: R = U Vᵀ, LS = V diag(log S) Vᵀ, neg = det F <= 0
__global__ void an_rsp(int N, const double* F, double* R, double* LS, unsigned char* neg) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double U[9], S[3], V[9];
    ssvd(F + (size_t)p*9, U, S, V);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[(size_t)p*9+a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    bool ng = !(S[2] > 0.0);
    neg[p] = ng ? 1 : 0;
    double l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.0 : log(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) LS[(size_t)p*9+a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
}
// 점 마무리: R̄ = polar(sR) (부호 있는 SVD 의 U Vᵀ = 가장 가까운 회전), S̄ = exp(sym(sL) / sw), RS = R̄ S̄; 비유한이면 nf
__global__ void an_rsf(int nq, const double* sR, const double* sL, const double* sw, double* RS, unsigned char* nf) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nq) return;
    double U[9], S[3], V[9];
    ssvd(sR + (size_t)q*9, U, S, V);
    double Rb[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Rb[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    double w = sw[q];
    double L[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) L[i][j] = 0.5*(sL[(size_t)q*9+i*3+j] + sL[(size_t)q*9+j*3+i])/w;
    jac3(L, Ev);
    double ex[3]; for (int i=0;i<3;i++) ex[i] = exp(L[i][i]);
    double Sb[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Sb[a*3+b] = Ev[a][0]*ex[0]*Ev[b][0] + Ev[a][1]*ex[1]*Ev[b][1] + Ev[a][2]*ex[2]*Ev[b][2];
    bool bad = false;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        double t = Rb[a*3]*Sb[b] + Rb[a*3+1]*Sb[3+b] + Rb[a*3+2]*Sb[6+b];
        RS[(size_t)q*9+a*3+b] = t;
        if (!isfinite(t)) bad = true;
    }
    nf[q] = bad ? 1 : 0;
}
}
'''
_MOD = [None]
NAMES = ("an_elem", "an_mv", "an_bd", "an_rsp", "an_rsf")


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def mod():
    if _MOD[0] is None:
        _MOD[0] = cp.RawModule(code=_K, options=("--std=c++14",))
        for nm in NAMES:
            _MOD[0].get_function(nm)
    return _MOD[0]


def launch(name, n, args, b=128):
    m = mod()
    with RTC._Stream():
        m.get_function(name)(((int(n) + b - 1) // b,), (b,), args)
