"""절 AO: 암시적 B-스플라인 하이브리드 판 (절 AN) 의 모래 재질 (PG Drucker-Prager) 커널. 절 AN 커널 원본 (fem_an_cuda._K: ssvd·jac3·seg8) 에 덧붙여 따로 컴파일한다
(절 AN 파일은 고치지 않음). 헤시안 저장 형식은 절 AN 과 같아 an_mv·an_bd·an_mv32·an_bd32 를 그대로 쓴다.

정답 식 (PG mpm_utils.py sand_return_mapping + kirchoff_stress_drucker_prager, 클러스터 사본 /home/dkta/work/PhysGaussian):
  시험 X -> SVD -> ε_i = log max(|σ_i|, 1e-14), tr = Σε, e = ε − tr/3, ρ = |e|, δγ = ρ + c tr, c = (3λ + 2μ) α / (2μ), α = √(2/3) 2 sinφ / (3 − sinφ)
  δγ ≤ 0: 탄성 (F_e = X), δγ > 0 이고 tr > 0: 꼭짓점 (F_e = U Vᵀ, 응력 0), 그 밖: ε̂ = ε − e δγ/ρ (꼬리 반경 ρ̂ = −c tr, 대각합 유지).
  응력 (Hencky StVK): 키르히호프 주값 τ̂_i = 2μ ε̂_i + λ tr̂ (정답은 투영된 F_e 로 τ̂ 를 만들어 현재 배치 ∇w 에 곱한다).
  우리 판은 프레임 시작 기준 총 라그랑주라 같은 힘이 되는 시험 X 위 제 1 PK 를 쓴다: P = τ̂ X⁻ᵀ = U diag(τ̂_i / σ_i) Vᵀ (σ 부호 있음).
접선: 일관 접선 dτ̂/dε 는 비결합 흐름 (부피 소성 흐름 없음) 때문에 비대칭 항 −2μ c ê 1ᵀ 하나를 가진다. 이 항만 뺀
  C = K₃ 1 1ᵀ + 2μ s (I − 1 1ᵀ/3 − ê êᵀ) (s = ρ̂/ρ, K₃ = λ + 2μ/3; 탄성은 2μ I + λ 1 1ᵀ, 꼭짓점은 0) 은 대칭 반양정치라 PCG 에 쓸 수 있다
  (대칭화 (C + Cᵀ)/2 는 (1, ê) 2x2 블록이 부정부호가 되어 투영이 접선을 크게 바꾼다). 주값 공간 D_ij = C_ij / (σ_i σ_j) − δ_ij τ̂_i / σ_i²,
  비틀림 (p_i + p_j)/(σ_i + σ_j), 뒤집기 (p_i − p_j)/(σ_i − σ_j) (같은 σ 면 D_ii − D_ij) -- 절 AN 과 같은 해석 고유계, 음의 고유값은 0.
  퍼텐셜이 없으므로 (비결합) 선탐색은 에너지 대신 잔차 노름 |g|² 로 한다 (fem_ao_sim.py).
커널:
  ao_elem : (mode 1) 힘 f_n += V_q P_q G_qn, (mode 2) 헤시안 H_q (mode 8 이면 float32 H32), pst 가 있으면 점마다 상태 0 탄성 / 1 꼬리 투영 / 2 꼭짓점.
  ao_rm   : 3x3 묶음에 정답 return mapping (G2P 입자 상태 갱신, 걸음 끝 가우스 점 상태 커밋) -- 탄성이면 F 그대로 (정답과 같음).
"""
import cupy as cp
import numpy as np

from . import fem_an_cuda as AN
from . import fem_rt_cuda as RTC

_SAND = r'''
__device__ __forceinline__ int ao_dp(const double* S, double cdp, double* sgn, double* sg, double* ep, double* e, double& tr, double& rho, double& s) {
    for (int i=0;i<3;i++) { sgn[i] = (S[i] < 0.0) ? -1.0 : 1.0; sg[i] = fmax(fabs(S[i]), 1e-14); ep[i] = log(sg[i]); }
    tr = ep[0] + ep[1] + ep[2];
    for (int i=0;i<3;i++) e[i] = ep[i] - tr/3.0;
    rho = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    double dg = rho + cdp*tr;
    s = 1.0;
    if (!(dg > 0.0)) return 0;
    if (tr > 0.0) { s = 0.0; return 2; }
    s = -cdp*tr/rho;
    return 1;
}
__global__ void ao_elem(int nq, const int* qn, const double* G, const double* Fq, const double* Vq, const double* y,
                        double mu, double lam, double cdp, int mode, double* f, double* H, float* H32, unsigned char* pst) {
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
    double sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dp(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (pst && on) pst[qq] = (unsigned char)st;
    double K3 = lam + 2.0*mu/3.0;
    double tau[3], ss[3], p[3];
    for (int i=0;i<3;i++) {
        tau[i] = (st == 2) ? 0.0 : ((st == 1) ? (K3*tr + 2.0*mu*s*e[i]) : (2.0*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i];
        p[i] = tau[i]/ss[i];
    }
    double vq = on ? Vq[qq] : 0.0;
    if (mode & 1) {
        double P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
        for (int n=0;n<27;n++) {
            const double* g = G + ((size_t)qq*27+n)*3;
            double sv[3];
            for (int a=0;a<3;a++) sv[a] = vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2]);
            for (int a=0;a<3;a++) sv[a] = seg8(sv[a]);
            if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&f[id*3+a], sv[a]); }
        }
    }
    if ((mode & 2) && on) {
        double C[3][3], D[3][3], Ev[3][3];
        double eh[3] = {0.0, 0.0, 0.0};
        if (st == 1 && rho > 0.0) for (int i=0;i<3;i++) eh[i] = e[i]/rho;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            double c;
            if (st == 0) c = ((i==j) ? 2.0*mu : 0.0) + lam;
            else if (st == 1) c = K3 + 2.0*mu*s*(((i==j) ? 1.0 : 0.0) - 1.0/3.0 - eh[i]*eh[j]);
            else c = 0.0;
            C[i][j] = c;
        }
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = C[i][j]/(ss[i]*ss[j]) - ((i==j) ? p[i]/ss[i] : 0.0);
        double Dk[3][3]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dk[i][j] = D[i][j];
        jac3(D, Ev);
        double h[81];
        for (int k=0;k<81;k++) h[k] = 0.0;
        for (int m=0;m<3;m++) {
            double dm = fmax(D[m][m], 0.0);
            if (!(dm > 0.0) || !isfinite(dm)) continue;
            double ee[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) ee[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*ee[r]*ee[c];
        }
        const int PI[3] = {0, 0, 1}, PJ[3] = {1, 2, 2};
        const double r2 = 0.70710678118654752440;
        for (int pp=0; pp<3; pp++) {
            int i = PI[pp], j = PJ[pp];
            double den = ss[i] + ss[j];
            double lt = (fabs(den) > 1e-12) ? (p[i] + p[j])/den : 0.0;
            double dd = ss[i] - ss[j];
            double lf = (fabs(dd) > 1e-9*(fabs(ss[i]) + fabs(ss[j]))) ? (p[i] - p[j])/dd : (Dk[i][i] - Dk[i][j]);
            lt = isfinite(lt) ? fmax(lt, 0.0) : 0.0;
            lf = isfinite(lf) ? fmax(lf, 0.0) : 0.0;
            double et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                double x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i];
                et[a*3+b] = r2*(x1 - x2); ef[a*3+b] = r2*(x1 + x2);
            }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c];
        }
        if (mode & 8) { for (int k=0;k<81;k++) H32[(size_t)qq*81+k] = (float)h[k]; }
        else { for (int k=0;k<81;k++) H[(size_t)qq*81+k] = h[k]; }
    }
}
// 정답 return mapping (3x3 묶음): 탄성이면 그대로, 꼭짓점이면 U diag(부호) Vᵀ, 꼬리 투영이면 U diag(부호 exp ε̂) Vᵀ
__global__ void ao_rm(int N, const double* Fin, double cdp, double* Fout, unsigned char* pst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    const double* X = Fin + (size_t)p*9;
    double U[9], S[3], V[9];
    ssvd(X, U, S, V);
    double sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dp(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (pst) pst[p] = (unsigned char)st;
    if (st == 0) { for (int k=0;k<9;k++) Fout[(size_t)p*9+k] = X[k]; return; }
    double sn[3];
    for (int i=0;i<3;i++) sn[i] = (st == 2) ? sgn[i] : sgn[i]*exp(tr/3.0 + s*e[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Fout[(size_t)p*9+a*3+b] = U[a*3]*sn[0]*V[b*3] + U[a*3+1]*sn[1]*V[b*3+1] + U[a*3+2]*sn[2]*V[b*3+2];
}
'''
_MOD = [None]
NAMES = ("ao_elem", "ao_rm")


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def mod():
    if _MOD[0] is None:
        src = AN._K.rstrip()
        i = src.rfind("}")                                                            # extern "C" 닫는 괄호 앞에 끼운다
        _MOD[0] = cp.RawModule(code=src[:i] + _SAND + "\n}\n", options=("--std=c++14",))
        for nm in NAMES:
            _MOD[0].get_function(nm)
    return _MOD[0]


def launch(name, n, args, b=128):
    m = mod()
    with RTC._Stream():
        m.get_function(name)(((int(n) + b - 1) // b,), (b,), args)


def cdp_of(cfg, mu, lam):
    """정답 PG 기본값 그대로 (friction_angle 없으면 25°, sin 은 PG 의 3.14159265 로): c = (3λ + 2μ) α / (2μ)."""
    import math
    phi = float(cfg.get("friction_angle", 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    return (3.0 * lam + 2.0 * mu) / (2.0 * mu) * alpha, alpha, phi
