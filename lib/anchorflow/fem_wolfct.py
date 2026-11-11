"""절 CM108 묶음 wolfct (사용자 11-11 지시): 모래 (PG Drucker-Prager) 반환 사상의 일관 접선 (Simo & Hughes algorithmic tangent) 을 뉴턴 헤시안에.

지금 판 (ao_elem_s, --hspec): 힘은 반환 사상 뒤 τ̂ (P = U diag(τ̂/σ) Vᵀ), 헤시안은 시험 X 의 Hencky 탄성 접선 (2μI + λ11ᵀ 와 시험 τ 의 기하 항, 투영) = 초기 강성법.
--ct 1 (새 판): 주값 공간 접선을 dτ̂/dε 로 바꿈 -- 탄성 2μδ + λ, 꼬리 K₃ − 2μc ê_i + 2μs (δ_ij − 1/3 − ê_i ê_j) (s = −c tr/ρ, ê = dev ε/ρ), 꼭짓점 0;
  비결합 흐름이라 비대칭 (−2μc ê 1ᵀ) -> 대칭 몫 ½(C + Cᵀ) 을 D_ij = C_ij/(σ_iσ_j) − δ_ij τ̂_i/σ_i² 에 넣고 지금과 같은 고유값 투영 (jac3f, 음수 0) -- PSD 투영.
  비틀림·뒤집기 쌍 계수도 반환 뒤 τ̂ 로 (p_i ± p_j)/(σ_i ± σ_j) (음수 0, 지금과 같은 클램프). 힘·커밋·G2P 는 그대로 (헤시안만 바뀜).
  watermelon (NACC) 은 aq_elem 경로라 이 바꿈을 안 받음 (소스에 ao_elem_s 가 있어도 모래만 부름).
--ctfd 1 (진단, 결과 불변): 다시 풀기 판에서 뉴턴 반복마다 ao_elem_s 바로 뒤 같은 그래프·스트림에서 가우스 점마다
  X = F_q + (Σ y⊗∇w) F_q 와 무작위 노드 방향 dX (점마다 ‖dX‖ = h‖X‖, h = --cth) 로 P 의 중앙 차분 FD = (P(X+dX) − P(X−dX))/2 를
  세 접선 곱과 대조: T0 = 지금 판 (탄성·투영), T1 = 정확한 야코비안 (일관 접선 비대칭·투영 없음), T2 = 새 판 (대칭 몫 + PSD).
  상태 (탄성 0 · 꼬리 1 · 꼭짓점 2) 별 합: 점 수, Σ‖FD‖², Σ‖Ti − FD‖², Σ‖T0‖², Σ‖T2‖², X±dX 가 상태를 넘는 점 수 (꺾인 점, 제외). 기록 cw.jsonl 반복마다 'fd'.
출력 ctinfo.json (소스 바꿈 수·훅 수·오류).
"""
import json
import os

import torch

ST = dict(ct=0, fd=0, h=1e-3, out=".", n_src_ct=0, n_src_fd=0, n_fd_launch=0, n_alloc=0, err=None)

OLD1 = ("{ float D[3][3], Ev[3][3], Dk[3][3];\n"
        "      for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }\n"
        "      jac3f(D, Ev);")
NEW1 = ("{ float D[3][3], Ev[3][3], Dk[3][3];\n"
        "      { float Ct_[3][3]; ct_tan(st, mu, lam, cdp, e, rho, s, Ct_);   // 절 CM108 일관 접선 (대칭 몫)\n"
        "        for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = 0.5f*(Ct_[i][j] + Ct_[j][i])/(ss[i]*ss[j]) - ((i==j) ? p[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; } }\n"
        "      jac3f(D, Ev);")
OLD2 = ("ao_pairs(pp, i, j);\n"
        "          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;\n"
        "          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);")
NEW2 = ("ao_pairs(pp, i, j);\n"
        "          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (p[i] + p[j])/den : 0.f;\n"
        "          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (p[i] - p[j])/dd : (Dk[i][i] - Dk[i][j]);")
ANCHOR = "__global__ void __launch_bounds__(128) ao_elem_s("

TAN_SRC = r'''
// 절 CM108: PG Drucker-Prager 반환 사상 τ̂(ε) 의 일관 접선 dτ̂_i/dε_j (ao_dpf 와 같은 정의; st 0 탄성 · 1 꼬리 · 2 꼭짓점)
__device__ __forceinline__ void ct_tan(int st, float mu, float lam, float cdp, const float* e, float rho, float s, float Ct[3][3]) {
    float K3 = lam + 2.f*mu/3.f;
    float eh[3]; for (int i=0;i<3;i++) eh[i] = (rho > 0.f) ? e[i]/rho : 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        float d = (i==j) ? 1.f : 0.f;
        Ct[i][j] = (st == 0) ? (2.f*mu*d + lam) : ((st == 2) ? 0.f : (K3 - 2.f*mu*cdp*eh[i] + 2.f*mu*s*(d - 1.f/3.f - eh[i]*eh[j])));
    }
}
'''

FD_SRC = r'''
// 절 CM108 진단 (결과 불변): P(X) = U diag(τ̂/σ) Vᵀ (ao_elem_s 와 같은 식)
__device__ int ct_P(const float* X, float mu, float lam, float cdp, float* Pm) {
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float K3 = lam + 2.f*mu/3.f, p[3];
    for (int i=0;i<3;i++) { float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr)); p[i] = tau/(sgn[i]*sg[i]); }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
    return st;
}
// 접선 곱 dP = U P̂ Vᵀ, P̂ = 주값 블록 D (var 0: 탄성 시험 τ + 투영, 1: 일관 비대칭 그대로, 2: 일관 대칭 몫 + 투영) · 쌍 계수 (0: 시험 τ, 1·2: τ̂; 0·2 음수 0)
__device__ void ct_act(int var, const float* U, const float* V, const float* ss, const float* p, const float* pe, int st, const float* e, float rho, float s,
                       float mu, float lam, float cdp, const float* dX, float* dP) {
    float D[3][3], Dk[3][3];
    const float* q = (var == 0) ? pe : p;
    if (var == 0) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); }
    else { float Ct[3][3]; ct_tan(st, mu, lam, cdp, e, rho, s, Ct);
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((var == 2) ? 0.5f*(Ct[i][j] + Ct[j][i]) : Ct[i][j])/(ss[i]*ss[j]) - ((i==j) ? p[i]/ss[i] : 0.f); }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dk[i][j] = D[i][j];
    float Dp[3][3];
    if (var == 1) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dp[i][j] = D[i][j]; }
    else { float Ev[3][3]; jac3f(D, Ev); float dm[3]; for (int m=0;m<3;m++) { float d_ = fmaxf(D[m][m], 0.f); dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dp[i][j] = dm[0]*Ev[i][0]*Ev[j][0] + dm[1]*Ev[i][1]*Ev[j][1] + dm[2]*Ev[i][2]*Ev[j][2]; }
    float Xh[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) { float t = 0.f; for (int a=0;a<3;a++) for (int b=0;b<3;b++) t += U[a*3+i]*dX[a*3+b]*V[b*3+j]; Xh[i*3+j] = t; }
    float Ph[9];
    for (int i=0;i<3;i++) Ph[i*3+i] = Dp[i][0]*Xh[0] + Dp[i][1]*Xh[4] + Dp[i][2]*Xh[8];
    for (int pp=0; pp<3; pp++) { int i = (pp == 2) ? 1 : 0, j = (pp == 0) ? 1 : 2;
        float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (q[i] + q[j])/den : 0.f;
        float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (q[i] - q[j])/dd : (Dk[i][i] - Dk[i][j]);
        if (var != 1) { lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f; }
        float c1 = 0.5f*(lt + lf), c2 = 0.5f*(lf - lt);
        Ph[i*3+j] = c1*Xh[i*3+j] + c2*Xh[j*3+i]; Ph[j*3+i] = c2*Xh[i*3+j] + c1*Xh[j*3+i]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int i=0;i<3;i++) for (int j=0;j<3;j++) t += U[a*3+i]*Ph[i*3+j]*V[b*3+j]; dP[a*3+b] = t; }
}
// 가우스 점마다 FD 대 세 접선; acc[상태*12 + k]: 0 점 수, 1 Σ‖FD‖², 2·3·4 Σ‖T0·T1·T2 − FD‖², 5 Σ‖T0‖², 6 Σ‖T2‖², 7 꺾인 점 수, 8 Σ‖T1‖²
__global__ void ct_fd(const int* IP, const int* cb, const float* BG, const float* Fq, const float* y, const float* dy, float mu, float lam, float cdp, float hrel, double* acc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float A[9] = {0,0,0,0,0,0,0,0,0}, dA[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a], da = dy[id*3+a]; for (int b=0;b<3;b++) { A[a*3+b] += ya*bg[b]; dA[a*3+b] += da*bg[b]; } } }
    float X[9], dX[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b]; dX[a*3+b] = dA[a*3]*F[b] + dA[a*3+1]*F[3+b] + dA[a*3+2]*F[6+b]; }
    float nX = 0.f, nd = 0.f; for (int k=0;k<9;k++) { nX += X[k]*X[k]; nd += dX[k]*dX[k]; }
    if (!(nd > 0.f) || !isfinite(nX)) return;
    float sc = hrel*sqrtf(nX/nd); for (int k=0;k<9;k++) dX[k] *= sc;
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float Xp[9], Xm[9], Pp[9], Pm[9];
    for (int k=0;k<9;k++) { Xp[k] = X[k] + dX[k]; Xm[k] = X[k] - dX[k]; }
    int sp = ct_P(Xp, mu, lam, cdp, Pp), sm = ct_P(Xm, mu, lam, cdp, Pm);
    double* ac = acc + st*12;
    if (sp != st || sm != st) { atomicAdd(&ac[7], 1.0); return; }
    float K3 = lam + 2.f*mu/3.f, ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) { float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i]; }
    float T0[9], T1[9], T2[9];
    ct_act(0, U, V, ss, p, pe, st, e, rho, s, mu, lam, cdp, dX, T0);
    ct_act(1, U, V, ss, p, pe, st, e, rho, s, mu, lam, cdp, dX, T1);
    ct_act(2, U, V, ss, p, pe, st, e, rho, s, mu, lam, cdp, dX, T2);
    double f2 = 0, e0 = 0, e1 = 0, e2 = 0, n0 = 0, n2 = 0, n1 = 0;
    for (int k=0;k<9;k++) { double fd = 0.5*((double)Pp[k] - (double)Pm[k]);
        f2 += fd*fd; e0 += (T0[k]-fd)*(T0[k]-fd); e1 += (T1[k]-fd)*(T1[k]-fd); e2 += (T2[k]-fd)*(T2[k]-fd); n0 += (double)T0[k]*T0[k]; n2 += (double)T2[k]*T2[k]; n1 += (double)T1[k]*T1[k]; }
    if (!isfinite(f2) || !isfinite(e0) || !isfinite(e1) || !isfinite(e2)) { atomicAdd(&ac[7], 1.0); return; }
    atomicAdd(&ac[0], 1.0); atomicAdd(&ac[1], f2); atomicAdd(&ac[2], e0); atomicAdd(&ac[3], e1); atomicAdd(&ac[4], e2); atomicAdd(&ac[5], n0); atomicAdd(&ac[6], n2); atomicAdd(&ac[8], n1);
}
'''


def _dump():
    try:
        with open(os.path.join(ST["out"], "ctinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _patch(code):
    if ANCHOR not in code:
        return code
    if ST["ct"] or ST["fd"]:
        code = code.replace(ANCHOR, TAN_SRC + ANCHOR, 1)
    if ST["ct"]:
        n1, n2 = code.count(OLD1), code.count(OLD2)
        assert n1 == 1 and n2 == 1, f"절 CM108 바꿀 자리 수 {n1} {n2}"
        code = code.replace(OLD1, NEW1).replace(OLD2, NEW2)
        ST["n_src_ct"] += 1
    if ST["fd"]:
        i_ = code.rstrip().rfind("}")                                                  # extern "C" 닫는 괄호 앞
        code = code[:i_] + FD_SRC + "\n}\n"
        ST["n_src_fd"] += 1
    _dump()
    return code


def _fd_read(s):
    """반복 칸 (걸음 k x 반복 it) 마다 상태별 합 -> dict, 읽은 뒤 0."""
    A = s.ctacc.cpu().numpy()
    s.ctacc.zero_()
    return A


def install(out, ct=0, fd=0, h=1e-3):
    from . import fem_an_rt as ART
    from . import fem_mat_rt as MRT
    ST.update(out=out, ct=int(ct), fd=int(fd), h=float(h))
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        try:
            if "code" in kw:
                kw["code"] = _patch(kw["code"])
            elif a:
                a = (_patch(a[0]),) + tuple(a[1:])
        except AssertionError as e_:
            ST["err"] = repr(e_); _dump(); raise
        return orm(*a, **kw)
    cpm.RawModule = RawModule

    if ST["fd"]:
        from . import fem_convwhy as CW
        M = MRT.MatRT
        oa, oe = M.alloc, M.mat_elem

        def alloc(s, need):
            r = oa(s, need)
            try:
                KK = 4 * max(s.K, s.Kc or 0, getattr(s, "kcap", 0))
                s.ctacc = torch.zeros(KK * int(s.nt), 36, device=s.dev, dtype=torch.float64)
                g = torch.Generator(device="cpu"); g.manual_seed(108)
                s.ctdy = torch.randn(s.y.numel(), generator=g, dtype=torch.float32).to(s.dev).reshape(s.y.shape).contiguous()
                ST["n_alloc"] += 1
            except Exception as e_:
                ST["err"] = "alloc " + repr(e_)
            return r

        def mat_elem(s, kstep, it, nst, capQ):
            r = oe(s, kstep, it, nst, capQ)
            if getattr(s, "ctacc", None) is not None and not s.aq and s.hspec:
                i = int(kstep) * int(s.nt) + int(it)
                if i < s.ctacc.shape[0]:
                    f = lambda v: __import__("numpy").float32(v)
                    s.k("ct_fd", capQ, (ART.P(s.IP), ART.P(s.cb), ART.P(s.BG), ART.P(s.Fq), ART.P(s.y), ART.P(s.ctdy), f(s.mu), f(s.lam), f(s.cdp), f(ST["h"]), ART.P(s.ctacc[i])))
                    ST["n_fd_launch"] += 1
            return r
        M.alloc, M.mat_elem = alloc, mat_elem

        prev_read = CW._read

        def _read(s, label, K):
            steps = prev_read(s, label, K)
            if getattr(s, "ctacc", None) is None:
                return steps
            A = _fd_read(s)
            nt = int(s.nt)
            for stp in steps:
                for o in stp["its"]:
                    i = stp["k"] * nt + o["it"]
                    if i >= A.shape[0]:
                        continue
                    o["fd"] = [[float(v) for v in A[i, c * 12:(c + 1) * 12]] for c in range(3)]
            return steps
        CW._read = _read
    print(f"[절 CM108] wolfct 켬 -- 일관 접선 {ST['ct']} FD 대조 {ST['fd']} (h {ST['h']})", flush=True)
    _dump()
