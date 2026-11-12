"""절 CM117 묶음 extsolve: 확장 노드를 소거한 줄어든 문제 Ã = Eᵀ A E 의 MG 전처리 -- 층 0 평활을 D̃ 척도 체비셰프 다항식으로.

배경 (절 CM98·CM94, 로컬 모형 .xstmp/model*.py): H9 의 PCG 는 이미 줄어든 문제 Ã δ = Eᵀ(−g) 를 풂 (fem_wolfh8: 잔차 접기·방향 펼치기 = Eᵀ A E).
  그 전처리 (원래 A 의 V 순환, r_o = 0) 가 Eᵀ A_OO E 를 몰라 조건수 1e5~1e6. 절 CM94 wolfh13 이 층 0 블록 대각 D̃ = diag(Ã) 블록을 넣었으나
  야코비 한 번은 D̃⁻¹Ã 의 넓은 스펙트럼 (확장 외삽 이득 7³) 을 못 누름. 모형 (3 차원 공, 차수 2) 에서 κ: H9 282 · D̃ 야코비 (모드 2) 55 ·
  D̃ 체비셰프 2 차 6.4 · 3 차 3.4 (거친 층 정확 풀이).
정의 (fem_wolfh13 모드 2 위에; 풀리는 방정식·수렴 해 그대로, 전처리만 바뀜; 대칭 V 순환 유지):
  - 층 0 앞·뒤 평활 = D̃ 척도 체비셰프 k 차 (Saad, Iterative Methods 12.3 / PETSc KSPCHEBYSHEV 와 같은 점화식):
      구간 [lo, hi] = [0.1 λ̂, 1.1 λ̂] (PETSc KSPChebyshevEstEigSet 기본값), λ̂ = wolfh13 거듭제곱법의 λmax(D̃⁻¹Ã) (W[0] = 1/λ̂).
      θ = (hi+lo)/2, δ = (hi−lo)/2, σ = θ/δ, ρ0 = 1/σ;  d = D̃⁻¹e/θ, z += d;  i ≥ 1: e = Eᵀ(r − A E z), ρi = 1/(2σ − ρ_{i−1}), d = ρi ρ_{i−1} d + (2ρi/δ) D̃⁻¹ e, z += d.
    k = 1 이면 z += D̃⁻¹e/θ (= 감쇠 야코비, wolfh13 의 ω/λ̂ 대신 1/θ = 1/(0.6 λ̂)).
  - 앞 평활은 z = 0 에서 시작 (잔차 계산 없이 첫 걸음), 뒤 평활은 거친 층 보정 뒤 z 에서 (wolfh13 이 이미 하는 잔차 e 부터).
  - 거친 층 = 원래 V 순환 (wolfh13 모드 2 그대로), 외부 노드는 늘 z_o = Σ c (T z)_j 로 펼침, D̃⁻¹_o = 0.
  모든 새 커널은 허용치 조기 종료 (AF_SKIP) 를 지킴. 호스트 동기화 없음 (λ̂ 는 장치 W[0] 에서 커널이 읽음).
훅: fem_wolfh13.install(out, 2) 뒤 RTSolver._mg_vc 를 한 번 더 감쌈 -- 확장 판 (κ 0) 의 층 0 만; 나머지는 그대로 넘김.
기록: xsinfo.json (설정·V 순환 수).
"""
import json
import os

import numpy as np

ST = dict(k=2, lo=0.1, hi=1.1, out=".", nsrc=0, nvc=0, said=False)
MARK = "// ---- 절 CM117 extsolve"

XS_SRC = r'''
// ---- 절 CM117 extsolve: D̃ 척도 체비셰프 평활 (lib/anchorflow/fem_extsolve.py)
#ifdef AF_ADAPT
#define XS_SKIP if (AF_SKIP) return;
#else
#define XS_SKIP
#endif
// i 번째 체비셰프 걸음: i = 0 -> d = D e/θ (set 이면 z = d, 아니면 z += d); i ≥ 1 -> d = ρi ρ_{i−1} d + 2ρi/δ D e, z += d. Tz = T z. 외부 노드 (base ≥ 0) 건너뜀.
__global__ void xs_cheb(const int* IP, const int* ACT, const int* base, const float* D, const double* W, float lof, float hif, int i, int set,
                        const float* e, const float* sk, const int* pm, float n0, float n1, float n2, float* z, float* d, float* Tz) {
    XS_SKIP
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    int cnt = ACT ? ACT[0] : IP[9];
    if (ii >= cnt) return;
    int n = ACT ? ACT[1+ii] : ii;
    if (base[n] >= 0) return;
    double lam = 1.0 / fmax(W[0], 1e-300);
    double hi = hif*lam, lo = lof*lam, th = 0.5*(hi + lo), de = 0.5*(hi - lo), sg = th/de;
    double a = 0.0, b = 1.0/th;
    if (i > 0) { double rp = 1.0/sg, rc = rp; for (int t=1;t<=i;t++) { rp = rc; rc = 1.0/(2.0*sg - rp); } a = rc*rp; b = 2.0*rc/de; }
    const float* Dn = D + n*9;
    float ee[3] = {e[n*3], e[n*3+1], e[n*3+2]}, zz[3];
    for (int c=0;c<3;c++) {
        float de_ = Dn[c*3]*ee[0] + Dn[c*3+1]*ee[1] + Dn[c*3+2]*ee[2];
        float dd = (i > 0 ? (float)a*d[n*3+c] : 0.f) + (float)b*de_;
        d[n*3+c] = dd;
        zz[c] = (set ? 0.f : z[n*3+c]) + dd; z[n*3+c] = zz[c];
    }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, t); for (int c=0;c<3;c++) Tz[n*3+c] = t[c];
}
'''


def add_src(src):
    if MARK in src:
        return src
    assert "Tapply(" in src, "extsolve: Tapply 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + XS_SRC + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def install(out, k=2, lo=0.1, hi=1.1):
    import torch
    from . import fem_an_rt as ART
    from . import fem_wolfh8 as W8
    from . import fem_wolfh13 as W13
    ST.update(k=int(k), lo=float(lo), hi=float(hi), out=out)
    W13.install(out, 2)                                                                # 층 0 D̃ (rt_grad 뒤) · 거듭제곱법 λ̂ (Ã 로) · 모드 2 V 순환
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
    R = ART.RTSolver
    ovc = R._mg_vc
    i32, f32 = np.int32, np.float32

    def on(s):
        return getattr(s, "_xpon", False) and W8._on(s)

    def nrf(s):
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        return [f32(q) for q in nr]

    def it2m(s):
        return f32(1.0 / (getattr(s, "mg_tau", s.tau) ** 2))

    def act(s):
        return P(s.actl) if s.ncomp else P(None)

    def expand0(s, z, Tz):
        s.k("w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(z), P(Tz)), b=256)

    def fold0(s, v):
        s.k("w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(v), P(s.w8acc)), b=256)
        s.k("w8_fold2", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nrf(s), P(s.w8acc), P(v)), b=256)

    def fold0TT(s, v):
        s.k("w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(v), P(s.w8acc)), b=256)
        s.k("xp_f2TT", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nrf(s), P(s.w8acc), P(v)), b=256)

    def cheb(s, i, set_, e, z):
        s.k("xs_cheb", s.capN, (P(s.IP), act(s), P(s.w8b), P(s.Dinv), P(s.W), f32(ST["lo"]), f32(ST["hi"]), i32(i), i32(set_), P(e),
                                P(s.sk), P(s.pm), *nrf(s), P(z), P(s.xsd), P(s.m0u)), b=256)
        expand0(s, z, s.m0u)

    def node(s, mode, rr, kt, zz, rs):
        KB = s.bKn if (getattr(s, "bar", False) and s.mgbar and not getattr(s, "_pkc", None)) else None
        s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(mode), P(s.meff), it2m(s), P(s.sk), P(s.pm), *nrf(s), P(s.Dinv), P(s.W),
                                   P(rr), P(kt), P(zz), P(rs), P(KB)), b=256)

    def resid(s, r, z):                                                                # xpT = Eᵀ (r − A E z) (m0u = T z 펼친 상태에서)
        s.mg_Az(0, z, s.m0KT, tz_ready=True)
        node(s, 2, r, s.m0KT, z, s.xpT)
        fold0(s, s.xpT)

    def smooth(s, r, z, pre):
        K = ST["k"]
        if pre:
            cheb(s, 0, 1, r, z)                                                        # r 는 접힌 PCG 잔차 (r_o = 0)
        else:
            resid(s, r, z); cheb(s, 0, 0, s.xpT, z)
        for i in range(1, K):
            resid(s, r, z); cheb(s, i, 0, s.xpT, z)

    def vc0(s, r, z):
        if getattr(s, "xsd", None) is None or s.xsd.numel() < s.capN * 3:
            s.xsd = torch.zeros(s.capN * 3, device=s.dev, dtype=torch.float32)
        smooth(s, r, z, True)
        s.mg_Az(0, z, s.m0KT, tz_ready=True); node(s, 7, r, s.m0KT, z, s.m0u)         # m0u = T (r − A z)
        fold0TT(s, s.m0u)
        Lc = s.ML[0]
        s.k(*(("rt_mg_restrict3", Lc["capn"] * 16) if s.mv2 else ("rt_mg_restrict", Lc["capn"])), (i32(0), P(s.LM), P(s.m0u), P(Lc["r"]), i32(3)), b=256)
        s.mg_vc(1, Lc["r"], Lc["z"])                                                   # 거친 층 = 원래 V 순환 (모드 2)
        s.k("rt_mg_prolong", s.capN, (i32(0), P(s.LM), P(Lc["z"]), P(s.m0u), i32(0)), b=256)
        if s.fuse2:
            node(s, 9, s.m0u, None, z, s.m0u)                                          # z += T P z_c, m0u = T z
        else:
            node(s, 5, s.m0u, None, z, None); node(s, 3, None, None, z, s.m0u)
        expand0(s, z, s.m0u)
        smooth(s, r, z, False)
        ST["nvc"] += 1

    def _mg_vc(s, l, r, z):
        if l == 0 and on(s):
            return vc0(s, r, z)
        return ovc(s, l, r, z)

    R._mg_vc = _mg_vc
    print(f"[절 CM117] extsolve 켬 -- 줄어든 문제 Ã = Eᵀ A E 의 층 0 평활 = D̃ 척도 체비셰프 {ST['k']} 차 [{ST['lo']} λ̂, {ST['hi']} λ̂] (wolfh13 모드 2 위; 확장 판 (κ 0) 에서만)", flush=True)
    import atexit
    atexit.register(_dump)
    _dump()


def _dump():
    try:
        with open(os.path.join(ST["out"], "xsinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


# ---------------------------------------------------------------- 절 CM119: 헤시안 투영을 0 자르기 대신 절댓값으로 (모래 ao_elem_s 본문 안만)
ABS = dict(n=0)
ABS_OLD = ("float d_ = fmaxf(D[m][m], 0.f);", "lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;")
ABS_NEW = ("float d_ = fabsf(D[m][m]);   /* 절 CM119 --xabs */", "lt = isfinite(lt) ? fabsf(lt) : 0.f; lf = isfinite(lf) ? fabsf(lf) : 0.f;   /* 절 CM119 --xabs */")


def _abs_patch(code):
    h = "ao_elem_s("
    i = code.find("__global__ void __launch_bounds__(128) " + h)
    if i < 0 or "절 CM119 --xabs" in code:
        return code
    j = code.find("\n}\n", i)
    body = code[i:j]
    for a, b in zip(ABS_OLD, ABS_NEW):
        assert body.count(a) == 1, ("xabs: 바꿀 자리", a, body.count(a))
        body = body.replace(a, b)
    ABS["n"] += 1
    return code[:i] + body + code[j:]


def install_abs(out):
    """헤시안 주값 블록 고유값·쌍 계수의 음수를 0 대신 절댓값으로 (Chen et al. 2024 'Trust-region eigenvalue filtering for projected Newton' 의 abs 투영).
    양정치 유지, 힘·커밋 그대로 (헤시안만). 모래 경로 ao_elem_s 본문 안만 (수박 aq_elem 안 건드림)."""
    from . import fem_an_rt as ART
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = _abs_patch(kw["code"])
        elif a:
            a = (_abs_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    ST["abs"] = ABS
    print("[절 CM119] --xabs 켬 -- 모래 헤시안 투영 = 절댓값 (ao_elem_s)", flush=True)


# ---------------------------------------------------------------- 절 CM119: 꼭짓점 (st 2, τ̂ = 0) 가우스 점의 헤시안 = 참 접선 0 (모래 ao_elem_s 본문 안만)
APX = dict(n=0)
APX_OLD = ("dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }", "C[6+pp] = vq*0.5f*(lt + lf);")
APX_NEW = ("dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; if (st == 2) dm[m] = 0.f; }   /* 절 CM119 --xapex */",
           "if (st == 2) { lt = 0.f; lf = 0.f; }   /* 절 CM119 --xapex */ C[6+pp] = vq*0.5f*(lt + lf);")


def _apex_patch(code):
    i = code.find("__global__ void __launch_bounds__(128) ao_elem_s(")
    if i < 0 or "절 CM119 --xapex" in code:
        return code
    j = code.find("\n}\n", i)
    body = code[i:j]
    for a, b in zip(APX_OLD, APX_NEW):
        assert body.count(a) == 1, ("xapex: 바꿀 자리", a, body.count(a))
        body = body.replace(a, b)
    APX["n"] += 1
    return code[:i] + body + code[j:]


def install_apex(out):
    """꼭짓점 상태 (원뿔 꼭짓점, 반환 뒤 응력 0) 점만 헤시안 기여 0 (그 점의 참 접선; 힘은 이미 0). 탄성·꼬리 점은 그대로 (절 CM108 T2 는 꼬리까지 바꿔 나빠짐)."""
    from . import fem_an_rt as ART
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = _apex_patch(kw["code"])
        elif a:
            a = (_apex_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    ST["apex"] = APX
    print("[절 CM119] --xapex 켬 -- 꼭짓점 점 헤시안 0 (ao_elem_s)", flush=True)
