"""절 CM128 뒤집힌 가우스 점 = 압축 (--xsinv, 기본 0 = 안 부름): 모래 반환 사상·Hencky 응력·헤시안을 부호 있는 SVD 로.

지금 판 (ao_dpf): 부호 있는 SVD (U, V 회전, 가장 작은 σ 만 음수 가능) 를 하고도 변형률을 ε_i = log|σ_i| 로 잰다 -- 정답 PG sand_return_mapping 과 같은 글자.
  그래서 뒤집힌 점 (σ_3 < 0) 은 거울상 (|σ|) 으로 취급되어, |σ| 곱이 1 보다 크면 '부피 늘어남 -> 꼭짓점 -> 응력 0', 그 밖에서도 힘이 σ_3 = −1 (거울 정지점) 쪽으로 민다.
  걸음 끝 커밋·G2P 는 F_E = U diag(sgn ...) Vᵀ 로 뒤집힘을 그대로 저장 (정답은 U e^H Vᵀ · U Vᵀ 로 늘 풀린 F_E 를 저장 -- 절 CM110.1).
--xsinv 1 정의 (부호 있는 σ; 뒤집히지 않은 점 (σ_i > 0 모두) 은 비트까지 지금과 같음):
  - 변형률 ε_i = log σ_i (σ_i > 0, 정답과 같음) · ε_i = σ_i − 1 (σ_i ≤ 0: 로그를 정지 상태 σ = 1 에서 1 차로 이은 값 = 작은 변형률; 상수·문턱 없음).
    σ ≤ 0 이면 ε ≤ −1 (늘 압축) 이고 dε/dσ = 1 > 0 이라 압축 응력의 힘 −τ dε/dσ 가 σ_3 를 키우는 쪽 (뒤집힘을 푸는 쪽) 으로 민다.
  - 반환 사상 (꼬리 투영·꼭짓점 판정) 은 정답 식 그대로 이 ε 에 (항복 판정 Δγ = ‖ε̂‖ + cdp tr).
  - 1 차 PK: P = U diag(τ_i ε'_i) Vᵀ, ε' = 1/σ (σ > 0) · 1 (σ ≤ 0).
  - 헤시안 (시험 X 의 Hencky 탄성 접선, 지금과 같은 투영): 주값 D_ij = (2μ δ_ij + λ) ε'_i ε'_j + δ_ij τ_i ε''_i (ε'' = −1/σ² · 0), 쌍 (lt·lf) 식은 그대로 (P̂·σ 부호 있는 값).
  - 커밋·G2P (ao_rm3): F_E = U diag(exp(ε_E)) Vᵀ (탄성 갈래 뒤집힌 점은 H = ε, 꼬리 H = tr/3 + s ε̂, 꼭짓점 U Vᵀ) -- 정답 식과 같은 꼴로 늘 풀린 F_E,
    같은 ε 를 저장 (log exp(ε) = ε) 하므로 응력이 이어짐. 뒤집히지 않은 탄성 점은 F_E = X 그대로.
훅: cupy.RawModule 를 감싸개가 시작하자마자 (다른 감싸개 훅보다 먼저 = 가장 안쪽, 최종 글자를 봄) 감싼다. 치환 자리 수는 <out>/sinvinfo.json.
"""
import json
import os

ST = dict(mode=0, out=".", n=dict(dpf=0, str=0, D=0, rm=0), said=False)
TAG = "/* 절 CM128 --xsinv */"

SUBS = [
    ("dpf",
     "for (int i=0;i<3;i++) { sgn[i] = (S[i] < 0.f) ? -1.f : 1.f; sg[i] = fmaxf(fabsf(S[i]), 1e-14f); ep[i] = logf(sg[i]); }",
     "for (int i=0;i<3;i++) { sgn[i] = (S[i] > 0.f) ? 1.f : -1.f; sg[i] = fmaxf(fabsf(S[i]), 1e-14f); ep[i] = (S[i] > 0.f) ? logf(sg[i]) : (S[i] - 1.f); } " + TAG),
    ("str",
     "ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];",
     "ss[i] = sgn[i]*sg[i]; p[i] = (sgn[i] > 0.f) ? tau/ss[i] : tau; pe[i] = (sgn[i] > 0.f) ? (2.f*mu*ep[i] + lam*tr)/ss[i] : (2.f*mu*ep[i] + lam*tr); " + TAG),
    ("D",
     "D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f);",
     "D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(((sgn[i] > 0.f) ? ss[i] : 1.f)*((sgn[j] > 0.f) ? ss[j] : 1.f)) - ((i==j && sgn[i] > 0.f) ? pe[i]/ss[i] : 0.f); " + TAG),
    ("rm",
     "    if (st == 0) { for (int k=0;k<9;k++) Y[k] = X[k]; return false; }\n    float sn[3]; for (int i=0;i<3;i++) sn[i] = (st == 2) ? sgn[i] : sgn[i]*expf(tr/3.f + s*e[i]);",
     "    if (st == 0 && sgn[0] > 0.f && sgn[1] > 0.f && sgn[2] > 0.f) { for (int k=0;k<9;k++) Y[k] = X[k]; return false; }   " + TAG + "\n"
     "    float sn[3]; for (int i=0;i<3;i++) sn[i] = (st == 2) ? 1.f : expf(tr/3.f + s*e[i]);"),
]


def patch(code):
    for k, old, new in SUBS:
        c = code.count(old)
        if c:
            code = code.replace(old, new)
            ST["n"][k] += c
    _rec()
    return code


def _rec():
    try:
        json.dump(dict(mode=ST["mode"], n=ST["n"]), open(os.path.join(ST["out"], "sinvinfo.json"), "w"))
    except OSError:
        pass
    n = ST["n"]
    if not ST["said"] and n["dpf"] and n["str"] and n["rm"]:
        ST["said"] = True
        print(f"[절 CM128] --xsinv {ST['mode']}: 부호 있는 σ 치환 ao_dpf {n['dpf']} · 응력 {n['str']} · 헤시안 {n['D']} · 반환 F_E {n['rm']} 곳", flush=True)


def install(out, mode=1):
    import cupy as cp
    orm = cp.RawModule
    ST.update(mode=int(mode), out=out)

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cp.RawModule = RawModule
    print(f"[절 CM128] --xsinv {mode} 설치 (뒤집힌 가우스 점 = 압축, 부호 있는 SVD)", flush=True)
