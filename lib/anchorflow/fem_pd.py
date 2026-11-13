"""절 CM129 local-global (Projective Dynamics 꼴) 모래 판 (`--xpd 1`, 기본 0 = 안 부름).

local  = 가우스점마다 반환 사상 (ao_dpf, 정답 PG sand 식) 을 정확히 적용한 τ̂ 로 힘 (ao_elem_s 의 힘 경로 그대로 = 투영 변형 p_q 를 w (f_q − p_q) = τ̂ f⁻ᵀ 로 쓴 것).
global = 헤시안 자리에 걸음 안 상수인 SPD 행렬 M/τ² + Σ_q V_q Bᵀ C0 B (B: 노드 변위 -> 증분 dA = Σ y ⊗ ∇w, 걸음 시작 격자 기준;
         C0 = 쉼 상태 선형 탄성 = Hencky 헤시안 at I: 주값 2μδ + λ, 쌍 lt 0 · lf 2μ). 확장 접기 Eᵀ(·)E·바닥 장벽 헤시안은 실행기 그대로 (global 에 들어감).
  ao_elem_s 의 저장 계수 (U, M, C) 를 U = M = I, C = V_q C0 로 바꿈 -> ao_mv_q 곱·노드 대각 블록이 같은 상수 행렬. 반복 x <- x − A⁻¹ g(x) (선형 풀이는 실행기 PCG).
  g 가 참 잔차 (반환 사상 뒤 τ̂) 라 수렴하면 뉴턴 (cvb) 과 같은 해.
모래 커널 (ao_elem_s, 재질 판에서 모래만 씀) 만 바꿈 -- 레고·ficus·bread (κ ≠ 0 재질 커널) 는 글자 그대로. 새 문턱·클램프 없음.
훅: cupy.RawModule 글자 치환 (fem_lagpl 과 같은 방식), 자리 못 찾으면 assert.
"""
import json
import os

ST = dict(mode=0, out=".", n_sub=0, said=False)
TAG = "/* 절 CM129 --xpd */"
HEAD_S = "__global__ void __launch_bounds__(128) ao_elem_s("
ANCHOR = "    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; }"
NEW = ("    " + TAG + " { for (int k=0;k<9;k++) { U[k] = (k % 4 == 0) ? 1.f : 0.f; M[k] = U[k]; }\n"
       "      C[0] = C[1] = C[2] = vq*(2.f*mu + lam); C[3] = C[4] = C[5] = vq*lam;\n"
       "      for (int pp=0; pp<3; pp++) { C[6+pp] = vq*mu; C[9+pp] = vq*mu; } }\n")


def _patch(code):
    if TAG in code or HEAD_S not in code:
        return code
    pos = 0
    n = 0
    while True:
        i = code.find(HEAD_S, pos)
        if i < 0:
            break
        j = code.find("\n}\n", i)
        j = len(code) if j < 0 else j
        body = code[i:j]
        assert body.count(ANCHOR) == 1, f"fem_pd: ao_elem_s 저장 자리 {body.count(ANCHOR)} 곳"
        body = body.replace(ANCHOR, NEW + ANCHOR)
        code = code[:i] + body + code[j:]
        n += 1
        pos = i + len(HEAD_S)
    ST["n_sub"] += n
    _rec()
    return code


def _rec():
    try:
        json.dump(dict(mode=ST["mode"], n_sub=ST["n_sub"]), open(os.path.join(ST["out"], "pdinfo.json"), "w"))
    except OSError:
        pass
    if not ST["said"] and ST["n_sub"]:
        ST["said"] = True
        print(f"[절 CM129] --xpd {ST['mode']}: 모래 헤시안 = 상수 PD 행렬 (V_q C0, U = M = I) -- ao_elem_s {ST['n_sub']} 곳", flush=True)


def install(out, mode=1):
    from . import fem_an_rt as ART
    cpm = ART.cp
    orm = cpm.RawModule
    ST.update(mode=int(mode), out=out)

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = _patch(kw["code"])
        elif a:
            a = (_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    print(f"[절 CM129] --xpd {mode} 설치 (local = 반환 사상 힘, global = 상수 SPD 행렬)", flush=True)
