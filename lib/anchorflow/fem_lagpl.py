"""절 CM122 소성 지연 판 (--xlag, 기본 0 = 안 부름): 뉴턴 반복 중 모래 소성 상태를 걸음 시작 값 F_q 로 고정 (순수 초탄성 Hencky 문제), 수렴 뒤 걸음 끝에 한 번 반환 사상.

정답 MPM (PhysGaussian) 의 서브스텝 = 탄성 예측 (시험 F) -> 걸음 끝 반환 사상 한 번 과 같은 꼴을 우리 걸음 (0.04 s) 에 적용:
  - 힘 (ao_elem_s · ao_elem): τ = 2μ ε + λ tr ε (ε = log|σ(X)|, X = (I + Σ y ⊗ ∇w) F_q) -- 반환 사상 (꼬리 투영·꼭짓점 0) 을 빼고 시험 X 의 Hencky 탄성 그대로.
    헤시안은 원래부터 시험 X 의 Hencky 탄성 접선 (PSD 투영) 이라 기울기의 미분과 맞음 (투영된 고유값 몫만 다름).
    가우스 점 상태 수 (cnt: 꼬리·꼭짓점) 는 진단이라 시험 X 의 반환 사상 판정 그대로 셈.
  - 걸음 끝 커밋 ao_commit (F_q ← F_q X⁻¹ RM(X)) · G2P ao_g2p (입자 F 걸음별 RM) 는 그대로 = 걸음 끝 반환 한 번.
  - 모드 2 (--xlag 2): 선탐색 에너지도 같은 초탄성 포텐셜로 -- rt_elem_lsN 의 탄성 에너지 (μ Σ(σ−1)² + λ/2 (J−1)²) 를 Hencky W = μ Σ ε_i² + λ/2 (Σ ε_i)² 로,
    또 c = 0 (현재 반복점) 의 Hencky 에너지를 SC[10] 에 더함 (재질 판은 rt_elem 이 없어 SC[10] 에 탄성 몫이 없었음) -> 아르미호 E(c) ≤ E(0) + 1e-4 c g·δ 가
    풀리는 포텐셜 (관성 + Hencky + 장벽) 과 그 기울기로 맞음. 장벽·고스트 등 다른 항은 손대지 않음.
훅: cupy.RawModule 글자 치환 (fem_extsolve.install_apex 와 같은 방식). 끄면 (설치 안 하면) 소스 그대로.
기록: <out>/lagplinfo.json (모드, 치환한 커널 수).
"""
import json
import os

ST = dict(mode=0, out=".", n_elem=0, n_ls=0, said=False)
TAG = "/* 절 CM122 --xlag */"

ELEM_S_OLD = "if (cnt && on && st > 0) atomicAdd(&cnt[st-1], 1);"
ELEM_OLD = "if (cnt && on && st > 0 && (mode & 1)) atomicAdd(&cnt[st-1], 1);"
HEAD_S = "__global__ void __launch_bounds__(128) ao_elem_s("
HEAD_E = "__global__ void __launch_bounds__(128) ao_elem("
HEAD_LS = "__global__ void __launch_bounds__(128) rt_elem_lsN("

LS_SUBS = [
    ("    for (int i=i0;i<NC;i++) {\n        float X[9];",
     "    for (int i=(i0 == 0 ? -1 : i0);i<NC;i++) {   " + TAG + " // i = −1: c = 0 (현재 반복점) -> SC[10]\n        float cc_ = (i < 0) ? 0.f : c;\n        float X[9];"),
    ("e[a] = eta[id*3+a] + c*dl[id*3+a];", "e[a] = eta[id*3+a] + cc_*dl[id*3+a];"),
    ("        double en = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;",
     "        double l0_ = log(fmax(fabs((double)S[0]), 1e-14)), l1_ = log(fmax(fabs((double)S[1]), 1e-14)), l2_ = log(fmax(fabs((double)S[2]), 1e-14)), t_ = l0_ + l1_ + l2_;   " + TAG + " // Hencky\n"
     "        double en = on ? (double)vq*((double)mu*(l0_*l0_ + l1_*l1_ + l2_*l2_) + 0.5*(double)lam*t_*t_) : 0.0; (void)J;"),
    ("        blocksum_add(en, &SC[40+i]);\n        c *= 0.5f;",
     "        blocksum_add(en, (i < 0) ? &SC[10] : &SC[40+i]);\n        if (i >= 0) c *= 0.5f;"),
]


def _body(code, head, start):
    i = code.find(head, start)
    if i < 0:
        return None
    j = code.find("\n}\n", i)
    return i, (len(code) if j < 0 else j)


def _patch(code):
    if TAG in code:
        return code
    for head, old in ((HEAD_S, ELEM_S_OLD), (HEAD_E, ELEM_OLD)):
        pos = 0
        while True:
            b = _body(code, head, pos)
            if b is None:
                break
            i, j = b
            body = code[i:j]
            if body.count(old) == 1:
                body = body.replace(old, old + " st = 0; " + TAG)
                code = code[:i] + body + code[j:]
                ST["n_elem"] += 1
            pos = i + len(head)
    if ST["mode"] >= 2:
        pos = 0
        while True:
            b = _body(code, HEAD_LS, pos)
            if b is None:
                break
            i, j = b
            body = code[i:j]
            if all(body.count(o) == 1 for o, _ in LS_SUBS):
                for o, n in LS_SUBS:
                    body = body.replace(o, n)
                code = code[:i] + body + code[j:]
                ST["n_ls"] += 1
            pos = i + len(HEAD_LS)
    _rec()
    return code


def _rec():
    try:
        json.dump(dict(mode=ST["mode"], n_elem=ST["n_elem"], n_ls=ST["n_ls"]), open(os.path.join(ST["out"], "lagplinfo.json"), "w"))
    except OSError:
        pass
    if not ST["said"] and ST["n_elem"]:
        ST["said"] = True
        print(f"[절 CM122] --xlag {ST['mode']}: 소성 지연 -- 재질 커널 {ST['n_elem']} 곳 반환 사상 뺌 (커밋 그대로)"
              + (f", 선탐색 Hencky {ST['n_ls']} 곳" if ST["mode"] >= 2 else ""), flush=True)


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
    print(f"[절 CM122] --xlag {mode} 설치 (모래 반환 사상 = 걸음 끝만)", flush=True)
