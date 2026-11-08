"""절 CM70 묶음 wolfh5 표 (json 만 읽음 -- torch 없이 로컬에서도): wolf 판별 결과 + 무리 요약 (절 CM68 표 형식 + 낙하 몫) + 진단 (w5dg.json, 판 안 걸음 기록) + 입자 수준 대조 (w5gt.json 정답 · w5p.json 판).

사용: python exe/wolfh5_tab.py <출력 디렉토리>   (판 목록 exe/wolfh5_spec.txt)
"""
import json
import math
import os
import re
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
LIM = 1000.0 / 60.0
G = 9.8
SPEC = {}
for ln in open(os.path.join(HERE, "wolfh5_spec.txt")):
    if ln.startswith("#") or not ln.strip():
        continue
    t = ln.split()
    SPEC[t[1]] = dict(stg=t[0], sc=t[2], md=t[3], fl=" ".join(t[4:]))
EXTRA = {}                                                                         # 앞 묶음 대조 판 (같은 표 형식): 환경 W5_EXTRA="무리=디렉토리/판,디렉토리/판;..."
for it in filter(None, os.environ.get("W5_EXTRA", "").split(";")):
    g_, ps = it.split("=", 1)
    EXTRA[g_] = ps.split(",")


def jl(n, f):
    p = n if os.path.isabs(n) else os.path.join(O, n)
    try:
        return json.load(open(os.path.join(p, f)))
    except Exception:
        return None


def rec(n):
    p = os.path.join(n if os.path.isabs(n) else os.path.join(O, n), "fem.jsonl")
    if not os.path.isfile(p):
        return None
    L = []
    for l_ in open(p):
        try:
            q = json.loads(l_)
        except Exception:
            continue
        if "t" in q:
            L.append(q)
    return L


def err(s):
    if s is None:
        return None
    e = s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def tm(r):
    q = [x["ms"] for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    if not q:
        return None
    return sum(q) / len(q), max(q), sum(1 for b in q if b > LIM)


def com(r, t):
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


def bbox(n):
    p = os.path.join(n if os.path.isabs(n) else os.path.join(O, n), "bbox.txt")
    if not os.path.isfile(p):
        return None
    for ln in open(p):
        if "pred.pt" in ln:
            try:
                D = json.loads(ln[ln.index("{"):])
            except Exception:
                return None
            return max((v["nfar"], v["dmax"]) for v in D.values())
    return None


def div(s):
    return bool(s is None or s.get("diverged") or s.get("div_at") or s.get("oom") or (s.get("frames_done") or 0) < 50)


def relfall(n):
    """relfall.md 의 판 줄 -> τ 칸별 몫 목록."""
    p = os.path.join(n if os.path.isabs(n) else os.path.join(O, n), "relfall.md")
    if not os.path.isfile(p):
        return None
    for ln in open(p):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if c and c[0] == os.path.basename(n):
            return [float(re.findall(r"\(([-0-9.]+)\)", x)[0]) if "(" in x else None for x in c[1:]]
    return None


def relfall_gt():
    for n in list(SPEC) + [q for v in EXTRA.values() for q in v]:
        p = os.path.join(n if os.path.isabs(n) else os.path.join(O, n), "relfall.md")
        if os.path.isfile(p):
            for ln in open(p):
                c = [x.strip() for x in ln.strip().strip("|").split("|")]
                if c and c[0] == "정답":
                    return [float(re.findall(r"\(([-0-9.]+)\)", x)[0]) for x in c[1:]]
    return None


AN, GT = {}, None
for n in list(SPEC) + [q for v in EXTRA.values() for q in v]:
    J = jl(n, "an.json")
    if J:
        GT = GT or J["gt"]
        for k_, v_ in J["runs"].items():
            AN[n if os.path.isabs(n) else k_] = v_


def an(n, f, k):
    r = AN.get(n)
    if not r:
        return None
    s = r["series"].get(str(f)) or r["series"].get(f)
    return None if s is None else s.get(k)


def vt0(get):
    v = [get(f) for f in range(10, 31)]
    v = [x[0] for x in v if x and x[0] is not None]
    return sum(v) / len(v) if v else None


def fm(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def rng(v, d=0):
    v = [x for x in v if x is not None]
    return "-" if not v else (f"{sum(v) / len(v):.{d}f} [{min(v):.{d}f}~{max(v):.{d}f}]" if len(v) > 1 else f"{v[0]:.{d}f}")


def res(s):
    if s is None:
        return "없음/실패"
    if s.get("diverged") or s.get("div_at") or s.get("oom"):
        return f"발산 f{s.get('div_at') or s.get('diverged')}" + (" (메모리)" if s.get("oom") else "")
    return f"완주 {s.get('frames_done')}"


TB = ["[0,.04)", "[.04,.08)", "[.08,.12)", "[.12,.16)", "[.16,.24)", "[.24,.32)", "[.32,.48)"]
print("# 절 CM70 묶음 wolfh5 -- wolf 높게 쌓임 5: 방출 직후 낙하 매달림 진단 + 정의, 공통 후보 G'' (= G' + 절 CM67 (2)) 위\n")
print("정의 (절 CM68 과 같음): f10·f20·f30·f50 = 질량중심 세계 z (mm; 바닥 0.48 m, 정답 f50 533). 낙하 몫 = 바닥 위 10 cm 넘는 풀린 입자 수직 속도 / 자유 낙하 −g τ "
      "(exe/wolfh4_relfall.py, f1~f24, τ = 방출 뒤 시간 칸 s). 면적 = 바닥 1 cm 안 입자 5 mm 칸 점유 (cm²), 회전 반경 = 평면 회전 반경 f50 (mm). 이탈 = 몸 상자 4 칸 밖 > 20 개 판. "
      "발산 = 판정 발산·메모리·50 프레임 못 감. 시간 = 렌더 포함 재생 GPU ms (첫 2 프레임 뺌), 동시 판이라 참고.\n")
GR = {}
for n, v in SPEC.items():
    if v["sc"] == "W" and v["md"] != "z" and os.path.isdir(os.path.join(O, n)) and "--frames 8" not in v["fl"]:
        GR.setdefault(re.sub(r"_\d+$", "", n), []).append(n)
for g_, ps in EXTRA.items():
    GR[g_] = [p for p in ps if os.path.isdir(p)]
for st in sorted({v["stg"] for v in SPEC.values()}):
    ns = [n for n, v in SPEC.items() if v["stg"] == st and v["sc"] == "W" and v["md"] != "z" and os.path.isdir(os.path.join(O, n))]
    if not ns:
        continue
    print(f"## 단계 {st} wolf 판별\n")
    print("| 판 | 플래그 | 결과 | 오차 % | f10 · f20 · f30 · f50 mm | 낙하 몫 τ [.04,.08) · [.16,.24) · [.32,.48) | 면적 f30 · f50 | 이탈 | ms 평균 / 최대 / >16.7 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in ns:
        s = jl(n, "summary.json"); r = rec(n); t = tm(r); b = bbox(n); rf = relfall(n) or [None] * 7
        print(f"| {n} | `{SPEC[n]['fl']}` | {res(s)} | {fm(err(s), 2)} | {fm(com(r, 9), 0)} · {fm(com(r, 19), 0)} · {fm(com(r, 29), 0)} · {fm(com(r, 49), 0)} | "
              f"{fm(rf[1], 2)} · {fm(rf[4], 2)} · {fm(rf[6], 2)} | {fm(an(n, 30, 'area_cm2'), 0)} · {fm(an(n, 50, 'area_cm2'), 0)} | "
              f"{'-' if b is None else f'{b[0]}·{b[1]:.2f}'} | {'-' if t is None else f'{t[0]:.2f} / {t[1]:.1f} / {t[2]}'} |")
    print()

# ---- 절 CM68 표 형식 + 낙하 몫
if GR:
    print("## 절 CM68 표 형식 (+ 낙하 몫)\n")
    print("| 후보 | 판 | 발산 | f10 · f20 mm | f30 mm | f50 mm [범위] (완주) | 오차 % [범위] | 낙하 몫 τ [.04,.08) · [.16,.24) · [.32,.48) | 면적 f30 · f50 cm² | 회전 반경 f50 mm | 이탈 판 · 최대 m | 동시 ms 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rg_ = relfall_gt()
    if GT:
        print(f"| 정답 | - | - | {480 + GT['10']['com']:.0f} · {480 + GT['20']['com']:.0f} | {480 + GT['30']['com']:.0f} | 533 | - | "
              f"{'-' if not rg_ else f'{rg_[1]:.2f} · {rg_[4]:.2f} · {rg_[6]:.2f}'} | {GT['30']['area_cm2']:.0f} · {GT['50']['area_cm2']:.0f} | {GT['50']['rg_mm']:.0f} | - | - |")
    for g_, ns in GR.items():
        f10, f20, f30, f50, er, a30, a50, rg, nd, ne, dm, mx, R1, R4, R6 = [], [], [], [], [], [], [], [], 0, 0, [], [], [], [], []
        for n in ns:
            s = jl(n, "summary.json"); r = rec(n)
            if s is None and r is None:
                continue
            if div(s):
                nd += 1
            else:
                f50.append(com(r, 49)); er.append(err(s)); a50.append(an(n, 50, "area_cm2")); rg.append(an(n, 50, "rg_mm"))
            f10.append(com(r, 9)); f20.append(com(r, 19)); f30.append(com(r, 29)); a30.append(an(n, 30, "area_cm2"))
            rf = relfall(n)
            if rf:
                R1.append(rf[1]); R4.append(rf[4]); R6.append(rf[6])
            b = bbox(n)
            if b:
                dm.append(b[1])
                if b[0] > 20:
                    ne += 1
            t = tm(r)
            if t:
                mx.append(t[1])
        print(f"| {g_} | {len(ns)} | {nd} | {rng(f10)} · {rng(f20)} | {rng(f30)} | {rng(f50)} | {rng(er, 2)} | {rng(R1, 2)} · {rng(R4, 2)} · {rng(R6, 2)} | {rng(a30)} · {rng(a50)} | {rng(rg)} | "
              f"{ne}/{len(ns)} · {fm(max(dm) if dm else None, 2)} | {fm(max(mx) if mx else None, 1)} |")
    print()
    print("### 낙하 몫 τ 칸 전부 (무리 평균)\n")
    print("| 무리 | " + " | ".join(TB) + " |")
    print("|---|" + "---|" * 7)
    if rg_:
        print("| 정답 | " + " | ".join(f"{x:.2f}" for x in rg_) + " |")
    for g_, ns in GR.items():
        L = [relfall(n) for n in ns]
        L = [x for x in L if x]
        if L:
            print(f"| {g_} | " + " | ".join(fm(sum(q[i] for q in L if q[i] is not None) / max(1, sum(1 for q in L if q[i] is not None)), 2) for i in range(7)) + " |")
    print()
    if GT:
        FR = [5, 10, 15, 20, 25, 30, 40, 50]
        print("### 질량중심 바닥 위 mm (무리 평균)\n")
        print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
        print("|---|" + "---|" * len(FR))
        print("| 정답 | " + " | ".join(fm(GT[str(f)].get("com"), 0) for f in FR) + " |")
        for g_, ns in GR.items():
            row = []
            for f in FR:
                vv = [an(n, f, "com") for n in ns]
                vv = [x for x in vv if x is not None]
                row.append(fm(sum(vv) / len(vv), 0) if vv else "-")
            print(f"| {g_} | " + " | ".join(row) + " |")
        print()

# ---- 진단 (w5dg.json)
DB = ["(−∞,−6)", "[−6,−3)", "[−3,0)", "[0,3)", "[3,6)", "[6,11)", "[11,21)", "[21,40)", "[40,∞)"]
F0, F1 = int(os.environ.get("W5_F0", "1")), int(os.environ.get("W5_F1", "24"))


def dg_sum(ns):
    P = Q = None; dt = 0.0
    for n in ns:
        J = jl(n, "w5dg.json")
        if not J:
            continue
        for fr in J["frames"]:
            if F0 <= fr["f"] <= F1:
                dt += sum(fr["dt"])
                if P is None:
                    P = [[[0.0] * len(fr["P"][0][0]) for _ in range(2)] for _ in range(9)]
                    Q = [[[0.0] * len(fr["Q"][0][0]) for _ in range(2)] for _ in range(9)]
                for i in range(9):
                    for h in range(2):
                        for j in range(len(P[i][h])):
                            P[i][h][j] += fr["P"][i][h][j]
                        for j in range(len(Q[i][h])):
                            Q[i][h][j] += fr["Q"][i][h][j]
    return P, Q, dt


DGR = {g_: [n for n in ns if jl(n, "w5dg.json")] for g_, ns in GR.items()}
DGR = {g_: ns for g_, ns in DGR.items() if ns}
if DGR:
    print(f"## 진단 (--w5dg; 판 안 걸음 끝 기록, f{F0}~f{F1} 걸음 Δt 가중, 무리 판 합): 방출 앞면 거리 칸 (cm, 방출된 쪽 +)\n")
    print("입자 줄 (거리 = (t − t_방출) v_앞면, 프레임 시작 위치 높이): M 몫 = 그 칸 질량 / 그 높이 무리 질량. k̄ = Σ w k_구속 (노드 구속 몫을 입자로; 1 = 안 묶임), "
          "ᾱ = Σ w α (노드 구속 질량 몫), 걸린 무게 = Σ w (1 − k_구속). v1 = 걸음 끝 노드 속도 ·ẑ 를 입자로 (m/s), 몫 = v1 / (−g τ̄). "
          "힘 (중력 무게 M g 로 나눔, 위 +): F_net = 노드 운동에서 거꾸로 낸 중력 말고의 힘 (구속 반력·재질·ghost·장벽·덜 풀림 합), F_el = 재질 응력 (마지막 뉴턴 반복), "
          "F_ghost = 요소 커널 사슬 기울기 − 재질 (ghost penalty 강성 + 고스트 질량), F_net 풀린 몫 = k_구속 가중 (구속 노드 몫 뺀 것), 나머지 = F_net 풀린 몫 − F_el 풀린 몫. "
          "가우스 점 줄 (거리 = |x − 끝 위치| − (반폭 − v_앞면 t), 걸음 끝 시험 X 의 DP 투영): p · |s| kPa (압축 +), 원뿔·꼭짓점 % (부피), k_q = Σ BW k_구속, tr ε 1e-3.\n")
    for hh, hl in ((1, "바닥 위 ≥ 10 cm (공중)"), (0, "바닥 위 < 10 cm")):
        print(f"### {hl}\n")
        print("| 무리 | 값 | " + " | ".join(DB) + " |")
        print("|---|---|" + "---|" * 9)
        for g_, ns in DGR.items():
            P, Q, dt = dg_sum(ns)
            if P is None:
                continue
            Mt = sum(P[i][hh][1] for i in range(9))

            def pr(j, sc=1.0, d=2, i=None):
                return "-" if P[i][hh][1] <= 0 else f"{sc * P[i][hh][j] / P[i][hh][1]:.{d}f}"

            def fg(j, i, sub=None):
                M = P[i][hh][1]
                if M <= 0:
                    return "-"
                v = P[i][hh][j] - (P[i][hh][sub] if sub is not None else 0.0)
                return f"{v / (M * G):.2f}"

            def vr(i):
                M = P[i][hh][1]
                if M <= 0 or i < 3:
                    return "-"
                tau = P[i][hh][7] / M
                return "-" if tau <= 0 else f"{(P[i][hh][6] / M) / (-G * tau):.2f}"
            print(f"| {g_} | M 몫 % | " + " | ".join("-" if Mt <= 0 else f"{100 * P[i][hh][1] / Mt:.1f}" for i in range(9)) + " |")
            print(f"| {g_} | k̄ · ᾱ · 걸린 무게 | " + " | ".join("-" if P[i][hh][1] <= 0 else f"{pr(2, i=i)}·{pr(3, i=i)}·{pr(4, i=i)}" for i in range(9)) + " |")
            print(f"| {g_} | v1 m/s · 낙하 몫 | " + " | ".join("-" if P[i][hh][1] <= 0 else f"{pr(6, i=i)}·{vr(i)}" for i in range(9)) + " |")
            print(f"| {g_} | F_net · F_el · F_ghost (/Mg) | " + " | ".join(f"{fg(8, i)}·{fg(9, i)}·{fg(10, i, 9)}" for i in range(9)) + " |")
            print(f"| {g_} | 풀린 몫 F_net · F_el · 나머지 (/Mg) | " + " | ".join(f"{fg(11, i)}·{fg(12, i)}·{fg(11, i, 12)}" for i in range(9)) + " |")

            def qq(j, sc=1.0, d=1, i=None):
                return "-" if Q[i][hh][1] <= 0 else f"{sc * Q[i][hh][j] / Q[i][hh][1]:.{d}f}"
            print(f"| {g_} | 가우스 p · \\|s\\| kPa | " + " | ".join(f"{qq(4, 1e-3, i=i)}·{qq(5, 1e-3, i=i)}" for i in range(9)) + " |")
            print(f"| {g_} | 가우스 원뿔 · 꼭짓점 % · k_q · tr ε (1e-3) | " + " | ".join(f"{qq(2, 100, 0, i)}·{qq(3, 100, 0, i)}·{qq(7, 1, 2, i)}·{qq(9, 1e3, 2, i)}" for i in range(9)) + " |")
        print()

# ---- 입자 수준 대조 (w5gt.json · w5p.json)
PJ = {}
J = jl(O, "w5gt.json") if os.path.isfile(os.path.join(O, "w5gt.json")) else None
if J:
    PJ.update(J["res"])
for g_, ns in GR.items():
    for n in ns:
        j_ = jl(n, "w5p.json")
        if j_:
            for k_, v_ in j_["res"].items():
                PJ[f"{g_} ({os.path.basename(n)})"] = v_
if PJ:
    print(f"## 입자 수준 대조 (exe/wolfh5_gt.py; 정답 = PG h5 원본 f32, 판 = --dump_F 덤프; f{F0}~f{min(F1, 30)} 합; 정답 dx 1 cm 격자)\n")
    print("k̄ = Σ_g w k_g (정답 격자 노드가 그 프레임 유지하는 속도 몫; 판 줄도 같은 정답 격자 식 -- 판 자신의 노드 구속은 위 진단 표), ᾱ = 정답 격자 구속 질량 몫, "
          "낙하 몫 = 입자 중심 차분 수직 속도 / (−g τ̄), F_net · F_el = /Mg (F_el = 입자 F 의 Hencky 응력을 정답 격자 P2G 한 힘의 입자 몫), p · |s| kPa (입자 F 부피 가중).\n")
    for hh, hl in ((1, "바닥 위 ≥ 10 cm (공중)"), (0, "바닥 위 < 10 cm")):
        print(f"### {hl}\n")
        print("| 판 | 값 | " + " | ".join(DB) + " |")
        print("|---|---|" + "---|" * 9)
        for n, L in PJ.items():
            S = [[[0.0] * 12 for _ in range(2)] for _ in range(9)]
            for f, fr in enumerate(L, start=1):
                if F0 <= f <= F1:
                    for i in range(9):
                        for h in range(2):
                            for j in range(12):
                                S[i][h][j] += fr[i][h][j]
            Mt = sum(S[i][hh][1] for i in range(9))

            def s_(j, sc=1.0, d=2, i=None):
                return "-" if S[i][hh][1] <= 0 else f"{sc * S[i][hh][j] / S[i][hh][1]:.{d}f}"

            def vr2(i):
                M = S[i][hh][1]
                if M <= 0 or i < 3:
                    return "-"
                tau = S[i][hh][6] / M
                return "-" if tau <= 0 else f"{(S[i][hh][5] / M) / (-G * tau):.2f}"
            V = [S[i][hh][1] / 2000.0 for i in range(9)]
            print(f"| {n} | M 몫 % | " + " | ".join("-" if Mt <= 0 else f"{100 * S[i][hh][1] / Mt:.1f}" for i in range(9)) + " |")
            print(f"| {n} | k̄ · ᾱ · 걸린 무게 | " + " | ".join("-" if S[i][hh][1] <= 0 else f"{s_(2, i=i)}·{s_(3, i=i)}·{s_(4, i=i)}" for i in range(9)) + " |")
            print(f"| {n} | v m/s · 낙하 몫 | " + " | ".join("-" if S[i][hh][1] <= 0 else f"{s_(5, i=i)}·{vr2(i)}" for i in range(9)) + " |")
            print(f"| {n} | F_net · F_el (/Mg) | " + " | ".join("-" if S[i][hh][1] <= 0 else f"{S[i][hh][8] / (S[i][hh][1] * G):.2f}·{S[i][hh][9] / (S[i][hh][1] * G):.2f}" for i in range(9)) + " |")
            print(f"| {n} | p · \\|s\\| kPa | " + " | ".join("-" if V[i] <= 0 else f"{1e-3 * S[i][hh][10] / V[i]:.1f}·{1e-3 * S[i][hh][11] / V[i]:.1f}" for i in range(9)) + " |")
        print()
