"""절 CM66 묶음 wolfh3 표 (exe/wolfh1_tab.py 꼴): wolf 판별 결과 + 무리 요약 (절 CM52 표 형식 + 면적·바닥 층 접선 속력) + 정답 대 무리 평균 곡선.
진단은 판마다 <판>/an.json (exe/wolfh1_an.py, pred.pt 지우기 전), 이탈은 <판>/bbox.txt, 높이·시간은 fem.jsonl.

사용: python exe/wolfh3_tab.py <출력 디렉토리>   (판 목록 exe/wolfh3_spec.txt; 다른 씬 판은 exe/wolfh3_tabreg.py)
"""
import json
import math
import os
import re
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
LIM = 1000.0 / 60.0
SPEC = {}
for ln in open(os.path.join(HERE, "wolfh3_spec.txt")):
    if ln.startswith("#") or not ln.strip():
        continue
    t = ln.split()
    SPEC[t[1]] = dict(stg=t[0], sc=t[2], md=t[3], fl=" ".join(t[4:]))


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return json.load(open(p))
    except Exception:
        return None


def rec(n):
    p = os.path.join(O, n, "fem.jsonl")
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
    p = os.path.join(O, n, "bbox.txt")
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


def res(s):
    if s is None:
        return "없음/실패"
    if s.get("diverged") or s.get("div_at") or s.get("oom"):
        return f"발산 f{s.get('div_at') or s.get('diverged')}" + (" (메모리)" if s.get("oom") else "")
    return f"완주 {s.get('frames_done')}"


AN, GT = {}, None
for n in SPEC:
    J = jl(n, "an.json")
    if J:
        GT = GT or J["gt"]
        AN.update(J["runs"])


def an(n, f, k):
    r = AN.get(n)
    if not r:
        return None
    s = r["series"].get(str(f)) or r["series"].get(f)
    return None if s is None else s.get(k)


def vt0(get):
    """바닥 층 [0,2) cm 접선 속력 f10~f30 평균."""
    v = [get(f) for f in range(10, 31)]
    v = [x[0] for x in v if x and x[0] is not None]
    return sum(v) / len(v) if v else None


def fm(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def rng(v, d=0):
    v = [x for x in v if x is not None]
    return "-" if not v else (f"{sum(v) / len(v):.{d}f} [{min(v):.{d}f}~{max(v):.{d}f}]" if len(v) > 1 else f"{v[0]:.{d}f}")


print("# 절 CM66 묶음 wolfh3 -- wolf 높게 쌓임 3: --y0eb 걸음별 진단 + Y5 (장벽 에너지 판정을 법선 몫에만), 공통 후보 G' 위\n")
print("정의 (절 CM52 와 같음): f30·f50 = 30·50 프레임 질량중심 세계 z (mm, 실행기 --hgt; 정답 f50 533). 면적 = 바닥 1 cm 안 입자 5 mm 칸 점유 넓이 (cm²). "
      "바닥 층 접선 속력 = 바닥 위 [0,2) cm 입자 질량 가중 평균 |v_t| 의 f10~f30 평균 (m/s). 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 최대 수·거리 m. "
      "발산 = 판정 발산·메모리·50 프레임 못 감. 시간 = 렌더 포함 재생 GPU ms (첫 2 프레임 뺌), 동시 판이라 참고.\n")
if GT:
    print(f"정답: 면적 f30 {GT['30']['area_cm2']:.0f} · f50 {GT['50']['area_cm2']:.0f} cm², 바닥 층 접선 속력 f10~f30 {fm(vt0(lambda f: GT[str(f)].get('vt_layers')), 3)} m/s, "
          f"회전 반경 f50 {GT['50']['rg_mm']:.0f} mm\n")
for st in sorted({v["stg"] for v in SPEC.values()}):
    ns = [n for n, v in SPEC.items() if v["stg"] == st and v["sc"] == "W" and v["md"] != "z" and os.path.isdir(os.path.join(O, n))]
    if not ns:
        continue
    print(f"## 단계 {st} wolf 판별\n")
    print("| 판 | 플래그 | 결과 | 오차 % | f30 · f50 mm | 면적 f30 · f50 | 바닥 층 접선 f10~30 | 갈라짐 5·10·20 | 이탈 | 면 빼기 | ms 평균 / 최대 / >16.7 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ns:
        s = jl(n, "summary.json"); r = rec(n); t = tm(r); b = bbox(n); w2 = jl(n, "w2info.json") or {}
        dv = (AN.get(n) or {}).get("diverge") or {}
        print(f"| {n} | `{SPEC[n]['fl']}` | {res(s)} | {fm(err(s), 2)} | {fm(com(r, 29), 0)} · {fm(com(r, 49), 0)} | {fm(an(n, 30, 'area_cm2'), 0)} · {fm(an(n, 50, 'area_cm2'), 0)} | "
              f"{fm(vt0(lambda f, n=n: an(n, f, 'vt_layers')), 3)} | {dv.get('5', '-')}·{dv.get('10', '-')}·{dv.get('20', '-')} | "
              f"{'-' if b is None else f'{b[0]}·{b[1]:.2f}'} | {w2.get('gs_removed_faces_total', '-')} | {'-' if t is None else f'{t[0]:.2f} / {t[1]:.1f} / {t[2]}'} |")
    print()
GR = {}
for n, v in SPEC.items():
    if v["sc"] == "W" and v["md"] != "z" and os.path.isdir(os.path.join(O, n)):
        GR.setdefault(re.sub(r"_\d+$", "", n), []).append(n)
print("## 무리별 요약 (절 CM52 표 형식 + 면적·바닥 층 접선 속력)\n")
print("| 무리 | 플래그 | 판 | 발산 | f50 mm [범위] (완주) | f30 mm [범위] | 오차 % [범위] | 면적 f30 · f50 cm² | 바닥 층 접선 f10~30 m/s | 이탈 판 (>20 개) · 최대 거리 m | 동시 ms 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
if GT:
    print(f"| 정답 | - | - | - | 533 | - | - | {GT['30']['area_cm2']:.0f} · {GT['50']['area_cm2']:.0f} | {fm(vt0(lambda f: GT[str(f)].get('vt_layers')), 3)} | - | - |")
for g, ns in GR.items():
    f50, f30, er, a30, a50, vt, nd, ne, dm, mx = [], [], [], [], [], [], 0, 0, [], []
    for n in ns:
        s = jl(n, "summary.json"); r = rec(n)
        if s is None and r is None:
            continue
        if div(s):
            nd += 1
        else:
            f50.append(com(r, 49)); er.append(err(s)); a50.append(an(n, 50, "area_cm2"))
        f30.append(com(r, 29)); a30.append(an(n, 30, "area_cm2")); vt.append(vt0(lambda f, n=n: an(n, f, "vt_layers")))
        b = bbox(n)
        if b:
            dm.append(b[1])
            if b[0] > 20:
                ne += 1
        t = tm(r)
        if t:
            mx.append(t[1])
    print(f"| {g} | `{SPEC[ns[0]]['fl']}` | {len(ns)} | {nd} | {rng(f50)} | {rng(f30)} | {rng(er, 2)} | {rng(a30)} · {rng(a50)} | {rng(vt, 3)} | "
          f"{ne}/{len(ns)} · {fm(max(dm) if dm else None, 2)} | {fm(max(mx) if mx else None, 1)} |")
print()
if GT and GR:
    FR = [5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    for key, lab, dd in (("com", "질량중심 높이 (바닥 위) mm", 0), ("area_cm2", "바닥 접촉 면적 cm²", 0), ("rg_mm", "평면 회전 반경 mm", 0), ("ke", "운동 에너지 J", 2)):
        print(f"## 곡선: {lab} (무리 평균)\n")
        print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
        print("|---|" + "---|" * len(FR))
        print("| 정답 | " + " | ".join(fm(GT[str(f)].get(key), dd) for f in FR) + " |")
        for g, ns in GR.items():
            row = []
            for f in FR:
                vv = [an(n, f, key) for n in ns]
                vv = [x for x in vv if x is not None]
                row.append(fm(sum(vv) / len(vv), dd) if vv else "-")
            print(f"| {g} | " + " | ".join(row) + " |")
        print()
    print("## 곡선: 높이 층별 접선 속력 m/s (층 [0,2)·[2,4)·[4,8) cm, 무리 평균)\n")
    print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
    print("|---|" + "---|" * len(FR))

    def lay(get):
        o = []
        for f in FR:
            v = get(f)
            o.append("-" if not v else "·".join("-" if x is None else f"{x:.2f}" for x in v[:3]))
        return o
    print("| 정답 | " + " | ".join(lay(lambda f: GT[str(f)].get("vt_layers"))) + " |")
    for g, ns in GR.items():
        def gm(f, ns=ns):
            L = [an(n, f, "vt_layers") for n in ns]
            L = [x for x in L if x]
            if not L:
                return None
            return [None if any(q[i] is None for q in L) else sum(q[i] for q in L) / len(L) for i in range(3)]
        print(f"| {g} | " + " | ".join(lay(gm)) + " |")
    print()
if os.path.isfile(os.path.join(HERE, "wolfh3_tabreg.py")):
    sys.stdout.flush()
    os.system(f"{sys.executable} {os.path.join(HERE, 'wolfh3_tabreg.py')} {O}")


# ---- 절 CM66 진단 (--w3dg): 높이 칸별 시작점·걸음 끝 실현 몫 (w3dg.json), 무리 평균
def w3(n):
    return jl(n, "w3dg.json")


ED = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,4)", "[4,8)", "[8,16)", "[16,32)", "[32,∞)"]
WIN = [(10, 19), (20, 29), (30, 39), (40, 49)]
DG = {g: [n for n in ns if w3(n)] for g, ns in GR.items()}
DG = {g: ns for g, ns in DG.items() if ns}
if DG:
    print("## 걸음별 진단 (--w3dg): 높이 칸 (걸음 시작 바닥 높이 d0, cm) 별 관성 예측 대비 실현 몫\n")
    print("정의: 법선 λn = Σm (d − d0) / Σm (d_ŷ − d0) (1 = 관성 예측만큼, 0 = 걸음 시작에 묶임), 접선 λt = Σm (t − t0)·(t_ŷ − t0) / Σm |t_ŷ − t0|². "
          "S = 뉴턴 시작점 (섞기 뒤), E = 걸음 끝. 묶임 % = 27 노드 실효 k ≥ 20 (β ≤ 2^-20) 인 점 수 몫, eb 올림 % = 지금 --y0eb 규칙이 실현 가능 k 보다 k 를 올리는 점 몫 (시작점, 활성 점). "
          "프레임 구간 합 (걸음 모두), 무리 판 합.\n")
    for (a, b) in WIN:
        print(f"### f{a}~f{b}\n")
        print("| 무리 | 값 | " + " | ".join(ED) + " |")
        print("|---|---|" + "---|" * len(ED))
        for g, ns in DG.items():
            S = [[0.0] * 12 for _ in range(8)]; E = [[0.0] * 12 for _ in range(8)]
            for n in ns:
                for fr in w3(n)["frames"]:
                    if a <= fr["f"] <= b:
                        for i in range(8):
                            for j in range(12):
                                S[i][j] += fr["S"][i][j]; E[i][j] += fr["E"][i][j]

            def rt(x, y):
                return "-" if abs(y) < 1e-30 else f"{x / y:.2f}"
            print(f"| {g} | λn S·E | " + " | ".join(f"{rt(S[i][2], S[i][3])}·{rt(E[i][2], E[i][3])}" for i in range(8)) + " |")
            print(f"| {g} | λt S·E | " + " | ".join(f"{rt(S[i][4], S[i][5])}·{rt(E[i][4], E[i][5])}" for i in range(8)) + " |")
            print(f"| {g} | 묶임 % · eb 올림 % | " + " | ".join("-" if S[i][0] == 0 else f"{100 * S[i][8] / S[i][0]:.0f}·{100 * S[i][6] / S[i][0]:.0f}" for i in range(8)) + " |")
            print(f"| {g} | 점 수 (걸음 평균) | " + " | ".join(f"{S[i][0] / max(1, sum(1 for n in ns for fr in w3(n)['frames'] if a <= fr['f'] <= b and fr['rows'][1] > fr['rows'][0])):.0f}" for i in range(8)) + " |")
        print()
    print("### Y5 기록 (w3info.json)\n")
    print("| 판 | Y5 올린 점 (누적) | Y5 β 0 점 | Y5 판정 점 | m_q 합 · m_n 합 | 진단 행 |")
    print("|---|---|---|---|---|---|")
    for g, ns in DG.items():
        for n in ns:
            i = jl(n, "w3info.json") or {}
            print(f"| {n} | {i.get('y5_raised', '-')} | {i.get('y5_beta0', '-')} | {i.get('y5_pts', '-')} | {fm(i.get('mq_sum'), 3)} · {fm(i.get('mn_sum'), 3)} | {i.get('dg_rows', '-')} |")
    print()
