"""절 CM15 표 (vres): python exe/fem_vres_table.py <묶음 디렉토리 vres> <(A) a_wolf 디렉토리 (절 CM12)> <(B) x_wolf 디렉토리 (절 CM10)> <정답 wolf.pt> <wolf config.json>
판: o_<씬> = (B) 그대로 (이 인스턴스, 새 코드 플래그 없음 + --pen 진단), vs = −z 저항 강 (--fl_vres 9), vw = 약 (--fl_vres 0.9).
높이 = 바닥 법선 (z), 질량 중심 z 는 질량 가중 (실행기 --hgt), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (exe/fem_barnode_table.py 와 같은 정의).
바닥 아래 = 바닥 평면 부호 거리 < 0 인 입자 수와 최대 깊이 (실행기 --pen: 우리 np·dmax, 정답 gnp·gdmax; 저항 판은 걸음 시작 위치가 평면 아래인 표본점 수 nq, 마지막 걸음).
(A)·(B) wolf (앞 묶음 재사용) 의 바닥 아래는 pred.pt 덤프 입자 위치에서 같은 정의로 잰다."""
import json
import os
import re
import sys

import numpy as np

B, AW, BW_, GTP, CFG = sys.argv[1:6]
VS = ["o", "vs", "vw"]
VN = {"o": "(B) 장벽 (표본점)", "vs": "저항 강 k 9", "vw": "저항 약 k 0.9", "A": "(A) 가우스 점 고착"}
SCN = ("L000", "0016", "drop1", "wolf")


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def state(d):
    s = sm(d)
    if s is None:
        ri = os.path.join(d, "run_info.txt")
        t = open(ri).read() if os.path.exists(ri) else ""
        return ("실패 (" + t.split("rc=")[-1].split()[0] + ")") if "rc=" in t else ("없음" if not t else "도는 중")
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    return ("**x%d** (발산 전 %.2f)" % (dv, e)) if dv else "%.2f" % e


def drop(d):
    q = os.path.join(d, "drop_ana.txt")
    if not os.path.exists(q):
        return ""
    m = re.search(r"\n[^\n]*?:[^\n]*반발 계수 ([0-9.\-]+), 튀어 오름 ([0-9.\-]+) mm", "\n" + "\n".join(l for l in open(q).read().splitlines() if not l.startswith("mpm:")))
    return (" / 반발 %s · 튀어 오름 %s mm" % m.groups()) if m else ""


H = lambda d: (sm(d) or {}).get("x_sha1", "") or "-"
P = lambda v, t: os.path.join(B, f"{v}_{t}")
print("# 절 CM15 vres -- 채택 판 (B) 의 바닥 장벽·띠 고착 대신 −z 저항 바닥 (--fl_vres k, c = k / (τ² thinv²)), 결정적")
print()
print("## 1. 판정 (칸 = 평균 오차 % 또는 x발산 프레임; 낙하 정답 반발 0.750 · 튀어 오름 83.4 mm)")
print()
print("| 씬 | " + " | ".join(VN[v] for v in VS) + " | 해시 " + " / ".join(VS) + " |")
print("|---|" + "---|" * (len(VS) + 1))
for t in SCN:
    print(f"| {t} | " + " | ".join(state(P(v, t)) + (drop(P(v, t)) if t == "drop1" else "") for v in VS) + " | " + " / ".join(H(P(v, t)) for v in VS) + " |")
print(f"| wolf (A) (절 CM12 a_wolf) | {state(AW)} | | | {H(AW)} |")
print(f"| wolf (B) (절 CM10 x_wolf) | {state(BW_)} | | | {H(BW_)} |")
print()
print("- 끄면 같은지: o 해시를 앞 인스턴스 값과 대조 -- wolf 9b119d8fd3, L000 c43a42c7d0, 0016 f89f6fb659, drop1 de2e854bf7 (절 CM14.2; --pen 은 풀이 불변).")
print()


def mm(v, fmt="%.1f"):
    v = [q for q in v if q is not None]
    return ((fmt + " / " + fmt) % (np.mean(v), max(v))) if v else "-"


def itr(J):
    v = [q["it"] for q in J if isinstance(q.get("it"), dict) and "nN" in q["it"]]
    return ("%.1f · %.1f" % (np.mean([q["nN"] for q in v]), np.mean([q["nP"] for q in v]))) if v else "-"


print("## 2. 바닥 아래 (프레임 평균 / 최대; 입자 수, 최대 침투 깊이 mm, 저항 판 표본점 수 = 걸음 시작 위치가 평면 아래인 표본점, 마지막 걸음)")
print()
print("| 씬 | 판 | 아래 입자 수 | 최대 깊이 mm | 아래 표본점 수 | 정답 아래 입자 수 | 정답 최대 깊이 mm | 프레임당 뉴턴 · PCG |")
print("|---|---|---|---|---|---|---|---|")
for t in SCN:
    for v in VS:
        J = jl(P(v, t))
        pe = [q["pen"] for q in J if "pen" in q]
        if not pe:
            continue
        print(f"| {t} | {VN[v]} | {mm([p['np'] for p in pe])} | {mm([1000 * p['dmax'] for p in pe if p['dmax'] is not None], '%.2f')} | {mm([p.get('nq') for p in pe]) if v != 'o' else '-'} | "
              f"{mm([p['gnp'] for p in pe])} | {mm([1000 * p['gdmax'] for p in pe], '%.2f')} | {itr(J)} |")
print()
print("## 3. 프레임별 바닥 아래 (lego·낙하; 칸 = 아래 입자 수 / 최대 깊이 mm, 저항 판은 + 표본점 수)")
for t in ("L000", "0016", "drop1"):
    JS = {v: jl(P(v, t)) for v in VS}
    n = max([len(J) for J in JS.values()] + [0])
    if not n:
        continue
    print()
    print(f"### {t}")
    print()
    print("| 프레임 | 정답 | " + " | ".join(VN[v] for v in VS) + " |")
    print("|---|---|" + "---|" * len(VS))
    for i in range(n):
        cells, g = [], "-"
        for v in VS:
            J = JS[v]
            p = J[i].get("pen") if i < len(J) else None
            if p is None:
                cells.append("-"); continue
            g = f"{p['gnp']} / {1000 * p['gdmax']:.2f}"
            cells.append(f"{p['np']} / {1000 * p['dmax']:.2f}" + (f" · {p['nq']}" if 'nq' in p else "") if p['dmax'] is not None else "비유한")
        print(f"| {i + 1} | {g} | " + " | ".join(cells) + " |")
print()
WD = [("A", AW), ("o", BW_)] + [(v, P(v, "wolf")) for v in VS[1:]]
JS = {k: jl(d) for k, d in WD}
JO = jl(P("o", "wolf"))
ref = next((J for J in [JO] + list(JS.values()) if J and "hg" in J[0]), None)
if ref is None:
    sys.exit(0)
cfg = json.load(open(CFG))
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = np.array(pl["normal"], dtype=np.float64); NR /= np.linalg.norm(NR); PN = float(np.dot(np.array(pl["point"], dtype=np.float64), NR))
try:
    import torch

    def ld(p):
        return torch.load(p, map_location="cpu", weights_only=False)
    G = ld(GTP)["x"]
    PR = {k: (ld(os.path.join(d, "pred.pt"))["pred"] if os.path.exists(os.path.join(d, "pred.pt")) else None) for k, d in WD}
except Exception as ex:                                                              # torch 없는 곳에서는 폭·덤프 바닥 아래 칸 비움
    print(f"(퍼짐 폭 생략: {ex})")
    G, PR = None, {k: None for k, _ in WD}


def spread(x):
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


def pen_of(k, t):
    J = JS[k]
    if t < len(J) and "pen" in J[t]:
        p = J[t]["pen"]
        return p["np"], p["dmax"]
    Pk = PR.get(k)
    if Pk is None or t >= Pk.shape[0]:
        return None, None
    x = Pk[t].numpy().astype(np.float64)
    if not (np.isfinite(x).all() and np.abs(x).max() > 0):
        return None, None
    h = x @ NR - PN
    return int((h < 0).sum()), float(max(0.0, -h.min()))


f = lambda v: ("%.1f" % (1000 * v)) if v is not None and np.isfinite(v) else "-"
pc = lambda e: "%.2f" % (100 * e) if e is not None else "-"
K = [k for k, _ in WD]
n = max(len(J) for J in JS.values())
rows = []
for t in range(n):
    g = lambda J, k: (J[t]["hp"][k] if t < len(J) and "hp" in J[t] else None)
    def sp(k):
        Pk = PR.get(k)
        if Pk is None or t >= Pk.shape[0]:
            return float("nan")
        x = Pk[t].numpy()
        return spread(x) if np.isfinite(x).all() and np.abs(x).max() > 0 else float("nan")
    hg = ref[t]["hg"] if t < len(ref) else [float("nan")] * 3
    gp = ref[t].get("pen") if t < len(ref) else None
    pens = [pen_of(k, t) for k in K]
    rows.append(dict(fr=t + 1, hg=hg, c=[g(JS[k], 0) for k in K], m=[g(JS[k], 1) for k in K],
                     w=[spread(G[t + 1].numpy()) if G is not None and t + 1 < G.shape[0] else float("nan")] + [sp(k) for k in K],
                     e=[JS[k][t].get("err") if t < len(JS[k]) else None for k in K],
                     pn=[(f"{gp['gnp']} / {1000 * gp['gdmax']:.2f}" if gp else "-")] + [("-" if a is None else f"{a} / {1000 * b:.2f}") for a, b in pens]))
hd = " | ".join(VN[k] for k in K)
for title, sel in (("4. wolf 곡선 요약 (프레임 10·20·30·40·50; mm, 바닥 아래 = 입자 수 / 최대 깊이 mm)", lambda R: R["fr"] % 10 == 0), ("5. wolf 곡선 프레임별 (mm)", lambda R: True)):
    print(f"## {title}")
    print()
    print(f"| 프레임 | 질량중심 z 정답 | {hd} | 최대 z 정답 | {hd} | 퍼짐 폭 정답 | {hd} | 오차 % {hd} | 바닥 아래 정답 | {hd} |")
    print("|---|" + "---|" * (5 * len(K) + 4))
    for R in filter(sel, rows):
        print(f"| {R['fr']} | {f(R['hg'][0])} | " + " | ".join(f(v) for v in R["c"]) + f" | {f(R['hg'][1])} | " + " | ".join(f(v) for v in R["m"]) + " | "
              + " | ".join(f(v) for v in R["w"]) + " | " + " | ".join(pc(e) for e in R["e"]) + " | " + " | ".join(R["pn"]) + " |")
    print()
