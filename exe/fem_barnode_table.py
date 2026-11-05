"""절 CM14 표 (barnode): python exe/fem_barnode_table.py <묶음 디렉토리 barnode> <(A) a_wolf 디렉토리 (절 CM12)> <(B) x_wolf 디렉토리 (절 CM10)> <정답 wolf.pt>
판: o_<씬> = (B) 그대로 (이 인스턴스, 새 코드 플래그 없음), na5·na7 = (가) 표본점 장벽 + 노드 장벽 (d̂_n ½·¾ dx_정답), no5·no7 = (나) 노드 장벽만.
높이 = 바닥 법선 (z), 질량 중심 z 는 질량 가중 (실행기 --hgt), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (exe/fem_lift_table.py 와 같은 정의).
장벽 걸린 노드 = fem.jsonl bn.nneng (그 프레임 어느 뉴턴 반복에서든 활성이고 d_n < d̂_n), 표본점 = bc.neng (--bar_cnt)."""
import json
import os
import re
import sys

import numpy as np

B, AW, BW_, GTP = sys.argv[1:5]
VS = ["o", "na5", "na7", "no5", "no7"]
VN = {"o": "(B) 장벽 (표본점)", "na5": "(가) 표본점+노드 ½dx", "na7": "(가) 표본점+노드 ¾dx", "no5": "(나) 노드만 ½dx", "no7": "(나) 노드만 ¾dx", "A": "(A) 가우스 점 고착"}


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
print("# 절 CM14 barnode -- 채택 판 (B) (장벽 + --y0eb) 에 B-스플라인 노드 위치 장벽 (--bar_node), 결정적")
print()
print("## 1. 판정 (칸 = 평균 오차 % 또는 x발산 프레임; 낙하 정답 반발 0.750 · 튀어 오름 83.4 mm)")
print()
print("| 씬 | " + " | ".join(VN[v] for v in VS) + " | 해시 " + " / ".join(VS) + " |")
print("|---|" + "---|" * (len(VS) + 1))
for t in ("L000", "0016", "drop1", "wolf"):
    print(f"| {t} | " + " | ".join(state(P(v, t)) + (drop(P(v, t)) if t == "drop1" else "") for v in VS) + " | " + " / ".join(H(P(v, t)) for v in VS) + " |")
print(f"| wolf (A) (절 CM12 a_wolf) | {state(AW)} | | | | | {H(AW)} |")
print()
print("- 끄면 같은지: o 해시를 앞 인스턴스 값과 대조 -- barx o_wolf 9b119d8fd3, barstick o_L000 c43a42c7d0 · o_0016 f89f6fb659 · o_drop1 de2e854bf7 (다른 vast 기계면 NVRTC 환경 차로 다를 수 있음 -- 절 CM10.2).")
print()
print("## 2. 장벽 걸린 노드·표본점 (프레임 평균 / 최대; 노드 = bn.nneng, 부피 > 0 노드 = bn.nnv, 걸음 시작 d ≤ 0 이라 빠진 노드 = bn.nnbelow, 표본점 = bc.neng)")
print()
print("| 판 | 씬 | 걸린 노드 평균 / 최대 | 부피 > 0 노드 평균 | 빠진 노드 평균 / 최대 | 걸린 표본점 평균 / 최대 | 프레임당 뉴턴 · PCG |")
print("|---|---|---|---|---|---|---|")


def mm(J, k1, k2):
    v = [q[k1][k2] for q in J if k1 in q and k2 in q[k1]]
    return ("%.1f / %d" % (np.mean(v), max(v))) if v else "-"


def itr(J):
    v = [q["it"] for q in J if isinstance(q.get("it"), dict) and "nN" in q["it"]]
    return ("%.1f · %.1f" % (np.mean([q["nN"] for q in v]), np.mean([q["nP"] for q in v]))) if v else "-"


for t in ("L000", "0016", "drop1", "wolf"):
    for v in VS:
        J = jl(P(v, t))
        if not J:
            continue
        nv = [q["bn"]["nnv"] for q in J if "bn" in q]
        print(f"| {VN[v]} | {t} | {mm(J, 'bn', 'nneng')} | {('%.0f' % np.mean(nv)) if nv else '-'} | {mm(J, 'bn', 'nnbelow')} | {mm(J, 'bc', 'neng')} | {itr(J)} |")
print()
WD = [("A", AW), ("o", BW_)] + [(v, P(v, "wolf")) for v in VS[1:]]
JS = {k: jl(d) for k, d in WD}
JO = jl(P("o", "wolf"))
if not JS["o"] and JO:
    JS["o"] = JO
ref = next((J for J in JS.values() if J and "hg" in J[0]), None)
if ref is None:
    sys.exit(0)
try:
    import torch

    def ld(p):
        return torch.load(p, map_location="cpu", weights_only=False)
    G = ld(GTP)["x"]
    PR = {k: (ld(os.path.join(d, "pred.pt"))["pred"] if os.path.exists(os.path.join(d, "pred.pt")) else None) for k, d in WD}
except Exception as ex:                                                              # torch 없는 곳에서는 폭 칸 비움
    print(f"(퍼짐 폭 생략: {ex})")
    G, PR = None, {k: None for k, _ in WD}


def spread(x):
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


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
    rows.append(dict(fr=t + 1, hg=hg, c=[g(JS[k], 0) for k in K], m=[g(JS[k], 1) for k in K],
                     w=[spread(G[t + 1].numpy()) if G is not None and t + 1 < G.shape[0] else float("nan")] + [sp(k) for k in K],
                     e=[JS[k][t].get("err") if t < len(JS[k]) else None for k in K],
                     bn=[JS[k][t].get("bn", {}).get("nneng") if t < len(JS[k]) else None for k in K[2:]]))
hd = " | ".join(VN[k] for k in K)
for title, sel in (("3. wolf 곡선 요약 (프레임 10·20·30·40·50; mm)", lambda R: R["fr"] % 10 == 0), ("4. wolf 곡선 프레임별 (mm)", lambda R: True)):
    print(f"## {title}")
    print()
    print(f"| 프레임 | 질량중심 z 정답 | {hd} | 최대 z 정답 | {hd} | 퍼짐 폭 정답 | {hd} | 오차 % {hd} | 걸린 노드 " + " / ".join(K[2:]) + " |")
    print("|---|" + "---|" * (4 * len(K) + 4))
    for R in filter(sel, rows):
        print(f"| {R['fr']} | {f(R['hg'][0])} | " + " | ".join(f(v) for v in R["c"]) + f" | {f(R['hg'][1])} | " + " | ".join(f(v) for v in R["m"]) + " | "
              + " | ".join(f(v) for v in R["w"]) + " | " + " | ".join(pc(e) for e in R["e"]) + " | " + " / ".join("-" if b is None else str(b) for b in R["bn"]) + " |")
    print()
