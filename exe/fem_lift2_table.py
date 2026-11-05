"""절 CM11.2 표 (lift2): python exe/fem_lift2_table.py <CM11 l_wolf (1h, (B))> <l2_wolf (2h, --bar_cross)> <l3_wolf (3h, --bar_cross)> <barx 디렉토리 (x_wolf·d_wolf·o_wolf)> <정답 wolf.pt>
형식은 exe/fem_lift_table.py (절 CM11) 그대로: 높이 = 바닥 법선 (z) 방향, 질량 중심 z 는 질량 가중 (실행기 --hgt), −Δ = 그 판의 띄운 거리 뺀 값,
퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (입자 수 기준), 장벽 걸린 점 = --bar_cnt neng, 그중 관통 칸 비율 = engcr / neng.
안 띄움 = 절 CM10 barx 재사용 (높이·덤프 x_wolf, 장벽 걸린 점 d_wolf, 오차 o_wolf)."""
import json
import os
import sys

import numpy as np
import torch

L1, L2, L3, BX, GTP = sys.argv[1:6]


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def spread(x):
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


def pred(d):
    p = os.path.join(d, "pred.pt")
    return torch.load(p, map_location="cpu", weights_only=False)["pred"] if os.path.exists(p) else None


NM = ["1h (B)", "2h bar_cross", "3h bar_cross"]
DS = [L1, L2, L3]
J = [jl(d) for d in DS]
S = [sm(d) for d in DS]
sx, sd, so = jl(os.path.join(BX, "x_wolf")), jl(os.path.join(BX, "d_wolf")), sm(os.path.join(BX, "o_wolf"))
print("# 절 CM11.2 lift2 -- wolf (B) 를 2h·3h 띄우고 관통 칸 장벽 (--bar_cross) 으로 실행, 절 CM11 1h (B) (--bar_cross 없음) 와 비교")
print()
for nm, s, j in zip(NM, S, J):
    if s is None or s.get("lift") is None:
        print(f"- {nm}: 결과 없음 (도는 중 또는 실패)" + (f" -- fem.jsonl {len(j)} 프레임" if j else ""))
        continue
    L = s["lift"]
    e = s["err_mean"] if s.get("err_mean") is not None else s["err_mean_done"]
    print(f"- {nm}: 띄운 거리 칸 h {1000 * L['h']:.2f} mm x {L['n']} = **{1000 * L['dist']:.2f} mm**; 처음 관통 칸 {L['orig']['cross']} -> {L['tried'][-1]['cross']} (실행 프레임 1 bc {j[0]['bc']['ccr'] if j and 'bc' in j[0] else '?'}),"
          f" 입자 최저 평면 거리 {1000 * L['tried'][-1]['dmin_part']:.2f} mm, 칸 꼭짓점 최저 {1000 * L['tried'][-1]['dmin_corner']:.2f} mm;"
          f" {s['frames_done']}/{s['frames']} 프레임{' 발산 ' + str(s['diverged']) if s.get('diverged') else ' 발산 없음'}, 오차 {100 * e:.3f} %, 띄운 거리 빼고 {100 * L['err_lift_mean_done']:.3f} %, 끝 해시 {s.get('x_sha1')}")
print(f"- 안 띄움 (barx o_wolf): 오차 {100 * so['err_mean']:.3f} %, 끝 해시 {so.get('x_sha1')}" if so else "- 안 띄운 판 요약 없음")
print()
if not all(s and s.get("lift") for s in S):
    sys.exit(0)
DZ = [s["lift"]["dist"] for s in S]
G = torch.load(GTP, map_location="cpu", weights_only=False)["x"]
P0 = pred(os.path.join(BX, "x_wolf"))
PP = [pred(d) for d in DS]
f = lambda v, k=1000: ("%.1f" % (k * v)) if v is not None and np.isfinite(v) else "-"
pc = lambda e: "%.2f" % (100 * e) if e is not None else "-"


def bar(b):
    n = b.get("neng")
    if n is None:
        return "-"
    return f"{n} ({100 * b['engcr'] / n:.0f}%)" if n else "0"


rows = []
for t in range(min(len(j) for j in J)):
    g = G[t + 1].numpy()
    rx = sx[t] if t < len(sx) else {}
    rd = sd[t] if t < len(sd) else {}
    r = [j[t] for j in J]
    sp = lambda P: spread(P[t].numpy()) if P is not None and t < P.shape[0] else float("nan")
    rows.append(dict(fr=t + 1, hg=r[0]["hg"], h0=rx.get("hp"), h=[q["hp"] for q in r], w=[spread(g), sp(P0)] + [sp(P) for P in PP],
                     b=[rd.get("bc", {})] + [q.get("bc", {}) for q in r], cr=[rd.get("bc", {}).get("ccr", "-")] + [q.get("bc", {}).get("ccr", "-") for q in r],
                     e0=rx.get("err"), e=[q.get("err") for q in r], el=[q.get("err_lift") for q in r]))
for title, sel in (("1. 요약 (프레임 10·20·30·40·50)", lambda R: R["fr"] % 10 == 0), ("2. 프레임별", lambda R: True)):
    print(f"## {title}")
    print()
    for k, kn in ((0, "질량중심 z"), (1, "최대 z")):
        print(f"### {kn} (mm; −Δ = 띄운 거리 뺀 값, " + " · ".join(f"{n.split()[0]} −{1000 * dz:.1f}" for n, dz in zip(NM, DZ)) + ")")
        print()
        print("| 프레임 | 정답 | 안 띄움 | 1h (B) | 1h−Δ | 2h bx | 2h−Δ | 3h bx | 3h−Δ |")
        print("|---|---|---|---|---|---|---|---|---|")
        for R in filter(sel, rows):
            print(f"| {R['fr']} | {f(R['hg'][k])} | {f(R['h0'][k] if R['h0'] else None)} | " + " | ".join(f"{f(h[k])} | **{f(h[k] - dz)}**" for h, dz in zip(R["h"], DZ)) + " |")
        print()
    print("### 퍼짐 폭 (mm) · 장벽 걸린 표본점 (--bar_cnt neng, 괄호 = 그중 관통 칸 점 비율 engcr/neng) · 관통 칸 수 (ccr)")
    print()
    print("| 프레임 | 폭 정답 | 안 띄움 | 1h (B) | 2h bx | 3h bx | 걸린 점 안 띄움 | 1h (B) | 2h bx | 3h bx | 관통 칸 안 띄움 | 1h | 2h | 3h |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for R in filter(sel, rows):
        print(f"| {R['fr']} | " + " | ".join(f(w) for w in R["w"]) + " | " + " | ".join(bar(b) for b in R["b"]) + " | " + " | ".join(str(c) for c in R["cr"]) + " |")
    print()
    print("### 오차 % (정답 그대로 / 띄운 거리 뺀 값)")
    print()
    print("| 프레임 | 안 띄움 | 1h (B) | 1h−Δ | 2h bx | 2h−Δ | 3h bx | 3h−Δ |")
    print("|---|---|---|---|---|---|---|---|")
    for R in filter(sel, rows):
        print(f"| {R['fr']} | {pc(R['e0'])} | " + " | ".join(f"{pc(e)} | {pc(el)}" for e, el in zip(R["e"], R["el"])) + " |")
    print()
tot = [(sum(r["b"][i].get("neng", 0) for r in rows), sum(r["b"][i].get("engcr", 0) for r in rows)) for i in range(4)]
print("## 3. 장벽 걸린 표본점 합 (프레임 합, 그중 관통 칸)")
print()
print(" · ".join(f"{n} {a} ({b})" for n, (a, b) in zip(["안 띄움"] + NM, tot)))
