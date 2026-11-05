"""절 CM11 표: python exe/fem_lift_table.py <띄운 판 디렉토리 l_wolf> <barx 디렉토리 (x_wolf·d_wolf·o_wolf)> <정답 wolf.pt>
안 띄운 (B) = 절 CM10 barx 재사용: 높이·덤프 = x_wolf (--bar_cross 판이지만 끝 해시 = o_wolf, 비트 같음), 장벽 걸린 점·관통 칸 = d_wolf (--bar_cnt, 결과 불변), 오차 = o_wolf.
높이 = 바닥 법선 (z) 방향, 질량 중심 z 는 질량 가중 (실행기 --hgt), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (입자 수 기준)."""
import json
import os
import sys

import numpy as np
import torch

LD, BX, GTP = sys.argv[1:4]


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def spread(x):
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


l1, sx, sd, so = jl(LD), jl(os.path.join(BX, "x_wolf")), jl(os.path.join(BX, "d_wolf")), sm(os.path.join(BX, "o_wolf"))
s1 = sm(LD)
lift = (s1 or {}).get("lift")
print("# 절 CM11 lift -- wolf 채택 판 (B) (절 CM10 의 (B) 명령 그대로, --bar_cross 없음) 를 처음 관통 칸 0 이 되게 바닥 법선 방향으로 띄워 실행")
print()
if lift is None:
    print("띄운 판 결과 없음 (도는 중 또는 실패)" + (f" -- fem.jsonl {len(l1)} 프레임" if l1 else ""))
    sys.exit(0)
dz = lift["dist"]
print(f"- 띄운 거리: 칸 h {1000 * lift['h']:.2f} mm x {lift['n']} = **{1000 * dz:.2f} mm** (법선 {lift['normal']}); 시도 n·처음 관통 칸 = {[(q['n'], q['cross']) for q in lift['tried']]}")
print(f"- 처음 상태 (U = 0) 관통 칸: 안 띄움 {lift['orig']['cross']} / 점유 칸 {lift['orig']['occ']} (실행 barx d_wolf 프레임 1 bc 관통 칸 {sd[0]['bc']['ccr'] if sd else '?'}) -> 띄움 **{lift['tried'][-1]['cross']}** / {lift['tried'][-1]['occ']}"
      f" (실행 프레임 1 bc 관통 칸 {l1[0]['bc']['ccr'] if l1 and 'bc' in l1[0] else '?'}); 입자 최저 평면 거리 안 띄움 {1000 * lift['orig']['dmin_part']:.2f} mm -> {1000 * lift['tried'][-1]['dmin_part']:.2f} mm, 칸 꼭짓점 최저 {1000 * lift['orig']['dmin_corner']:.2f} -> {1000 * lift['tried'][-1]['dmin_corner']:.2f} mm")
e1 = s1["err_mean"] if s1.get("err_mean") is not None else s1["err_mean_done"]
print(f"- 띄운 판: {s1['frames_done']}/{s1['frames']} 프레임{' 발산 ' + str(s1['diverged']) if s1.get('diverged') else ''}, 오차 (정답 그대로 대비) {100 * e1:.3f} %, 띄운 거리 빼고 {100 * lift['err_lift_mean_done']:.3f} % (참고), 끝 해시 {s1.get('x_sha1')}")
print(f"- 안 띄운 판 (barx o_wolf): 오차 {100 * so['err_mean']:.3f} %, 끝 해시 {so.get('x_sha1')}" if so else "- 안 띄운 판 요약 없음")
print()
d = torch.load(GTP, map_location="cpu", weights_only=False)
G = d["x"]
P0 = P1 = None
p0, p1 = os.path.join(BX, "x_wolf", "pred.pt"), os.path.join(LD, "pred.pt")
if os.path.exists(p0):
    P0 = torch.load(p0, map_location="cpu", weights_only=False)["pred"]
if os.path.exists(p1):
    P1 = torch.load(p1, map_location="cpu", weights_only=False)["pred"]
rows = []
for i, r1 in enumerate(l1):
    t = r1["t"]
    rx = sx[t] if t < len(sx) else {}
    rd = sd[t] if t < len(sd) else {}
    hg, h0, h1 = rx.get("hg") or r1.get("hg"), rx.get("hp"), r1.get("hp")
    g = G[t + 1].numpy()
    w = [spread(g), spread(P0[t].numpy()) if P0 is not None and t < P0.shape[0] else float("nan"), spread(P1[t].numpy()) if P1 is not None and t < P1.shape[0] else float("nan")]
    b0, b1 = rd.get("bc", {}), r1.get("bc", {})
    rows.append((t + 1, hg, h0, h1, w, b0, b1, rx.get("err"), r1.get("err"), r1.get("err_lift")))
f = lambda v, k=1000: ("%.1f" % (k * v)) if v is not None and np.isfinite(v) else "-"
print("## 1. 요약 (프레임 10·20·30·40·50; 높이·폭 mm, 띄움−Δ = 띄운 판 값에서 띄운 거리 뺀 값)")
print()
print("| 프레임 | 질량중심 z 정답 | 안 띄움 | 띄움 | 띄움−Δ | 최대 z 정답 | 안 띄움 | 띄움 | 띄움−Δ | 퍼짐 폭 정답 | 안 띄움 | 띄움 | 장벽 걸린 점 안 띄움 / 띄움 | 오차 % 안 띄움 / 띄움 / 띄움−Δ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for R in rows:
    if R[0] % 10:
        continue
    fr, hg, h0, h1, w, b0, b1, e0_, e1_, el = R
    print(f"| {fr} | {f(hg[0])} | {f(h0[0] if h0 else None)} | {f(h1[0])} | **{f(h1[0] - dz)}** | {f(hg[1])} | {f(h0[1] if h0 else None)} | {f(h1[1])} | **{f(h1[1] - dz)}** | {f(w[0])} | {f(w[1])} | **{f(w[2])}** | {b0.get('neng', '-')} / {b1.get('neng', '-')} | {'%.2f' % (100 * e0_) if e0_ is not None else '-'} / {'%.2f' % (100 * e1_) if e1_ is not None else '-'} / {'%.2f' % (100 * el) if el is not None else '-'} |")
print()
print("## 2. 프레임별 (높이·폭 mm; 장벽 걸린 점 = 그 프레임 어느 걸음·반복에서든 활성이고 d < d̂ 인 표본점, 관통 칸 = 어느 걸음 시작에서든 관통인 칸)")
print()
print("| 프레임 | 질량중심 z 정답 | 안 띄움 | 띄움 | 띄움−Δ | 최대 z 정답 | 안 띄움 | 띄움 | 띄움−Δ | 퍼짐 폭 정답 | 안 띄움 | 띄움 | 걸린 점 안 띄움 | 띄움 | 관통 칸 안 띄움 | 띄움 | 오차 % 안 띄움 | 띄움 | 띄움−Δ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for fr, hg, h0, h1, w, b0, b1, e0_, e1_, el in rows:
    print(f"| {fr} | {f(hg[0])} | {f(h0[0] if h0 else None)} | {f(h1[0])} | {f(h1[0] - dz)} | {f(hg[1])} | {f(h0[1] if h0 else None)} | {f(h1[1])} | {f(h1[1] - dz)} | {f(w[0])} | {f(w[1])} | {f(w[2])} | {b0.get('neng', '-')} | {b1.get('neng', '-')} | {b0.get('ccr', '-')} | {b1.get('ccr', '-')} | {'%.3f' % (100 * e0_) if e0_ is not None else '-'} | {'%.3f' % (100 * e1_) if e1_ is not None else '-'} | {'%.3f' % (100 * el) if el is not None else '-'} |")
