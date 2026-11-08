"""절 CM44 (2) 첫 접촉 진단 (읽기만): 정답·우리 판의 높이 층별 눌림·logJp 변화 (f3 기준).
python exe/wmshape3_prof.py <결과 디렉토리> <gt_jp 디렉토리> <wm.pt> <판 이름,...> > 표.md
층 = f3 (첫 접촉 앞) 에 바닥 (z 0.02) 위 높이 h3 로 5 cm 구간. 칸마다: 층 높이 비 = 중앙 (h_t / h3) (1 = 안 눌림), Δ logJp 평균 (t − f3),
다져짐 몫 (logJp < −0.2, 절 CM40 정의), 손상 몫 (logJp > 0), 수평 반경 비 = 중앙 (r_t / r3) (r = 처음 질량중심 축 거리).
우리 판 = <판>/wsh/xs_*.npz (x·logJp float16, 정답과 같은 입자 번호)."""
import json
import os
import sys

import numpy as np

O, JP, WM, NS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
FL = 0.02
TS = (4, 5, 6, 8)
EDG = np.arange(0.0, 0.80, 0.05)


def ld_run(n, t):
    p = os.path.join(O, n, "wsh", f"xs_{t:03d}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["x"].astype(np.float64), z["lj"].astype(np.float64)


import torch
GX = torch.load(WM, map_location="cpu", weights_only=False)["x"]


def ld_gt(t):
    return GX[t].numpy().astype(np.float64), np.load(os.path.join(JP, "sc_%04d.npy" % t)).astype(np.float64)


def prof(get):
    x3, l3 = get(3)
    h3 = x3[:, 2] - FL
    c0 = x3[:, :2].mean(0)
    r3 = np.linalg.norm(x3[:, :2] - c0, axis=1)
    b = np.digitize(h3, EDG) - 1
    out = {}
    for t in TS:
        z = get(t)
        if z is None:
            continue
        x, l = z
        ok = np.isfinite(x).all(1) & np.isfinite(l)
        h = x[:, 2] - FL
        r = np.linalg.norm(x[:, :2] - c0, axis=1)
        rows = []
        for k in range(len(EDG) - 1):
            m = ok & (b == k) & (h3 > 0.01)
            if m.sum() < 50:
                rows.append(None)
                continue
            rows.append(dict(n=int(m.sum()), hr=float(np.median(h[m] / h3[m])), dl=float((l[m] - l3[m]).mean()),
                             cp=float((l[m] < -0.2).mean()), dm=float((l[m] > 0).mean()), rr=float(np.median(r[m] / np.maximum(r3[m], 1e-3)))))
        out[t] = rows
    return out


R = {"정답": prof(ld_gt)}
for n in NS:
    if ld_run(n, 3) is not None:
        R[n] = prof(lambda t, n=n: ld_run(n, t))
print("# 절 CM44 (2) 첫 접촉 층별 진단 (f3 기준, 5 cm 층; hr = 층 높이 비 중앙, Δlj = logJp 변화 평균, 다져짐·손상 몫 %, rr = 수평 반경 비 중앙)\n")
for t in TS:
    print(f"## f{t}\n")
    print("| 층 (f3 바닥 위 cm) | " + " | ".join(f"{k} hr · Δlj · 다져짐 % · 손상 % · rr" for k in R) + " |")
    print("|---|" + "---|" * len(R))
    for k in range(len(EDG) - 1):
        cells = []
        for nm, P in R.items():
            q = (P.get(t) or [None] * 99)[k]
            cells.append("-" if q is None else f"{q['hr']:.3f} · {q['dl']:+.3f} · {100 * q['cp']:.1f} · {100 * q['dm']:.1f} · {q['rr']:.3f}")
        print(f"| {100 * EDG[k]:.0f}~{100 * EDG[k + 1]:.0f} | " + " | ".join(cells) + " |")
    print()
json.dump({k: {str(t): v for t, v in P.items()} for k, P in R.items()}, open(os.path.join(O, "wmshape3_prof.json"), "w"))
