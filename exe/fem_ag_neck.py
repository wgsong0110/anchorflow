"""절 AG.18 bread 진단: 목 (necking) 이 생기는 자리 -- 정답·우리의 초기 y (라그랑주) 구간별 늘어남과 옆 수축.
구간 = 프레임 0 의 y 를 등분 (구속 영역 포함). 늘어남 = 이웃 구간 y 중앙값 간격 / 프레임 0 간격, 옆 수축 = xz 공분산 행렬식^(1/4) / 프레임 0.
사용: python -I exe/fem_ag_neck.py <우리 roll_fem.pt> <정답 .pt> <출력 json> [구간 수 30]"""
import json
import sys

import torch

R = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
G = torch.load(sys.argv[2], map_location="cpu", weights_only=False, mmap=True)
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 30
xo, xg = R["pred"], G["x"]
print("keys ours", list(R.keys())[:12], "gt", list(G.keys())[:12], flush=True)
x0 = xg[0].float()
y0 = x0[:, 1]
ed = torch.linspace(float(y0.min()), float(y0.max()) + 1e-6, NB + 1)
b = torch.bucketize(y0, ed) - 1
bc_hi = (y0 > 1.25 - 0.05) & (y0 < 1.25 + 0.05)
bc_lo = (y0 > 0.75 - 0.05) & (y0 < 0.75 + 0.05)
def prof(x):
    x = x.float()
    yc, lat, n = [], [], []
    for k in range(NB):
        m = b == k
        n.append(int(m.sum()))
        if int(m.sum()) < 10:
            yc.append(float("nan")); lat.append(float("nan")); continue
        yc.append(float(x[m, 1].median()))
        xz = x[m][:, [0, 2]]
        c = torch.cov(xz.T)
        lat.append(float(torch.det(c).clamp_min(0) ** 0.25))
    return yc, lat, n
yc0, lat0, n0 = prof(x0)
out = dict(edges=ed.tolist(), n=n0, bc_hi_bins=sorted(set(b[bc_hi].tolist())), bc_lo_bins=sorted(set(b[bc_lo].tolist())), lat0=lat0, frames=[])
T = xo.shape[0]
for t in list(range(19, T, 20)) + [T - 1]:
    r = dict(frame=t + 1)
    for nm, x in (("gt", xg[t + 1]), ("ours", xo[t])):
        yc, lat, _ = prof(x)
        st = [((yc[k + 1] - yc[k]) / (yc0[k + 1] - yc0[k])) if (yc0[k + 1] - yc0[k]) > 0 else float("nan") for k in range(NB - 1)]
        r[nm + "_stretch"] = [round(v, 3) for v in st]
        r[nm + "_lat"] = [round(l / l0, 3) if l0 > 0 else float("nan") for l, l0 in zip(lat, lat0)]
    out["frames"].append(r)
    print(f"프레임 {t + 1}\n 정답 늘어남 {r['gt_stretch']}\n 우리 늘어남 {r['ours_stretch']}\n 정답 옆 {r['gt_lat']}\n 우리 옆 {r['ours_lat']}", flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("bc 구간", out["bc_lo_bins"], out["bc_hi_bins"], "초기 옆 크기", [round(v, 4) for v in lat0], flush=True)
