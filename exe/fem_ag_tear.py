"""절 AG.17 bread 진단: 정답이 찢어지는 곳에서 우리 판이 항복 (손상) 을 못 시키는지, 항복은 하는데 메시가 못 갈라지는지.
정답·우리 위치에서 입자 국소 늘어남 (0 프레임 k 최근접 이웃과의 거리 비, 중앙값) 을 프레임마다 재고, 정답 '찢어진' 입자 (늘어남 > 1.5) 집합에서
우리 판의 늘어남·손상 (al) · 탄성 F 최대 특이값을 비교. 사용: python -I exe/fem_ag_tear.py <우리 roll_fem.pt> <우리 pl_dump.pt> <정답 .pt> <출력 json>"""
import json
import sys

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
R = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
P = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["ys_lj_al"]
G = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
xo, xg = R["pred"], G["x"][1:R["pred"].shape[0] + 1]
x0 = G["x"][0].float().to(dev)
N = x0.shape[0]
K = 8
sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:40000].to(dev)       # 비교 입자 표본 (이웃은 전체에서)
nb = []
for c in sub.split(4000):
    d = torch.cdist(x0[c], x0)
    nb.append(d.topk(K + 1, largest=False).indices[:, 1:])
nb = torch.cat(nb)
d0 = (x0[sub][:, None] - x0[nb]).norm(dim=-1)
def stretch(x):
    x = x.float().to(dev)
    return ((x[sub][:, None] - x[nb]).norm(dim=-1) / d0.clamp_min(1e-9)).median(-1).values
out = []
Fo = R.get("F")
for t in range(9, xo.shape[0], 10):
    sg, so = stretch(xg[t]), stretch(xo[t])
    torn = sg > 1.5
    al = P[t, 2].float().to(dev)[sub]
    ys = P[t, 0].float().to(dev)[sub]
    sv = torch.linalg.svdvals(Fo[t].float().to(dev)[sub].view(-1, 3, 3)) if Fo is not None else None
    r = dict(frame=t + 1, gt_torn=int(torn.sum()), ours_torn=int((so > 1.5).sum()),
             both=int((torn & (so > 1.5)).sum()))
    if bool(torn.any()):
        r.update(ours_stretch_med_at_gt_torn=float(so[torn].median()), ours_damaged_frac_at_gt_torn=float((al[torn] <= 0).float().mean()),
                 ours_damaged_frac_elsewhere=float((al[~torn] <= 0).float().mean()), ours_ys_med_at_gt_torn=float(ys[torn].median()),
                 ours_ys_med_elsewhere=float(ys[~torn].median()),
                 gt_torn_y_q=[float(torch.quantile(xg[t].float().to(dev)[sub][torn][:, 1], q)) for q in (0.1, 0.5, 0.9)])
        if sv is not None:
            r.update(ours_Fe_smax_med_at_gt_torn=float(sv[torn][:, 0].median()), ours_Fe_smax_med_elsewhere=float(sv[~torn][:, 0].median()))
    ot = so > 1.5
    if bool(ot.any()):
        r["ours_torn_y_q"] = [float(torch.quantile(xo[t].float().to(dev)[sub][ot][:, 1], q)) for q in (0.1, 0.5, 0.9)]
    out.append(r)
    print(r, flush=True)
json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
