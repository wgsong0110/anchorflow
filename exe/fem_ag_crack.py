"""절 AG 파괴 진단: 고정 격자 메시가 갈라진 양쪽을 한 덩어리로 담는가 (GF 수박).

정답 궤적에서 '끊긴 짝' = 0 프레임 k-NN 이웃 (k 16) 중 프레임 t 거리가 0 프레임 거리의 thr 배를 넘은 짝.
우리 lag 판은 프레임마다 입자가 든 정육면체 (Kuhn 격자, 변 = node_h/√2) 의 사면체로 메시를 다시 짜고, 이웃 정육면체끼리는 노드를 나눠
그 프레임 동안 한 연속체로 움직인다 -> 끊긴 짝의 두 입자가 같은 정육면체이거나 이웃 정육면체 (지표 차 최대 1) 면 그 프레임 메시는 둘을 잇는다
(그 사이 틈을 표현하지 못한다). 정답 위치에 우리 격자를 깔아 그 비율을 잰다. 우리 롤아웃의 끊긴 짝 비율도 같은 짝·같은 문턱으로.

  python exe/fem_ag_crack.py --gt wm.pt --dump roll_fem.pt --node_h 0.0094281 --out crack.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--dump", default="")
ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--thr", type=float, nargs="+", default=[1.5, 2.0, 3.0])
ap.add_argument("--frames", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40, 60, 80, 100])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.gt, map_location="cpu", weights_only=False)
X = d["x"]
T = X.shape[0]
g = torch.Generator().manual_seed(a.seed)
x0 = X[0].to(dev).float()
ids = torch.randperm(x0.shape[0], generator=g)[:a.n].to(dev)
# 0 프레임 k-NN (전체 입자에서, 블록 거리)
nb = []
for s in range(0, ids.numel(), 2048):
    q = x0[ids[s:s + 2048]]
    best_d = torch.full((q.shape[0], a.k + 1), float("inf"), device=dev)
    best_i = torch.zeros((q.shape[0], a.k + 1), dtype=torch.long, device=dev)
    for t0 in range(0, x0.shape[0], 200000):
        dd = torch.cdist(q, x0[t0:t0 + 200000])
        cd, ci = torch.cat([best_d, dd], 1), torch.cat([best_i, torch.arange(t0, t0 + dd.shape[1], device=dev).expand(q.shape[0], -1)], 1)
        best_d, o = cd.topk(a.k + 1, dim=1, largest=False)
        best_i = torch.gather(ci, 1, o)
    nb.append(best_i[:, 1:])
nb = torch.cat(nb)                                                                   # [n, k]
pi = ids[:, None].expand_as(nb).reshape(-1)
pj = nb.reshape(-1)
d0 = (x0[pi] - x0[pj]).norm(dim=-1).clamp_min(1e-9)
lat = SX.lattice(float(a.node_h), SX.HZ_CUBE * float(a.node_h), device=dev, dtype=torch.float32)
Ai = lat.Ai
P = None
if a.dump:
    P = torch.load(a.dump, map_location="cpu", weights_only=False)["pred"]          # [L, N, 3] (프레임 1..L)
out = dict(node_h=a.node_h, cube_edge=float(lat.A[:, 0].norm()), n_pairs=int(pi.numel()), thr=a.thr, frames=[])
for t in a.frames:
    if t >= T:
        continue
    xt = X[t].to(dev).float()
    ok = torch.isfinite(xt[pi]).all(-1) & torch.isfinite(xt[pj]).all(-1)
    dt_ = (xt[pi] - xt[pj]).norm(dim=-1)
    ci = torch.floor(xt[pi] @ Ai.T)
    cj = torch.floor(xt[pj] @ Ai.T)
    cheb = (ci - cj).abs().max(-1).values
    rec = dict(t=t, n_ok=int(ok.sum()))
    for th in a.thr:
        br = ok & (dt_ > th * d0)
        nbr = int(br.sum())
        rec[f"gt_broken_{th}"] = nbr / max(int(ok.sum()), 1)
        rec[f"gt_broken_same_cube_{th}"] = float(((cheb == 0) & br).sum()) / max(nbr, 1)
        rec[f"gt_broken_mesh_connected_{th}"] = float(((cheb <= 1) & br).sum()) / max(nbr, 1)
        if P is not None and t - 1 < P.shape[0]:
            pt = P[t - 1].to(dev).float()
            dp = (pt[pi] - pt[pj]).norm(dim=-1)
            rec[f"ours_broken_{th}"] = float((ok & (dp > th * d0)).sum()) / max(int(ok.sum()), 1)
            rec[f"ours_broken_given_gt_broken_{th}"] = float((br & (dp > th * d0)).sum()) / max(nbr, 1)
    out["frames"].append(rec)
    print(f"[t {t}] " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[저장] {a.out}", flush=True)
