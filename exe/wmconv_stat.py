"""절 CM90 묶음 wmconv: 덤프 판 손상 지표 (읽기만, CPU). python exe/wmconv_stat.py --run DIR --jp GT_JP_DIR -> DIR/wmconv_stat.json
- 손상 = logJp > 0 몫 (분모 = 그 프레임 위치가 유한한 입자; exe/wmdmg_stats.py 와 같은 정의), 프레임 1..L
- 손상 입자 J·Jp = 탄성 det F · exp(logJp + 0.04) 의 10·50·90 % 분위 (절 CM59 와 같은 정의) -- f5·f6·f8·끝
- 정답 IoU = 같은 입자 번호에서 손상 집합 (정답 = GF 다시 돌린 logJp gt_jp/sc_<t>.npy > 0) 교집합 / 합집합
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--jp", default="")
a = ap.parse_args()
lj = torch.load(f"{a.run}/lj.pt", map_location="cpu", weights_only=False)["logJp"].numpy()
D = torch.load(f"{a.run}/pred.pt", map_location="cpu", weights_only=False, mmap=True)
P, F = D["pred"], D["F"]
L, N = lj.shape
S = json.load(open(f"{a.run}/summary.json"))
R = dict(L=L, N=N, err=S.get("err_mean"), err_done=S.get("err_mean_done"), diverged=S.get("diverged"), dmg=[], fr={})
for i in range(L):
    fin = np.isfinite(P[i].numpy()).all(1)
    R["dmg"].append(float((lj[i][fin] > 0).sum()) / max(int(fin.sum()), 1))
for t in (4, 5, 6, 8, 10, 20, 50, L):
    if t > L:
        continue
    i = t - 1
    x = P[i].numpy()
    fin = np.isfinite(x).all(1)
    dm = fin & (lj[i] > 0)
    r = dict(dmg=R["dmg"][i], ndmg=int(dm.sum()))
    if dm.any():
        J = np.linalg.det(F[i].numpy()[dm].astype(np.float64))
        jj = J * np.exp(lj[i][dm].astype(np.float64) + 0.04)
        jj = jj[np.isfinite(jj)]
        r["JJp_q"] = [float(q) for q in np.percentile(jj, [10, 50, 90])] if jj.size else None
        r["J_q"] = [float(q) for q in np.percentile(J[np.isfinite(J)], [10, 50, 90])]
        r["z_med"] = float(np.median(x[dm, 2]))
    gp = f"{a.jp}/sc_{t:04d}.npy"
    if a.jp and os.path.exists(gp):
        g = np.load(gp)
        gd = np.isfinite(g) & (g > 0)
        r["gt_dmg_raw"] = float(gd.sum()) / max(int(np.isfinite(g).sum()), 1)
        u = int((gd | dm).sum())
        r["iou"] = float((gd & dm).sum()) / u if u else None
    R["fr"][str(t)] = r
json.dump(R, open(f"{a.run}/wmconv_stat.json", "w"), indent=1)
print("[wmconv 통계]", a.run, json.dumps({k: R["fr"][k] for k in R["fr"] if k in ("5", "6", "8", str(L))}))
