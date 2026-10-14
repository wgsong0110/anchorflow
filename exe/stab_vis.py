"""서브스텝 안정성 시각 지표: 같은 렌더러(rep_ip.py, F 반영 공분산)로 그린 PNG 프레임을 기준(PG N1024)과 비교.

PSNR, SSIM (skimage, 채널별), LPIPS (alex). 프레임 1..60 평균 (0 프레임은 모두 같은 초기 상태).

  python exe/stab_vis.py --ref DIR_ref --runs name=DIR ... --out vis.json
"""
import argparse
import glob
import json

import imageio.v2 as imageio
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
import lpips
from skimage.metrics import structural_similarity as ssim
LP = lpips.LPIPS(net="alex").cuda().eval()
R = sorted(glob.glob(f"{a.ref}/*.png"))
res = {}
for spec in a.runs:
    name, d = spec.split("=", 1)
    F = sorted(glob.glob(f"{d}/*.png"))
    T = min(len(F), len(R))
    ps, ss, ls = [], [], []
    for t in range(1, T):
        x = imageio.imread(F[t])[..., :3].astype(np.float32) / 255; y = imageio.imread(R[t])[..., :3].astype(np.float32) / 255
        mse = float(((x - y) ** 2).mean()); ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        ss.append(float(ssim(x, y, channel_axis=2, data_range=1.0)))
        with torch.no_grad():
            tx = torch.as_tensor(x).permute(2, 0, 1)[None].cuda() * 2 - 1; ty = torch.as_tensor(y).permute(2, 0, 1)[None].cuda() * 2 - 1
            ls.append(float(LP(tx, ty)))
    res[name] = dict(frames=T - 1, psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(ls)),
                     psnr_last=ps[-1], lpips_last=ls[-1])
    print(name, res[name], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
