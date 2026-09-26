"""렌더된 영상의 시각 품질. PG 렌더를 기준으로 PSNR/SSIM/LPIPS 와 시간축 흔들림.

두 덤프를 같은 카메라로 프레임마다 나란히 그려 바로 비교한다 (프레임을 디스크에
쌓으면 240 프레임 x 800x800 이 궤적당 0.5 GB 다).

  python exe/bench_visual.py --ref bench_pg/mic_clayC_s00.pt \
      --tgt bench_ipg8/mic_clayC_s00.pt --model pgmodel/mic_whitebg-trained \
      --cfg bench_cfg/mic_clayC.json --tag ipg8 --combo mic_clayC --seed 0 \
      --out bench/visual_ipg8.csv --mp4 bench/vis_ipg8_mic_clayC_s00.mp4
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from anchorflow.gsrender import GSScene

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--tgt", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--combo", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--mp4", default="")
ap.add_argument("--white_bg", type=int, default=1)
ap.add_argument("--every", type=int, default=1, help="이 간격으로만 잰다")
a = ap.parse_args()

dev = "cuda"
XR = torch.load(a.ref, map_location="cpu", weights_only=False)
XT = torch.load(a.tgt, map_location="cpu", weights_only=False)
xr, xt = XR["x"].float(), XT["x"].float()
fr = XR.get("F")
ft = XT.get("F")
T = min(xr.shape[0], xt.shape[0])
N = min(xr.shape[1], xt.shape[1])

scene = GSScene(a.model, a.cfg, xr[0][:N].to(dev), device=dev,
                white_bg=bool(a.white_bg))
print(f"[대응] 가우시안 {scene.gs_num} / 입자 {N}, 첫 프레임 최대 차 "
      f"{scene.fit:.3e}", flush=True)

import lpips as _lp
from skimage.metrics import structural_similarity as _ssim
LP = _lp.LPIPS(net="alex").to(dev)
W = None
if a.mp4:
    import imageio
    W = imageio.get_writer(a.mp4, fps=30, macro_block_size=1)

ps, ss, lp, fl_r, fl_t = [], [], [], [], []
prev_r = prev_t = None
for t in range(0, T, a.every):
    with torch.no_grad():
        ir = scene.render(xr[t][:N], fr[t][:N].float().to(dev) if fr is not None
                          else None, frame=t)
        it = scene.render(xt[t][:N], ft[t][:N].float().to(dev) if ft is not None
                          else None, frame=t)
        ir = ir.clamp(0, 1) if torch.is_tensor(ir) else torch.as_tensor(ir)
        it = it.clamp(0, 1) if torch.is_tensor(it) else torch.as_tensor(it)
        if ir.dim() == 3 and ir.shape[0] == 3:
            ir, it = ir.permute(1, 2, 0), it.permute(1, 2, 0)
        mse = float(((ir - it) ** 2).mean())
        ps.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        A = ir.cpu().numpy()
        B = it.cpu().numpy()
        ss.append(float(_ssim(A, B, channel_axis=2, data_range=1.0)))
        lp.append(float(LP(ir.permute(2, 0, 1)[None] * 2 - 1,
                           it.permute(2, 0, 1)[None] * 2 - 1)))
        if prev_r is not None:
            fl_r.append(float((ir - prev_r).abs().mean()))
            fl_t.append(float((it - prev_t).abs().mean()))
        prev_r, prev_t = ir, it
        if W is not None:
            W.append_data((torch.cat([ir, it], 1).cpu().numpy() * 255)
                          .astype(np.uint8))
if W is not None:
    W.close()
row = dict(solver=a.tag, combo=a.combo, seed=a.seed, frames=len(ps),
           PSNR=float(np.mean(ps)), SSIM=float(np.mean(ss)),
           LPIPS=float(np.mean(lp)),
           flicker=float(np.mean(fl_t)) if fl_t else float("nan"),
           flicker_pg=float(np.mean(fl_r)) if fl_r else float("nan"))
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(json.dumps(row, ensure_ascii=False))
