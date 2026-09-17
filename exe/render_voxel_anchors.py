"""복셀 앵커가 프레임 사이에 어떻게 흔들리는지 눈으로 보게 그린다.

떨림은 지금까지 숫자 하나로만 봤는데, 그 숫자는 회전·전단 같은 정상적인 국소
변형까지 섞여 들어가 절대값이 과대하다. 눈으로 보면 그 구별이 바로 된다 --
매끄럽게 흐르는 색은 재질을 따라가는 것이고, 깜빡이는 색은 배정이 끊긴 것이다.

각 가우시안을 **자기 앵커의 위치**로 칠한다. 앵커가 연속으로 움직이면 색도
연속으로 변하고, 배정이 바뀌어 앵커가 튀면 그 자리에서 색이 튄다. 색은 정준
배치가 아니라 현재 앵커 위치로 정하므로, 앞서 궤적 영상에서 쓴 "조각이 어디서
왔는지" 와는 다른 것을 본다.

세 방식을 같은 화면에 둔다: 하드 배정 / B-스플라인 / 오프셋 앙상블.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True, help="전체 입자 h5 (sim_*.h5)")
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--n_anchors", type=int, default=512,
                help="복셀 한 변을 정하는 기준 앵커 수")
ap.add_argument("--ens", type=int, default=4)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--width", type=int, default=420)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--point", type=int, default=1)
ap.add_argument("--elev", type=float, default=12.0)
ap.add_argument("--azim", type=float, default=35.0)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                     # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402

from anchorflow import ptrender, voxel                           # noqa: E402
from anchorflow.deform import fps                                # noqa: E402

files = sorted(glob.glob(os.path.join(a.h5_dir, "*.h5")))[::a.stride][:a.frames]
if not files:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")
import json                                                      # noqa: E402

cfg = json.load(open(a.config))
FRAME_DT = float(cfg["frame_dt"])


def load(p, key="x"):
    with h5py.File(p, "r") as f:
        d = np.array(f[key])
    return torch.from_numpy(d.T if d.shape[0] in (3, 9) else d).float()


X0 = load(files[0]).to(dev)
N = X0.shape[0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
aid0 = fps(X0, a.n_anchors)
H = float(torch.cdist(X0[aid0], X0[aid0]).topk(2, largest=False).values[:, 1]
          .median())
print(f"[씬] 입자 {N}, 물체 {EXT:.4f}, 복셀 한 변 {H:.5f}, 프레임 {len(files)}",
      flush=True)

# 격자는 공간에 고정한다 (궤적 전체를 담도록 첫/끝 프레임에서 여유 있게)
allb = torch.stack([load(files[0]).to(dev), load(files[-1]).to(dev)])
fin = torch.isfinite(allb).all(-1).all(0)
LO = allb[:, fin].reshape(-1, 3).min(0).values - 4 * H
OFFS = np.array([np.random.RandomState(i).rand(3) for i in range(a.ens)])
H_ENS = H * (a.ens ** (1.0 / 3.0))
MODES = [("hard (격자 1)", None, H, False),
         ("B-spline", None, H, True),
         (f"ensemble x{a.ens}", OFFS, H_ENS, False)]

R = ptrender.camera(a.elev, a.azim, dev)
ctr, half, W, Hh = ptrender.frame_box(allb[:, fin].reshape(-1, 3), R, a.width)
os.makedirs(a.out, exist_ok=True)

frames = []
m = torch.ones(N, device=dev)
for t, f in enumerate(files):
    x = load(f).to(dev)
    v = ((x - load(files[max(t - 1, 0)]).to(dev)) / FRAME_DT)
    ok = torch.isfinite(x).all(-1) & torch.isfinite(v).all(-1)
    x = torch.where(ok.unsqueeze(-1), x, X0)
    v = torch.where(ok.unsqueeze(-1), v, torch.zeros_like(v))
    tiles = []
    for name, offs, cell, soft in MODES:
        vb = voxel.build(x, X0, v, m, cell, lo=LO, offsets=offs, soft=soft)
        gi, _ = voxel.neighbors(x, vb, a.k)
        ap_ = vb.pos[gi[:, 0].clamp(min=0)]          # 자기 앵커의 위치
        # 앵커 위치를 색으로. 범위는 첫 프레임 물체 상자로 고정해 프레임 간
        # 비교가 되게 한다.
        c = (ap_ - LO) / (EXT * 0.8)
        col = (0.15 + 0.8 * c.clamp(0, 1))
        img = ptrender.splat(x, col, R, ctr, half, W, Hh, a.point)
        arr = (img.cpu().numpy() * 255).astype("uint8")
        im = Image.fromarray(arr)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 15], fill=(0, 0, 0))
        dr.text((4, 2), f"{name}  앵커 {vb.M}", fill=(255, 255, 255))
        tiles.append(np.array(im))
    row = np.concatenate(tiles, 1)
    strip = Image.fromarray(row)
    ImageDraw.Draw(strip).text((4, Hh - 14), f"frame {t:03d}  (색 = 자기 앵커 위치)",
                               fill=(0, 0, 0))
    frames.append(np.array(strip))
    if t % 10 == 0:
        print(f"  {t}/{len(files)}", flush=True)

p_out = os.path.join(a.out, "voxel_anchors.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}  {len(frames)} 프레임, {row.shape[1]}x{row.shape[0]}",
      flush=True)
print("VOXVID_OK")
