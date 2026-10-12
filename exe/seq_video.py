"""프레임 순차 과적합(--seq_frames) 체크포인트에 남은 위치 궤적을 정답과 나란히 3DGS 로 렌더.

체크포인트에는 위치만 있고 변형구배가 없어서, 모델 쪽 가우시안 모양은 0 프레임 정답 모양(F_render[0]) 그대로 두고
위치만 옮긴다. 왼쪽 정답(lbdata h5 의 x·F), 오른쪽 모델.

  python exe/seq_video.py --ckpt /home/dkta/work/lbours/seq0000/MAIN_last.pt --traj /home/dkta/work/lbdata/test/0000 \
      --config /home/dkta/work/lbdata/config.json --out OUT.mp4
"""
import argparse
import json
import os
import sys

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow.lb_gs_render import LegoRenderer                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"
sq = torch.load(a.ckpt, map_location="cpu", weights_only=False)["seq"]
X, err = sq["traj"], sq["err"]
R = LegoRenderer(json.load(open(a.config)), dev=dev)


def h5(f):
    with h5py.File(f"{a.traj}/sim_{f:010d}.h5") as h:
        return (torch.as_tensor(np.array(h["x"][:R.NG]), device=dev).float(),
                torch.as_tensor(np.array(h["F"][:R.NG]), device=dev).float().reshape(-1, 3, 3))


def tag(img, s):
    import cv2
    img = np.ascontiguousarray(img)
    cv2.putText(img, s, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return img


x0, F0 = h5(0)
W = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
for f in range(len(X) + 1):
    xg, Fg = h5(f)
    xm = x0 if f == 0 else X[f - 1][:R.NG].to(dev).float()
    e = "" if f == 0 else f"  err {100 * err[f - 1]:.1f}%"
    W.append_data(np.concatenate([tag(R.image(xg, Fg), f"GT  frame {f}"), tag(R.image(xm, F0), f"model  frame {f}{e}")], 1))
W.close()
print(f"[영상] {a.out}  {len(X) + 1} 프레임", flush=True)
