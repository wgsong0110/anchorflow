"""PGND 예측(pgnd_eval_lb.py --save_quat 의 pred_XXXX.pt)과 PG 정답 궤적(lbdata)을 3DGS 로 렌더해 좌우로 붙인 비교 영상.

PGND 의 가우시안 렌더 규약(experiments/train/gs.py rollout_and_render) 그대로: 위치 = 상향 표본 위치,
회전 = 시작 프레임 이후 누적 강체 회전 q (뼈 SVD 회전의 가중 평균). 공분산 = R(q) Σ_start R(q)ᵀ,
Σ_start = 시작 프레임(12) 정답의 F C0 Fᵀ. PGND 는 0.1 s 스텝(물리 6 프레임)이라 두 쪽 모두 12, 18, .., 78 을 10 fps(실시간)로.

  python exe/lb_pred_video.py --pred DIR/pred_0100.pt --config /home/dkta/work/lbdata/config.json --out DIR/cmp_0100.mp4
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
ap.add_argument("--pred", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--label", default="PGND")
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"
P = torch.load(a.pred, weights_only=False)
assert P.get("q") is not None, "pred 에 회전이 없다 -- pgnd_eval_lb.py --save_quat 로 다시"
src = P["src"]
R = LegoRenderer(json.load(open(a.config)), dev=dev)
NG = R.NG


def h5(f):
    with h5py.File(f"{src}/sim_{f:010d}.h5") as h:
        x = torch.as_tensor(np.array(h["x"][:NG]), device=dev).float()
        F = torch.as_tensor(np.array(h["F"][:NG]), device=dev).float().reshape(-1, 3, 3)
    return x, F


def qmat(q):                                         # wxyz -> 회전 행렬 (kornia 규약과 같음)
    q = torch.nn.functional.normalize(q.float(), dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def tag(img, s):
    try:
        import cv2
    except ImportError:
        return img
    img = np.ascontiguousarray(img)
    cv2.putText(img, s, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    return img


f0 = int(P["start_frame"]); frames = [f0] + list(P["frames"])
x0, F0 = h5(f0)
W = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
for i, f in enumerate(frames):
    xg, Fg = h5(f)
    left = tag(R.image(xg, Fg), f"PG (GT)  frame {f}")
    if i == 0:
        xp, Fp = x0, F0
    else:
        xp = P["x"][i - 1][:NG].to(dev).float()
        Fp = qmat(P["q"][i - 1][:NG].to(dev)) @ F0       # R Σ0 Rᵀ = (R F0) C0 (R F0)ᵀ
    right = tag(R.image(xp, Fp), f"{a.label}  frame {f}")
    W.append_data(np.concatenate([left, right], 1))
W.close()
print(f"[영상] {a.out}  {len(frames)} 프레임 (물리 {frames[0]}..{frames[-1]}, {a.fps} fps)", flush=True)
