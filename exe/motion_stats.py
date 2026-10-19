"""롤아웃 덤프·정답 궤적의 프레임별 입자 평균 속도 -> json (바닥 닿은 뒤 움직임이 유지되는지 보기용).

  python exe/motion_stats.py --out m.json --gt L000=/home/dkta/work/lbdata/test_low/L000 \
      --dump ipg_ov_L000=roll.pt:L000 --dump old_ov_L000=roll_old.pt:L000

속도_t = mean_p ||x_t - x_{t-1}|| / frame_dt  (t = 1..80, x_0 = 정답 처음 상태). 덤프의 pred[i] 가 프레임 i+1.
같은 줄에 질량중심 높이(z 평균)도 남긴다.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--gt", action="append", default=[], help="이름=lbdata 궤적 디렉토리")
ap.add_argument("--dump", action="append", default=[], help="이름=덤프.pt:정답이름")
ap.add_argument("--frame_dt", type=float, default=1.0 / 60)
ap.add_argument("--frames", type=int, default=80)
a = ap.parse_args()

gts = {}
res = {}
for s in a.gt:
    nm, dr = s.split("=", 1)
    fs = sorted(glob.glob(os.path.join(dr, "sim_*.h5")))[: a.frames + 1]
    X = np.stack([np.array(h5py.File(f)["x"], dtype=np.float32) for f in fs])
    gts[nm] = X
    sp = np.linalg.norm(X[1:] - X[:-1], axis=-1).mean(-1) / a.frame_dt
    res["gt_" + nm] = {"speed": sp.tolist(), "zmean": X[1:, :, 2].mean(-1).tolist()}
    print(f"[정답] {nm} {X.shape}", flush=True)
for s in a.dump:
    nm, rest = s.split("=", 1)
    pt, g = rest.rsplit(":", 1)
    P = torch.load(pt, map_location="cpu", weights_only=False)["pred"].float().numpy()[: a.frames]
    X0 = gts[g][0]
    if P.shape[1] != X0.shape[0]:
        raise SystemExit(f"{nm}: 입자 수 {P.shape[1]} != 정답 {X0.shape[0]}")
    X = np.concatenate([X0[None], P], 0)
    sp = np.linalg.norm(X[1:] - X[:-1], axis=-1).mean(-1) / a.frame_dt
    res[nm] = {"speed": sp.tolist(), "zmean": X[1:, :, 2].mean(-1).tolist(), "gt": g}
    print(f"[덤프] {nm} {P.shape}", flush=True)
json.dump(res, open(a.out, "w"))
print(f"[저장] {a.out}", flush=True)
