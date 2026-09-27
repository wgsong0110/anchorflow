"""궤적을 앞 N+1 프레임만 남겨 잘라 낸다 (앞 N 프레임만 학습시키기 위해)."""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--frames", type=int, required=True, help="남길 프레임 수 N+1")
a = ap.parse_args()

d = torch.load(a.src, map_location="cpu", weights_only=False)
n = int(a.frames)
for k in ("x", "v", "F", "motion"):
    if k in d and torch.is_tensor(d[k]) and d[k].dim() >= 1:
        d[k] = d[k][:n].clone()
for k in ("ctrl_id", "ctrl_vel", "ctrl_pos", "ctrl_R"):
    if k in d and torch.is_tensor(d[k]) and d[k].shape[0] > n:
        d[k] = d[k][:n].clone()
if "cfg" in d and isinstance(d["cfg"], dict):
    d["cfg"] = dict(d["cfg"])
    d["cfg"]["frame_num"] = n - 1
os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
torch.save(d, a.dst)
print(f"[자름] {os.path.basename(a.src)} -> {a.dst}  {d['x'].shape[0]} 프레임")
