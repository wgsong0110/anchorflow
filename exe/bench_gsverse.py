"""GS-Verse 포팅으로 벤치 시나리오를 돌려 PG 와 같은 입자 집합의 궤적을 낸다."""
import argparse
import json
import time

import numpy as np
import torch

from anchorflow.gsverse import GSVerseSim

ap = argparse.ArgumentParser()
ap.add_argument("--scen", required=True)
ap.add_argument("--fill", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--spring", type=float, default=20.0)
ap.add_argument("--damping", type=float, default=5.0)
ap.add_argument("--substeps", type=int, default=4)
a = ap.parse_args()

dev = "cuda"
cfg = json.load(open(a.cfg))
h = float(cfg["frame_dt"])
sn = np.load(a.scen)
XF = torch.from_numpy(np.load(a.fill)).float().to(dev)
g = torch.Generator().manual_seed(0)
sel = torch.randperm(XF.shape[0], generator=g)[:a.n_pts].sort().values.to(dev)
hid = torch.as_tensor(sn["hid"], dtype=torch.long, device=dev)
vel = torch.as_tensor(sn["vel"], dtype=torch.float32, device=dev)

sim = GSVerseSim(XF[sel], int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0)),
                 spring=a.spring, damping=a.damping,
                 radius=float(np.asarray(sn.get("radius", 0.15)).reshape(-1)[0])
                 if "radius" in sn else 0.15, substeps=a.substeps)
# 손잡이 중심은 전체 채우기의 제어 입자 위치에서 시작한다 (PG 와 같은 자리)
hc = XF[hid].clone()
XS, FS = [sim.positions().cpu().half()], [sim.F().cpu().half()]
t0 = time.time()
with torch.no_grad():
    for fr in range(vel.shape[0]):
        vc = vel[fr]
        sim.step(h, handle_pos=hc, handle_vel=vc)
        hc = hc + h * vc
        XS.append(sim.positions().cpu().half())
        FS.append(sim.F().cpu().half())
wall = time.time() - t0
torch.save(dict(x=torch.stack(XS), F=torch.stack(FS), cfg=cfg,
                n=int(sel.numel()), wall_s=wall,
                fps=(len(XS) - 1) / max(wall, 1e-9)), a.out)
print(f"[GS-Verse] 저장 {a.out} {len(XS)}프레임, {wall:.2f}s, "
      f"{(len(XS) - 1) / max(wall, 1e-9):.2f} FPS", flush=True)
