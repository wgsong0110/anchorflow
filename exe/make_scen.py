"""벤치 시나리오를 **파일로 고정**한다. 모든 솔버가 같은 파일을 읽는다.

제어 입자와 목표점 열을 뽑고, 손잡이 지령 속도를 미리 적분해 둔다. 지령은 학생
학습과 같은 규약이다:

    s = min(vmax, a·t, sqrt(2 a d)),   목표에 닿으면 다음 목표로

손잡이는 운동학적으로 구동되므로(속도를 박는다) 그 경로는 시뮬레이션 결과와
무관하게 미리 정해진다. 그래서 속도열을 파일에 담아 두면 PG·i-PG·학생이 **글자
그대로 같은 구동**을 받는다 -- 솔버마다 컨트롤러를 따로 구현해 생기는 차이가
없어진다. 제어 입자는 시나리오 안에서 고정하고 목표점만 도달할 때마다 바꾼다.

  python exe/make_scen.py --fill pgfill_mic.npy --cfg wmats/mic_clayC_t.json \
      --out bench --tag mic_clayC --seeds 0-7 --frames 240
"""
import argparse
import json
import os

import numpy as np
import torch

from anchorflow.scene_pool import target_grid

ap = argparse.ArgumentParser()
ap.add_argument("--fill", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", default="0-7")
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--n_ctrl", type=int, default=2)
ap.add_argument("--radius", type=float, default=0.15)
ap.add_argument("--acc", type=float, default=2.4)
ap.add_argument("--vmax", type=float, default=0.6)
ap.add_argument("--targets", type=int, default=5)
ap.add_argument("--margin", type=float, default=0.30)
ap.add_argument("--tol", type=float, default=5e-3)
a = ap.parse_args()

cfg = json.load(open(a.cfg))
h = float(cfg["frame_dt"])
domain = float(cfg.get("grid_lim", 2.0))
cand = target_grid(domain, a.margin, a.targets, cfg, clear=a.radius)
X = torch.from_numpy(np.load(a.fill)).float()
os.makedirs(a.out, exist_ok=True)
lo, hi = (a.seeds.split("-") + [a.seeds])[:2]

for sd in range(int(lo), int(hi) + 1):
    g = torch.Generator().manual_seed(1000 + sd)
    # 제어 입자: 서로 2R 이상 떨어진 것에서 뽑는다 (전체 채우기 색인)
    idx = []
    for _ in range(200000):
        c = int(torch.randint(X.shape[0], (1,), generator=g))
        if all(float((X[c] - X[j]).norm()) >= 2.0 * a.radius for j in idx):
            idx.append(c)
        if len(idx) == a.n_ctrl:
            break
    while len(idx) < a.n_ctrl:
        idx.append(int(torch.randint(X.shape[0], (1,), generator=g)))
    idx = torch.tensor(idx, dtype=torch.long)

    c = X[idx].clone()                                  # [K,3] 손잡이 중심
    vel = torch.zeros(a.frames, a.n_ctrl, 3)
    pos = torch.zeros(a.frames, a.n_ctrl, 3)
    tgts, tg_used = [], None
    t_since = 0
    n_arr = 0
    for f in range(a.frames):
        if tg_used is None:
            tg_used = cand[torch.randint(cand.shape[0], (a.n_ctrl,),
                                         generator=g)].clone()
            tgts.append(tg_used.tolist())
            t_since = 0
        pos[f] = c
        vec = tg_used - c
        d = vec.norm(dim=-1)
        dirv = vec / d.clamp_min(1e-9).unsqueeze(-1)
        s_ramp = a.acc * max(t_since * h, h)
        s = torch.clamp((2.0 * a.acc * d.clamp_min(0.0)).sqrt(),
                        max=min(a.vmax, s_ramp))
        s = s * (d > a.tol).to(s.dtype)
        v = dirv * s.unsqueeze(-1)
        vel[f] = v
        c = c + h * v
        t_since += 1
        if bool(((tg_used - c).norm(dim=-1) <= a.tol).all()):
            tg_used = None                              # 도달 -> 다음 목표
            n_arr += 1
    dst = os.path.join(a.out, f"scen_{a.tag}_s{sd:02d}")
    np.savez(dst + ".npz", hid=idx.numpy(), vel=vel.numpy(), hpos=pos.numpy(),
             targets=np.asarray(tgts, dtype=np.float32))
    json.dump(dict(tag=a.tag, seed=sd, frames=a.frames, n_ctrl=a.n_ctrl,
                   radius=a.radius, acc=a.acc, vmax=a.vmax, tol=a.tol,
                   frame_dt=h, hid=idx.tolist(), targets=tgts,
                   n_arrived=n_arr, n_cand=int(cand.shape[0])),
              open(dst + ".json", "w"), indent=1)
    print(f"[시나리오] {a.tag} s{sd:02d}: 목표 {len(tgts)} 개, 도달 {n_arr} 회, "
          f"최고속도 {float(vel.norm(dim=-1).max()):.3f}, "
          f"이동거리 {float((pos[-1] - pos[0]).norm(dim=-1).max()):.3f}", flush=True)
