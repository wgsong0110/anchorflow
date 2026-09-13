"""세기를 고정해도 반경이 바뀌면 실제로 가해지는 힘이 달라지는가.

random_multi_poke 는 **RMS 만** magnitude 로 맞춘다. RMS 가 같아도

  * 힘을 받는 입자 수가 다르다 -- 좁으면 몇 개, 넓으면 전부
  * 총 힘 크기 sum|f| 와 알짜 운동량 |sum f| 가 다르다
  * 좁은 반경에서는 peak_cap 이 걸려 RMS 자체가 magnitude 아래로 내려간다

그래서 (K, r) 격자의 칸들은 "같은 세기"가 아니다. 반경 축에 세기가 섞여 있으면
그 격자로 "반경이 어렵다"를 말할 수 없다. 여기서 그 섞임을 수치로 낸다.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--mag", type=float, default=1.0)
ap.add_argument("--n_rep", type=int, default=4)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=777)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
BASE = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        BASE = torch.tensor(bc["force"], device=dev)
MAG = BASE.norm().item() * args.mag
EXT = float(sc.extent)
KEEP = sc.keep
print(f"[setup] 가우시안 {sc.N} (시뮬 {int(KEEP.sum())}), 물체 크기 {EXT:.4f}, "
      f"목표 RMS {MAG:.5f}", flush=True)

KS = [1, 4, 32]
RS = [0.125, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
print(f"\n{'K':>3}{'r':>8}{'받는입자':>9}{'실제RMS':>10}{'RMS/목표':>9}"
      f"{'총힘 S|f|':>12}{'알짜 |Sf|':>11}{'최대변위':>10}{'물체대비':>9}")
for K in KS:
    for R in RS:
        n_hot = rms = l1 = net = disp = 0.0
        ok = 0
        for rep in range(args.n_rep):
            g = torch.Generator(device=dev)
            g.manual_seed(args.seed + 1000 * K + 17 * int(R * 1000) + rep)
            f = sc.random_multi_poke(g, K, R * sc.sim.radius, MAG)
            m = f[KEEP].norm(dim=-1)
            n_hot += float((m > 0.05 * m.max().clamp(min=1e-20)).float().mean())
            rms += float(m.pow(2).mean().sqrt())
            l1 += float(m.sum())
            net += float(f[KEEP].sum(0).norm())
            dv = sc.impulse_dv(f)
            v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
            T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
            bad = False
            for _ in range(args.frames):
                for k in range(args.dt_mult):
                    T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
                    if (k + 1) % 8 == 0 and not T._in_domain():
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                x = T.solver.export_particle_x_to_torch()
                disp += float((x - T.pos_m).norm(dim=-1).max())
                ok += 1
        n = args.n_rep
        d = disp / ok if ok else float("nan")
        print(f"{K:>3}{R:>8.3f}{100*n_hot/n:>8.2f}%{rms/n:>10.5f}"
              f"{rms/n/MAG:>9.3f}{l1/n:>12.1f}{net/n:>11.2f}"
              f"{d:>10.5f}{100*d/EXT:>8.2f}%", flush=True)

print("\nRMS 가 목표보다 낮으면 peak_cap 이 걸린 것이다 -- 좁은 반경에서 일어난다.")
print("RADIUS_DONE")
