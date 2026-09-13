"""앵커가 정말 부풀었는지 직접 확인한다.

fs_x1 은 앵커가 1024 에서 멈춘 뒤에도 짝(pair) 수가 3.35M -> 12.0M 로 계속 늘었고,
같은 시점부터 목적함수가 오르기 시작했다. 짝이 느는 경로는 두 가지뿐이다 --
앵커가 늘거나, 앵커가 커지거나. 앵커 수는 멈췄으니 후자여야 한다. 파라미터를
직접 열어 확인한다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--ckpt", nargs="+", required=True, help="이름=경로 형태")
ap.add_argument("--anchors", type=int, default=512)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse

sc = scene_setup.build(args.ply, args.config, args.anchors, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
R = float(sc.sim.radius)
print(f"[setup] 앵커 기준 반경(sim.radius) = {R:.5f}, "
      f"허용 구간 [{0.25*R:.5f}, {4.0*R:.5f}] (s_lo 0.25x ~ s_hi 4x)\n")

rows = []
for spec in args.ckpt:
    name, path = spec.split("=", 1)
    fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
    if path == "init":
        fit.init_from_geometry()
    else:
        b = torch.load(path, map_location=dev, weights_only=False)
        fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                     b.get("log_k"), b.get("log_amp"))
    s = fit.log_s.exp() / R                                   # [M,3] 배수
    vol = s.prod(-1)                                          # 부피 배수
    amp = fit.log_amp.exp()
    P = int(fit.pair_g.shape[0])
    hi = float((s > 3.99).any(-1).float().mean())
    rows.append((name, fit.M, P, P / fit.N,
                 float(s.mean()), float(s.median()), float(s.max()),
                 float(vol.mean()), float(vol.median()),
                 100 * hi, float(amp.mean()), float(amp.max())))
    print(f"  {name}: M={fit.M}, 짝={P:,}", flush=True)
    del fit
    torch.cuda.empty_cache()

H = ("체크포인트", "앵커", "짝", "가우시안당", "반경평균", "반경중앙", "반경최대",
     "부피평균", "부피중앙", "상한도달%", "amp평균", "amp최대")
print("\n" + " ".join(f"{h:>10}" for h in H))
for r in rows:
    print(f"{r[0]:>10} {r[1]:>10} {r[2]:>10,} {r[3]:>10.1f} {r[4]:>10.2f} {r[5]:>10.2f} "
          f"{r[6]:>10.2f} {r[7]:>10.2f} {r[8]:>10.2f} {r[9]:>10.1f} {r[10]:>10.2f} {r[11]:>10.2f}")
print("\n반경/부피는 sim.radius 배수. 상한도달% 는 어느 축이든 4x 클램프에 붙은 앵커 비율.")
print("SIZE_DONE")
