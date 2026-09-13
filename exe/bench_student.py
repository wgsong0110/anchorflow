"""학생 스테퍼의 프레임당 비용. Simplicits 축소 시뮬과 같은 잣대로 재기 위한 것.

한 프레임 = 스테퍼 한 번 + 앵커를 가우시안으로 스키닝하는 비용까지. 축소 모델의
"프레임당 ms" 도 x_p = X_p + J_p q 복원을 포함하므로 여기서도 포함한다.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow import scene_setup
from anchorflow.nextstate import apply_step, net_from_ckpt

sc = scene_setup.build(a.ply, a.config, a.anchors, a.k, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = net_from_ckpt(ck, dev)
net.eval()
dt = float(sc.frame_dt) if hasattr(sc, "frame_dt") else 0.04
print(f"[설정] 앵커 {sc.anchor_canonical.shape[0]}, 가우시안 {sc.pos.shape[0]}, dt {dt}")


def one_rollout(skin=True):
    p, v = sc.anchor_canonical.clone(), sc.initial_velocity(None)
    gp = sc.pos.clone()
    k = 0
    while k < a.frames:
        for q, d in apply_step(net, p, v, None, dt, sc.fixed_mask):
            p, v = q, d / dt
            if skin:
                gp = sc.skin(p, gp)
            k += 1
            if k >= a.frames:
                break
    return gp


for tag, skin in (("스테퍼만", False), ("스테퍼 + 가우시안 스키닝", True)):
    for _ in range(2):
        one_rollout(skin)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(a.reps):
        one_rollout(skin)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / (a.reps * a.frames)
    print(f"  {tag:<26} {ms:7.2f} ms/프레임")
