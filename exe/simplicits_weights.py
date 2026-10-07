"""Simplicits 스키닝 가중치를 정지 형상에서 학습한다 (kaolin 공식 구현 그대로).

kaolin.physics.simplicits.SimplicitsObject.create_trained 로 신경 스키닝 가중치를
학습하고(논문의 탄성 에너지 + 직교 손실), 목표 입자 X0 와 영상용 가우시안(aux)에서
가중치를 뽑아 저장한다. rep_track.py --method simplicits --simp_w 가 읽는다.
핸들 수는 자유도 예산에 맞춘다 (핸들당 3x4 = 12).

kaolin 은 warp 1.x 가 필요해 별도 환경(simp)에서 돈다.

  conda activate simp
  python exe/simplicits_weights.py --flow repflow/flow_wolf.npz \
      --aux repflow/aux_wolf_pts.npy --out repflow/simpw_wolf.npz
"""
import argparse
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--aux", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--dof", type=int, default=3000)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--ym", type=float, default=1e5)
ap.add_argument("--pr", type=float, default=0.45)
ap.add_argument("--rho", type=float, default=1000.0)
a = ap.parse_args()

from kaolin.physics.simplicits import SimplicitsObject          # noqa: E402

dev = "cuda"
X0 = torch.as_tensor(np.load(a.flow)["X0"], dtype=torch.float32, device=dev)
AUX = (torch.as_tensor(np.load(a.aux), dtype=torch.float32, device=dev)
       if a.aux else torch.zeros(0, 3, device=dev))
ext = X0.max(0).values - X0.min(0).values
vol = float(ext.prod())
K = a.dof // 12
t0 = time.time()
obj = SimplicitsObject.create_trained(
    X0, a.ym, a.pr, a.rho, vol, num_handles=K,
    training_num_steps=a.steps, training_log_every=max(a.steps // 10, 1))
with torch.no_grad():
    P = torch.cat([X0, AUX], 0)
    Wl = [obj.skinning_weight_function(P[i:i + 20000]) for i in range(0, P.shape[0], 20000)]
    W = torch.cat(Wl, 0)
print(f"[simplicits] 핸들 요청 {K}, 가중치 열 {W.shape[1]} (자유도 {12 * W.shape[1]}), "
      f"점 {W.shape[0]}, 학습 {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(a.out, W=W.cpu().numpy().astype(np.float32), K=W.shape[1])
print(f"[저장] {a.out}", flush=True)
