"""Simplicits 스키닝 가중치를 정지 형상에서 학습한다 (kaolin 공식 구현·기본값 그대로).

kaolin.physics.simplicits.SimplicitsObject.create_trained 를 **기본 인자**로 부른다
(핸들 10, 표본 1000, 층 6, 1 만 스텝, le 0.1, lo 1e6). 학습 점은 원본 3DGS 가우시안 중심
(내부 채움 없음, 정규화 좌표)이다. 가중치 함수(신경망) 자체를 저장한다 -- 추적에서
가우시안 중심의 가중치와 그 기울기(F)를 구할 때 쓴다.

kaolin 은 warp 1.x 가 필요해 별도 환경(simp)에서 돈다.

  conda activate simp
  python exe/simplicits_weights.py --fill pgfill_wolf.npy --aux repflow/aux_wolf.npz \
      --out repflow/simp_wolf.pt
"""
import argparse
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pts", required=True, help="정규화된 가우시안 중심 (rep_aux 의 *_pts.npy)")
ap.add_argument("--aux", required=True, help="정규화 상수 (rep_aux.py 출력)")
ap.add_argument("--out", required=True)
ap.add_argument("--ym", type=float, default=1e5)
ap.add_argument("--pr", type=float, default=0.45)
ap.add_argument("--rho", type=float, default=1000.0)
a = ap.parse_args()

from kaolin.physics.simplicits import SimplicitsObject          # noqa: E402

dev = "cuda"
A = np.load(a.aux, allow_pickle=True)
X = np.load(a.pts).astype(np.float64)                 # 내부 채움 없는 3DGS 가우시안
P = torch.as_tensor(X, dtype=torch.float32, device=dev)
ext = P.max(0).values - P.min(0).values
t0 = time.time()
obj = SimplicitsObject.create_trained(P, a.ym, a.pr, a.rho, float(ext.prod()))
fcn = obj.skinning_weight_function
with torch.no_grad():
    K = fcn(P[:2]).shape[1]
torch.save(fcn, a.out)
print(f"[simplicits] 기본 설정 학습: 핸들 열 {K} (자유도 {12 * K}), 점 {P.shape[0]}, "
      f"{time.time() - t0:.0f}s -> {a.out}", flush=True)
