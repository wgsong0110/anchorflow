"""학습 비교 공통 데이터셋(lbdata, exe/lb_gen.py) 궤적을 train_deform.py 가 읽는 .pt 로.

  x [T,N,3], v [T,N,3], F [T,N,3,3] 는 half (train_deform 이 half 로 상주시키는 형식), 입자 전부 (부분표본 없음).
  F 는 lb_gen 이 렌더용으로 초기 회전 R 을 실어 둔 F_sim·R 이므로 R 을 떼어 F_sim (회전한 정지 자세 기준 탄성 F) 로 되돌린다.

  python exe/lb_to_pt.py --src /home/dkta/work/lbdata/test --out /home/dkta/work/lbpt/test
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="궤적 디렉토리들이 든 곳 (각각 sim_*.h5 + meta.json)")
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="", help="쉼표 목록: 이 궤적 이름만")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = sorted(d for d in os.listdir(a.src) if os.path.isfile(f"{a.src}/{d}/meta.json"))
if a.only:
    names = [n for n in names if n in set(a.only.split(","))]
for nm in names:
    dst = f"{a.out}/{nm}.pt"
    if os.path.exists(dst):
        print(f"[있음] {dst}", flush=True)
        continue
    meta = json.load(open(f"{a.src}/{nm}/meta.json"))
    cfg = json.load(open(meta["config"]))
    Rt = torch.as_tensor(np.array(meta["R"]), dtype=torch.float64).T          # F_sim = (F_sim·R)·Rᵀ
    fs = sorted(glob.glob(f"{a.src}/{nm}/sim_*.h5"))
    X, V, F = [], [], []
    for f in fs:
        with h5py.File(f) as h:
            X.append(torch.from_numpy(np.array(h["x"])).half())
            V.append(torch.from_numpy(np.array(h["v"])).half())
            Fm = torch.from_numpy(np.array(h["F"])).double().reshape(-1, 3, 3) @ Rt
            F.append(Fm.half())
    X, V, F = torch.stack(X), torch.stack(V), torch.stack(F)
    assert torch.isfinite(X.float()).all() and torch.isfinite(F.float()).all(), nm
    err0 = float((F[0].float() - torch.eye(3)).abs().max())
    torch.save({"x": X, "v": V, "F": F, "sel": torch.arange(X.shape[1]), "cfg": cfg, "n_full": int(X.shape[1]),
                "meta": meta, "gaussians_first": int(meta.get("gaussians_first", X.shape[1]))}, dst)
    print(f"[저장] {dst}  {X.shape[0]} 프레임 x {X.shape[1]} 입자, F0-I 최대 {err0:.2e}, "
          f"{os.path.getsize(dst) / 1e6:.0f} MB", flush=True)
