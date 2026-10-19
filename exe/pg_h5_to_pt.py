"""PhysGaussian(gs_simulation.py --output_h5) 프레임 h5 -> train_deform 이 읽는 .pt + lb_render 가 읽는 궤적 디렉토리.

PG h5 는 x (3,N), v (3,N), f_tensor (9,N) 배치다 (lb_gen 은 (N,3)/(N,9)). 전치해서 담는다. 입자 전부 (부분표본 없음).
cfg 는 그 실행의 공식 config 그대로 두되, PG 가 config 에 없을 때 쓰는 기본값(n_grid 50, grid_lim 2.0 --
utils/decode_param.py) 을 **명시적으로** 채운다 (우리 코드의 기본값 100 이 대신 쓰이지 않게).
x 는 float32 (영역 밀도·충격 상자 판정이 PG 와 같은 위치로), v·F 는 half.

  python exe/pg_h5_to_pt.py --h5_dir W/topo/pg_ficus_elastic --config W/PhysGaussian/config/ficus_config.json \
      --out_pt W/one/F_pg/F_pg.pt --out_dir W/lbours/ficus_gt
"""
import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out_pt", required=True)
ap.add_argument("--out_dir", default="", help="lb_render·lb_roll_to_h5 용 궤적 디렉토리 (x (N,3), F (N,9), meta.json R=I)")
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.h5_dir, "**", "sim_*.h5"), recursive=True),
            key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
assert fs, a.h5_dir
cfg = json.load(open(a.config))
cfg.setdefault("n_grid", 50)        # PG decode_param 기본값
cfg.setdefault("grid_lim", 2.0)
X, V, F = [], [], []
for f in fs:
    with h5py.File(f, "r") as h:
        x = np.array(h["x"]); v = np.array(h["v"]); Fm = np.array(h["f_tensor"])
    assert x.shape[0] == 3 and Fm.shape[0] == 9, (f, x.shape, Fm.shape)
    X.append(torch.from_numpy(x.T.copy()).float()); V.append(torch.from_numpy(v.T.copy()).half())
    F.append(torch.from_numpy(Fm.T.copy()).reshape(-1, 3, 3).half())
X, V, F = torch.stack(X), torch.stack(V), torch.stack(F)
if float(F[0].float().abs().max()) == 0.0:
    # PG 는 0 프레임 h5 를 첫 서브스텝 전에 쓰는데 그때 f_tensor 는 아직 채워지지 않은 0 이다 (1 프레임은 ≈I).
    # PG 솔버의 시작 F 는 항등이므로 0 프레임 F 를 항등으로 둔다.
    F[0] = torch.eye(3).half().expand_as(F[0])
    print("[F] 0 프레임 f_tensor 가 전부 0 (PG 미초기화 저장) -> 항등으로", flush=True)
assert torch.isfinite(X).all() and torch.isfinite(F.float()).all()
err0 = float((F[0].float() - torch.eye(3)).abs().max())
os.makedirs(os.path.dirname(a.out_pt), exist_ok=True)
meta = {"h5": fs[0], "config": os.path.abspath(a.config), "R": np.eye(3).tolist(), "src": "PhysGaussian gs_simulation",
        "n_grid_default_filled": True}
torch.save({"x": X, "v": V, "F": F, "sel": torch.arange(X.shape[1]), "cfg": cfg, "n_full": int(X.shape[1]),
            "meta": meta, "gaussians_first": int(X.shape[1])}, a.out_pt)
print(f"[저장] {a.out_pt}  {X.shape[0]} 프레임 x {X.shape[1]} 입자, F0-I 최대 {err0:.2e}, "
      f"0 프레임 |v| 최대 {float(V[0].float().norm(dim=-1).max()):.3g}, 1 프레임 |v| 평균 {float(V[1].float().norm(dim=-1).mean()):.4f}, "
      f"n_grid {cfg['n_grid']} grid_lim {cfg['grid_lim']}", flush=True)
if a.out_dir:
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(cfg, open(f"{a.out_dir}/config.json", "w"), indent=1)
    meta["config"] = f"{a.out_dir}/config.json"
    json.dump(meta, open(f"{a.out_dir}/meta.json", "w"), indent=1)
    for i in range(X.shape[0]):
        with h5py.File(f"{a.out_dir}/sim_{i:010d}.h5", "w") as o:
            o.create_dataset("x", data=X[i].numpy().astype(np.float32))
            o.create_dataset("F", data=F[i].float().reshape(-1, 9).numpy().astype(np.float32))
    print(f"[저장] {a.out_dir}  렌더용 {X.shape[0]} 프레임", flush=True)
