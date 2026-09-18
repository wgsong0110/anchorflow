"""PhysGaussian 으로 궤적을 여러 개 만들고 학습에 쓸 크기로 압축한다.

gen_gf_trajs.py 의 PhysGaussian 판이다. 다른 점은 셋뿐이다 -- 러너가 최상위
`gs_simulation.py` 이고, h5 가 하위 폴더 없이 `--output_path` 에 바로 떨어지며,
물성 키 이름이 다르다 (xi 대신 friction_angle 등).

궤적이 하나뿐이면 자기회귀 모델이 그 수열을 외운다. 그래서 물성과 중력을 바꿔
여러 개를 만든다. **모델 입력에 들어가는 축**(E, nu, density, g)만 바꾼다 --
friction_angle 처럼 입력에 없는 축을 바꾸면 모델은 구별할 방법 없이 서로 다른
동역학을 같은 조건으로 배우게 되어 평균만 낸다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pg_root", required=True)
ap.add_argument("--model_path", required=True, help="3DGS 학습 결과 폴더")
ap.add_argument("--base_config", required=True)
ap.add_argument("--scene", default="wolf")
ap.add_argument("--out", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--variants", default=None)
ap.add_argument("--variants_file", default=None,
                help="JSON 목록을 담은 파일. 셸을 거치며 따옴표가 망가지는 것을 "
                     "피하려면 이쪽이 낫다")
ap.add_argument("--n_pts", type=int, default=40000)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--keep_h5", action="store_true")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
os.makedirs(a.work, exist_ok=True)
base = json.load(open(a.base_config))
if a.variants_file:
    variants = json.load(open(a.variants_file))
elif a.variants:
    variants = json.loads(a.variants)
else:
    raise SystemExit("--variants 나 --variants_file 중 하나는 있어야 한다")


def load_x(p, key):
    with h5py.File(p, "r") as f:
        d = np.array(f[key])
    if d.ndim == 2 and d.shape[0] in (3, 9):
        d = d.T
    return d


for var in variants:
    tag = var.pop("tag")
    dst = os.path.join(a.out, f"{a.scene}_{tag}.pt")
    if os.path.exists(dst):
        print(f"[건너뜀] {dst} 이미 있다", flush=True)
        continue
    cfg = dict(base)
    cfg.update(var)
    cpath = os.path.join(a.work, f"cfg_{a.scene}_{tag}.json")
    json.dump(cfg, open(cpath, "w"), indent=1)
    odir = os.path.join(a.work, f"sim_{a.scene}_{tag}")
    print(f"[실행] {tag}: " + ", ".join(f"{k}={v}" for k, v in var.items()),
          flush=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    r = subprocess.run(
        [sys.executable, "gs_simulation.py", "--model_path", a.model_path,
         "--output_path", odir, "--config", cpath, "--output_h5", "--white_bg"],
        cwd=a.pg_root, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[실패] {tag} rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}",
              flush=True)
        continue

    files = sorted(glob.glob(os.path.join(odir, "*.h5")))
    if not files:
        print(f"[실패] {tag}: h5 가 없다", flush=True)
        continue
    X0 = torch.from_numpy(load_x(files[0], "x")).float()
    XL = torch.from_numpy(load_x(files[-1], "x")).float()
    ok = torch.isfinite(X0).all(1) & torch.isfinite(XL).all(1)
    cand = torch.nonzero(ok).squeeze(-1)
    g = torch.Generator().manual_seed(a.seed)
    sel = cand[torch.randperm(cand.numel(), generator=g)[:a.n_pts]].sort().values
    xs, vs, Fs = [], [], []
    for p in files:
        xs.append(torch.from_numpy(load_x(p, "x")).float()[sel])
        vs.append(torch.from_numpy(load_x(p, "v")).float()[sel])
        Fs.append(torch.from_numpy(load_x(p, "f_tensor")).float()[sel])
    X = torch.stack(xs); V = torch.stack(vs)
    Fm = torch.stack(Fs).reshape(len(files), sel.numel(), 3, 3)
    bad = ~torch.isfinite(X).all(-1)
    for t in range(1, X.shape[0]):
        msk = bad[t]
        if msk.any():
            X[t][msk] = X[t - 1][msk]; V[t][msk] = 0; Fm[t][msk] = Fm[t - 1][msk]
    Fm = torch.where(torch.isfinite(Fm), Fm, torch.eye(3).reshape(1, 1, 3, 3))
    torch.save({"x": X, "v": V, "F": Fm, "sel": sel, "cfg": cfg,
                "n_full": int(X0.shape[0]), "nonfinite_last": int((~ok).sum())},
               dst)
    print(f"[저장] {dst}  {X.shape[0]} 프레임 x {sel.numel()} 입자, "
          f"{os.path.getsize(dst)/1e6:.0f} MB", flush=True)
    if not a.keep_h5:
        shutil.rmtree(odir, ignore_errors=True)
print("GEN_OK")
