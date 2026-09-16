"""GaussianFluent 로 파괴 궤적을 여러 개 만들고, 학습에 쓸 크기로 압축한다.

궤적이 하나뿐이면 자기회귀 모델은 그 수열을 외운다 -- watermelon 은 임펄스 같은
변화 요인이 없는 한 줄짜리 궤적이라 그대로는 학습 데이터가 되지 못한다. 그래서
물성(E, nu, xi)과 중력을 바꿔 여러 궤적을 만든다. 물성은 모델 입력에도 들어가므로,
바뀌는 축을 물성으로 두면 "외우기" 대신 "물성에 따라 달라지는 변형" 을 배우게 된다.

원본 h5 는 한 궤적에 9.4GB(입자 138 만 x 101 프레임)라 여덟 개면 홈을 다 먹는다.
그래서 돌리자마자 입자를 부분표본해 .pt 로 압축하고 h5 는 지운다. 남기는 것은
위치·속도·변형구배 셋이다 -- 변형구배는 야코비안을 지도할 정답이다.
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
ap.add_argument("--gf_root", required=True, help="GaussianFluent 클론")
ap.add_argument("--model_root", required=True, help="model/ 가 있는 곳")
ap.add_argument("--base_config", required=True)
ap.add_argument("--scene", default="watermelon")
ap.add_argument("--out", required=True, help="압축된 .pt 를 둘 곳")
ap.add_argument("--work", required=True, help="h5 를 잠시 둘 곳")
ap.add_argument("--variants", required=True,
                help="JSON 목록. 예 '[{\"tag\":\"a\",\"E\":1e3}]'")
ap.add_argument("--n_pts", type=int, default=40000)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--keep_h5", action="store_true")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
os.makedirs(a.work, exist_ok=True)
base = json.load(open(a.base_config))
variants = json.loads(a.variants)


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
    env["PYTHONPATH"] = (a.gf_root + ":" + os.path.join(a.gf_root, "gaussian-splatting")
                         + ":" + env.get("PYTHONPATH", ""))
    env["GF_MODEL_ROOT"] = a.model_root
    r = subprocess.run(
        [sys.executable, f"gs_simulation/{a.scene}/gs_simulation_{a.scene}.py",
         "--model_path", os.path.join(a.model_root, a.scene),
         "--output_path", odir, "--config", cpath, "--output_h5"],
        cwd=a.gf_root, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[실패] {tag} rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}",
              flush=True)
        continue

    files = sorted(glob.glob(os.path.join(odir, "simulation_ply", "*.h5")))
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
    X = torch.stack(xs)
    V = torch.stack(vs)
    Fm = torch.stack(Fs).reshape(len(files), sel.numel(), 3, 3)
    # 중간에 떨어져 나간 비유한값은 직전 프레임 값으로 채운다 (프레임을 통째로
    # 버리면 자기회귀 학습의 연속성이 끊긴다)
    bad = ~torch.isfinite(X).all(-1)
    for t in range(1, X.shape[0]):
        msk = bad[t]
        if msk.any():
            X[t][msk] = X[t - 1][msk]
            V[t][msk] = 0
            Fm[t][msk] = Fm[t - 1][msk]
    Fm = torch.where(torch.isfinite(Fm), Fm, torch.eye(3).reshape(1, 1, 3, 3))
    torch.save({"x": X, "v": V, "F": Fm, "sel": sel, "cfg": cfg,
                "n_full": int(X0.shape[0]), "nonfinite_last": int((~ok).sum())},
               dst)
    print(f"[저장] {dst}  {X.shape[0]} 프레임 x {sel.numel()} 입자, "
          f"{os.path.getsize(dst)/1e6:.0f} MB", flush=True)
    if not a.keep_h5:
        shutil.rmtree(odir, ignore_errors=True)
print("GEN_OK")
