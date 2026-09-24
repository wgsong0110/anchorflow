"""손잡이 제어 MPM 롤아웃을 형상x물성x시드 로 만들고 학습용 .pt 로 압축한다.

씬별 하이퍼파라미터는 전부 동일하다 (n_grid, substep_dt, frame_dt, 프레임 수,
입자 간격, 손잡이 반경·개수·이동·속도·가속도·회차). 조합마다 시드만 달라서
손잡이 위치와 방향이 전부 다른 궤적이 된다.

  python exe/gen_handle_trajs.py --pg <PG> --work <작업폴더> --out <pt 폴더> \
      --shape wolf --material clayC --seeds 0-15 --gpu 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True, help="형상_물성")
ap.add_argument("--seeds", default="0-15")
ap.add_argument("--pairs", default="", help="tag:seed 쉼표 목록 (여러 조합을 한 작업에서)")
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--fill_cache", default="")
ap.add_argument("--rounds_cycle", default="",
                help="쉼표 목록. 시드마다 회차 수를 돌려가며 써서 "
                     "롤아웃 길이를 다양하게 한다 (길이 = 회차 x RF)")
a = ap.parse_args()

if a.pairs:
    JOBS = [(t, int(sd)) for t, sd in
            (x.split(":") for x in a.pairs.split(",") if x)]
else:
    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
    JOBS = [(a.tag, sd) for sd in range(int(lo), int(hi) + 1)]
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.work, exist_ok=True)
CFGDIR = os.path.dirname(a.config)
MODELDIR = os.path.dirname(a.model)
SHAPE_MODEL = {"wolf": "wolf_whitebg-trained", "bread": "bread-trained",
               "lego": "lego_whitebg-trained", "mic": "mic_whitebg-trained",
               "hotdog": "hotdog_whitebg-trained"}

for _tag, sd in JOBS:
    dst = os.path.join(a.out, f"{_tag}_s{sd:02d}.pt")
    if os.path.exists(dst):
        print(f"[건너뜀] {dst}", flush=True)
        continue
    odir = os.path.join(a.work, f"{_tag}_s{sd:02d}")
    shutil.rmtree(odir, ignore_errors=True)
    os.makedirs(odir, exist_ok=True)
    env = dict(os.environ)
    env["AF_H_SEED"] = str(sd)
    env["AF_H_DUMP"] = os.path.join(odir, "handle.npz")
    if a.fill_cache:
        env["AF_FILL_CACHE"] = a.fill_cache
    _shape = _tag.split("_")[0]
    _cfgp = os.path.join(CFGDIR, f"{_tag}.json") if a.pairs else a.config
    if a.rounds_cycle:
        _rc = [int(x) for x in a.rounds_cycle.split(",") if x]
        _nr = _rc[sd % len(_rc)]
        _rf = int(env.get("AF_H_RF", 60))
        _c = json.load(open(_cfgp))
        _c["frame_num"] = _nr * _rf
        _cfgp = os.path.join(a.work, f"cfg_{_tag}_s{sd}.json")
        json.dump(_c, open(_cfgp, "w"))
        env["AF_H_ROUNDS"] = str(_nr)
        print(f"[길이] seed {sd}: 회차 {_nr} x {_rf} = {_nr * _rf} 프레임", flush=True)
    _mdl = (os.path.join(MODELDIR, SHAPE_MODEL[_shape]) if a.pairs else a.model)
    cfg = json.load(open(_cfgp))
    if a.pairs:
        env["AF_FILL_CACHE"] = os.path.join(
            os.path.dirname(a.fill_cache or "/home/dkta/work/x"), f"fill_{_shape}.npy")
    r = subprocess.run(
        ["python", "-u", "gs_simulation.py", "--model_path", _mdl,
         "--config", _cfgp, "--output_path", odir, "--output_h5"],
        cwd=a.pg, env=env, capture_output=True, text=True)
    open(os.path.join(a.work, f"sim_{_tag}_s{sd:02d}.log"), "w").write(
        (r.stdout or "")[-4000:] + "\n" + (r.stderr or "")[-2000:])
    files = sorted(glob.glob(os.path.join(odir, "**", "*.h5"), recursive=True))
    if not files:
        print(f"[실패] seed {sd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
        continue

    def rd(p, k):
        with h5py.File(p, "r") as f:
            v = np.array(f[k])
        return v.T if v.shape[0] in (3, 9) and v.shape[0] != v.shape[-1] else v

    X0 = torch.from_numpy(rd(files[0], "x")).float()
    g = torch.Generator().manual_seed(1234 + sd)
    sel = torch.randperm(X0.shape[0], generator=g)[:min(a.n_pts, X0.shape[0])]
    xs, vs, fs = [], [], []
    for p in files:
        xs.append(torch.from_numpy(rd(p, "x")).float()[sel])
        try:
            vs.append(torch.from_numpy(rd(p, "v")).float()[sel])
        except Exception:
            vs.append(torch.zeros_like(xs[-1]))
        try:
            # PG 는 변형구배를 **f_tensor** 로 저장한다. "F" 로 읽으면 예외가 나고
            # 항등으로 대체되는데, 그러면 Psi 가 항등적으로 0 이라 물리 손실이
            # 관성과 중력만 남는다 -- 한동안 그렇게 굴러갔다.
            fs.append(torch.from_numpy(rd(p, "f_tensor")).float()
                      .reshape(-1, 3, 3)[sel])
        except Exception as _e:
            print(f"  [경고] F 를 읽지 못해 항등으로 둔다: {_e}", flush=True)
            fs.append(torch.eye(3).repeat(sel.numel(), 1, 1))
    X, V, Fm = torch.stack(xs), torch.stack(vs), torch.stack(fs)
    h = np.load(env["AF_H_DUMP"])
    torch.save({"x": X.half(), "v": V.half(), "F": Fm.half(), "sel": sel,
                "cfg": cfg, "seed": sd, "tag": _tag,
                "ctrl_pos": torch.from_numpy(h["hpos"]).float(),
                "ctrl_vel": torch.from_numpy(h["hvel"]).float(),
                "ctrl_id": torch.from_numpy(h["hid"]).long(),
                "ctrl_R": torch.from_numpy(h["R"]).float(),
                "n_full": int(X0.shape[0])}, dst)
    print(f"[저장] {dst} {X.shape[0]}프레임 x {sel.numel()}입자 "
          f"{os.path.getsize(dst)/1e6:.0f}MB", flush=True)
    shutil.rmtree(odir, ignore_errors=True)
print("HTRAJ_OK")
