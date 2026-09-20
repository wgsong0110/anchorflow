"""제어점 조작 롤아웃을 여러 개 만들어 학습 데이터로 저장한다.

한 씬에서 파라미터를 범위에서 뽑아 롤아웃을 여러 개 굴린다. 학습용과 평가용은
**같은 범위에서 서로 다른 시드**로 뽑는다 -- 범위를 다르게 하면 일반화가 아니라
외삽을 재는 것이 된다.

저장은 h5 를 그대로 두지 않고 .pt 로 줄인다 (한 롤아웃이 수 GB 라 금방 홈을 먹는다).
남기는 것은 위치·속도·제어점 색인·제어점 궤적이다. **제어점 입자는 부분표본에
반드시 넣는다** -- 빠지면 학생이 덮어쓸 자리가 없다.

  python exe/gen_control_rollouts.py --gf <GF> --base <config.json> --out DIR \
      --n 20 --split train
"""
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
ap.add_argument("--gf", required=True)
ap.add_argument("--base", required=True, help="씬 config (clouds/material/경계)")
ap.add_argument("--out", required=True)
ap.add_argument("--work", default=None, help="h5 를 잠시 둘 곳")
ap.add_argument("--n", type=int, default=20, help="롤아웃 개수")
ap.add_argument("--split", default="train", choices=("train", "eval"))
ap.add_argument("--frames", type=int, default=40, help="롤아웃 길이(스텝)")
ap.add_argument("--n_pts", type=int, default=40000, help="저장할 입자 수")
ap.add_argument("--seed0", type=int, default=None, help="시드 블록 시작")
ap.add_argument("--keep_h5", action="store_true")
# --- 파라미터 범위 (학습/평가가 **같은** 범위를 쓴다) ---
ap.add_argument("--every_n", type=int, nargs=2, default=[10, 30])
ap.add_argument("--p_touch", type=float, nargs=2, default=[0.3, 0.9])
ap.add_argument("--depth", type=float, nargs=2, default=[0.03, 0.15])
ap.add_argument("--v_max", type=float, nargs=2, default=[0.2, 1.0])
ap.add_argument("--n_points", type=int, nargs=2, default=[4, 4])
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
work = a.work or os.path.join(a.out, "_h5")
os.makedirs(a.out, exist_ok=True)
os.makedirs(work, exist_ok=True)
base = json.load(open(a.base))

# 학습용과 평가용의 시드 블록을 겹치지 않게 띄운다
SEED0 = a.seed0 if a.seed0 is not None else (1000 if a.split == "train" else 900000)
rng = np.random.default_rng(SEED0)

rows = []
for i in range(a.n):
    seed = SEED0 + i
    r = np.random.default_rng(seed)
    ctl = dict(
        seed=int(seed),
        n_points=int(r.integers(a.n_points[0], a.n_points[1] + 1)),
        every_n=int(r.integers(a.every_n[0], a.every_n[1] + 1)),
        p_touch=float(r.uniform(*a.p_touch)),
        depth=float(r.uniform(*a.depth)),
        v_max=float(r.uniform(*a.v_max)),
    )
    cfg = dict(base)
    cfg["frame_num"] = a.frames
    cfg["control"] = ctl
    cpath = os.path.join(work, f"cfg_{a.split}_{i:03d}.json")
    json.dump(cfg, open(cpath, "w"), indent=1)
    odir = os.path.join(work, f"roll_{a.split}_{i:03d}")
    shutil.rmtree(odir, ignore_errors=True)
    print(f"[{a.split} {i+1}/{a.n}] 시드 {seed} " +
          " ".join(f"{k}={v}" for k, v in ctl.items() if k != "seed"), flush=True)
    rc = subprocess.run([sys.executable, os.path.join(HERE, "run_warp_mpm.py"),
                         "--gf", a.gf, "--config", cpath, "--out", odir],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"  실패 rc={rc.returncode}\n{(rc.stdout+rc.stderr)[-800:]}", flush=True)
        continue
    fs = sorted(glob.glob(os.path.join(odir, "sim_*.h5")))
    if len(fs) < 2:
        print("  h5 가 모자라다", flush=True); continue

    cz = np.load(os.path.join(odir, "control.npz"))
    cidx = cz["idx"]
    with h5py.File(fs[0], "r") as h:
        n_all = np.array(h["x"]).shape[-1]
    # 제어점은 **반드시** 남기고 나머지를 채운다
    keep = set(int(v) for v in cidx)
    pool = np.setdiff1d(np.arange(n_all), np.array(sorted(keep)))
    extra = np.random.default_rng(seed).choice(
        pool, max(0, min(a.n_pts - len(keep), len(pool))), replace=False)
    sel = np.sort(np.concatenate([np.array(sorted(keep)), extra]))
    remap = {int(v): j for j, v in enumerate(sel)}

    X, V = [], []
    for f in fs:
        with h5py.File(f, "r") as h:
            x = np.array(h["x"]).T[sel]
            v = np.array(h["v"]).T[sel] if "v" in h else np.zeros_like(x)
        X.append(x.astype(np.float32)); V.append(v.astype(np.float32))
    dst = os.path.join(a.out, f"{a.split}_{i:03d}.pt")
    torch.save(dict(x=torch.from_numpy(np.stack(X)),
                    v=torch.from_numpy(np.stack(V)),
                    ctrl=torch.tensor([remap[int(c)] for c in cidx], dtype=torch.long),
                    ctrl_pos=torch.from_numpy(cz["pos"]),
                    ctrl_vel=torch.from_numpy(cz["vel"]),
                    params=ctl, split=a.split, seed=int(seed)), dst)
    rows.append(dict(file=os.path.basename(dst), **ctl, frames=len(fs),
                     n_pts=int(len(sel))))
    print(f"  [저장] {dst}  {len(fs)} 프레임 x {len(sel)} 입자", flush=True)
    if not a.keep_h5:
        shutil.rmtree(odir, ignore_errors=True)

json.dump(rows, open(os.path.join(a.out, f"index_{a.split}.json"), "w"), indent=1)
print(f"[요약] {a.split} {len(rows)}/{a.n} 개, 시드 {SEED0}~{SEED0+a.n-1}", flush=True)
print("ROLLOUTS_DONE", flush=True)
