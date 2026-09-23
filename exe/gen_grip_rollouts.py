"""집게 조작 롤아웃을 만들어 학생 학습 데이터로 저장한다.

제어점(위치 강제) 방식이 아니라 **물리 집게**(판 충돌체)로 굴린 궤적이다. 학생에게는
집게를 강제 신호가 아니라 **조건 입력**으로 준다 -- 판의 자세·간격·속도를 앵커마다
상대좌표로 넣어 준다. 덮어쓸 입자가 없으므로 손실 가림도 필요 없다.

  python exe/gen_grip_rollouts.py --pg <PhysGaussian> --gs <gaussian-splatting> \
      --model <3DGS> --base <config.json> --out DIR --n 20 --split train
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
ap.add_argument("--pg", required=True, help="집게 패치가 적용된 PhysGaussian")
ap.add_argument("--gs", required=True, help="gaussian-splatting 경로")
ap.add_argument("--model", required=True, help="3DGS 모델 디렉토리")
ap.add_argument("--base", required=True, help="씬 config (grip_plate 포함)")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--split", default="train", choices=("train", "eval"))
ap.add_argument("--n_pts", type=int, default=30000)
ap.add_argument("--seed0", type=int, default=None)
ap.add_argument("--keep_h5", action="store_true")
# 조작 파라미터 범위 (학습/평가가 **같은** 범위, 시드만 다르다)
ap.add_argument("--speed", type=float, nargs=2, default=[0.6, 1.4])
ap.add_argument("--twist", type=float, default=1.0)
ap.add_argument("--grip", type=float, nargs=2, default=[0.45, 0.65])
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
work = os.path.join(a.out, "_h5")
os.makedirs(a.out, exist_ok=True)
os.makedirs(work, exist_ok=True)
base = json.load(open(a.base))
SEED0 = a.seed0 if a.seed0 is not None else (1000 if a.split == "train" else 900000)

rows = []
for i in range(a.n):
    seed = SEED0 + i
    r = np.random.default_rng(seed)
    gp = dict(base.get("grip_plate", {}))
    gp.update(seed=int(seed), speed=[float(a.speed[0]), float(a.speed[1])],
              twist=float(r.uniform(0.3, a.twist)),
              grip=float(r.uniform(*a.grip)))
    cfg = dict(base)
    cfg["grip_plate"] = gp
    cpath = os.path.join(work, f"cfg_{a.split}_{i:03d}.json")
    json.dump(cfg, open(cpath, "w"), indent=1)
    odir = os.path.join(work, f"roll_{a.split}_{i:03d}")
    shutil.rmtree(odir, ignore_errors=True)
    print(f"[{a.split} {i+1}/{a.n}] 시드 {seed} 무는 정도 {gp['grip']:.2f} "
          f"비틀기 {gp['twist']:.2f}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{a.pg}:{a.gs}:" + env.get("PYTHONPATH", "")
    env["AF_EXE"] = HERE
    rc = subprocess.run([sys.executable, os.path.join(a.pg, "gs_simulation.py"),
                         "--model_path", a.model, "--output_path", odir,
                         "--config", cpath, "--output_h5"],
                        capture_output=True, text=True, cwd=a.pg, env=env)
    h5dir = os.path.join(odir, "simulation_ply")
    fs = sorted(glob.glob(os.path.join(h5dir, "sim_*.h5")))
    if rc.returncode != 0 or len(fs) < 2:
        print(f"  실패 rc={rc.returncode}\n{(rc.stdout + rc.stderr)[-500:]}", flush=True)
        continue
    pose = np.load(os.path.join(h5dir, "gripseq_pose.npy"), allow_pickle=True)
    log = np.load(os.path.join(h5dir, "gripseq.npy"), allow_pickle=True)
    with h5py.File(fs[0], "r") as h:
        n_all = np.array(h["x"]).shape[-1]
    sel = np.sort(np.random.default_rng(seed).choice(
        n_all, min(a.n_pts, n_all), replace=False))

    X, V, FF = [], [], []
    for f in fs:
        with h5py.File(f, "r") as h:
            x = np.array(h["x"]).T[sel]
            v = np.array(h["v"]).T[sel] if "v" in h else np.zeros_like(x)
            ft = (np.array(h["f_tensor"]).T[sel].reshape(-1, 3, 3)
                  if "f_tensor" in h else None)
        X.append(x.astype(np.float32)); V.append(v.astype(np.float32))
        if ft is not None:
            FF.append(ft.astype(np.float32))
    # PhysGaussian 은 0 프레임 f_tensor 를 전부 0 으로 쓴다. 그대로 두면 모양
    # 손실에서 det=0 -> NaN 이 되어 학습이 첫 스텝에 죽는다 (겪었다).
    if FF:
        Fa = np.stack(FF)
        det = np.linalg.det(Fa)
        bad = (~np.isfinite(det)) | (np.abs(det) < 1e-8)
        if bad.any():
            Fa[bad] = np.eye(3, dtype=np.float32)
            FF = list(Fa)
            print(f"  [고침] 퇴화한 F {int(bad.sum())} 개를 단위행렬로", flush=True)
    Xa = np.stack(X)
    if not np.isfinite(Xa).all():
        print("  [버림] 비유한 값 -- 교사가 터졌다", flush=True)
        if not a.keep_h5:
            shutil.rmtree(odir, ignore_errors=True)
        continue
    # 집게 자세: 프레임마다 (팔, 중심 3, 회전 9, 간격 1)
    T = len(fs)
    arms = max(len(p) for p in pose) if len(pose) else 0
    G = np.zeros((T, arms, 13), np.float32)
    for t in range(min(T, len(pose))):
        for k, (c, R, g) in enumerate(pose[t]):
            G[t, k] = np.concatenate([np.asarray(c, np.float32).ravel(),
                                      np.asarray(R, np.float32).ravel(),
                                      [np.float32(g)]])
    dst = os.path.join(a.out, f"{a.split}_{i:03d}.pt")
    torch.save(dict(x=torch.from_numpy(Xa), v=torch.from_numpy(np.stack(V)),
                    **({"F": torch.from_numpy(np.stack(FF))} if FF else {}),
                    cfg=cfg, grip=torch.from_numpy(G),
                    grip_cfg=gp, split=a.split, seed=int(seed)), dst)
    rows.append(dict(file=os.path.basename(dst), seed=int(seed), frames=T,
                     n_pts=int(len(sel)), arms=int(arms), **{
                         k: gp[k] for k in ("grip", "twist")}))
    print(f"  [저장] {dst}  {T} 프레임 x {len(sel)} 입자, 팔 {arms}", flush=True)
    if not a.keep_h5:
        shutil.rmtree(odir, ignore_errors=True)

json.dump(rows, open(os.path.join(a.out, f"index_{a.split}.json"), "w"), indent=1)
print(f"[요약] {a.split} {len(rows)}/{a.n} 개", flush=True)
print("GRIP_ROLLOUTS_DONE", flush=True)
