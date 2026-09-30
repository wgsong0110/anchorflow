"""i-PG 교사의 하드 Dirichlet 이 실제로 구속하는지 h5 에서 직접 잰다.

솔버 로그(gv_max, max_v)만 보고 "걸렸다" 고 판단했었다. 실제로는 손잡이 입자가
명령 속도대로 움직였는지를 궤적에서 재야 확인이 된다.

  손잡이 입자 p:  x_{k+1}[p] - x_k[p]  ==  frame_dt * v_cmd_k[p]
  반경 밖 입자 :  명령과 무관해야 한다 (전부 끌려가면 구속이 과하게 퍼진 것)
"""
from __future__ import annotations
import argparse, glob, os, sys

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5dir", required=True)
ap.add_argument("--scen", required=True)
ap.add_argument("--frame_dt", type=float, default=1.0 / 60)
ap.add_argument("--radius", type=float, default=0.15)
a = ap.parse_args()
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


def rd(p, k):
    with h5py.File(p, "r") as f:
        v = np.array(f[k])
    return v.T if v.shape[0] in (3, 9) and v.shape[0] != v.shape[-1] else v


files = sorted(glob.glob(os.path.join(a.h5dir, "**", "*.h5"), recursive=True))
chk("h5 프레임이 2 개 이상", len(files) >= 2, f"{len(files)} 개")
if len(files) < 2:
    print(f"\n{OK[0]}/{OK[1]}  SOME-FAIL"); raise SystemExit(1)

z = np.load(a.scen)
hid, vel = z["hid"], z["vel"]
X = np.stack([rd(p, "x") for p in files])            # [T,N,3]
print(f"  프레임 {X.shape[0]}, 입자 {X.shape[1]}, 손잡이 {hid.tolist()}")

x0 = X[0]
mem = [np.linalg.norm(x0 - x0[j], axis=1) < a.radius for j in hid]
chk("손잡이 반경 안 입자가 있다", all(int(m.sum()) > 0 for m in mem),
    " ".join(f"{int(m.sum())}" for m in mem))

# 손잡이 **중심 입자** 의 변위가 명령과 맞는가
errs, cmds = [], []
for k in range(X.shape[0] - 1):
    vc = vel[min(k, vel.shape[0] - 1)]
    for j, p in enumerate(hid):
        d = X[k + 1][p] - X[k][p]
        c = a.frame_dt * vc[j]
        cmds.append(np.linalg.norm(c))
        errs.append(np.linalg.norm(d - c))
errs, cmds = np.array(errs), np.array(cmds)
rel = errs / np.maximum(cmds, 1e-12)
chk("손잡이 중심이 명령대로 움직인다 (상대오차 중앙 < 5%)",
    float(np.median(rel)) < 0.05,
    f"중앙 {float(np.median(rel)):.3f}, 최대 {float(rel.max()):.3f}, "
    f"명령크기 중앙 {float(np.median(cmds)):.4f}")

# 반경 밖 입자가 통째로 끌려가지는 않는가 (구속이 과하게 퍼졌나)
far = ~np.any(np.stack(mem), axis=0)
mv_near = np.linalg.norm(X[-1][~far] - X[0][~far], axis=1).mean()
mv_far = np.linalg.norm(X[-1][far] - X[0][far], axis=1).mean()
chk("반경 밖이 손잡이만큼 끌려가지 않는다", mv_far < mv_near,
    f"안 {mv_near:.4f} vs 밖 {mv_far:.4f}")
chk("물체가 실제로 움직였다", mv_near > 1e-3, f"손잡이 근처 평균 {mv_near:.4f}")

# 물리적 온전성
chk("좌표가 유한", bool(np.isfinite(X).all()), "")
F = None
try:
    F = np.stack([rd(p, "f_tensor").reshape(-1, 3, 3) for p in files])
except Exception as e:
    print(f"  (F 를 못 읽었다: {e})")
if F is not None:
    det = np.linalg.det(F)
    chk("변형구배 det > 0", float(det.min()) > 0,
        f"최소 {float(det.min()):.4f}, 중앙 {float(np.median(det)):.4f}")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
