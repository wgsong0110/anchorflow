"""바닥 접촉 진단 (미끄러짐 방지 --bc_stick 확인용) -> json. 정답 궤적·롤아웃 덤프의 프레임별
  n_contact : 바닥 법선 거리 d = n·(x-p) < --thr (0.01) 인 입자 수
  com_x, com_y, com_z : 무게중심 (입자 평균)
  tan_disp  : 그 프레임 접촉 입자의 직전 프레임 대비 접선 변위 크기 평균 (접촉 입자 없으면 null)
  d_min, d_contact_mean : 최소 법선 거리, 접촉 입자 평균 법선 거리 ("살짝 띄워 미끄러지는" 회피 확인용)
  n_near    : d < 3·thr 인 입자 수
  detF_contact : (덤프에 F 가 있으면) 접촉 입자 누적 det F 평균
프레임 t = 1..L (x_0 = 정답 처음 상태). 덤프의 pred[i] 가 프레임 i+1.

  python exe/stick_diag.py --cfg cfg.json --out d.json --gt L000=/home/dkta/work/lbdata/test_low/L000 \
      --dump new_ov_L000=roll.pt:L000
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gt", action="append", default=[], help="이름=lbdata 궤적 디렉토리")
ap.add_argument("--dump", action="append", default=[], help="이름=덤프.pt:정답이름")
ap.add_argument("--thr", type=float, default=0.01)
ap.add_argument("--frames", type=int, default=80)
a = ap.parse_args()

cfg = json.load(open(a.cfg))
bc = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"][0]
P0 = np.asarray(bc["point"], np.float64)
NR = np.asarray(bc["normal"], np.float64); NR = NR / np.linalg.norm(NR)
print(f"[바닥] point {P0.tolist()} normal {NR.tolist()} 접촉 기준 d < {a.thr}", flush=True)


def stats(X, F=None):
    """X [L+1,N,3] (0 = 처음 상태), F [L,N,3,3] 또는 None."""
    d = (X - P0) @ NR                                              # [L+1,N]
    r = {k: [] for k in ("n_contact", "n_near", "com_x", "com_y", "com_z", "tan_disp", "d_min", "d_contact_mean")}
    if F is not None:
        r["detF_contact"] = []
    for t in range(1, X.shape[0]):
        c = d[t] < a.thr
        r["n_contact"].append(int(c.sum())); r["n_near"].append(int((d[t] < 3 * a.thr).sum()))
        m = X[t].mean(0); r["com_x"].append(float(m[0])); r["com_y"].append(float(m[1])); r["com_z"].append(float(m[2]))
        r["d_min"].append(float(d[t].min()))
        if c.any():
            dx = X[t][c] - X[t - 1][c]
            dt_ = dx - (dx @ NR)[:, None] * NR
            r["tan_disp"].append(float(np.linalg.norm(dt_, axis=-1).mean()))
            r["d_contact_mean"].append(float(d[t][c].mean()))
        else:
            r["tan_disp"].append(None); r["d_contact_mean"].append(None)
        if F is not None:
            if c.any():
                r["detF_contact"].append(float(torch.linalg.det(F[t - 1][torch.from_numpy(c)].double()).mean()))
            else:
                r["detF_contact"].append(None)
    return r


def summ(nm, r):
    nc = r["n_contact"]; td = [v for v in r["tan_disp"] if v is not None]
    s = (f"[{nm}] 접촉 입자 최대 {max(nc)} (마지막 {nc[-1]}), 중심 x {r['com_x'][0]:.3f}->{r['com_x'][-1]:.3f}, "
         f"y {r['com_y'][0]:.3f}->{r['com_y'][-1]:.3f}, 접촉 접선 변위 평균 {np.mean(td) if td else float('nan'):.2e}, "
         f"최소 d {min(r['d_min']):.4f}")
    if "detF_contact" in r:
        dv = [v for v in r["detF_contact"] if v is not None]
        s += f", 접촉 det F 평균 최소 {min(dv) if dv else float('nan'):.3e}"
    print(s, flush=True)


gts, res = {}, {}
for s in a.gt:
    nm, dr = s.split("=", 1)
    fs = sorted(glob.glob(os.path.join(dr, "sim_*.h5")))[: a.frames + 1]
    X = np.stack([np.array(h5py.File(f)["x"], dtype=np.float64) for f in fs])
    gts[nm] = X
    res["gt_" + nm] = stats(X); summ("정답 " + nm, res["gt_" + nm])
for s in a.dump:
    nm, rest = s.split("=", 1)
    pt, g = rest.rsplit(":", 1)
    D = torch.load(pt, map_location="cpu", weights_only=False)
    Pr = D["pred"].double().numpy()[: a.frames]
    X0 = gts[g][0]
    if Pr.shape[1] != X0.shape[0]:
        raise SystemExit(f"{nm}: 입자 수 {Pr.shape[1]} != 정답 {X0.shape[0]}")
    F = D.get("F")
    F = F[: a.frames] if F is not None else None
    res[nm] = stats(np.concatenate([X0[None], Pr], 0), F); res[nm]["gt"] = g
    summ(nm, res[nm]); del D, F
json.dump(res, open(a.out, "w"))
print(f"[저장] {a.out}", flush=True)
