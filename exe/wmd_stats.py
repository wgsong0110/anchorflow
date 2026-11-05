"""절 CM13.4: 손상 변수 (NACC 경화 logJp) 통계 + 우리 판 색 렌더용 프레임별 값 파일.

- 우리 판: lj.pt (fem_ao_rt_sim --dump_lj, 프레임 1..L) -> sc_<t:04d>.npy (프레임 0 = 처음 logJp, --lj0)
- 정답 다시 돌림 (gf_wm_jp.py): jp/sc_<t:04d>.npy 그대로
- 프레임별: logJp > 0 몫 (응집력 p0 = κ(1e-5 + sinh(ξ max(−logJp, 0))) 이 사라진 입자 = 손상 입자), > 0.4 몫 (GF 러너가 렌더에서 숨기는 문턱),
  평균·백분위, 색 범위 밖 몫; 정답 다시 돌림 위치 대 원래 정답 h5 위치 차 (평균 거리 / 장면 크기 _ext, 우리 오차와 같은 잣대).
  몫의 분모 = 그 프레임 위치가 유한한 입자 (GF 는 진행하며 입자를 비유한으로 떨굼).
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ours_lj", required=True)
ap.add_argument("--ours_pred", required=True)
ap.add_argument("--ours_sc", required=True)
ap.add_argument("--lj0", type=float, default=-0.04)
ap.add_argument("--gt_jp", required=True)
ap.add_argument("--gt_new", required=True, help="다시 돌린 정답 h5 폴더")
ap.add_argument("--gt_orig", required=True, help="원래 정답 h5 폴더")
ap.add_argument("--data", required=True, help="wm.pt (_ext)")
ap.add_argument("--vmin", type=float, default=-0.2)
ap.add_argument("--vmax", type=float, default=0.4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
EXT = float(torch.load(a.data, map_location="cpu", weights_only=False, mmap=True)["_ext"])


def st(v, fin):
    v = v[fin]
    return dict(n=int(v.size), gt0=float((v > 0).mean()), gt04=float((v > 0.4).mean()), mean=float(v.mean()),
                p50=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)), mn=float(v.min()), mx=float(v.max()),
                below=float((v < a.vmin).mean()), above=float((v > a.vmax).mean()))


def h5x(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    return x.T if x.shape[0] == 3 else x


O = dict(ext=EXT, vmin=a.vmin, vmax=a.vmax, ours=[], gt=[], gt_pos=[])
LJ = torch.load(a.ours_lj, map_location="cpu", weights_only=False)["logJp"].numpy()
P = torch.load(a.ours_pred, map_location="cpu", weights_only=False, mmap=True)["pred"]
os.makedirs(a.ours_sc, exist_ok=True)
N = LJ.shape[1]
np.save(f"{a.ours_sc}/sc_0000.npy", np.full(N, a.lj0, np.float32))
O["ours"].append(st(np.full(N, a.lj0, np.float32), np.ones(N, bool)))
for i in range(LJ.shape[0]):
    np.save(f"{a.ours_sc}/sc_{i + 1:04d}.npy", LJ[i].astype(np.float32))
    O["ours"].append(st(LJ[i], np.isfinite(P[i].numpy()).all(1) & np.isfinite(LJ[i])))
fo = sorted(glob.glob(f"{a.gt_orig}/sim_*.h5")); fn = sorted(glob.glob(f"{a.gt_new}/sim_*.h5"))
for t in range(len(fn)):
    jp = np.load(f"{a.gt_jp}/sc_{t:04d}.npy"); xn = h5x(fn[t])
    O["gt"].append(st(jp, np.isfinite(xn).all(1) & np.isfinite(jp)))
    xo = h5x(fo[t]) if t < len(fo) else None
    if xo is not None and xo.shape != xn.shape:
        O['gt_pos'].append(dict(t=t, shape_new=list(xn.shape), shape_orig=list(xo.shape)))
    elif xo is not None:
        ok = np.isfinite(xo).all(1) & np.isfinite(xn).all(1)
        dd = np.linalg.norm(xn[ok] - xo[ok], axis=1)
        O["gt_pos"].append(dict(t=t, rel=float(dd.mean()) / EXT, max=float(dd.max()), n_ok=int(ok.sum()),
                                nf_new=int((~np.isfinite(xn).all(1)).sum()), nf_orig=int((~np.isfinite(xo).all(1)).sum())))
    print(f"[통계] 프레임 {t} 정답 > 0 몫 {O['gt'][-1]['gt0']:.4f} 우리 {O['ours'][t]['gt0'] if t < len(O['ours']) else float('nan'):.4f} 위치 차 {O['gt_pos'][-1].get('rel', float('nan')) if O['gt_pos'] else 0:.4%}", flush=True)
O["gt_frame0_vs_lj0"] = float(np.abs(np.load(f"{a.gt_jp}/sc_0000.npy") - a.lj0).max())
json.dump(O, open(a.out, "w"), indent=1)
print("[통계 저장]", a.out, "정답 0 프레임 logJp - lj0 최대", O["gt_frame0_vs_lj0"], flush=True)
