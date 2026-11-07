"""절 CM23 (사용자 요청 11-07): 수박 손상 변수 logJp 상위 10·20·30 % 입자 범주 (wm_render.py --cat_dir 용) -- 결과 불변 후처리.

  python exe/wm_dmgtop.py --gt_jp DIR --out_gt DIR [--ours RUN --out_ours DIR --out_ours_gtthr DIR] --json OUT.json [--pad 100]
- 프레임마다 그 프레임 logJp 분포 (유한 값만; 비유한 logJp 는 범주 4 로 따로, 백분위 계산에서 뺌) 의 90·80·70 백분위 문턱 q90·q80·q70:
  범주 1 = v > q90 (상위 10 %), 2 = q80 < v ≤ q90, 3 = q70 < v ≤ q80, 0 = 나머지, 4 = 비유한. 값이 같은 입자가 많으면 (예: 처음 값 −0.04) 실제 몫은 10 % 와 다를 수 있음 -> 실제 몫을 json 에.
- 정답 = GF 다시 돌린 궤적의 particle_Jp (절 CM13.4 wmJd gt_jp/sc_<t>.npy, t = 0..100).
- 우리 = RUN/lj.pt (fem_ao_rt_sim --dump_lj, 프레임 1..L) + 프레임 0 은 처음 값 −0.04; 발산 판은 마지막 유한 프레임을 --pad 까지 되풀이 (영상의 멈춘 칸).
  --out_ours_gtthr: 같은 프레임 정답 문턱 (q90·q80·q70) 을 우리 값에 그대로 적용한 범주.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--gt_jp", required=True)
ap.add_argument("--out_gt", required=True)
ap.add_argument("--ours", default="")
ap.add_argument("--out_ours", default="")
ap.add_argument("--out_ours_gtthr", default="")
ap.add_argument("--json", required=True)
ap.add_argument("--lj0", type=float, default=-0.04)
ap.add_argument("--pad", type=int, default=100)
a = ap.parse_args()


def thr_of(v):
    f = np.isfinite(v)
    if not f.any():
        return [float("nan")] * 3, f
    return [float(x) for x in np.quantile(v[f], [0.9, 0.8, 0.7])], f


def cat_of(v, thr):
    f = np.isfinite(v)
    c = np.zeros(v.shape[0], np.int8)
    q90, q80, q70 = thr
    with np.errstate(invalid="ignore"):
        c[f & (v > q70)] = 3
        c[f & (v > q80)] = 2
        c[f & (v > q90)] = 1
    c[~f] = 4
    n = v.shape[0]
    return c, dict(frac=[float((c == k).sum()) / n for k in (1, 2, 3)], nf=float((~f).sum()) / n)


J = dict(gt=[], ours=[], ours_gtthr=[])
os.makedirs(a.out_gt, exist_ok=True)
G = sorted(glob.glob(f"{a.gt_jp}/sc_*.npy"))
GT_THR = []
for p in G:
    t = int(os.path.basename(p)[3:7])
    v = np.load(p).astype(np.float64)
    thr, _ = thr_of(v)
    c, st = cat_of(v, thr)
    np.save(f"{a.out_gt}/cat_{t:04d}.npy", c)
    GT_THR.append(thr)
    J["gt"].append(dict(t=t, thr=thr, **st))
if a.ours:
    LJ = torch.load(f"{a.ours}/lj.pt", map_location="cpu", weights_only=False)["logJp"].numpy()
    P = torch.load(f"{a.ours}/pred.pt", map_location="cpu", weights_only=False, mmap=True)["pred"]
    L, N = LJ.shape
    nfin = 0
    while nfin < L and bool(torch.isfinite(P[nfin]).all()):
        nfin += 1
    nfin = max(nfin, 1)
    for d in (a.out_ours, a.out_ours_gtthr):
        os.makedirs(d, exist_ok=True)
    for t in range(0, a.pad + 1):
        v = np.full(N, a.lj0) if t == 0 else LJ[min(t, nfin) - 1].astype(np.float64)
        thr, _ = thr_of(v)
        c, st = cat_of(v, thr)
        np.save(f"{a.out_ours}/cat_{t:04d}.npy", c)
        J["ours"].append(dict(t=t, src=min(t, nfin), thr=thr, **st))
        gthr = GT_THR[min(t, len(GT_THR) - 1)]
        c2, st2 = cat_of(v, gthr)
        np.save(f"{a.out_ours_gtthr}/cat_{t:04d}.npy", c2)
        J["ours_gtthr"].append(dict(t=t, thr=gthr, **st2))
    J["ours_nfin"] = nfin
json.dump(J, open(a.json, "w"), indent=0)
for k in ("gt", "ours", "ours_gtthr"):
    if J[k]:
        r = J[k][-1]
        print(f"[{k}] 마지막 t {r['t']} 문턱 {['%.4f' % x for x in r['thr']]} 몫 {['%.3f' % x for x in r['frac']]} 비유한 {r['nf']:.3f}", flush=True)
