"""절 CM19: 수박 손상 판 한 실행의 손상 통계 + 색 렌더용 프레임별 값 파일 (인스턴스에서, 결과 불변 후처리).

  python exe/wmdmg_stats.py --run DIR [--sc_lj DIR] [--sc_d DIR] [--pad 100]
- DIR/lj.pt (fem_ao_rt_sim --dump_lj: logJp [L, N], 손상 판이면 d [L, N]) 와 DIR/pred.pt (위치 유한 여부 = 분모) -> DIR/dmg_stats.json:
  프레임 (1..L) 마다 logJp > 0 몫 (NACC 응집력 0 = 손상), > 0.4 몫 (GF 숨김 문턱), d > 0.5 · d > 0.1 몫, 합집합 (d > 0.5 또는 logJp > 0) 몫, d 평균.
  분모 = 그 프레임 위치가 유한한 입자 (절 CM13.4 와 같은 규칙).
- --sc_lj / --sc_d: wm_render.py --scalar_dir 용 sc_<t:04d>.npy (t = 0 은 처음 값 logJp −0.04 · d 0). 발산 판은 마지막 유한 프레임을 --pad 프레임까지 되풀이 (영상의 멈춘 칸).
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--sc_lj", default="")
ap.add_argument("--sc_d", default="")
ap.add_argument("--lj0", type=float, default=-0.04)
ap.add_argument("--pad", type=int, default=100)
a = ap.parse_args()
LJ = torch.load(f"{a.run}/lj.pt", map_location="cpu", weights_only=False)
lj = LJ["logJp"].numpy(); dd = LJ["d"].numpy() if "d" in LJ else None
P = torch.load(f"{a.run}/pred.pt", map_location="cpu", weights_only=False, mmap=True)["pred"]
L, N = lj.shape
O = dict(frames=[], n_fin_frames=0)
nfin = 0
for i in range(L):
    fin = np.isfinite(P[i].numpy()).all(1)
    if fin.all():
        nfin = i + 1
    n = max(int(fin.sum()), 1)
    r = dict(t=i + 1, n=int(fin.sum()), lj0=float((lj[i][fin] > 0).sum()) / n, lj04=float((lj[i][fin] > 0.4).sum()) / n)
    if dd is not None:
        d = np.nan_to_num(dd[i][fin], nan=0.0)
        r.update(d05=float((d > 0.5).sum()) / n, d01=float((d > 0.1).sum()) / n, dmean=float(d.mean()) if d.size else None,
                 uni=float(((d > 0.5) | (lj[i][fin] > 0)).sum()) / n)
    O["frames"].append(r)
O["n_fin_frames"] = nfin
json.dump(O, open(f"{a.run}/dmg_stats.json", "w"), indent=1)
last = O["frames"][-1] if O["frames"] else {}
print(f"[손상 통계] {a.run} 프레임 {L} (전부 유한 {nfin}) 마지막: logJp>0 {last.get('lj0', 0):.4f} d>0.5 {last.get('d05', float('nan')):.4f} 합집합 {last.get('uni', float('nan')):.4f}", flush=True)


def sc(out, arr, v0):
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/sc_0000.npy", np.full(N, v0, np.float32))
    n = max(nfin, 1)
    for t in range(1, a.pad + 1):
        i = min(t, n) - 1
        np.save(f"{out}/sc_{t:04d}.npy", np.nan_to_num(arr[i].astype(np.float32), nan=v0))


if a.sc_lj:
    sc(a.sc_lj, lj, a.lj0)
if a.sc_d and dd is not None:
    sc(a.sc_d, dd, 0.0)
