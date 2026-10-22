"""롤아웃 덤프의 처음 상태(x0)·정답 첫 프레임(gt[0])이 어느 lbdata 궤적 디렉토리와 맞는지 판정 -> GTDIR <경로>.

덤프 x0 = 학습 데이터 .pt 의 프레임 0, gt[0] = 프레임 1 (같은 .pt). 후보 디렉토리의 sim_0000000000.h5·sim_0000000001.h5 의 x 와
입자 수·최대 절대차를 비교해, 둘 다 1e-3 이내인 (.pt 가 half 라 반올림 차 ~4.9e-4) 후보를 고른다 (여럿이면 차가 가장 작은 것, 없으면 GTDIR 를 내지 않는다).

  python exe/allmodels_gtcheck.py --dump roll.pt --cand test=/home/dkta/work/lbdata/test/0000 --cand train=... --out c.json
"""
import argparse
import json

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--cand", action="append", required=True, help="이름=lbdata 궤적 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=1e-3)   # .pt 위치가 half 라 반올림 4.9e-4 (L000 대조 때도 같음)
a = ap.parse_args()
D = torch.load(a.dump, map_location="cpu", weights_only=False)
x0 = D["x0"].float().numpy(); g1 = D["gt"][0].float().numpy()
print(f"[덤프] {a.dump} tag {D.get('tag')} t0 {D.get('t0')} 입자 {x0.shape[0]} 프레임 {D['gt'].shape[0]} "
      f"F {None if D.get('F') is None else (tuple(D['F'].shape), str(D['F'].dtype))}", flush=True)
res = {}
for c in a.cand:
    nm, dr = c.split("=", 1)
    r = dict(dir=dr)
    try:
        with h5py.File(f"{dr}/sim_0000000000.h5") as h:
            h0 = np.array(h["x"], dtype=np.float32)
        with h5py.File(f"{dr}/sim_0000000001.h5") as h:
            h1 = np.array(h["x"], dtype=np.float32)
        r["n"] = int(h0.shape[0])
        if h0.shape == x0.shape:
            r["d0"] = float(np.abs(h0 - x0).max()); r["d1"] = float(np.abs(h1 - g1).max())
    except Exception as e:  # noqa: BLE001
        r["err"] = repr(e)
    res[nm] = r
    print(f"[후보] {nm} {dr}: 입자 {r.get('n')}  프레임0 최대차 {r.get('d0')}  프레임1 최대차 {r.get('d1')} {r.get('err', '')}", flush=True)
ok = {k: v for k, v in res.items() if "d0" in v and v["d0"] < a.tol and v["d1"] < a.tol}
best = min(ok, key=lambda k: ok[k]["d0"] + ok[k]["d1"]) if ok else None
json.dump(dict(dump=a.dump, cand=res, choice=best, tol=a.tol), open(a.out, "w"), ensure_ascii=False, indent=1)
if best:
    print(f"[판정] {best} (기준 최대차 < {a.tol})", flush=True)
    print(f"GTDIR {res[best]['dir']}", flush=True)
else:
    print("[판정] 맞는 후보 없음", flush=True)
