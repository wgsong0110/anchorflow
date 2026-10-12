"""서브스텝 안정성: 진행 중 실행(체크포인트)과 끝난 실행(npz)의 프레임별 곡선을 JSON 으로 (그림·보고용).

프레임마다 (운동+위치 에너지)/처음, 최대 속력, det F 최소, 부분표본 무게중심 z, PG N1024 대비 부분표본 위치 오차 / 물체 크기.
모든 방법이 같은 부분표본 번호(stabstat.subset)를 쓰므로 위치를 바로 비교한다.

  python exe/stab_curves.py --out /home/dkta/work/stab/curves.json
"""
import argparse
import glob
import json
import os

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))   # 체크포인트 안 객체 복원용

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--out", required=True)
a = ap.parse_args()
K = ("ke", "pe", "nan", "out", "vmax", "detneg", "detmin")


def load(path):
    """-> (행 [T,7], 부분표본 위치 [T,S,3], 끝남 여부)"""
    if path.endswith(".npz"):
        z = np.load(path)
        keys = list(z["keys"]) if "keys" in z else list(K)
        st = z["stats"]
        rows = np.stack([st[:, keys.index(k)] if k in keys else np.full(len(st), np.nan) for k in K], 1)
        return rows, z["x"], True
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("stab"):                                   # rep_ip (우리)
        R, X = ck["stab"]["STROWS"], ck["stab"]["STX"]
    else:                                                # stab_mpm (i-PG 등)
        R, X = ck["rows"], ck["XS"]
    rows = np.array([[r.get(k, np.nan) for k in K] for r in R])
    return rows, np.stack([np.asarray(x) for x in X]), False


out = {}
for S in ("lego", "mic", "ficus"):
    res = f"{a.root}/{S}/res"
    ref = next((p for p in (f"{res}/pg_N1024r.npz", f"{res}/pg_N1024.npz") if os.path.exists(p)), None)
    runs = {}
    for p in sorted(glob.glob(f"{res}/*_N*.npz") + glob.glob(f"{res}/*_N*.ckpt.pt")):
        b = os.path.basename(p)
        name = b.split(".")[0]
        if name in runs and runs[name]["done"]:
            continue
        if b.endswith(".ckpt.pt") and os.path.exists(f"{res}/{name}.npz"):
            continue
        rows, X, done = load(p)
        runs[name] = dict(done=done, rows=rows, X=X)
    rX = load(ref)[1] if ref else None
    L = float(np.linalg.norm(rX[0].max(0) - rX[0].min(0))) if rX is not None else 1.0
    out[S] = {}
    for name, r in runs.items():
        rows, X = r["rows"], r["X"]
        e = rows[:, 0] + rows[:, 1]
        T = min(len(X), len(rX)) if rX is not None else 0
        err = [float(np.sqrt(((X[t] - rX[t]) ** 2).sum(1)).mean() / L) for t in range(T)]
        # 자기 방법 N1024 (진행 중이면 체크포인트) 대비 -- 안정성 판정의 기준
        own = runs.get(name.split("_N")[0] + "_N1024") or runs.get(name.split("_N")[0] + "_N1024r")
        To = min(len(X), len(own["X"])) if own is not None else 0
        err_own = [float(np.sqrt(((X[t] - own["X"][t]) ** 2).sum(1)).mean() / L) for t in range(To)]
        out[S][name] = dict(done=r["done"], frames=len(rows), energy=(e / e[0]).tolist(), vmax=rows[:, 4].tolist(),
                            err_vs_own1024=err_own,
                            detmin=rows[:, 6].tolist(), comz=[float(x[:, 2].mean()) for x in X], err_vs_pg1024=err)
        print(f"{S} {name}: {'끝' if r['done'] else '진행'} {len(rows)} 프레임, PG1024 대비 마지막 {err[-1] * 100 if err else float('nan'):.3f}%", flush=True)
json.dump(out, open(a.out, "w"))
print("CURVES_DONE", flush=True)
