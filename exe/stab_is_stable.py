"""한 실행이 안정인지 (stab_metrics.py 와 같은 판정): 60 프레임 끝 + 총에너지 변화 최대 ≤ 1 + 바닥 관통 ≤ 0.5 L.
출력: 1 (안정) / 0 (발산) / - (결과 없음).   python exe/stab_is_stable.py <res 폴더> <방법> <N> [형상 폴더]
형상 폴더(config.json 이 있는 곳)를 안 주면 res 폴더의 상위로 본다."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stab_metrics as sm                                   # noqa: E402

d, m, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = sm.load(d, m, n)
if r is None or r["status"] == "진행":
    print("-"); sys.exit(0)
if r["status"] != "끝":
    print(0); sys.exit(0)
root = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(d.rstrip("/"))
cfg = json.load(open(f"{root}/config.json"))
zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
X = r["X"]
L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
mt = sm.metrics(r, None, L, zf, float(cfg["g"][2]), float(cfg["frame_dt"]), 1.0, float(cfg["grid_lim"]) / int(cfg["n_grid"]))
ok = np.isfinite(mt.get("e_drift_max", 0.0)) and mt.get("e_drift_max", 0.0) <= 1.0 and (mt.get("pen_depth_max") or 0.0) <= 0.5
print(1 if ok else 0)
