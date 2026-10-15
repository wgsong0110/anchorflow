"""한 실행이 안정인지 (stab_metrics.py 와 같은 판정): 60 프레임 끝 + 총에너지 변화 최대 ≤ 1 + 바닥 관통 ≤ 0.5 L.
바닥 없는 공식 씬은 기준 궤적(PG N1024) 최대 총에너지 대비 (아래 주석).
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
_fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"]
X = r["X"]
if not _fl:
    # 바닥이 없는 공식 씬 (plane·pillow2sofa): 경계조건이 에너지를 넣어 처음 에너지가 0 일 수 있다. 낙하 씬의 판정
    # "(E - E0)/|E0| ≤ 1" 에서 E0 를 **기준 궤적 (PG N1024) 의 최대 총에너지**로 바꾼다 (낙하 씬에서는 E0 가 곧 최대라 같은 판정).
    # 관통 판정은 바닥이 없어 뺀다.
    ref = sm.load(f"{root}/res", "pg", 1024)
    if ref is None or ref["status"] != "끝":
        print("-"); sys.exit(0)
    _E = lambda rr: rr["rows"]["ke"] + rr["rows"]["pe"]
    e_ref = float(np.nanmax(np.abs(_E(ref))))
    E = _E(r)
    ok = bool(np.isfinite(E).all()) and float(np.max((E - e_ref) / e_ref)) <= 1.0 and len(E) == len(_E(ref))
    print(1 if ok else 0); sys.exit(0)
zf = float(_fl[0]["point"][2])
L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
# 판정 기준은 모든 방법·모든 실행에 같게: 총에너지 = 운동 + 위치 (탄성에너지 ee 는 10-12 이후 실행에만 기록돼 섞으면
# 같은 궤적도 판정이 갈린다 -- 10-14 GF 사고). 정수 스캔 때 판정과 같다.
if r["rows"] is not None:
    r["rows"] = {k: v for k, v in r["rows"].items() if k != "ee"}
mt = sm.metrics(r, None, L, zf, float(cfg["g"][2]), float(cfg["frame_dt"]), 1.0, float(cfg["grid_lim"]) / int(cfg["n_grid"]))
ok = np.isfinite(mt.get("e_drift_max", 0.0)) and mt.get("e_drift_max", 0.0) <= 1.0 and (mt.get("pen_depth_max") or 0.0) <= 0.5
print(1 if ok else 0)
