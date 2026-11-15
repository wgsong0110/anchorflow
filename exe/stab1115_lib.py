"""[stab1115] 공용: 씬 상수, 결과 읽기 (PG·i-PG·GF = stab_metrics.load 형식 npz, 우리 = 육면체 res.npz), 안정 판정 (기존 표와 같은 정의)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import stab_metrics as sm  # noqa: E402

ST = os.environ.get("S1115_ROOT", "/home/dkta/work/stab")
SCENES = ("ficus", "vasedeck", "plane", "pillow2sofa")
REPS = ("a", "b", "c")
POW = [2 ** i for i in range(12)]
NSHAPE = ("ship_plastic_lego", "mic_metal_lego")      # [stabfinal] 새 형상 장면 = 레고 L000 세팅 (정답 PG 사본 R2 datasets/anchorflow/nshape/gt_pg/<장면>), 80 프레임


def nframes(S):
    """[stabfinal] 장면 프레임 수 = config frame_num (기존 낙하 장면 60, 새 형상 80)."""
    try:
        return int(json.load(open(f"{ST}/{S}/config.json"))["frame_num"])
    except Exception:
        return 60


def res_dir(S, rep):
    return f"{ST}/{S}/res" if rep == "a" else f"{ST}/{S}/res_{rep}"


def hex_dir(S, n, rep):
    return f"{ST}/{S}/hex/{S}_N{n}_{rep}"


def hex_load(d, nf=60):
    p = f"{d}/res.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    keys = [str(k) for k in z["keys"]]
    rows = {k: z["stats"][:, i] for i, k in enumerate(keys)}
    st = "끝" if (len(z["x"]) >= nf + 1 and np.nanmax(rows["nan"]) == 0 and np.isfinite(z["x"]).all()) else "발산"
    stop = str(z["stop"]) if "stop" in z.files else ""
    if st == "발산" and "CUDA error: out of memory" in stop and len(z["x"]) <= 3:
        st = "메모리부족"
    return dict(status=st, rows=rows, X=z["x"], stop=stop, K=int(z["substeps"]))


def load(S, m, n, rep):
    if m == "ours":
        return hex_load(hex_dir(S, n, rep), nframes(S))
    return sm.load(res_dir(S, rep), m, n, nframes(S))


def scene_consts(S):
    cfg = json.load(open(f"{ST}/{S}/config.json"))
    zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
    return cfg, zf, float(cfg["g"][2]), float(cfg["frame_dt"]), float(cfg["grid_lim"]) / int(cfg["n_grid"])


def judge(r, S):
    """안정 = 전 프레임 (60 또는 80) 끝 + 운동+위치 에너지 변화 최대 <= 1 + 바닥 관통 <= 0.5 L (stabhex_an.judge 와 같음). -> (bool|None, 지표 dict)"""
    if r is None or r["status"] == "진행":
        return None, {}
    if r["status"] != "끝":
        return False, {}
    cfg, zf, g, dt, dxg = scene_consts(S)
    X = r["X"]
    L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
    rr = dict(r)
    rr["rows"] = {k: v for k, v in r["rows"].items() if k != "ee"} if r["rows"] is not None else None
    mt = sm.metrics(rr, None, L, zf, g, dt, 1.0, dxg)
    ok = bool(np.isfinite(mt.get("e_drift_max", 0.0)) and mt.get("e_drift_max", 0.0) <= 1.0 and (mt.get("pen_depth_max") or 0.0) <= 0.5)
    return ok, mt
