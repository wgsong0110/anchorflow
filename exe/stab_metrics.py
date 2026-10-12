"""서브스텝 안정성: N 에 따른 지표 (사용자 2026-10-12 지정). 이미 끝난 결과(npz)와 진행 중(체크포인트)을 그대로 읽는다.

지표 (모든 방법 같은 정의, 물체 크기 L = 0 프레임 부분표본 대각선):
  에너지 드리프트   총에너지 E = 운동 + 중력 위치 + 탄성(고정 공회전, 같은 E·ν). max_t (E_t − E_0)/E_0 와 마지막 값.
                    탄성 에너지는 2026-10-12 이후 실행에만 있다 (예전 PG·GF 는 운동+중력만, 표시해 둔다)
  반전·부피 이상    det F ≤ 0 비율의 최대, |J−1| 평균의 최대, |J−1|>0.1 비율의 최대 (뒤 둘은 새 실행만)
  바닥 관통         바닥 아래 입자 비율의 최대, 최대 깊이 / L (부분표본 32,768 점의 위치로 -- 모든 실행 공통)
  떨림(jitter)      위치 2 차 차분 |x_{t+1} − 2x_t + x_{t−1}| / L 의 평균, 입자 속도 시계열 스펙트럼에서 위쪽 절반 주파수 에너지 비율
  자유낙하 운동량   접촉 전 프레임에서 운동량 크기 |p| 와 이론값 M g t 의 상대 오차 최대. 자유낙하는 강체 병진이라
                    |p| = sqrt(2 M KE) (KE 는 모든 실행에 있다 -- 예전 실행도 같은 자로). M = 밀도·dx³·점유 셀 수(시뮬과 같은 질량).
                    접촉 = 부분표본 최저점이 바닥 + 1.5 dx 아래로 내려온 첫 프레임 (PG 충돌체는 바닥 위 격자 노드에서 멈춘다)
  MPM 대비 CD       PG N1024 부분표본과의 대칭 Chamfer (평균 최근접 거리) / L 의 프레임 평균
자기 관통(self-penetration)은 계산하지 않는다 (표면 정보가 없어 정의가 방법마다 달라진다).

  python exe/stab_metrics.py --out /home/dkta/work/stab/metrics.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--out", required=True)
ap.add_argument("--nmax", type=int, default=128)
a = ap.parse_args()
NS = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= a.nmax]


def load(res, m, n):
    """-> dict(status, rows{key:[T]}, X[T,S,3]) 또는 None (실행 없음)"""
    npz, log = f"{res}/{m}_N{n}.npz", f"{res}/{m}_N{n}.log"
    ck = next((p for p in (f"{res}/{m}_N{n}.npz.ckpt.pt", f"{res}/{m}_N{n}.main.npz.ckpt.pt") if os.path.exists(p)), None)
    if os.path.exists(npz):
        z = np.load(npz)
        keys = [str(k) for k in z["keys"]]
        rows = {k: z["stats"][:, i] for i, k in enumerate(keys)}
        st = "끝"
        if np.nanmax(rows["nan"]) > 0 or len(z["x"]) < 61:
            st = "발산"
        return dict(status=st, rows=rows, X=z["x"])
    if ck:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        R, X = (c["stab"]["STROWS"], c["stab"]["STX"]) if c.get("stab") else (c["rows"], c["XS"])
        keys = sorted({k for r in R for k in r})
        return dict(status="진행", rows={k: np.array([r.get(k, np.nan) for r in R]) for k in keys},
                    X=np.stack([np.asarray(x) for x in X]))
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        if re.search(r"illegal memory|Error|error|Traceback|발산", txt):
            return dict(status="발산", rows=None, X=None)
    return None


def metrics(r, ref, L, zf, g, dt, M, dxg):
    rows, X = r["rows"], r["X"]
    out = {}
    if rows is not None:
        E = rows["ke"] + rows["pe"] + (np.nan_to_num(rows["ee"]) if "ee" in rows and np.isfinite(rows["ee"]).any() else 0.0)
        out["e_drift_max"] = float(np.nanmax((E - E[0]) / abs(E[0])))
        out["e_drift_last"] = float((E[-1] - E[0]) / abs(E[0]))
        out["e_has_elastic"] = bool("ee" in rows and np.isfinite(rows["ee"]).any())
        out["detneg_max"] = float(np.nanmax(rows["detneg"])) if "detneg" in rows else None
        for k in ("jdev", "jdev10"):
            out[k + "_max"] = float(np.nanmax(rows[k])) if k in rows and np.isfinite(rows[k]).any() else None
    if X is not None and len(X) >= 3:
        z = X[..., 2]
        out["pen_ratio_max"] = float((z < zf).mean(1).max())
        out["pen_depth_max"] = float(max((zf - z).max(), 0.0) / L)
        d2 = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1)
        out["jitter_d2"] = float(np.nanmean(d2) / L)
        v = X[1:] - X[:-1]                                         # [T-1,S,3]
        if len(v) >= 8:
            P = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2   # 시간 축 스펙트럼
            P = P.sum(-1)                                          # [F,S]
            h = P.shape[0] // 2
            out["jitter_hf"] = float(np.nanmean(P[h:].sum(0) / np.maximum(P.sum(0), 1e-30)))
        # 자유낙하: 접촉(최저점 < 바닥 + 1.5 dx) 전 프레임에서 |p| = sqrt(2 M KE) 와 M g t
        hit = np.where(z.min(1) < zf + 1.5 * dxg)[0]
        T_c = int(hit[0]) if len(hit) else len(z)
        out["freefall_frames"] = T_c
        if rows is not None:
            ke = rows["ke"]
            errs = [abs(np.sqrt(2 * M * max(ke[t], 0.0)) - M * abs(g) * t * dt) / (M * abs(g) * t * dt)
                    for t in range(1, min(T_c, len(ke)))]
            out["freefall_mom_err"] = float(max(errs)) if errs else None
        if ref is not None:
            T = min(len(X), len(ref))
            cds = []
            for t in range(1, T):
                if not np.isfinite(X[t]).all():
                    break
                ka, kb = cKDTree(X[t]), cKDTree(ref[t])
                cds.append(0.5 * (kb.query(X[t])[0].mean() + ka.query(ref[t])[0].mean()) / L)
            out["cd_mpm"] = float(np.mean(cds)) if cds else None
            out["cd_frames"] = len(cds)
    return out


res_all = {}
for S in ("lego", "mic", "ficus"):
    res = f"{a.root}/{S}/res"
    cfg = json.load(open(f"{a.root}/{S}/config.json"))
    zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
    g = float(cfg["g"][2]); dt = float(cfg["frame_dt"])
    import h5py
    with h5py.File(f"{a.root}/{S}/init/sim_0000000000.h5") as h:
        x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
    dxg = float(cfg["grid_lim"]) / int(cfg["n_grid"])
    M = float(cfg["density"]) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))
    refr = load(res, "pg", 1024)
    ref = refr["X"] if refr else None
    L = float(np.linalg.norm(ref[0].max(0) - ref[0].min(0)))
    res_all[S] = {}
    for m in ("pg", "gf", "ipg", "ours"):
        res_all[S][m] = {}
        for n in NS:
            r = load(res, m, n)
            if r is None:
                continue
            d = dict(status=r["status"])
            d.update(metrics(r, ref, L, zf, g, dt, M, dxg))
            # 결과 파일이 있어도 값이 터졌으면 발산으로 (GF N8 은 NaN 없이 1e30 대로 커졌다)
            if d["status"] == "끝" and (not np.isfinite(d.get("e_drift_max", 0.0)) or d.get("e_drift_max", 0.0) > 1.0
                                       or (d.get("pen_depth_max") or 0.0) > 0.5):
                d["status"] = "발산"
            res_all[S][m][n] = d
            print(S, m, n, d["status"], {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items() if k != "status"}, flush=True)
json.dump(res_all, open(a.out, "w"), indent=1)
print("METRICS_DONE", flush=True)
