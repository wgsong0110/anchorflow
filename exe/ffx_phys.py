#!/usr/bin/env python3
"""[final_fix_phys] 정답 없이 재는 물리 지표 (파레토 방법 x 장면 한 칸) -> json.

정의 = 안정성 실험 stabfinal 과 같은 코드 (lib/anchorflow/stabstat.frame_stats 프레임 통계 + exe/stab_metrics.metrics 궤적 지표):
  뒤집힘      det F <= 0 입자 비율의 프레임 최대 (detneg_max), det F 최소 (detmin, 전 프레임)
  바닥 관통   부분표본 (stabstat.subset 32768 점, 씨앗 0) 위치에서 바닥 아래 입자 비율 최대 (pen_ratio_max), 최대 깊이 / L (pen_depth_max)
  에너지      E = 운동 + 중력 위치 (stabfinal 판정과 같이 탄성 에너지 뺌). e_drift_max = max_t (E_t - E_0) / |E_0|, e_ratio_last = E_T / E_0
  자유낙하    접촉 (부분표본 최저점 < 바닥 + 1.5 dx) 전 프레임에서 |sqrt(2 M KE_t) - M |g| t h| / (M |g| t h) 최대 (freefall_mom_err)
  떨림        부분표본 위치 2 차 차분 |x_{t+1} - 2 x_t + x_{t-1}| / L 의 평균 (jitter_d2), 속도 스펙트럼 위쪽 절반 에너지 비율 (jitter_hf)
  부피        |J - 1| 평균의 프레임 최대 (jdev_max, J = det F), 전체 부피 변화 dV_t = sum vol_i (J_i - 1) / sum vol_i 의 마지막 값·절댓값 최대
L = 0 프레임 부분표본 대각선. M·질량 = deform_model.traj_mass (밀도 x dx^3 / 칸 입자 수, 시뮬과 같은 질량).
바닥 없는 장면 (--floor none): 관통·자유낙하 = 해당 없음 (None), 위치 에너지 기준 z = 0 프레임 최저 입자 z.
속도 = 위치 중앙 차분 (x_{t+1} - x_{t-1}) / 2h (모든 방법 같은 자: 베이스라인 덤프에 속도가 없음; 등가속 운동에서 정확), 0 프레임 = 자료의 처음 속도, 마지막 프레임 = 뒤 차분.
F 가 없는 표현 (스프링-질량: PhysTwin 덤프 F = 렌더용 누적 회전, Spring-Gaus = 단위 행렬) 은 --no_F -> 뒤집힘·부피 = 해당 없음.

  python exe/ffx_phys.py --method ours --scene wolf --N 1 --data wolf.pt --pred RUN/pred.pt --floor vel --out phys.json
  python exe/ffx_phys.py --method mpm --scene wolf --N 1999 --data wolf.pt --gt_h5 GTDIR --floor vel --out phys.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import stab_metrics as sm  # noqa: E402
from anchorflow import stabstat as ss  # noqa: E402
from anchorflow import deform_model as DM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--N", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--pred", default="", help="덤프 pred.pt {pred [T,N,3] = 프레임 1..T, F [T,N,3,3]} (프레임 0 = 자료 x[0], F = I)")
ap.add_argument("--gt_h5", default="", help="정답 렌더용 궤적 폴더 (sim_*.h5 의 x, F = F_sim R; meta.json 의 R) -- 위치는 자료 x, F 는 h5")
ap.add_argument("--frames", type=int, default=0, help="0 = 덤프 길이 전부")
ap.add_argument("--floor", default="vel")
ap.add_argument("--no_F", action="store_true")
ap.add_argument("--note", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"]); g = np.array(cfg.get("g", [0.0, 0.0, -9.8]), np.float64)
dxg = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
m = DM.traj_mass(d, "cpu").double().numpy()
Np = m.shape[0]
vol = m / float(cfg["density"])
XD = d["x"]
x0 = XD[0].double().numpy()
HAS_FLOOR = a.floor != "none"
zf = None
if HAS_FLOOR:
    fl = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"]
    zf = float(fl[0]["point"][2]) if fl else float(cfg.get("floor_z", 0.0))
zref = zf if HAS_FLOOR else float(x0[:, 2].min())
if "v" in d and d["v"] is not None:
    v0 = d["v"][0].double().numpy()
else:
    v0 = np.tile(np.array(cfg.get("init_velocity", [0.0, 0.0, 0.0]), np.float64), (Np, 1))

# ---- 위치·F 공급자 ----
note = [a.note] if a.note else []
if a.pred:
    P = torch.load(a.pred, map_location="cpu", weights_only=False)
    PX, PF = P["pred"], P.get("F")
    T = PX.shape[0] if not a.frames else min(a.frames, PX.shape[0])
    if P.get("part"):
        note.append(f"덤프 일부 (발산 프레임 {P.get('diverged')})")

    def X(t):
        return x0 if t == 0 else PX[t - 1].double().numpy()

    def Fm(t):
        if a.no_F or PF is None:
            return None
        return np.tile(np.eye(3), (Np, 1, 1)) if t == 0 else PF[t - 1].double().numpy().reshape(-1, 3, 3)
else:
    import h5py
    R = np.array(json.load(open(f"{a.gt_h5}/meta.json"))["R"], np.float64)
    H5 = sorted(glob.glob(f"{a.gt_h5}/sim_*.h5"))
    T = min(len(H5) - 1, XD.shape[0] - 1)
    if a.frames:
        T = min(T, a.frames)

    def X(t):
        return XD[t].double().numpy()

    def Fm(t):
        if a.no_F:
            return None
        with h5py.File(H5[t]) as f_:
            Fr = np.array(f_["F"], np.float64)
        Fr = Fr.T if (Fr.shape[0] == 9 and Fr.shape[1] != 9) else Fr
        return Fr.reshape(-1, 3, 3) @ R.T                                           # F_sim = F_render R^T (det 같음)

SUB = ss.subset(Np)
rows, XS, dV = [], [], []
xp, xc = None, X(0)
for t in range(T + 1):
    xn = X(t + 1) if t < T else None
    if t == 0:
        v = v0
    elif xn is not None:
        v = (xn - xp) / (2 * h)
    else:
        v = (xc - xp) / h
    F = Fm(t)
    st = ss.frame_stats(xc, v, F, m, g, zref)
    rows.append(st)
    XS.append(xc[SUB].astype(np.float32))
    if F is not None:
        J = np.linalg.det(F)
        dV.append(float((vol * (J - 1)).sum() / vol.sum()))
    xp, xc = xc, xn
keys = sorted({k for r in rows for k in r})
RW = {k: np.array([r.get(k, np.nan) for r in rows], np.float64) for k in keys}
XS = np.stack(XS)
Lc = float(np.linalg.norm(XS[0].max(0) - XS[0].min(0)))
M = float(m.sum())
rr = dict(rows={k: v for k, v in RW.items() if k != "ee"}, X=XS)
mt = sm.metrics(rr, None, Lc, zf if HAS_FLOOR else zref, g[2], h, M, dxg)

E = RW["ke"] + RW["pe"]
p0 = np.array([RW["px"][0], RW["py"][0], RW["pz"][0]])
out = dict(method=a.method, scene=a.scene, N=a.N, frames=int(T), n_particles=int(Np), L=Lc, floor_z=zf, pe_ref_z=zref,
           detneg_max=None, detmin=None, jdev_max=None, dV_last=None, dV_absmax=None,
           pen_ratio_max=None, pen_depth_max=None,
           e_drift_max=mt.get("e_drift_max"), e_ratio_last=float(E[-1] / E[0]) if E[0] != 0 else None, E0=float(E[0]),
           freefall_mom_err=None, freefall_frames=None,
           jitter_d2=mt.get("jitter_d2"), jitter_hf=mt.get("jitter_hf"),
           nan_max=float(np.nanmax(RW["nan"])))
if not (a.no_F or (a.pred and PF is None)):
    out.update(detneg_max=float(np.nanmax(RW["detneg"])), detmin=float(np.nanmin(RW["detmin"])), jdev_max=float(np.nanmax(RW["jdev"])),
               dV_last=dV[-1], dV_absmax=float(np.max(np.abs(dV))))
else:
    note.append("F 없음 (스프링-질량: 변형 기울기가 표현에 없음) -> 뒤집힘·부피 해당 없음")
if HAS_FLOOR:
    out.update(pen_ratio_max=mt.get("pen_ratio_max"), pen_depth_max=mt.get("pen_depth_max"), freefall_frames=mt.get("freefall_frames"))
    if np.linalg.norm(p0) <= 1e-9 * M:
        out["freefall_mom_err"] = mt.get("freefall_mom_err")
    else:                                                                            # 처음 속도가 0 이 아니면 stab 식 (처음 운동량 0 가정) 대신 벡터 운동량
        Tc = mt.get("freefall_frames") or 0
        gv = np.array([0.0, 0.0, g[2]])
        errs = [np.linalg.norm(np.array([RW["px"][t], RW["py"][t], RW["pz"][t]]) - (p0 + M * gv * t * h)) / (M * abs(g[2]) * t * h) for t in range(1, min(Tc, T + 1))]
        out["freefall_mom_err"] = float(max(errs)) if errs else None
        note.append("처음 속도 0 아님 -> 자유낙하 = |p_t - (p_0 + M g t h)| / (M |g| t h)")
else:
    note.append("바닥 없음 (--floor none) -> 관통·자유낙하 해당 없음; 위치 에너지 기준 = 0 프레임 최저 입자 z")
out["note"] = "; ".join(note)
out["dV_series"] = dV
out["E_series"] = E.tolist()
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_series")}, ensure_ascii=False), flush=True)
