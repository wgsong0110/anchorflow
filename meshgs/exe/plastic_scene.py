"""실제 3DGS 씬에서 메시-가우시안(VR-GS 구조) 소성 시험.

합성 큐브가 아니라 진짜 가우시안 구름에서 같은 결론이 나오는지 본다:
  3DGS -> 점유 격자 -> 사면체 케이지(고정 위상) -> 가우시안 무게중심 결속
  -> 탄성 / 소성 FEM 롤아웃 -> MPM 정답과 비교 + 영상

측정: 뒤집힌 요소 비율의 시간 추이, 첫 뒤집힘 시점, 발산 시점, 잔류 변형.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "lib"))

import numpy as np
import torch

from meshgs.fem import TetFEM
from meshgs.tetcage import (bind_gaussians, build_tet_mesh, dilate_fill,
                            occupancy, skin)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--cameras", required=True)
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=48)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--sub", type=int, default=40, help="프레임당 FEM 서브스텝")
ap.add_argument("--E", type=float, default=None)
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--yields", type=float, nargs="+", default=[1e9, 1e2])
ap.add_argument("--damping", type=float, default=4.0)
ap.add_argument("--width", type=int, default=400)
ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.dreamphysics)
import warp as wp

wp.init()
from anchorflow import scene_setup
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev, frozen_weights=True,
                       rot_fallback=True, eig_floor=0.02)
EXT = float(sc.extent)
FRAME_DT = float(sc.cfg.get("frame_dt", 0.04))
E = a.E if a.E is not None else float(sc.cfg.get("E", 1e5)) * 1e7
NU = a.nu if a.nu is not None else float(sc.cfg.get("nu", 0.3))
RHO = float(sc.cfg.get("density", 200.0))
print(f"[씬] 재질 {sc.cfg.get('material')}, E {E:.3e}, nu {NU}, rho {RHO}, "
      f"물체 {EXT:.4f}", flush=True)

G0 = sc.pos[sc.keep]
occ, org, h = occupancy(G0, res=a.res)
occ = dilate_fill(occ, 1)
V0, T = build_tet_mesh(occ, org, h)
tid, bw = bind_gaussians(G0, V0, T)
print(f"[메시] 정점 {V0.shape[0]}, 사면체 {T.shape[0]}, h {h:.4f}, "
      f"결속 실패 {int((tid < 0).sum())}", flush=True)
err0 = float((skin(V0, T, tid, bw) - G0).norm(dim=-1).max())
print(f"[메시] 결속 복원 오차 {err0:.3e}", flush=True)

# 고정 정점: MPM 의 cuboid 경계와 같은 영역
FIXV = torch.zeros(V0.shape[0], dtype=torch.bool, device=dev)
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] != "cuboid":
        continue
    v = torch.tensor(bc.get("velocity", [0., 0., 0.]), device=dev)
    if float(v.abs().max()) > 0:
        continue
    c = torch.tensor(bc["point"], device=dev)
    s_ = torch.tensor(bc["size"], device=dev)
    FIXV |= ((V0 - c).abs() <= s_).all(-1)
print(f"[메시] 고정 정점 {int(FIXV.sum())}/{V0.shape[0]}", flush=True)

# --- MPM 정답과 같은 초기 속도 ---
T_ = MPMTeacher(sc, horizon=a.frames * FRAME_DT)
n_sub = max(1, int(round(FRAME_DT / float(sc.sub_dt))))
BASE = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        BASE = torch.tensor(bc["force"], device=dev)
if BASE is not None:
    f = BASE.unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
    dv = sc.impulse_dv(f)
    v0_p = (T_.w.unsqueeze(-1) * dv[T_.idx]).sum(1).contiguous()
else:
    v0_p = torch.zeros(T_.n, 3, device=dev)
T_._set(T_.pos_m.clone(), v0_p, T_.eye.clone(), torch.zeros_like(T_.eye))
truth = [T_.pos_m[sc.keep].clone()]
for _ in range(a.frames - 1):
    bad = False
    for k in range(n_sub):
        T_.solver.p2g2p(None, float(sc.sub_dt), device=T_.wp_dev)
        if (k + 1) % 4 == 0 and (not T_._in_domain() or not T_._vel_safe(4)):
            bad = True
            break
    if bad:
        break
    truth.append(T_.solver.export_particle_x_to_torch()[sc.keep].clone())
TR = torch.stack(truth)
NF = TR.shape[0]
print(f"[mpm] {NF} 프레임, 최대 변위 "
      f"{100 * float((TR - TR[0]).norm(dim=-1).max()) / EXT:.2f}% of 물체", flush=True)

# 가우시안 초기 속도를 정점 속도로 옮긴다 (최근접)
from scipy.spatial import cKDTree

_, nn = cKDTree(G0.detach().cpu().numpy()).query(V0.detach().cpu().numpy(), k=1)
vg = v0_p[sc.keep] if v0_p.shape[0] == sc.pos.shape[0] else v0_p
V_INIT = vg[torch.from_numpy(nn).long().to(dev)]
V_INIT[FIXV] = 0
print(f"[초기속도] 정점 평균 속력 {float(V_INIT.norm(dim=-1).mean()):.4f}", flush=True)

DT = FRAME_DT / a.sub
rows, traj = [], {}
for ys in a.yields:
    kind = "none" if ys >= 1e8 else ("drucker_prager"
                                     if sc.cfg.get("material") == "sand"
                                     else "von_mises")
    fem = TetFEM(V0, T, density=RHO, E=E, nu=NU, plastic=kind, yield_stress=ys,
                 damping=a.damping)
    V, vel = V0.clone(), V_INIT.clone()
    xs, first_inv, fail = [G0.clone()], -1, -1
    for fr in range(1, NF):
        for s in range(a.sub):
            V, vel = fem.step(V, vel, DT, fixed=FIXV)
            if not torch.isfinite(V).all():
                fail = fr * a.sub + s
                break
            iv, dmin, _ = fem.quality(V)
            if first_inv < 0 and iv > 0:
                first_inv = fr * a.sub + s
        if fail >= 0:
            break
        xs.append(skin(V, T, tid, bw))
    ok = fail < 0
    if ok:
        P = torch.stack(xs)
        e3 = float((P - TR[:P.shape[0]]).norm(dim=-1).mean()) / EXT
        iv, dmin, ar = fem.quality(V)
        res = float((V - V0).norm(dim=-1).mean()) / EXT
    else:
        P, e3, iv, dmin, res = None, float("nan"), float("nan"), float("nan"), float("nan")
    traj[ys] = P
    rows.append((ys, kind, ok, fail, first_inv, e3, iv, res))
    print(f"  yield {ys:8.1e} ({kind:14s}) {'완주' if ok else '발산'}  "
          f"실패스텝 {fail:6d}  첫뒤집힘 {first_inv:6d}  "
          f"3D오차 {100*e3:6.2f}%  뒤집힘 {100*iv:6.2f}%  잔류 {100*res:5.2f}%",
          flush=True)

json.dump([{"yield": r[0], "kind": r[1], "ok": r[2], "fail_step": r[3],
            "first_inversion": r[4], "err3d": r[5], "inverted": r[6],
            "residual": r[7]} for r in rows],
          open(os.path.join(a.out, "plastic_scene.json"), "w"), indent=1)
torch.save({"traj": {k: (v.cpu() if v is not None else None)
                     for k, v in traj.items()},
            "truth": TR.cpu()}, os.path.join(a.out, "traj.pt"))
print("PLASTIC_SCENE_OK")
