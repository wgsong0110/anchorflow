"""XPBD 롤아웃 (사용자 10-24): 한 스텝을 망/증분 포텐셜 대신 XPBD (lib/anchorflow/xpbd.py) 로.

데이터·격자·초기 상태는 train_deform --infer 롤아웃과 같게 읽는다:
  질량 DM.traj_mass (--pg_region_density 면 PG 영역 밀도), 부피 = 질량/밀도, 초기 x = x[0], v = 궤적 v[0] (--v0_traj),
  F = DM.traj_F(d)[0], 격자 SX.grid_for_nodes(h_fix=--node_h, hz=1/√6), 물성 cfg E·ν, 중력 cfg g,
  바닥 = cfg surface_collider, --cfg_bc = cfg 의 cuboid·enforce_particle_* 노드 하드 구속 (train_deform.cfg_bc_nodes 를 옮김),
  --pg_vdamp = cfg grid_v_damping_scale 을 PG 처럼 PG 서브스텝마다 (우리 서브스텝 Δt_s 만큼 s^(Δt_s/substep_dt)).
오차 = 프레임별 입자 평균 |x − x_gt| / 물체 대각선 (DM.frame_err, EXT = DM.extent(x[0])) -- train_deform 과 같은 정의.
덤프 = AF_ROLL_DUMP 형식 (pred, gt, x0, EXT, F (float32, 누적 F = J F), v, t0, tag) -> lb_roll_to_h5·lb_render·psi_color 그대로.
발산 (위치 비유한 또는 시작 대비 이동 > --div_ext x EXT) 이면 그 프레임까지 기록하고 멈춘다.

  python exe/xpbd_roll.py --pt L000.pt --frames 80 --node_h 0.028284 --sub 4 --iters 1 --out DIR --tag L000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402

from anchorflow import deform_model as DM                        # noqa: E402
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import xpbd as XP                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--hz_ratio", type=float, default=XP.HZ)
ap.add_argument("--sub", type=int, default=1, help="서브스텝 N")
ap.add_argument("--iters", type=int, default=1, help="반복 K")
ap.add_argument("--mu_s", type=float, default=math.inf, help="바닥 정적 마찰 계수 (inf = 닿으면 접선 고정 = sticky)")
ap.add_argument("--contact_sweeps", type=int, default=10, help="반복마다 바닥 쓸기 최대 횟수 (사면체별 최심 입자 사영을 침투가 없어질 때까지)")
ap.add_argument("--rot", default="particle", choices=["particle", "cell"], help="회전 구속: 입자별 R_p (정확) | 셀 공통 R_c (상계, 비교용)")
ap.add_argument("--m_floor", type=float, default=0.05, help="역질량에 쓰는 노드 질량 바닥 (노드 질량 중앙값 대비)")
ap.add_argument("--pg_region_density", action="store_true")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--div_ext", type=float, default=20.0)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--name", default="xpbd", help="덤프 파일 이름 roll_<name>.pt")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = a.device
t_all = time.time()
DM.REGION_DENSITY[0] = bool(a.pg_region_density)
d = torch.load(a.pt, map_location="cpu", weights_only=False)
cfg = d["cfg"]
FRAME_DT = float(cfg["frame_dt"])
T = d["x"].shape[0]
L = min(a.frames, T - 1)
m = DM.traj_mass(d, dev).double()
rho = d.get("_rho")
Vp = m / (rho.to(dev).double() if rho is not None else float(cfg["density"]))
EXT = float(d["_ext"])
x = DM.take(d["x"][0], torch.arange(d["x"].shape[1]), dev).double()
v = DM.take(d["v"][0], torch.arange(d["x"].shape[1]), dev).double()
F = DM.traj_F(d, FRAME_DT, dev)[0].to(dev).double()
x0 = x.clone()
N = x.shape[0]
mu, la = XP.lame(cfg["E"], cfg["nu"])
g = [float(q) for q in cfg["g"]]
planes = XP.planes_of(cfg)
vd = 1.0
if a.pg_vdamp:
    s_ = float(cfg.get("grid_v_damping_scale", 1.0))
    if s_ < 1.0:
        vd = s_ ** ((FRAME_DT / a.sub) / float(cfg["substep_dt"]))
solver = XP.Solver(mu, la, a.sub, a.iters, planes=planes, mu_s=a.mu_s, vdamp=vd, sweeps=a.contact_sweeps, rot=a.rot)
_bcs = [b.get("type") for b in (cfg.get("boundary_conditions") or [])]
print(f"[설정] {a.tag}: 입자 {N}, 프레임 {L} (dt {FRAME_DT:g}), XPBD 서브스텝 N={a.sub} 반복 K={a.iters}, node_h {a.node_h}, "
      f"E {cfg['E']:g} ν {cfg['nu']:g} -> μ {mu:.6g} λ {la:.6g} (fixed corotated 구속), g {g}, 질량 합 {float(m.sum()):.6g}, 부피 합 {float(Vp.sum()):.6g}, "
      f"EXT {EXT:.6f}, 경계조건 {_bcs}, 바닥 평면 {len(planes)} 개 (μ_s {a.mu_s}), 감쇠 서브스텝당 {vd:.8f}, 회전 구속 {a.rot}, 바닥 쓸기 ≤{a.contact_sweeps}, 노드 질량 바닥 중앙값 x{a.m_floor}, "
      f"|v0| 평균 {float(v.norm(dim=-1).mean()):.5f}, F0 det 범위 {float(torch.linalg.det(F).min()):.4f}~{float(torch.linalg.det(F).max()):.4f}",
      flush=True)
for bc in (cfg.get("boundary_conditions") or []):
    print(f"[경계조건] {json.dumps(bc, ensure_ascii=False)}", flush=True)

# ---- --cfg_bc: train_deform.cfg_bc_nodes 를 그대로 옮김 (입자 선택은 0 프레임 위치, 노드는 질량 가중 절반 초과) ----
_SEL = []
if a.cfg_bc:
    for bc in (cfg.get("boundary_conditions") or []):
        tp = bc.get("type")
        if tp == "enforce_particle_translation":
            q = torch.tensor(bc["point"], device=dev, dtype=torch.float64); sz = torch.tensor(bc["size"], device=dev, dtype=torch.float64)
            _SEL.append((bc, ((x0 - q).abs() < sz).all(-1)))
        elif tp == "enforce_particle_velocity_rotation":
            nr = torch.tensor(bc["normal"], device=dev, dtype=torch.float64); nr = nr / nr.norm()
            q = torch.tensor(bc["point"], device=dev, dtype=torch.float64)
            off = x0 - q; vd_ = (off @ nr).abs(); hd = (off - (off @ nr).unsqueeze(-1) * nr).norm(dim=-1)
            hh, rr = (float(z) for z in bc["half_height_and_radius"])
            _SEL.append((bc, (vd_ < hh) & (hd < rr)))
    for bc, pm in _SEL:
        print(f"[경계조건] {bc['type']} 입자 {int(pm.sum())} 개", flush=True)


def _rot_v(pos, bc):
    n = torch.tensor(bc["normal"], device=pos.device, dtype=pos.dtype); n = n / n.norm()
    q = torch.tensor(bc["point"], device=pos.device, dtype=pos.dtype)
    off = pos - q; rp = off - (off @ n).unsqueeze(-1) * n
    return float(bc["rotation_scale"]) * torch.cross(rp, n.expand_as(rp), dim=-1) + float(bc["translation_scale"]) * n


NOW = [0.0]


def bc_fn(npos, rows, lam):
    if not a.cfg_bc:
        return None, None
    M = npos.shape[0]; now = NOW[0]
    msk = torch.zeros(M, dtype=torch.bool, device=npos.device); vel = torch.zeros(M, 3, device=npos.device, dtype=npos.dtype)
    hit = False
    wl = lam * m.unsqueeze(1)
    den = None
    for bc, pm in _SEL:
        if not (float(bc.get("start_time", 0.0)) <= now < float(bc.get("end_time", 1e3))):
            continue
        if den is None:
            den = torch.zeros(M, device=npos.device, dtype=npos.dtype).index_add_(0, rows.reshape(-1), wl.reshape(-1))
        num = torch.zeros(M, device=npos.device, dtype=npos.dtype).index_add_(0, rows.reshape(-1), (wl * pm.to(wl.dtype).unsqueeze(1)).reshape(-1))
        nm = num > 0.5 * den.clamp_min(1e-30)
        vv = (torch.tensor(bc["velocity"], device=npos.device, dtype=npos.dtype).expand(M, 3)
              if bc["type"] == "enforce_particle_translation" else _rot_v(npos, bc))
        vel = torch.where(nm.unsqueeze(-1), vv, vel); msk = msk | nm; hit = True
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") != "cuboid":
            continue
        t0_, t1_ = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
        if not (t0_ <= now < t1_):
            continue
        vb = torch.tensor(bc["velocity"], device=npos.device, dtype=npos.dtype)
        q = torch.tensor(bc["point"], device=npos.device, dtype=npos.dtype) + (now - t0_) * vb
        sz = torch.tensor(bc["size"], device=npos.device, dtype=npos.dtype)
        nm = ((npos - q).abs() < sz).all(-1)
        vel = torch.where(nm.unsqueeze(-1), vb.expand(M, 3), vel); msk = msk | nm; hit = True
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid" and int(bc.get("reset", 0)) == 1:
            t1_ = float(bc.get("end_time", 1e3))
            if t1_ <= now < t1_ + 15.0 * FRAME_DT:
                msk = torch.ones(M, dtype=torch.bool, device=npos.device); vel = torch.zeros(M, 3, device=npos.device, dtype=npos.dtype); hit = True
    return (msk, vel) if hit else (None, None)


psi_fcr = lambda Fq: phys_resid.psi_of(Fq.float(), cfg, FRAME_DT)[0].double()   # noqa: E731  씬 물성(jelly FCR) 탄성 밀도
gvec = torch.tensor(g, device=dev, dtype=torch.float64)


def energies(xq, vq, Fq):
    K = 0.5 * float((m * (vq * vq).sum(-1)).sum())
    Ef = float((Vp * psi_fcr(Fq)).sum())
    Es = float((Vp * XP.psi_fcr(Fq, mu, la)).sum())             # 구속 에너지와 같은 식 (float64, 특이값 자르기 없음)
    U = -float((m * ((xq - x0) @ gvec)).sum())
    return K, Ef, Es, U


K0, Ef0, Es0, U0 = energies(x, v, F)
print(f"[에너지 0] 운동 {K0:.6e} 탄성(FCR) {Ef0:.6e} 탄성(FCR float64) {Es0:.6e}", flush=True)
P, G, FD, VD, errs, recs = [], [], [], [], [], []
div_at = None
stl = open(os.path.join(a.out, f"stats_{a.name}.jsonl"), "w")
pbar = tqdm(range(L), desc=f"XPBD {a.tag} N{a.sub} K{a.iters}", mininterval=10)
for i in pbar:
    NOW[0] = i * FRAME_DT
    t_ = time.time()
    with torch.no_grad():
        x2, v2, F2, Jp, st = XP.advance_frame(solver, x, v, F, m, Vp, a.node_h, g, FRAME_DT, bc_fn=bc_fn, hz_ratio=a.hz_ratio, m_floor=a.m_floor)
    if dev == "cuda":
        torch.cuda.synchronize()
    gt = DM.take(d["x"][i + 1], torch.arange(N), dev).double()
    fin = bool(torch.isfinite(x2).all() and torch.isfinite(F2).all() and torch.isfinite(v2).all())
    mv = float((x2 - x0).norm(dim=-1).max()) if fin else math.inf
    if not fin or mv > a.div_ext * EXT:
        div_at = i + 1
        rec = dict(frame=i + 1, diverged=True, finite=fin, max_move_over_ext=mv / EXT, **st)
        stl.write(json.dumps(rec) + "\n"); stl.flush()
        print(f"[발산] 프레임 {i + 1}: 유한 {fin}, 최대 이동/EXT {mv / EXT:.3g} -- 여기서 멈춘다 (기록은 프레임 {i} 까지)", flush=True)
        break
    e = DM.frame_err(x2, gt, EXT)
    detF = torch.linalg.det(F2)
    K, Ef, Es, U = energies(x2, v2, F2)
    pen = None
    if planes:
        p0, n, _ = planes[0]
        pen = float(((x2 - p0.to(dev)) @ n.to(dev)).min())
    rec = dict(frame=i + 1, err=e, detF_min=float(detF.min()), inv_particles=int((detF <= 0).sum()),
               K=K, E_fcr=Ef, E_fcr64=Es, U_grav=U, floor_min_s=pen, wall_s=time.time() - t_, **st)
    stl.write(json.dumps(rec) + "\n"); stl.flush()
    recs.append(rec); errs.append(e)
    P.append(x2.float().cpu()); G.append(gt.float().cpu()); FD.append(F2.float().cpu()); VD.append(v2.float().cpu())
    pbar.set_postfix_str(f"오차 {100 * e:.2f}% detJ최소 {st['detJ_min']:.3f} detF최소 {float(detF.min()):.3f}")
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [프레임 {i + 1}] 오차 {100 * e:.3f}% 평균(지금까지) {100 * np.mean(errs):.3f}% | 사면체 {st['tets']} 노드 {st['nodes']} 색 {st['colors']} "
              f"고정 노드 {st['pinned']} | det J 최소 {st['detJ_min']:.4f} 뒤집힌 사면체 {st['tets_inv']} | det F 최소 {float(detF.min()):.4f} "
              f"뒤집힌 입자 {int((detF <= 0).sum())} | 운동 {K:.4e} 탄성 FCR {Ef:.4e} (float64 {Es:.4e}) 중력위치 {U:.4e} 합(FCR) {K + Ef + U:.4e}"
              + (f" | 바닥 최소 거리 {pen:.3e}" if pen is not None else "") + f" | {time.time() - t_:.2f} s", flush=True)
    x, v, F = x2, v2, F2
stl.close()
nf = len(P)
if nf:
    torch.save({"pred": torch.stack(P), "gt": torch.stack(G), "x0": x0.float().cpu(), "ctrl_pos": None, "ctrl_R": None, "ctrl_id": None,
                "t0": 0, "tag": a.tag, "nodes": None, "fscal": None, "gt_fscal": None, "EXT": EXT,
                "F": torch.stack(FD), "v": torch.stack(VD),
                "xpbd": dict(N=a.sub, K=a.iters, node_h=a.node_h, mu=mu, lam=la, model="fcr", rot=a.rot, mu_s=a.mu_s, vdamp_sub=vd,
                             diverged_at=div_at, frames=nf)},
               os.path.join(a.out, f"roll_{a.name}.pt"))
summ = dict(tag=a.tag, N=a.sub, K=a.iters, frames_req=L, frames_done=nf, diverged_at=div_at,
            err=errs, err_mean=float(np.mean(errs)) if errs else None, E0=dict(K=K0, E_fcr=Ef0, E_fcr64=Es0),
            detF_min=min((r["detF_min"] for r in recs), default=None), detJ_min=min((r["detJ_min"] for r in recs), default=None),
            inv_particles_max=max((r["inv_particles"] for r in recs), default=None), tets_inv_max=max((r["tets_inv"] for r in recs), default=None),
            E_fcr_mean=float(np.mean([r["E_fcr"] for r in recs])) if recs else None, wall_s=time.time() - t_all,
            mu=mu, lam=la, model="fcr", rot=a.rot, mu_s=a.mu_s, vdamp_sub=vd, node_h=a.node_h, pt=a.pt)
json.dump(summ, open(os.path.join(a.out, f"summary_{a.name}.json"), "w"), ensure_ascii=False, indent=1)
print(f"[롤아웃] {a.tag} t0=  0: {nf} 프레임, 평균 {100 * (summ['err_mean'] or float('nan')):.3f}%"
      + (f" (발산 프레임 {div_at})" if div_at else "") + f" -- XPBD N={a.sub} K={a.iters}", flush=True)
print("XPBD_ROLL_DONE", flush=True)
