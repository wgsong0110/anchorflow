"""벤치 지표. PG 덤프를 기준으로 한 솔버의 궤적을 재고 CSV 한 줄을 낸다.

재는 것: CD, EMD, 물리잔차(바닥 접촉항 포함), 부피비, 운동량·에너지 변동,
영역 관통량, det F < 0 비율. FPS 와 실행시간은 실행 쪽 로그에서 받아 붙인다.

  python exe/bench_metrics.py --ref bench_pg/mic_clayC_s00.pt \
      --tgt bench_ipg/mic_clayC_s00.pt --cfg bench_cfg/mic_clayC.json \
      --tag ipg --combo mic_clayC --seed 0 --out bench/metrics_ipg.csv
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="PG 덤프 (.pt)")
ap.add_argument("--tgt", required=True, help="대상 덤프 (.pt). ref 와 같으면 PG 자신")
ap.add_argument("--cfg", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--combo", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--cd_pts", type=int, default=2048)
ap.add_argument("--n_pts", type=int, default=8000, help="잔차 계산 부분표본")
ap.add_argument("--fps", type=float, default=float("nan"))
ap.add_argument("--wall", type=float, default=float("nan"))
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(a.cfg))
h = float(cfg["frame_dt"])
gl = float(cfg.get("grid_lim", 2.0))
n_grid = int(cfg["n_grid"])
gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    x = d["x"].float()
    F = d["F"].float() if d.get("F") is not None else None
    return x, F


XR, _ = load(a.ref)
XT, FT = load(a.tgt)
T = min(XR.shape[0], XT.shape[0])
N = min(XR.shape[1], XT.shape[1])
XR, XT = XR[:T, :N].to(dev), XT[:T, :N].to(dev)
if FT is not None:
    FT = FT[:T, :N].to(dev)

rho = float(cfg["density"])
# 질량은 균일 채우기라 입자당 같다 (절대 스케일은 비에만 들어간다)
mass = torch.full((N,), rho, device=dev)
vol = mass / rho
ext = float((XR[0].max(0).values - XR[0].min(0).values).norm())

g0 = torch.Generator().manual_seed(0)
cd_i = torch.randperm(N, generator=g0)[:a.cd_pts].to(dev)      # 같은 색인으로
res_i = torch.randperm(N, generator=g0)[:a.n_pts].to(dev)


def chamfer(p, q):
    d = torch.cdist(p, q)
    return float(d.min(1).values.mean() + d.min(0).values.mean()) / 2.0


def emd(p, q):
    """헝가리안 대신 Sinkhorn 근사 (같은 표본 수, 정규화된 수송비)."""
    d = torch.cdist(p, q)
    K = torch.exp(-d / (0.01 * ext))
    u = torch.ones(p.shape[0], device=p.device) / p.shape[0]
    v = torch.ones(q.shape[0], device=q.device) / q.shape[0]
    for _ in range(60):
        u = 1.0 / p.shape[0] / (K @ v).clamp_min(1e-30)
        v = 1.0 / q.shape[0] / (K.t() @ u).clamp_min(1e-30)
    P = u.unsqueeze(-1) * K * v.unsqueeze(0)
    return float((P * d).sum() / P.sum())


cds, emds, res, pen, dneg, dets, mom, ene = [], [], [], [], [], [], [], []
for t in range(1, T):
    pr, pt = XR[t][cd_i], XT[t][cd_i]
    cds.append(chamfer(pr, pt))
    emds.append(emd(pr, pt) / ext)
    x = XT[t - 1][res_i]
    du = (XT[t] - XT[t - 1])[res_i]
    v = (XT[t - 1] - XT[max(t - 2, 0)])[res_i] / h
    F = (FT[t - 1][res_i] if FT is not None
         else torch.eye(3, device=dev).expand(res_i.numel(), 3, 3).contiguous())
    m = mass[res_i]
    du = du.clone().requires_grad_(True)
    nrm = float(m.sum()) * ext ** 2 / h ** 2
    E, _, _, _ = phys_resid.grid_ip_energy(
        x, du, v, F, m, m / rho, cfg, h, n_grid, gl, g=gv, norm=nrm)
    gx, = torch.autograd.grad(E * nrm, du)
    res.append(float((gx * h ** 2 / m.unsqueeze(-1) / ext).norm(dim=-1).mean()))
    # 영역 관통: 바닥면 아래 + 상자 밖 깊이의 질량가중 합 (물체 크기로 정규화)
    with torch.no_grad():
        xt = XT[t]
        dep = torch.zeros(xt.shape[0], device=dev)
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "surface_collider":
                p0 = torch.as_tensor(bc["point"], device=dev, dtype=xt.dtype)
                nr = torch.as_tensor(bc["normal"], device=dev, dtype=xt.dtype)
                nr = nr / nr.norm().clamp_min(1e-12)
                dep = dep + (-((xt - p0) * nr).sum(-1)).clamp_min(0.0)
            elif bc.get("type") == "bounding_box":
                b = float(cfg.get("bound", 3)) * gl / n_grid
                dep = dep + ((b - xt).clamp_min(0.0)
                             + (xt - (gl - b)).clamp_min(0.0)).sum(-1)
        pen.append(float(dep.mean()) / ext)
        if FT is not None:
            dj = torch.linalg.det(FT[t])
            dneg.append(float((dj < 0).float().mean()))
            dets.append(float(dj.median()))
        vv = (XT[t] - XT[t - 1]) / h
        mom.append((mass.unsqueeze(-1) * vv).sum(0).tolist())
        ke = float(0.5 * (mass * (vv * vv).sum(-1)).sum())
        pe = -float((mass * (XT[t] * gv).sum(-1)).sum())
        if FT is not None:
            psi, _ = phys_resid.psi_of(FT[t], cfg, h)
            el = float((vol * psi).sum())
        else:
            el = 0.0
        ene.append(ke + pe + el)

mom = np.asarray(mom)
ene = np.asarray(ene)
dets = np.asarray(dets) if dets else np.asarray([float("nan")])
row = dict(
    solver=a.tag, combo=a.combo, seed=a.seed, frames=T,
    CD=float(np.mean(cds)), EMD=float(np.mean(emds)),
    residual=float(np.mean(res)),
    vol_ratio=float(np.nanmax(dets) / max(np.nanmin(dets), 1e-12)),
    det_neg_mean=float(np.mean(dneg)) if dneg else float("nan"),
    det_neg_max=float(np.max(dneg)) if dneg else float("nan"),
    mom_span=float(np.abs(mom.max(0) - mom.min(0)).max()),
    ene_span=float(ene.max() - ene.min()),
    ene_ratio=float(ene.max() / ene.min()) if ene.min() > 0 else float("nan"),
    penetration=float(np.mean(pen)), penetration_max=float(np.max(pen)),
    fps=a.fps, wall_s=a.wall)
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(json.dumps(row, ensure_ascii=False))
