"""i-PG 목적함수만으로 **출력 공간**을 최적화했을 때 교사와 얼마나 벌어지는가.

`fit_floor.py` 는 정답 위치를 보고 격자점 변위를 맞춘다 (위치 손실의 하한).
여기서는 정답을 **보지 않고** 같은 변수를 i-PG 격자 증분 포텐셜로만 최적화한 뒤,
나온 위치를 교사와 견준다. 두 값의 차이가 곧 "목적함수를 물리잔차로 바꿨을 때
잃는 정보" 다 -- 학습이 완벽해도 이 아래로는 못 내려간다.

네트워크도 일반화도 끼지 않는다. 창마다 따로 푼다.
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from anchorflow import phys_resid                              # noqa: E402
from anchorflow import trilinear as TRI                        # noqa: E402
from anchorflow import vox_anchor                              # noqa: E402
from anchorflow.deform import skin                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--files", required=True, help="쉼표로 구분한 궤적 파일명")
ap.add_argument("--t0", type=int, nargs="+", required=True)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ctrl_R_scale", type=float, default=1.0)
ap.add_argument("--var", default="grid", choices=("grid", "pts"),
                help="최적화 변수. grid 는 격자점 변위(학생의 출력 공간), "
                     "pts 는 가우시안 위치 자체 -- 둘의 차이가 격자로 제한해서 "
                     "잃는 몫이다")
ap.add_argument("--obj_log", default="", help="창별 목적함수 곡선을 npy 로 남긴다")
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()
dev = a.dev
FRAME_DT = 1.0 / 60.0

tot_p = tot_s = tot_r = 0.0
n = 0
curves = []
for fn in a.files.split(","):
    f = os.path.join(a.data, fn if fn.endswith(".pt") else fn + ".pt")
    d = torch.load(f, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    X = d["x"].float()
    N = X.shape[1]
    gsel = torch.arange(0, N, max(1, N // a.n_pts))[:a.n_pts]
    x0 = X[0][gsel].to(dev)
    ext = float((X[0].max(0).values - X[0].min(0).values).norm())
    # 입자 질량 (궤적 자기 배치·밀도)
    ng_ = int(cfg.get("n_grid", 100))
    dx_ = float(cfg.get("grid_lim", 2.0)) / ng_
    xa = X[0].to(dev)
    vi_ = (xa / dx_).long().clamp(0, ng_ - 1)
    fl_ = (vi_[:, 0] * ng_ + vi_[:, 1]) * ng_ + vi_[:, 2]
    cn_ = torch.zeros(ng_ ** 3, device=dev).index_add_(
        0, fl_, torch.ones(xa.shape[0], device=dev))
    mass_full = ((dx_ ** 3) / cn_[fl_]) * float(cfg["density"])
    mass = mass_full[gsel.to(dev)] * (float(N) / gsel.numel())
    vol = mass / float(cfg["density"])
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    norm = float(mass.sum()) * (ext ** 2) / (FRAME_DT * FRAME_DT)

    for t0 in a.t0:
        if t0 + 1 >= X.shape[0]:
            continue
        x = X[t0][gsel].to(dev)
        v = (x - X[max(t0 - 1, 0)][gsel].to(dev)) / FRAME_DT
        F = d["F"][t0][gsel].float().to(dev)
        gt = X[t0 + 1][gsel].to(dev)
        # 손잡이 입자: 학습과 같은 규약으로 명령 변위를 박고 손실에서 뺀다
        R = float(d["ctrl_R"][min(t0, d["ctrl_R"].numel() - 1)]) * a.ctrl_R_scale
        cid = d["ctrl_id"][min(t0, d["ctrl_id"].shape[0] - 1)]
        sel = d["sel"]
        inv = torch.full((int(d["n_full"]),), -1, dtype=torch.long)
        inv[sel] = torch.arange(sel.numel())
        l20 = inv[cid.clamp(0, inv.numel() - 1)]
        invg = torch.full((N,), -1, dtype=torch.long)
        invg[gsel] = torch.arange(gsel.numel())
        loc = invg[l20.clamp(0, N - 1)]
        bad = loc < 0
        if bool(bad.any()):
            loc[bad] = torch.cdist(X[0][l20[bad].clamp(0, N - 1)],
                                   X[0][gsel]).argmin(1)
        c = x[loc.to(dev)]
        q = ((x.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / max(R, 1e-6)
             ).clamp(0, 1)
        w = (1.0 - q * q) ** 2                              # [N,K]
        vc = d["ctrl_vel"][min(t0, d["ctrl_vel"].shape[0] - 1)].to(dev)
        wm, ki = w.max(1)
        d_cmd = FRAME_DT * vc[ki]
        free = ~(wm > 0.5)

        if a.var == "grid":
            lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
            sidx, _w8 = TRI.corners(x, lo, hh, nn3)
            gpos = (torch.stack(torch.meshgrid(
                *[torch.arange(int(nn3[i]), device=dev, dtype=x.dtype)
                  for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
                ) * float(hh) + lo
            log_r = torch.full((gpos.shape[0],), math.log(float(hh)),
                               device=dev)
            log_t = torch.zeros_like(log_r)
            var = torch.zeros(gpos.shape[0], 3, device=dev, requires_grad=True)

            def warp(_v):
                return skin(x, gpos, _v, log_r, log_t, sidx, float(hh))[0]
        else:
            # 가우시안 위치 자체가 변수다 (격자 제한 없음)
            var = torch.zeros(x.shape[0], 3, device=dev, requires_grad=True)

            def warp(_v):
                return x + _v
        dp = var
        opt = torch.optim.Adam([var], lr=a.lr)
        cur = []
        for it in range(a.steps):
            opt.zero_grad(set_to_none=True)
            xe = warp(var)
            x2 = x + (1.0 - wm.unsqueeze(-1)) * (xe - x) + \
                wm.unsqueeze(-1) * d_cmd
            E, _dl, _F, _pt = phys_resid.grid_ip_energy(
                x, x2 - x, v, F, mass, vol, cfg, FRAME_DT,
                ng_, float(cfg.get("grid_lim", 2.0)), g=g, norm=norm, free=free)
            E.backward()
            opt.step()
            cur.append(float(E))
        with torch.no_grad():
            xe = warp(var)
            x2 = x + (1.0 - wm.unsqueeze(-1)) * (xe - x) + \
                wm.unsqueeze(-1) * d_cmd
            pe = float((x2[free] - gt[free]).norm(dim=-1).mean()) / ext
            st = float((x[free] - gt[free]).norm(dim=-1).mean()) / ext
        curves.append(cur)
        tot_p += pe
        tot_s += st
        tot_r += pe / max(st, 1e-20)
        n += 1
        print(f"  {fn} t0={t0:3d}  물리최적 {100*pe:.4f}%  정지 {100*st:.4f}%  "
              f"비 {pe/max(st,1e-20):.3f}", flush=True)
if a.obj_log:
    import numpy as _np
    _np.save(a.obj_log, _np.asarray(curves, dtype=_np.float64))
    print(f"[목적함수 곡선] {a.obj_log}  {len(curves)} 창 x {a.steps} 반복",
          flush=True)
print(f"[물리 하한] {n} 창 평균  물리최적 {100*tot_p/n:.4f}%  "
      f"정지 {100*tot_s/n:.4f}%  비 {tot_r/n:.3f}", flush=True)
