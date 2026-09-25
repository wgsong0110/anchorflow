"""매 스텝 정답을 보고 격자점 변위를 맞추며 굴리는 **표현 한계 롤아웃**.

학생도 학습도 없다. 프레임마다 현재 상태에서 격자를 잡고, 교사의 다음 프레임에
맞도록 격자점 변위를 최소제곱으로 푼 뒤, 그 결과를 다음 스텝의 상태로 이어 쓴다.
손잡이 입자는 학습과 같은 규약으로 명령 변위를 박는다.

그래서 여기 남는 오차는 **학습 오차가 아니라 표현의 한계가 누적된 것**이다.
학습된 모델이 이 곡선에서 얼마나 떨어져 있는지가 곧 "못 배운 몫" 이다.
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from anchorflow import phys_resid                             # noqa: E402
from anchorflow import trilinear as TRI                        # noqa: E402
from anchorflow import vox_anchor                              # noqa: E402
from anchorflow.deform import skin                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True, help="덤프 .pt (렌더용)")
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--transfer", default="skin", choices=("skin", "tri"))
ap.add_argument("--clamp", type=float, default=0.5,
                help="격자점 변위를 셀 크기의 이 배수로 자른다. 학습 경로가 "
                     "망 출력에 거는 것과 같은 제한 (기본 0.5). 0 이면 안 자른다")
ap.add_argument("--obj", default="pos", choices=("pos", "phys"),
                help="매 스텝 무엇을 맞출지. pos 는 교사의 다음 프레임(표현 한계), "
                     "phys 는 i-PG 격자 증분 포텐셜(물리손실의 한계)")
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()
dev = a.dev
FRAME_DT = 1.0 / 60.0

d = torch.load(a.traj, map_location="cpu", weights_only=False)
X = d["x"].float()
N = X.shape[1]
gsel = torch.arange(0, N, max(1, N // a.n_pts))[:a.n_pts]
ext = float((X[0].max(0).values - X[0].min(0).values).norm())
x = X[a.t0][gsel].to(dev)
still0 = x.clone()
cfg = d["cfg"]
NG = int(cfg.get("n_grid", 100))
GL = float(cfg.get("grid_lim", 2.0))
if a.obj == "phys":
    # 물리손실을 쓰려면 질량·부피·중력·F 가 있어야 한다 (학습과 같은 규약)
    _dx = GL / NG
    _xa = X[0].to(dev)
    _vi = (_xa / _dx).long().clamp(0, NG - 1)
    _fl = (_vi[:, 0] * NG + _vi[:, 1]) * NG + _vi[:, 2]
    _cn = torch.zeros(NG ** 3, device=dev).index_add_(
        0, _fl, torch.ones(_xa.shape[0], device=dev))
    _mf = ((_dx ** 3) / _cn[_fl]) * float(cfg["density"])
    MASS = _mf[gsel.to(dev)] * (float(N) / gsel.numel())
    VOL = MASS / float(cfg["density"])
    GVEC = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    NORM = float(MASS.sum()) * (ext ** 2) / (FRAME_DT * FRAME_DT)
    Fst = d["F"][a.t0][gsel].float().to(dev)
    v = (x - X[max(a.t0 - 1, 0)][gsel].to(dev)) / FRAME_DT

# 제어 입자 색인 (전체 -> 부분표본)
sel, cid_all = d["sel"], d["ctrl_id"]
inv = torch.full((int(d["n_full"]),), -1, dtype=torch.long)
inv[sel] = torch.arange(sel.numel())
invg = torch.full((N,), -1, dtype=torch.long)
invg[gsel] = torch.arange(gsel.numel())

preds, gts = [], []
err_s, still_s = [], []
for i in range(a.frames):
    t = a.t0 + i
    if t + 1 >= X.shape[0]:
        break
    gt = X[t + 1][gsel].to(dev)
    # 손잡이: 학습과 같은 규약 (명령 속도 + 감쇠 가중치, 학생 상태 기준)
    R = float(d["ctrl_R"][min(t, d["ctrl_R"].numel() - 1)])
    l20 = inv[cid_all[min(t, cid_all.shape[0] - 1)].clamp(0, inv.numel() - 1)]
    loc = invg[l20.clamp(0, N - 1)]
    bad = loc < 0
    if bool(bad.any()):
        loc[bad] = torch.cdist(X[0][l20[bad].clamp(0, N - 1)],
                               X[0][gsel]).argmin(1)
    c = x[loc.to(dev)]
    q = ((x.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / max(R, 1e-6)).clamp(0, 1)
    w = (1.0 - q * q) ** 2
    vc = d["ctrl_vel"][min(t, d["ctrl_vel"].shape[0] - 1)].to(dev)
    wm, ki = w.max(1)
    d_cmd = FRAME_DT * vc[ki]
    free = ~(wm > 0.5)

    lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
    sidx, w8 = TRI.corners(x, lo, hh, nn3)
    gpos = (torch.stack(torch.meshgrid(
        *[torch.arange(int(nn3[k]), device=dev, dtype=x.dtype)
          for k in range(3)], indexing="ij"), -1).reshape(-1, 3)) * float(hh) + lo
    log_r = torch.full((gpos.shape[0],), math.log(float(hh)), device=dev)
    log_t = torch.zeros_like(log_r)
    dp = torch.zeros(gpos.shape[0], 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp], lr=a.lr)
    _lim = a.clamp * float(hh)
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        dpc = dp.clamp(-_lim, _lim) if a.clamp > 0 else dp
        xe = (skin(x, gpos, dpc, log_r, log_t, sidx, float(hh))[0]
              if a.transfer == "skin" else x + TRI.g2p(sidx, w8, dpc))
        x2 = x + (1.0 - wm.unsqueeze(-1)) * (xe - x) + wm.unsqueeze(-1) * d_cmd
        if a.obj == "phys":
            E, _dl, _Ft, _pt = phys_resid.grid_ip_energy(
                x, x2 - x, v, Fst, MASS, VOL, cfg, FRAME_DT, NG, GL,
                g=GVEC, norm=NORM, free=free)
            E.backward()
        else:
            ((x2[free] - gt[free]) ** 2).sum(-1).mean().backward()
        opt.step()
    with torch.no_grad():
        dpc = dp.clamp(-_lim, _lim) if a.clamp > 0 else dp
        xe = (skin(x, gpos, dpc, log_r, log_t, sidx, float(hh))[0]
              if a.transfer == "skin" else x + TRI.g2p(sidx, w8, dpc))
        x2 = x + (1.0 - wm.unsqueeze(-1)) * (xe - x) + wm.unsqueeze(-1) * d_cmd
        e = float((x2[free] - gt[free]).norm(dim=-1).mean()) / ext
        st = float((still0[free] - gt[free]).norm(dim=-1).mean()) / ext
    if a.obj == "phys":
        # 상태를 교사와 같은 절차로 이어 나른다: F 를 밀고 속도를 갱신한다
        with torch.no_grad():
            _m, _duI, _vI, _info, _fr = phys_resid.p2g_increment(
                x, x2 - x, v, MASS, NG, GL)
            _gu = phys_resid.g2p_grad(x, _duI, _info, NG)
            _Ftr = (torch.eye(3, device=dev) + _gu) @ Fst
            Fst = phys_resid.plastic_step(
                _Ftr, phys_resid.psi_of(_Ftr, cfg, FRAME_DT)[1])
            v = (x2 - x) / FRAME_DT
    err_s.append(e)
    still_s.append(st)
    preds.append(x2.cpu())
    gts.append(gt.cpu())
    print(f"  프레임 {i+1:2d}  적합 {100*e:.4f}%  정지 {100*st:.4f}%  "
          f"비 {e/max(st,1e-20):.3f}", flush=True)
    x = x2

import numpy as np                                             # noqa: E402
torch.save({"pred": torch.stack(preds), "gt": torch.stack(gts),
            "x0": X[a.t0][gsel].cpu(), "ctrl_pos": d.get("ctrl_pos"),
            "t0": a.t0, "tag": d.get("tag", "fit"), "EXT": ext}, a.out)
print(f"[표현 한계 롤아웃] {len(err_s)} 프레임 평균 {100*np.mean(err_s):.3f}% "
      f"(정지 {100*np.mean(still_s):.3f}%, 비 "
      f"{np.mean(err_s)/max(np.mean(still_s),1e-20):.3f})  -> {a.out}", flush=True)
