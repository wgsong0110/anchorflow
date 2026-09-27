"""교사 상태에서 각 항이 얼마나 미는지 **기울기 크기**로 잰다.

에너지 값이 아니라 기울기가 해를 정한다. 관성·탄성·중력·접촉을 따로 미분해
크기와 방향(중력과의 정렬)을 본다. 손잡이 구속은 학습과 같게 제외한다.
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=20000)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
NG, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
gv = torch.tensor(cfg["g"], device=dev)
X = d["x"].float()
F_all = d.get("F")
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
X = X[:, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
I3 = torch.eye(3, device=dev)
gdir = gv / gv.norm()


def free_of(t0):
    cid, cp, cr = d.get("ctrl_id"), d.get("ctrl_pos"), d.get("ctrl_R")
    if cid is None:
        return None
    R = float(cr[min(t0, cr.numel() - 1)])
    c = cp[min(t0, cp.shape[0] - 1)].float().to(dev)
    dd = (X[t0].unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1)
    return (dd.min(1).values > R)


print(f"[궤적] {os.path.basename(a.traj)}")
for t0 in a.t0:
    x = X[t0]
    du_t = X[t0 + 1] - x
    v = (X[t0] - X[t0 - 1]) / h
    F = (F_all[t0, sel].float().to(dev) if F_all is not None
         else I3.expand(sel.numel(), 3, 3).contiguous())
    fm = free_of(t0)
    nf = int(fm.sum()) if fm is not None else sel.numel()

    def term_grads(du):
        du = du.detach().clone().requires_grad_(True)
        m_I, du_I, v_I, info, frac = phys_resid.p2g_increment(
            x, du, v, mass, NG, gl, free=fm)
        wf = torch.ones_like(m_I) if frac is None else (frac > 0.5).to(m_I.dtype)
        dd = du_I - h * v_I
        e_in = (0.5 * wf * m_I / (h * h) * (dd * dd).sum(-1)).sum()
        e_g = -(wf * m_I * (du_I * gv).sum(-1)).sum()
        gu = phys_resid.g2p_grad(x, du_I, info, NG)
        psi, _ = phys_resid.psi_of((I3 + gu) @ F, cfg, h)
        e_el = (vol * psi).sum()
        e_bc = phys_resid.bc_energy(x, du, mass, cfg, h, gl, NG)
        out = {}
        for nm, e in (("관성", e_in), ("탄성", e_el), ("중력", e_g), ("접촉", e_bc)):
            if float(e) == 0.0 and nm == "접촉":
                out[nm] = (0.0, 0.0)
                continue
            gx, = torch.autograd.grad(e, du, retain_graph=True)
            n = float(gx.norm(dim=-1).mean())
            al = float((gx / gx.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                        @ gdir).mean())
            out[nm] = (n, al)
        return out

    print(f"[t0={t0}] 자유입자 {nf}/{sel.numel()}  (기울기는 입자당 평균 크기, "
          f"정렬은 중력방향 내적: +1 이면 힘이 위로 = 낙하를 막는다)")
    for tag, du in (("교사", du_t), ("정지", torch.zeros_like(x))):
        o = term_grads(du)
        s = "  ".join(f"{k} {o[k][0]:.3e}(정렬 {o[k][1]:+.2f})" for k in o)
        print(f"   {tag}: {s}")
