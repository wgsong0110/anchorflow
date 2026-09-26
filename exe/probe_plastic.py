"""탄성항이 교사와 어긋나는 원인을 소성 사영 횟수로 가린다.

PG 는 프레임당 서브스텝(2e-5 기준 833 회)마다 항복면으로 사영한다. 우리 격자
목적함수는 프레임 변위를 한 번에 받아 **한 번만** 사영하므로, 소성 재료에서는
한 프레임 분량의 변형이 탄성으로 다 쌓인 뒤 잘린다 -> 탄성 에너지가 과대평가되고
목적함수가 "덜 변형하는 답" 을 교사보다 좋아하게 된다.

같은 프레임 변위를 K 등분해 매 등분마다 사영하며 탄성 에너지를 다시 재고,
K 를 키우면 교사의 탄성 에너지가 얼마나 내려가는지 본다.

  python exe/probe_plastic.py --traj traj_h2/mic_clayC_t_s400706.pt --t0 3 10 20
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 16, 64])
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
ng, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
X = d["x"].float()
F_all = d.get("F")
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
X = X[:, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
print(f"[궤적] {os.path.basename(a.traj)}  재질 {phys_resid.mat_name(cfg)} "
      f"항복 {cfg.get('yield_stress')}  서브스텝 {cfg.get('substep_dt')}")

for t0 in a.t0:
    x = X[t0]
    du = X[t0 + 1] - x
    F0 = (F_all[t0, sel].float().to(dev) if F_all is not None
          else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
    # 격자를 거친 변위기울기 (목적함수가 쓰는 것과 같은 경로)
    m_I, du_I, v_I, info, _ = phys_resid.p2g_increment(
        x, du, torch.zeros_like(x), mass, ng, gl)
    gu = phys_resid.g2p_grad(x, du_I, info, ng)
    I3 = torch.eye(3, device=dev)
    print(f"[t0={t0}] |∇Δu| 중앙 {float(gu.reshape(-1, 9).norm(dim=-1).median()):.4f}")
    for K in a.K:
        F = F0.clone()
        e_el = 0.0
        for _ in range(K):
            F_tr = (I3 + gu / K) @ F
            psi, dlog = phys_resid.psi_of(F_tr, cfg, h)
            e_el = float((vol * psi).sum())      # 마지막 등분의 값을 쓴다
            F = phys_resid.plastic_step(F_tr, dlog)
        det = torch.linalg.det(F)
        print(f"   K={K:3d}  탄성 {e_el:.4e}  det F 중앙 {float(det.median()):.4f} "
              f"음수 {float((det < 0).float().mean()) * 100:.3f}%")
