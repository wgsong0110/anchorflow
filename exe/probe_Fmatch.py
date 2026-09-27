"""목적함수가 보는 변형이 PG 의 F 와 맞는지 대조한다.

PG 는 서브스텝마다 격자 ∇v 로 F 를 갱신한다. 우리 목적함수는 프레임 변위를
MPM 격자에 올려 ∇Δu 를 뽑고 F <- (I+∇Δu) F 로 민다. 두 경로가 같은 양을 보는지
확인하려면 궤적의 F^{n+1} 과 우리가 민 F 를 직접 비교해야 한다.

  python exe/probe_Fmatch.py --traj traj_micF/mic_clayC_t_s200000.pt --t0 3 10 20
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--n_grid", type=int, default=0, help="0 이면 cfg 값")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
NG = a.n_grid or int(cfg["n_grid"])
gl = float(cfg.get("grid_lim", 2.0))
X = d["x"].float()
F_all = d.get("F")
if F_all is None:
    raise SystemExit("궤적에 F 가 없다")
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
X = X[:, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
I3 = torch.eye(3, device=dev)
print(f"[궤적] {os.path.basename(a.traj)}  격자 {NG}^3  입자 {sel.numel()}")

for t0 in a.t0:
    x0 = X[t0]
    du = X[t0 + 1] - x0
    F0 = F_all[t0, sel].float().to(dev)
    F1 = F_all[t0 + 1, sel].float().to(dev)
    # 실제 PG 가 만든 증분: F1 = dF F0  ->  dF = F1 F0^{-1}
    dF = F1 @ torch.linalg.inv(F0)
    tn = (dF - I3).reshape(-1, 9).norm(dim=-1)
    print(f"[t0={t0}] PG 증분 |dF-I| 중앙 {float(tn.median()):.5f} "
          f"상위10% {float(tn.quantile(0.9)):.5f}")
    for nm in ("평균+도함수", "최소제곱"):
        if nm == "평균+도함수":
            _m, du_I, _v, info, _ = phys_resid.p2g_increment(
                x0, du, torch.zeros_like(x0), mass, NG, gl)
            gu = phys_resid.g2p_grad(x0, du_I, info, NG)
        else:
            _m, _u, G_I, info, _xI = phys_resid.p2g_ls(x0, du, mass, NG, gl)
            gu = phys_resid.g2p_from_nodes(x0, G_I, info, NG)
        rel = ((gu - (dF - I3)).reshape(-1, 9).norm(dim=-1)
               / tn.clamp_min(1e-12))
        gn = gu.reshape(-1, 9).norm(dim=-1)
        u, vv = gu.reshape(-1, 9), (dF - I3).reshape(-1, 9)
        cc = float(((u - u.mean(0)) * (vv - vv.mean(0))).mean()
                   / (u.std(0).clamp_min(1e-20) * vv.std(0).clamp_min(1e-20)).mean())
        print(f"   [{nm:10s}] |∇Δu| 중앙 {float(gn.median()):.5f} "
              f"크기비 {float(gn.median() / tn.median().clamp_min(1e-12)):.3f}  "
              f"오차 {float(rel.median()):.4f}  상관 {cc:+.4f}")
