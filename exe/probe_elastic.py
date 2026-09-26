"""탄성항이 무엇 때문에 큰지 가린다.

가설: 교사 상태에 이미 쌓인 F 자체가 탄성 에너지를 크게 만들고, 목적함수는
그 변형을 **풀어 버리는** 답을 교사보다 좋아한다. 확인 방법은 세 가지 비교다.

  (가) Psi(F^n)            지금 상태를 유지만 해도 드는 탄성 에너지
  (나) Psi((I+grad du)F^n) 교사의 다음 프레임
  (다) F^n 을 항등으로 바꿔 같은 계산 -> F 가 원인인지 분리

또 PG 의 F 와 우리가 궤적에서 읽은 F 가 같은지 (det, 특이값) 같이 낸다.
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=8000)
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
mu, lam = phys_resid.lame(cfg["E"], cfg["nu"])
print(f"[궤적] {os.path.basename(a.traj)}  F {'있음' if F_all is not None else '없음'}"
      f"  mu {mu:.4g} lam {lam:.4g}")

I3 = torch.eye(3, device=dev)
for t0 in a.t0:
    x = X[t0]
    du = X[t0 + 1] - x
    F0 = (F_all[t0, sel].float().to(dev) if F_all is not None
          else I3.expand(sel.numel(), 3, 3).contiguous())
    sig = phys_resid._sig(F0)
    det = torch.linalg.det(F0)
    m_I, du_I, _v, info, _ = phys_resid.p2g_increment(
        x, du, torch.zeros_like(x), mass, ng, gl)
    gu = phys_resid.g2p_grad(x, du_I, info, ng)

    def el(F):
        psi, _ = phys_resid.psi_of(F, cfg, h)
        return float((vol * psi).sum()), float(psi.median())

    e_keep, p_keep = el(F0)
    e_next, p_next = el((I3 + gu) @ F0)
    e_id, p_id = el((I3 + gu))
    e_zero, _ = el(I3.expand(sel.numel(), 3, 3).contiguous())
    print(f"[t0={t0}] F 특이값 중앙 {[round(float(q),4) for q in sig.median(0).values]}"
          f"  det 중앙 {float(det.median()):.4f} 최소 {float(det.min()):.4f}")
    print(f"   유지 Psi(F^n)          {e_keep:.4e}  (중앙 {p_keep:.4e})")
    print(f"   교사 Psi((I+G)F^n)     {e_next:.4e}  (중앙 {p_next:.4e})")
    print(f"   F=I 로 바꾸면 Psi(I+G) {e_id:.4e}  (중앙 {p_id:.4e})")
    print(f"   완전 무변형 Psi(I)     {e_zero:.4e}")
