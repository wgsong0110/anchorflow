"""Kuhn 사면체 분할 barycentric 전달의 정합성 검사.

  python exe/test_kuhn.py
"""
import torch

from anchorflow.sitreg_warp import (BoundedWarp, WARP_BOUND,
                                    _kuhn_locate, bary_g2p, bary_g2p_jac)

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
n3 = torch.tensor([9, 9, 9])
lo = torch.zeros(3, device=dev, dtype=DT)
h = 0.125
M = 9 ** 3
gpos = (torch.stack(torch.meshgrid(
    *[torch.arange(9, device=dev, dtype=DT)] * 3, indexing="ij"),
    -1).reshape(-1, 3)) * h
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


q = torch.rand(20000, 3, device=dev, dtype=DT) * (8 * h)

# 1) 분할합: barycentric 가중치는 합이 1, 전부 0 이상
idx, lam, perm = _kuhn_locate(q, lo, h, n3)
chk("가중치 합=1", float((lam.sum(-1) - 1).abs().max()) < 1e-12,
    f"max err {float((lam.sum(-1)-1).abs().max()):.1e}")
chk("가중치 >= 0", bool((lam > -1e-12).all()),
    f"min {float(lam.min()):.1e}")

# 2) 아핀 재현: 꼭짓점 값이 dp = A v + b 면 보간이 **정확히** A q + b
A = torch.randn(3, 3, device=dev, dtype=DT) * 0.3
b = torch.randn(3, device=dev, dtype=DT)
dp_aff = gpos @ A.T + b
u = bary_g2p(q, lo, h, n3, dp_aff)
err = (u - (q @ A.T + b)).norm(dim=-1).max()
chk("아핀 장 정확 재현 (PL)", float(err) < 1e-12, f"max {float(err):.1e}")

# 3) 연속성: 사면체 면(정렬 동률)과 큐브 면 양쪽에서 값이 이어진다
dp = torch.randn(M, 3, device=dev, dtype=DT) * 0.1
eps = 1e-9
qq = torch.rand(4000, 3, device=dev, dtype=DT) * (8 * h)
qq[:1000, 0] = qq[:1000, 1]                    # 사면체 면 (f_x = f_y)
qq[1000:2000, 0] = (qq[1000:2000, 0] / h).round() * h   # 큐브 면
d = torch.randn_like(qq); d = d / d.norm(dim=-1, keepdim=True)
va = bary_g2p((qq + eps * d).clamp(0, 8 * h - 1e-7), lo, h, n3, dp)
vb = bary_g2p((qq - eps * d).clamp(0, 8 * h - 1e-7), lo, h, n3, dp)
cerr = (va - vb).norm(dim=-1).max()
chk("면에서 연속 (C0)", float(cerr) < 1e-6, f"max jump {float(cerr):.1e}")

# 4) 해석 ∇u == 자동미분 (면에서 떨어진 내부점)
_ci = torch.randint(0, 8, (500, 3), device=dev).to(DT)
f0 = torch.rand(500, 3, device=dev, dtype=DT)
f0 = f0 + torch.where(f0.diff(dim=-1, prepend=f0[:, :1] + 0.2).abs() < 0.05,
                      0.07, 0.0)               # 동률 근처 피함
xs = ((_ci + f0.clamp(0.03, 0.97)) * h).requires_grad_(True)
u2, G = bary_g2p_jac(xs, lo, h, n3, dp)
Ga = torch.stack([torch.autograd.grad(
    bary_g2p(xs, lo, h, n3, dp)[:, k].sum(), xs, retain_graph=True)[0]
    for k in range(3)], 1)
r = float((G - Ga).norm() / Ga.norm().clamp_min(1e-30))
chk("해석 ∇u == autograd", r < 1e-10, f"상대오차 {r:.1e}")

# 5) 상한 + K 합성: det > 0 (h/6 성분 상한이면 Lipschitz<1)
w = BoundedWarp(WARP_BOUND * h, 5)
big = torch.randn(M, 3, device=dev, dtype=DT) * 10.0
_, J = w.apply_jac(xs.detach(), big,
                   lambda z, c: bary_g2p_jac(z, lo, h, n3, c))
det = torch.linalg.det(J)
chk("상한 K=5 합성 det > 0", bool((det > 0).all()),
    f"최소 {float(det.min()):.4f}")

# 6) dp 에 선형: 값이 dp 의 선형함수라 dp ∝ t 면 dPhi/dt == 할선 (배선 검사용)
u_a = bary_g2p(q, lo, h, n3, dp)
u_b = bary_g2p(q, lo, h, n3, 2.0 * dp)
chk("dp 에 선형", float((u_b - 2 * u_a).abs().max()) < 1e-12)

print("ALL-OK" if ok else "SOME-FAIL", flush=True)
raise SystemExit(0 if ok else 1)
