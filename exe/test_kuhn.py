"""방향 비의존(대칭 8분할) barycentric 전달의 정합성 검사.

  python exe/test_kuhn.py
"""
import itertools

import torch

from anchorflow.sitreg_warp import (BoundedWarp, WARP_BOUND,
                                    _sym_locate, bary_g2p, bary_g2p_jac)

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
K = 9
n3 = torch.tensor([K, K, K])
lo = torch.zeros(3, device=dev, dtype=DT)
h = 0.125
M = K ** 3
gpos = (torch.stack(torch.meshgrid(
    *[torch.arange(K, device=dev, dtype=DT)] * 3, indexing="ij"),
    -1).reshape(-1, 3)) * h
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


q = torch.rand(20000, 3, device=dev, dtype=DT) * ((K - 1) * h)

# 1) 분할합: 8꼭짓점 가중치는 합이 1, 전부 0 이상
idx, w, _aux = _sym_locate(q, lo, h, n3)
chk("가중치 합=1", float((w.sum(-1) - 1).abs().max()) < 1e-12,
    f"max err {float((w.sum(-1)-1).abs().max()):.1e}")
chk("가중치 >= 0", bool((w > -1e-12).all()), f"min {float(w.min()):.1e}")

# 2) 아핀 재현: 꼭짓점 값이 dp = A v + b 면 보간이 **정확히** A q + b
A = torch.randn(3, 3, device=dev, dtype=DT) * 0.3
b = torch.randn(3, device=dev, dtype=DT)
dp_aff = gpos @ A.T + b
u = bary_g2p(q, lo, h, n3, dp_aff)
err = (u - (q @ A.T + b)).norm(dim=-1).max()
chk("아핀 장 정확 재현 (PL)", float(err) < 1e-12, f"max {float(err):.1e}")

# 3) 연속성: 사면체 면(정렬 동률)·소큐브 경계(f=0.5)·큐브 면 전부에서 잇긴다
dp = torch.randn(M, 3, device=dev, dtype=DT) * 0.1
eps = 1e-9
qq = torch.rand(6000, 3, device=dev, dtype=DT) * ((K - 1) * h)
qq[:1500, 0] = qq[:1500, 1]                       # 동률면 (같은 옥탄트)
_c5 = (qq[1500:3000, 0] / h).floor() * h + 0.5 * h
qq[1500:3000, 0] = _c5                            # 소큐브 경계 f=0.5
qq[3000:4500, 0] = (qq[3000:4500, 0] / h).round() * h   # 큐브 면
d = torch.randn_like(qq); d = d / d.norm(dim=-1, keepdim=True)
hi = (K - 1) * h - 1e-7
va = bary_g2p((qq + eps * d).clamp(0, hi), lo, h, n3, dp)
vb = bary_g2p((qq - eps * d).clamp(0, hi), lo, h, n3, dp)
cerr = (va - vb).norm(dim=-1).max()
chk("면에서 연속 (C0)", float(cerr) < 1e-6, f"max jump {float(cerr):.1e}")

# 4) **방향 비의존**: 큐브 대칭군 48개 전부에 대해 등변 (이게 이 분할의 목적)
ctr = torch.full((3,), (K - 1) * h / 2.0, device=dev, dtype=DT)
ii = torch.arange(K, device=dev)
grid_idx = torch.stack(torch.meshgrid(ii, ii, ii, indexing="ij"),
                       -1).reshape(-1, 3)                     # [M,3]
qs = torch.rand(3000, 3, device=dev, dtype=DT) * ((K - 1) * h)
worst = 0.0
n_ok = 0
for perm in itertools.permutations(range(3)):
    for flips in itertools.product([1, -1], repeat=3):
        P = torch.zeros(3, 3, device=dev, dtype=DT)
        for r, (pr, fl) in enumerate(zip(perm, flips)):
            P[r, pr] = fl
        # 격자 색인 변환: n' = P 적용 (뒤집힌 축은 K-1-n)
        src = grid_idx[:, list(perm)]
        for r, fl in enumerate(flips):
            if fl < 0:
                src[:, r] = (K - 1) - src[:, r]
        # dp'_{v'} = P dp_v  즉 dp'[flat(n')] = P dp[flat(n)]
        dst = (src[:, 0] * K + src[:, 1]) * K + src[:, 2]
        dp2 = torch.zeros_like(dp)
        dp2[dst] = dp @ P.T
        Tq = (qs - ctr) @ P.T + ctr
        lhs = bary_g2p(Tq, lo, h, n3, dp2)
        rhs = bary_g2p(qs, lo, h, n3, dp) @ P.T
        e = float((lhs - rhs).norm(dim=-1).max())
        worst = max(worst, e)
        n_ok += int(e < 1e-10)
chk("48 대칭 전부 등변", n_ok == 48,
    f"{n_ok}/48, 최악 어긋남 {worst:.1e}")

# 5) 해석 ∇u == autograd (동률·0.5 면에서 떨어진 내부점)
g0 = torch.rand(500, 3, device=dev, dtype=DT) * 0.34 + 0.05   # (0.05,0.39)
g0 = g0 + torch.arange(3, device=dev, dtype=DT) * 0.02        # 동률 회피
sb = torch.randint(0, 2, (500, 3), device=dev, dtype=torch.bool)
f0 = torch.where(sb, 1.0 - g0, g0)
_ci = torch.randint(0, K - 1, (500, 3), device=dev).to(DT)
xs = ((_ci + f0) * h).requires_grad_(True)
u2, G = bary_g2p_jac(xs, lo, h, n3, dp)
Ga = torch.stack([torch.autograd.grad(
    bary_g2p(xs, lo, h, n3, dp)[:, k].sum(), xs, retain_graph=True)[0]
    for k in range(3)], 1)
r = float((G - Ga).norm() / Ga.norm().clamp_min(1e-30))
chk("해석 ∇u == autograd", r < 1e-10, f"상대오차 {r:.1e}")
rv = float((u2 - bary_g2p(xs, lo, h, n3, dp)).abs().max())
chk("jac 판 값 일치", rv < 1e-14, f"max {rv:.1e}")

# 6) 상한 + K 합성: det > 0 (성분 상한 h/6 이면 Lipschitz<1, 분할 무관)
wp = BoundedWarp(WARP_BOUND * h, 5)
big = torch.randn(M, 3, device=dev, dtype=DT) * 10.0
_, J = wp.apply_jac(xs.detach(), big,
                    lambda z, c: bary_g2p_jac(z, lo, h, n3, c))
det = torch.linalg.det(J)
chk("상한 K=5 합성 det > 0", bool((det > 0).all()),
    f"최소 {float(det.min()):.4f}")

# 7) dp 에 선형 (dPhi/dt 정확검사의 전제)
u_a = bary_g2p(q, lo, h, n3, dp)
u_b = bary_g2p(q, lo, h, n3, 2.0 * dp)
chk("dp 에 선형", float((u_b - 2 * u_a).abs().max()) < 1e-12)

print("ALL-OK" if ok else "SOME-FAIL", flush=True)
raise SystemExit(0 if ok else 1)
