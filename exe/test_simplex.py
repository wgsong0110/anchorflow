"""사면체 복합체(사면체가 셀) 의 정합성 검사.

  python exe/test_simplex.py
"""
import itertools

import torch

from anchorflow import simplex as SX

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


# 노드 간격만 준다 (발판 격자는 내부 구현)
pts = torch.rand(20000, 3, device=dev, dtype=DT)
lo, hn, nn = SX.grid_for_nodes(pts, 16)
M = int(nn[0] * nn[1] * nn[2])
print(f"[격자] 노드 간격 {hn:.5f}, 노드 {tuple(int(c) for c in nn)} = {M}, "
      f"발판 셀 {tuple((int(c)-1)//2 for c in nn)}")
chk("축별 노드 수 홀수 (발판 셀로 떨어짐)",
    all(int(c) % 2 == 1 for c in nn), f"{[int(c) for c in nn]}")

npos = (torch.stack(torch.meshgrid(
    *[torch.arange(int(nn[k]), device=dev, dtype=DT) for k in range(3)],
    indexing="ij"), -1).reshape(-1, 3)) * hn + lo          # 노드 좌표

idx, lam, aux = SX.locate(pts, lo, hn, nn)

# 1) barycentric 기본 성질
chk("가중치 합=1", float((lam.sum(-1) - 1).abs().max()) < 1e-12,
    f"max {float((lam.sum(-1)-1).abs().max()):.1e}")
chk("가중치 >= 0", bool((lam > -1e-12).all()), f"min {float(lam.min()):.1e}")

# 2) 사면체 꼭짓점이 실제로 그 점을 감싼다: 재구성이 원점과 일치
rec = (lam.unsqueeze(-1) * npos[idx]).sum(1)
chk("사면체가 점을 감싼다 (재구성 일치)",
    float((rec - pts).norm(dim=-1).max()) < 1e-12,
    f"max {float((rec - pts).norm(dim=-1).max()):.1e}")

# 3) 아핀 재현: 노드 값이 A v + b 면 보간이 정확히 A q + b
A = torch.randn(3, 3, device=dev, dtype=DT) * 0.3
b = torch.randn(3, device=dev, dtype=DT)
u = SX.g2p(pts, lo, hn, nn, npos @ A.T + b)
chk("아핀 장 정확 재현 (PL)",
    float((u - (pts @ A.T + b)).norm(dim=-1).max()) < 1e-12)

# 4) 연속성 (C0): 사면체 면·소큐브 경계·발판 셀 면 전부
dp = torch.randn(M, 3, device=dev, dtype=DT) * 0.02
eps = 1e-9
qq = torch.rand(6000, 3, device=dev, dtype=DT)
qq[:1500, 0] = qq[:1500, 1]                                   # 동률면
_h2 = 2.0 * hn
qq[1500:3000, 0] = ((qq[1500:3000, 0] - lo[0]) / _h2).floor() * _h2 + lo[0] + hn
qq[3000:4500, 0] = ((qq[3000:4500, 0] - lo[0]) / _h2).round() * _h2 + lo[0]
dd = torch.randn_like(qq); dd = dd / dd.norm(dim=-1, keepdim=True)
va = SX.g2p(qq + eps * dd, lo, hn, nn, dp)
vb = SX.g2p(qq - eps * dd, lo, hn, nn, dp)
chk("면에서 연속 (C0)", float((va - vb).norm(dim=-1).max()) < 1e-6,
    f"max jump {float((va - vb).norm(dim=-1).max()):.1e}")

# 5) 48 대칭 등변 -- 방향 비의존 분할인지
ii = torch.arange(int(nn[0]), device=dev)
gidx = torch.stack(torch.meshgrid(ii, ii, ii, indexing="ij"), -1).reshape(-1, 3)
K0 = int(nn[0])
ctr = lo + (K0 - 1) * hn / 2.0
qs = torch.rand(2000, 3, device=dev, dtype=DT)
n_ok, worst = 0, 0.0
for perm in itertools.permutations(range(3)):
    for flips in itertools.product([1, -1], repeat=3):
        P = torch.zeros(3, 3, device=dev, dtype=DT)
        for r, (pr, fl) in enumerate(zip(perm, flips)):
            P[r, pr] = fl
        src = gidx[:, list(perm)].clone()
        for r, fl in enumerate(flips):
            if fl < 0:
                src[:, r] = (K0 - 1) - src[:, r]
        dst = (src[:, 0] * K0 + src[:, 1]) * K0 + src[:, 2]
        dp2 = torch.zeros_like(dp)
        dp2[dst] = dp @ P.T
        Tq = (qs - ctr) @ P.T + ctr
        e = float((SX.g2p(Tq, lo, hn, nn, dp2)
                   - SX.g2p(qs, lo, hn, nn, dp) @ P.T).norm(dim=-1).max())
        worst = max(worst, e); n_ok += int(e < 1e-10)
chk("48 대칭 전부 등변", n_ok == 48, f"{n_ok}/48, 최악 {worst:.1e}")

# 6) 해석 grad u == autograd (면에서 떨어진 내부점)
g0 = torch.rand(500, 3, device=dev, dtype=DT) * 0.3 + 0.1
g0 = g0 + torch.arange(3, device=dev, dtype=DT) * 0.03      # 동률 회피
sb = torch.randint(0, 2, (500, 3), device=dev, dtype=torch.bool)
ff = torch.where(sb, 1.0 - g0, g0)
nc = [(int(nn[k]) - 1) // 2 for k in range(3)]
ci = torch.stack([torch.randint(0, nc[k], (500,), device=dev) for k in range(3)],
                 -1).to(DT)
xs = (lo + (ci + ff) * _h2).requires_grad_(True)
u2, G = SX.g2p_jac(xs, lo, hn, nn, dp)
Ga = torch.stack([torch.autograd.grad(
    SX.g2p(xs, lo, hn, nn, dp)[:, k].sum(), xs, retain_graph=True)[0]
    for k in range(3)], 1)
r = float((G - Ga).norm() / Ga.norm().clamp_min(1e-30))
chk("해석 grad u == autograd", r < 1e-10, f"상대오차 {r:.1e}")
chk("jac 판 값 일치",
    float((u2 - SX.g2p(xs, lo, hn, nn, dp)).abs().max()) < 1e-14)

# 7) 사면체별 det 가 상수인가 (같은 사면체 안 두 점)
tid = SX.tet_id(lo, hn, nn, aux)
det_all = SX.tet_det(SX.g2p_jac(pts, lo, hn, nn, dp)[1])
_, inv, cnt = torch.unique(tid, return_inverse=True, return_counts=True)
# E[x^2]-E[x]^2 은 상쇄로 정밀도를 잃는다 -- 평균 대비 최대 편차로 본다
mean = torch.zeros(int(cnt.numel()), device=dev, dtype=DT)
mean.index_add_(0, inv, det_all)
mean = mean / cnt.to(DT)
dev_max = (det_all - mean[inv]).abs().max()
chk("같은 사면체 안 det 일정", float(dev_max) < 1e-12,
    f"최대 편차 {float(dev_max):.1e}, 사면체 {int(cnt.numel())}")

# 8) **점유 사면체만** 쓰는가 -- 활성 노드·간선이 점유분에서만 나오는지
rows, uniq = SX.active_nodes(idx)
chk("활성 노드가 전체보다 적다", uniq.numel() < M,
    f"{uniq.numel()}/{M} = {100*uniq.numel()/M:.1f}%")
chk("활성 노드 = 점유 사면체 꼭짓점 합집합",
    bool(torch.equal(uniq, torch.unique(idx.reshape(-1)))))
src, dst, cls = SX.edges_of(rows, uniq, nn)
chk("간선 클래스가 26 종 안", bool((cls >= 0).all() and (cls < 26).all()),
    f"쓰인 종류 {int(cls.unique().numel())}/26, 간선 {src.numel()}")
# 간선 오프셋이 실제로 {-1,0,1}^3 인지 (2배 격자 기준)
nnl = [int(nn[k]) for k in range(3)]
pz = uniq % nnl[2]; py = (uniq // nnl[2]) % nnl[1]
px = uniq // (nnl[1] * nnl[2])
P3 = torch.stack([px, py, pz], -1)
off = P3[dst] - P3[src]
chk("간선 오프셋 in {-1,0,1}^3", bool((off.abs() <= 1).all()))
chk("간선 양방향 대칭",
    bool(torch.equal(torch.unique(torch.stack([src, dst], -1), dim=0),
                     torch.unique(torch.stack([dst, src], -1), dim=0))))

print("ALL-OK" if ok else "SOME-FAIL", flush=True)
raise SystemExit(0 if ok else 1)
