"""사면체 내부 재배열(stick-breaking + 단조 RQS) 의 정합성 검사."""
import torch

from anchorflow import simplex as SX

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
K = 8
P = SX.tet_n_params(K)
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


pts = torch.rand(8000, 3, device=dev, dtype=DT)
lo, hn, nn = SX.grid_for_nodes(pts, 12)
M = int(nn[0] * nn[1] * nn[2])

# 1) 파라미터 0 이면 정확히 항등
th0 = torch.zeros(M, P, device=dev, dtype=DT)
q0, lam0, idx0 = SX.tet_remap(pts, lo, hn, nn, th0, K)
chk("항등 초기화", float((q0 - pts).norm(dim=-1).max()) < 2e-3,
    f"max {float((q0 - pts).norm(dim=-1).max()):.2e}")

# 2) 무작위 파라미터: 점이 자기 사면체를 벗어나지 않는다
th = torch.randn(M, P, device=dev, dtype=DT) * 2.0
q1, lam1, idx1 = SX.tet_remap(pts, lo, hn, nn, th, K)
chk("새 barycentric 합=1", float((lam1.sum(-1) - 1).abs().max()) < 1e-12)
chk("새 barycentric >= 0", bool((lam1 > -1e-12).all()),
    f"min {float(lam1.min()):.1e}")
i2, _, _ = SX.locate(q1, lo, hn, nn)
chk("사면체 유지 (꼭짓점 집합 동일)",
    bool((torch.sort(i2, -1).values == torch.sort(idx1, -1).values).all()))

# 3) 단사: 서로 다른 점이 같은 곳으로 가지 않는다 (같은 사면체 안에서 확인)
tid = SX.tet_id(lo, hn, nn, SX.locate(pts, lo, hn, nn)[2])
u, inv, cnt = torch.unique(tid, return_inverse=True, return_counts=True)
big = int(cnt.argmax())
sel = (inv == big).nonzero().squeeze(-1)[:200]
if sel.numel() > 5:
    d0 = torch.cdist(pts[sel], pts[sel])
    d1 = torch.cdist(q1[sel], q1[sel])
    m = ~torch.eye(sel.numel(), dtype=torch.bool, device=dev)
    chk("같은 사면체 안 단사 (거리 0 없음)", float(d1[m].min()) > 1e-12,
        f"최소거리 {float(d1[m].min()):.1e} (원본 {float(d0[m].min()):.1e})")

# 4) 급격한 재배열이 가능한가 (좁은 빈에 높이 몰기)
ths = torch.zeros(M, P, device=dev, dtype=DT)
ths[:, 4] = -8.0
ths[:, K + 4] = 8.0
line = torch.stack([torch.linspace(0.02, 0.98, 400, device=dev, dtype=DT)] * 3,
                   -1) * float(hn) * 0.9 + lo + 0.5 * float(hn)
qa, _, _ = SX.tet_remap(line, lo, hn, nn, ths, K)
sp = float((qa.diff(dim=0).norm(dim=-1) / line.diff(dim=0).norm(dim=-1)).max())
chk("가파른 재배열 가능", sp > 5, f"최대 배율 {sp:.1f}x")

# 5) 기울기가 theta 와 x 로 흐른다
thg = (torch.randn(M, P, device=dev, dtype=DT) * 0.5).requires_grad_(True)
xg = pts.clone().requires_grad_(True)
SX.tet_remap(xg, lo, hn, nn, thg, K)[0].sum().backward()
chk("기울기 흐름", thg.grad is not None and xg.grad is not None
    and float(thg.grad.abs().sum()) > 0
    and bool(torch.isfinite(thg.grad).all()))

print("ALL-OK" if ok else "SOME-FAIL")
raise SystemExit(0 if ok else 1)
