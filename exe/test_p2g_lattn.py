"""--node_feat p2g / --gnn_layer lattn 단위 확인 (사용자 10-23). GPU 하나, 수 초.

  P2G: bspline27 가중 합 1 · 선형 재현(Σ w x_i = x_p, 노드 = 격자 정수점이 맞는지) · p2g_feats == 노드별 직접 합
  lattn: 같은 dst 끼리 어텐션 가중 합 1 · 이웃 수 · 출력 0 초기화(항등) · 기울기 · torch.func.jvp(τ) 통과
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX                       # noqa: E402
from anchorflow import simplex_gnn as SG                   # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OK = [True]


def chk(name, ok, msg=""):
    print(f"[{'통과' if ok else '실패'}] {name} {msg}", flush=True)
    OK[0] &= bool(ok)


N = 4000
x = torch.rand(N, 3, device=dev, dtype=torch.float64) * 0.3 + 0.5
v = torch.randn(N, 3, device=dev, dtype=torch.float64)
F = torch.eye(3, device=dev, dtype=torch.float64) + 0.1 * torch.randn(N, 3, 3, device=dev, dtype=torch.float64)
m = torch.rand(N, device=dev, dtype=torch.float64) + 0.5
lo, lat, nn = SX.grid_for_nodes(x, 0, h_fix=0.028284)
idx, lam, aux = SX.locate(x, lo, lat, nn)
rows, uniq = SX.active_nodes(idx)
Mn = uniq.numel()
npos = SX.node_pos(lo, lat, nn, uniq)
print(f"격자 h={lat.s} hz={lat.hz:.6f} (정육면체 변 {lat.s / 2 ** 0.5:.5f}), 입자 {N}, 활성 노드 {Mn}")

# --- P2G
f27, w27, _ = SX.bspline27(x, lo, lat, nn)
chk("B-스플라인 가중 합 1", float((w27.sum(1) - 1).abs().max()) < 1e-12, f"{float((w27.sum(1) - 1).abs().max()):.2e}")
allpos = SX.node_pos(lo, lat, nn, f27.reshape(-1)).reshape(N, 27, 3)
rep = (w27.unsqueeze(-1) * allpos).sum(1)
chk("선형 재현 Σw x_i = x_p", float((rep - x).abs().max()) < 1e-12, f"{float((rep - x).abs().max()):.2e}")
pos = torch.searchsorted(uniq, f27).clamp(max=Mn - 1)
hit = uniq[pos] == f27
r27 = torch.where(hit, pos, torch.full_like(pos, Mn))
ww = torch.where(hit, w27, torch.zeros_like(w27))
out = SG.p2g_feats(v, F, m, r27, ww, Mn + 1)[:Mn]
chk("폭 13", out.shape == (Mn, SG.P2G_NCH), str(tuple(out.shape)))
# 직접: 노드마다
j = torch.randint(0, Mn, (20,), device=dev)
err = 0.0
for jj in j.tolist():
    wj = (ww * (r27 == jj)).sum(1) * m                       # [N]
    mi = wj.sum()
    vi = (wj.unsqueeze(-1) * v).sum(0) / mi
    Fi = (wj.reshape(-1, 1, 1) * F).sum(0) / mi
    ref = torch.cat([torch.log1p(mi * 1e3).reshape(1), vi, (Fi - torch.eye(3, device=dev, dtype=F.dtype)).reshape(9)])
    err = max(err, float((ref - out[jj]).abs().max()))
chk("노드별 직접 합과 같음", err < 1e-10, f"{err:.2e}")
chk("nan 없음", bool(torch.isfinite(out).all()))
chk("활성 노드 전부 질량 > 0", bool((out[:, 0] > 0).all()), f"최소 log1p(1e3 m) {float(out[:, 0].min()):.3g}")
print("P2G 칸별 [최소, 최대] " + " ".join(f"{c}:[{float(out[:, c].min()):.3g},{float(out[:, c].max()):.3g}]" for c in range(13)))

# --- 국소 어텐션
src, dst, cls = SX.edges_of(rows, uniq, nn)
nb = torch.bincount(dst, minlength=Mn)
chk("이웃 수 > 0 (모든 노드)", bool((nb > 0).all()), f"최소 {int(nb.min())} 평균 {float(nb.float().mean()):.2f} 최대 {int(nb.max())}")
net = SG.LocalAttnGNN(n_feat=20, hidden=128, layers=8, heads=8, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
mp = SG.SimplexGNN(n_feat=20, hidden=128, layers=8, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
print(f"파라미터 수: lattn {sum(p.numel() for p in net.parameters())}, mp {sum(p.numel() for p in mp.parameters())}")
feat = torch.randn(Mn, 20, device=dev)
s32, d32, c32 = src, dst, cls
o = net(feat, s32, d32, c32, 1 / 24)[0]
chk("출력 0 초기화", float(o.abs().max()) == 0.0)
torch.nn.init.normal_(net.out.weight, std=0.01)
# 어텐션 가중 합 1: 첫 층 내부 재현
L0 = net.body[0]
h0 = torch.randn(Mn, 128, device=dev)
q, k, vv = L0.qkv(L0.ln1(h0)).split(128, -1)
rt = L0.rpe(L0.off)
lg = (q[d32].reshape(-1, 8, 16) * (k[s32] + rt[c32, :128]).reshape(-1, 8, 16)).sum(-1) / 4 + rt[c32, 256:]
mx = torch.full((Mn, 8), -1e30, device=dev).scatter_reduce(0, d32.unsqueeze(-1).expand(-1, 8), lg.detach(), reduce="amax")
ex = torch.exp(lg - mx[d32])
den = torch.zeros(Mn, 8, device=dev).index_add(0, d32, ex)
al = ex / den[d32]
ssum = torch.zeros(Mn, 8, device=dev).index_add(0, d32, al)
chk("같은 dst 끼리 가중 합 1", float((ssum - 1).abs().max()) < 1e-5, f"{float((ssum - 1).abs().max()):.2e}")
o = net(feat, s32, d32, c32, 1 / 24)[0]
o.square().sum().backward()
g = [p.grad for p in net.body.parameters() if p.grad is not None]
chk("몸통 기울기 흐름", len(g) > 0 and all(torch.isfinite(t).all() for t in g) and sum(float(t.abs().sum()) for t in g) > 0)
# jvp (--v_from_dt 경로가 쓰는 것)
tau = torch.tensor(1 / 24, device=dev)
y, ty = torch.func.jvp(lambda t_: net(feat, s32, d32, c32, t_)[0], (tau,), (torch.ones_like(tau),))
chk("torch.func.jvp 통과", bool(torch.isfinite(ty).all()), f"|dy/dτ| {float(ty.abs().max()):.3g}")
print("TEST_P2G_LATTN", "OK" if OK[0] else "FAIL", flush=True)
sys.exit(0 if OK[0] else 1)
