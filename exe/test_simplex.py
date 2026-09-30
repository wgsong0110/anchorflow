"""사면체 복합체(몸대각선을 z 로 세운 고정 Kuhn) 단위검사.

확인하는 것:
  격자    |a_i-a_j|=h, sum a_i 가 순수 z, 정육면체면 a_i.a_j=0
  쌓기    층이 xy 평행 정삼각 격자이고 ABCABC, 몸대각선이 3 층을 잇는다
  복합체  barycentric 이 합 1·비음수, 아핀 재현이 정확, 셀 경계에서 C0
  야코비안 해석 == autograd, 변위 0 이면 det=1
  간선    14 종이고 전부 양방향, 같은 층 안에는 변이 없다
"""
from __future__ import annotations
import argparse, math, os, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import simplex as SX

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--n_nodes", type=int, default=12)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


# ---------------------------------------------------------------- 격자
h = 0.37
L = SX.lattice(h, SX.HZ_CUBE * h, device=dev, dtype=torch.float64)
A = L.A
d12 = [float((A[:, i] - A[:, j]).norm()) for i, j in ((0, 1), (0, 2), (1, 2))]
chk("면내 최근접 |a_i-a_j| = h", max(abs(v - h) for v in d12) < 1e-12,
    f"{[round(v, 6) for v in d12]}")
sm = A.sum(1)
chk("sum a_i 가 순수 z (몸대각선 ∥ z)",
    float(sm[:2].abs().max()) < 1e-12 and float(sm[2]) > 0,
    f"({float(sm[0]):.1e}, {float(sm[1]):.1e}, {float(sm[2]):.4f})")
dots = [float(A[:, i] @ A[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))]
chk("hz=h/sqrt(6) 이면 정육면체 (a_i.a_j=0)",
    max(abs(v) for v in dots) < 1e-12, f"{[f'{v:.1e}' for v in dots]}")
ln = [float(A[:, i].norm()) for i in range(3)]
chk("세 기저의 길이가 같다", max(ln) - min(ln) < 1e-12,
    f"|a|={ln[0]:.6f} (h/sqrt2={h / math.sqrt(2):.6f})")

# ---------------------------------------------------------------- 쌓기
ij = torch.tensor([[i, j, k] for i in range(4) for j in range(4)
                   for k in range(4)], device=dev, dtype=torch.float64)
P = ij @ A.T
zs = torch.unique((P[:, 2] / L.hz).round().long())
chk("층이 z 로 균일 (정수 배수)", int(zs.numel()) == int(ij.sum(1).max()) + 1,
    f"층 {int(zs.numel())} 개")
# 같은 층(i+j+k 일정) 안에서 최근접 거리가 h 이고 xy 평면에 있다
lay = P[(ij.sum(1) == 3)]
dz = float((lay[:, 2] - lay[0, 2]).abs().max())
chk("한 층이 xy 에 평행", dz < 1e-12, f"dz {dz:.1e}")
if lay.shape[0] > 1:
    dd = torch.cdist(lay, lay) + torch.eye(lay.shape[0], device=dev,
                                           dtype=torch.float64) * 1e9
    chk("층 안 최근접 = h", abs(float(dd.min()) - h) < 1e-12,
        f"{float(dd.min()):.6f}")
# 몸대각선은 같은 사영 위치를 3 층 건너 잇는다
b = A.sum(1)
chk("몸대각선이 3 층을 건넌다",
    abs(float(b[2] / L.hz) - 3.0) < 1e-12, f"{float(b[2] / L.hz):.4f} 층")

# ---------------------------------------------------------------- 복합체
x = torch.rand(a.n, 3, device=dev, dtype=torch.float64) * 0.9 + 0.05
lo, lat, nn = SX.grid_for_nodes(x, a.n_nodes)
idx, lam, aux = SX.locate(x, lo, lat, nn)
chk("barycentric 합 = 1", float((lam.sum(1) - 1).abs().max()) < 1e-12,
    f"{float((lam.sum(1) - 1).abs().max()):.1e}")
chk("barycentric 비음수", float(lam.min()) >= -1e-12, f"min {float(lam.min()):.1e}")
Mtot = int(nn[0] * nn[1] * nn[2])
npos_all = SX.node_pos(lo, lat, nn, torch.arange(Mtot, device=dev))
rec = (lam.unsqueeze(-1) * npos_all[idx]).sum(1)
chk("아핀 재현 (사면체가 점을 담는다)",
    float((rec - x).norm(dim=-1).max()) < 1e-10,
    f"최대 {float((rec - x).norm(dim=-1).max()):.2e}")

# C0: 임의 노드 변위를 주고 아주 가까운 두 점의 변위 차가 작아야 한다
dp = torch.randn(Mtot, 3, device=dev, dtype=torch.float64) * 0.01
u1 = SX.g2p(x, lo, lat, nn, dp)
eps = 1e-7
u2 = SX.g2p(x + eps, lo, lat, nn, dp)
chk("셀 경계 넘어 C0", float((u2 - u1).norm(dim=-1).max()) < 1e-6,
    f"최대 {float((u2 - u1).norm(dim=-1).max()):.2e}")

# ---------------------------------------------------------------- 야코비안
xs = x[:2000].clone().requires_grad_(True)
u, G = SX.g2p_jac(xs, lo, lat, nn, dp)
Ja = torch.zeros(xs.shape[0], 3, 3, device=dev, dtype=torch.float64)
for i in range(3):
    gi, = torch.autograd.grad(u[:, i].sum(), xs, retain_graph=True)
    Ja[:, i] = gi
pj = (G - Ja).abs().amax(dim=(1, 2))
chk("해석 야코비안 == autograd", float(pj.median()) < 1e-10,
    f"중앙 {float(pj.median()):.1e}, 최대 {float(pj.max()):.1e}")
_, G0 = SX.g2p_jac(x, lo, lat, nn, torch.zeros_like(dp))
chk("변위 0 이면 det=1", float((SX.tet_det(G0) - 1).abs().max()) < 1e-12,
    f"{float((SX.tet_det(G0) - 1).abs().max()):.1e}")

# ---------------------------------------------------------------- 간선
rows, uniq = SX.active_nodes(idx)
src, dst, cls = SX.edges_of(rows, uniq, nn)
chk("간선 클래스가 표 안에 있다", int((cls < 0).sum()) == 0,
    f"미분류 {int((cls < 0).sum())}")
used = torch.unique(cls)
chk("14 종이 모두 쓰인다", int(used.numel()) == SX.N_EDGE_CLASS == 14,
    f"{int(used.numel())}/{SX.N_EDGE_CLASS}")
key = src * (uniq.numel() + 1) + dst
rkey = dst * (uniq.numel() + 1) + src
chk("간선이 전부 양방향",
    bool(torch.isin(rkey, key).all()), "")
nnl = [int(nn[k]) for k in range(3)]
zc = uniq % nnl[2]
yc = (uniq // nnl[2]) % nnl[1]
xc = uniq // (nnl[1] * nnl[2])
lvl = xc + yc + zc                       # i+j+k 가 층 번호다
chk("같은 층 안에는 변이 없다",
    int((lvl[dst] == lvl[src]).sum()) == 0,
    f"층 내 변 {int((lvl[dst] == lvl[src]).sum())}")
dl = (lvl[dst] - lvl[src]).abs()
chk("변은 1·2·3 층만 건넌다",
    bool(torch.isin(dl, torch.tensor([1, 2, 3], device=dev)).all()),
    f"고유 {sorted(set(dl.tolist()))[:6]}")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
