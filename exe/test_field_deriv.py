"""변형장의 t-미분(속도)과 야코비안(변형구배 갱신)을 검사한다.

학생의 물리량 갱신이 여기에 얹혀 있다:
    v_{n+1} = dPhi_t(x)/dt        (jvp, 유한차분과 맞아야 한다)
    F_{n+1} = grad_x Phi . F_n    (해석 야코비안, autograd 와 맞아야 한다)
둘 중 하나만 틀려도 롤아웃이 조용히 어긋난다.

PL·C0 사상이라 셀 면(비평활 집합)에서는 해석과 차분이 서로 다른 한쪽 값을 줄 수
있다 -- 개수로 FAIL 하지 않고 **중앙값**으로 본다 (예전에 전역 노름비로 재서 몇
개의 면집합 점이 판정을 지배한 적이 있다).
"""
from __future__ import annotations
import argparse, os, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import simplex as SX
from anchorflow.simplex_gnn import SimplexGNN, node_moments

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--n_nodes", type=int, default=16)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
DT = torch.float64
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


x = (torch.rand(a.n, 3, device=dev, dtype=DT) * 0.8 + 0.6)
v = torch.randn_like(x) * 0.1
X = x.clone()
mw = torch.full((a.n,), 1.0 / a.n, device=dev, dtype=DT)
lo, lat, nn = SX.grid_for_nodes(x, a.n_nodes)
idx, lam, aux = SX.locate(x, lo, lat, nn)
rows, uniq = SX.active_nodes(idx)
Mn = int(uniq.numel())
npos = SX.node_pos(lo, lat, nn, uniq)
feat = node_moments(x, v, X, mw, rows, lam, Mn, npos, lat.s)
net = SimplexGNN(feat.shape[-1], hidden=64, layers=2, scale=1.0,
                 dt_cond=True, dt_ref=1 / 60, dt_scale=True).to(dev).double()
# 출력 가중을 0 이 아니게 (초기화가 0 이라 그대로면 항등이다)
with torch.no_grad():
    net.out.weight.normal_(0, 0.02)
    net.out.bias.normal_(0, 0.01)
src, dst, cls = SX.edges_of(rows, uniq, nn)


def field(tau):
    dp = net(feat, src, dst, cls, tau)[0]
    return x + SX.g2p_pre(rows, lam, dp)


# ---------------------------------------------------- 속도 = dPhi/dt
tau0 = torch.tensor(1 / 60, device=dev, dtype=DT)
(q0, ), (vt, ) = torch.func.jvp(lambda t: (field(t),), (tau0,),
                                (torch.ones_like(tau0),))
h = 1e-6
fd = (field(tau0 + h) - field(tau0 - h)) / (2 * h)
rel = (vt - fd).norm(dim=-1) / fd.norm(dim=-1).clamp_min(1e-12)
chk("v = dPhi/dt == t 중심차분", float(rel.median()) < 1e-6,
    f"중앙 {float(rel.median()):.2e}, 1e-3 초과 {int((rel > 1e-3).sum())}/{a.n} (면집합)")

# 출력이 dt 에 비례(--dt_scale)하므로 tau -> 0 에서 Phi -> x 여야 한다
q_small = field(tau0 * 1e-6)
chk("tau -> 0 이면 Phi -> x (dt_scale)",
    float((q_small - x).norm(dim=-1).max()) < 1e-6,
    f"최대 {float((q_small - x).norm(dim=-1).max()):.2e}")

# ---------------------------------------------------- 야코비안
dp = net(feat, src, dst, cls, tau0)[0].detach()
xs = x[:3000].clone().requires_grad_(True)
r2, l2, aux2 = SX.locate(xs, lo, lat, nn)
rw2, uq2 = SX.active_nodes(r2)
# 같은 노드 집합을 쓰도록 전체 격자로 펴서 비교한다
Mtot = int(nn[0] * nn[1] * nn[2])
dpf = torch.zeros(Mtot, 3, device=dev, dtype=DT).index_copy(0, uniq, dp)
u, G = SX.g2p_jac(xs, lo, lat, nn, dpf)
Ja = torch.zeros(xs.shape[0], 3, 3, device=dev, dtype=DT)
for i in range(3):
    gi, = torch.autograd.grad(u[:, i].sum(), xs, retain_graph=True)
    Ja[:, i] = gi
pj = (G - Ja).abs().amax(dim=(1, 2))
chk("grad_x Phi == autograd", float(pj.median()) < 1e-10,
    f"중앙 {float(pj.median()):.1e}, 최대 {float(pj.max()):.1e}")
# g2p_jac 와 g2p_jac_pre 가 같은 값을 줘야 한다 (최적화가 정의를 안 바꿨나)
u2, G2 = SX.g2p_jac_pre(rw2, l2, aux2, lat, dp[uq2] if False else dp)
chk("g2p_jac_pre == g2p_jac",
    float((G2 - G).abs().max()) < 1e-12 and float((u2 - u).abs().max()) < 1e-12,
    f"G 차 {float((G2 - G).abs().max()):.1e}, u 차 {float((u2 - u).abs().max()):.1e}")
# 아핀 사상이면 야코비안이 사면체별 상수 -- 같은 사면체 안 두 점이 같아야 한다
tid = SX.tet_id(lo, lat, nn, aux2)
o = torch.argsort(tid)
same = tid[o][:-1] == tid[o][1:]
if int(same.sum()):
    d = (G[o][:-1][same] - G[o][1:][same]).abs().amax(dim=(1, 2))
    chk("야코비안이 사면체별 상수", float(d.max()) < 1e-12,
        f"최대 {float(d.max()):.1e} ({int(same.sum())} 쌍)")
else:
    chk("야코비안이 사면체별 상수", False, "같은 사면체 쌍이 없다")
# det = det(I + grad u) 가 1 근처 (작은 변위)
det = SX.tet_det(G)
chk("det > 0", float(det.min()) > 0, f"최소 {float(det.min()):.4f}")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
