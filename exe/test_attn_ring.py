"""--attn_ring K 단위 확인 (사용자 10-23). GPU 하나, 수 초.

  1) K=1 기존 동작 불변: 이전 커밋의 simplex_gnn.py (--old 경로) 의 LocalAttnGNN 과 같은 시드로 만들어
     파라미터·출력·기울기가 **비트 단위로 같은지**
  2) ring_edges K=2,3 == 조밀 BFS (인접행렬 거듭제곱, 자기 제외) 이웃 집합, 오프셋 == pos[dst]-pos[src]
  3) K=2,3 망: 출력 0 초기화·기울기 흐름·같은 dst 끼리 어텐션 가중 합 1
  python exe/test_attn_ring.py --old <옛 simplex_gnn.py>
"""
import argparse
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX                       # noqa: E402
from anchorflow import simplex_gnn as SG                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True, help="이전 커밋 simplex_gnn.py 사본 (같은 패키지 안에 두어 상대 import 가 되게)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
OK = [True]


def chk(name, ok, msg=""):
    print(f"[{'통과' if ok else '실패'}] {name} {msg}", flush=True)
    OK[0] &= bool(ok)


torch.manual_seed(0)
N = 4000
x = torch.rand(N, 3, device=dev, dtype=torch.float64) * 0.3 + 0.5
lo, lat, nn = SX.grid_for_nodes(x, 0, h_fix=0.028284)
idx, lam, aux = SX.locate(x, lo, lat, nn)
rows, uniq = SX.active_nodes(idx)
M = uniq.numel()
src, dst, cls = SX.edges_of(rows, uniq, nn)
nnl = [int(nn[k]) for k in range(3)]
pos = torch.stack([uniq // (nnl[1] * nnl[2]), (uniq // nnl[2]) % nnl[1], uniq % nnl[2]], -1)
print(f"노드 {M}, 1 링 간선 {src.numel()}, 노드당 평균 {src.numel() / M:.2f}")

# 1) K=1 불변
spec = importlib.util.spec_from_file_location("anchorflow._old_simplex_gnn", a.old)
OLD = importlib.util.module_from_spec(spec)
sys.modules["anchorflow._old_simplex_gnn"] = OLD
spec.loader.exec_module(OLD)
for L in (4, 8):
    torch.manual_seed(123)
    n_old = OLD.LocalAttnGNN(n_feat=20, hidden=128, layers=L, heads=8, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
    torch.manual_seed(123)
    n_new = SG.LocalAttnGNN(n_feat=20, hidden=128, layers=L, heads=8, ring=1, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
    so, sn = n_old.state_dict(), n_new.state_dict()
    chk(f"K=1 {L}층 파라미터 이름·값 같음", list(so) == list(sn) and all(torch.equal(so[k], sn[k]) for k in so))
    for n_ in (n_old, n_new):
        torch.manual_seed(7)
        torch.nn.init.normal_(n_.out.weight, std=0.01)
    feat = torch.randn(M, 20, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
    yo = n_old(feat, src, dst, cls, 1 / 24)[0]
    yn = n_new(feat, src, dst, cls, 1 / 24)[0]
    chk(f"K=1 {L}층 출력 비트 같음", torch.equal(yo, yn), f"최대차 {float((yo - yn).abs().max()):.3e}")
    yo.square().sum().backward(); yn.square().sum().backward()
    go = [p.grad for p in n_old.parameters()]; gn = [p.grad for p in n_new.parameters()]
    chk(f"K=1 {L}층 기울기 비트 같음", all((p is None and q is None) or torch.equal(p, q) for p, q in zip(go, gn)))
    rs = n_new.ring_stats
    chk(f"K=1 이웃 목록 = 입력 그대로", rs["E"] == src.numel(), str(rs))

# 2) ring_edges == 조밀 BFS
A = torch.zeros(M, M, dtype=torch.bool, device=dev)
A[dst, src] = True
reach = A.clone()
cur = A.clone()
eye = torch.eye(M, dtype=torch.bool, device=dev)
for K in (2, 3):
    cur = (cur.float() @ A.float() > 0) | cur
    reach = cur & ~eye
    s2, d2, c2 = SG.ring_edges(src, dst, cls, K, n_node=M)
    R = torch.zeros(M, M, dtype=torch.bool, device=dev)
    R[d2, s2] = True
    chk(f"K={K} 이웃 집합 == 조밀 BFS", torch.equal(R, reach) and int(R.sum()) == d2.numel(),
        f"간선 {d2.numel()} (BFS {int(reach.sum())}), 중복 {d2.numel() - int(R.sum())}")
    chk(f"K={K} 자기 제외", bool((s2 != d2).all()))
    tab = torch.tensor(SG.ring_offset_table(K), device=dev)
    chk(f"K={K} 오프셋 == pos[dst]-pos[src]", torch.equal(tab[c2], pos[d2] - pos[s2]))
    chk(f"K={K} 1 링 포함", bool(R[dst, src].all()))
    nb = torch.bincount(d2, minlength=M).float()
    print(f"    K={K} 노드당 이웃 최소 {int(nb.min())} 평균 {float(nb.mean()):.2f} 최대 {int(nb.max())}", flush=True)
    # 3) 망
    net = SG.LocalAttnGNN(n_feat=20, hidden=128, layers=4, heads=8, ring=K, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
    feat = torch.randn(M, 20, device=dev)
    chk(f"K={K} 출력 0 초기화", float(net(feat, src, dst, cls, 1 / 24)[0].abs().max()) == 0.0)
    chk(f"K={K} ring_stats", net.ring_stats["E"] == d2.numel(), str(net.ring_stats))
    torch.nn.init.normal_(net.out.weight, std=0.01)
    L0 = net.body[0]
    h0 = torch.randn(M, 128, device=dev)
    q, k, vv = L0.qkv(L0.ln1(h0)).split(128, -1)
    rt = L0.rpe(L0.off)
    lg = (q[d2].reshape(-1, 8, 16) * (k[s2] + rt[c2, :128]).reshape(-1, 8, 16)).sum(-1) / 4 + rt[c2, 256:]
    mx = torch.full((M, 8), -1e30, device=dev).scatter_reduce(0, d2.unsqueeze(-1).expand(-1, 8), lg.detach(), reduce="amax")
    ex = torch.exp(lg - mx[d2])
    den = torch.zeros(M, 8, device=dev).index_add(0, d2, ex)
    ssum = torch.zeros(M, 8, device=dev).index_add(0, d2, ex / den[d2])
    chk(f"K={K} 같은 dst 끼리 가중 합 1", float((ssum - 1).abs().max()) < 1e-5)
    o = net(feat, src, dst, cls, 1 / 24)[0]
    o.square().sum().backward()
    g = [p.grad for p in net.body.parameters() if p.grad is not None]
    chk(f"K={K} 몸통 기울기 흐름", len(g) > 0 and all(torch.isfinite(t).all() for t in g) and sum(float(t.abs().sum()) for t in g) > 0)
print("TEST_ATTN_RING", "OK" if OK[0] else "FAIL", flush=True)
sys.exit(0 if OK[0] else 1)
