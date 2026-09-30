"""전체 입자 수에서 **순전파 단계별 GPU 비용**을 잰다.

물음: "속도 때문에 서브샘플링이 필요하다" 면 같은 집계가 추론 순전파에도 들어가니
실시간도 안 된다는 뜻이다. 그래서 전체 입자에서 집계·GNN·전달의 GPU 시간을 재서
프레임 예산(60FPS = 16.7ms) 안에 들어가는지 직접 본다.

CUDA 이벤트로 **GPU 실행 구간만** 재므로 CPU 공유와 무관하다. 단계마다 반복 후
중앙값을 쓴다 (첫 호출의 커널 컴파일·캐시는 예열로 버린다).

  python exe/bench_fullN.py --traj W/traj_h2/...pt --n 8000,20000,64000,251001
"""
from __future__ import annotations
import argparse, os, statistics, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import simplex as SX
from anchorflow.simplex_gnn import SimplexGNN, node_moments

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--n", default="8000,20000,64000,251001")
ap.add_argument("--n_nodes", type=int, default=32)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--rep", type=int, default=20)
ap.add_argument("--warm", type=int, default=5)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


d = _load(a.traj)
x0 = d["x"][3].float()
v0 = (d["x"][3].float() - d["x"][2].float()) / (1.0 / 60)
print(f"[궤적] 저장 입자 {x0.shape[0]}, n_full {int(d.get('n_full', 0))}")


def cloud(n):
    """n 개 점 구름. 저장 입자보다 많이 필요하면 **같은 부피 안에서** 국소 지터로
    늘린다 -- 점유 사면체 집합은 거의 같고 노드당 입자만 늘어나므로, 실제 전체
    밀도 상황의 비용 구조를 그대로 재현한다."""
    if n <= x0.shape[0]:
        s = torch.randperm(x0.shape[0])[:n]
        return x0[s].to(dev), v0[s].to(dev)
    r = (n + x0.shape[0] - 1) // x0.shape[0]
    xs = x0.repeat(r, 1)[:n].clone()
    vs = v0.repeat(r, 1)[:n].clone()
    ext = float((x0.max(0).values - x0.min(0).values).max())
    sp = ext / max(x0.shape[0], 1) ** (1 / 3)
    xs += (torch.rand_like(xs) - 0.5) * sp
    return xs.to(dev), vs.to(dev)


def timed(fn, rep, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        e0, e1 = (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True))
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


print(f"\n{'N':>8} {'집계':>9} {'GNN':>8} {'전달':>8} "
      f"{'합':>9} {'FPS':>7} {'노드':>7} {'간선':>9}")
for n in [int(q) for q in a.n.split(",")]:
    x, v = cloud(n)
    X = x.clone()
    mw = torch.full((n,), 1.0 / n, device=dev)
    lo, lat, nn_ = SX.grid_for_nodes(x, a.n_nodes)
    with torch.no_grad():
        idx, lam, aux = SX.locate(x, lo, lat, nn_)
        rows, uniq = SX.active_nodes(idx)
        Mn = int(uniq.numel())
        npos = SX.node_pos(lo, lat, nn_, uniq)
        hn = lat.s
        src, dst, cls = SX.edges_of(rows, uniq, nn_)
        feat = node_moments(x, v, X, mw, rows, lam, Mn, npos, hn)
        nf = feat.shape[-1]
        net = SimplexGNN(nf, hidden=a.hidden, layers=a.layers, scale=1.0,
                         dt_cond=True, dt_ref=1 / 60,
                         dt_scale=True).to(dev).eval()
        tau = torch.tensor(1 / 60, device=dev)

        def f_agg():
            i2, l2, _ = SX.locate(x, lo, lat, nn_)
            r2, u2 = SX.active_nodes(i2)
            return node_moments(x, v, X, mw, r2, l2, int(u2.numel()),
                                npos[:int(u2.numel())], hn)

        def f_gnn():
            return net(feat, src, dst, cls, tau)

        out = net(feat, src, dst, cls, tau)
        Mtot = int(nn_[0] * nn_[1] * nn_[2])
        dpf = torch.zeros(Mtot, 3, device=dev).index_copy(0, uniq, out[0])

        def f_g2p():
            return x + SX.g2p(x, lo, lat, nn_, dpf)

        t_a = timed(f_agg, a.rep, a.warm)
        t_g = timed(f_gnn, a.rep, a.warm)
        t_p = timed(f_g2p, a.rep, a.warm)
    tot = t_a + t_g + t_p
    print(f"{n:>8} {t_a:>8.2f}ms {t_g:>7.2f}ms {t_p:>7.2f}ms "
          f"{tot:>8.2f}ms {1000/tot:>6.1f} {Mn:>7} {int(src.numel()):>9}")
print("\n60FPS 예산 16.67ms / 30FPS 33.3ms")
