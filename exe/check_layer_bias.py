"""단계별 중간값 몰림 정도 측정.

각 레이어 출력의 bias = F.normalize(x, dim=-1).mean(dim=0).norm()
  0 → 완전 isotropic, 1 → 모든 벡터가 동일 방향
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from anchorflow.physim import GNNSim, _fourier_pe
from anchorflow.graph import knn_graph


def bias(t: torch.Tensor) -> float:
    """노드(행) 방향 편향. t: [N, D]"""
    if t.shape[0] < 2:
        return float("nan")
    return F.normalize(t.float(), dim=-1).mean(dim=0).norm().item()


def run(graph_path, n_nodes, k_nn, hidden_dim, latent_dim, node_dim, dev):
    if graph_path and os.path.exists(graph_path):
        gd = torch.load(graph_path, map_location="cpu", weights_only=False)
        edge_index = gd["edge_index"]
        rest_len   = gd["rest_len"]
        canonical  = torch.randn(n_nodes, 3) * 0.5
    else:
        canonical  = torch.randn(n_nodes, 3) * 0.5
        edge_index = knn_graph(canonical, k=k_nn)
        src, dst   = edge_index
        rest_len   = (canonical[src] - canonical[dst]).norm(dim=-1)

    sim = GNNSim(
        canonical=canonical,
        anchor_colors=torch.rand(n_nodes, 3),
        edge_index=edge_index, rest_len=rest_len,
        T=2, hidden_dim=hidden_dim,
        latent_dim=latent_dim, node_dim=node_dim,
        k_restore=0.0, gravity=0.0,
    ).to(dev).eval()

    src, dst = sim.edge_index
    static   = sim._static

    with torch.no_grad():
        x = sim.canonical.clone()
        v = torch.zeros_like(x)

        # ── 1. state_mlp ────────────────────────────────────────── #
        state = sim.state_mlp(torch.cat([_fourier_pe(x), _fourier_pe(v)], dim=-1))  # [M, H]

        # ── 2. edge_mlp ─────────────────────────────────────────── #
        feat = torch.cat([state[src], state[dst],
                          static[src], static[dst]], dim=-1)
        msg  = sim.edge_mlp(feat)                           # [E, H]

        # ── 3. agg (mean pool) ──────────────────────────────────── #
        agg = torch.zeros(sim.M, sim.hidden_dim, device=dev)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        deg = torch.zeros(sim.M, device=dev).scatter_add_(
            0, dst, torch.ones(dst.shape[0], device=dev))
        agg = agg / deg.unsqueeze(1).clamp(min=1)          # [M, H]

        # ── 4. enc_mlp → node_enc ───────────────────────────────── #
        node_enc = sim.enc_mlp(torch.cat([agg, state], dim=-1))  # [M, H]

        # ── 5. pool → z (global, [1, L]) ──────────────────────────── #
        z = sim.pool_mlp(node_enc.mean(dim=0, keepdim=True))     # [1, L]

        # ── 6. GRU → h (h=zeros) ───────────────────────────────── #
        h_zero = torch.zeros(1, sim.latent_dim, device=dev, dtype=x.dtype)
        h_new  = sim.ssm(z, h_zero)                              # [1, L]

        # ── 7. dec_mlp (h=zeros) ──────────────────────────────── #
        h_broad = h_zero.expand(sim.M, -1)
        a_zero  = sim.dec_mlp(torch.cat([node_enc, h_broad], dim=-1))  # [M, 3]

        # ── 8. dec_mlp (h=random) ────────────────────────────── #
        h_rand  = torch.randn(1, sim.latent_dim, device=dev)
        h_broad2 = h_rand.expand(sim.M, -1)
        a_rand  = sim.dec_mlp(torch.cat([node_enc, h_broad2], dim=-1))

    header = f"{'layer':<25} {'shape':<18} {'bias':>6}"
    print(header)
    print("-" * len(header))

    rows = [
        ("x (position)",        x,        ),
        ("static (node_emb)",   static,   ),
        ("state  = state_mlp",  state,    ),
        ("msg    = edge_mlp",   msg,      ),
        ("agg    = mean_pool",  agg,      ),
        ("node_enc = enc_mlp",  node_enc, ),
        ("a_gnn (h=zeros)",     a_zero,   ),
        ("a_gnn (h=random)",    a_rand,   ),
    ]
    for name, t in rows:
        b = bias(t)
        print(f"  {name:<23} {str(tuple(t.shape)):<18} {b:>6.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",      default=None)
    ap.add_argument("--n_nodes",    type=int, default=512)
    ap.add_argument("--k_nn",       type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--node_dim",   type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(a.graph, a.n_nodes, a.k_nn, a.hidden_dim, a.latent_dim, a.node_dim, dev)


if __name__ == "__main__":
    main()
