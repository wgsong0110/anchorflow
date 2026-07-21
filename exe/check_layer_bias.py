"""단계별 중간값 몰림 정도 측정.

bias = F.normalize(x, dim=-1).mean(dim=0).norm()
  0 → 완전 isotropic, 1 → 모든 벡터가 동일 방향
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import torch
import torch.nn.functional as F
from anchorflow.physim import GNNSim, _fourier_pe
from anchorflow.graph import knn_graph


def bias(t: torch.Tensor) -> float:
    if t.shape[0] < 2:
        return float("nan")
    return F.normalize(t.float(), dim=-1).mean(dim=0).norm().item()


def run(graph_path, n_nodes, k_nn, hidden_dim, d_state, node_dim, dev):
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
        d_state=d_state, node_dim=node_dim,
        k_restore=0.0, gravity=0.0,
    ).to(dev).eval()

    src, dst = sim.edge_index
    static   = sim._static

    with torch.no_grad():
        x = sim.canonical.clone()
        v = torch.zeros_like(x)

        # ── 1. state_mlp ──────────────────────────────────────── #
        state = sim.state_mlp(
            torch.cat([_fourier_pe(x), _fourier_pe(v)], dim=-1))   # [M, H]

        # ── 2. edge_mlp ───────────────────────────────────────── #
        feat = torch.cat([state[src], state[dst],
                          static[src], static[dst]], dim=-1)
        msg  = sim.edge_mlp(feat)                                    # [E, H]

        # ── 3. agg (mean pool) ───────────────────────────────── #
        agg = torch.zeros(sim.M, sim.hidden_dim, device=dev)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        deg = torch.zeros(sim.M, device=dev).scatter_add_(
            0, dst, torch.ones(dst.shape[0], device=dev))
        agg = agg / deg.unsqueeze(1).clamp(min=1)                   # [M, H]

        # ── 4. enc_mlp → node_enc ───────────────────────────── #
        node_enc = sim.enc_mlp(torch.cat([agg, state], dim=-1))     # [M, H]

        # ── 5. SSM (h=zeros) ────────────────────────────────── #
        h = sim.ssm.init_state(sim.M, dev, x.dtype)
        y_zero, _ = sim.ssm(node_enc, h)                            # [M, H]

        # ── 6. SSM (h=random, run 1 step with random node_enc) ─ #
        h_rand = torch.randn_like(h)
        y_rand, _ = sim.ssm(node_enc, h_rand)                       # [M, H]

        # ── 7. dec_mlp ──────────────────────────────────────── #
        a_zero = torch.tanh(sim.dec_mlp(y_zero))                    # [M, 3]
        a_rand = torch.tanh(sim.dec_mlp(y_rand))                    # [M, 3]

    header = f"{'layer':<28} {'shape':<18} {'bias':>6}"
    print(header)
    print("-" * len(header))
    rows = [
        ("x (position)",         x),
        ("static (node_emb)",    static),
        ("state  = state_mlp",   state),
        ("msg    = edge_mlp",    msg),
        ("agg    = mean_pool",   agg),
        ("node_enc = enc_mlp",   node_enc),
        ("y  = SSM(h=zeros)",    y_zero),
        ("y  = SSM(h=random)",   y_rand),
        ("a_gnn (h=zeros)",      a_zero),
        ("a_gnn (h=random)",     a_rand),
    ]
    for name, t in rows:
        print(f"  {name:<26} {str(tuple(t.shape)):<18} {bias(t):>6.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",      default=None)
    ap.add_argument("--n_nodes",    type=int, default=512)
    ap.add_argument("--k_nn",       type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--d_state",    type=int, default=16)
    ap.add_argument("--node_dim",   type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(a.graph, a.n_nodes, a.k_nn, a.hidden_dim, a.d_state, a.node_dim, dev)


if __name__ == "__main__":
    main()
