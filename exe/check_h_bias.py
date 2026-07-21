"""Compare acceleration distribution with vs without h in decoder.

Random init GNN:
  Case A: h = random  (current architecture)
  Case B: h = zeros   (h disabled, only node_enc drives output)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anchorflow.physim import GNNSim
from anchorflow.graph import knn_graph


def build_sim(n_nodes, k_nn, hidden_dim, latent_dim, node_dim, dev):
    canonical = torch.randn(n_nodes, 3) * 0.5
    edge_index = knn_graph(canonical, k=k_nn)
    src, dst = edge_index
    rest_len = (canonical[src] - canonical[dst]).norm(dim=-1)
    sim = GNNSim(
        canonical=canonical,
        anchor_colors=torch.rand(n_nodes, 3),
        edge_index=edge_index,
        rest_len=rest_len,
        T=2, hidden_dim=hidden_dim,
        latent_dim=latent_dim, node_dim=node_dim,
        k_restore=0.0, gravity=0.0,
    ).to(dev)
    return sim


def collect(sim, n_samples, latent_dim, dev, use_h: bool):
    M = sim.M
    static = sim._static
    src, dst = sim.edge_index
    raws = []
    with torch.no_grad():
        for _ in range(n_samples):
            x = sim.canonical + torch.randn(M, 3, device=dev) * 0.1
            v = torch.randn(M, 3, device=dev) * 0.05
            node_enc, _ = sim._encode(x, v, static, src, dst)
            if use_h:
                h = torch.randn(1, latent_dim, device=dev)
            else:
                h = torch.zeros(1, latent_dim, device=dev)
            a = sim._decode(node_enc, h)
            raws.append(a)
    return torch.cat(raws, 0)   # [N*M, 3]


def stats(label, raw):
    dirs = F.normalize(raw, dim=-1)
    mean_dir = dirs.mean(0)
    bias = mean_dir.norm().item()
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  bias (0=isotropic): {bias:.4f}")
    print(f"  mean dir : [{mean_dir[0]:.4f}, {mean_dir[1]:.4f}, {mean_dir[2]:.4f}]")
    print(f"  raw mean : [{raw[:,0].mean():.4f}, {raw[:,1].mean():.4f}, {raw[:,2].mean():.4f}]")
    print(f"  raw std  : [{raw[:,0].std():.4f},  {raw[:,1].std():.4f},  {raw[:,2].std():.4f}]")
    print(f"  magnitude: {raw.norm(dim=-1).mean():.4f}")
    return dirs.cpu().numpy(), bias


def plot(cases, out_path):
    n = len(cases)
    fig = plt.figure(figsize=(6 * n, 12))
    for col, (label, raw, dirs, bias) in enumerate(cases):
        # ── 3D sphere ──────────────────────────────────────────────────────── #
        ax = fig.add_subplot(2, n, col + 1, projection="3d")
        idx = np.random.choice(len(dirs), min(600, len(dirs)), replace=False)
        d = dirs[idx]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=2, alpha=0.4,
                   c=d[:, 2], cmap="coolwarm")
        md = dirs.mean(0)
        ax.quiver(0, 0, 0, md[0], md[1], md[2], color="red",
                  linewidth=3, label=f"bias={bias:.3f}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{label}", fontsize=10)
        ax.legend(fontsize=9)

        # ── per-axis histogram ─────────────────────────────────────────────── #
        ax2 = fig.add_subplot(2, n, n + col + 1)
        raw_np = raw.cpu().numpy()
        for axis, color, name in zip([0,1,2], ["r","g","b"], ["X","Y","Z"]):
            ax2.hist(raw_np[:, axis], bins=60, alpha=0.5, color=color,
                     label=f"{name} μ={raw_np[:,axis].mean():.3f}")
        ax2.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax2.set_title(f"{label}\nRaw accel histogram", fontsize=9)
        ax2.set_xlabel("acceleration"); ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n[plot] {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",      default=None)
    ap.add_argument("--n_nodes",    type=int, default=512)
    ap.add_argument("--n_samples",  type=int, default=200)
    ap.add_argument("--k_nn",       type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--node_dim",   type=int, default=32)
    ap.add_argument("--out",        default="/workspace/h_bias.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Use real graph if provided
    if args.graph and os.path.exists(args.graph):
        gd = torch.load(args.graph, map_location="cpu", weights_only=False)
        edge_index = gd["edge_index"]
        rest_len   = gd["rest_len"]
        M = args.n_nodes
        canonical  = torch.randn(M, 3) * 0.5
        sim = GNNSim(
            canonical=canonical,
            anchor_colors=torch.rand(M, 3),
            edge_index=edge_index, rest_len=rest_len,
            T=2, hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim, node_dim=args.node_dim,
            k_restore=0.0, gravity=0.0,
        ).to(dev)
    else:
        sim = build_sim(args.n_nodes, args.k_nn,
                        args.hidden_dim, args.latent_dim, args.node_dim, dev)
    sim.eval()

    cases = []
    for use_h, label in [(True, "with h (random)"), (False, "h = zeros")]:
        raw = collect(sim, args.n_samples, args.latent_dim, dev, use_h)
        dirs, bias = stats(label, raw)
        cases.append((label, raw, dirs, bias))

    plot(cases, args.out)


if __name__ == "__main__":
    main()
