"""Check GNN acceleration direction distribution for random latents.

Runs N random (x, v, h) inputs through the model and checks whether
the per-node acceleration directions are isotropically distributed
or systematically biased toward a particular direction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from anchorflow.physim import GNNSim
from anchorflow.graph import knn_graph


def build_sim(args, dev):
    M = args.n_nodes
    canonical = torch.randn(M, 3) * 0.5
    anchor_colors = torch.rand(M, 3)
    edge_index = knn_graph(canonical, k=args.k_nn)
    src, dst = edge_index
    rest_len = (canonical[src] - canonical[dst]).norm(dim=-1)
    sim = GNNSim(
        canonical=canonical,
        anchor_colors=anchor_colors,
        edge_index=edge_index,
        rest_len=rest_len,
        T=2,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        node_dim=args.node_dim,
        k_restore=0.0,
        gravity=0.0,
    ).to(dev)
    return sim


def collect_dirs(sim, n_samples, latent_dim, dev):
    """Returns (raw_accels [N*M, 3], unit_dirs [N*M, 3])."""
    M = sim.M
    static = sim._static
    src_t, dst_t = sim.edge_index
    raws, dirs = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            x = sim.canonical + torch.randn(M, 3, device=dev) * 0.1
            v = torch.randn(M, 3, device=dev) * 0.05
            h = torch.randn(1, latent_dim, device=dev)
            node_enc, _ = sim._encode(x, v, static, src_t, dst_t)
            a = sim._decode(node_enc, h)          # [M, 3]
            raws.append(a)
            dirs.append(a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return torch.cat(raws, 0), torch.cat(dirs, 0)


def print_stats(label, raw, dirs):
    mean_dir  = dirs.mean(0)
    bias_norm = mean_dir.norm().item()
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  samples         : {raw.shape[0]}")
    print(f"  mean direction  : [{mean_dir[0]:.4f}, {mean_dir[1]:.4f}, {mean_dir[2]:.4f}]")
    print(f"  bias (0=isotropic, 1=fully biased): {bias_norm:.4f}")
    print(f"  raw mean/axis   : [{raw[:,0].mean():.4f}, {raw[:,1].mean():.4f}, {raw[:,2].mean():.4f}]")
    print(f"  raw std/axis    : [{raw[:,0].std():.4f}, {raw[:,1].std():.4f}, {raw[:,2].std():.4f}]")
    print(f"  magnitude mean  : {raw.norm(dim=-1).mean():.4f}")


def plot_results(raws_dict, dirs_dict, out_path):
    labels = list(raws_dict.keys())
    n = len(labels)
    fig = plt.figure(figsize=(6 * n, 14))

    for col, label in enumerate(labels):
        raw  = raws_dict[label].cpu().numpy()
        dirs = dirs_dict[label].cpu().numpy()

        # ── Row 1: 3D unit-direction scatter ─────────────────────────────── #
        ax = fig.add_subplot(3, n, col + 1, projection="3d")
        idx = np.random.choice(len(dirs), min(500, len(dirs)), replace=False)
        d = dirs[idx]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=2, alpha=0.4, c=d[:, 2],
                   cmap="coolwarm")
        mean_d = dirs.mean(0)
        ax.quiver(0, 0, 0, mean_d[0], mean_d[1], mean_d[2],
                  color="red", linewidth=3, label=f"bias={np.linalg.norm(mean_d):.3f}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (gravity axis)")
        ax.set_title(f"{label}\nUnit direction sphere", fontsize=9)
        ax.legend(fontsize=8)

        # ── Row 2: per-axis histogram of raw accelerations ────────────────── #
        ax2 = fig.add_subplot(3, n, n + col + 1)
        for axis, color, name in zip([0, 1, 2], ["r", "g", "b"], ["X", "Y", "Z"]):
            ax2.hist(raw[:, axis], bins=60, alpha=0.5, color=color,
                     label=f"{name}  μ={raw[:,axis].mean():.3f}")
        ax2.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax2.set_title(f"{label}\nRaw accel distribution", fontsize=9)
        ax2.set_xlabel("acceleration"); ax2.set_ylabel("count")
        ax2.legend(fontsize=7)

        # ── Row 3: polar plot (XZ plane — gravity plane) ──────────────────── #
        ax3 = fig.add_subplot(3, n, 2 * n + col + 1, projection="polar")
        theta = np.arctan2(dirs[:, 0], dirs[:, 2])   # XZ plane angle
        ax3.hist(theta, bins=36, color="steelblue", alpha=0.7)
        ax3.set_title(f"{label}\nPolar (XZ plane, Z=gravity axis)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n[plot] saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default=None)
    ap.add_argument("--n_nodes",    type=int, default=512)
    ap.add_argument("--n_samples",  type=int, default=200)
    ap.add_argument("--k_nn",       type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--node_dim",   type=int, default=32)
    ap.add_argument("--out",        default="/workspace/accel_dist.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    raws_dict = {}
    dirs_dict = {}

    # ── Random init ──────────────────────────────────────────────────────── #
    sim_rand = build_sim(args, dev)
    sim_rand.eval()
    raw, dirs = collect_dirs(sim_rand, args.n_samples, args.latent_dim, dev)
    raws_dict["random init"] = raw
    dirs_dict["random init"] = dirs
    print_stats("random init", raw, dirs)

    # ── Trained checkpoint ────────────────────────────────────────────────── #
    if args.ckpt and os.path.exists(args.ckpt):
        sim_tr = build_sim(args, dev)
        ck = torch.load(args.ckpt, map_location=dev)
        state = ck.get("sim", ck)
        sim_tr.load_state_dict(state, strict=False)
        sim_tr.eval()
        raw_tr, dirs_tr = collect_dirs(sim_tr, args.n_samples, args.latent_dim, dev)
        raws_dict["trained"] = raw_tr
        dirs_dict["trained"] = dirs_tr
        print_stats("trained", raw_tr, dirs_tr)

    plot_results(raws_dict, dirs_dict, args.out)


if __name__ == "__main__":
    main()
