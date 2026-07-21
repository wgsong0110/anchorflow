"""Check GNN acceleration direction distribution for random latents.

Runs N random (x, v) inputs through the model and checks whether
the per-node acceleration directions are isotropically distributed
or systematically biased toward a particular direction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import argparse
import torch
import torch.nn.functional as F

from anchorflow.physim import GNNSim
from anchorflow.graph import knn_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      default=None,  help="checkpoint path (optional)")
    ap.add_argument("--n_nodes",   type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=100, help="number of random latent samples")
    ap.add_argument("--k_nn",      type=int, default=16)
    ap.add_argument("--hidden_dim",type=int, default=256)
    ap.add_argument("--latent_dim",type=int, default=256)
    ap.add_argument("--node_dim",  type=int, default=32)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Random canonical positions (unit cube)
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

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        state = ck.get("sim", ck)
        sim.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("Using randomly initialized weights")

    sim.eval()
    static = sim._static
    src_t, dst_t = sim.edge_index

    all_dirs = []   # [N_samples * M, 3] normalized acceleration directions

    with torch.no_grad():
        for i in range(args.n_samples):
            # Random position perturbation + random velocity
            x = sim.canonical + torch.randn(M, 3, device=dev) * 0.1
            v = torch.randn(M, 3, device=dev) * 0.05
            h = torch.randn(1, args.latent_dim, device=dev)   # random SSM state

            node_enc, z = sim._encode(x, v, static, src_t, dst_t)
            # Use random h instead of SSM step to sample diverse latent contexts
            a_gnn = sim._decode(node_enc, h)                   # [M, 3]

            # Normalize to unit direction
            norms = a_gnn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dirs  = a_gnn / norms                               # [M, 3]
            all_dirs.append(dirs)

    all_dirs = torch.cat(all_dirs, dim=0)   # [N*M, 3]

    mean_dir = all_dirs.mean(dim=0)
    std_dir  = all_dirs.std(dim=0)
    mean_norm = mean_dir.norm()

    print(f"\n=== Acceleration Direction Distribution ===")
    print(f"Samples: {args.n_samples} × {M} nodes = {all_dirs.shape[0]} directions")
    print(f"Mean direction : {mean_dir.cpu().numpy()}  (norm={mean_norm:.4f})")
    print(f"Std  per axis  : {std_dir.cpu().numpy()}")
    print()
    print("  If isotropic (no bias): mean direction ≈ [0,0,0], norm ≈ 0")
    print("  If biased (e.g. -Z)   : mean direction ≈ [0,0,-1], norm ≈ 1")
    print()

    # Per-axis mean of raw (non-normalized) accelerations
    all_raw = []
    with torch.no_grad():
        for i in range(args.n_samples):
            x = sim.canonical + torch.randn(M, 3, device=dev) * 0.1
            v = torch.randn(M, 3, device=dev) * 0.05
            h = torch.randn(1, args.latent_dim, device=dev)
            node_enc, z = sim._encode(x, v, static, src_t, dst_t)
            a_gnn = sim._decode(node_enc, h)
            all_raw.append(a_gnn)

    all_raw = torch.cat(all_raw, dim=0)
    print(f"=== Raw Acceleration Stats (max_accel={sim.max_accel}) ===")
    print(f"Mean per axis  : {all_raw.mean(dim=0).cpu().numpy()}")
    print(f"Std  per axis  : {all_raw.std(dim=0).cpu().numpy()}")
    print(f"Mean magnitude : {all_raw.norm(dim=-1).mean().item():.4f}")


if __name__ == "__main__":
    main()
