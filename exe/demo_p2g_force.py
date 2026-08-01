"""Demo: does an externally applied per-particle force actually reach the
anchor-level GNN's decision, via P2G aggregation?

Loads REAL anchor topology (from an existing HopDynamics checkpoint) and
REAL Gaussian (particle) positions (from the SC-GS point_cloud.ply that
scene's canonical anchors were derived from), picks one particle, applies a
synthetic force to it ONLY, aggregates via AnchorSet.scatter_particle_to_anchor
(P2G), and runs ONE hop of a randomly-initialized HopDynamics(stateless=True,
f_ext_dim=3) with vs without that force -- comparing the resulting d_p per
anchor. This is a plumbing/sanity check (random weights, no training signal
involving f_ext exists yet), not a physically meaningful result: the question
is only "does injecting a force at a particle measurably change the hop
output, more so near that particle's bound anchors than far away?"
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from plyfile import PlyData

from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import HopDynamics, build_graph

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, required=True,
                 help="any existing HopDynamics ckpt_last.pt -- used only for real anchor "
                      "topology/hyperparameters, weights are NOT loaded (fresh random init)")
ap.add_argument("--ply", type=str, required=True,
                 help="SC-GS point_cloud.ply for the same scene, for real particle positions")
ap.add_argument("--force_particle_idx", type=int, default=None,
                 help="which particle gets the synthetic force (default: random)")
ap.add_argument("--force_mag", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
hargs = ckpt["args"]
M = ckpt["anchors"]["canonical"].shape[0]
anchors = AnchorSet(torch.zeros(M, 3), latent_dim=hargs["latent_dim"], e_dim=hargs["e_dim"],
                     K=hargs["anchor_k"]).to(dev)
anchors.load_state_dict(ckpt["anchors"])
canonical, e = anchors.canonical.detach(), anchors.e.detach()
edge_index = build_graph(canonical, {"graph": "knn", "k": hargs["k_graph"]})
print(f"[demo] M={M} anchors, edges={edge_index.shape[1]}")

ply = PlyData.read(args.ply)
v = ply["vertex"]
gaussian_xyz = torch.tensor(
    list(zip(v["x"], v["y"], v["z"])), dtype=torch.float32, device=dev)
N = gaussian_xyz.shape[0]
print(f"[demo] N={N} particles loaded from {args.ply}")

force_idx = args.force_particle_idx if args.force_particle_idx is not None \
    else int(torch.randint(0, N, (1,)).item())
particle_force = torch.zeros(N, 3, device=dev)
force_dir = torch.randn(3, device=dev)
force_dir = force_dir / force_dir.norm()
particle_force[force_idx] = force_dir * args.force_mag
print(f"[demo] force applied to particle {force_idx} at {gaussian_xyz[force_idx].tolist()}, "
      f"direction={force_dir.tolist()}, magnitude={args.force_mag}")

anchor_force = anchors.scatter_particle_to_anchor(gaussian_xyz, particle_force)
nonzero_anchors = (anchor_force.norm(dim=-1) > 1e-6).nonzero().flatten().tolist()
print(f"[demo] P2G: force reached {len(nonzero_anchors)} anchor(s) (nonzero): {nonzero_anchors}")
for a in nonzero_anchors:
    dist_to_particle = (canonical[a] - gaussian_xyz[force_idx]).norm().item()
    print(f"  anchor {a}: |f_ext|={anchor_force[a].norm().item():.4f}  "
          f"dist to forced particle={dist_to_particle:.4f}")

model = HopDynamics(hidden=hargs["hidden"], mp_steps=hargs["mp_steps"],
                     e_dim=hargs["e_dim"], n_time_freqs=hargs["n_time_freqs"],
                     stateless=True, f_ext_dim=3).to(dev)
model.eval()
v0 = torch.zeros(M, 3, device=dev)
dt = 0.03

with torch.no_grad():
    d_p_with, _, _, _ = model.hop_pv(e, edge_index, canonical, v0, dt, f_ext=anchor_force)
    d_p_without, _, _, _ = model.hop_pv(e, edge_index, canonical, v0, dt,
                                         f_ext=torch.zeros_like(anchor_force))

diff = (d_p_with - d_p_without).norm(dim=-1)   # [M]
print(f"\n[demo] |d_p(with force) - d_p(without force)| per anchor -- "
      f"mean={diff.mean().item():.6f}  max={diff.max().item():.6f}")
order = diff.argsort(descending=True)[:10]
print("[demo] top-10 most-affected anchors (idx, |Δd_p|, canonical dist to forced particle):")
for a in order.tolist():
    dist_to_particle = (canonical[a] - gaussian_xyz[force_idx]).norm().item()
    print(f"  anchor {a}: Δd_p={diff[a].item():.6f}  dist={dist_to_particle:.4f}")

# sanity: correlation between "close to forced particle" and "affected more"
dists_all = (canonical - gaussian_xyz[force_idx]).norm(dim=-1)
corr = torch.corrcoef(torch.stack([dists_all, diff]))[0, 1].item()
print(f"\n[demo] correlation(distance to forced particle, |Δd_p|) = {corr:.3f} "
      f"(negative = closer anchors affected more, as expected if P2G+GNN propagation works)")
