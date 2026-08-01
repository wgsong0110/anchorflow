"""Demo: does the anchor graph "split apart" under a random-init HopDynamics?

Loads a REAL anchor set (canonical positions + k-NN graph topology) from an
existing trained checkpoint, but instantiates a FRESH, randomly-initialized
HopDynamics (no weights loaded) -- the point is to show the no-splitting
guarantee holds structurally, regardless of what the network has learned,
not just for a converged model.

Rolls the same random model out twice from the same initial state:
  (a) unconstrained  -- raw p = p + d_p every hop
  (b) xpbd-projected -- p = xpbd_distance_project(p + d_p, edge_index, rest_len)
      after every hop (see ssm_dynamics.xpbd_distance_project)

and renders both as side-by-side 3D wireframe (canonical edges) + scatter
animations so the difference is visible frame-by-frame.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch

from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import (
    HopDynamics, build_graph, rest_edge_lengths, xpbd_distance_project,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, required=True,
                 help="any existing HopDynamics ckpt_last.pt -- only used for its "
                      "real anchors/graph topology + hyperparameters, its trained "
                      "weights are NOT loaded")
ap.add_argument("--hops", type=int, default=60)
ap.add_argument("--dt_base", type=float, default=0.033)
ap.add_argument("--tolerance", type=float, default=0.15,
                 help="XPBD: allowed edge stretch/compression vs canonical length")
ap.add_argument("--xpbd_iters", type=int, default=4)
ap.add_argument("--decoder_init_std", type=float, default=0.3,
                 help="HopDynamics zero-inits decoder/rot_decoder's last layer "
                      "for training stability -- an untrained model with that "
                      "init produces exactly zero motion forever, which is a "
                      "trivial (uninteresting) demo. Re-randomize it so the "
                      "random model actually moves.")
ap.add_argument("--max_step_frac", type=float, default=0.4,
                 help="clamp each hop's raw |d_p| to at most this fraction of the "
                      "median canonical edge length, applied identically to BOTH "
                      "branches before accumulation -- an unbounded random decoder "
                      "can output an astronomically large one-hop delta (no tanh/ "
                      "accel_scale clamp like SSMDynamics has), which would blow "
                      "the unconstrained branch to float overflow in a couple hops "
                      "and swamp the comparison. This only bounds the PROPOSED "
                      "step magnitude, not inter-anchor distance -- the no-splitting "
                      "property still has to come from XPBD, not from this.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=str, default="./demo_out")
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

graph_cfg = {"graph": "knn", "k": hargs["k_graph"]}
edge_index = build_graph(canonical, graph_cfg)
rest_len = rest_edge_lengths(canonical, edge_index)
max_step = args.max_step_frac * float(rest_len.median())
print(f"[demo] M={M} edges={edge_index.shape[1]} k={hargs['k_graph']} max_step={max_step:.4f}")

model = HopDynamics(hidden=hargs["hidden"], mp_steps=hargs["mp_steps"], ssm_dim=hargs["ssm_dim"],
                     e_dim=hargs["e_dim"], n_time_freqs=hargs["n_time_freqs"]).to(dev)
# random init on purpose (no load_state_dict) -- but undo the class's own
# zero-init of decoder/rot_decoder's last layer, see --decoder_init_std above.
for head in (model.decoder, model.rot_decoder):
    last = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.normal_(last.weight, std=args.decoder_init_std)
    torch.nn.init.normal_(last.bias, std=args.decoder_init_std)
model.eval()

h0 = 0.3 * torch.randn(M, hargs["ssm_dim"], device=dev)
n_hops = args.hops

# hop() takes no positional input at all (spatial is precomputed once from
# canonical, fixed for the whole rollout) -- so the sequence of raw d_p/h
# updates the random model produces is IDENTICAL regardless of which branch
# consumes it. Compute it once, then apply to the two position sequences
# separately (raw accumulation vs XPBD-projected) so both branches see
# exactly the same underlying network output and only differ in what
# happens to the position update.
with torch.no_grad():
    spatial = model.spatial_embed(e, edge_index, canonical)
    h = h0.clone()
    d_p_seq = []
    for i in range(n_hops):
        d_p, _, h = model.hop(spatial, h, e, args.dt_base)
        step_norm = d_p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_p = d_p / step_norm * step_norm.clamp(max=max_step)
        d_p_seq.append(d_p)

    p_raw, p_xpbd = canonical.clone(), canonical.clone()
    seq_raw, seq_xpbd = [p_raw.clone()], [p_xpbd.clone()]
    for d_p in d_p_seq:
        p_raw = p_raw + d_p
        p_xpbd = xpbd_distance_project(p_xpbd + d_p, edge_index, rest_len,
                                        tolerance=args.tolerance, iters=args.xpbd_iters)
        seq_raw.append(p_raw.clone())
        seq_xpbd.append(p_xpbd.clone())

seq_raw = torch.stack(seq_raw).cpu().numpy()      # [T+1,M,3]
seq_xpbd = torch.stack(seq_xpbd).cpu().numpy()

# max edge-stretch ratio per frame, as a quantitative "did it split" signal
src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
rest_np = rest_len.cpu().numpy()


def stretch_ratio(seq):
    d = np.linalg.norm(seq[:, dst] - seq[:, src], axis=-1)   # [T+1,E]
    return (d / rest_np[None, :]).max(axis=-1)                # [T+1]


ratio_raw, ratio_xpbd = stretch_ratio(seq_raw), stretch_ratio(seq_xpbd)
print(f"[demo] max edge-stretch ratio over rollout -- unconstrained: {ratio_raw.max():.2f}x  "
      f"xpbd: {ratio_xpbd.max():.2f}x (tolerance={1 + args.tolerance:.2f}x)")

os.makedirs(args.out, exist_ok=True)

edges_arr = np.stack([src, dst], axis=1)


def render(seq, ratio, title, path):
    # per-branch bounding box (NaN-safe) -- the unconstrained branch is
    # EXPECTED to grow much larger than the xpbd one over the rollout, so
    # sharing one global box would either blow up the projected branch's
    # view to a speck or (if any NaN/inf slipped through) crash both.
    finite = np.isfinite(seq)
    seq_f = np.where(finite, seq, 0.0)
    lo, hi = seq_f.min(axis=(0, 1)), seq_f.max(axis=(0, 1))
    center, half = (lo + hi) / 2, max((hi - lo).max() / 2 * 1.1, 1e-3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=15)
    with writer.saving(fig, path, dpi=120):
        for t in range(seq.shape[0]):
            ax.cla()
            pts = np.nan_to_num(seq[t], nan=0.0, posinf=0.0, neginf=0.0)
            segs = pts[edges_arr]                                # [E,2,3]
            ax.add_collection3d(Line3DCollection(segs, colors="steelblue", linewidths=0.5, alpha=0.5))
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c="firebrick")
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.set_title(f"{title}\nhop {t}  max stretch {ratio[t]:.2f}x")
            ax.set_axis_off()
            writer.grab_frame()
    plt.close(fig)
    print(f"[demo] wrote {path}")


render(seq_raw, ratio_raw, "unconstrained (raw d_p)", os.path.join(args.out, "rollout_unconstrained.mp4"))
render(seq_xpbd, ratio_xpbd, "XPBD distance-projected", os.path.join(args.out, "rollout_xpbd.mp4"))
