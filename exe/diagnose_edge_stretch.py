"""One-off diagnostic: measure per-edge stretch ratio (current length / canonical
rest length) over a FULL dense (every-frame, T-1 consecutive hops) rollout of a
trained HopDynamics checkpoint -- the structural "did it tear apart" signal used
throughout the 2026-08-01 XPBD no-splitting experiments, applied here to the new
"SSM kept + GNN every hop" (--stateless_pv, use_ssm=True) checkpoint for a
same-methodology comparison against xpbd_none / no_ssm / stateless_pv(no-ssm).
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import (
    HopDynamics, build_graph, hop_rollout, hop_rollout_pv, rest_edge_lengths,
    xpbd_distance_project, xpbd_directional_project,
)

ap = argparse.ArgumentParser()
ap.add_argument("--hop_ckpt", required=True)
ap.add_argument("--traj", required=True)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.hop_ckpt, map_location=dev, weights_only=False)
hargs = ckpt["args"]
traj = torch.load(args.traj, map_location=dev, weights_only=False)
canonical = traj["canonical"].to(dev) if "canonical" in traj else ckpt["anchors"]["canonical"].to(dev)
T = traj["pos"].shape[0] if "pos" in traj else 200

anchors = AnchorSet(canonical, latent_dim=hargs["latent_dim"], e_dim=hargs["e_dim"],
                     K=hargs["anchor_k"]).to(dev)
anchors.load_state_dict(ckpt["anchors"])
edge_index = build_graph(anchors.canonical.detach(), {"graph": "knn", "k": hargs["k_graph"]})
xpbd_rest_len = rest_edge_lengths(anchors.canonical.detach(), edge_index)

stateless_pv = hargs.get("stateless_pv", False)
model = HopDynamics(hidden=hargs["hidden"], mp_steps=hargs["mp_steps"], ssm_dim=hargs["ssm_dim"],
                     e_dim=hargs["e_dim"], n_time_freqs=hargs["n_time_freqs"],
                     use_ssm=not hargs.get("no_ssm", False), stateless=stateless_pv,
                     gnn_layer=hargs.get("gnn_layer", "interaction")).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()

dt_base = 1.0 / (T - 1)
xpbd_mode = hargs.get("xpbd_mode", "distance") if hargs.get("xpbd") else None
if xpbd_mode == "none" or xpbd_mode is None:
    project_fn = None
elif xpbd_mode == "directional":
    project_fn = lambda p, d_p: xpbd_directional_project(
        p, edge_index, xpbd_rest_len, direction=d_p,
        tolerance=hargs["xpbd_tolerance"], iters=hargs["xpbd_iters"])
else:
    project_fn = lambda p, d_p: xpbd_distance_project(
        p, edge_index, xpbd_rest_len, tolerance=hargs["xpbd_tolerance"], iters=hargs["xpbd_iters"])

frame_indices = list(range(1, T))
init_h = ckpt["init_h"].to(dev)
with torch.no_grad():
    if stateless_pv:
        p_by_t, _, _ = hop_rollout_pv(model, anchors.canonical, anchors.e, edge_index,
                                       times=frame_indices, dt_base=dt_base, grad=False,
                                       h0=init_h if model.use_ssm else None,
                                       project_fn=project_fn)
    else:
        p_by_t, _, _ = hop_rollout(model, anchors.canonical, anchors.e, edge_index,
                                    times=frame_indices, dt_base=dt_base, grad=False,
                                    h0=init_h, project_fn=project_fn)

src, dst = edge_index
max_err = 0.0
sum_err = 0.0
n = 0
max_err_by_hop = []
for t in frame_indices:
    p = p_by_t[t]
    cur_len = (p[dst] - p[src]).norm(dim=-1)
    ratio = cur_len / xpbd_rest_len.clamp(min=1e-8)
    err = (ratio - 1.0).abs()
    max_err_by_hop.append(err.max().item())
    max_err = max(max_err, err.max().item())
    sum_err += err.sum().item()
    n += err.numel()

print(f"[diagnose] ckpt={args.hop_ckpt}")
print(f"[diagnose] stateless_pv={stateless_pv} use_ssm={model.use_ssm} xpbd_mode={xpbd_mode} "
      f"gnn_layer={hargs.get('gnn_layer','interaction')}")
print(f"[diagnose] full dense rollout ({len(frame_indices)} consecutive hops): "
      f"max_stretch_err={max_err*100:.1f}%  mean_stretch_err={(sum_err/n)*100:.1f}%")
print(f"[diagnose] max_err at hop 50/100/150/199: "
      f"{max_err_by_hop[min(49,len(max_err_by_hop)-1)]*100:.1f}% / "
      f"{max_err_by_hop[min(99,len(max_err_by_hop)-1)]*100:.1f}% / "
      f"{max_err_by_hop[min(149,len(max_err_by_hop)-1)]*100:.1f}% / "
      f"{max_err_by_hop[-1]*100:.1f}%")
