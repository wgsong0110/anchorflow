#!/usr/bin/env python
"""
Extract a per-frame anchor (SC-GS control node) trajectory from a converged
SC-GS checkpoint, for HopDynamics anchor-trajectory distillation
(train_hop_autoreg_anchortraj.py).

Uses deform.deform.node_deform() directly (the raw per-node network call,
NOT deform.step() which internally LBS-blends over Gaussians and does not
expose local_rotation) to get, for every training frame time t:
  - d_xyz          -> anchor_pos[t] = canonical_nodes + d_xyz
  - d_rotation     -> anchor_rot[t]
  - local_rotation -> anchor_local_rot[t] (only meaningful if the checkpoint
                       was trained with --local_frame; harmless zeros otherwise)

Must be run with CWD = the SC-GS repo root.

Usage:
  cd /workspace/SC-GS && python /workspace/anchorflow/exe/extract_scgs_anchor_traj.py \\
      --model_path /workspace/scgs_hellwarrior_official_node \\
      --source_path /workspace/dnerf_hellwarrior \\
      --out /workspace/scgs_hellwarrior_anchor_traj.pt
"""
import sys, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--source_path", required=True)
ap.add_argument("--iteration", type=int, default=-1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, os.getcwd())
import torch
from scene import Scene, DeformModel
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

sys.argv = [sys.argv[0], "--model_path", args.model_path, "--source_path", args.source_path,
            "--deform_type", "node"]
parser = argparse.ArgumentParser()
model_p = ModelParams(parser, sentinel=True)
pipeline_p = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
cargs = get_combined_args(parser)
cargs.source_path = args.source_path
dataset = model_p.extract(cargs)

dev = "cuda"
torch.set_grad_enabled(False)

deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                      skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num,
                      pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash,
                      hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame,
                      progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
deform.load_weights(dataset.model_path, iteration=args.iteration)
deform.deform.eval()

gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == "node" else dataset.hyper_dim
gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

# readNerfSyntheticInfo appends test cams after train cams when eval=False --
# only the first `n_real_train` entries (the JSON's own frame count) are the
# real, temporally-ordered training frames we want here.
import json
with open(os.path.join(args.source_path, "transforms_train.json")) as f:
    n_real_train = len(json.load(f)["frames"])
views = scene.getTrainCameras()[:n_real_train]
print(f"[extract] local_frame={dataset.local_frame} d_rot_as_res={dataset.d_rot_as_res} node_num={dataset.node_num} n_frames={len(views)}")

nodes = deform.deform.nodes[..., :3].detach()
M = nodes.shape[0]
T = len(views)
anchor_pos = torch.zeros(T, M, 3)
anchor_rot = torch.zeros(T, M, 4)
anchor_local_rot = torch.zeros(T, M, 4)
times = []

for i, view in enumerate(views):
    fid = view.fid
    times.append(float(fid))
    time_input = deform.deform.expand_time(fid)
    raw = deform.deform.node_deform(t=time_input, detach_node=True)
    d_xyz, d_rotation, local_rotation = raw["d_xyz"], raw["d_rotation"], raw["local_rotation"]
    anchor_pos[i] = (nodes + d_xyz).cpu()
    anchor_rot[i] = d_rotation.cpu()
    anchor_local_rot[i] = local_rotation.cpu()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({
    "canonical_nodes": nodes.cpu(),
    "anchor_pos": anchor_pos,
    "anchor_rot": anchor_rot,
    "anchor_local_rot": anchor_local_rot,
    "times": times,
}, args.out)
print(f"[extract] saved T={T} M={M} -> {args.out}")
