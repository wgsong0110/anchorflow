"""Render a real ficus rollout using the grid-free anchor-elastodynamics
module (lib/anchorflow/anchor_mpm.py) instead of a learned GNN, reusing
SC-GS's OWN trained skinning weights for the actual Gaussian-level motion --
same pattern as exe/eval_hop_autoreg_psnr.py's HopNodeDeform monkeypatch,
but the "dynamics" driving the anchor nodes is physics (Fixed Corotated
elasticity via anchor_mpm), not a learned network.

anchor_mpm's own internal shape-matching (used to compute the elastic
energy/force on anchors) is a SEPARATE computation from the actual
render-time Gaussian skinning here -- the latter uses SC-GS's own already-
fitted cal_nn_weight (proven, used to reach the checkpoint's real
photometric quality), which is more reliable for a first "does this look
reasonable" look than trusting anchor_mpm's own (still being debugged)
shape-matching G2P for the final visual result too.

Must be run with CWD = the SC-GS repo root, e.g.:
  cd /workspace/SC-GS && python /workspace/anchorflow/exe/render_anchor_mpm_ficus.py \\
      --model_path /workspace/ficus_scgs_baseline_node --iteration 80000 \\
      --out /workspace/anchor_mpm_ficus.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
import numpy as np

from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--iteration", type=int, default=-1)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=2000, help="physics substeps")
ap.add_argument("--dt", type=float, default=2e-4)
ap.add_argument("--save_every", type=int, default=40, help="render every N substeps")
ap.add_argument("--E", type=float, default=0.1, help="ficus_config.json fitted value")
ap.add_argument("--nu", type=float, default=0.4)
ap.add_argument("--gravity", type=float, default=-1.0)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--fixed_view_idx", type=int, default=0)
args = ap.parse_args()

sys.argv = [sys.argv[0], "--model_path", args.model_path, "--source_path", "/workspace/ficus_ds_wind",
            "--deform_type", "node"]
import argparse as _argparse
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, DeformModel
from gaussian_renderer import GaussianModel, render

parser = _argparse.ArgumentParser()
model_p = ModelParams(parser, sentinel=True)
pipeline_p = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
cargs = get_combined_args(parser)
dataset = model_p.extract(cargs)
pipeline = pipeline_p.extract(cargs)

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
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device=dev)

print(f"[render] local_frame={dataset.local_frame} d_rot_as_res={dataset.d_rot_as_res} "
      f"node_num={dataset.node_num} gaussians={gaussians.get_xyz.shape[0]}")

anchor_canonical = deform.deform.nodes[..., :3].detach().clone()
gaussian_canonical = gaussians.get_xyz.detach().clone()
M = anchor_canonical.shape[0]

sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=args.K)
mu, lam = lame_from_E_nu(torch.tensor(args.E, device=dev), torch.tensor(args.nu, device=dev))
gaussian_volume = torch.full((gaussian_canonical.shape[0],), 1.0 / gaussian_canonical.shape[0], device=dev)
gravity = torch.tensor([0.0, 0.0, args.gravity], device=dev)

anchor_pos = anchor_canonical.clone()
anchor_vel = torch.zeros(M, 3, device=dev)
anchor_mass = torch.full((M,), 1.0, device=dev)
gaussian_pos_prev = gaussian_canonical.clone()

rot_bias = torch.zeros(M, 4, device=dev)
rot_bias[:, 0] = 1.0  # identity quaternion (w,x,y,z)

class PhysicsNodeDeform:
    def __init__(self, d_xyz):
        self.d_xyz = d_xyz

    def __call__(self, t, **kwargs):
        return {
            "d_xyz": self.d_xyz,
            "d_rotation": rot_bias,
            "d_scaling": torch.zeros(M, 3, device=dev),
            "d_opacity": None,
            "d_color": None,
        }

fixed_view = scene.getTrainCameras()[args.fixed_view_idx]
if dataset.load2gpu_on_the_fly:
    fixed_view.load2device()

with torch.enable_grad():
    frames = []
    nan_hit = False
    for step_i in range(1, args.steps + 1):
        anchor_pos, anchor_vel, gaussian_pos_prev, F = sim.step(
            anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev,
            gaussian_volume, mu, lam, args.dt, gravity=gravity, damping=0.999)
        if torch.isnan(anchor_pos).any():
            print(f"[render] NaN at step {step_i}, stopping.")
            nan_hit = True
            break
        if step_i % args.save_every == 0 or step_i == 1:
            with torch.no_grad():
                d_xyz_node = anchor_pos - anchor_canonical
                deform.deform.node_deform = PhysicsNodeDeform(d_xyz_node)
                time_input = deform.deform.expand_time(fixed_view.fid if hasattr(fixed_view, "fid") else torch.tensor(0.0, device=dev))
                d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.feature,
                                        motion_mask=gaussians.motion_mask, is_training=False)
                d_xyz, d_rotation, d_scaling = d_values["d_xyz"], d_values["d_rotation"], d_values["d_scaling"]
                results = render(fixed_view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling,
                                  d_opacity=d_values["d_opacity"], d_color=d_values["d_color"],
                                  d_rot_as_res=deform.d_rot_as_res)
                image = torch.clamp(results["render"], 0.0, 1.0)
                frames.append((image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
                max_disp = d_xyz_node.norm(dim=-1).max().item()
                print(f"[render] step {step_i}/{args.steps} max_anchor_disp={max_disp:.4f}")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, frames, fps=20, quality=8)
print(f"[render] wrote {len(frames)} frames -> {args.out} (nan_hit={nan_hit})")
