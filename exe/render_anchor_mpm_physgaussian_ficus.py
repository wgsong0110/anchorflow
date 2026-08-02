"""Render a real rollout of the grid-free anchor-elastodynamics module
(lib/anchorflow/anchor_mpm.py) on PhysGaussian's OWN static ficus 3DGS
reconstruction (NOT SC-GS's ficus_ds_wind dynamic reconstruction) -- the
same scene used in the DreamPhysics/PhysGaussian MPM comparison earlier this
session (config/physics/ficus_config.json: E=0.1, nu=0.4, jelly, density=200).

Uses SC-GS's own (already-working-on-this-instance) GaussianModel/render
pipeline just as a renderer -- no SC-GS deform network is involved at all
since this checkpoint has none; a synthetic orbit camera is built directly
(no dataset/cameras.json needed for a static point cloud with no source
photos here). This is the FULL end-to-end path: anchors -> physics -> this
module's OWN shape-matching G2P for the final Gaussian positions (unlike
render_anchor_mpm_ficus.py, which offloaded G2P to SC-GS's pretrained
skinning) -- a genuine test of the whole pipeline including the parts still
being debugged (branch-like, anisotropic local anchor neighborhoods).

Must be run with CWD = the SC-GS repo root.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True, help="path to point_cloud.ply")
ap.add_argument("--out", required=True)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--E", type=float, default=0.1)
ap.add_argument("--nu", type=float, default=0.4)
ap.add_argument("--gravity", type=float, default=-2.0, help="applied along scene's up axis")
ap.add_argument("--up_axis", type=int, default=1, choices=[0, 1, 2], help="0=x,1=y,2=z (Blender/NeRF-synthetic default up=y)")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--dt", type=float, default=1e-4)
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--azimuth", type=float, default=-36.7)
ap.add_argument("--elevation", type=float, default=8.96)
ap.add_argument("--radius_scale", type=float, default=2.2, help="camera distance = radius_scale * scene extent")
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--fov_x", type=float, default=0.6911112070083618)
ap.add_argument("--impulse", type=float, nargs=3, default=None,
                 help="initial velocity kick applied to all anchors at t=0 (matches "
                      "ficus_config.json's particle_impulse -- gravity alone just sags "
                      "into a new rest shape, a push makes branches visibly swing)")
ap.add_argument("--pot_height_frac", type=float, default=0.18,
                 help="anchors within this fraction of the bbox range from the bottom "
                      "(along up_axis) are treated as the pot/base and held fixed -- "
                      "matches PhysGaussian's own boundary_conditions cuboid pin region "
                      "(ficus_config.json), which this script otherwise omits, letting "
                      "gravity free-fall/drift the WHOLE object (pot included) instead of "
                      "just the branches swaying relative to a planted base.")
ap.add_argument("--show_anchors", action="store_true",
                 help="overlay anchor positions (projected to screen space) as dots on "
                      "each frame, red=fixed(pot) / cyan=free -- to directly see anchor "
                      "motion, not just the skinned Gaussian render.")
ap.add_argument("--density", type=float, default=200.0,
                 help="material density (ficus_config.json value). With volumes from "
                      "voxel occupancy this gives PHYSICAL anchor masses -- the earlier "
                      "uniform mass=1.0 per anchor was ~10x too heavy for the DreamPhysics "
                      "stiffness scale, so elastic forces couldn't overcome inertia and the "
                      "motion was monotone creep (no oscillation) instead of a natural "
                      "swing-and-settle.")
ap.add_argument("--damping", type=float, default=0.9999,
                 help="per-substep velocity retention. The earlier 0.999 at dt=5e-5 "
                      "compounds to ~0.05x over 3000 steps -- overdamped, killing any "
                      "oscillation that would read as natural elastic motion.")
ap.add_argument("--voxel_grid", type=int, default=64,
                 help="voxel grid resolution for particle volume estimation "
                      "(DreamPhysics get_particle_volume convention: V_i = voxel_vol / "
                      "count_in_voxel).")
args = ap.parse_args()

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

dev = "cuda"
torch.set_grad_enabled(False)

gaussians = GaussianModel(args.sh_degree, fea_dim=0)
gaussians.load_ply(args.ply)
gaussian_canonical = gaussians.get_xyz.detach().clone()
N = gaussian_canonical.shape[0]
print(f"[render] loaded {N} Gaussians from {args.ply}")

bbox_min = gaussian_canonical.min(0).values
bbox_max = gaussian_canonical.max(0).values
center = ((bbox_min + bbox_max) / 2).cpu().numpy()
extent = float((bbox_max - bbox_min).norm().item())
radius = args.radius_scale * extent
print(f"[render] bbox center={center} extent={extent:.4f} camera_radius={radius:.4f}")


class _Pipe:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = _Pipe()
background = torch.tensor([1.0, 1.0, 1.0], device=dev)

# --- synthetic orbit camera (OpenCV/COLMAP convention: Y-down, Z-forward) ---
az = math.radians(args.azimuth)
el = math.radians(args.elevation)
eye = center + radius * np.array([
    math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az),
])
forward = center - eye
forward = forward / np.linalg.norm(forward)
world_up = np.array([0.0, 1.0, 0.0]) if args.up_axis == 1 else (
    np.array([0.0, 0.0, 1.0]) if args.up_axis == 2 else np.array([1.0, 0.0, 0.0]))
right = np.cross(forward, world_up)
right = right / (np.linalg.norm(right) + 1e-9)
true_up = np.cross(right, forward)
R = np.stack([right, -true_up, forward], axis=1)   # camera-to-world rotation (OpenCV: +Y down, +Z forward)
T = -R.T @ eye

fovx = args.fov_x
fovy = focal2fov(args.width / (2 * math.tan(fovx / 2)), args.height)
world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().to(dev)
projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev)
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, world_view_transform, full_proj_transform)

# --- anchors + physics ---
anchor_set, _ = AnchorSet.from_gaussians(gaussian_canonical, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
anchor_canonical = anchor_set.canonical.clone()
M = anchor_canonical.shape[0]
print(f"[render] {M} anchors (FPS)")

sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=args.K)
mu, lam = lame_from_E_nu(torch.tensor(args.E, device=dev), torch.tensor(args.nu, device=dev))

# per-particle volume via voxel occupancy (DreamPhysics get_particle_volume
# convention: V_i = voxel_volume / n_particles_in_that_voxel), then physical
# masses m_i = density * V_i, P2G-scattered onto anchors -- replaces the
# arbitrary uniform anchor mass that made the dynamics creep unnaturally.
G = args.voxel_grid
span = (bbox_max - bbox_min).max() + 1e-6
voxel_dx = span / G
vidx = ((gaussian_canonical - bbox_min) / voxel_dx).long().clamp(0, G - 1)
flat = (vidx[:, 0] * G + vidx[:, 1]) * G + vidx[:, 2]
counts = torch.zeros(G * G * G, device=dev).index_add_(0, flat, torch.ones(N, device=dev))
gaussian_volume = (voxel_dx ** 3) / counts[flat]
particle_mass = args.density * gaussian_volume
w0 = sim._weights(gaussian_canonical, anchor_canonical)
anchor_mass = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (particle_mass.unsqueeze(-1) * w0).reshape(-1))
anchor_mass = anchor_mass.clamp(min=1e-8)
print(f"[render] total volume={gaussian_volume.sum().item():.4f} total mass={particle_mass.sum().item():.4f} "
      f"anchor mass: min={anchor_mass.min().item():.2e} mean={anchor_mass.mean().item():.2e} max={anchor_mass.max().item():.2e}")

gravity = torch.zeros(3, device=dev)
gravity[args.up_axis] = args.gravity

# pot/base pin region: bottom pot_height_frac of the bbox along up_axis (see
# --pot_height_frac help). Without this the pot has nothing holding it in
# place and free-falls/drifts along with the branches under gravity.
anchor_up = anchor_canonical[:, args.up_axis]
up_min, up_max = gaussian_canonical[:, args.up_axis].min(), gaussian_canonical[:, args.up_axis].max()
pot_threshold = up_min + args.pot_height_frac * (up_max - up_min)
fixed_mask = anchor_up < pot_threshold
print(f"[render] {fixed_mask.sum().item()}/{M} anchors pinned as pot/base "
      f"(up_axis={args.up_axis} < {pot_threshold.item():.4f})")

anchor_pos = anchor_canonical.clone()
anchor_vel = torch.zeros(M, 3, device=dev)
if args.impulse is not None:
    anchor_vel = anchor_vel + torch.tensor(args.impulse, device=dev)
    print(f"[render] initial impulse velocity={args.impulse}")
gaussian_pos_prev = gaussian_canonical.clone()

d_rotation = torch.zeros(N, 4, device=dev)
d_rotation[:, 0] = 1.0
d_scaling = torch.zeros(N, 3, device=dev)

def project_to_screen(pts, view_proj, width, height):
    """pts [M,3] world -> screen-space pixel coords [M,2] (row-vector
    convention matching this codebase's full_proj_transform, i.e. clip =
    [x,y,z,1] @ view_proj)."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)
    clip = pts_h @ view_proj
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
    x = (ndc[:, 0] * 0.5 + 0.5) * width
    y = (ndc[:, 1] * 0.5 + 0.5) * height
    return torch.stack([x, y], dim=-1)


def draw_dots(image_uint8, xy, color, radius=4):
    h, w = image_uint8.shape[:2]
    xy_np = xy.detach().cpu().numpy()
    for x, y in xy_np:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
            x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
            image_uint8[y0:y1, x0:x1] = color
    return image_uint8


frames = []
nan_hit = False
with torch.enable_grad():
    for step_i in range(1, args.steps + 1):
        anchor_pos, anchor_vel, gaussian_pos_prev, F = sim.step(
            anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev,
            gaussian_volume, mu, lam, args.dt, gravity=gravity, damping=args.damping,
            fixed_mask=fixed_mask)
        if torch.isnan(anchor_pos).any():
            print(f"[render] NaN at step {step_i}, stopping.")
            nan_hit = True
            break
        if step_i % args.save_every == 0 or step_i == 1:
            with torch.no_grad():
                d_xyz = gaussian_pos_prev - gaussian_canonical
                results = render(cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, d_rot_as_res=True)
                image = torch.clamp(results["render"], 0.0, 1.0)
                frame = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8").copy()
                if args.show_anchors:
                    xy = project_to_screen(anchor_pos, full_proj_transform, args.width, args.height)
                    frame = draw_dots(frame, xy[fixed_mask], color=[255, 0, 0], radius=5)
                    frame = draw_dots(frame, xy[~fixed_mask], color=[0, 200, 255], radius=3)
                frames.append(frame)
                max_disp = d_xyz.norm(dim=-1).max().item()
                pot_disp = (anchor_pos[fixed_mask] - anchor_canonical[fixed_mask]).norm(dim=-1).max().item() if fixed_mask.any() else 0.0
                print(f"[render] step {step_i}/{args.steps} max_gaussian_disp={max_disp:.4f} "
                      f"max_pot_anchor_disp={pot_disp:.6f}")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, frames, fps=20, quality=8)
print(f"[render] wrote {len(frames)} frames -> {args.out} (nan_hit={nan_hit})")
