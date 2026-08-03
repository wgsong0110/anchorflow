"""Faithful PhysGaussian/DreamPhysics reproduction with the ONLY intended
difference being grid -> anchors.

Everything else follows config/physics/ficus_config.json and DreamPhysics's
own code paths exactly, rather than the ad-hoc substitutions an earlier
version of this experiment used:

  * world -> MPM space transform (utils/transformation_utils.py):
      opacity filter (opacity_threshold) -> rotations (rotation_degree/axis)
      -> transform2origin ((p-mean)/max_extent) -> shift2center111 (+[1,1,1])
    Previously skipped entirely, which made the config's MPM-space numbers
    (boundary cuboid, viewpoint center) unusable and forced a made-up
    "bottom 18% of the bbox" pin heuristic.
  * boundary condition: the config's actual cuboid (point +/- size, i.e.
    set_velocity_on_cuboid with velocity=[0,0,0] => Dirichlet-fixed), applied
    to ANCHORS -- the grid-node BC's direct analogue, which is the whole
    point of the substitution.
  * loading: the config's particle_impulse, applied EXACTLY as DreamPhysics'
    apply_force kernel does -- a uniform FORCE (not acceleration, and not a
    height-weighted wind): v_p += (force / m_p) * dt for num_dt substeps.
    Since m_p * dv_p = force * dt is then the same for every particle, the
    momentum handed to each anchor is just (sum of its P2G weights) * force *
    dt, so this transfers to the anchor state with no extra assumption.
  * per-region material: additional_material_params (the softer leaf/canopy
    region) instead of one global E/nu/density.
  * particle volume: DreamPhysics get_particle_volume convention
    (voxel_volume / count_in_voxel) at the config's n_grid/grid_lim, computed
    in MPM space, mass = density * volume.
  * timing: substep_dt, frame_dt, frame_num from the config; g from the
    config (ficus: zero gravity).

Rendering transforms positions back to world space (undo_all_transforms)
before rasterizing, same as ms_simulation.py.

Must be run with CWD = the SC-GS repo root (for the renderer imports).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True, help="DreamPhysics physics config json")
ap.add_argument("--out", required=True)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--grid_lim", type=float, default=2.0,
                 help="MPM domain is [0,grid_lim]^3; shift2center111 puts the object at "
                      "center (1,1,1), so the domain must cover it (solver default n_grid=100).")
ap.add_argument("--frame_num", type=int, default=None, help="override config frame_num")
ap.add_argument("--show_anchors", action="store_true")
ap.add_argument("--radius_scale", type=float, default=1.5)
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--fov_x", type=float, default=0.6911112070083618)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))
print(f"[cfg] {json.dumps({k: v for k, v in cfg.items() if not isinstance(v, list)}, indent=None)}")

gaussians = GaussianModel(args.sh_degree, fea_dim=0)
gaussians.load_ply(args.ply)
xyz_world_all = gaussians.get_xyz.detach().clone()
opacity_all = gaussians.get_opacity.detach().clone()

# ---- 1. opacity filter (DreamPhysics does this before anything else) ----
keep = (opacity_all[:, 0] > cfg["opacity_threshold"])
kept_idx = keep.nonzero().flatten()
xyz_world = xyz_world_all[keep]
N = xyz_world.shape[0]
print(f"[prep] opacity>{cfg['opacity_threshold']}: kept {N}/{xyz_world_all.shape[0]} Gaussians")
# DreamPhysics masks the dropped kernels out of position, covariance, opacity
# AND SH (ms_simulation.py:142) -- they leave the render, not just the sim. On
# ficus the dropped 16% are interleaved through the foliage (median distance to
# a simulated Gaussian: 0.002 in a scene spanning 1.0), so leaving them in the
# render pins them at their canonical position and smears an afterimage over
# the moving plant. Drive their opacity logit to zero to match.
gaussians._opacity.data[~keep] = -20.0

# ---- 2. rotations (ficus: [0.0] deg about axis 0 -> identity, kept for fidelity) ----
def rot_mat(deg, axis):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == 0:
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1:
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return torch.tensor(m, dtype=torch.float32, device=dev)

rot_mats = [rot_mat(d, a) for d, a in zip(cfg["rotation_degree"], cfg["rotation_axis"])]
pos = xyz_world
for R_ in rot_mats:
    pos = pos @ R_.T

# ---- 3. transform2origin + shift2center111 ----
pmin, pmax = pos.min(0).values, pos.max(0).values
orig_mean = (pmin + pmax) / 2.0
scale_origin = 1.0 / (pmax - pmin).max()
pos_mpm = (pos - orig_mean) * scale_origin + torch.tensor([1.0, 1.0, 1.0], device=dev)
print(f"[prep] MPM space: scale={scale_origin.item():.4f} bbox={pos_mpm.min(0).values.tolist()} .. {pos_mpm.max(0).values.tolist()}")


def undo_all(p_mpm):
    """MPM space -> world (inverse of the above, matching undo_all_transforms)."""
    p = (p_mpm - torch.tensor([1.0, 1.0, 1.0], device=dev)) / scale_origin + orig_mean
    for R_ in reversed(rot_mats):
        p = p @ R_
    return p

# ---- 4. particle volume (get_particle_volume convention) + per-region material ----
grid_dx = args.grid_lim / args.n_grid
vi = (pos_mpm / grid_dx).long().clamp(0, args.n_grid - 1)
flat = (vi[:, 0] * args.n_grid + vi[:, 1]) * args.n_grid + vi[:, 2]
cnt = torch.zeros(args.n_grid ** 3, device=dev).index_add_(0, flat, torch.ones(N, device=dev))
volume = (grid_dx ** 3) / cnt[flat]

E_p = torch.full((N,), float(cfg["E"]), device=dev)
nu_p = torch.full((N,), float(cfg["nu"]), device=dev)
dens_p = torch.full((N,), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev)
    s = torch.tensor(reg["size"], device=dev)
    inside = ((pos_mpm - c).abs() <= s).all(-1)
    E_p[inside] = reg["E"]; nu_p[inside] = reg["nu"]; dens_p[inside] = reg["density"]
    print(f"[prep] region {reg['point']}+-{reg['size']}: {inside.sum().item()} particles -> E={reg['E']} density={reg['density']}")
mass_p = dens_p * volume
print(f"[prep] total volume={volume.sum().item():.4f} total mass={mass_p.sum().item():.4f}")

# ---- 5. anchors (the grid replacement) ----
anchor_set, _ = AnchorSet.from_gaussians(pos_mpm, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
anchor_canonical = anchor_set.canonical.clone().contiguous()
M = anchor_canonical.shape[0]
sim = AnchorElasticSim(pos_mpm.contiguous(), anchor_canonical, K=args.K)
w0 = sim._weights(pos_mpm, anchor_canonical)
anchor_mass = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
# PER-PARTICLE Lame params -- the config's additional_material_params make the
# leaf canopy 10x softer than the trunk; averaging them into one scalar (as an
# earlier version did, when the fused kernel was scalar-only) erases exactly
# the stiffness contrast that gives ficus its characteristic motion.
mu, lam = lame_from_E_nu(E_p, nu_p)
print(f"[prep] {M} anchors; per-particle mu in [{mu.min():.3e}, {mu.max():.3e}] "
      f"lam in [{lam.min():.3e}, {lam.max():.3e}] "
      f"({int((E_p != cfg['E']).sum())} particles use a region override)")

# ---- 6. boundary conditions from the config, applied to ANCHORS ----
fixed_mask = torch.zeros(M, dtype=torch.bool, device=dev)
impulse_force = None
impulse_num_dt = 0
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev)
        s = torch.tensor(bc["size"], device=dev)
        assert all(v == 0 for v in bc["velocity"]), "only zero-velocity (fixed) cuboids supported"
        inside = ((anchor_canonical - c).abs() <= s).all(-1)
        fixed_mask |= inside
        print(f"[bc] cuboid {bc['point']}+-{bc['size']}: pinned {inside.sum().item()} anchors")
    elif bc["type"] == "particle_impulse":
        impulse_force = torch.tensor(bc["force"], device=dev)
        impulse_num_dt = int(bc.get("num_dt", 1))
        print(f"[bc] particle_impulse force={bc['force']} num_dt={impulse_num_dt}")
print(f"[bc] total pinned anchors: {fixed_mask.sum().item()}/{M}")

gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
substep_dt = float(cfg["substep_dt"])
frame_dt = float(cfg["frame_dt"])
frame_num = args.frame_num if args.frame_num is not None else int(cfg["frame_num"])
steps_per_frame = max(1, int(round(frame_dt / substep_dt)))
damping = float(cfg.get("grid_v_damping_scale", 1.0))
print(f"[sim] substep_dt={substep_dt} frame_dt={frame_dt} frames={frame_num} "
      f"steps/frame={steps_per_frame} total={frame_num*steps_per_frame} damping={damping} g={cfg['g']}")

# ---- 7. camera (config's MPM-space viewpoint center, mapped back to world) ----
center_world = undo_all(torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev).unsqueeze(0))[0].cpu().numpy()
up_mpm = (torch.tensor(cfg["mpm_space_vertical_upward_axis"], device=dev)
          + torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev)).unsqueeze(0)
up_world = (undo_all(up_mpm)[0].cpu().numpy() - center_world)
up_world = up_world / (np.linalg.norm(up_world) + 1e-9)
world_extent = float((xyz_world.max(0).values - xyz_world.min(0).values).norm())
radius = args.radius_scale * world_extent
az, el = math.radians(cfg["init_azimuthm"]), math.radians(cfg["init_elevation"])
# build an orbit basis around the config's up axis
tmp = np.array([1.0, 0.0, 0.0])
if abs(np.dot(tmp, up_world)) > 0.9:
    tmp = np.array([0.0, 1.0, 0.0])
h1 = np.cross(up_world, tmp); h1 /= np.linalg.norm(h1)
h2 = np.cross(up_world, h1)
eye = center_world + radius * (math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up_world)
fwd = center_world - eye; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up_world); right /= (np.linalg.norm(right) + 1e-9)
tup = np.cross(right, fwd)
Rcam = np.stack([right, -tup, fwd], axis=1)
Tcam = -Rcam.T @ eye
fovx = args.fov_x
fovy = focal2fov(args.width / (2 * math.tan(fovx / 2)), args.height)
wvt = torch.tensor(getWorld2View2(Rcam, Tcam)).transpose(0, 1).float().to(dev)
pm_ = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev)
fpt = (wvt.unsqueeze(0).bmm(pm_.unsqueeze(0))).squeeze(0)
cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, wvt, fpt)


class _Pipe:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = _Pipe()
background = torch.tensor([1.0, 1.0, 1.0], device=dev)
d_rotation = torch.zeros(xyz_world_all.shape[0], 4, device=dev); d_rotation[:, 0] = 1.0
d_scaling = torch.zeros(xyz_world_all.shape[0], 3, device=dev)


def project(pts):
    o = torch.ones(pts.shape[0], 1, device=pts.device)
    clip = torch.cat([pts, o], -1) @ fpt
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
    return torch.stack([(ndc[:, 0] * .5 + .5) * args.width, (ndc[:, 1] * .5 + .5) * args.height], -1)


def dots(img, xy, color, r=4):
    h, w = img.shape[:2]
    for x, y in xy.cpu().numpy():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            img[max(0, yi-r):min(h, yi+r+1), max(0, xi-r):min(w, xi+r+1)] = color
    return img


# ---- 8. simulate ----
anchor_pos = anchor_canonical.clone()
anchor_vel = torch.zeros(M, 3, device=dev)
gaussian_pos_prev = pos_mpm.clone().contiguous()
frames = []
nan_hit = False
step_global = 0
total_steps = frame_num * steps_per_frame

with torch.enable_grad():
    for step_global in tqdm(range(1, total_steps + 1), desc="physgaussian_faithful", ncols=100):
        # DreamPhysics applies the impulse as a pre-P2G operation for the first
        # num_dt substeps: v_p += (force/m_p)*dt. m_p*dv_p = force*dt is then
        # particle-independent, so each anchor receives (sum of its P2G weights)
        # * force * dt of momentum.
        f_ext = None
        if impulse_force is not None and step_global <= impulse_num_dt:
            wsum = torch.zeros(M, device=dev).index_add_(
                0, sim.nn_idx.reshape(-1), w0.reshape(-1))
            dv = (wsum.unsqueeze(-1) * impulse_force.unsqueeze(0) * substep_dt) / anchor_mass.unsqueeze(-1)
            anchor_vel = anchor_vel + dv
            print(f"\n[impulse] applied at step {step_global}: max|dv|={dv.norm(dim=-1).max().item():.4e}")

        anchor_pos, anchor_vel, gaussian_pos_prev, F = sim.step(
            anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev, volume,
            mu, lam, substep_dt,
            gravity=(gravity if gravity.abs().sum() > 0 else None),
            damping=damping, fixed_mask=fixed_mask)
        if torch.isnan(anchor_pos).any():
            print(f"\n[sim] NaN at step {step_global}")
            nan_hit = True
            break
        if step_global % steps_per_frame == 0:
            with torch.no_grad():
                world_now = undo_all(gaussian_pos_prev)
                d_xyz = torch.zeros_like(xyz_world_all)
                d_xyz[kept_idx] = world_now - xyz_world
                res = render(cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, d_rot_as_res=True)
                img = torch.clamp(res["render"], 0, 1)
                fr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8").copy()
                if args.show_anchors:
                    xy = project(undo_all(anchor_pos))
                    fr = dots(fr, xy[fixed_mask], [255, 0, 0], 5)
                    fr = dots(fr, xy[~fixed_mask], [0, 200, 255], 3)
                frames.append(fr)
                if (step_global // steps_per_frame) % 10 == 0:
                    disp = (anchor_pos[~fixed_mask] - anchor_canonical[~fixed_mask]).norm(dim=-1).max().item()
                    print(f"\n[frame {step_global//steps_per_frame}/{frame_num}] max_free_anchor_disp={disp:.5f} (MPM units)")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, frames, fps=25, quality=8)
print(f"[done] wrote {len(frames)} frames -> {args.out} (nan_hit={nan_hit}, steps={step_global}/{total_steps})")
