"""Autoregressive rollout with the trained one-shot implicit-step GNN, rendered
side by side against the explicit physics reference.

The GNN was trained on SINGLE steps (minimize the within-step momentum residual
from states the explicit simulator visits). Chaining it is a strictly harder
test: nothing during training told it to stay on-distribution after its own
error feeds back in. This script exists to measure that, so it reports, per
frame, both the residual ratio it was trained on and the drift away from the
explicit reference trajectory.

Left panel = GNN rollout (one network step per frame), right = explicit physics
(steps_per_frame substeps per frame), same camera.
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
from anchorflow.implicit import (incremental_potential, newmark_accel,
                                  newmark_velocity, DuCorrector, predict_du,
                                  anchor_elastic_accel)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--dt_mult", type=float, default=None,
                 help="step size as a multiple of the config's substep_dt; default is the "
                      "checkpoint's dt_big. The step the network can actually sustain "
                      "autoregressively is the number this experiment is after, so it must "
                      "be sweepable independently of what it was trained at.")
ap.add_argument("--sweep", default=None,
                 help="comma-separated dt multipliers to test for stability instead of "
                      "rendering; reports how far each gets before diverging")
ap.add_argument("--radius_scale", type=float, default=1.5)
ap.add_argument("--width", type=int, default=600)
ap.add_argument("--height", type=int, default=600)
ap.add_argument("--fov_x", type=float, default=0.6911112070083618)
ap.add_argument("--show_anchors", action="store_true")
args = ap.parse_args()

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from anchorflow import graph as G

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
targs = ck["args"]
beta = targs["beta"]; gamma = targs["gamma"]
sub_dt = float(cfg["substep_dt"])
dt_big = args.dt_mult * sub_dt if args.dt_mult else targs["dt_big"]
ACC_SCALE = ck.get("accel_scale")
assert ACC_SCALE is not None, "checkpoint predates the constant-gain parameterisation"
steps_per_frame = max(1, int(round(dt_big / sub_dt)))
print(f"[cfg] dt={dt_big} (= {steps_per_frame} explicit substeps, trained at "
      f"{targs['dt_big']/sub_dt:.0f}x), beta={beta} gamma={gamma} accel_scale={ACC_SCALE:.2f}")

gaussians = GaussianModel(3, fea_dim=0)
gaussians.load_ply(args.ply)
xyz_all = gaussians.get_xyz.detach().clone(); op = gaussians.get_opacity.detach().clone()
# The opacity threshold selects PHYSICS MATERIAL, nothing else. DreamPhysics
# additionally deletes the rejected kernels from the render (ms_simulation.py:142
# masks position, covariance, opacity and SH alike), but on ficus those are 16%
# of the Gaussians sitting *inside* the foliage (median distance to a kept one:
# 0.002 in a scene spanning 1.0) -- deleting them throws away real appearance,
# and leaving them undeformed pins them at their canonical position and smears
# an afterimage over the moving plant. Both are wrong. Every Gaussian is skinned
# by the anchors; the rejected ones simply carry zero volume, which makes their
# elastic energy and their force contribution exactly zero (the fused kernel only
# ever multiplies by volume, never divides), so the physics is bit-for-bit the
# filtered simulation while the render stays complete.
keep = op[:, 0] > cfg["opacity_threshold"]
xyz_w = xyz_all[keep]
pmin, pmax = xyz_w.min(0).values, xyz_w.max(0).values     # MPM space from the material only
orig_mean = (pmin + pmax) / 2; scale_origin = 1.0 / (pmax - pmin).max()
to_mpm = lambda p: (p - orig_mean) * scale_origin + torch.tensor([1., 1., 1.], device=dev)
POS = to_mpm(xyz_all).contiguous()          # ALL Gaussians live in the sim
POS_MAT = POS[keep]                          # ...but only these are material
N = POS.shape[0]
print(f"[prep] opacity>{cfg['opacity_threshold']}: {int(keep.sum())}/{N} carry material, "
      f"all {N} are skinned")


def undo(p):
    return (p - torch.tensor([1., 1., 1.], device=dev)) / scale_origin + orig_mean


# voxel-occupancy volumes are a property of the MATERIAL point cloud, so they
# are computed over the kept subset exactly as before and scattered back; the
# rejected Gaussians keep volume 0.
ng, grid_lim = 100, 2.0
dx = grid_lim / ng
vi = (POS_MAT / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(POS_MAT.shape[0], device=dev))
VOL = torch.zeros(N, device=dev)
VOL[keep] = (dx ** 3) / cnt[flat]
VOL = VOL.contiguous()
E_p = torch.full((N,), float(cfg["E"]), device=dev)
nu_p = torch.full((N,), float(cfg["nu"]), device=dev)
dens_p = torch.full((N,), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev); s = torch.tensor(reg["size"], device=dev)
    inside = ((POS - c).abs() <= s).all(-1)
    E_p[inside] = reg["E"]; nu_p[inside] = reg["nu"]; dens_p[inside] = reg["density"]
mass_p = dens_p * VOL                       # zero wherever VOL is zero
MU, LAM = lame_from_E_nu(E_p, nu_p)

# anchors are seeded from the material cloud, and the RBF radius is taken from
# it too, so adding the skinned-only Gaussians cannot shift the physics.
aset, _ = AnchorSet.from_gaussians(POS_MAT, node_num=targs["n_anchors"], latent_dim=0,
                                    e_dim=0, K=targs["K"])
AC = aset.canonical.clone().contiguous(); M = AC.shape[0]
radius_mat = AnchorElasticSim(POS_MAT, AC, K=targs["K"]).radius
sim = AnchorElasticSim(POS, AC, K=targs["K"], radius=radius_mat)
w0 = sim._weights(POS, AC)
MASS = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
fixed_mask = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed_mask |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
edge_index = G.knn_graph(AC, k=targs["k_graph"])

net = DuCorrector(targs["hidden"], targs["mp_steps"],
                   use_force=not targs.get("no_force_feature", True),
                   processor=targs.get("processor", "mpnn"),
                   heads=targs.get("heads", 4),
                   raw_io=targs.get("raw_io", False)).to(dev)
net.load_state_dict(ck["model"]); net.eval()

# ---------- camera (same construction as the faithful renderer) ----------
center = undo(torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev).unsqueeze(0))[0].cpu().numpy()
up_mpm = (torch.tensor(cfg["mpm_space_vertical_upward_axis"], device=dev)
          + torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev)).unsqueeze(0)
up = undo(up_mpm)[0].cpu().numpy() - center; up /= (np.linalg.norm(up) + 1e-9)
extent = float((xyz_w.max(0).values - xyz_w.min(0).values).norm())
radius = args.radius_scale * extent
az, el = math.radians(cfg["init_azimuthm"]), math.radians(cfg["init_elevation"])
tmp = np.array([1., 0., 0.]) if abs(np.dot(np.array([1., 0., 0.]), up)) < 0.9 else np.array([0., 1., 0.])
h1 = np.cross(up, tmp); h1 /= np.linalg.norm(h1); h2 = np.cross(up, h1)
eye = center + radius * (math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up)
fwd = center - eye; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
tup = np.cross(right, fwd)
Rc = np.stack([right, -tup, fwd], axis=1); Tc = -Rc.T @ eye
fovx = args.fov_x; fovy = focal2fov(args.width / (2 * math.tan(fovx / 2)), args.height)
wvt = torch.tensor(getWorld2View2(Rc, Tc)).transpose(0, 1).float().to(dev)
pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev)
fpt = (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0)
cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, wvt, fpt)


class _P:
    debug = False; compute_cov3D_python = False; convert_SHs_python = False


pipe = _P()
bg = torch.tensor([1., 1., 1.], device=dev)
d_rot = torch.zeros(xyz_all.shape[0], 4, device=dev); d_rot[:, 0] = 1.
d_sc = torch.zeros(xyz_all.shape[0], 3, device=dev)


def render_state(gauss_mpm):
    d_xyz = undo(gauss_mpm) - xyz_all
    im = torch.clamp(render(cam, gaussians, pipe, bg, d_xyz, d_rot, d_sc, d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def initial_impulse():
    v = torch.zeros(M, 3, device=dev)
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "particle_impulse":
            force = torch.tensor(bc["force"], device=dev)
            # material particles only -- the zero-volume Gaussians carry no mass
            wsum = torch.zeros(M, device=dev).index_add_(
                0, sim.nn_idx.reshape(-1), (w0 * keep.unsqueeze(-1)).reshape(-1))
            v = v + (wsum.unsqueeze(-1) * force.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
    v[fixed_mask] = 0
    return v


def rollout_only(dt, n):
    """autoregressive chain at this dt, no rendering -- returns (frames survived,
    peak displacement, residual ratio at the last surviving frame)."""
    p, v = AC.clone(), initial_impulse()
    gp = POS.clone()
    # seed the acceleration with the real f/m rather than zero: a is a physical
    # acceleration everywhere the trainer looks, and starting it at 0 puts the
    # very first frame outside anything the network has seen
    a = anchor_elastic_accel(sim, p, gp, VOL, MU, LAM, MASS, fixed_mask)
    damp = float(cfg.get("grid_v_damping_scale", 1.0))
    peak, rel = 0.0, float("nan")
    for f in range(n):
        fa = anchor_elastic_accel(sim, p, gp, VOL, MU, LAM, MASS, fixed_mask) \
            if net.use_force else None
        du, pred0 = predict_du(net, p, v, a, dt, edge_index, ACC_SCALE, beta, fixed_mask, fa,
                                targs.get("direct_du", False), targs.get("velocity_out", False))
        _, R = incremental_potential(du, sim, p, gp, VOL, MU, LAM, MASS, v, a,
                                      dt, None, beta, fixed_mask)
        _, R0 = incremental_potential(pred0, sim, p, gp, VOL, MU, LAM, MASS, v, a,
                                       dt, None, beta, fixed_mask)
        a_next = newmark_accel(du, v, a, dt, beta)
        v = damp * newmark_velocity(a_next, v, a, dt, gamma)
        p = p + du; a = a_next
        v = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(v), v)
        p = torch.where(fixed_mask.unsqueeze(-1), AC, p)
        a = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(a), a)
        if not torch.isfinite(p).all():
            return f, peak, rel
        d = (p[~fixed_mask] - AC[~fixed_mask]).norm(dim=-1).max().item()
        peak = max(peak, d)
        rel = R.norm().item() / max(R0.norm().item(), 1e-20)
        _, _, gp, _ = sim.step(p, torch.zeros_like(v), MASS, gp, VOL, MU, LAM, 0.0,
                                gravity=None, damping=1.0, fixed_mask=fixed_mask)
    return n, peak, rel


if args.sweep:
    # the explicit reference's own peak displacement is the yardstick: a rollout
    # that "survives" while drifting 100x further than the physics is not stable.
    p_e, v_e = AC.clone(), initial_impulse(); gp_e = POS.clone(); ref_peak = 0.0
    with torch.enable_grad():
        for _ in tqdm(range(args.frames * int(round(targs["dt_big"] / sub_dt))),
                       desc="reference", ncols=90):
            p_e, v_e, gp_e, _ = sim.step(p_e, v_e, MASS, gp_e, VOL, MU, LAM, sub_dt,
                                          gravity=(gravity if gravity.abs().sum() > 0 else None),
                                          damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                                          fixed_mask=fixed_mask)
            ref_peak = max(ref_peak, (p_e[~fixed_mask] - AC[~fixed_mask]).norm(dim=-1).max().item())
    # every dt must cover the SAME physical time as the reference, or a small dt
    # looks stable merely for having simulated less of the motion
    total_sub = args.frames * int(round(targs["dt_big"] / sub_dt))
    print(f"\n[sweep] explicit reference over {total_sub * sub_dt:.3f}s: "
          f"peak anchor displacement = {ref_peak:.5f}")
    print(f"  {'dt/sub':>7} {'steps':>7} {'sim time':>9} {'peak disp':>12} {'vs ref':>9} {'R/R_expl':>9}")
    for m in [float(x) for x in args.sweep.split(",")]:
        n = max(1, int(round(total_sub / m)))
        n_ok, peak, rel = rollout_only(m * sub_dt, n)
        flag = "ok" if (n_ok == n and peak < 2 * ref_peak) else "DIVERGED"
        print(f"  {m:7.0f} {n_ok:4d}/{n:<4d} {n_ok*m*sub_dt:9.3f} {peak:12.5f} "
              f"{peak/ref_peak:9.2f} {rel:9.4f}  {flag}")
    sys.exit(0)

# ---------------- GNN rollout: ONE network step per frame ----------------
p_g, v_g = AC.clone(), initial_impulse()
gp_g = POS.clone()
a_g = anchor_elastic_accel(sim, p_g, gp_g, VOL, MU, LAM, MASS, fixed_mask)
frames_g, stats = [], []
damping = float(cfg.get("grid_v_damping_scale", 1.0))
for f in tqdm(range(args.frames), desc="gnn", ncols=90):
    fa_g = anchor_elastic_accel(sim, p_g, gp_g, VOL, MU, LAM, MASS, fixed_mask) \
        if net.use_force else None
    du, pred0 = predict_du(net, p_g, v_g, a_g, dt_big, edge_index, ACC_SCALE,
                            beta, fixed_mask, fa_g, targs.get("direct_du", False), targs.get("velocity_out", False))
    _, R = incremental_potential(du, sim, p_g, gp_g, VOL, MU, LAM, MASS, v_g, a_g,
                                  dt_big, None, beta, fixed_mask)
    _, R0 = incremental_potential(pred0, sim, p_g, gp_g, VOL, MU, LAM, MASS, v_g, a_g,
                                   dt_big, None, beta, fixed_mask)
    a_next = newmark_accel(du, v_g, a_g, dt_big, beta)
    v_g = damping * newmark_velocity(a_next, v_g, a_g, dt_big, gamma)
    p_g = p_g + du
    a_g = a_next
    v_g = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(v_g), v_g)
    p_g = torch.where(fixed_mask.unsqueeze(-1), AC, p_g)
    a_g = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(a_g), a_g)
    _, _, gp_g, _ = sim.step(p_g, torch.zeros_like(v_g), MASS, gp_g, VOL, MU, LAM, 0.0,
                              gravity=None, damping=1.0, fixed_mask=fixed_mask)
    if torch.isnan(p_g).any():
        print(f"[gnn] NaN at frame {f}"); break
    frames_g.append(render_state(gp_g))
    stats.append((R.norm().item() / max(R0.norm().item(), 1e-20),
                  (p_g[~fixed_mask] - AC[~fixed_mask]).norm(dim=-1).max().item()))

# ---------------- explicit reference: steps_per_frame substeps ----------------
p_e, v_e = AC.clone(), initial_impulse()
gp_e = POS.clone(); frames_e, disp_e = [], []
with torch.enable_grad():
    for f in tqdm(range(len(frames_g)), desc="explicit", ncols=90):
        for _ in range(steps_per_frame):
            p_e, v_e, gp_e, _ = sim.step(p_e, v_e, MASS, gp_e, VOL, MU, LAM, sub_dt,
                                          gravity=(gravity if gravity.abs().sum() > 0 else None),
                                          damping=damping, fixed_mask=fixed_mask)
        if torch.isnan(p_e).any():
            print(f"[explicit] NaN at frame {f}"); break
        with torch.no_grad():
            frames_e.append(render_state(gp_e))
            disp_e.append((p_e[~fixed_mask] - AC[~fixed_mask]).norm(dim=-1).max().item())

n = min(len(frames_g), len(frames_e))
print(f"\n[rollout] {n} frames  (GNN: 1 net step/frame, explicit: {steps_per_frame} substeps/frame)")
print(f"  {'frame':>6} {'R/R_expl':>10} {'disp_gnn':>10} {'disp_expl':>10}")
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:>6} {stats[i][0]:10.4f} {stats[i][1]:10.5f} {disp_e[i]:10.5f}")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
combo = [np.concatenate([frames_g[i], frames_e[i]], axis=1) for i in range(n)]
imageio.mimwrite(args.out, combo, fps=20, quality=8)
print(f"[done] wrote {n} side-by-side frames (left=GNN, right=explicit) -> {args.out}")
