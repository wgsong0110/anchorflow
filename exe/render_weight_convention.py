"""Render the simulator with its lagged weights beside the same one with the
weights frozen at canonical.

The blend weights are recomputed every step from the Gaussians' previous
positions, which is what makes the simulator path-dependent: the same anchor
configuration and velocity can step to different places depending on how it got
there. Freezing the weights at their canonical values removes that -- the state
becomes a genuine state, and the elastic acceleration becomes a function of the
anchor positions alone -- but it is a different discretisation, so the question
is whether it is a different simulation.

The numbers say 0.2% of peak motion at the config's impulse and 0.7% at three
times it, over 20 coarse steps. This puts those numbers on screen.

The frozen side runs through the torch reference path, since the fused kernel
computes its weights internally; the lagged side runs the same path with the
same weights-detached convention, so the only difference between the panels is
where the weights come from.
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
from tqdm import tqdm

from anchorflow import scene_setup, anchor_mpm
import anchorstep

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--impulse", type=float, default=1.0,
                 help="multiplier on the config's impulse. The weight convention only "
                      "matters where there is local stretch, so the amplitude is the "
                      "variable that decides whether it matters at all.")
ap.add_argument("--width", type=int, default=600)
ap.add_argument("--height", type=int, default=600)
ap.add_argument("--fov_x", type=float, default=0.6911112070083618)
ap.add_argument("--radius_scale", type=float, default=1.5)
ap.add_argument("--show_anchors", action="store_true")
args = ap.parse_args()

from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
W0 = sc.sim._weights(sc.sim.gaussian_canonical, AC).clone()
_orig_weights = anchor_mpm.AnchorElasticSim._weights

gaussians = GaussianModel(3, fea_dim=0)
gaussians.load_ply(args.ply)


class _P:
    debug = False; compute_cov3D_python = False; convert_SHs_python = False


pipe = _P()
bg = torch.tensor([1., 1., 1.], device=dev)
d_rot = torch.zeros(sc.N, 4, device=dev); d_rot[:, 0] = 1.
d_sc = torch.zeros(sc.N, 3, device=dev)

cfg = sc.cfg
center = sc.undo(torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev).unsqueeze(0))[0].cpu().numpy()
up_mpm = (torch.tensor(cfg["mpm_space_vertical_upward_axis"], device=dev)
          + torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev)).unsqueeze(0)
up = sc.undo(up_mpm)[0].cpu().numpy() - center
up /= (np.linalg.norm(up) + 1e-9)
xw = sc.xyz_world[sc.keep]
extent = float((xw.max(0).values - xw.min(0).values).norm())
az, el = math.radians(cfg["init_azimuthm"]), math.radians(cfg["init_elevation"])
tmp = np.array([1., 0., 0.]) if abs(np.dot(np.array([1., 0., 0.]), up)) < 0.9 else np.array([0., 1., 0.])
h1 = np.cross(up, tmp); h1 /= np.linalg.norm(h1); h2 = np.cross(up, h1)
eye = center + args.radius_scale * extent * (
    math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up)
fwd = center - eye; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
tup = np.cross(right, fwd)
Rc = np.stack([right, -tup, fwd], axis=1); Tc = -Rc.T @ eye
fovx = args.fov_x
fovy = focal2fov(args.width / (2 * math.tan(fovx / 2)), args.height)
wvt = torch.tensor(getWorld2View2(Rc, Tc)).transpose(0, 1).float().to(dev)
pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev)
fpt = (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0)
cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, wvt, fpt)


def frame(gauss_mpm):
    d_xyz = sc.undo(gauss_mpm) - sc.xyz_world
    im = torch.clamp(render(cam, gaussians, pipe, bg, d_xyz, d_rot, d_sc,
                             d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def project(pts):
    o = torch.ones(pts.shape[0], 1, device=pts.device)
    clip = torch.cat([pts, o], -1) @ fpt
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
    return torch.stack([(ndc[:, 0] * .5 + .5) * args.width,
                        (ndc[:, 1] * .5 + .5) * args.height], -1)


def dots(img, xy, color, r=3):
    h, w = img.shape[:2]
    for x, y in xy.cpu().numpy():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            img[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)] = color
    return img


base = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
force = base * args.impulse
print(f"[impulse] {args.impulse}x the config's: {['%.4f' % x for x in force.tolist()]}")


def rollout(mode, colour):
    """mode: 'lagged' (weights from the previous Gaussian positions) or 'frozen'."""
    anchorstep.HAVE_CUDA = False          # the kernel computes its own weights
    if mode == "frozen":
        anchor_mpm.AnchorElasticSim._weights = lambda self, gpp, ap: W0
    else:
        # detach so the force matches the kernel's frozen-weight convention and
        # the only difference between the two panels is the weights themselves
        anchor_mpm.AnchorElasticSim._weights = \
            lambda self, gpp, ap: _orig_weights(self, gpp, ap.detach())
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    frames, traj = [], [p.clone()]
    with torch.enable_grad():
        for _ in tqdm(range(args.frames + 1), desc=mode, ncols=90):
            with torch.no_grad():
                fr = frame(gp)
                frames.append(dots(fr, project(sc.undo(p)), colour) if args.show_anchors else fr)
            p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
            traj.append(p.clone())
    anchorstep.HAVE_CUDA = True
    anchor_mpm.AnchorElasticSim._weights = _orig_weights
    return frames, torch.stack(traj)


fr_l, tr_l = rollout("lagged", [0, 160, 255])
fr_f, tr_f = rollout("frozen", [0, 200, 90])

span = (tr_l - AC).norm(dim=-1).max().clamp(min=1e-12).item()
err = (tr_l - tr_f)[:, ~fixed].norm(dim=-1).mean(-1)
print(f"\n[divergence] peak motion {span:.5f}")
print(f"  {'frame':>6} {'mean anchor diff':>18} {'% of peak motion':>18}")
n = tr_l.shape[0]
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:6d} {err[i].item():18.6f} {100 * err[i].item() / span:17.3f}%")
print(f"[divergence] final {100 * err[-1].item() / span:.3f}% of peak motion")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
n = min(len(fr_l), len(fr_f))
imageio.mimwrite(args.out, [np.concatenate([fr_l[i], fr_f[i]], axis=1) for i in range(n)],
                 fps=15, quality=8)
print(f"[done] {n} frames (left = lagged weights, right = frozen canonical) -> {args.out}")
