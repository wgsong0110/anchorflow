"""Render a learned rollout beside the explicit reference, same camera, same clock.

The network drives the anchors on its own -- position from its own output,
velocity from its own positions, elastic acceleration re-evaluated at whatever
configuration it has produced -- and the Gaussians are skinned onto them for
display only. The right panel is the explicit simulator over the same physical
time, so any divergence is visible rather than inferred from a number.
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

from anchorflow import scene_setup
from anchorflow.nextstate import NextStep

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=60)
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
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
targs = ck["args"]
sc = scene_setup.build(args.ply, args.config, targs["n_anchors"], targs["K"], device=dev)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = targs["dt_mult"] * sc.sub_dt

net = NextStep(targs["hidden"], targs["depth"], targs["heads"],
                ck["disp_scale"], ck["vel_scale"], ck["acc_scale"]).to(dev)
net.load_state_dict(ck["model"]); net.eval()
print(f"[cfg] {targs['dt_mult']} substeps/step, {targs['n_traj']} training trajectories, "
      f"noise {targs.get('noise', 0)}")

# the renderer needs its own copy of the Gaussians
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


# ---- the explicit reference, sampled at the coarse step ----
p_e, v_e = AC.clone(), sc.initial_velocity()
gp_e = sc.pos.clone()
ref, frames_e = [p_e.clone()], []
with torch.enable_grad():
    for _ in tqdm(range(args.frames + 1), desc="reference", ncols=90):
        with torch.no_grad():
            fr = frame(gp_e)
            frames_e.append(dots(fr, project(sc.undo(p_e)), [0, 160, 255]) if args.show_anchors else fr)
        p_e, v_e, gp_e = sc.explicit_step(p_e, v_e, gp_e, targs["dt_mult"])
        ref.append(p_e.clone())
REF = torch.stack(ref)

# ---- the network, from the same initial condition ----
p = REF[1].clone()
v = (REF[1] - REF[0]) / dt
gp = sc.skin(p, sc.pos.clone())
frames_n, err = [], []
for i in tqdm(range(len(frames_e)), desc="network", ncols=90):
    fr = frame(gp)
    frames_n.append(dots(fr, project(sc.undo(p)), [255, 120, 0]) if args.show_anchors else fr)
    k = min(i + 1, REF.shape[0] - 1)
    err.append((p - REF[k]).norm(dim=-1).mean().item())
    a = sc.elastic_accel(p, gp)
    du = net(p, v, a, dt)
    du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
    p = p + du
    v = du / dt
    if not torch.isfinite(p).all():
        print(f"[network] NaN at frame {i}")
        break
    gp = sc.skin(p, gp)

span = (REF - AC).norm(dim=-1).max().item()
print(f"\n[rollout] reference peak anchor displacement = {span:.5f}")
print(f"  {'frame':>6} {'mean anchor err':>16} {'% of reference motion':>22}")
n = len(err)
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:6d} {err[i]:16.5f} {100 * err[i] / span:21.2f}%")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
n = min(len(frames_n), len(frames_e))
imageio.mimwrite(args.out, [np.concatenate([frames_n[i], frames_e[i]], axis=1)
                             for i in range(n)], fps=15, quality=8)
print(f"[done] {n} frames (left = network, right = explicit) -> {args.out}")
