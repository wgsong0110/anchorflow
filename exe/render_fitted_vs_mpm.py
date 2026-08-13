"""MPM, the anchor simulator, and the fitted one, side by side.

The fit halved its one-step error and did not improve the rollout -- 10.31% to
11.62% against MPM's particles, with the motion falling to about 70% of MPM's.
That is a summary statistic over two trajectories, and it says nothing about
what the difference looks like: whether the fitted simulator damps uniformly,
lags, or bends somewhere else entirely.

Three panels from one camera, the same impulse in each, MPM on the left as the
thing being chased. Every panel is the Gaussian cloud a renderer would draw, so
what is compared is what would be seen.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import imageio
import numpy as np
import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.anchor_sparse import AnchorSparse

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True)
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--which", type=int, default=0, help="held-out impulse to show")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=640)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()


def render(cam, g, pipe, bg, d_xyz, d_rot, d_sc):
    return _render_scgs(cam, g, pipe, bg, d_xyz, d_rot, d_sc, d_rot_as_res=True)


class MiniCam:
    def __init__(self, W, H, fovy, fovx, zn, zf, wvt, fpt):
        self.image_width, self.image_height = W, H
        self.FoVy, self.FoVx = fovy, fovx
        self.znear, self.zfar = zn, zf
        self.world_view_transform = wvt
        self.full_proj_transform = fpt
        self.camera_center = wvt.inverse()[3, :3]


sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
mat = T.mat
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
X, _V = blob["chk"][args.which]
force = blob["forces"]["chk"][args.which]

fb = torch.load(args.fit, map_location=dev, weights_only=False)
fit = AnchorSparse(sc, c=fb.get("c", 0.25), eig_floor=fb.get("eig_floor", 0.02)).to(dev)
fit._rebuild(fb["pos"].to(dev), fb["quat"].to(dev), fb["log_s"].to(dev))
print(f"[fit] {args.fit} at iteration {fb.get('iter')}, {fit.M} anchors")

# ---- the three trajectories, as Gaussian clouds ---------------------------
runs = {}
runs["MPM"] = X[: args.frames + 1]

p, v, gp = sc.anchor_canonical.clone(), sc.initial_velocity(force), sc.pos.clone()
out = [gp.clone()]
for _ in tqdm(range(args.frames), desc="anchor 8-NN", ncols=90):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
    out.append(gp.clone())
runs["anchor simulator"] = torch.stack(out)

cache = fit.prepare()
p = fit.project(X[0], cache)
v = fit.project_v(_V[0], cache)
full = sc.pos.clone()
out = []
for k in tqdm(range(args.frames + 1), desc="fitted", ncols=90):
    g = fit.gaussian_pos(p, cache)
    full = full.clone(); full[mat] = g
    out.append(full.clone())
    if k < args.frames:
        p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
runs["fitted to MPM"] = torch.stack(out)

# MPM's rows are material only; the renderer wants the whole cloud.
#
# The opacity-rejected Gaussians -- a sixth of the cloud -- carry zero volume and
# are not simulated by anything, but they are interleaved through the foliage and
# they do get drawn. Left at their canonical positions they stand still while the
# tree moves around them, which reads as a ghost of the original and is a
# property of this script rather than of MPM. They are carried by the material
# around them: the weighted mean displacement of their nearest material
# particles, which is what the anchor panels do for them through skinning.
K_CARRY = 8
rest = torch.nonzero(~sc.keep, as_tuple=False).squeeze(-1)
d = torch.cdist(sc.pos[rest], T.pos_m)
nd, ni = torch.topk(d, K_CARRY, dim=-1, largest=False)
cw = 1.0 / nd.clamp(min=1e-6)
cw = cw / cw.sum(-1, keepdim=True)
mpm_full = []
for k in range(args.frames + 1):
    q = sc.pos.clone()
    q[mat] = runs["MPM"][k]
    disp = runs["MPM"][k] - T.pos_m
    q[rest] = sc.pos[rest] + (cw.unsqueeze(-1) * disp[ni]).sum(1)
    mpm_full.append(q)
runs["MPM"] = torch.stack(mpm_full)
print(f"[render] {rest.shape[0]} zero-volume Gaussians carried by their "
      f"{K_CARRY} nearest material particles")

# ---- camera, as the config specifies it -----------------------------------
gaussians = GaussianModel(3, fea_dim=0)
gaussians.load_ply(args.ply)


class _P:
    debug = False; compute_cov3D_python = False; convert_SHs_python = False


pipe, bg = _P(), torch.tensor([1., 1., 1.], device=dev)
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
cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, wvt,
               (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))


def frame(gauss_mpm):
    d_xyz = sc.undo(gauss_mpm) - sc.xyz_world
    im = torch.clamp(render(cam, gaussians, pipe, bg, d_xyz, d_rot, d_sc)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, img.shape[1] - 1, 24], fill=(40, 40, 40))
        d.text((8, 7), text, fill=(255, 255, 255))
        return np.array(im)
    except Exception:
        img[:24] = [40, 40, 40]
        return img


names = ["MPM", "anchor simulator", "fitted to MPM"]
span = (runs["MPM"] - runs["MPM"][0]).norm(dim=-1).max().clamp(min=1e-12)
err = {n: ((runs[n][:, mat] - runs["MPM"][:, mat]).norm(dim=-1).mean(-1) / span)
       for n in names}
vid = []
for k in tqdm(range(args.frames + 1), desc="render", ncols=90):
    row = []
    for n in names:
        tag = n if n == "MPM" else f"{n}   {100 * err[n][k]:.1f}%"
        row.append(label(frame(runs[n][k]), tag))
    vid.append(np.concatenate(row, axis=1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimsave(args.out, vid, fps=args.fps, quality=8)
print(f"\n[saved] {args.out}")
for n in names[1:]:
    print(f"  {n:>18}: all-frame {100 * err[n].mean():.2f}%, "
          f"final {100 * err[n][-1]:.2f}%")
