"""The explicit simulator at several substep counts, side by side.

The stability limit sits between 11 and 10 substeps per 4e-3 coarse step: at 11
the trajectory matches a 320-substep reference to 0.06%, at 10 it is gone within
four frames. That is a cliff, not a gradual loss of accuracy, and a table of
error percentages does not convey what falling off it looks like.

A panel that has produced a non-finite position holds its last valid frame and
is tinted, so the moment each one dies is visible rather than the video simply
ending.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--substeps", type=int, nargs="+", default=[40, 20, 10, 2, 1])
ap.add_argument("--coarse_dt", type=float, default=4e-3)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--width", type=int, default=460)
ap.add_argument("--height", type=int, default=460)
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
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
NATIVE = int(args.coarse_dt / sc.sub_dt)

base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
# the impulse is deposited as force * sub_dt and the damping is applied once per
# substep, so both are pinned to the native substep count -- otherwise the panels
# differ in how hard they were hit as well as in how they integrate
V0 = sc.initial_velocity(base_force)
D_REF = sc.damping ** NATIVE

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


def dots(img, xy, color, r=2):
    h, w = img.shape[:2]
    for x, y in xy.cpu().numpy():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            img[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)] = color
    return img


def label(img, text, dead):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, img.shape[1] - 1, 22],
                     fill=(200, 40, 40) if dead else (40, 40, 40))
        d.text((8, 6), text, fill=(255, 255, 255))
        return np.array(im)
    except Exception:
        img[:22] = [200, 40, 40] if dead else [40, 40, 40]
        return img


panels = {}
for n in args.substeps:
    sc.sub_dt = args.coarse_dt / n
    sc.damping = D_REF ** (1.0 / n)
    p, v, gp = AC.clone(), V0.clone(), sc.pos.clone()
    frames, died, died_amp = [], None, None
    last = None
    for k in tqdm(range(args.frames + 1), desc=f"{n} substeps", ncols=90):
        amp = (p - AC)[~fixed].norm(dim=-1).max().item() if died is None else None
        if died is None:
            fr = frame(gp)
            if args.show_anchors:
                fr = dots(fr, project(sc.undo(p)), [255, 120, 0])
            last = fr
            # a panel that has left the scene entirely is as dead as one that
            # produced a NaN, and holding its last sane frame is more legible
            # than watching it fill the viewport
            if amp > 20 * 0.12194:
                died, died_amp = k, amp
        # the displacement goes in the label because a panel that died at frame 2
        # is frozen on a picture of an object that has barely moved, which reads
        # as "nothing happened" rather than "this blew up a frame later"
        txt = f"{n} substeps  dt={args.coarse_dt / n:.1e}   "
        txt += (f"displacement {amp:.3e}" if died is None
                else f"DIVERGED at frame {died}, displacement {died_amp:.2e}")
        frames.append(label(last.copy(), txt, died is not None))
        if died is not None:
            continue
        p, v, gp = sc.explicit_step(p, v, gp, n)
        if not torch.isfinite(p).all():
            died, died_amp = k + 1, float("inf")
    panels[n] = frames
    print(f"  {n} substeps: " + (f"diverged at frame {died}" if died is not None
                                  else "survived"))
sc.sub_dt = args.coarse_dt / NATIVE
sc.damping = float(cfg.get("grid_v_damping_scale", 1.0))

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
n_f = min(len(v) for v in panels.values())
out = [np.concatenate([panels[n][i] for n in args.substeps], axis=1) for i in range(n_f)]
imageio.mimwrite(args.out, out, fps=15, quality=8)
print(f"[done] {n_f} frames, {len(args.substeps)} panels -> {args.out}")
