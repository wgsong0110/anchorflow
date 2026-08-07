"""Draw the streaming training data, so what the network is being taught is visible.

The from-rest trajectories are easy to picture: a kick, then decay. A run that
keeps being hit is not -- impulses land while the object is still moving, and
the peak displacement measured during collection reached 4.6x what the config's
own impulse produces. That number is worth seeing as motion rather than trusting
as a scalar, because it is also the number that decides whether this data is
useful coverage or a set of configurations the simulator should not be trusted
in.

Frames where an impulse lands are marked, and those are exactly the samples the
training set drops.
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
from anchorflow.streams import stream

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--impulse_every", type=int, default=20)
ap.add_argument("--impulse_range", type=float, default=4.0)
ap.add_argument("--stream_amp_cap", type=float, default=3.0)
ap.add_argument("--field", action="store_true",
                 help="impulses become smooth random force fields rather than one vector on "
                      "the whole object -- see scene_setup.random_force_field")
ap.add_argument("--skip", type=int, default=0,
                 help="draw the n-th run rather than the first, by advancing the same "
                      "seeded generator the training script advances")
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
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True)
AC, fixed = sc.anchor_canonical, sc.fixed_mask

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

gen = torch.Generator(device=dev); gen.manual_seed(1234)
ref_peak = None
# the training script draws its held-out trajectories first, then the runs, off
# the same generator -- advance it the same way so this is one of those runs
for t in range(5):
    if t == 0:
        continue
    torch.rand(1, device=dev, generator=gen)
    torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))

# the cap is measured against the config impulse's own peak
p, v, gp = AC.clone(), sc.initial_velocity(base), sc.pos.clone()
peak = 0.0
for _ in range(60):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
    peak = max(peak, (p - AC)[~fixed].norm(dim=-1).max().item())
cap = args.stream_amp_cap * peak
print(f"[cap] config impulse peaks at {peak:.5f}; impulses skipped above {cap:.5f}")

for _ in range(args.skip):
    stream(sc, args.frames, args.dt_mult, base, gen, cap,
           impulse_every=args.impulse_every, impulse_range=args.impulse_range,
           keep_accel=False, field=args.field)

frames, hits = [], []
bar = tqdm(total=args.frames, desc="stream", ncols=90)


def draw(k, p_, gp_, fired):
    fr = frame(gp_)
    if args.show_anchors:
        fr = dots(fr, project(sc.undo(p_)), [255, 120, 0])
    if fired:
        fr[:6, :] = [220, 40, 40]; fr[-6:, :] = [220, 40, 40]
        fr[:, :6] = [220, 40, 40]; fr[:, -6:] = [220, 40, 40]
    frames.append(fr); hits.append(fired)
    bar.update(1)


ps, _, bad = stream(sc, args.frames, args.dt_mult, base, gen, cap,
                     impulse_every=args.impulse_every, impulse_range=args.impulse_range,
                     keep_accel=False, on_step=draw, field=args.field)
bar.close()

amp = (ps - AC)[:, ~fixed].norm(dim=-1).max(-1).values
print(f"\n[stream] {ps.shape[0]} steps, {int(bad.sum())} impulses "
      f"({int(bad.sum())} samples dropped)")
print(f"[stream] displacement: peak {amp.max():.5f} = {amp.max() / peak:.1f}x the config "
      f"impulse's, mean {amp.mean():.5f}")
print(f"  {'step':>6} {'displacement':>14} {'x config':>9} {'impulse':>8}")
n = ps.shape[0]
for i in range(0, n, max(1, n // 15)):
    hit = "*" if i < len(hits) and hits[i] else ""
    print(f"  {i:6d} {amp[i].item():14.5f} {amp[i].item() / peak:8.2f}x {hit:>8}")

import imageio
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, frames, fps=15, quality=8)
print(f"[done] {len(frames)} frames (red border = impulse, that sample is dropped) "
      f"-> {args.out}")
