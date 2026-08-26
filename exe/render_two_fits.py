"""Two fitted discretisations against MPM, side by side.

exe/render_eval_case.py draws MPM, the sampled anchors and one fit. What is
wanted now is a different comparison -- two FITS against the same MPM, because
the question is whether a 5.54% run and a 5.53% run differ in any way a person
can see, or whether a hundredth of a percent is a number and nothing else.

Everything is built the way exe/eval_vs_mpm.py builds a row: from rest, under
the impulse the cached reference was made with, with the same stepping. The
caption on each panel is that panel's error so far against MPM, in the
normalisation the table uses, so the video and the number are the same claim.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import imageio
import numpy as np
import torch
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--cache", required=True, help="the eval_vs_mpm reference cache")
ap.add_argument("--fit", action="append", required=True,
                 help="NAME=path.pt, repeatable; each becomes a panel")
ap.add_argument("--cases", nargs="+", default=["field:0"])
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--width", type=int, default=560)
ap.add_argument("--height", type=int, default=560)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--unfitted", action="store_true",
                 help="add the geometry-initialised simulator as its own panel -- "
                      "what the discretisation does before anything is fitted")
ap.add_argument("--metrics", action="store_true",
                 help="score every panel against MPM's own render: PSNR, SSIM and "
                      "LPIPS. The position error says how far the particles are; "
                      "this says what a person would see.")
ap.add_argument("--only_material", action="store_true",
                 help="hide the Gaussians no simulator moved, so what is left "
                      "on screen is what the physics actually said")
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.view import (build_camera, label, local_deformation,
                              make_renderer)

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
blob = torch.load(args.cache, map_location=dev, weights_only=False)

FITS = []
for spec in args.fit:
    name, path = spec.split("=", 1)
    fs, fb = load_fitted(sc, path, dev)
    print(f"[fit] {name}: {os.path.basename(path)} at iteration {fb.get('iter')}, "
          f"{fs.M} anchors")
    FITS.append((name, fs))

# the zero-volume Gaussians, carried by the material around them: a sixth of the
# cloud is simulated by nothing and would stand still while the tree moves
K_CARRY = 8
from scipy.spatial import cKDTree

rest = torch.nonzero(~sc.keep, as_tuple=False).squeeze(-1)
_tree = cKDTree(T.pos_m.cpu().numpy())
nd_np, ni_np = _tree.query(sc.pos[rest].cpu().numpy(), k=K_CARRY)
nd = torch.from_numpy(nd_np).float().to(dev)
ni = torch.from_numpy(ni_np).long().to(dev)
cw = (1.0 / nd.clamp(min=1e-6))
cw = cw / cw.sum(-1, keepdim=True)
mat = T.mat


def full(xm):
    """material Gaussians -> the whole cloud, the rest carried by their neighbours"""
    out = sc.pos.clone()
    out[mat] = xm
    if rest.numel():
        disp = xm - sc.pos[mat]
        out[rest] = sc.pos[rest] + (cw.unsqueeze(-1) * disp[ni]).sum(1)
    return out


# ---- how the two renders compare as pictures, not as point sets -------------
LPIPS = None
if args.metrics:
    from utils.loss_utils import ssim as ssim_fn

    def to_t(img):
        return torch.from_numpy(img).to(dev).permute(2, 0, 1).float() / 255.0

    def psnr(a, b):
        m = float((a - b).pow(2).mean())
        return 100.0 if m <= 0 else float(-10 * torch.log10(torch.tensor(m)))

    try:
        import lpips as _lp
        LPIPS = _lp.LPIPS(net="alex").to(dev)
    except Exception as e:
        print(f"[metrics] LPIPS unavailable ({type(e).__name__}); PSNR and SSIM only")

cam = build_camera(sc, args.width, args.height, args.fov_x, args.radius_scale)
frame = make_renderer(sc, args.ply, cam)
# the splats have to stretch with the cloud, or both panels speckle and the one
# that deformed more -- MPM -- looks like it fell apart
defgrad = local_deformation(sc.pos)
os.makedirs(args.out_dir, exist_ok=True)

for case in args.cases:
    kind, idx = case.split(":")
    idx = int(idx)
    truth = blob["ref"][kind][idx][: args.frames + 1]
    force = blob["force"][kind][idx]
    span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    print(f"\n[case] {kind} #{idx}, MPM moved {float(span):.4f}")

    runs = {"MPM": truth}
    if args.unfitted:
        # the same interface the fitted simulators expose, on the sampled
        # anchors scene_setup built, with nothing learned
        p_, v_, gp_ = (sc.anchor_canonical.clone(), sc.initial_velocity(force),
                       sc.pos.clone())
        out = [gp_[mat].clone()]
        alive = True
        for _ in tqdm(range(args.frames), desc="  unfitted", ncols=90):
            p_, v_, gp_ = sc.explicit_step(p_, v_, gp_, args.dt_mult)
            if alive and not torch.isfinite(p_).all():
                print("  the unfitted simulator went non-finite; it is held at "
                      "its last finite state from here so the panel stays legible")
                alive = False
            if alive:
                gp_ = sc.skin(p_, gp_)
            out.append(gp_[mat].clone())
        runs["unfitted"] = torch.stack(out)
    for name, fs in FITS:
        # exactly what eval_vs_mpm.py does: the anchors start canonical, the
        # impulse becomes an anchor velocity, and skinning is a separate call --
        # explicit_step hands the cloud back untouched
        p, v, gp = fs.anchor_canonical.clone(), fs.initial_velocity(force), fs.pos.clone()
        out = [gp[mat].clone()]
        for _ in tqdm(range(args.frames), desc=f"  {name}", ncols=90):
            p, v, gp = fs.explicit_step(p, v, gp, args.dt_mult)
            gp = fs.skin(p, gp)
            out.append(gp[mat].clone())
        runs[name] = torch.stack(out)

    names = ["MPM"] + (["unfitted"] if args.unfitted else []) + [n for n, _ in FITS]
    err = {n: 0.0 for n in names}
    vis = {n: {"psnr": [], "ssim": [], "lpips": []} for n in names[1:]}
    W, H = args.width, args.height
    vid = []
    for k in tqdm(range(args.frames + 1), desc="  render", ncols=90):
        row = []
        imgs = {}
        for n in names:
            x = runs[n][k]
            if n != "MPM":
                e = float((x - truth[k]).norm(dim=-1).mean() / span) * 100
                err[n] = (err[n] * k + e) / (k + 1)
            xf = full(x)
            img = frame(xf, defgrad(xf),
                        hide=rest if args.only_material else None)
            imgs[n] = img
            cap = n if n == "MPM" else f"{n}   err {err[n]:.2f}%"
            row.append(label(img, cap))
        if args.metrics:
            ref = to_t(imgs["MPM"])
            for n in names[1:]:
                cur = to_t(imgs[n])
                vis[n]["psnr"].append(psnr(cur, ref))
                vis[n]["ssim"].append(float(ssim_fn(cur[None], ref[None])))
                if LPIPS is not None:
                    with torch.no_grad():
                        vis[n]["lpips"].append(float(
                            LPIPS(cur[None] * 2 - 1, ref[None] * 2 - 1)))
        vid.append(np.concatenate(row, axis=1))
    path = os.path.join(args.out_dir, f"{kind}{idx}_" + "_vs_".join(names[1:]) + ".mp4")
    imageio.mimsave(path, vid, fps=args.fps, quality=8)
    print(f"  -> {path}")
    for n in names[1:]:
        print(f"     {n:24} {err[n]:.2f}%")
    if args.metrics:
        print(f"\n     {'panel':24} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}"
              "   (MPM 의 렌더와 비교)")
        for n in names[1:]:
            v = vis[n]
            lp = sum(v["lpips"]) / len(v["lpips"]) if v["lpips"] else float("nan")
            print(f"     {n:24} {sum(v['psnr']) / len(v['psnr']):7.2f} "
                  f"{sum(v['ssim']) / len(v['ssim']):7.4f} {lp:7.4f}")
