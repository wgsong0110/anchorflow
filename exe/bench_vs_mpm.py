"""What the chain actually costs, per coarse frame, against MPM itself.

Every speed number this project has quoted so far compared a stepper against
another stepper -- the learned step against the explicit substeps it replaces.
That is the right number for "was the network worth it" and the wrong one for
"is this faster than PhysGaussian", which is the question the method exists to
answer. MPM has never been in the same process as the thing replacing it.

So all of it here, on one device, in one process, over the same scene:

  MPM                    dt_mult substeps of p2g2p, ending in particle positions
  anchor simulator       the same interval, explicitly integrated, torch
  anchor simulator       the same, through the fused kernel
  learned stepper        one network call, amortised over the frames it covers

Two rows for the last two, because they answer different questions. Stepping
alone is what a rollout costs; stepping plus skinning is what a *frame* costs,
and only the second is like for like -- MPM moves the particles as it goes, so
its positions are free where ours have to be reconstructed from the anchors.

The explicit baseline is also reported at eleven substeps, which is as far down
as that scheme goes before it comes apart (exe/bench_explicit_dt.py). Forty is
what the config asks for and twenty-nine of them do nothing, so quoting a
speedup against forty flatters the result.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True, help="the fitted anchor set the student steps")
ap.add_argument("--ckpt", default=None, help="a trained student; without one only "
                                              "the simulators are timed")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--fair_substeps", type=int, default=11)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--repeat", type=int, default=20)
ap.add_argument("--render", action="store_true",
                 help="also time the rasteriser, from the camera the config specifies. "
                      "Physics on its own answers 'did the stepper get cheap'; only "
                      "this answers 'does a frame arrive in time', which is a different "
                      "question once the physics stops being the expensive part.")
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=2.2)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

import sparsestep
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import apply_step, net_from_ckpt

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
fs, _ = load_fitted(sc, args.fit, device=dev)
fit = fs.fit
cache = fit.prepare()
teacher = MPMTeacher(sc, n_grid=args.n_grid, sparse=fit)

print(f"[setup] {torch.cuda.get_device_name(0)}")
print(f"        {teacher.n} material particles on a {args.n_grid}^3 grid, "
      f"{fit.M} anchors, {fit.pair_g.shape[0]} pairs")
print(f"        one coarse frame = {args.dt_mult} substeps of dt={sc.sub_dt:g}")
print(f"        fused kernel: {'yes' if sparsestep.HAVE_CUDA else 'NO -- torch only'}")

# a deformed state, so nothing is timed at rest where branches are untypical
p0 = fit.pos.detach().clone()
v0 = torch.zeros_like(p0)
p_state, v_state, _ = fit.rollout(p0, v0, args.dt_mult, cache)
gp = sc.pos.clone()


def timeit(fn, repeat=None):
    repeat = repeat or args.repeat
    for _ in range(args.warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(repeat):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeat


# ---- MPM -------------------------------------------------------------------
x0, vp0, F0, C0 = teacher.lift(p_state, v_state)


def mpm_frame():
    teacher._set(x0, vp0, F0, C0)
    for _ in range(args.dt_mult):
        teacher.solver.p2g2p(None, sc.sub_dt, device=teacher.wp_dev)
    teacher.solver.export_particle_x_to_torch()


ms_mpm = timeit(mpm_frame)

# ---- the anchor simulator, both paths --------------------------------------
def sim_steps(n, fused):
    saved = sparsestep.HAVE_CUDA

    def run():
        sparsestep.HAVE_CUDA = saved and fused
        try:
            fit.rollout(p_state, v_state, n, cache)
        finally:
            sparsestep.HAVE_CUDA = saved
    return run


def skin_only(fused):
    """the WHOLE cloud, not just the material.

    gaussian_pos returns the simulated particles; a frame also needs the
    opacity-rejected sixth carried along, and FittedScene.skin is what does
    both. Timing only the first understated what a drawn frame costs.
    """
    saved = sparsestep.HAVE_CUDA

    def run():
        sparsestep.HAVE_CUDA = saved and fused
        try:
            fs.skin(p_state, gp)
        finally:
            sparsestep.HAVE_CUDA = saved
    return run


ms_sim_torch = timeit(sim_steps(args.dt_mult, False))
ms_skin_torch = timeit(skin_only(False))
if sparsestep.HAVE_CUDA:
    ms_sim_fused = timeit(sim_steps(args.dt_mult, True))
    ms_skin_fused = timeit(skin_only(True))
    ms_fair_fused = timeit(sim_steps(args.fair_substeps, True))
else:
    ms_sim_fused = ms_skin_fused = ms_fair_fused = float("nan")

# ---- the student -----------------------------------------------------------
ms_net = ms_student = float("nan")
chunk = 1
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    t = ck["args"]
    chunk = t.get("chunk", 1)
    use_a = not t.get("no_accel", False)
    net = net_from_ckpt(ck, dev)
    dt = t["dt_mult"] * sc.sub_dt
    v_c = v_state.clone()

    def student_call():
        a = fs.elastic_accel(p_state, gp) if use_a else None
        apply_step(net, p_state, v_c, a, dt, fs.fixed_mask)

    ms_net = timeit(student_call) / chunk
    ms_student = ms_net + (ms_skin_fused if sparsestep.HAVE_CUDA else ms_skin_torch)
    print(f"        student: chunk {chunk}, "
          f"{'with' if use_a else 'without'} the acceleration input")


# ---- the rasteriser --------------------------------------------------------
#
# Same camera as exe/render_fitted_vs_mpm.py builds, so the number is for the
# view this scene is actually looked at from rather than an arbitrary one. It is
# charged to both sides equally -- MPM and the student draw the same cloud from
# the same place -- which is the point: it is the term that does not shrink.
ms_render = float("nan")
if args.render:
    import math

    import numpy as np
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as _render_scgs
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    class MiniCam:
        def __init__(self, W, H, fovy, fovx, zn, zf, wvt, fpt):
            self.image_width, self.image_height = W, H
            self.FoVy, self.FoVx = fovy, fovx
            self.znear, self.zfar = zn, zf
            self.world_view_transform = wvt
            self.full_proj_transform = fpt
            self.camera_center = wvt.inverse()[3, :3]

    class _P:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    gaussians = GaussianModel(3, fea_dim=0)
    gaussians.load_ply(args.ply)
    pipe, bg = _P(), torch.tensor([1., 1., 1.], device=dev)
    d_rot = torch.zeros(sc.N, 4, device=dev); d_rot[:, 0] = 1.
    d_sc = torch.zeros(sc.N, 3, device=dev)
    cfg = sc.cfg
    center = sc.undo(torch.tensor(cfg["mpm_space_viewpoint_center"],
                                   device=dev).unsqueeze(0))[0].cpu().numpy()
    up_mpm = (torch.tensor(cfg["mpm_space_vertical_upward_axis"], device=dev)
              + torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev)).unsqueeze(0)
    up = sc.undo(up_mpm)[0].cpu().numpy() - center
    up /= (np.linalg.norm(up) + 1e-9)
    xw = sc.xyz_world[sc.keep]
    extent = float((xw.max(0).values - xw.min(0).values).norm())
    az, el = math.radians(cfg["init_azimuthm"]), math.radians(cfg["init_elevation"])
    tmp = np.array([1., 0., 0.]) if abs(np.dot(np.array([1., 0., 0.]), up)) < 0.9 \
        else np.array([0., 1., 0.])
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
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                               fovY=fovy).transpose(0, 1).to(dev)
    cam = MiniCam(args.width, args.height, fovy, fovx, 0.01, 100.0, wvt,
                   (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))
    d_xyz = torch.zeros_like(sc.xyz_world)

    def draw():
        _render_scgs(cam, gaussians, pipe, bg, d_xyz, d_rot, d_sc, d_rot_as_res=True)

    ms_render = timeit(draw)
    print(f"        rasteriser: {sc.N} Gaussians at {args.width}x{args.height}")


# ---- report ----------------------------------------------------------------
def row(name, ms, note=""):
    sp = f"{ms_mpm / ms:8.1f}x" if ms == ms and ms > 0 else f"{'--':>9}"
    fps = f"{1000.0 / ms:9.1f}" if ms == ms and ms > 0 else f"{'--':>9}"
    print(f"  {name:<44} {ms:8.3f} ms {sp} {fps} fps  {note}")


print(f"\n{'':46}{'per frame':>11} {'vs MPM':>9} {'':>9}")
row("PhysGaussian MPM", ms_mpm, f"{args.dt_mult} substeps")
print()
row("anchor sim, torch", ms_sim_torch, f"{args.dt_mult} substeps")
row("anchor sim, fused kernel", ms_sim_fused, f"{args.dt_mult} substeps")
row("anchor sim, fused, at the stability limit", ms_fair_fused,
    f"{args.fair_substeps} substeps")
print()
row("skinning alone, torch", ms_skin_torch)
row("skinning alone, fused kernel", ms_skin_fused)
if args.ckpt:
    print()
    row("learned step, anchors only", ms_net, f"chunk {chunk}")
    row("learned step + skinning (a drawn frame)", ms_student)

if args.render:
    print()
    row("rasteriser alone", ms_render, f"{args.width}x{args.height}")
    print(f"\n  {'end to end, a frame on screen':<44}")
    row("  MPM + rasteriser", ms_mpm + ms_render)
    if args.ckpt:
        row("  learned step + skinning + rasteriser", ms_student + ms_render)
        budget = 1000.0 / 60.0
        print(f"\n[budget] at 60 fps a frame has {budget:.1f} ms. MPM's physics alone "
              f"is {ms_mpm / budget:.1f}x that;\n         the learned step is "
              f"{100 * ms_student / budget:.1f}% of it, and the rasteriser "
              f"{100 * ms_render / budget:.1f}%.")

if sparsestep.HAVE_CUDA and ms_sim_torch == ms_sim_torch:
    print(f"\n[kernel] the fused path is {ms_sim_torch / ms_sim_fused:.2f}x the torch "
          f"path on the substeps and {ms_skin_torch / ms_skin_fused:.2f}x on skinning")
print("\n[note] 'vs MPM' divides MPM's frame by that row's. The rows above the gap\n"
      "       all end in the same thing -- the material's positions after one coarse\n"
      "       frame -- except the anchor-only rows, which stop at the anchors.")
