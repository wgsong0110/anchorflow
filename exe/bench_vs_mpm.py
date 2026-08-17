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
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

import sparsestep
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import NextStep, apply_step

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
    saved = sparsestep.HAVE_CUDA

    def run():
        sparsestep.HAVE_CUDA = saved and fused
        try:
            fit.gaussian_pos(p_state, cache)
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
    net = NextStep(t["hidden"], t["depth"], t["heads"], ck["disp_scale"],
                    ck["vel_scale"], ck["acc_scale"], use_accel=use_a,
                    chunk=chunk).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    dt = t["dt_mult"] * sc.sub_dt
    v_c = v_state.clone()

    def student_call():
        a = fs.elastic_accel(p_state, gp) if use_a else None
        apply_step(net, p_state, v_c, a, dt, fs.fixed_mask)

    ms_net = timeit(student_call) / chunk
    ms_student = ms_net + (ms_skin_fused if sparsestep.HAVE_CUDA else ms_skin_torch)
    print(f"        student: chunk {chunk}, "
          f"{'with' if use_a else 'without'} the acceleration input")


# ---- report ----------------------------------------------------------------
def row(name, ms, note=""):
    sp = f"{ms_mpm / ms:8.1f}x" if ms == ms and ms > 0 else f"{'--':>9}"
    print(f"  {name:<44} {ms:8.3f} ms {sp}   {note}")


print(f"\n{'':46}{'per frame':>11} {'vs MPM':>9}")
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

if sparsestep.HAVE_CUDA and ms_sim_torch == ms_sim_torch:
    print(f"\n[kernel] the fused path is {ms_sim_torch / ms_sim_fused:.2f}x the torch "
          f"path on the substeps and {ms_skin_torch / ms_skin_fused:.2f}x on skinning")
print("\n[note] 'vs MPM' divides MPM's frame by that row's. The rows above the gap\n"
      "       all end in the same thing -- the material's positions after one coarse\n"
      "       frame -- except the anchor-only rows, which stop at the anchors.")
