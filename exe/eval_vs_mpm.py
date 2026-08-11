"""Every stepper against PhysGaussian's MPM, on one ruler.

The project's numbers have always been measured against the anchor simulator,
in anchor space. That answers "did the student learn its teacher" and not "is
this PhysGaussian", and the two differ by however far the teacher is -- 5.5%.
A student trained on MPM directly cannot be compared on the old ruler at all,
since its teacher is a different trajectory.

So: MPM's own particles, every frame, for everything. Students are rolled out
in anchor space and skinned out to Gaussians, which is what a renderer would
draw. The anchor simulator is included as the teacher one of them imitates, and
the projection floor as what 512 anchors can represent of MPM at all -- no
stepper on this state, learned or not, can beat it.

Held-out impulses only, in the three families the project has used: the config's
own uniform push, smooth force fields, and localised pokes.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.nextstate import NextStep, apply_step
from anchorflow.streams import draw_impulse, rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_uniform", type=int, default=8)
ap.add_argument("--n_field", type=int, default=8)
ap.add_argument("--n_poke", type=int, default=8)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=20260811)
ap.add_argument("--cache", default=None, help="where to keep the MPM references")
ap.add_argument("--per_traj", action="store_true")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = args.dt_mult * sc.sub_dt
T = MPMTeacher(sc)
mat = T.mat
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
print(f"[setup] {sc.M} anchors, {T.n} material particles, eig_floor {args.eig_floor}")


def forces():
    """the three impulse families, drawn the way every evaluation here draws them"""
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    out = {"uniform": [base], "field": [], "poke": []}
    for _ in range(args.n_uniform - 1):
        s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=g).item())
        out["uniform"].append((rand_rot(g, dev) @ base) * s)
    for _ in range(args.n_field):
        f, _ = draw_impulse(sc, base, g, args.impulse_range, field=True)
        out["field"].append(f)
    for _ in range(args.n_poke):
        rad = sc.sim.radius * (2.0 ** (1.0 + 2.0 * torch.rand(1, device=dev, generator=g).item()))
        pf, _, _ = sc.random_poke(g, rad, base.norm().item())
        out["poke"].append(pf)
    return out


FORCE = forces()
key = f"{args.seed}_{args.n_uniform}_{args.n_field}_{args.n_poke}_{args.frames}_{args.dt_mult}"
REF = None
if args.cache and os.path.exists(args.cache):
    blob = torch.load(args.cache, map_location=dev, weights_only=False)
    if blob.get("key") == key:
        REF = blob["ref"]
        print(f"[ref] {args.cache}")
if REF is None:
    REF = {}
    for kind, fs in FORCE.items():
        runs = []
        for f in tqdm(fs, desc=f"MPM {kind}", ncols=90):
            dv = sc.impulse_dv(f)
            v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
            T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
            xs = [T.pos_m.clone()]
            for _ in range(args.frames):
                for _ in range(args.dt_mult):
                    T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
                xs.append(T.solver.export_particle_x_to_torch().clone())
            runs.append(torch.stack(xs))
        REF[kind] = runs
    if args.cache:
        torch.save({"ref": REF, "key": key}, args.cache)
        print(f"[ref] cached to {args.cache}")


def score(pred, truth):
    """mean particle error over every frame, as a fraction of MPM's own motion"""
    span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    e = (pred - truth).norm(dim=-1).mean(-1) / span
    return e.mean().item(), e[-1].item(), \
        ((pred - pred[0]).norm(dim=-1).max() / span).item()


def run_anchor(force):
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            return None
        out.append(gp[mat].clone())
    return torch.stack(out)


def run_floor(truth):
    """MPM's own trajectory, squeezed through the anchor state and back"""
    return torch.stack([sc.skin(T.project(truth[t]), sc.pos.clone())[mat]
                        for t in range(truth.shape[0])])


def run_net(net, force):
    p, v = AC.clone(), sc.initial_velocity(force)
    gp = sc.pos.clone()
    out = [gp[mat].clone()]
    k = 0
    while k < args.frames:
        for q, d in apply_step(net, p, v, None, dt, fixed):
            p, v = q, d / dt
            if not torch.isfinite(p).all():
                return None
            gp = sc.skin(p, gp)
            out.append(gp[mat].clone())
            k += 1
            if k >= args.frames:
                break
    return torch.stack(out)


rows = [("MPM projected (floor)", None), ("anchor simulator", None)]
for path in args.ckpt:
    ck = torch.load(path, map_location=dev, weights_only=False)
    ta = ck["args"]
    if not ta.get("no_accel", False):
        raise SystemExit(f"{path} was trained with the acceleration input; "
                         f"this evaluation does not feed it")
    net = NextStep(ta["hidden"], ta["depth"], ta["heads"], ck["disp_scale"],
                   ck["vel_scale"], ck["acc_scale"], use_accel=False,
                   chunk=ta.get("chunk", 1)).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    tag = os.path.basename(path).replace(".pt", "")
    rows.append((f"{tag} [{ta.get('teacher', 'anchor')}]", net))

print(f"\n{'':>28}" + "".join(f" | {k + ' mean':>14} {'final':>7} {'amp':>6} {'lost':>5}"
                              for k in REF))
for name, net in rows:
    line = f"{name:>28}"
    for kind, truth_runs in REF.items():
        ms, fs, amps, lost = [], [], [], 0
        for i, truth in enumerate(truth_runs):
            if net is not None:
                pred = run_net(net, FORCE[kind][i])
            elif name.startswith("MPM"):
                pred = run_floor(truth)
            else:
                pred = run_anchor(FORCE[kind][i])
            if pred is None:
                lost += 1
                continue
            m, f, a = score(pred, truth)
            ms.append(m); fs.append(f); amps.append(a)
        if ms:
            t = torch.tensor(ms)
            line += f" | {100*t.mean():8.2f}+-{100*t.std():4.1f} " \
                    f"{100*sum(fs)/len(fs):6.2f}% {100*sum(amps)/len(amps):5.0f}% {lost:5d}"
        else:
            line += f" | {'all lost':>14} {'':>7} {'':>6} {lost:5d}"
    print(line, flush=True)

if args.per_traj:
    print(f"\n[per trajectory] mean error, % of MPM's own peak displacement")
    for kind, truth_runs in REF.items():
        print(f"\n  {kind}")
        print(f"    {'#':>3} " + "".join(f"{n[:22]:>24}" for n, _ in rows))
        for i, truth in enumerate(truth_runs):
            line = f"    {i:>3} "
            for name, net in rows:
                pred = run_net(net, FORCE[kind][i]) if net is not None else (
                    run_floor(truth) if name.startswith("MPM") else run_anchor(FORCE[kind][i]))
                line += f"{'diverged':>24}" if pred is None else \
                    f"{100*score(pred, truth)[0]:23.2f}%"
            print(line)

print(f"\n[note] every number is against MPM's own particles. The floor row is MPM's\n"
      f"       trajectory reduced to 512 anchors and skinned back, which no stepper\n"
      f"       on this state can beat.")
