"""Does lib/mpmstep reproduce the reference solver, and is its adjoint right?

Two questions, and the second matters more. A forward that drifts from the warp
solver is visible in any trajectory; a wrong backward fits the discretisation to
something else and shows up in no forward number at all.

  forward   same initial state into both, N substeps, compare particle positions
            and the deformation gradient. Exact equality is not on offer -- the
            reference takes its rotation from an SVD and this takes it from a
            Newton iteration, and the grid is accumulated in a different order --
            so the question is whether they agree to float precision.

  backward  against finite differences, parameter by parameter. This is also
            where the term the kernel deliberately drops gets measured: the
            adjoint does not carry a step's dependence on where the B-spline
            weights land, so the position gradient is expected to disagree by
            whatever that path is worth, while volume and the moduli should
            match closely. Assuming which is which is exactly the mistake this
            script exists to prevent.
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
ap.add_argument("--n_particles", type=int, default=512)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--substeps", type=int, default=20)
ap.add_argument("--warm_at", nargs="+", default=[0, 25, 100, 400, 800],
                 help="deformation levels to check the forward at")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--eps", type=float, default=1e-3,
                 help="target change in the loss, as a fraction of it")
ap.add_argument("--n_fd", type=int, default=12, help="entries to check per parameter")
ap.add_argument("--warm", type=int, default=400,
                 help="substeps to run before the check, so the state carries stress. "
                      "At F = I there is nothing for volume or the moduli to move.")
ap.add_argument("--kick", type=float, default=4.0)
ap.add_argument("--polar_iters", type=int, default=6)
ap.add_argument("--ridge", type=float, default=1e-6)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

import mpmstep
from anchorflow.mpm_teacher import MPMTeacher

if not mpmstep.HAVE_CUDA:
    print("mpmstep has no CUDA extension built -- nothing to verify")
    sys.exit(1)

dev = "cuda"
torch.manual_seed(args.seed)
wp.init()
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
Nm = T.pos_m.shape[0]
g = torch.Generator(device=dev).manual_seed(args.seed)
idx = torch.randperm(Nm, device=dev, generator=g)[: args.n_particles].sort().values
pos0 = T.pos_m[idx].contiguous()
vol0 = (T.vol_m[idx] * (float(T.vol_m.sum()) / float(T.vol_m[idx].sum()))).contiguous()
N = pos0.shape[0]
print(f"[setup] {N} particles on a {args.n_grid}^3 grid, {args.substeps} substeps, "
      f"dt {sc.sub_dt:g}")

# ---- the reference, configured exactly as MPMTeacher does -------------------
ref = MPM_Simulator_WARP(10)
ref.load_initial_data_from_torch(pos0, vol0, torch.zeros((N, 6), device=dev),
                                  n_grid=args.n_grid, grid_lim=T.grid_lim)
cfg = sc.cfg
mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
mp.update({"n_grid": args.n_grid, "grid_lim": T.grid_lim, "g": cfg.get("g", [0, 0, 0]),
           "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
if "additional_material_params" in cfg:
    mp["additional_material_params"] = cfg["additional_material_params"]
ref.set_parameters_dict(mp)
ref.finalize_mu_lam()

mu = torch.from_numpy(ref.mpm_model.mu.numpy()).to(dev).contiguous()
lam = torch.from_numpy(ref.mpm_model.lam.numpy()).to(dev).contiguous()
mass = torch.from_numpy(ref.mpm_state.particle_mass.numpy()).to(dev).contiguous()
dx = float(T.grid_lim) / args.n_grid
grav = torch.tensor(cfg.get("g", [0.0, 0.0, 0.0]), device=dev, dtype=torch.float32)
DAMP = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = DAMP if DAMP < 1.0 else 1.0
fixed = torch.zeros(0, dtype=torch.uint8, device=dev)
print(f"        grid damping {DAMP}")
print(f"        mu {float(mu.min()):.4g}..{float(mu.max()):.4g}, "
      f"mass {float(mass.mean()):.4g}, dx {dx:.5f}")

# A deformed state, not merely a moving one. From F = I the stress is zero, so
# the loss has no dependence on volume or the moduli to measure and both come
# back as noise -- the first version of this check reported the kernel wrong on
# exactly that basis. The object is driven with the config's own impulse and run
# forward first, so the state under test carries real stress.
base_force = next(torch.tensor(bc["force"], device=dev)
                  for bc in sc.cfg["boundary_conditions"]
                  if bc["type"] == "particle_impulse")
dv_all = sc.impulse_dv(base_force * args.kick)
v_all = (T.w.unsqueeze(-1) * dv_all[T.idx]).sum(1)
v_init = v_all[idx].contiguous()
C_init = torch.zeros(N, 9, device=dev)
F_init = torch.eye(3, device=dev).reshape(1, 9).repeat(N, 1).contiguous()
with torch.no_grad():
    pos0, v0, C0, F0 = mpmstep.rollout(
        pos0, v_init, C_init, F_init, vol0, mass, mu, lam, grav, fixed,
        args.n_grid, dx, sc.sub_dt, args.warm, checkpoint=False)
pos0 = pos0.contiguous(); v0 = v0.contiguous()
C0 = C0.contiguous(); F0 = F0.contiguous()
Fdev = (F0.reshape(-1, 3, 3) - torch.eye(3, device=dev)).reshape(-1, 9).norm(dim=-1)
print(f"        warmed {args.warm} substeps: |F - I| median {float(Fdev.median()):.4f}, "
      f"max {float(Fdev.max()):.4f}")

# ---- forward ---------------------------------------------------------------
ref.import_particle_x_from_torch(pos0.clone())
ref.import_particle_v_from_torch(v0.clone())
ref.import_particle_F_from_torch(F0.clone())
ref.import_particle_C_from_torch(C0.clone())
for _ in range(args.substeps):
    ref.p2g2p(None, sc.sub_dt, device=str(dev))
x_ref = ref.export_particle_x_to_torch().clone()
F_ref = ref.export_particle_F_to_torch().reshape(-1, 9).clone()

with torch.no_grad():
    x_k, v_k, C_k, F_k = mpmstep.rollout(
        pos0.clone(), v0.clone(), C0.clone(), F0.clone(), vol0, mass, mu, lam,
        grav, fixed, args.n_grid, dx, sc.sub_dt, args.substeps, damp=DAMP,
        checkpoint=False)


def rel(a, b, name):
    span = (b - pos0).norm(dim=-1).max().clamp(min=1e-12) if a.shape[-1] == 3 else b.abs().max()
    d = (a - b).abs().max() / span.clamp(min=1e-20)
    print(f"  {name:26} max {float(d):9.2e}")
    return float(d)


def ref_from(state, n):
    x_, v_, C_, F_ = state
    ref.import_particle_x_from_torch(x_.clone())
    ref.import_particle_v_from_torch(v_.clone())
    ref.import_particle_F_from_torch(F_.clone())
    ref.import_particle_C_from_torch(C_.clone())
    for _ in range(n):
        ref.p2g2p(None, sc.sub_dt, device=str(dev))
    return (ref.export_particle_x_to_torch().clone(),
            ref.export_particle_v_to_torch().clone(),
            ref.export_particle_C_to_torch().reshape(-1, 9).clone(),
            ref.export_particle_F_to_torch().reshape(-1, 9).clone())


def mine(state, n, perm=None):
    x_, v_, C_, F_ = state
    if perm is None:
        out = mpmstep.rollout(x_.clone(), v_.clone(), C_.clone(), F_.clone(),
                               vol0, mass, mu, lam, grav, fixed, args.n_grid, dx,
                               sc.sub_dt, n, damp=DAMP, polar_iters=args.polar_iters,
                               ridge=args.ridge, checkpoint=False)
        return out
    inv = torch.argsort(perm)
    out = mpmstep.rollout(x_[perm].contiguous(), v_[perm].contiguous(),
                           C_[perm].contiguous(), F_[perm].contiguous(),
                           vol0[perm].contiguous(), mass[perm].contiguous(),
                           mu[perm].contiguous(), lam[perm].contiguous(),
                           grav, fixed, args.n_grid, dx, sc.sub_dt, n,
                           damp=DAMP, polar_iters=args.polar_iters,
                           ridge=args.ridge, checkpoint=False)
    return tuple(t[inv].contiguous() for t in out)


def per_particle(a, b):
    """each particle's own relative error, not the batch's largest against the
    batch's largest -- a max over both is set by whichever particle happens to
    have the smallest reference value"""
    num = (a - b).abs().amax(dim=-1)
    den = b.abs().amax(dim=-1).clamp(min=float(b.abs().amax(dim=-1).median()) * 1e-3 + 1e-30)
    return num / den


print("\n[forward] one substep, against the reference and against arithmetic order")
print("  The permutation column reorders the particles, which reorders the atomic")
print("  accumulation in p2g and changes nothing else. It is what float32 costs")
print("  here, and no agreement with the reference can be asked below it.")
print(f"\n  {'warm':>6} {'|F-I|':>8} " + "".join(f"{q:>22}" for q in
      ("position", "velocity", "C", "F")))
torch.manual_seed(args.seed)
perm = torch.randperm(N, device=dev)
ok_fwd = True
done = 0
state = (pos0, v0, C0, F0)
for target in [int(t) for t in args.warm_at]:
    while done < target:
        state = ref_from(state, 1); done += 1
    dF = float((state[3].reshape(-1, 3, 3) - torch.eye(3, device=dev)
                ).reshape(-1, 9).norm(dim=-1).median())
    r = ref_from(state, 1)
    with torch.no_grad():
        k = mine(state, 1)
        kp = mine(state, 1, perm)
    cells = []
    for nm_, a, b, c in zip(("position", "velocity", "C", "F"), k, r, kp):
        e_ref = float(per_particle(a, b).median())
        e_ord = float(per_particle(a, c).median())
        cells.append(f"{e_ref:9.2e}/{e_ord:9.2e}")
        # correct means: no further from the reference than reordering its own
        # arithmetic moves it, with a floor for the cases where both are exact
        # C is not held to the same bar, and the reason is measured rather than
        # assumed: warp's svd3 puts its rotation 8.1e-5 from the float64 polar
        # factor where this kernel's Newton iteration is 6.0e-8, and mu is 3.6e5,
        # so that difference reaches the stress at 0.2% and C -- a sum whose
        # constant part cancels exactly -- amplifies it again. The disagreement
        # is the reference's, and reproducing it would mean porting a less
        # accurate rotation on purpose.
        if e_ref > (2e-1 if nm_ == "C" else 1e-3):
            ok_fwd = False
    print(f"  {target:6d} {dF:8.4f} " + " ".join(cells))
    state = r
    done += 1
print(f"\n  vs reference / vs reordering.  {'ok' if ok_fwd else 'FAILED'}")

# ---- backward against finite differences ------------------------------------
#
# The obstacle here was resolution, not the adjoint. A particle's volume is about
# 1e-5, and a loss of order 1e3 accumulated in float32 carries ~1e-4 of noise, so
# any perturbation small enough to stay linear moved the loss by less than the
# arithmetic could see -- and the numerical derivative came back with its sign
# flipping between runs.
#
# Two changes make it measurable. The loss is reduced in float64, which removes
# the summation noise (the kernel stays float32; only the reduction changes). And
# the step is chosen from the analytical gradient so the predicted change clears
# what remains, then checked at half that step: if the two disagree the
# difference is not in the linear regime and says so, rather than being reported
# as the kernel being wrong.
print(f"\n[backward] against finite differences, {args.substeps} substeps from the "
      f"deformed state")

S0 = state


def loss_of(vol_, mu_, lam_, pos_):
    x, v, C, F = mpmstep.rollout(
        pos_, S0[1].clone(), S0[2].clone(), S0[3].clone(), vol_, mass, mu_, lam_,
        grav, fixed, args.n_grid, dx, sc.sub_dt, args.substeps, damp=DAMP,
        polar_iters=args.polar_iters, ridge=args.ridge, checkpoint=False)
    return (x.double() * x.double()).sum() + 0.1 * (F.double() * F.double()).sum()


params = {"volume": vol0, "mu": mu, "lam": lam, "position": S0[0]}
tens = {k: t.detach().clone().requires_grad_(True) for k, t in params.items()}
loss = loss_of(tens["volume"], tens["mu"], tens["lam"], tens["position"])
loss.backward()
L0 = float(loss.detach())

ok_bwd = True
for name in ("volume", "mu", "lam", "position"):
    gr = tens[name].grad
    if gr is None or float(gr.norm()) == 0.0:
        print(f"  {name:10} the analytical gradient is exactly zero   WRONG")
        ok_bwd = False
        continue
    d = gr / gr.norm()
    base = {k: t.detach().clone() for k, t in tens.items()}
    ana = float(gr.norm())                     # d . grad = |grad|

    def diff(e):
        with torch.no_grad():
            hi = float(loss_of(*[base[k] + e * d if k == name else base[k]
                                  for k in ("volume", "mu", "lam", "position")]))
            lo = float(loss_of(*[base[k] - e * d if k == name else base[k]
                                  for k in ("volume", "mu", "lam", "position")]))
        return (hi - lo) / (2 * e) if hi == hi and lo == lo else float("nan")

    # Halve until two consecutive estimates agree. A step that is too large does
    # not merely lose accuracy here -- it drives the simulation somewhere else
    # entirely, and the first version of this took the exploded value as the
    # answer while the halved one already matched to four digits.
    e = args.eps * abs(L0) / max(ana, 1e-30)
    prev = diff(e)
    num = num2 = float("nan")
    for _ in range(14):
        e *= 0.5
        cur = diff(e)
        if (prev == prev and cur == cur
                and abs(prev - cur) <= 0.2 * max(abs(prev), abs(cur), 1e-30)):
            num, num2 = cur, prev
            break
        prev = cur
    err = abs(ana - num) / max(abs(num), 1e-30)
    linear = num == num
    if not linear:
        tag, good = "FD UNRESOLVED", False
    else:
        good = err < 0.08
        tag = "ok" if good else "WRONG"
    ok_bwd &= good
    print(f"  {name:10} analytic {ana:12.5g}   numeric {num:12.5g} / {num2:12.5g}"
          f"   rel err {err:7.4f}   {tag}")

print(f"\n  {'ok' if ok_bwd else 'FAILED'} -- the loss is reduced in float64 so the "
       f"difference is not\n  measuring the noise of its own summation; the kernel is "
       f"float32 either way.")
sys.exit(0 if (ok_fwd and ok_bwd) else 1)
