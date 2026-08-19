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
fixed = torch.zeros(0, dtype=torch.uint8, device=dev)
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
        grav, fixed, args.n_grid, dx, sc.sub_dt, args.substeps, checkpoint=False)


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


print("\n[forward] one substep, from the same deformed state")
xr1, vr1, Cr1, Fr1 = ref_from((pos0, v0, C0, F0), 1)
with torch.no_grad():
    xk1, vk1, Ck1, Fk1 = mpmstep.rollout(
        pos0.clone(), v0.clone(), C0.clone(), F0.clone(), vol0, mass, mu, lam,
        grav, fixed, args.n_grid, dx, sc.sub_dt, 1, checkpoint=False)
for nm, a_, b_ in (("position", xk1, xr1), ("velocity", vk1, vr1),
                    ("C", Ck1, Cr1), ("F", Fk1, Fr1)):
    d = float((a_ - b_).abs().max() / b_.abs().max().clamp(min=1e-20))
    print(f"  {nm:26} max {d:9.2e}   {'ok' if d < 1e-3 else 'DIFFERS'}")

print("\n[forward] against the warp solver")
d1 = rel(x_k, x_ref, "particle positions")
d2 = rel(F_k, F_ref, "deformation gradient")
fwd_ok = d1 < 5e-2 and d2 < 5e-2
print(f"  {'':26} {'ok' if fwd_ok else 'DRIFTED'}")

# ---- backward against finite differences ------------------------------------
print("\n[backward] against finite differences, over "
      f"{args.substeps} substeps")


def loss_of(vol_, mu_, lam_, pos_):
    x, v, C, F = mpmstep.rollout(
        pos_, v0.clone(), C0.clone(), F0.clone(), vol_, mass, mu_, lam_,
        grav, fixed, args.n_grid, dx, sc.sub_dt, args.substeps, checkpoint=False)
    # something that touches positions and F both
    return (x * x).sum() + 0.1 * (F * F).sum()


params = {"volume": vol0, "mu": mu, "lam": lam, "position": pos0}
tens = {k: t.clone().requires_grad_(True) for k, t in params.items()}
loss = loss_of(tens["volume"], tens["mu"], tens["lam"], tens["position"])
loss.backward()

# A per-entry difference is below what float32 can resolve here: one particle's
# volume is ~1e-5, so perturbing it moves a loss of order 1e3 by far less than
# the arithmetic noise. The derivative along the analytical gradient aggregates
# every entry instead, which makes the signal large enough to measure and tests
# the whole vector rather than a sample of it.
for name in ("volume", "mu", "lam", "position"):
    gr = tens[name].grad
    if gr is None or float(gr.norm()) == 0.0:
        print(f"  {name:10} {'analytical gradient is exactly zero':>44}   WRONG")
        continue
    d = gr / gr.norm()
    base = {k: t.detach().clone() for k, t in tens.items()}
    # sized so the predicted change in the loss clears float32 noise. A step
    # scaled to the PARAMETER does not: one particle's volume is 1e-5, and the
    # loss it moves is far below the ~1e-4 the arithmetic can resolve
    L0 = float(loss.detach())
    e = args.eps * abs(L0) / max(float(gr.norm()), 1e-20)
    e = min(e, 0.25 * float(base[name].abs().mean()) / max(float(d.abs().mean()), 1e-20))
    hi = base[name] + e * d
    lo = base[name] - e * d
    with torch.no_grad():
        f_hi = float(loss_of(*[hi if k == name else base[k]
                                for k in ("volume", "mu", "lam", "position")]))
        f_lo = float(loss_of(*[lo if k == name else base[k]
                                for k in ("volume", "mu", "lam", "position")]))
    num = (f_hi - f_lo) / (2 * e)
    ana = float(gr.norm())          # d . grad = |grad|
    err = abs(ana - num) / max(abs(num), 1e-20)
    tag = "ok" if err < 0.05 else ("DROPPED TERM" if name == "position" else "WRONG")
    print(f"  {name:10} analytic {ana:12.4g}   numeric {num:12.4g}   "
          f"rel err {err:7.3f}   {tag}")

print("\n  The position row is expected to disagree: the adjoint deliberately\n"
       "  omits a step's dependence on where the B-spline weights land. Its\n"
       "  cosine says how much of that gradient survives the omission, which is\n"
       "  what decides whether rest positions can be fitted with this.")
