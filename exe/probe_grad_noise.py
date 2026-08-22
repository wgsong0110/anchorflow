"""Is the gradient noisy because the samples differ, or because the arithmetic does?

The fit stops improving after about forty iterations and then walks the
parameters somewhere worse. The loss is noisy enough to explain that -- one
iteration draws two trajectories at two random frames, and the raw loss ranges
over 0.06 to 4.19 -- but so is 480 substeps of float32 with a checkpointed
backward: rollout() wraps every substep in torch.utils.checkpoint, so the
backward RE-RUNS the forward, and index_add_ on CUDA reduces with atomics, whose
order is not reproducible. A gradient taken at a slightly different state than
the one the loss was measured at is noise that no batch size can average away.

The two are separated by taking the gradient twice from the SAME sample:

  same sample, twice     any disagreement is arithmetic. Should be cosine 1.
  different samples      disagreement is what a bigger batch would average.
  checkpointing off      the same sample without the recompute, if it fits.

Reported as cosines between gradient vectors, per parameter, because that is
what the optimiser actually consumes -- the norm is thrown away by the clip.
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
ap.add_argument("--ckpt", default=None, help="fitted anchors; default the sampled set")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--unroll", type=int, default=12)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--n_traj", type=int, default=3)
ap.add_argument("--repeats", type=int, default=4, help="gradients per sample")
ap.add_argument("--samples", type=int, default=4, help="different samples")
ap.add_argument("--impulse_range", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=777)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

import warp as wp

from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.streams import draw_impulse

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")

fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor).to(dev)
fit.init_from_geometry()
if args.ckpt:
    b = torch.load(args.ckpt, map_location=dev, weights_only=False)
    fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                  b["log_k"].to(dev) if "log_k" in b else None)
    print(f"[setup] {os.path.basename(args.ckpt)} at iteration {b.get('iter')}, "
          f"{fit.M} anchors")
TRAIN = ("pos", "log_s", "quat", "log_k")
for n, prm in fit.named_parameters():
    prm.requires_grad_(n in TRAIN)


# ---- one trajectory set, so the samples are the ones the fit would draw -------
@torch.no_grad()
def mpm_traj(force):
    cache = fit.prepare()
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.to(torch.float16).cpu()]
    vs = [v0.to(torch.float16).cpu()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
        vs.append(T.solver.export_particle_v_to_torch().to(torch.float16).cpu())
    return torch.stack(xs), torch.stack(vs)


g = torch.Generator(device=dev); g.manual_seed(args.seed)
TRAJ = []
for _ in range(args.n_traj):
    f_ = draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
    t_ = mpm_traj(f_)
    if t_ is not None:
        TRAJ.append(t_)
print(f"[data] {len(TRAJ)} trajectories\n")


def one_gradient(X, V, t):
    """the loss the fit minimises, and the gradient it steps on"""
    for n in TRAIN:
        getattr(fit, n).grad = None
    cache = fit.prepare()
    w, rc, q, Binv, blocked, _ = cache
    p = fit.project(X[t].to(dev, torch.float32), cache)
    v = fit.project_v(V[t].to(dev, torch.float32), cache)
    loss = 0.0
    hi = min(args.unroll, X.shape[0] - 1 - t)
    for j in range(hi):
        p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        tgt = X[t + j + 1].to(dev, torch.float32)
        d = (tgt - X[t].to(dev, torch.float32)).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - tgt).norm(dim=-1).mean() / d
    (loss / hi).backward()
    return float(loss / hi), {n: getattr(fit, n).grad.detach().reshape(-1).clone()
                              for n in TRAIN}


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-30))


torch.set_grad_enabled(True)

print("=== 같은 표본, 같은 파라미터로 반복 ===  (1.0 이 아니면 수치 잡음)")
X, V = TRAJ[0]
t0 = 10
gs, ls = [], []
for r in range(args.repeats):
    l, gg = one_gradient(X, V, t0)
    gs.append(gg); ls.append(l)
print(f"  손실: " + "  ".join(f"{x:.6f}" for x in ls))
for n in TRAIN:
    cs = [cos(gs[0][n], gs[i][n]) for i in range(1, len(gs))]
    rel = [float((gs[i][n] - gs[0][n]).norm() / gs[0][n].norm().clamp(min=1e-30))
           for i in range(1, len(gs))]
    print(f"  {n:7} cos {min(cs):.6f}~{max(cs):.6f}   상대차 {min(rel):.2e}~{max(rel):.2e}")

print("\n=== 다른 표본 ===  (표집 잡음)")
gs2, ls2 = [], []
for i in range(args.samples):
    Xi, Vi = TRAJ[i % len(TRAJ)]
    ti = 5 + 11 * i
    l, gg = one_gradient(Xi, Vi, ti)
    gs2.append(gg); ls2.append(l)
print(f"  손실: " + "  ".join(f"{x:.4f}" for x in ls2))
for n in TRAIN:
    cs = [cos(gs2[i][n], gs2[j][n])
          for i in range(len(gs2)) for j in range(i + 1, len(gs2))]
    print(f"  {n:7} 쌍별 cos  중앙값 {sorted(cs)[len(cs)//2]:+.4f}   "
          f"{min(cs):+.4f} ~ {max(cs):+.4f}")

print("\n=== 체크포인팅 끄고, 같은 표본 ===")
try:
    fit.checkpoint_substeps = False
    gs3 = [one_gradient(X, V, t0)[1] for _ in range(2)]
    for n in TRAIN:
        print(f"  {n:7} cos(끔,끔) {cos(gs3[0][n], gs3[1][n]):.6f}   "
              f"cos(끔,켬) {cos(gs3[0][n], gs[0][n]):.6f}")
except torch.OutOfMemoryError:
    print("  메모리 부족 -- 480 서브스텝을 체크포인트 없이 들고 있을 수 없음")
finally:
    fit.checkpoint_substeps = True
