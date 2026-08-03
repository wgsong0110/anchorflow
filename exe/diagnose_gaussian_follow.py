"""Find which Gaussians fail to follow the anchors during an explicit rollout.

Two distinct ways a Gaussian can appear frozen in a render:

  (a) it never entered the simulation at all -- the opacity filter dropped it,
      so the renderer leaves its d_xyz at zero and it stays pinned at its
      canonical position forever (a literal afterimage);
  (b) it IS simulated, but its skinning source (the weighted centroid of its K
      anchors, plus F) barely moves while other parts of the object do.

These need opposite fixes, so this script separates them: it reports the
dropped count and their spatial overlap with the moving region, and for the
kept Gaussians it compares each one's displacement against the displacement of
the anchors actually driving it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--steps", type=int, default=4000)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))

g = GaussianModel(3, fea_dim=0)
g.load_ply(args.ply)
xyz_all = g.get_xyz.detach().clone()
op = g.get_opacity.detach().clone()
keep = op[:, 0] > cfg["opacity_threshold"]
n_all = xyz_all.shape[0]
xyz_w = xyz_all[keep]
pmin, pmax = xyz_w.min(0).values, xyz_w.max(0).values
mid = (pmin + pmax) / 2
scale = 1.0 / (pmax - pmin).max()
to_mpm = lambda p: (p - mid) * scale + torch.tensor([1., 1., 1.], device=dev)
POS = to_mpm(xyz_w).contiguous()
POS_DROP = to_mpm(xyz_all[~keep])
N = POS.shape[0]

print(f"[opacity] threshold={cfg['opacity_threshold']}")
print(f"[opacity] kept {N}/{n_all}  dropped {n_all - N}  ({100*(n_all-N)/n_all:.1f}%)")
print(f"[opacity] dropped opacity: max={op[~keep,0].max():.4f} mean={op[~keep,0].mean():.4f}")

ng, grid_lim = 100, 2.0
dx = grid_lim / ng
vi = (POS / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(N, device=dev))
VOL = ((dx ** 3) / cnt[flat]).contiguous()
E_p = torch.full((N,), float(cfg["E"]), device=dev)
nu_p = torch.full((N,), float(cfg["nu"]), device=dev)
dens_p = torch.full((N,), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev); s = torch.tensor(reg["size"], device=dev)
    inside = ((POS - c).abs() <= s).all(-1)
    E_p[inside] = reg["E"]; nu_p[inside] = reg["nu"]; dens_p[inside] = reg["density"]
MU, LAM = lame_from_E_nu(E_p, nu_p)
mass_p = dens_p * VOL

aset, _ = AnchorSet.from_gaussians(POS, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
AC = aset.canonical.clone().contiguous(); M = AC.shape[0]
sim = AnchorElasticSim(POS, AC, K=args.K)
w0 = sim._weights(POS, AC)
MASS = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
fixed_mask = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed_mask |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
sub_dt = float(cfg["substep_dt"])
damping = float(cfg.get("grid_v_damping_scale", 1.0))

# does a dropped Gaussian sit inside the moving part of the object, or is it
# scattered background floaters? measure against the kept cloud's own extent
d_all = torch.cdist(POS_DROP, POS[torch.randperm(N, device=dev)[:20000]]).min(1).values
print(f"[dropped] distance to nearest kept Gaussian: "
      f"median={d_all.median():.4f} p90={d_all.quantile(0.9):.4f} max={d_all.max():.4f}"
      f"  (object spans 1.0 in MPM units)")

p_a, v_a = AC.clone(), torch.zeros(M, 3, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        force = torch.tensor(bc["force"], device=dev)
        wsum = torch.zeros(M, device=dev).index_add_(0, sim.nn_idx.reshape(-1), w0.reshape(-1))
        v_a = v_a + (wsum.unsqueeze(-1) * force.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
v_a[fixed_mask] = 0

gp = POS.clone()
gp_max = torch.zeros(N, device=dev)      # peak displacement over the whole run
ap_max = torch.zeros(M, device=dev)
for i in tqdm(range(args.steps), desc="explicit", ncols=90):
    p_a, v_a, gp, _F = sim.step(p_a, v_a, MASS, gp, VOL, MU, LAM, sub_dt,
                                 gravity=(gravity if gravity.abs().sum() > 0 else None),
                                 damping=damping, fixed_mask=fixed_mask)
    if torch.isnan(p_a).any():
        print(f"[sim] NaN at {i}"); break
    gp_max = torch.maximum(gp_max, (gp - POS).norm(dim=-1))
    ap_max = torch.maximum(ap_max, (p_a - AC).norm(dim=-1))

print(f"\n[anchors] peak displacement: max={ap_max.max():.5f} median={ap_max.median():.5f} "
      f"pinned={int(fixed_mask.sum())}/{M}")
print(f"[gaussians] peak displacement: max={gp_max.max():.5f} median={gp_max.median():.5f}")

# the driver of each Gaussian: the weighted mean of its anchors' peak motion,
# using the SAME weights the skinning uses. If a Gaussian's driver moved but
# the Gaussian didn't, the skinning is at fault; if neither moved, the Gaussian
# is simply in a stationary region (the pot) and is correct to stay put.
w = sim._weights(POS, AC)
drive = (w * ap_max[sim.nn_idx]).sum(-1)
thr = 0.1 * ap_max.max()
lag = (drive > thr) & (gp_max < 0.2 * drive)
print(f"[follow] Gaussians whose anchors clearly moved (>{thr:.4f}): {int((drive>thr).sum())}")
print(f"[follow]   of those, moved <20% of their anchors' motion: {int(lag.sum())}"
      f"  ({100*int(lag.sum())/max(1,int((drive>thr).sum())):.2f}%)")
if int(lag.sum()) > 0:
    q = torch.linspace(0, 1, 5, device=dev)
    print(f"[follow]   their gp/drive ratio quantiles: "
          f"{[f'{v:.3f}' for v in (gp_max[lag]/drive[lag]).quantile(q).tolist()]}")
    print(f"[follow]   their weight sum on pinned anchors: "
          f"median={(w * fixed_mask[sim.nn_idx].float()).sum(-1)[lag].median():.3f}")

# how much of the *rendered* image is frozen: dropped Gaussians vs moving ones
frozen_near_motion = int((d_all < 0.05).sum())
print(f"\n[verdict] frozen-by-opacity Gaussians sitting within 0.05 of the "
      f"simulated cloud: {frozen_near_motion}/{n_all - N}")
