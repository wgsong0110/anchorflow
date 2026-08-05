"""Can the model fit anything at all?

A 2.9M-parameter network stalls at a 0.16 relative error on 60 training pairs,
and the lagged-cloud hypothesis was measured and ruled out (a cloud differing by
60% of a step's displacement changes the answer by 0.04%). Capacity, data and
hidden state are all eliminated, which leaves the model or the training code.

So strip the problem to one sample and three targets of increasing difficulty:

  * v * dt        -- readable straight off an input channel; anything that
                     cannot fit this is broken, not undertrained.
  * a smooth random field over the anchors -- no relation to the inputs, so this
                     is pure memorisation capacity for one sample.
  * the real displacement.

If the first two fit and the third does not, the target is genuinely hard. If
none of them fit, the fault is in the model or the loop.
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
from anchorflow.nextstate import NextStep

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--warm", type=int, default=10)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = args.dt_mult * sc.sub_dt

p, v = AC.clone(), sc.initial_velocity()
gp = sc.pos.clone()
for _ in tqdm(range(args.warm), desc="warm", ncols=80, leave=False):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
p_prev = p.clone()
p2, _, gp2 = sc.explicit_step(p.clone(), v.clone(), gp.clone(), args.dt_mult)

P = p.clone()
V = (p2 - p) / dt * 0 + v.clone()          # the state's own velocity
A = sc.elastic_accel(P, gp)
DU = p2 - p
DISP = float(DU.norm(dim=-1).mean())
VEL = float(V.norm(dim=-1).mean())
ACC = float(A.norm(dim=-1).mean())
print(f"[state] |du|={DISP:.6f}  |v|={VEL:.4f}  |a|={ACC:.2f}")
print(f"[state] du/|du| spread across anchors: "
      f"min={DU.norm(dim=-1).min():.2e} max={DU.norm(dim=-1).max():.2e}")

smooth = torch.randn(sc.M, 3, device=dev)
for _ in range(6):                          # smooth it over neighbours in space
    d = torch.cdist(AC, AC)
    w = torch.softmax(-d / (0.05 * d.mean()), dim=-1)
    smooth = w @ smooth
smooth = smooth / smooth.norm(dim=-1).mean() * DISP

TARGETS = {
    "v * dt (an input channel)": V * dt,
    "smooth random field": smooth,
    "the real displacement": DU,
}

for name, target in TARGETS.items():
    torch.manual_seed(0)
    net = NextStep(args.hidden, args.depth, 4, DISP, VEL, ACC).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    tgt = torch.where(fixed.unsqueeze(-1), torch.zeros_like(target), target)
    trace = []
    for it in tqdm(range(1, args.iters + 1), desc=name[:24], ncols=90, leave=False):
        du = net(P, V, A, dt)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        loss = ((du - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it in (1, 100, 500, 1000, 2000, 4000, args.iters):
            with torch.no_grad():
                rel = ((du - tgt).norm() / tgt.norm()).item()
            trace.append((it, rel))
    print(f"\n{name}")
    print("   " + "  ".join(f"it{it}={r:.4f}" for it, r in trace))
