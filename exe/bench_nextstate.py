"""Is one learned step actually cheaper than the substeps it replaces?

The point of predicting a coarse step is to skip explicit substeps, so the
comparison that matters is one network step against dt_mult of them, on the same
device, in the same process. Both sides of the learned step are counted: the
network forward, the elastic acceleration it takes as input, and the skinning
that produces the Gaussian cloud that acceleration is evaluated against.
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
ap.add_argument("--ckpt", required=True)
ap.add_argument("--warm", type=int, default=10)
ap.add_argument("--repeat", type=int, default=30)
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
targs = ck["args"]
sc = scene_setup.build(args.ply, args.config, targs["n_anchors"], targs["K"], device=dev,
                        frozen_weights=targs.get("frozen_weights", False))
USE_A = not targs.get("no_accel", False)
dt = targs["dt_mult"] * sc.sub_dt
net = NextStep(targs["hidden"], targs["depth"], targs["heads"],
                ck["disp_scale"], ck["vel_scale"], ck["acc_scale"],
                use_accel=USE_A).to(dev)
net.load_state_dict(ck["model"]); net.eval()
print(f"[setup] {torch.cuda.get_device_name(0)}  M={sc.M} N={sc.N}  "
      f"coarse step = {targs['dt_mult']} substeps")

p, v = sc.anchor_canonical.clone(), sc.initial_velocity()
gp = sc.pos.clone()
for _ in tqdm(range(args.warm), desc="warm", ncols=80, leave=False):
    p, v, gp = sc.explicit_step(p, v, gp, targs["dt_mult"])
v_c = (p - p) / dt + v


def timeit(fn, n=None):
    n = n or args.repeat
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def explicit():
    sc.explicit_step(p.clone(), v.clone(), gp.clone(), targs["dt_mult"])


def accel():
    sc.elastic_accel(p, gp)


def forward():
    net(p, v_c, sc.elastic_accel(p, gp) if USE_A else None, dt)


def skin():
    sc.skin(p, gp)


def learned():
    # without the acceleration input neither the force nor the skinning is on
    # the rollout's critical path: the cloud exists to evaluate the force
    # against, and is otherwise only needed when a frame is actually drawn
    a = sc.elastic_accel(p, gp) if USE_A else None
    du = net(p, v_c, a, dt)
    q = p + du
    if USE_A:
        sc.skin(q, gp)


ms_e = timeit(explicit)
ms_a = timeit(accel) if USE_A else 0.0
# forward() only evaluates the acceleration when the network takes it, so
# subtracting it unconditionally charged the no-accel path for work it never
# did and made the parts add up to more than the whole
ms_f = timeit(forward) - ms_a
ms_s = timeit(skin) if USE_A else 0.0
ms_l = timeit(learned)

print(f"\n{'':>34} {'ms':>8} {'share':>8}")
print(f"{'explicit substeps (the baseline)':>34} {ms_e:8.3f}")
print(f"{'  network forward':>34} {ms_f:8.3f} {100*ms_f/ms_l:7.1f}%")
if USE_A:
    print(f"{'  elastic acceleration (input)':>34} {ms_a:8.3f} {100*ms_a/ms_l:7.1f}%")
    print(f"{'  skinning (cloud for next step)':>34} {ms_s:8.3f} {100*ms_s/ms_l:7.1f}%")
else:
    print(f"{'  (no acceleration input, so no':>34}")
    print(f"{'   force and no skinning per step)':>34}")
print(f"{'one learned step (total)':>34} {ms_l:8.3f}")
print(f"\n[speedup] {ms_e / ms_l:.2f}x over the substeps it replaces")
print(f"[note] the learned step also needs the skinning the explicit path does "
      f"internally, so the two are counted on equal terms.")
