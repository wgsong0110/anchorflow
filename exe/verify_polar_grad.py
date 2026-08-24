"""Is the closed-form polar derivative the same one the iteration gives?

polar_R now returns its gradient from a 3x3 solve instead of differentiating six
Newton steps. That is worth 22 ms of the 50 ms a force costs with its backward,
and it is only worth taking if the two agree.

Checked three ways, because each catches something the others do not:

  against the iteration   autograd through the Newton steps, same inputs. This
                          is the gradient the fit has been using, so agreement
                          here means nothing downstream changes.
  against finite          a directional derivative of the loss, computed in
  differences             float64 with the step halved until two estimates
                          agree. Catches an error the two analytic paths could
                          share.
  at rest                 F = I, where S = I and the system is 2 I w = c. The
                          simulator spends most of its time here and this is
                          where an eigen-based derivative would divide by zero.
"""
from __future__ import annotations

import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.anchor_fit import polar_R

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def cases(n=4096):
    eye = torch.eye(3, device=dev).expand(n, 3, 3)
    out = {
        "rest (F = I)": eye.clone(),
        "near rest (1e-4)": eye + 1e-4 * torch.randn(n, 3, 3, device=dev),
        "small strain (1e-2)": eye + 1e-2 * torch.randn(n, 3, 3, device=dev),
        "large strain (0.3)": eye + 0.3 * torch.randn(n, 3, 3, device=dev),
    }
    # a rotation times a stretch, which is what the simulator actually sees
    a = torch.randn(n, 3, device=dev) * 0.7
    c, s = torch.cos(a[:, :1]), torch.sin(a[:, :1])
    Rz = torch.zeros(n, 3, 3, device=dev)
    Rz[:, 0, 0] = c[:, 0]; Rz[:, 0, 1] = -s[:, 0]
    Rz[:, 1, 0] = s[:, 0]; Rz[:, 1, 1] = c[:, 0]
    Rz[:, 2, 2] = 1.0
    st = eye + 0.05 * torch.randn(n, 3, 3, device=dev)
    out["rotation x stretch"] = Rz @ st
    return out


def grads(F, analytic, weight):
    Fx = F.detach().clone().requires_grad_(True)
    R = polar_R(Fx, 6, 1e-6, analytic=analytic)
    (R * weight).sum().backward()
    return Fx.grad.detach().clone()


print(f"{'case':22} {'max rel diff':>13} {'cos':>9}   (해석 vs 반복 미분)")
W = None
for name, F in cases().items():
    if W is None or W.shape != F.shape:
        W = torch.randn_like(F)
    ga = grads(F, True, W)
    gi = grads(F, False, W)
    d = (ga - gi).reshape(ga.shape[0], -1).norm(dim=-1)
    s = gi.reshape(gi.shape[0], -1).norm(dim=-1).clamp(min=1e-30)
    cs = float((ga * gi).sum() / (ga.norm() * gi.norm()).clamp(min=1e-30))
    print(f"{name:22} {float((d / s).max()):13.2e} {cs:9.6f}")

# ---- finite differences, in float64 ----------------------------------------
print("\n방향 유한차분 (float64):")
n = 256
eye = torch.eye(3, device=dev, dtype=torch.float64).expand(n, 3, 3)
for tag, F0 in (("near rest", eye + 1e-4 * torch.randn(n, 3, 3, device=dev, dtype=torch.float64)),
                 ("small strain", eye + 1e-2 * torch.randn(n, 3, 3, device=dev, dtype=torch.float64)),
                 ("large strain", eye + 0.3 * torch.randn(n, 3, 3, device=dev, dtype=torch.float64))):
    W64 = torch.randn_like(F0)
    D = torch.randn_like(F0)
    D = D / D.reshape(n, -1).norm(dim=-1).reshape(n, 1, 1)

    Fx = F0.clone().requires_grad_(True)
    (polar_R(Fx, 6, 1e-6, analytic=True) * W64).sum().backward()
    pred = float((Fx.grad * D).sum())

    prev, agreed = None, None
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        with torch.no_grad():
            hi = (polar_R(F0 + h * D, 6, 1e-6, analytic=False) * W64).sum()
            lo = (polar_R(F0 - h * D, 6, 1e-6, analytic=False) * W64).sum()
        est = float((hi - lo) / (2 * h))
        if prev is not None and abs(est - prev) < 1e-6 * max(1.0, abs(est)):
            agreed = est
            break
        prev = est
    got = agreed if agreed is not None else prev
    rel = abs(got - pred) / max(abs(got), 1e-30)
    print(f"  {tag:14} 해석 {pred:+.8e}   유한차분 {got:+.8e}   상대차 {rel:.2e}"
          f"   {'ok' if rel < 2e-4 else 'FAILED'}")
