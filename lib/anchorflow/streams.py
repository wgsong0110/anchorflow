"""Training data from an object that keeps being hit, instead of restarts.

Every trajectory in the original scheme began at the canonical rest state, took
one impulse, and decayed. That covers a narrow set of states -- what a single
kick reaches before it dies out -- and the later steps of every trajectory look
alike, near rest. Running continuously and delivering impulses along the way
lets kicks land on an object that is still moving, which reaches configurations
no single impulse produces.

Lives here rather than in the training script so the renderer shows the data
the training actually sees, rather than a second implementation of it.
"""
from __future__ import annotations

import torch


def rand_rot(gen, device):
    q, r = torch.linalg.qr(torch.randn(3, 3, device=device, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def stream(sc, n_steps, dt_mult, base_force, gen, cap, impulse_every=20,
           impulse_range=4.0, keep_accel=True, on_step=None):
    """One continuous run. Returns (positions, accelerations, contaminated).

    contaminated[k] marks a step whose target carries an impulse the inputs
    cannot see: the kick lands between p_k and p_{k+1}, so no function of
    (p_k, v_k, a_k) reaches the answer. One sample per impulse, dropped. Every
    later step is fine -- by then the kick is in the positions, and the velocity
    is read off them.

    cap skips an impulse while the object is already displaced further than
    this; impulses land on top of each other by design, and the shape-matching
    fit degrades once the local neighbourhoods are far from their rest
    arrangement. Note this only gates the moment of delivery -- a kick landing
    just under the cap can still carry the object well past it.

    on_step(k, p, gp, fired) is called after each step, for a caller that wants
    to draw the run rather than train on it.
    """
    AC, fixed = sc.anchor_canonical, sc.fixed_mask
    dev = AC.device
    p, v = AC.clone(), torch.zeros(sc.M, 3, device=dev)
    gp = sc.pos.clone()
    ps, acc, bad = [p.clone()], [sc.elastic_accel(p, gp) if keep_accel else None], []
    due = 0
    for k in range(n_steps):
        fired = False
        if k >= due and base_force is not None:
            amp = (p - AC)[~fixed].norm(dim=-1).max().item()
            if amp < cap:
                s = 0.5 * (impulse_range ** torch.rand(1, device=dev, generator=gen).item())
                s = s if impulse_range > 1 else 1.0
                v = v + sc.impulse_dv((rand_rot(gen, dev) @ base_force) * s)
                fired = True
            jitter = 0.5 + torch.rand(1, device=dev, generator=gen).item()
            due = k + max(1, int(round(impulse_every * jitter)))
        bad.append(fired)
        p, v, gp = sc.explicit_step(p, v, gp, dt_mult)
        if not torch.isfinite(p).all():
            print(f"  [stream] diverged at step {k}, truncating", flush=True)
            bad.pop()
            break
        ps.append(p.clone())
        acc.append(sc.elastic_accel(p, gp) if keep_accel else None)
        if on_step is not None:
            on_step(k, p, gp, fired)
    out_a = torch.stack(acc) if keep_accel else None
    return torch.stack(ps), out_a, torch.tensor(bad, device=dev)
