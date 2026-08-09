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


def draw_impulse(sc, base_force, gen, impulse_range=4.0, field=False,
                  sigma_lo=None, sigma_hi=None):
    """One impulse: how hard, which way, and -- with field -- where.

    Without field this is the config's own impulse rotated and rescaled: one
    vector on the whole object. With it, a smooth random force whose
    correlation length is drawn log-uniformly between the anchor spacing and
    the object's size, so a single dataset spans everything from bending one
    twig to the uniform shove that used to be the only case.
    """
    dev = sc.anchor_canonical.device
    s = 0.5 * (impulse_range ** torch.rand(1, device=dev, generator=gen).item())
    s = s if impulse_range > 1 else 1.0
    if not field:
        return (rand_rot(gen, dev) @ base_force) * s, None
    lo = sigma_lo if sigma_lo is not None else sc.sim.radius
    hi = sigma_hi if sigma_hi is not None else sc.extent
    u = torch.rand(1, device=dev, generator=gen).item()
    sigma = lo * ((hi / lo) ** u)
    return sc.random_force_field(gen, sigma, base_force.norm().item() * s), sigma


def draw_field_shape(sc, gen, sigma_lo=None, sigma_hi=None):
    """A unit-RMS force field and the correlation length it was drawn at.

    Separated from the magnitude because the two are not independent in effect:
    the RMS normalisation makes a localised field move the object much less than
    a spread one of the same nominal strength, so drawing strength and length
    independently produces almost nothing that is both localised and large. In
    120 training trajectories that corner held 2, and the one held-out rollout
    that runs away sits 10% past the largest localised trajectory trained on.
    """
    dev = sc.anchor_canonical.device
    lo = sigma_lo if sigma_lo is not None else sc.sim.radius
    hi = sigma_hi if sigma_hi is not None else sc.extent
    u = torch.rand(1, device=dev, generator=gen).item()
    sigma = lo * ((hi / lo) ** u)
    return sc.random_force_field(gen, sigma, 1.0), sigma


def stream(sc, n_steps, dt_mult, base_force, gen, cap, impulse_every=20,
           impulse_range=4.0, keep_accel=True, on_step=None, field=False,
           sigma_lo=None, sigma_hi=None):
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
                f, _ = draw_impulse(sc, base_force, gen, impulse_range, field,
                                     sigma_lo, sigma_hi)
                v = v + sc.impulse_dv(f)
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
