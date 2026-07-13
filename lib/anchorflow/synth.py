"""Synthetic anchor-motion sequences for validating GNN autoregressive dynamics.

Every generator returns a dict with:
    positions : float32 [T, N, 3]   anchor centres over time
    fixed     : bool    [N]         True for kinematically-pinned anchors
    name      : str

The sequences are deterministic given a seed and use pure NumPy so they can be
generated (and sanity-checked) on any host without torch.  They are intentionally
small (N in the tens) — the point is to verify that graph message passing learns
the *spatial coupling* between anchors, not to scale.

Three regimes, increasing difficulty:

    traveling_wave  kinematic gait-like motion.  Each node oscillates with a
                    phase offset along the chain -> a wave travels through it.
                    Tests whether the GNN can propagate a phase relationship
                    between neighbours.  This is our stand-in for a *self-actuated*
                    object: the motion is internally driven, not a response to
                    external force.
    pendulum_chain  mass-spring chain pinned at the top, swinging under gravity.
                    Coupled second-order dynamics -> neighbour state genuinely
                    determines a node's acceleration ("move the shoulder and the
                    hand follows").
    cloth           mass-spring lattice falling and settling under gravity.
                    2D connectivity, larger N, a passive-physics baseline that
                    mirrors GNS.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Kinematic: traveling wave (self-actuated stand-in)                          #
# --------------------------------------------------------------------------- #
def traveling_wave(n=16, T=240, dt=1.0, wavelength=6.0, speed=0.15,
                   amplitude=0.35, seed=0):
    """A chain of nodes along x; each oscillates in z with a phase offset.

    z_i(t) = A * sin(2*pi*(i/wavelength - speed*t))

    The motion is *internally generated* (no forces) which is exactly the class
    of object MPM cannot represent -- a good minimal target for learned dynamics.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    x = i.astype(np.float64) * (1.0 / max(1, n - 1)) * (n - 1) * 0.25  # even spacing
    t = np.arange(T) * dt
    phase = 2.0 * np.pi * (i[None, :] / wavelength - speed * t[:, None])
    z = amplitude * np.sin(phase)                       # [T, n]
    pos = np.zeros((T, n, 3), dtype=np.float64)
    pos[:, :, 0] = x[None, :]
    pos[:, :, 2] = z
    # tiny per-node y jitter so the graph is not perfectly 1-D / degenerate
    pos[:, :, 1] = rng.normal(0.0, 1e-3, size=(1, n))
    fixed = np.zeros(n, dtype=bool)                     # nothing pinned
    return _pack(pos, fixed, "traveling_wave")


# --------------------------------------------------------------------------- #
#  Dynamic: pinned mass-spring pendulum chain                                  #
# --------------------------------------------------------------------------- #
def pendulum_chain(n=12, T=300, dt=0.02, rest=0.25, k=180.0, damping=0.6,
                   gravity=9.81, seed=0):
    """Chain of point masses connected by stiff springs, top node pinned.

    Semi-implicit (symplectic) Euler integration.  Starts from a horizontal
    configuration so it swings -- strongly coupled, second-order motion.
    """
    rng = np.random.default_rng(seed)
    # rest configuration: horizontal chain along +x, node 0 pinned at origin
    p = np.zeros((n, 3), dtype=np.float64)
    p[:, 0] = np.arange(n) * rest
    p += rng.normal(0.0, 1e-3, size=p.shape)            # break exact symmetry
    v = np.zeros((n, 3), dtype=np.float64)
    fixed = np.zeros(n, dtype=bool)
    fixed[0] = True

    edges = [(i, i + 1) for i in range(n - 1)]
    g = np.array([0.0, 0.0, -gravity], dtype=np.float64)

    out = np.empty((T, n, 3), dtype=np.float64)
    for t in range(T):
        out[t] = p
        f = np.tile(g, (n, 1))                          # gravity on every node
        for a, b in edges:                              # spring forces
            d = p[b] - p[a]
            L = np.linalg.norm(d) + 1e-9
            fs = k * (L - rest) * (d / L)
            f[a] += fs
            f[b] -= fs
        f -= damping * v                                # viscous damping
        v = v + dt * f                                  # unit mass
        v[fixed] = 0.0
        p = p + dt * v
        p[fixed] = out[0][fixed]                        # hard-pin
    return _pack(out, fixed, "pendulum_chain")


# --------------------------------------------------------------------------- #
#  Dynamic: mass-spring cloth (passive-physics baseline, a la GNS)            #
# --------------------------------------------------------------------------- #
def cloth(nx=8, ny=8, T=260, dt=0.02, rest=0.2, k=140.0, shear_k=90.0,
          damping=0.5, gravity=9.81, floor=-1.4, restitution=0.35, seed=0):
    """A grid of masses with structural + shear springs, dropped onto a floor.

    Two corners of the top row are pinned so it drapes rather than falls away.
    """
    rng = np.random.default_rng(seed)
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    p = np.zeros((n, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    p[:, 0] = xs.reshape(-1) * rest
    p[:, 1] = ys.reshape(-1) * rest
    p[:, 2] = 0.4 + rng.normal(0.0, 1e-3, size=n)       # start slightly above
    v = np.zeros((n, 3), dtype=np.float64)

    fixed = np.zeros(n, dtype=bool)
    fixed[idx[-1, 0]] = True
    fixed[idx[-1, -1]] = True

    edges = []
    for r in range(ny):
        for c in range(nx):
            a = idx[r, c]
            if c + 1 < nx:
                edges.append((a, idx[r, c + 1], rest, k))         # structural
            if r + 1 < ny:
                edges.append((a, idx[r + 1, c], rest, k))
            if r + 1 < ny and c + 1 < nx:
                edges.append((a, idx[r + 1, c + 1], rest * np.sqrt(2), shear_k))
            if r + 1 < ny and c - 1 >= 0:
                edges.append((a, idx[r + 1, c - 1], rest * np.sqrt(2), shear_k))
    edges = np.array(edges, dtype=np.float64)
    ea = edges[:, 0].astype(int)
    eb = edges[:, 1].astype(int)
    er = edges[:, 2]
    ek = edges[:, 3]
    g = np.array([0.0, 0.0, -gravity], dtype=np.float64)

    out = np.empty((T, n, 3), dtype=np.float64)
    for t in range(T):
        out[t] = p
        f = np.tile(g, (n, 1))
        d = p[eb] - p[ea]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        fs = (ek[:, None] * (L - er[:, None])) * (d / L)
        np.add.at(f, ea, fs)
        np.add.at(f, eb, -fs)
        f -= damping * v
        v = v + dt * f
        v[fixed] = 0.0
        p = p + dt * v
        # floor collision (z = floor), simple restitution
        below = p[:, 2] < floor
        p[below, 2] = floor
        v[below, 2] = np.abs(v[below, 2]) * restitution
        p[fixed] = out[0][fixed]
    return _pack(out, fixed, "cloth")


# --------------------------------------------------------------------------- #
def _pack(pos, fixed, name):
    pos = np.asarray(pos, dtype=np.float32)
    # re-centre and scale to a roughly unit box so learning is well-conditioned
    c = pos.reshape(-1, 3).mean(0)
    pos = pos - c
    scale = float(np.percentile(np.abs(pos), 99)) + 1e-6
    pos = pos / scale
    return {
        "positions": pos.astype(np.float32),
        "fixed": np.asarray(fixed, dtype=bool),
        "name": name,
        "scale": scale,
    }


GENERATORS = {
    "traveling_wave": traveling_wave,
    "pendulum_chain": pendulum_chain,
    "cloth": cloth,
}


def make(name, **kwargs):
    if name not in GENERATORS:
        raise KeyError(f"unknown sequence '{name}', choose from {list(GENERATORS)}")
    return GENERATORS[name](**kwargs)
