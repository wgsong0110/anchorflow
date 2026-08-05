# Learning the implicit step — retired

The idea: replace i-PhysGaussian's Newton–GMRES solve of the within-step
momentum residual with one network forward pass, so a single big step stands in
for many explicit substeps. Removed on 2026-08-05. This file exists so the
measurements are not repeated.

Code removed: `lib/anchorflow/implicit.py`, `exe/train_implicit_gnn.py`,
`exe/rollout_implicit_gnn.py`, `exe/diagnose_rollout_gap.py`,
`exe/verify_processor.py`, `exe/bench_solver.py`. All recoverable from git
history before this commit.

## The premise did not hold on this scene

Same physical step (4e-3 s = 40 explicit substeps), ficus, M=512 anchors,
N=171,553 material Gaussians, RTX 4090:

| method | ms/step | vs explicit | R/R_explicit |
|---|---|---|---|
| 40 explicit substeps | 10.04 | 1.00x | — |
| LBFGS 1 iter | 4.80 | 2.09x | 0.199 |
| LBFGS 3 iters | 4.23 | 2.37x | 0.199 |
| LBFGS 5 iters | 9.34 | 1.08x | 0.136 |
| LBFGS 10 iters | 21.34 | 0.47x | 0.092 |
| LBFGS 20 iters | 53.77 | 0.19x | 0.016 |
| LBFGS 40 iters | 100.64 | 0.10x | 0.008 |

Advancing 4 ms of physical time costs 10 ms explicitly and 100 ms implicitly at
the iteration count that is actually accurate. A network forward is 2.2 ms, so
even a perfect one-shot predictor saves at most 4.5x against explicit — and the
implicit route only pays at all where explicit is stability-limited to far
smaller substeps than the 1e-4 this config uses. That regime was never
measured; if the direction is ever revisited, measure it first.

The implicit step itself is sound: driven by an LBFGS solve every step, a 40x
rollout tracks the explicit reference for 100 frames with a residual ratio of
0.0005–0.011 and no divergence. The target is reachable. It is the network that
could not reach it.

## What was tried, and what it did

Metric is `||R(du)|| / ||R(du_predictor)||`, i.e. against ONE explicit-sized
step (not against "nothing" — the predictor already carries the elastic
acceleration). Rollout numbers are peak anchor displacement over 0.4 s of
physical time, against the explicit reference's own 0.122.

| change | on training states | in its own rollout |
|---|---|---|
| gain `beta*dt^2` | 0.033 | diverges, R/R = 1.0 |
| gain `\|predictor\|` | 0.043 | exponential, NaN by frame 40 |
| gain `dt*vel_scale` | — | 4.9x WORSE than no correction at 1x dt |
| gain `beta*dt^2*accel_scale` | 0.128 | diverges at every dt |
| dt curriculum | 0.128 | diverges at every dt |
| smooth state noise 0.02/0.05/0.10 | 0.069/0.100/0.112 | 71-131x |
| chained (autoregressive) training | 0.185 | 2.4-2.7x at 5-20x dt — the best result |
| elastic force as an input | 0.070 | 207-224x |
| attention instead of 4-hop message passing | 0.078 | 3.1-4.2x at 5-10x dt |
| 12 trajectories instead of 1 | 0.122 | 7.6-14.3x |
| supervised regression on du* | 0.060 | 110x |
| (p,v,a) in, absolute position out | 1.97-4.34 | 8-11x, object collapsed |

Two measurements explain most of this.

**Message passing could not represent the target.** One Newton step is
`(M/(beta dt^2) + K) delta = -R`, whose solution operator is a dense global
inverse — pinning the pot changes the branch tips within the same step. At k=8
neighbours and depth 4, perturbing one anchor moved 29 of 512 outputs, against a
k-NN graph diameter of 73 hops. Attention fixed the reach (512/512) and made
early convergence ~10x faster, and changed the rollout not at all.

**The residual objective was not the obstacle either.** Regression on an LBFGS
solve reaches 0.016 relative error on the correction — the map is learnable and
the network can express it. It still contributes exactly nothing in rollout.

**Absolute-position output is a precision trap.** Anchor coordinates are O(1),
one step's displacement is 4e-3, and the part not already given by the inertial
predictor is 4e-4. Emitting the position requires ~3.5 significant digits before
beating a single explicit step; the measured position RMS error was 9.2e-3,
i.e. 2.3x larger than the displacement being predicted.

## The one number that never moved

In every configuration above, the residual ratio during the network's own
rollout was 1.0000 to four decimals — from the very first frame, whose state is
the same initial condition the training states start from. A distribution
argument explains drift after several steps; it does not explain a contribution
of exactly zero at frame 0. That was never chased down, and it is the first
thing to look at if this is ever picked up again.
