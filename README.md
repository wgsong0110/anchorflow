# anchorflow

Replacing PhysGaussian's MPM with something that runs at interactive rates on
the same Gaussian cloud, without giving up the physics.

PhysGaussian animates a 3D Gaussian scene by treating the Gaussians as material
points and stepping them with MPM on a background grid. It is faithful and it is
slow: for ficus, 171,553 material particles on a 100^3 grid, forty substeps per
frame. This repository replaces that with a two-stage chain.

```
MPM (ground truth)  →  anchor simulator (teacher)  →  learned stepper (student)
     171,553 particles      ~600 anchors                 one forward pass
     100^3 grid             40 explicit substeps         per coarse frame
```

The anchor simulator drops the grid. A few hundred anchors carry the state;
every Gaussian's deformation gradient comes from a weighted Procrustes fit of
its neighbouring anchors, the Fixed Corotated stress is evaluated there, and the
forces are scattered back. The student then replaces that simulator's forty
substeps with a single attention pass over the anchors.

Both stages are fitted, and both are measured against MPM's own particles.

## Where it stands

Rollout error against MPM, sixty frames, held-out impulses, as a fraction of how
far MPM moved the object:

| | vs MPM |
|---|---|
| anchor simulator, as originally built | 13.12% |
| anchor simulator, fitted to MPM | **5.08%** |
| what 512 anchors can represent of MPM at all | 0.75% |

The student reproduces its teacher to 3.33% in anchor space and runs 18.9x
faster than the fair explicit baseline. Retraining it against the fitted
simulator is the remaining step.

## Layout

```
lib/anchorflow/
  anchor_mpm.py      the anchor simulator: shape matching, Fixed Corotated,
                     analytic forces. The fused CUDA path lives in lib/anchorstep
  anchor_sparse.py   the same physics with the discretisation as parameters --
                     anchors with orientation, extent and stiffness, a compact
                     support that follows the fit, and density control
  anchor_fit.py      a differentiable rewrite used to fit it (analytic force,
                     Newton polar factor, substep checkpointing)
  mpm_teacher.py     PhysGaussian's MPM, wrapped so it can be asked "from HERE,
                     what happens next?" in the anchor state the student uses
  nextstate.py       the student: attention over anchors -> displacement
  scene_setup.py     scene, material, boundary conditions, impulses
  streams.py         impulse distributions (uniform, force fields, pokes)

exe/
  fit_anchor_sparse.py    fit the anchor simulator to MPM
  train_nextstate.py      train the student (--teacher anchor|mpm)
  eval_vs_mpm.py          every stepper against MPM's particles, one ruler
  verify_mpm_teacher.py   is MPM usable as a teacher for a 512-anchor state?
  verify_anchor_fit.py    does the differentiable rewrite reproduce the original?
  mpm_setup.sh            stand up an instance (DreamPhysics, warp, kernels, data)
  keepalive.sh            keep a run alive across instance restarts
```

## What was learned the hard way

**A one-step loss does not give a good rollout.** Fitting the simulator on
single coarse steps halved the one-step error four separate times and left the
sixty-frame rollout unchanged or worse. The loss has to unroll -- twelve frames,
with the simulator running on its own output after the first.

**The teacher's data distribution matters more than its parameter count.** The
fit was run on three uniform impulses while the student it feeds was trained on
125 force fields. Matching them was worth more than any architectural change.

**Per-anchor stiffness beats per-anchor geometry.** 512 stiffness multipliers
reach 6.29%; 5,890 position, rotation and extent parameters reach 7.68%. The
fitted stiffness has a median of 0.226 -- the discretisation was several times
too stiff, which an earlier measurement had already suggested.

**Two numerical faults were real physics bugs.** The polar decomposition returns
a reflection when an element inverts, which makes Fixed Corotated treat the
inverted state as an energy minimum and never recover; and the eigendecomposition
of the anchor scatter matrix has a NaN derivative wherever a Gaussian's
neighbourhood is degenerate, which a compact support makes common.

See `~/workspace/result/anchorflow/outputs.md` for the full experimental record.

## Running

An instance needs the trained ficus, DreamPhysics at a pinned commit, warp 1.0.2
and the CUDA kernels:

```bash
bash exe/mpm_setup.sh          # idempotent; fails loudly if anything is missing
```

Fitting the simulator, and training the student against it:

```bash
python3 exe/fit_anchor_sparse.py --ply ... --config ... \
    --params all --unroll 12 --eval_rollout 1 --traj_cache ... --r2 ...

python3 exe/train_nextstate.py --ply ... --config ... \
    --frozen_weights --no_accel --rot_fallback --field --dagger --chunk 4
```

Everything long-running resumes: parameters, optimiser, RNG, the DAgger pool and
the MPM trajectories all round-trip through R2, so losing an instance costs the
minutes since the last save and nothing else.
