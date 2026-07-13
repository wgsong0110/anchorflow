# anchorflow — learned GNN dynamics over 3DGS anchors

## Idea

MPM-style simulators only move **passive** objects (they respond to external
force). Self-actuated objects — people, animals, machines — carry internal
dynamics MPM cannot express. anchorflow replaces the physics rule with a **GNN
that learns anchor dynamics autoregressively**, so internally-driven motion is
representable, while the graph inductive bias makes spatial coupling between
anchors explicit.

```
3DGS  ──►  anchors (FPS/k-means)  ──►  anchor graph (kNN/radius)
                                             │
                                   GNN autoregressive rollout
                                             │
                                   anchor Δ ──► Gaussian deform (LBS)
                                             │
                                          render  ──► compare to frame ──► train
```

## Library stack (decision)

- **PyTorch** — base; needed anyway for the eventual deform + differentiable render.
- **GNN: hand-rolled GNS Interaction Network on pure-PyTorch scatter**
  (`index_add_`), *not* PyG/DGL. Why:
  1. The message-passing core is ~40 lines and we will customise it heavily for
     *learned self-actuation* (e.g. per-node phase/actuation latents) — a
     framework hides exactly the part we are researching.
  2. Zero PyG / torch-scatter / DGL build dependency — clean install on the
     arm64 host *and* the x86 GPU instances.
  3. `InteractionNetwork.forward(h, edge_index, e)` keeps the PyG
     `MessagePassing` signature, so PyG's neighbour samplers are a drop-in if we
     ever need large-N (thousands of anchors) scaling.
- NumPy + PyYAML only otherwise. Synthetic data is pure NumPy so it (and the
  core math) are testable without torch.

## Model — Encode-Process-Decode (Sanchez-Gonzalez et al., ICML 2020)

Second-order formulation, dt folded into units:

```
v_t     = p_t   - p_{t-1}
a_t     = p_{t+1} - 2 p_t + p_{t-1}      (regression target)
p_{t+1} = p_t + v_t + a_pred             (rollout integration)
```

- **node features** `[v_t (3), fixed_flag (1)]` → normalised
- **edge features** `[p_j - p_i (3), ‖·‖ (1)]` — relative only ⇒ translation invariant
- **encoder** node/edge MLPs → hidden (default 128)
- **processor** `M` interaction steps (default 6–8) with residual edge+node updates,
  sum aggregation at the receiver
- **decoder** MLP → normalised acceleration; `Normalizer.inverse` → world accel
- **fixed anchors** are masked from the loss and pinned during rollout

Training tricks that matter: GNS **random-walk input noise** (`noise` cfg) so the
model tolerates its own rollout drift; running-stat **input/target normalisation**
(primed once before training).

## Why GNN over MLP (the thing to prove)

A node's next state depends on its neighbours (move the shoulder → the hand
follows). An MLP over a single node cannot see that; message passing aggregates
it. The **success metric is rollout skill vs a constant-velocity baseline** on
*held-out extrapolation frames* — reported every eval as `xN better`. If the GNN
only matched the baseline it would mean no coupling was learned.

## Validation ladder

| seq | regime | what it tests |
|-----|--------|---------------|
| `traveling_wave` | kinematic, self-actuated | propagate a phase relationship (our target class) |
| `pendulum_chain` | coupled 2nd-order | neighbour state determines acceleration |
| `cloth` | passive, larger N, 2D + contact | pipeline scales past a 1D chain (GNS baseline) |

## Package layout

```
lib/anchorflow/
  synth.py     synthetic sequences (pure NumPy)      [Priority 1 data]
  graph.py     knn_graph / radius_graph + edge_features
  dynamics.py  InteractionNetwork, GNSDynamics, Normalizer, rollout
  deform.py    AnchorBinding (translation + local-rigid LBS), FPS anchor extract  [Priority 2]
exe/train_dynamics.py   train + rollout-eval CLI (records git commit)
cfg/{wave,pendulum,cloth}.yaml
```

## Run

```bash
cd master
python exe/train_dynamics.py --cfg cfg/pendulum.yaml     # coupled dynamics
python exe/train_dynamics.py --cfg cfg/wave.yaml         # self-actuated stand-in
python exe/train_dynamics.py --cfg cfg/cloth.yaml        # passive baseline
```
Outputs to `~/workspace/result/anchorflow/`: `rollout_<seq>_<commit>.npz`
(gt+pred+err_curve), `model_*.pt`, `history_*.json`. Runs on CPU (models are
tiny); `--cpu` forces it.

## Status

- **Priority 1 implemented.** Validated *without torch* on host: synthetic
  physics stable & pin-correct; second-order integration identity exact (0.0);
  kNN graph symmetric/loop-free. **Torch training run is pending** — needs an env
  with torch (the project image / an instance); not run on the arm64 host per the
  no-build rule.
- **Priority 2 foundation in place** (`deform.py`): FPS anchor extraction +
  translation/local-rigid skinning with the rest-pose binding computed once.
  Next: wire GNN-predicted anchor deltas → Gaussian deform → differentiable
  render → photometric loop.

## Open design questions (for discussion)

1. **Self-actuation signal.** Pure GNS is passive. To get internal drive we
   likely need per-node learnable actuation/phase latents (or a small recurrent
   state) fed into the node encoder — otherwise a deterministic autonomous system
   can still be fit, but a *controllable* gait cannot. Candidate: per-anchor
   latent `z_i` optimised jointly, decoded into an actuation force term.
2. **Fixed vs dynamic graph.** Articulated bodies favour a rest-pose graph;
   contact/large deformation favours per-step rebuild. Currently a cfg switch.
3. **Anchor count / binding.** FPS count and skinning `k` trade sharpness vs
   smoothness of the induced Gaussian field — sweep once real 3DGS is attached.
```
