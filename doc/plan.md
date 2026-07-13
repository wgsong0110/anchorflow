# anchorflow v1 — architecture & build order (confirmed)

## One-line

Take **SC-GS's scene representation** (canonical 3DGS + sparse control nodes +
RBF-LBS warp), **replace its per-node deformation MLP(node, t) with a GNN that
rolls control-node state forward autoregressively**, and **train by SDS from a
video-diffusion prior instead of photometric reconstruction** — distilling
self-actuated motion into a spatially-coherent, forward-simulatable GNN dynamics.

Novelty framing: **GNN dynamics as a structural (spatial-coherence) prior for
video-SDS 4D generation of self-actuated objects.**

## Decisions locked

- **Task = generation (per-object SDS distillation)**, not reconstruction. No
  observed trajectory; the video-diffusion prior defines plausible motion.
- **Representation = SC-GS** (canonical Gaussians + control nodes + RBF-LBS + ARAP).
- **Dynamics = autoregressive GNN** (spatial message passing). Justified by the
  headline claim: roll forward *past* the diffusion window (autonomous simulation),
  not just animate one clip.
- **Temporal state (v1) = minimal**: per-node actuation latent `z_i` + Markov
  (2-frame) state. **SSM/Mamba deferred to v2**, adopted only if long-horizon
  autonomous rollout becomes the headline and needs bounded-recurrence stability.
- **Loss = video-diffusion SDS** (+ ARAP/regularizers from SC-GS).
- **Video backbone = SVD (Stable Video Diffusion, image-conditioned)**. The
  canonical render is frame-0; SVD supplies the motion prior. Identity-preserving,
  easy to attach. (Action-controllability is weak — revisit text+video in v2 if we
  need *specified* actions.)
- **Canonical 3DGS asset = TRELLIS (image→3DGS `.ply`, MIT, `JeffreyXiang/TRELLIS-image-large`)**.
  SC-GS already demonstrates consuming a TRELLIS `.ply` (`edit_gui.py`). Same
  subject image seeds both the canonical GS *and* the SVD prior → consistent.
- Synthetic `synth.py` demoted from "idea validation" to **GNN-core unit test**.

## Canonical asset (subject) — how we get it

We generate on demand rather than hunt a specific pre-made `.ply`:

1. Pick a **neutral rest-pose** image of a self-actuated subject (v1 first target:
   a simple quadruped or humanoid where SVD has a strong motion prior — e.g. a
   standing horse/dog or a front-facing person). Source: stock/CC image or a
   text-to-image gen; keep the pose canonical (standing, arms/legs neutral).
2. `TRELLIS(image) → save_ply()` → canonical 3DGS `.ply` (xyz, opacity, scale, rot, SH).
3. Feed `.ply` as SC-GS's static Gaussians; FPS → control nodes.
4. Same image = SVD frame-0 conditioning during SDS.

Zero-generation smoke-test fallback: grab any public object `.ply` (a TRELLIS
example / Voxel51 GS) just to exercise the LBS warp before wiring TRELLIS.

Runs on GPU env (TRELLIS ~5GB weights + CUDA deps; not on this arm64 host).
Local `~/datasets/dnerf` (jumpingjacks/mutant/trex/standup) kept as an alt
subject source, but D-NeRF needs full dynamic training to yield a canonical GS —
TRELLIS is the faster path.

## What we reuse vs replace (grounded in the SC-GS code we read)

| SC-GS component (file:sym) | anchorflow v1 |
|---|---|
| `farthest_point_sample` node init (`time_utils.py:461`) | **reuse** (≈ our `deform.extract_anchors_fps`) |
| `cal_nn_weight` RBF-LBS, learnable `_node_radius`/`_node_weight` (`:934`) | **reuse** — upgrade our `deform.py` softmax→RBF |
| `forward` LBS warp, `local_frame` `R_k(x−node_k)+node_k+t_k` (`:1133`) | **reuse** as the anchor→Gaussian deform |
| `node_deform(t) = MLP(node_pos, t) → d_xyz,d_rot,d_scale` (`:990`) | **REPLACE with GNN autoregressive rollout** |
| `p2dR` SVD rotation from node trajectory (`:1044`) | optional (= our Procrustes `deform_affine`) |
| `arap_loss` / `elastic_loss` / `acc_loss` (`:1080–1120`) | **reuse** as motion regularizers |
| photometric loss (`train.py`) | **REPLACE with video-SDS** |
| 3DGS backbone + differentiable rasterizer (`gaussian_renderer/`) | **reuse** |

## Module plan (`lib/anchorflow/`)

```
anchors.py    control-node state container + FPS init + SC-GS RBF-LBS binding
              (absorbs current deform.py; add learnable per-node radius/weight,
               per-node actuation latent z_i, per-node rotation state)
dynamics.py   GNN over the node graph. v1 output = per-node transform delta
              (d_xyz ∈ R^3, d_rot ∈ quaternion-residual), NOT bare acceleration.
              Input node feat = [vel(3), z_i(actuation latent), fixed(1)].
              Autoregressive rollout: node_state_{t+1} = state_t ⊕ GNN(...).
warp.py       node transforms → Gaussian (μ, rot, scale) via RBF-LBS blend
              (port SC-GS forward warp; wire to the rasterizer).
sds.py        video-diffusion SDS: render rollout → video → noise → ε-pred →
              SDS grad. Backbone TBD (see open Q1).
reg.py        ARAP / elastic / acc regularizers on node trajectory (port SC-GS).
synth.py      unchanged — GNN-core unit test (rollout stability, spatial coupling).
graph.py      unchanged — knn/radius node graph + relative edge features.
```

Training script `exe/train_gen.py`: canonical GS (pretrained, frozen or lightly
tuned) → FPS nodes → GNN rollout → warp → render clip → SDS + ARAP → backprop to
{GNN weights, actuation latents z_i, node radius/weight}. Records git hash.

## Build order

1. **Pipeline skeleton on a GPU env** (this host is arm64/no-torch — build/run
   where torch + rasterizer + diffusion exist): load a pretrained canonical 3DGS,
   FPS nodes, port SC-GS `cal_nn_weight` + LBS `forward` into `warp.py`, render a
   static frame. Verify LBS deform works with *hand-set* node transforms.
2. **GNN as deformation driver, minimal loop**: node graph + GNN outputs per-node
   (d_xyz, d_rot); autoregressive rollout for T frames; render clip. No SDS yet —
   sanity that a rollout produces a non-degenerate, spatially-coherent moving clip.
3. **Attach video-SDS** (open Q1 backbone). Start with a short window (~14–25 fr),
   gradient clip + TBPTT. Add ARAP. Get *any* coherent self-actuated motion.
4. **Self-actuation `z_i`**: verify motion is driven/controllable, not trivial rest.
5. **Autonomous extrapolation**: roll past the SDS window; measure stability. This
   is the headline-claim experiment and the go/no-go for v2 SSM.

## Open questions (must resolve before step 3)

1. ~~Video-diffusion backbone~~ **RESOLVED → SVD (image-conditioned).**
2. **Actuation signal `z_i` form**: free per-node latent (SDS-optimized) vs learned
   phase oscillator vs initial-velocity seed. Determines controllability.
3. **SDS-through-rollout-through-render stability**: TBPTT window, per-step grad
   clip, rollout-length schedule. Central technical risk.
4. **Canonical GS**: freeze after static pretrain, or jointly tune under SDS?

## Status / constraints

- SC-GS cloned & read at `~/workspace/external/dynamic-scene/SC-GS`.
- Nothing in the SDS pipeline is runnable on this arm64 host (no torch/GPU/build).
  Steps 1+ execute in a GPU env (project image / instance).
- Current runnable-anywhere asset: `synth` + GNS core (unit test), validated
  without torch (integration identity exact, graph symmetric).
