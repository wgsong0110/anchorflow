# anchorflow v2 — MoSca-grounded, GNN⊗SSM simulator (locked)

Fixes v1's collapse/drift: the GNN no longer has to *invent* plausible motion from
scratch under high-DOF SDS. Instead it is **grounded on a reconstructed trajectory**
(MoSca) and then **generalised over initial conditions** by MDS.

## Pipeline
```
image
  → SVD img2vid                → monocular video (no real footage needed; same SVD as MDS)
  → MoSca (monocular anchor recon) → {anchor trajectory, canonical GS, LBS binding}  [freeze anchors/binding]
  → GNN⊗SSM supervised pretrain  (fit the MoSca trajectory — 1 initial condition; grounds dynamics, no collapse)
  → MDS refine                   (randomise z_i + init conditions → generalise over ICs → reusable simulator)
```
Why MoSca (not SC-GS): SC-GS assumes multi-view coverage; the generated video is
monocular, and MoSca's motion-scaffold *nodes are our anchors* with per-frame tracks.
Why MDS is essential (not optional): supervised fits ONE IC; MDS is the only stage that
generalises to arbitrary ICs → turns the GNN into a controllable simulator.

## Dynamics = GNN (spatial) ⊗ per-node SSM (temporal)   [`lib/anchorflow/ssm_dynamics.py`]
Per step (dt = hyperparameter, matched to MoSca frame dt):
```
m_i  = GNN message passing over the anchor graph        # spatial
u_i  = encode([v_i, m_i, e_i, z_i])
h_i  = SSM(h_i, u_i, dt)                                 # diagonal, decay∈(0,1) -> bounded/stable
a_i  = tanh(decode(h_i)) * accel_scale                   # acceleration ONLY
p_i' = p_i + v_i * dt                                    # explicit Euler (prev pos + prev vel)
v_i' = v_i + a_i * dt                                    # position is NEVER decoded from hidden
```

## State
```
physical (explicit):  p, v, a          # p,v by dt-integration only
SSM hidden (separate): h_i ∈ R^d        # recurrent memory -> produces a (gait phase/momentum)
init:  h_i^0 = encode([e_i, z_i, init_vel_i, init_pos_i])
       supervised: p^0,v^0 = MoSca first two frames (v^0 = p^1 - p^0)
       MDS:        randomise z_i + init_* (h^0 varies) → IC generalisation
```

## Per-anchor elements
| symbol | role | trained? | varies per IC? |
|---|---|---|---|
| canonical | rest position (MoSca scaffold) | no (buffer) | no |
| `e_i` | **intrinsic identity** (which part / joint-vs-rigid / material-like) | yes | **no (fixed)** |
| `z_i` | **actuation/control** | yes | **yes (MDS)** |
| `_radius,_node_weight` | RBF-LBS binding | yes | no |
| `init_vel/init_pos` | initial conditions | (opt.) | yes (MDS) |

## Status / TODO
- DONE: `ssm_dynamics.py` (DiagonalSSM + SSMDynamics + ssm_rollout, explicit p/v/a),
  `anchors.py` gains `e_i` intrinsic + `z_i` control. Long-rollout stability verified (numpy).
- TODO: (1) MoSca integration — repo + monocular video → scaffold-node trajectory / canonical /
  binding / frame dt; likely a separate CI image (its own CUDA deps). (2) `train_gen.py`: two-stage
  loop = supervised pretrain on MoSca trajectory (MSE on rollout positions) → MDS refine with z_i+init
  randomisation. (3) SVD img2vid video generation step.
