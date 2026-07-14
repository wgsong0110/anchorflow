# SC-GS reconstructor for anchorflow v3

Replaces the MoSca (monocular) reconstructor with **SC-GS** (Huang et al.,
"Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes", CVPR 2024,
repo [`yihua7/SC-GS`](https://github.com/yihua7/SC-GS)). SC-GS is architecturally
almost identical to anchorflow: a canonical 3DGS + **sparse control nodes** + an
LBS binding + a **deformation MLP** + ARAP regularization. Its control nodes ARE
our anchors and its kNN/RBF skinning IS our LBS binding, so its output drops
straight in.

Driver: `exe/scgs_run.sh` (train). Exporter: `exe/scgs_export.py` (checkpoint ->
anchorflow portable format). Both run inside a SC-GS GPU image (`/opt/SC-GS`).

---

## 1. SC-GS representation (the exact objects we consume)

All in `utils/time_utils.py:ControlNodeWarp` (`deform_type='node'`, the default;
built by `scene/deform_model.py:DeformModel`).

### Control nodes (= anchorflow anchors)
- `ControlNodeWarp.nodes` — `nn.Parameter` of shape **`[M, 3 + hyper_dim]`**.
  First 3 columns are the canonical node xyz; the remaining `hyper_dim` columns
  are "hyper coordinates" used only to separate spatially-close-but-disconnected
  parts during the kNN skinning (`hyper_dim=8` for D-NeRF, `2` for real scenes;
  `0` when `skinning=True`).
- `M = node_num` (default arg `1024`; `train_gui.sh` / our driver use `512`).
- Initialized by **farthest-point sampling** of the canonical Gaussian point
  cloud (`ControlNodeWarp.init` -> `farthest_point_sample`), or kept 1:1 with the
  pcl when `node_num >= N`. With `--is_blender` they instead start from random
  points; the readme insists real/multi-view scenes drop `--is_blender`.
- Per-node RBF kernel width `_node_radius` (`[M]`, used as
  `node_radius = exp(_node_radius)`) and per-node blend weight `_node_weight`
  (`[M,1]`, `node_weight = sigmoid(_node_weight)`). Nodes can be densified/pruned
  during training (`ControlNodeWarp.densify`) when `--node_enable_densify_prune`.

### Deformation MLP (control-node motion over time)
- `ControlNodeWarp.network` = `utils.time_utils.DeformNetwork` (an MLP; a hash
  variant `HashDeformNetwork` exists behind `--use_hash`, and a `StaticNetwork`
  behind `--is_scene_static`). **Not a HexPlane.**
- Input: **(node_xyz, t)** with positional encoding (`get_embedder`; time freq 6
  for D-NeRF / `is_blender`, else 10). Output per node:
  **`d_xyz` (Δposition, 3)**, **`d_rotation` (quaternion, 4)**, `d_scaling` (3),
  optionally `d_opacity`/`d_color`/`local_rotation`.
- `ControlNodeWarp.node_deform(t)` queries the MLP at the (detached) node
  positions; `get_trajectory(T)` returns `nodes[:, :3] + d_xyz` = node positions
  over time. **This is exactly what we sample to build `node_traj`.**
- `t` is a **normalized frame id `fid` in `[0,1]`**, `fid = frame_index/(T-1)`
  (`scene/dataset_readers.py`). So the training timestamps are `linspace(0,1,T)`.
- `d_rot_as_res` (default True): `d_rotation` is a *residual* quaternion added to
  the canonical Gaussian quaternion, then normalized
  (`gaussian_renderer.render` -> `get_rotation_bias`). We export node orientation
  as `normalize([1,0,0,0] + d_rotation)`.

### LBS skinning (= anchorflow's RBF-LBS binding)
`ControlNodeWarp.cal_nn_weight(x, feature)` and `ControlNodeWarp.forward`:
1. k-NN from each Gaussian `x` (concatenated with its `feature[:, :hyper_dim]`
   hyper coords) to the nodes (concatenated likewise), `K = self.K` (default 3),
   via `pytorch3d.ops.knn_points`.
2. RBF weight `w = exp(-dist / (2 * node_radius[idx]^2)) * node_weight[idx]`,
   then normalized over the K neighbors -> `nn_weight [N,K]`, `nn_idx [N,K]`.
3. Gaussian deformation = LBS blend:
   `d_xyz_gauss = Σ_k nn_weight[:,k] * node_d_xyz[nn_idx[:,k]]`
   (and likewise for rotation/scaling), scaled by `motion_mask`
   (`= sigmoid(feature[:,-1:])` when `--gs_with_motion_mask`, else 1).
   When `skinning=True`, weights are a learned dense `softmax(feature)` over all
   M nodes instead of kNN+RBF.

**This binding is what we dump to `lbs_weight.npz`** so anchorflow reuses SC-GS's
skinning verbatim rather than recomputing RBF weights.

### ARAP / as-rigid regularization
- `ControlNodeWarp.arap_loss` (default, added when `not --no_arap_loss`): samples
  node positions at nearby times, builds node connectivity (`K=10`,
  `utils/deform_utils.py:cal_connectivity_from_points`) and penalizes
  `cal_arap_error` (edge-length / local-rigidity change). Weight is annealed
  `1e-4 -> 0` over iters 0..20000 (`lambda_arap_landmarks/steps`), applied only
  after `warm_up`.
- Also available: `elastic_loss`, `acc_loss` (temporal smoothness), and an
  SVD-based per-node rotation `p2dR` used when `d_rot_as_res=False`.
- A separate **editing-time** ARAP lives in `utils/arap_deform.py` / `lap_deform.py`
  (used by `edit_gui.py`); not needed for reconstruction/export.

---

## 2. Training

### Entry point
The real trainer is **`train_gui.py`** (not `train.py`, which only holds
`training_report`). Run it in terminal mode by omitting `--gui`. Auto-detects the
dataset format in `scene/__init__.py`.

### Dataset formats (auto-detected by `Scene`)
| trigger file under `source_path`        | reader                    | notes |
|-----------------------------------------|---------------------------|-------|
| `poses_bounds.npy`                      | `readPlenopticVideoDataset` (Neu3D) | **multi-view video** — best fit for anchorflow |
| `sparse/` or `colmap_sparse/`           | `readColmapSceneInfo`     | monocular-style; `fid = int(digits in name)/(N-1)` |
| `transforms_train.json`                 | `readNerfSyntheticInfo`   | D-NeRF synthetic; use `--is_blender` |
| `dataset.json`                          | `readNerfiesInfo`         | Nerfies/HyperNeRF |
| `train_meta.json`                       | `readCMUSceneInfo`        | PanopticSports; `fid = t/150`, 20 timesteps hard-coded |

**Recommended multi-view video layout (Neu3D / plenoptic)** — this is the
per-view posed image-sequence format anchorflow targets:
```
$WS/
  poses_bounds.npy          # LLFF-style: one row per camera (3x5 pose + 2 bounds)
  frames/
    cam00/ 0000.png 0001.png ...   # camera 0's frame sequence (sorted)
    cam01/ 0000.png ...
    ...
  points3D.ply              # optional init pcl; else 100k random points are used
```
- `readCamerasFromNpy`: `frame_time = idx/(n_frames-1)` -> `fid`. Camera indices
  in `hold_id=[0]` are held out as the **test** view when `--eval`.
- **Gotcha:** `Scene.__init__` calls the reader with a **hard-coded
  `num_images=24`**. `scgs_run.sh` sed-patches this to the real per-camera frame
  count (`NUM_FRAMES`) so all timesteps are used.

### Train command (what `scgs_run.sh` runs, multi-view real scene)
```bash
cd /opt/SC-GS
python train_gui.py \
    --source_path $WS --model_path $WS/outputs/scene \
    --deform_type node --node_num 512 --hyper_dim 2 \
    --iterations 80000 --resolution 2 --eval \
    --init_isotropic_gs_with_all_colmap_pcl
```
D-NeRF synthetic (`IS_BLENDER=1`) instead appends
`--is_blender --gt_alpha_mask_as_scene_mask --local_frame --W 800 --H 800`
(mirrors upstream `train_gui.sh`).

### Key hyperparameters (`arguments/__init__.py`)
- `node_num` (M anchors, 512/1024), `hyper_dim` (2 real / 8 D-NeRF), `K=3` (LBS
  neighbors), `deform_type=node`.
- `iterations=80000`, `warm_up=3000` (static 3DGS warmup before deformation),
  `iterations_node_sampling=7500`, `iterations_node_rendering=10000` (control
  nodes are pre-trained as isotropic Gaussians for ~10k steps first),
  `dynamic_color_warm_up=20000`.
- Densify: `densify_until_iter=50000`; node densify/prune off by default
  (`node_enable_densify_prune=False`).
- Losses: `lambda_dssim=0.2`, optical-flow + motion-mask schedules, ARAP schedule
  (above). Real scenes with noisy poses: consider `--no_motion_mask_loss`,
  `--no_arap_loss` (ARAP is slow on large scenes).

### Checkpoint structure (what gets saved, and how we resume)
Saved at `save_iterations = [7000,10000,20000,30000,40000, iterations]` + the
best-PSNR iter + `warm_up-1`, under `model_path` (SC-GS **auto-appends `_node`**):
```
$WS/outputs/scene_node/
  cfg_args                                  # Namespace repr of ALL flags (we parse this)
  cameras.json  input.ply
  point_cloud/iteration_<it>/point_cloud.ply   # canonical dense 3DGS (Scene.save)
                                               #   INRIA fields + fea_* (hyper+motion)
  deform/iteration_<it>/deform.pth             # ControlNodeWarp.state_dict:
                                               #   nodes, _node_radius, _node_weight,
                                               #   network.* (deform MLP),
                                               #   gs_* (node isotropic gaussians)
```
There is **no single `chkpnt*.pth`** — the state is split ply + deform.pth.
**Resume is built-in:** on start `Scene(load_iteration=-1)` +
`deform.load_weights(-1)` reload the latest iteration, so re-running the same
command continues from the last save.

Runtime: ~1–2 h on one modern GPU for 80k iters at `--resolution 2` (D-NeRF
scale); larger multi-view scenes scale with Gaussian count.

---

## 3. Export: SC-GS control nodes -> anchorflow anchors (`exe/scgs_export.py`)

```bash
python exe/scgs_export.py --model_path $WS/outputs/scene_node \
    --num_frames <T> --out /workspace/scgs_out   [--iteration -1]
```
Reads `cfg_args` to rebuild `DeformModel` with the exact flags, loads
`deform.pth` + the latest `point_cloud.ply`, then writes:

| output | shape | how it is produced |
|--------|-------|--------------------|
| `node_traj.npy` | `[T, M, 3]` | `nodes[:, :3] + node_deform(t)['d_xyz']` at `t = linspace(0,1,T)` (chunked over time) |
| `node_rot.npy`  | `[T, M, 4]` (wxyz) | `normalize([1,0,0,0] + node_deform(t)['d_rotation'])`; omitted if `--is_scene_static` |
| `canonical.ply` | INRIA 3DGS | the trained `point_cloud.ply` with the trailing `fea_*` columns dropped -> identical field layout to `exe/mosca_export.py` (`x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*`) |
| `lbs_weight.npz` | `nn_idx [N,K]`, `nn_weight [N,K]`, `K`, `node_num`, `skinning` | `ControlNodeWarp.cal_nn_weight(x=gauss_xyz, feature=gauss_feature)` — SC-GS's own kNN+RBF (or dense softmax when `skinning=True`) |

Prints per-array shapes and a final `SCGS_EXPORT_OK`.

**Anchor mapping.** SC-GS control node `m` = anchorflow anchor `m`; its trajectory
`node_traj[:, m]` (+ orientation `node_rot[:, m]`) is the anchor dynamics the
GNN⊗SSM learns. Dense Gaussian `i` (in `canonical.ply`) is bound to anchors
`nn_idx[i]` with weights `nn_weight[i]` — feed these straight into anchorflow's
LBS to reuse SC-GS's binding instead of recomputing RBF weights. `M` matches
`node_traj.shape[1]`; `nn_idx` indexes `[0, M)`.

### Coordinate-frame notes
- SC-GS works in the dataset's world frame (COLMAP / LLFF / Blender), same frame
  as `canonical.ply` and the node positions — **no extra transform** between
  anchors and Gaussians; both are already world-space and consistent.
- Neu3D/CMU readers recenter/normalize the scene (`getNerfppNorm`,
  `translate_cam_info`); the applied translate is baked into the saved poses and
  point cloud, so exported anchors + Gaussians remain mutually consistent (just
  not in the original metric frame — fine for anchorflow, which is scale/offset
  agnostic).
- Time is normalized `[0,1]`; anchorflow uses `dt = 1/(T-1)` (integer-frame,
  like the MoSca path used `dt=1`).

### Gotchas / risks
- **`--num_frames` must equal the real T.** It is *not* stored in the checkpoint;
  pass the per-camera frame count. `t=linspace(0,1,T)` reproduces the exact
  training fids only if T matches.
- **`_node` suffix**: `--model_path` in export must point at the `..._node` dir
  SC-GS actually wrote (the driver reports it via `SCGS_CKPT`).
- **Node densification** can make final `M` differ from `--node_num`; the export
  reads the true `M` from the loaded `nodes` and realigns `hyper_dim` from its
  width, so this is handled.
- **Monocular bias**: SC-GS was designed for monocular video; on genuine
  multi-view rigs prefer the Neu3D loader (proper per-frame time) over the COLMAP
  loader (which derives `fid` from digits in the image name and assumes one image
  per timestep).
- Export needs a **GPU + pytorch3d + simple-knn** (imported transitively): run it
  inside the SC-GS image, not on the arm64 host.

---

## 4. Docker image (`docker/Dockerfile.scgs`)

Separate image, analogous to `docker/Dockerfile.mosca`. Base on a CUDA-devel
PyTorch image (`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`,
`TORCH_CUDA_ARCH_LIST="8.6;8.9"`, `FORCE_CUDA=1`). Must install:

1. **SC-GS repo** (with submodules):
   `git clone --recursive https://github.com/yihua7/SC-GS /opt/SC-GS`.
2. **CUDA rasterizer** — SC-GS's submodule
   `submodules/diff-gaussian-rasterization` (the **ashawkey fork**, adds depth +
   alpha rendering; NOT the vanilla INRIA one): `pip install ./submodules/diff-gaussian-rasterization`.
3. **simple-knn** — `submodules/simple-knn` (gitlab bkerbl): `pip install ./submodules/simple-knn`.
4. **pytorch3d** — REQUIRED (knn_points, ball_query, cot_laplacian). Use the
   prebuilt wheel for the torch/py/cu combo (as in `Dockerfile.mosca`:
   `-f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html`).
5. **`requirements.txt`** — `opencv-python`, `Pillow`, `PyYAML`, `scipy`, `tqdm`,
   `imageio`, `plyfile`, `piq`, `dearpygui`, `lpips`, `pytorch_msssim`,
   `matplotlib`, `scikit-image` (train_gui/train import `piq.LPIPS`,
   `pytorch_msssim.ms_ssim`, `dearpygui` at module top).
6. **Optional** (only if enabled, safe to skip for the default `node`+MLP recipe):
   - `tinycudann` — only for `--use_hash` (`HashDeformNetwork`).
   - `torch_batch_svd` — optional SVD speedup; falls back to `torch.svd` if absent.
7. Env: `ENV PYTHONPATH=/opt/SC-GS`, `WORKDIR /workspace`. System apt:
   `git build-essential ninja-build libglm-dev ffmpeg libgl1 libglib2.0-0 curl tmux`.

Build on GitHub Actions / x86 (never on the arm64 host).
