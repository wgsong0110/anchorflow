# MoSca run (`exe/mosca_run.sh`)

Driver that turns a folder of RGB video frames into a MoSca 4D reconstruction,
producing the `photometric_d_model_native_add3.pth` checkpoint that
`exe/mosca_export.py` reads (`DynSCFGaussian.load_from_ckpt`, giving
`scf._node_xyz` = `[T, M, 3]` motion-scaffold node trajectory).

Faithful to the upstream repo [JiahuiLei/MoSca](https://github.com/JiahuiLei/MoSca)
(pre-release, 2024-11-28). Runs inside our MoSca GPU image where the repo lives
at `/opt/MoSca` (`PYTHONPATH=/opt/MoSca`, `GS_BACKEND=native_add3`, torch
2.1.0/cu118, render CUDA ext + PyG + pytorch3d + `requirements.txt` all
pre-installed). The driver only downloads missing prior weights and runs the
pipeline — it does **not** build/install anything.

## Input / output

- Input: `$WS/images/00000.png .. NNNNN.png` (jpg also accepted). MoSca reads a
  plain sorted list of frames — this is exactly MoSca's own demo layout
  (`demo/<seq>/images/*.jpg`).
- Output checkpoint:
  `$WS/logs/demo_fit_native_add3_<YYYYMMDD_HHMMSS>/photometric_d_model_native_add3.pth`
  The run dir name is `<exp_name>_<gs_backend>_<datetime>` where `exp_name:
  demo_fit` comes from the fit yaml (`recon_utils.py:setup_recon_ws`); the
  `.pth` is saved by `lib_mosca/photo_recon.py:1144`
  (`{phase_name}_d_model_{GS_BACKEND.lower()}.pth`, `phase_name="photometric"`).
- The script prints the resolved path as `MOSCA_CKPT=<path>`.

Run: `WS=/workspace/svd_out bash exe/mosca_run.sh`

## Exact command sequence (what the driver runs)

Mirrors MoSca's `readme.md` "Run the Full Pipeline" and `example.sh`
(specifically the object-centric `breakdance-flare` / `train` recipe:
`--dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0`):

```bash
cd /opt/MoSca                       # src-backup step cp -r's relative repo dirs
export GS_BACKEND=native_add3
export CUDA_VISIBLE_DEVICES=0

# STAGE 1 — off-the-shelf 2D priors
python mosca_precompute.py \
    --cfg /opt/MoSca/profile/demo/demo_prep.yaml \
    --ws  "$WS" \
    --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0

# STAGE 2 — 4D fit
python mosca_reconstruct.py \
    --cfg /opt/MoSca/profile/demo/demo_fit.yaml \
    --ws  "$WS" --no_viz
```

### Stage 1 — `mosca_precompute.py` (`MoCaPrep.process`)
1. **Depth** (`compute_depth`): default `dep_mode=uni` → UniDepth (metric),
   auto-loaded via `torch.hub.load("lpiccinelli-eth/UniDepth", ...)`. Alternatives
   `metric3d` (`torch.hub yvanyin/metric3d`), `depthcrafter` (HF
   `tencent/DepthCrafter` + `stabilityai/stable-video-diffusion-img2vid-xt`; the
   yaml default, but heavy and needs metric alignment — we override to `uni`).
2. **Optical flow + epipolar error** (`compute_flow`): RAFT (`raft-things.pth`),
   `flow_steps: [1, 3]`; epipolar error per track drives the dynamic-region mask
   (no SAM/segmentation needed — sky/segformer is only used if
   `depth_exclude_sky=True`, which is off).
3. **Long-term 2D tracks** (`compute_tap`): default `tap_mode=bootstapir`
   (PyTorch TAPIR in `lib_prior/tracking/tapnet_pt`, `bootstapir_checkpoint_v2.pt`).
   First a uniform 16384-track pass, then a dynamic resample over the
   epipolar-dynamic mask (skipped only with `--skip_dynamic_resample`, which we
   do NOT pass — the full MoSca path needs the dynamic tracks). Alternatives:
   `spatracker` (`spaT_final.pth`, the yaml default), `cotracker` (`torch.hub`).

All artifacts (`*_depth/`, `flow/`, `epi/`, `*_tap.npz`,
`epi_resample_mask.gif`, `bundle/`) are written into `$WS`.

### Stage 2 — `mosca_reconstruct.py` (four sub-stages, run in order in `main`)
1. `static_reconstruct` — MoCa tracklet **bundle adjustment**: solves camera
   intrinsics (FoV search 20–90°, `iso_focal`) + poses + per-frame depth
   alignment from static tracks → `bundle/bundle_cams.pth`, `bundle/bundle.pth`.
2. `photometric_warmup` — optional static-background Gaussian warmup (returns
   early if not configured in the yaml).
3. `scaffold_reconstruct` — builds the **4D Motion Scaffold**: identifies
   dynamic tracks, lifts them to node curves, ARAP/geo regularized fit
   (`geo_mosca_steps: 4000`) → `mosca/mosca.pth`.
4. `photometric_reconstruct` — joint **photometric dynamic Gaussian fitting**
   (`photo_total_steps: 8000`): static + dynamic Gaussians rendered through
   `native_add3` rasterizer, RGB + depth + normal + track losses, skinning to
   scaffold nodes → saves `photometric_s_model_*.pth`, `photometric_cam.pth`,
   **`photometric_d_model_native_add3.pth`** (the file we export).

After saving, `main` runs an eval/FPS pass. For `mode: wild` (our fit yaml)
there is no GT eval — only `test_fps`. The driver runs stage 2 with `set +e` and
then verifies the checkpoint on disk, so a crash in that post-save pass does not
lose the reconstruction.

## Prior weights

| Model | File / id | Where | How |
|---|---|---|---|
| RAFT (flow) | `raft-things.pth` | `/opt/MoSca/weights/raft_models/` | gdrive bundle |
| BootsTAPIR (tracks) | `bootstapir_checkpoint_v2.pt` | `/opt/MoSca/weights/tapnet/` | gdrive bundle |
| SpaTracker (alt tracks) | `spaT_final.pth` | `/opt/MoSca/weights/` | gdrive bundle |
| UniDepth (depth) | `lpiccinelli-eth/UniDepth` | `TORCH_HOME` | `torch.hub`, auto @ runtime |
| Metric3D (alt depth) | `yvanyin/metric3d` | `TORCH_HOME` | `torch.hub`, auto @ runtime |
| DepthCrafter (alt depth) | `tencent/DepthCrafter` + SVD | HF cache | `from_pretrained`, auto @ runtime |

The RAFT/SpaTracker/BootsTAPIR **bundle is one gdrive zip**, id
`15tveiv7ZkvBBAN3qkkB7Zfky9d7vSqLD` (MoSca `readme.md` Install step 2), expected
layout:

```
/opt/MoSca/weights
├── raft_models/raft-things.pth
├── spaT_final.pth
└── tapnet/bootstapir_checkpoint_v2.pt
```

The driver downloads it with `gdown --fuzzy` only when the needed file is
missing, unzips to a temp dir, and copies each checkpoint to its exact path
(robust to the zip's internal folder layout). UniDepth downloads itself on first
run (needs network + writable `TORCH_HOME`). License: by downloading you accept
the upstream RAFT / SpaTracker / TAPNet licenses.

## Config knobs

- Default profiles: `profile/demo/demo_prep.yaml` + `profile/demo/demo_fit.yaml`
  (MoSca's casual-video demo configs; `mode: wild`). Override with `PREP_CFG` /
  `FIT_CFG` env vars.
- Prior choice via env: `DEP_MODE` (default `uni`), `TAP_MODE` (default
  `bootstapir`), `BOUNDARY_ENHANCE_TH` (default `-1.0`). These override the yaml
  (`dep_mode: depthcrafter`, `tap_mode: spatracker`, `boundary_enhance_th: 1.0`)
  via MoSca's OmegaConf dotlist CLI merge. We pick `uni + bootstapir` because it
  is MoSca's own recipe for object-centric clips (`example.sh` breakdance/train),
  is metric + robust, and avoids the heavy DepthCrafter+SVD download.
- Object-centric vs scene: `boundary_enhance_th=-1.0` disables the SpaTracker
  boundary enhancement (recommended off for BootsTAPIR); `dyn_id_cnt: 4`,
  `epi_th`/`ba_epi_th` govern dynamic/static track split; `gs_dynamic_n_init:
  30000`, `gs_static_n_init: 80000`, `gs_max_node_num: 10000` cap Gaussians/nodes.
- Set `--skip_dynamic_resample` only for the MoCa-only (camera-only) sub-module —
  do NOT use it here; the full MoSca 4D fit needs the dynamic tracks.

## Runtime & resources

- ~8000 photometric steps + 4000 scaffold steps + BA. On a single modern GPU
  (A100/RTX 4090) expect roughly **20–45 min** end-to-end for a ~25-frame clip;
  first run adds prior-weight + UniDepth download time.
- GPU memory: comfortably fits a 25×1024×576 clip; longer/higher-res clips scale
  the tracker and rasterizer memory.

## Known gotchas

- **Run from `/opt/MoSca`.** `setup_recon_ws` backs up source with `cp -r
  profile lib_prior ...` using **relative** paths from cwd; running elsewhere
  breaks the backup. The driver `cd`s there.
- **`GS_BACKEND` must be set before import** — it is read at module import
  (`lib_render/render_helper.py`) and is baked into the checkpoint filename
  (`photometric_d_model_native_add3.pth`). The driver exports `native_add3`.
- **Weights are found via paths relative to the wrapper `__file__`**
  (`.../weights/raft_models/...`, `.../weights/tapnet/...`), i.e. under
  `/opt/MoSca/weights`, independent of cwd — so the bundle must land exactly
  there.
- **Frame count**: needs enough frames for BA/tracking; the driver requires ≥4
  and warns below that. 25 frames is fine.
- **Post-fit eval pass** can error on `mode: wild` custom clips (no GT); the
  checkpoint is already saved before it, and the driver tolerates a non-zero
  exit as long as the `.pth` exists.
- **`min_valid_cnt` / dynamic tracks**: if the object barely moves relative to a
  near-static camera, few tracks are flagged dynamic; tune `epi_th`,
  `tap_loading_min_valid_cnt`, or force more dynamic tracks if the scaffold comes
  out empty.
- DepthCrafter (`dep_mode=depthcrafter`, the yaml default we override) pulls the
  full SVD img2vid diffusion weights and needs metric alignment — avoid unless
  UniDepth depth is poor.
