#!/usr/bin/env bash
# ============================================================================
# scgs_run.sh -- drive yihua7/SC-GS to reconstruct a 4D scene from a multi-view
# video dataset, producing the sparse-control-node checkpoint that anchorflow's
# exe/scgs_export.py consumes.
#
# Runs INSIDE the SC-GS GPU image where the repo is installed at /opt/SC-GS
# (PYTHONPATH=/opt/SC-GS, torch/cu, diff-gaussian-rasterization [ashawkey fork,
# depth+alpha] + simple-knn + pytorch3d + requirements.txt pre-installed). This
# script does NOT build/install anything -- it just runs SC-GS's own
# train_gui.py in terminal mode (no --gui).
#
# SC-GS represents the scene as a canonical 3DGS + sparse control nodes
# (ControlNodeWarp) driven by a deformation MLP; dense Gaussians are bound to the
# K-NN control nodes by an RBF/LBS weight. Checkpoints are saved as TWO pieces
# under <model_path> (Scene.save + DeformModel.save_weights):
#     point_cloud/iteration_<it>/point_cloud.ply   canonical dense 3DGS
#     deform/iteration_<it>/deform.pth             control nodes + deform MLP
# plus cfg_args (the exact flags). scgs_export.py reads that whole dir.
#
# Input : $WS holding a dataset in ONE of SC-GS's auto-detected layouts:
#   * SV4D multi-view video (transforms_*.json, THE v3 CONTRACT -- what
#     exe/gen_mv_video.py emits and what this script targets):
#         $WS/transforms_train.json   D-NeRF/blender: per-frame transform_matrix
#                                     (OpenGL c2w) + per-frame `time` in [0,1]
#         $WS/transforms_test.json    (copy of train)
#         $WS/images/view_XX/*.png    per-view posed frame sequence
#     Read by SC-GS's Blender loader (scene/dataset_readers.py:readCamerasFromTransforms
#     via sceneLoadTypeCallbacks["Blender"]; scene/__init__.py detects it on
#     transforms_train.json). That loader sets fid = frame['time'] when present, so
#     multi-view video with genuine per-frame time works WITHOUT --is_blender.
#     >>> Keep IS_BLENDER=0 for SV4D multi-view <<< (control nodes then init from the
#     point cloud, hyper_dim=2, time freq 10). Set IS_BLENDER=1 ONLY for true D-NeRF
#     synthetic single-object 360 scenes (random node init, hyper_dim=8, freq 6,
#     alpha-mask recipe).
#   * Neu3D / plenoptic (alt multi-view): $WS/poses_bounds.npy (LLFF, one row per
#     camera) + $WS/frames/<camXX>/*.png  [+ optional points3D.ply]. NOTE its reader
#     force-holds-out camera 0 (hold_id=[0]) under --eval, wasting the input view.
#   * COLMAP  : $WS/sparse/0/*.bin  + $WS/images/*   (monocular-style; fid from
#               the digits in each image name / (N-1))
#   * CMU/PanopticSports: $WS/train_meta.json + init_pt_cld.npz
# Output: $WS/outputs/<name>_node/{point_cloud,deform,cfg_args}
#   -> printed at the end as   SCGS_CKPT=<that dir>
#
# Usage:
#     WS=/workspace/scene bash exe/scgs_run.sh
#
# Env knobs (all optional unless noted):
#     WS              workspace dir containing the dataset            (required)
#     SCGS_ROOT       SC-GS repo root                    (default /opt/SC-GS)
#     GPU_ID          CUDA device index                  (default 0)
#     NAME            run name / output subdir           (default scene)
#     NODE_NUM        number of control nodes (anchors)  (default 512)
#     HYPER_DIM       hyper-coord dim (2 for real scenes,8 for D-NeRF) (default 2)
#     ITERS           training iterations (main phase)   (default 30000)
#                     A good SC-GS reconstruction does NOT need 90000; 30000 is a
#                     solid default that finishes far faster. Raise for hero runs.
#     SAVE_EVERY      main-phase checkpoint cadence       (default 2000)
#                     Saves point_cloud + deform + resume_state.json every SAVE_EVERY
#                     main-phase iters, so a preempted run never loses > SAVE_EVERY.
#     RESOLUTION      image downscale factor             (default 2)
#     NUM_FRAMES      T for Neu3D (# frames/cam); patches the hard-coded 24
#                     in scene/__init__.py               (default: auto-count)
#     IS_BLENDER      set to 1 for D-NeRF synthetic      (default 0)
#     EXTRA_ARGS      extra flags appended to train_gui.py verbatim
# ============================================================================
set -euo pipefail

WS="${WS:?set WS=<workspace dir containing the dataset>}"
SCGS_ROOT="${SCGS_ROOT:-/opt/SC-GS}"
GPU_ID="${GPU_ID:-0}"
NAME="${NAME:-scene}"
NODE_NUM="${NODE_NUM:-512}"
HYPER_DIM="${HYPER_DIM:-2}"
ITERS="${ITERS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
RESOLUTION="${RESOLUTION:-2}"
IS_BLENDER="${IS_BLENDER:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$SCGS_ROOT:${PYTHONPATH:-}"
WS="$(cd "$WS" && pwd)"
MODEL_PATH="$WS/outputs/$NAME"          # SC-GS auto-appends "_node" -> $NAME_node

log(){ echo -e "\n\033[1;36m[scgs_run]\033[0m $*"; }
log "SC-GS root : $SCGS_ROOT"
log "workspace  : $WS"
log "output     : ${MODEL_PATH}_node   | GPU: $GPU_ID"

# ---------------------------------------------------------------------------
# 1. Detect dataset layout (mirrors scene/__init__.py auto-detection).
# ---------------------------------------------------------------------------
if   [ -f "$WS/poses_bounds.npy" ];      then FMT="neu3d"
elif [ -d "$WS/sparse" ] || [ -d "$WS/colmap_sparse" ]; then FMT="colmap"
elif [ -f "$WS/transforms_train.json" ]; then FMT="blender"
elif [ -f "$WS/train_meta.json" ];       then FMT="cmu"
elif [ -f "$WS/dataset.json" ];          then FMT="nerfies"
else echo "ERROR: no recognized SC-GS dataset under $WS" >&2; exit 1; fi
log "detected dataset format: $FMT"

# transforms_*.json is the v3 SV4D handoff -> SC-GS's Blender loader
# (readCamerasFromTransforms) reads per-frame `time`. Make the --is_blender choice
# loud: OFF = SV4D multi-view recipe (correct for gen_mv_video output); ON = true
# D-NeRF synthetic single-object recipe.
if [ "$FMT" = "blender" ]; then
  if [ "$IS_BLENDER" = "1" ]; then
    log "transforms dataset + IS_BLENDER=1 -> D-NeRF SYNTHETIC recipe (--is_blender)."
  else
    log "transforms dataset + IS_BLENDER=0 -> SV4D MULTI-VIEW recipe (per-frame time via Blender loader, --is_blender OFF)."
  fi
fi

# ---------------------------------------------------------------------------
# 2. Neu3D only: patch the hard-coded num_images=24 in scene/__init__.py to the
#    real per-camera frame count so all T timesteps are used. Idempotent.
#    (Scene calls  sceneLoadTypeCallbacks["plenopticVideo"](path, eval, 24).)
# ---------------------------------------------------------------------------
if [ "$FMT" = "neu3d" ]; then
  if [ -z "${NUM_FRAMES:-}" ]; then
    CAM0="$(ls -d "$WS"/frames/*/ 2>/dev/null | head -n1 || true)"
    NUM_FRAMES="$(find "$CAM0" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)"
  fi
  if [ "${NUM_FRAMES:-0}" -gt 1 ]; then
    log "patching plenopticVideo num_images -> $NUM_FRAMES in scene/__init__.py"
    sed -i -E "s/(sceneLoadTypeCallbacks\[\"plenopticVideo\"\]\([^,]+, *args\.eval, *)[0-9]+\)/\1${NUM_FRAMES})/" \
        "$SCGS_ROOT/scene/__init__.py"
  fi
fi

# ---------------------------------------------------------------------------
# 2b. Headless: train_gui.py hard-imports dearpygui at module load, but every
#     dpg.* call is guarded by `if self.gui:` (we run WITHOUT --gui). dearpygui's
#     prebuilt .so needs a newer GLIBCXX than the conda base ships, so the import
#     alone crashes before training starts. Guard it (stub dpg=None). Idempotent —
#     removing the unused import, not installing libs to satisfy it.
# ---------------------------------------------------------------------------
if grep -q '^import dearpygui.dearpygui as dpg' "$SCGS_ROOT/train_gui.py"; then
  sed -i 's/^import dearpygui.dearpygui as dpg/try:\n    import dearpygui.dearpygui as dpg\nexcept Exception:\n    dpg = None/' \
      "$SCGS_ROOT/train_gui.py"
  log "guarded dearpygui import in train_gui.py (headless; GUI unused)"
fi

# SC-GS's blender/transforms loader builds the image with dtype=np.byte (signed
# int8) -> 255 overflows AND PIL can't map int8 RGBA ("Cannot handle (1,1,4) |i1").
# Must be uint8. Idempotent one-char fix to dataset_readers.py.
if grep -q 'dtype=np.byte' "$SCGS_ROOT/scene/dataset_readers.py"; then
  sed -i 's/dtype=np\.byte/dtype=np.uint8/g' "$SCGS_ROOT/scene/dataset_readers.py"
  log "patched dataset_readers.py np.byte -> np.uint8 (RGBA image build)"
fi

# ---------------------------------------------------------------------------
# 2c. Crash / preemption-safe RESUME. Instances can be stopped at any time, so
#     make SC-GS's train_gui.py: (a) reconcile partial checkpoints so the two
#     independent loaders (Scene point_cloud + DeformModel deform) always agree
#     on the same fully-written iteration, (b) write resume_state.json as an
#     atomic commit marker at each save, and (c) drive the main phase to
#     opt.iterations by self.iteration (not a fixed step count) so a restart
#     continues to the target instead of overshooting. scgs_resume_patch.py is
#     idempotent (guarded by '# [anchorflow-resume]' markers) -- safe every run.
# ---------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/scgs_resume_patch.py" "$SCGS_ROOT/train_gui.py"
log "ensured crash-safe resume patch on train_gui.py (idempotent)"

# ---------------------------------------------------------------------------
# 3. Assemble train_gui.py flags.
#    Multi-view / real scenes: NO --is_blender (nodes init from the point cloud,
#    not random), NO --gt_alpha_mask_as_scene_mask. --eval holds out a test view
#    (Neu3D: camera 0). --init_isotropic_gs_with_all_colmap_pcl guards against
#    control-node init failure on self-captured scenes (readme.md "2024-03-06").
# ---------------------------------------------------------------------------
# Dense --save_iterations so a preemption never costs more than SAVE_EVERY main-
# phase iters. These refer to the MAIN-phase self.iteration (the node-bootstrap
# phase is never checkpointed). SC-GS auto-appends args.iterations too.
SAVE_ITERS=""
i="$SAVE_EVERY"
while [ "$i" -lt "$ITERS" ]; do SAVE_ITERS="$SAVE_ITERS $i"; i=$((i + SAVE_EVERY)); done
SAVE_ITERS="$SAVE_ITERS $ITERS"
log "save cadence: every $SAVE_EVERY main-phase iters ->$SAVE_ITERS"

ARGS=( --source_path "$WS" --model_path "$MODEL_PATH"
       --deform_type node --node_num "$NODE_NUM" --hyper_dim "$HYPER_DIM"
       --iterations "$ITERS" --resolution "$RESOLUTION" --eval )
# shellcheck disable=SC2206
ARGS+=( --save_iterations $SAVE_ITERS )
if [ "$IS_BLENDER" = "1" ]; then
  # D-NeRF synthetic recipe (matches train_gui.sh)
  ARGS+=( --is_blender --gt_alpha_mask_as_scene_mask --local_frame --W 800 --H 800 )
else
  # Real multi-view video recipe
  ARGS+=( --init_isotropic_gs_with_all_colmap_pcl )
  # [anchorflow] BACKGROUND FIX: our SV4D multi-view frames are object-on-WHITE
  # RGB (no alpha; verified bg=255). Without --white_background SC-GS sets
  # self.background=[0,0,0] (train_gui.py:174) and renders/optimizes against a BLACK
  # bg while the GT images have a WHITE bg -> the model wastes capacity growing a
  # white "fake background" haze of Gaussians, opacity collapses (mean~0.02), the
  # object never forms, and renders come out blank (mixed black/white) at PSNR~14.
  # Matching the render bg to the data (white) frees the model to fit the object.
  ARGS+=( --white_background )
  # [anchorflow] ROOT-CAUSE FIX for the node-bootstrap "reshape 0 elements" crash
  # (train_gui.py L1390 at iterations_node_sampling). It is NOT a too-few-views
  # problem. The node-bootstrap gaussians are a StandardGaussianModel(all_the_same=
  # True): get_scaling returns ONE uniform scale (_scaling.mean()) for ALL of them.
  # SC-GS's world-space big-point prune (get_scaling.max > 0.1*cameras_extent)
  # activates once iteration > opacity_reset_interval (size_threshold=20). On our
  # SV4D scenes cameras sit at radius ~2 (cameras_extent~2.05 => 0.1*extent~0.205),
  # while the uniform node scale grows to ~0.34 -> EVERY node gaussian trips
  # big_points_ws -> ALL pruned -> N=0 -> reshape crash. D-NeRF survives only
  # because its radius-~4 cameras give a larger extent. The scale/extent ratio is
  # invariant to uniform camera rescaling, so moving cameras cannot fix it; instead
  # push opacity_reset_interval past iterations_node_sampling(7500) so size_threshold
  # never activates during the bootstrap. Verified: N holds ~97k through 7500, the
  # downsample to 512 control nodes succeeds, main phase trains normally.
  ARGS+=( --opacity_reset_interval 8000 )
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=( $EXTRA_ARGS )

log "train command:"
echo "    python train_gui.py ${ARGS[*]}"

# ---------------------------------------------------------------------------
# 4. Train. Re-running the SAME command auto-resumes (crash/preemption-safe):
#    _anchorflow_reconcile_checkpoints() prunes any partial newer save, then
#    Scene(load_iteration=-1) + deform.load_weights(-1) reload the latest
#    consistent iteration under ${MODEL_PATH}_node and the main phase continues
#    to --iterations. If preempted during the (unsaved, <=10000-iter) node
#    bootstrap, that short phase simply re-runs from scratch. Just relaunch this
#    script -- no extra flags needed to resume.
# ---------------------------------------------------------------------------
cd "$SCGS_ROOT"
python train_gui.py "${ARGS[@]}"

# ---------------------------------------------------------------------------
# 5. Locate and report the checkpoint dir (what scgs_export.py consumes).
# ---------------------------------------------------------------------------
CKPT_DIR="${MODEL_PATH}_node"
LATEST_DEFORM="$(ls -d "$CKPT_DIR"/deform/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 || true)"
LATEST_PLY="$(ls -d "$CKPT_DIR"/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 || true)"
if [ -z "$LATEST_DEFORM" ] || [ -z "$LATEST_PLY" ]; then
  echo "ERROR: SC-GS did not produce deform/point_cloud checkpoints under $CKPT_DIR" >&2
  exit 1
fi
log "DONE. latest deform: $LATEST_DEFORM"
log "      latest gaussians: $LATEST_PLY"
echo "SCGS_CKPT=$CKPT_DIR"
