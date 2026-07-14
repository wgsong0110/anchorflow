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
#   * Neu3D / plenoptic (multi-view video, RECOMMENDED):
#         $WS/poses_bounds.npy                LLFF poses, one row per camera
#         $WS/frames/<camXX>/<0000.png..>     per-camera posed frame sequence
#         $WS/points3D.ply                    (optional; else random init)
#   * COLMAP  : $WS/sparse/0/*.bin  + $WS/images/*   (monocular-style; fid from
#               the digits in each image name / (N-1))
#   * D-NeRF  : $WS/transforms_train.json + frames  (synthetic; add --is_blender)
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
#     ITERS           training iterations                (default 80000)
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
ITERS="${ITERS:-80000}"
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
# 3. Assemble train_gui.py flags.
#    Multi-view / real scenes: NO --is_blender (nodes init from the point cloud,
#    not random), NO --gt_alpha_mask_as_scene_mask. --eval holds out a test view
#    (Neu3D: camera 0). --init_isotropic_gs_with_all_colmap_pcl guards against
#    control-node init failure on self-captured scenes (readme.md "2024-03-06").
# ---------------------------------------------------------------------------
ARGS=( --source_path "$WS" --model_path "$MODEL_PATH"
       --deform_type node --node_num "$NODE_NUM" --hyper_dim "$HYPER_DIM"
       --iterations "$ITERS" --resolution "$RESOLUTION" --eval )
if [ "$IS_BLENDER" = "1" ]; then
  # D-NeRF synthetic recipe (matches train_gui.sh)
  ARGS+=( --is_blender --gt_alpha_mask_as_scene_mask --local_frame --W 800 --H 800 )
else
  # Real multi-view video recipe
  ARGS+=( --init_isotropic_gs_with_all_colmap_pcl )
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=( $EXTRA_ARGS )

log "train command:"
echo "    python train_gui.py ${ARGS[*]}"

# ---------------------------------------------------------------------------
# 4. Train. SC-GS auto-resumes: Scene(load_iteration=-1) + deform.load_weights(-1)
#    reload the latest saved iteration under $MODEL_PATH_node, so re-running the
#    same command continues from the last checkpoint.
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
