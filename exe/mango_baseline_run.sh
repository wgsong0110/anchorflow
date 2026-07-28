#!/usr/bin/env bash
# ============================================================================
# mango_baseline_run.sh -- official Mango-GS (Huang et al., ICLR 2026,
# https://github.com/htx0601/Mango-GS) node/anchor baseline for a
# D-NeRF-style dataset (transforms_*.json + white-background RGBA frames),
# e.g. ficus_ds_wind. Second anchor-based comparison point for AnchorFlow's
# GNN+SSM dynamics, alongside scgs_baseline_run.sh.
#
# Uses the official repo's own train.py entry point unmodified -- no glue
# code, no forked training loop. Every flag below was chosen by reading the
# actual repo source (arguments/__init__video.py, trainer.py) on 2026-07-28,
# not guessed -- read the comments before changing anything.
# ============================================================================
set -euo pipefail

WS="${WS:?set WS=<dataset dir containing transforms_train.json>}"
MANGO_ROOT="${MANGO_ROOT:-/workspace/Mango-GS}"
MODEL_PATH="${MODEL_PATH:-/workspace/ficus_mango}"
NODE_NUM="${NODE_NUM:-512}"     # matches scgs_baseline_run.sh's NODE_NUM for apples-to-apples
HYPER_DIM="${HYPER_DIM:-8}"     # matches scgs_baseline_run.sh's HYPER_DIM
ITERS="${ITERS:-60000}"        # matches scgs_baseline_run.sh's ITERS
RESOLUTION="${RESOLUTION:-1}"  # matches scgs_baseline_run.sh's RESOLUTION

# ── Lesson 1: clone + build submodules fresh, same convention as SC-GS ─────
# Mango-GS's own diff-gaussian-rasterization/simple-knn forks live under its
# submodules/ dir -- do NOT reuse SC-GS's built wheels, build Mango-GS's own
# copies against whatever torch is on this instance (FORCE_CUDA + arch list
# already set in the image env for exactly this kind of from-source build).
if [ ! -d "$MANGO_ROOT" ]; then
  git clone https://github.com/htx0601/Mango-GS "$MANGO_ROOT"
  cd "$MANGO_ROOT" && git submodule update --init --recursive
  pip install --no-cache-dir -r requirements.txt
  # Not in requirements.txt but imported by the repo (verified by reading
  # source, not guessed) -- PyYAML/matplotlib/scikit-image for config +
  # eval-plot utilities; skip if already satisfied by the base image.
  pip install --no-cache-dir PyYAML matplotlib scikit-image
  pip install --no-cache-dir -e submodules/diff-gaussian-rasterization
  pip install --no-cache-dir -e submodules/simple-knn
fi
cd "$MANGO_ROOT"

# ── Lesson 2: --no_profile, set every param explicitly ─────────────────────
# configs/profiles/global.json unconditionally applies iterations=24000,
# resolution=2, W/H=800 (tuned for HyperNeRF's cropped captures) before any
# dataset/scene profile lookup -- and ficus_ds_wind matches neither the n3v
# nor hypernerf scene-name allowlist in _detect_dataset_and_scene(), so it
# would silently fall back to "default" (no profile) while the always-on
# global.json values still clobber our own iterations/resolution. --no_profile
# disables the whole chain so only our explicit flags below + the repo's raw
# OptimizationParams defaults apply -- which is what we want for a genuine
# apples-to-apples comparison against the SC-GS baseline.
#
# ── Lesson 3: densify_until_iter / oneupSHdegree_step need NO override ──────
# Mango-GS's raw OptimizationParams defaults already are
# densify_until_iter=50_000, oneupSHdegree_step=1000 -- exactly the values
# scgs_baseline_run.sh had to manually patch into train_sc_anchorflow.py
# after finding the bug there. No fix needed here; do not add unnecessary
# overrides.
#
# ── Lesson 4: opacity_reset_interval must clear the node-bootstrap window ──
# Same failure family as SC-GS's Lesson-3 "reshape 0 elements" crash (see
# scgs_baseline_run.sh): Mango-GS runs its own node-bootstrap phase for
# `iterations_node_rendering` steps (default 12000) before switching to the
# main Gaussian training loop, and reset_opacity() firing mid-bootstrap risks
# pruning every bootstrap Gaussian if cameras_extent is small (true for
# ficus_ds_wind, same as it was for SC-GS). Push opacity_reset_interval to
# 15000 -- past the whole 12000-step bootstrap window -- as a preventive
# measure, not a reactive fix (this exact crash hasn't been hit here yet,
# but the risk is structurally identical to the one already hit in SC-GS).
OPACITY_RESET=15000

echo "[mango_baseline_run] node_num=$NODE_NUM hyper_dim=$HYPER_DIM iters=$ITERS opacity_reset=$OPACITY_RESET"

# ── Direct-to-R2 backup, not a manual afterthought ─────────────────────────
# Same convention as scgs_baseline_run.sh -- push checkpoints/logs to R2 as
# they're written, in the background, for the whole duration of training.
# deform_type defaults to 'mango_node', auto-appended to model_path by the
# repo itself (arguments/__init__video.py ModelParams.extract()).
ACTUAL_MODEL_PATH="${MODEL_PATH}_mango_node"
R2_DEST="r2:storage/result/anchorflow/$(basename "$ACTUAL_MODEL_PATH")/"
(
  while true; do
    sleep 120
    if ! rclone copy "$ACTUAL_MODEL_PATH" "$R2_DEST" --exclude "*.tmp" 2>/tmp/mango_r2_backup.err; then
      echo "[mango_baseline_run] WARNING: R2 backup failed: $(cat /tmp/mango_r2_backup.err)" >&2
    fi
  done
) &
BACKUP_PID=$!
trap 'kill $BACKUP_PID 2>/dev/null || true; rclone copy "$ACTUAL_MODEL_PATH" "$R2_DEST" 2>&1 || true' EXIT

python train.py \
  --no_profile \
  --source_path "$WS" \
  --model_path "$MODEL_PATH" \
  --resolution "$RESOLUTION" \
  --white_background \
  --is_blender \
  --gt_alpha_mask_as_scene_mask \
  --hyper_dim "$HYPER_DIM" \
  --node_num "$NODE_NUM" \
  --opacity_reset_interval "$OPACITY_RESET" \
  --iterations "$ITERS" \
  --save_iterations 10000 20000 30000 40000 50000 "$ITERS"
