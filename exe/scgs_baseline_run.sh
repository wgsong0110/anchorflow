#!/usr/bin/env bash
# ============================================================================
# scgs_baseline_run.sh -- stock SC-GS (anchor/control-node baseline, matching
# the paper's own architecture) for a D-NeRF-style dataset (transforms_*.json
# + white-background RGBA frames), e.g. ficus_ds_wind. Comparable baseline for
# AnchorFlow's GNN+SSM dynamics.
#
# Every flag/workaround here was hit and fixed the hard way in a live debug
# session (2026-07-26, ficus_ds_wind). Re-running this script should just
# work on a fresh instance -- read the comments before changing anything,
# each one documents a real crash this exact combination avoids.
# ============================================================================
set -euo pipefail

WS="${WS:?set WS=<dataset dir containing transforms_train.json>}"
SCGS_ROOT="${SCGS_ROOT:-/workspace/SC-GS}"
MODEL_PATH="${MODEL_PATH:-/workspace/ficus_scgs}"
NODE_NUM="${NODE_NUM:-512}"      # SC-GS's own D-NeRF reproduce recipe (train_gui.sh: jumpingjacks)
HYPER_DIM="${HYPER_DIM:-8}"      # 8 for D-NeRF/is_blender scenes, 2 for real multi-view
ITERS="${ITERS:-60000}"
RESOLUTION="${RESOLUTION:-1}"

cd "$SCGS_ROOT"

# ── Lesson 1: deform_type is NOT a flag here ────────────────────────────────
# SC-GS's own default (arguments/__init__.py: self.deform_type = 'node') is
# already the anchor/control-node architecture the paper describes and the
# one comparable to AnchorFlow's anchors. A prior run on this dataset passed
# --deform_type mlp by mistake and trained a completely different (dense,
# per-Gaussian, non-anchor) architecture for hours before anyone noticed --
# so this script deliberately never exposes --deform_type as a variable.

# ── Lesson 2: train.py is a helper-function MODULE, not an entry point ─────
# /workspace/SC-GS/train.py has no `if __name__ == "__main__"` -- it only
# defines prepare_output_and_logger() and training_report(), which train_gui.py
# imports (`from train import training_report`). Running `python train.py`
# with any flags silently does nothing: it parses no args, runs no training,
# and exits 0 after ~9s having only imported its dependencies. The real
# entry point -- the one with the argparse, the training loop, node
# bootstrapping, everything -- is train_gui.py. Use --gui is NOT required;
# it runs fine in terminal-only mode without it. Note: its
# `import dearpygui.dearpygui as dpg` at module top is NOT guarded (an
# earlier version of this comment claimed otherwise) -- dearpygui must be
# installed even when --gui is never passed, or the import crashes before
# argparse runs. It's in the Dockerfile now; if running on an older image,
# `pip install dearpygui` first.
ENTRY="train_gui.py"

# ── Lesson 3: node-bootstrap "reshape 0 elements" crash ────────────────────
# Traceback: train_gui.py train_node_rendering_step(), "cannot reshape tensor
# of 0 elements into shape [0, -1]". Root cause: the node-bootstrap Gaussians
# are uniform-scale (StandardGaussianModel), and SC-GS's world-space
# big-point prune (get_scaling.max > 0.1*cameras_extent) activates once
# iteration > opacity_reset_interval. If cameras_extent is small (true for
# ficus_ds_wind, unlike stock D-NeRF's radius-~4 cameras), the uniform node
# scale outgrows 0.1*extent before the ~7500-iter node-sampling bootstrap
# finishes downsampling to node_num control nodes -> every bootstrap
# Gaussian gets pruned -> N=0 -> the reshape crash. Fix: push the first
# opacity reset past the bootstrap window.
OPACITY_RESET=8000

# ── Lesson 4: --local_frame crashes main-phase rendering on this dataset ───
# Traceback: RuntimeError: Sizes of tensors must match except in dimension 1.
# Expected size 1 but got size 0 -- in GaussianModel.get_features, reached
# from render()'s d_color path when --local_frame is set. Do not pass
# --local_frame here (SC-GS's default local_frame=False still trains full
# per-node rotation via d_rot_as_res; --local_frame only adds an extra
# local-rotation refinement head that this dataset's Gaussian init doesn't
# support).

# ── Lesson 5: pkill -f <script.py> can kill its own invoking shell ─────────
# `ssh host 'pkill -9 -f train_gui.py'` matches train_gui.py as a SUBSTRING
# of the ssh command's own argv (the command string itself contains
# "train_gui.py"), so pkill -f killed the shell running the kill command --
# i.e. it took down the whole SSH session, not just the target process. Kill
# by exact PID (`ps aux | grep "python train_gui" | awk '{print $2}' | xargs
# kill -9`) instead of `pkill -f <name that also appears in your own command
# line>`.

# ── Lesson 6: tmux server on some vast.ai hosts dies unpredictably ─────────
# Seen mid-session: "no server running on /tmp/tmux-0/default" for a session
# that was alive seconds earlier -- not caused by anything this script did.
# For anything long-running, prefer `setsid nohup ... & disown` over tmux so
# the job survives even if tmux's own server process gets reaped:
#   setsid nohup bash scgs_baseline_run.sh > train.log 2>&1 < /dev/null &
#   disown

echo "[scgs_baseline_run] entry=$ENTRY node_num=$NODE_NUM hyper_dim=$HYPER_DIM iters=$ITERS opacity_reset=$OPACITY_RESET"

# ── Lesson 7: SC-GS silently appends "_<deform_type>" to model_path ────────
# train_gui.py: `if not args.model_path.endswith(args.deform_type):
# args.model_path = ... + f'_{args.deform_type}'`. deform_type defaults to
# 'node' (see Lesson 1) and is never overridden here, so the real output dir
# is always "${MODEL_PATH}_node", not the bare $MODEL_PATH passed in below --
# the R2 backup loop must target that actual path or it backs up nothing
# (silently retried-and-failed every 120s: "directory not found").
ACTUAL_MODEL_PATH="${MODEL_PATH}_node"

# ── Direct-to-R2 backup, not a manual afterthought ─────────────────────────
# Instances can be destroyed at any time -- push checkpoints/logs to R2 as
# they're written, in the background, for the whole duration of training
# (not just once at the end). Loud on failure (stderr), matching the
# convention already fixed into train_sc_anchorflow.py/train_hop_*.py after
# a checkpoint got lost this same session from relying on a manual step.
# Lesson 8: exclude tfevents from the periodic copy -- rclone refuses to
# copy a source file mid-transfer if its size changes (a real safety check,
# not a bug in rclone), and the tensorboard SummaryWriter appends to
# events.out.tfevents.* continuously while training runs, so every 120s
# cycle "fails" on it forever (loud WARNING spam, but harmless -- the actual
# checkpoints/point clouds aren't being written to during the copy window
# and always succeed). Exclude it from the periodic loop; the final EXIT-trap
# copy still captures whatever tfevents state exists at shutdown.
R2_DEST="r2:storage/result/anchorflow/$(basename "$ACTUAL_MODEL_PATH")/"
(
  while true; do
    sleep 120
    if ! rclone copy "$ACTUAL_MODEL_PATH" "$R2_DEST" --exclude "*.tmp" --exclude "events.out.tfevents.*" 2>/tmp/scgs_r2_backup.err; then
      echo "[scgs_baseline_run] WARNING: R2 backup failed: $(cat /tmp/scgs_r2_backup.err)" >&2
    fi
  done
) &
BACKUP_PID=$!
trap 'kill $BACKUP_PID 2>/dev/null || true; rclone copy "$ACTUAL_MODEL_PATH" "$R2_DEST" 2>&1 || true' EXIT

python "$ENTRY" \
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
