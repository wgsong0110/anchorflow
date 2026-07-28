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

  # Lesson 1b: is_blender noise-tensor rank bug -- crashes on first bootstrap
  # step with "IndexError: tuple index out of range" at mango/time_utils.py
  # forward()'s `t.shape[1]`. Root cause: trainer.py's is_blender branch
  # builds `noise` with one fewer dim than the non-blender branch (e.g.
  # torch.zeros(1) vs torch.randn(1,1)), so `t = time_input + noise` stays
  # 1D for Blender/D-NeRF datasets (ficus_ds_wind included) instead of being
  # broadcast to 2D -- forward() unconditionally assumes t has a dim 1.
  # Two occurrences (train_node_init_step's `noise = torch.zeros(1,...)` and
  # train_node_video_step's `noise = torch.zeros(T,...)`); fix both to match
  # their non-blender sibling's rank. Idempotent (skips if already patched).
  python3 - <<'PYEOF'
path = "trainer.py"
with open(path) as f:
    src = f.read()
if "2D to match the non-blender branch" in src:
    print("  already patched, skipping")
else:
    n1 = "            noise = torch.zeros(1, device='cuda')"
    n1_new = "            noise = torch.zeros(1, 1, device='cuda')  # 2D to match the non-blender branch (torch.randn(1,1,...)) -- forward() unconditionally does t.shape[1]"
    n2 = "                    noise = torch.zeros(T, device='cuda')"
    n2_new = "                    noise = torch.zeros(1, T, device='cuda')  # 2D to match the non-blender branch (torch.randn(1,T,...))"
    ok = True
    if src.count(n1) == 1:
        src = src.replace(n1, n1_new)
    else:
        print(f"  WARNING: expected 1 match for noise patch #1, found {src.count(n1)} -- not applied")
        ok = False
    if src.count(n2) == 1:
        src = src.replace(n2, n2_new)
    else:
        print(f"  WARNING: expected 1 match for noise patch #2, found {src.count(n2)} -- not applied")
        ok = False
    if ok:
        with open(path, "w") as f:
            f.write(src)
        print("  patched OK")
PYEOF
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

# ── Lesson 5: white_background triggers a SEPARATE, un-overridable opacity
# reset tied to densify_from_iter, not opacity_reset_interval ──────────────
# trainer.py has an extra OR-condition on both its bootstrap and main-phase
# reset_opacity() calls: `... or (self.dataset.white_background and
# self.iteration[_node_rendering] == self.opt.densify_from_iter)`. Since we
# pass --white_background (required for this dataset), this fires exactly
# once in bootstrap (at iteration_node_rendering==densify_from_iter) and
# once in main phase (at iteration==densify_from_iter), REGARDLESS of the
# OPACITY_RESET override above. Confirmed on a live run (2026-07-28):
# Mango-GS's default densify_from_iter=3000 let ~3000 real main-phase
# iterations of growth happen first, then this reset fired and collapsed the
# entire scene to invisible permanently (test PSNR/SSIM frozen bit-identical
# from iteration 3000 onward, byte-identical all-white renders -- the exact
# same failure mode diagnosed on the SC-GS baseline). SC-GS's own default
# densify_from_iter=500 (much earlier) doesn't have this problem because the
# equivalent reset fires before there's anything substantial to lose.
# Fix: match SC-GS's early densify_from_iter=500 so this reset fires at the
# start (negligible risk) instead of after real training has happened.
DENSIFY_FROM_ITER=500

# ── Lesson 6: node densify/prune corrupts rendering at node_force_densify_
# prune_step (default 10000) -- a SECOND, independent collapse cause ────────
# Even with Lesson 5's fix, a live run (2026-07-28) collapsed again: opacity
# was confirmed HEALTHY and steadily improving through iteration 10000
# (point_cloud/iteration_10000.ply: sigmoid(opacity) mean 0.048->0.15->0.39->
# 0.48 across saved checkpoints, N stable at 660) -- then eval froze to the
# same all-white PSNR signature (14.843050003051758) from iteration 11000
# onward. node_force_densify_prune_step=10000 (default, unconditional, fires
# once regardless of enable_dp) and node_densify_from_iter=10000 (default,
# starts the REGULAR periodic node densify schedule) both land exactly at
# this boundary. Reading mango/time_utils.py's DeformNetworkNode.densify():
# it prunes/splits `self.gs` -- a separate GaussianModel instance built
# during node-bootstrap (DeformNetworkNode.init_gaussians()) -- and syncs
# `self.gs._xyz.data = self.nodes[..., :3]`. This object is distinct from
# the trainer's own self.gaussians (the one actually rendered in main
# phase), and nothing in the surrounding code re-syncs the two afterward --
# so a node densify/prune event during main-phase training plausibly
# desyncs the skinning weights (cal_nn_weight) from the actual render
# Gaussians without raising an error, silently breaking every subsequent
# render. Fix: disable node densify/prune entirely for this run (push both
# past the total iteration count) -- trades away adaptive node refinement
# for avoiding the corruption; node count stays fixed at whatever bootstrap
# produced (comparable to SC-GS not needing this same tradeoff since its own
# node densify apparently doesn't hit the same desync).
NODE_DENSIFY_FROM_ITER=999999
NODE_FORCE_DENSIFY_PRUNE_STEP=999999

echo "[mango_baseline_run] node_num=$NODE_NUM hyper_dim=$HYPER_DIM iters=$ITERS opacity_reset=$OPACITY_RESET densify_from_iter=$DENSIFY_FROM_ITER node_densify_from_iter=$NODE_DENSIFY_FROM_ITER"

# ── Direct-to-R2 backup, not a manual afterthought ─────────────────────────
# Same convention as scgs_baseline_run.sh -- push checkpoints/logs to R2 as
# they're written, in the background, for the whole duration of training.
# deform_type defaults to 'mango_node', auto-appended to model_path by the
# repo itself (arguments/__init__video.py ModelParams.extract()).
ACTUAL_MODEL_PATH="${MODEL_PATH}_mango_node"
# events.out.tfevents.* is excluded from the periodic copy: rclone refuses
# to copy a file mid-transfer if its size changes while reading it (a real
# safety check), and tensorboard appends to this file continuously during
# training, so it would "fail" every single 120s cycle forever otherwise
# (harmless noise -- checkpoints/point clouds aren't touched during the copy
# window and always succeed; hit and diagnosed on the SC-GS baseline run,
# 2026-07-28). The final EXIT-trap copy still captures its end-of-run state.
R2_DEST="r2:storage/result/anchorflow/$(basename "$ACTUAL_MODEL_PATH")/"
(
  while true; do
    sleep 120
    if ! rclone copy "$ACTUAL_MODEL_PATH" "$R2_DEST" --exclude "*.tmp" --exclude "events.out.tfevents.*" 2>/tmp/mango_r2_backup.err; then
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
  --densify_from_iter "$DENSIFY_FROM_ITER" \
  --node_densify_from_iter "$NODE_DENSIFY_FROM_ITER" \
  --node_force_densify_prune_step "$NODE_FORCE_DENSIFY_PRUNE_STEP" \
  --iterations "$ITERS" \
  --save_iterations 10000 20000 30000 40000 50000 "$ITERS"
