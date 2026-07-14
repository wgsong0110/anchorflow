#!/usr/bin/env bash
# ============================================================================
# mosca_run.sh -- drive JiahuiLei/MoSca to reconstruct a 4D scene from a folder
# of RGB video frames, producing the photometric dynamic-Gaussian checkpoint
# that anchorflow's exe/mosca_export.py consumes.
#
# Runs INSIDE the MoSca GPU image where the repo is already installed at
# /opt/MoSca (PYTHONPATH=/opt/MoSca, GS_BACKEND=native_add3, torch 2.1.0/cu118,
# all render CUDA ext + PyG + pytorch3d + requirements.txt pre-installed).
# This script does NOT install MoSca -- it downloads missing prior weights and
# runs MoSca's own two-stage pipeline exactly as its demo/example.sh does:
#
#     python mosca_precompute.py  --cfg profile/demo/demo_prep.yaml --ws $WS ...
#     python mosca_reconstruct.py --cfg profile/demo/demo_fit.yaml  --ws $WS
#
# Input  : $WS/images/00000.png .. NNNNN.png   (jpg also accepted)
# Output : $WS/logs/demo_fit_native_add3_<datetime>/photometric_d_model_native_add3.pth
#          -> printed at the end as   MOSCA_CKPT=<path>
#
# Usage:
#     WS=/workspace/svd_out bash exe/mosca_run.sh
#
# Env knobs (all optional, sensible defaults):
#     WS                    workspace dir containing images/        (required)
#     MOSCA_ROOT            MoSca repo root         (default /opt/MoSca)
#     GPU_ID                CUDA device index       (default 0)
#     DEP_MODE              depth prior: uni|depthcrafter|metric3d  (default uni)
#     TAP_MODE              long-track prior: bootstapir|spatracker|cotracker
#                                                   (default bootstapir)
#     BOUNDARY_ENHANCE_TH   depth-boundary enhance; -1.0 disables   (default -1.0)
#     PREP_CFG / FIT_CFG    override the profile yamls (absolute paths)
#     WEIGHTS_GDRIVE_ID     gdrive id of the RAFT/SpaTracker/TAPIR bundle
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
WS="${WS:?set WS=<workspace dir containing images/>}"
MOSCA_ROOT="${MOSCA_ROOT:-/opt/MoSca}"
GPU_ID="${GPU_ID:-0}"
DEP_MODE="${DEP_MODE:-uni}"
TAP_MODE="${TAP_MODE:-bootstapir}"
BOUNDARY_ENHANCE_TH="${BOUNDARY_ENHANCE_TH:--1.0}"
PREP_CFG="${PREP_CFG:-$MOSCA_ROOT/profile/demo/demo_prep.yaml}"
FIT_CFG="${FIT_CFG:-$MOSCA_ROOT/profile/demo/demo_fit.yaml}"
# MoSca ships these RAFT / SpaTracker / BootsTAPIR checkpoints in one gdrive zip
# (see MoSca readme.md "Install" step 2).
WEIGHTS_GDRIVE_ID="${WEIGHTS_GDRIVE_ID:-15tveiv7ZkvBBAN3qkkB7Zfky9d7vSqLD}"

# MoSca reads GS_BACKEND at import time; the checkpoint filename embeds it.
export GS_BACKEND="${GS_BACKEND:-native_add3}"
export PYTHONPATH="$MOSCA_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Let torch.hub (UniDepth / Metric3D) cache in a stable, writable place.
export TORCH_HOME="${TORCH_HOME:-$MOSCA_ROOT/.torchhub}"

# Absolute WS so MoSca's src-backup step (which cp -r's relative repo dirs from
# cwd=$MOSCA_ROOT) and the profile paths both stay unambiguous.
WS="$(cd "$WS" && pwd)"

log(){ echo -e "\n\033[1;36m[mosca_run]\033[0m $*"; }

log "MoSca root : $MOSCA_ROOT"
log "workspace  : $WS"
log "GS_BACKEND : $GS_BACKEND  | GPU: $GPU_ID  | dep=$DEP_MODE tap=$TAP_MODE"

# ---------------------------------------------------------------------------
# 1. Sanity: input frames present
# ---------------------------------------------------------------------------
if [ ! -d "$WS/images" ]; then
  echo "ERROR: $WS/images not found (expected RGB frames 00000.png ..)." >&2
  exit 1
fi
N_FRAMES=$(find "$WS/images" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' \) | wc -l)
log "found $N_FRAMES input frames in $WS/images"
if [ "$N_FRAMES" -lt 4 ]; then
  echo "ERROR: need >= a handful of frames for BA/tracking (got $N_FRAMES)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Prior weights
#    - RAFT / SpaTracker / BootsTAPIR : one gdrive zip -> $MOSCA_ROOT/weights/
#         weights/raft_models/raft-things.pth
#         weights/spaT_final.pth
#         weights/tapnet/bootstapir_checkpoint_v2.pt
#    - UniDepth / Metric3D / DepthCrafter : auto (torch.hub / HF from_pretrained)
#      at runtime; nothing to stage here.
# ---------------------------------------------------------------------------
W="$MOSCA_ROOT/weights"
RAFT_PTH="$W/raft_models/raft-things.pth"
SPAT_PTH="$W/spaT_final.pth"
TAPIR_PTH="$W/tapnet/bootstapir_checkpoint_v2.pt"

need_bundle=0
[ -f "$RAFT_PTH" ] || need_bundle=1                                   # flow=raft always needed
[ "$TAP_MODE" = "bootstapir" ] && [ ! -f "$TAPIR_PTH" ] && need_bundle=1
[ "$TAP_MODE" = "spatracker" ] && [ ! -f "$SPAT_PTH" ]  && need_bundle=1

if [ "$need_bundle" -eq 1 ]; then
  log "downloading RAFT/SpaTracker/BootsTAPIR weight bundle (gdrive $WEIGHTS_GDRIVE_ID) ..."
  mkdir -p "$W"
  TMPD="$(mktemp -d)"
  # gdown is in MoSca's requirements.txt (pre-installed).
  gdown "https://drive.google.com/uc?id=${WEIGHTS_GDRIVE_ID}" \
        -O "$TMPD/mosca_weights.zip" || gdown "${WEIGHTS_GDRIVE_ID}" -O "$TMPD/mosca_weights.zip"
  mkdir -p "$TMPD/unz"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$TMPD/mosca_weights.zip" -d "$TMPD/unz"
  else
    python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$TMPD/mosca_weights.zip" "$TMPD/unz"
  fi
  # Place each checkpoint at its exact expected path regardless of the zip's
  # internal folder layout.
  place(){ # <basename> <dest>
    local src; src="$(find "$TMPD/unz" -type f -name "$1" | head -n1)"
    if [ -n "$src" ]; then mkdir -p "$(dirname "$2")"; cp -f "$src" "$2"; \
       echo "  placed $1 -> $2"; else echo "  (not in bundle: $1)"; fi
  }
  place "raft-things.pth"                "$RAFT_PTH"
  place "spaT_final.pth"                 "$SPAT_PTH"
  place "bootstapir_checkpoint_v2.pt"    "$TAPIR_PTH"
  rm -rf "$TMPD"
else
  log "prior weight bundle already present, skipping download"
fi
[ -f "$RAFT_PTH" ] || { echo "ERROR: RAFT weight missing at $RAFT_PTH" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2b. UniDepth <-> torch 2.1 TorchScript compat patch.
#     Latest UniDepth (pulled by torch.hub `main`) annotates jit-scripted fns with
#     PEP-604 unions (`int | tuple[int,int]`), which torch 2.1's TorchScript
#     compiler rejects (eager py3.10 is fine). We (a) flip the wrapper's
#     force_reload off so a patched cache survives, (b) pre-fetch the repo into the
#     hub cache (the import error is expected & ignored -- files land first), and
#     (c) disable @torch.jit.script across the cached tree so it runs eager.
#     UniDepth is needed for dep_mode=uni AND as the metric-alignment reference.
# ---------------------------------------------------------------------------
UNIWRAP="$MOSCA_ROOT/lib_prior/depth_models/unidepth_wrapper.py"
if [ -f "$UNIWRAP" ]; then
  log "applying UniDepth<->torch2.1 TorchScript compat patch"
  sed -i 's/force_reload=True/force_reload=False/g' "$UNIWRAP"
  python -c "import torch; torch.hub.load('lpiccinelli-eth/UniDepth','UniDepth',version='v2',backbone='vitl14',pretrained=False,trust_repo=True,force_reload=True)" >/dev/null 2>&1 || true
  UNIDIR="$(ls -d "$TORCH_HOME"/hub/*UniDepth* 2>/dev/null | head -n1)"
  if [ -n "$UNIDIR" ]; then
    find "$UNIDIR" -name '*.py' -exec sed -i 's/@torch\.jit\.script/# (torch2.1-compat off) @torch.jit.script/g' {} +
    echo "  patched jit.script in $UNIDIR"
  else
    echo "  WARN: UniDepth cache dir not found after prefetch (will retry live)"
  fi
fi

# ---------------------------------------------------------------------------
# 2c. Materialize in-repo symlinks. MoSca ships source symlinks (e.g.
#     lib_mosca/camera.py -> ../lib_moca/camera.py). Our image `git clone` stored
#     them as PLAIN-TEXT files (core.symlinks off on the builder), so Python reads
#     the link target as code -> SyntaxError. Rebuild every git symlink (mode
#     120000) from its stored target path. Idempotent.
# ---------------------------------------------------------------------------
cd "$MOSCA_ROOT"
# MoSca commits these as tiny TEXT files whose whole content is the link target
# (e.g. lib_mosca/camera.py = "../lib_moca/camera.py"); a plain `git clone` leaves
# them as-is. Detect by content: a small file whose sole line is a relative path
# that resolves to an existing file -> replace with a real symlink. Idempotent
# (already-symlinked files are skipped by the -type f filter).
n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  tgt="$(tr -d '\n' < "$f")"
  case "$tgt" in
    ./*|../*)
      d="$(dirname "$f")"
      if [ -e "$d/$tgt" ] && [ "$(wc -l < "$f")" -le 1 ]; then
        ln -sfn "$tgt" "$f" && n=$((n+1))
      fi ;;
  esac
done < <(find "$MOSCA_ROOT/lib_mosca" "$MOSCA_ROOT/lib_moca" "$MOSCA_ROOT/lib_render" \
              -type f -name '*.py' -size -128c 2>/dev/null)
log "materialized $n in-repo pointer files as symlinks"

# Pure-python runtime deps the reconstruct chain imports but that are missing from
# the image: lpips (perceptual loss), evo (trajectory eval). No compilation.
python -c "import lpips" 2>/dev/null || { log "installing lpips"; pip install -q lpips; }
python -c "import evo"   2>/dev/null || { log "installing evo";   pip install -q evo; }

# MoSca's src-backup (setup_recon_ws) does `cp -r profile lib_prior ...` from the
# current working directory, so we must run from the repo root.
cd "$MOSCA_ROOT"

# ---------------------------------------------------------------------------
# 3. Precompute: off-the-shelf 2D priors
#    depth (UniDepth) -> optical flow (RAFT) + epipolar error -> long 2D tracks
#    (BootsTAPIR), uniform + dynamic-region resample. Writes into $WS.
# ---------------------------------------------------------------------------
log "STAGE 1/2  precompute priors (depth/flow/epi/tracks)"
python mosca_precompute.py \
    --cfg "$PREP_CFG" \
    --ws  "$WS" \
    --dep_mode="$DEP_MODE" \
    --tap_mode="$TAP_MODE" \
    --boundary_enhance_th="$BOUNDARY_ENHANCE_TH"

# ---------------------------------------------------------------------------
# 4. Reconstruct: static BA -> photometric static warmup -> dynamic 4D motion
#    scaffold -> photometric dynamic-Gaussian fitting. Saves the d_model ckpt
#    at logs/<demo_fit_native_add3_datetime>/photometric_d_model_native_add3.pth.
#    (An optional eval/FPS pass runs *after* the ckpt is saved; tolerate its
#    failure since the checkpoint we need already exists on disk.)
# ---------------------------------------------------------------------------
log "STAGE 2/2  reconstruct 4D scene (BA -> scaffold -> photometric fit)"
set +e
python mosca_reconstruct.py \
    --cfg "$FIT_CFG" \
    --ws  "$WS" \
    --no_viz
RC=$?
set -e

# ---------------------------------------------------------------------------
# 5. Locate and report the checkpoint
# ---------------------------------------------------------------------------
CKPT="$(ls -t "$WS"/logs/*/photometric_d_model_${GS_BACKEND,,}.pth 2>/dev/null | head -n1 || true)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: reconstruction did not produce photometric_d_model_${GS_BACKEND,,}.pth (rc=$RC)" >&2
  echo "       inspect logs under $WS/logs/" >&2
  exit 1
fi
if [ "$RC" -ne 0 ]; then
  log "note: mosca_reconstruct exited rc=$RC (likely the post-fit eval/FPS pass); checkpoint is intact."
fi

log "DONE. 4D reconstruction checkpoint:"
echo "MOSCA_CKPT=$CKPT"
