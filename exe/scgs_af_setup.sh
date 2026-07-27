#!/usr/bin/env bash
# One-shot instance setup for the SC-GS baseline + AnchorFlow (rotation-head
# GNN+SSM dynamics) workflow -- exe/scgs_baseline_run.sh and
# exe/train_sc_anchorflow.py. Run once per fresh instance (wired into
# .wexec.json's "setup"). Idempotent: safe to re-run.
#
# What this replaces (all previously done by hand, see 2026-07-26 session):
#   - lib/lbs CUDA .so was never synced/downloaded -> both LBS kernels
#     silently fell back to pure torch with no warning.
#   - ficus_ds_wind dataset was scp'd in by hand from the local machine.
#   - SC-GS repo clone was already automated (kept as-is below).
set -uo pipefail
WS="${WS:-/workspace}"
AF="$WS/anchorflow"

echo "[setup] SC-GS repo"
git clone --depth 1 https://github.com/yihua7/SC-GS "$WS/SC-GS" 2>/dev/null || true

echo "[setup] anchorflow repo (self-heal if the instance wasn't pre-cloned)"
if [ ! -d "$AF/.git" ]; then
    git clone --depth 1 https://github.com/wgsong0110/anchorflow "$AF"
fi

echo "[setup] lib/lbs fused CUDA kernel (.so from the cuda-lbs release)"
LBS="$AF/lib/lbs"
mkdir -p "$LBS"
curl -sL "https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-lbs" \
  | grep -oE '"browser_download_url":[^,]*\.so"' | grep -oE 'https[^"]+' \
  | while read -r u; do curl -sL "$u" -o "$LBS/$(basename "$u")"; done
PYTHONPATH="$AF/lib" python3 -c "
import lbs
print('  lbs._HAVE_CUDA          =', lbs._HAVE_CUDA)
print('  lbs._HAVE_COV_CUDA      =', lbs._HAVE_COV_CUDA)
print('  lbs._HAVE_ROT_BATCH_CUDA=', lbs._HAVE_ROT_BATCH_CUDA)
" || echo "  WARNING: lib/lbs import failed -- training will fall back to (slower) pure torch"

echo "[setup] rclone + R2 (for the dataset step below; creds from env, never committed)"
if [ -n "${R2_ACCESS_KEY:-}" ] && [ ! -f ~/.config/rclone/rclone.conf ]; then
    command -v rclone >/dev/null 2>&1 || \
        (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rclone >/dev/null 2>&1)
    mkdir -p ~/.config/rclone
    printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\n' \
        "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
fi

echo "[setup] ficus_ds_wind dataset (R2: r2:storage/datasets/anchorflow/ficus_ds_wind)"
if [ -d "$WS/ficus_ds_wind" ] && [ -n "$(ls -A "$WS/ficus_ds_wind" 2>/dev/null)" ]; then
    echo "  already present, skipping"
elif command -v rclone >/dev/null 2>&1 && [ -f ~/.config/rclone/rclone.conf ]; then
    rclone copy r2:storage/datasets/anchorflow/ficus_ds_wind "$WS/ficus_ds_wind" --progress
else
    echo "  SKIPPED: rclone not configured on this instance."
    echo "  Either pass R2_ACCESS_KEY/R2_SECRET/R2_ENDPOINT at instance creation"
    echo "  (same convention as exe/instance_setup.sh), or copy the dataset by hand:"
    echo "    scp -r -P <port> ~/r2/datasets/anchorflow/ficus_ds_wind root@<host>:$WS/ficus_ds_wind"
fi

echo "[setup] done."
