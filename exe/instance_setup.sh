#!/usr/bin/env bash
# One-shot instance setup for anchorflow SVD-SDS runs.
# Assumes the anchorflow image (has /opt/gs = gaussian-splatting pinned to the
# DreamPhysics-era commit + its rasterizer .so) and a cloned anchorflow repo.
set -e
WS=${WS:-/workspace}
cd "$WS"

# DreamPhysics (renderer + SVDGuidance). Reuse the image's pinned gaussian-splatting
# at /opt/gs (Python + matching rasterizer .so) instead of cloning latest (which
# breaks Camera/rasterizer API).
[ -d DreamPhysics ] || git clone -q https://github.com/tyhuang0428/DreamPhysics
cd DreamPhysics
rm -rf gaussian-splatting && ln -s /opt/gs gaussian-splatting
bash "$WS/anchorflow/exe/patch_dreamphysics.sh" "$WS/DreamPhysics"

# ball asset (DreamPhysics known-good SVD asset) for loop validation
huggingface-cli download tyhuang/DreamPhysics --repo-type dataset \
    --include "model/ball/*" --local-dir . >/dev/null 2>&1 || \
    echo "warn: ball download issue"

# rclone + R2 for automatic result upload. Creds come from env (NOT committed):
#   R2_ACCESS_KEY / R2_SECRET / R2_ENDPOINT (pass from host at setup time).
if [ -n "${R2_ACCESS_KEY:-}" ]; then
    command -v rclone >/dev/null 2>&1 || \
        (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rclone >/dev/null 2>&1)
    mkdir -p ~/.config/rclone
    printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\n' \
        "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
    echo "rclone R2 configured (auto result upload enabled)"
fi

# fused LBS CUDA kernel — download the prebuilt .so from the cuda-lbs release
# (compiled in CI in the anchorflow image; never compiled on the instance).
LBS="$WS/anchorflow/lib/lbs"
curl -sL "https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-lbs" \
  | grep -oE '"browser_download_url":[^,]*\.so"' | grep -oE 'https[^"]+' \
  | while read u; do curl -sL "$u" -o "$LBS/$(basename "$u")"; done
PYTHONPATH="$WS/anchorflow/lib" python -c \
  "import lbs; print('lbs cuda kernel:', lbs._HAVE_CUDA)" 2>/dev/null || echo "lbs kernel: torch fallback"

echo "instance setup done: $(ls model/ball 2>/dev/null | tr '\n' ' ')"
