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

echo "instance setup done: $(ls model/ball 2>/dev/null | tr '\n' ' ')"
