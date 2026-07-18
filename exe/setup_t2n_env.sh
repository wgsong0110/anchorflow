#!/bin/bash
# Install T2N preprocessing dependencies on GPU instance.
# Run once after instance starts.
set -e

# ── repo setup ────────────────────────────────────────────────────────────────
cd /workspace
if [ ! -d anchorflow ]; then
  git clone https://github.com/wgsong0110/anchorflow.git
fi
cd anchorflow && git pull

# CUDA .so from release
mkdir -p lib/lbs
RELEASE_URL=$(curl -s https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-build \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(next(a['browser_download_url'] for a in d.get('assets',[]) if a['name'].endswith('.so')))" 2>/dev/null || echo "")
if [ -n "$RELEASE_URL" ]; then
  curl -L "$RELEASE_URL" -o lib/lbs/lbs_cuda.so
  echo "Downloaded lbs_cuda.so"
fi

# ── gs-splatting ──────────────────────────────────────────────────────────────
if [ ! -d /workspace/gaussian-splatting ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting --recursive
fi

# Install 3DGS rasterizer from anchorflow release
cd /workspace
RAST_URL=$(curl -s https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-build \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if 'diff_gaussian' in a['name']), ''))" 2>/dev/null || echo "")
if [ -n "$RAST_URL" ]; then
  pip install "$RAST_URL" --quiet
else
  echo "[warn] rasterizer wheel not found in release"
fi

# ── Python deps ───────────────────────────────────────────────────────────────
pip install omegaconf imageio imageio-ffmpeg --quiet

# ── TAPIR (tapnet) ────────────────────────────────────────────────────────────
pip install git+https://github.com/google-deepmind/tapnet.git --quiet || \
  echo "[warn] tapnet install failed"

# Download TAPIR checkpoint
mkdir -p /data/huggingface/tapir
if [ ! -f /data/huggingface/tapir/tapir_checkpoint_panning.pt ]; then
  wget -q -O /data/huggingface/tapir/tapir_checkpoint_panning.pt \
    "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt" || \
    echo "[warn] tapir checkpoint download failed"
fi

# ── SAM2 ─────────────────────────────────────────────────────────────────────
pip install git+https://github.com/facebookresearch/sam2.git --quiet || \
  echo "[warn] sam2 install failed"

mkdir -p /data/huggingface/sam2
if [ ! -f /data/huggingface/sam2/sam2_hiera_large.pt ]; then
  HF_HOME=/data/huggingface huggingface-cli download facebook/sam2-hiera-large \
    --local-dir /data/huggingface/sam2 --quiet || echo "[warn] sam2 checkpoint download failed"
fi

# ── DepthCrafter ──────────────────────────────────────────────────────────────
pip install git+https://github.com/TencentARC/DepthCrafter.git --quiet || \
  echo "[warn] depthcrafter install failed"

# Download DepthCrafter model
if [ ! -d /data/huggingface/hub/models--tencent--DepthCrafter ]; then
  HF_HOME=/data/huggingface huggingface-cli download tencent/DepthCrafter --quiet || \
    echo "[warn] depthcrafter model download failed"
fi

# ── gs_flame model ────────────────────────────────────────────────────────────
mkdir -p /workspace/gs_flame
if [ ! -f /workspace/gs_flame/cameras.json ]; then
  rclone copy r2:storage/result/anchorflow/expB_flame /workspace/gs_flame --progress
fi
# Copy N3DV cameras
if [ -f /workspace/anchorflow/exe/../cameras_n3dv_flame.json ]; then
  cp /workspace/anchorflow/cameras_n3dv_flame.json /workspace/gs_flame/cameras.json
fi

echo "[setup_t2n_env] done"
