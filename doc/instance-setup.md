# Instance / image setup for the GPU run

Single **RTX 3090 (24GB)** does everything: TRELLIS asset gen + SVD-SDS training.
**No CUDA builds on this arm64 host or the instance** — the image is built via
GitHub Actions (x86), per infra rules.

## Dependency stack (project image `ghcr.io/wgsong0110/anchorflow:latest`)

We reuse DreamPhysics's renderer + SVDGuidance and **do not** use its MPM path, so
warp/taichi are **not** needed. Required:

- CUDA runtime + PyTorch (2.1–2.4, cu118/cu121) + conda/mamba (base `run` image).
- **diff-gaussian-rasterization** (ashawkey fork `d986da0`) + **simple-knn** — CUDA
  build → GitHub Actions.
- **diffusers + transformers + accelerate** (SVD: `stabilityai/stable-video-diffusion-img2vid`).
- **TRELLIS** deps for `gen_canonical.py` (image→ply): `--basic --xformers --spconv
  --mipgaussian` (16GB VRAM; ATTN_BACKEND=xformers, SPCONV_ALGO=native). CUDA exts
  → GitHub Actions.
- pytorch3d (only if we keep any knn_points path; our lib uses cdist+topk, so
  optional — DreamPhysics utils may import it).
- plyfile, omegaconf, pyyaml, opencv-python-headless, imageio-ffmpeg.
- **anchorflow** (this repo) on PYTHONPATH.

Repos cloned on the instance (code only, no build): DreamPhysics fork (holds
`gaussian_renderer/`, `utils/`, `video_distillation/`, `scene/`) + anchorflow.
`train_gen.py`/`gen_canonical.py` run from inside the DreamPhysics dir with
`PYTHONPATH=$ANCHORFLOW/lib`.

## CI image build (GitHub Actions, x86)

`.github/workflows/image.yml` builds the Dockerfile above and pushes to GHCR.
Prereqs baked at build time: the two CUDA rasterizer exts + TRELLIS exts (the only
compilation). First build will need version pinning iteration — expect 1–2 CI cycles.

## Run sequence (on the 3090)

```bash
export HF_HOME=/data/huggingface ATTN_BACKEND=xformers SPCONV_ALGO=native
cd $DREAMPHYSICS_FORK
export PYTHONPATH=$ANCHORFLOW/lib:$PYTHONPATH

# 0. smoke test (no SVD/TRELLIS): placeholder GS -> LBS/rollout/render wiring
python $ANCHORFLOW/exe/smoke_render.py            # (to add: cheap plumbing check)

# 1. canonical asset (TRELLIS)
python $ANCHORFLOW/exe/gen_canonical.py --image horse.png \
    --out /data/datasets/anchorflow/horse.ply

# 2. per-scene distillation (resumable; run in tmux)
tmux new -d -s train "python $ANCHORFLOW/exe/train_gen.py \
    --model_path /data/datasets/anchorflow/horse.ply --cond horse.png \
    --config $ANCHORFLOW/cfg/anchorflow_horse.yaml \
    --out ~/workspace/result/anchorflow/horse --resume 2>&1 | tee train.log"
```

Staged smoke tests before the full run (minimise GPU time): (a) load .ply + LBS
warp with hand-set anchor motion → render 1 frame; (b) GNN rollout → render a clip
(no SVD); (c) attach SVDGuidance, 10 steps; then (d) full run. Stop the instance
immediately after measuring.

## Measurement

- Rendered rollout clips per checkpoint (video%02d.mp4) — qualitative self-actuated motion.
- SDS/MDS loss curve + reg terms (train.log / history).
- vs baselines: static (no motion), SC-GS-MLP deformation, DreamPhysics (MPM) —
  DreamPhysics is the key baseline (passive vs our self-actuated).
