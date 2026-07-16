# anchorflow instance image: 3DGS rasterizer + SVD(diffusers) + our lib.
# Built via GitHub Actions (x86). No MPM/warp/taichi, no tiny-cuda-nn/nvdiffrast
# (trimmed from the DreamPhysics path). CUDA exts compile ahead-of-time via
# TORCH_CUDA_ARCH_LIST + FORCE_CUDA (no GPU needed at build).
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV FORCE_CUDA=1

# libegl1/libgles2/libosmesa6: pyrender headless (EGL/OSMesa) for GLB multi-view render
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build libglm-dev ffmpeg libgl1 libglib2.0-0 \
        libegl1 libgles2 libosmesa6 \
        tmux openssh-client curl \
    && rm -rf /var/lib/apt/lists/*

# INRIA 3DGS pinned to the DreamPhysics-era commit (472689c0, parent of the
# 2024-08-21 "new release" that rewrote Camera + rasterizer). This keeps the
# rasterizer .so, scene/Camera, and gaussian_renderer all API-compatible with
# DreamPhysics (2-return rasterize, gt_alpha_mask Camera). /opt/gs is reused on
# the instance (symlinked as DreamPhysics/gaussian-splatting) so Python + .so match.
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting /opt/gs \
    && cd /opt/gs && git checkout 472689c0dc70417448fb451bf529ae532d32c095 \
    && git submodule update --init --recursive \
    && pip install --no-cache-dir /opt/gs/submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir /opt/gs/submodules/simple-knn

# Python deps. NB: SVD needs diffusers>=0.24 (DreamPhysics's requirements pin
# <0.20 for its ModelScope path — not applicable to the SVD path).
RUN pip install --no-cache-dir \
        "numpy<2" scipy tqdm packaging einops jaxtyping typeguard \
        omegaconf==2.3.0 plyfile==1.0.3 \
        opencv-python-headless imageio "imageio[ffmpeg]" \
        diffusers==0.29.2 transformers==4.41.2 accelerate==0.30.1 \
        safetensors huggingface_hub tensorboard rembg onnxruntime

# xformers matched to torch 2.1.2 (memory-efficient attention for SVD)
RUN pip install --no-cache-dir xformers==0.0.23.post1 || echo "xformers optional: skipped"

# pyrender+trimesh: headless GLB multi-view render; objaverse: asset download;
# gsplat: 3DGS rasterizer (CUDA ext compiles at build via FORCE_CUDA + arch list)
RUN pip install --no-cache-dir pyrender trimesh objaverse gsplat

WORKDIR /workspace
