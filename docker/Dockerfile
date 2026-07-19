# anchorflow instance image: SC-GS rasterizer + SVD(diffusers) + our lib.
# Built via GitHub Actions (x86). CUDA exts compile ahead-of-time via
# TORCH_CUDA_ARCH_LIST + FORCE_CUDA (no GPU needed at build).
#
# Lessons from past setup failures:
#   - INRIA 3DGS rasterizer returns (image, radii) — 2 values.
#     SC-GS rasterizer returns (image, radii, depth, alpha) — 4 values.
#     Our scripts use SC-GS's gaussian_renderer which expects the 4-return API,
#     so install SC-GS's rasterizer (not INRIA's).
#   - lpips was missing in the old image; it is required by train_seqgen_mds.py.
#   - SC-GS Python code (/workspace/SC-GS) is NOT baked into the image —
#     setup in .wexec.json clones it at instance start so code stays up-to-date.
#   - plyfile is needed by SC-GS model loading (load_ply).
#   - Base torch version must match the SC-GS rasterizer wheel build (torch 2.5.1).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV FORCE_CUDA=1

# libegl1/libgles2/libosmesa6: pyrender headless (EGL/OSMesa) for GLB multi-view render
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build libglm-dev ffmpeg libgl1 libglib2.0-0 \
        libegl1 libgles2 libosmesa6 \
        tmux openssh-client curl unzip \
    && rm -rf /var/lib/apt/lists/*

# SC-GS diff-gaussian-rasterization (4-return: color, radii, depth, alpha).
# Replaces the INRIA 2-return rasterizer — our train_seqgen_mds.py uses SC-GS's
# render() which requires the 4-return API.  simple-knn is also from SC-GS.
RUN git clone https://github.com/yihua7/SC-GS /opt/scgs \
    && cd /opt/scgs && git submodule update --init --recursive \
    && pip install --no-cache-dir /opt/scgs/submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir /opt/scgs/submodules/simple-knn

# Python deps.
# - lpips: required by train_seqgen_mds.py (perceptual loss).
# - plyfile: required by SC-GS GaussianModel.load_ply().
# - SVD needs diffusers>=0.24 (DreamPhysics's requirements pin <0.20 for its
#   ModelScope path — not applicable to the SVD path we use).
RUN pip install --no-cache-dir \
        "numpy<2" scipy tqdm packaging einops jaxtyping typeguard \
        omegaconf==2.3.0 plyfile==1.0.3 lpips piq \
        opencv-python-headless imageio "imageio[ffmpeg]" \
        diffusers==0.29.2 transformers==4.41.2 accelerate==0.30.1 \
        safetensors huggingface_hub tensorboard rembg onnxruntime

# xformers matched to torch 2.5.1 (memory-efficient attention for SVD UNet)
RUN pip install --no-cache-dir xformers==0.0.28.post3 || echo "xformers optional: skipped"

# pyrender+trimesh: headless GLB multi-view render;
# gsplat: 3DGS rasterizer (CUDA ext compiles at build via FORCE_CUDA + arch list)
RUN pip install --no-cache-dir pyrender trimesh gsplat

WORKDIR /workspace
