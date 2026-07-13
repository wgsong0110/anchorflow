# anchorflow instance image: 3DGS rasterizer + SVD(diffusers) + our lib.
# Built via GitHub Actions (x86). No MPM/warp/taichi, no tiny-cuda-nn/nvdiffrast
# (trimmed from the DreamPhysics path). CUDA exts compile ahead-of-time via
# TORCH_CUDA_ARCH_LIST + FORCE_CUDA (no GPU needed at build).
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build libglm-dev ffmpeg libgl1 libglib2.0-0 \
        tmux openssh-client curl \
    && rm -rf /var/lib/apt/lists/*

# INRIA 3DGS rasterizer (2-return, as DreamPhysics uses) + simple-knn
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting /opt/gs \
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

# xformers matched to torch 2.0.1 (memory-efficient attention for SVD)
RUN pip install --no-cache-dir xformers==0.0.22 || echo "xformers optional: skipped"

WORKDIR /workspace
