#!/usr/bin/env bash
# vast.ai 인스턴스를 클러스터 작업 환경(/home/dkta/work, conda env af)과 같게 맞춘다 (2026-10-15, 클러스터 잡 대기 중 대체 실행).
# 이미지 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel (torch 2.5.1+cu121, py3.11; 프로젝트 이미지는 호스트에서 받기가 멈춰 바꿈) 위에:
#   - 클러스터 af 와 같은 판의 바이너리 휠만 설치 (빌드 없음): warp-lang 0.10.1, taichi 1.6.0, h5py, scikit-image, lpips, cupy-cuda12x
#   - 3DGS 래스터라이저는 GHA 에서 torch 2.5.1 로 빌드한 휠 (release cuda-3dgs-torch251, INRIA 472689c0)
#   - anchorflow 레포를 /home/dkta/work/anchorflow 에 받는다 (실행은 커밋별 worktree)
# 데이터(i-physgaussian·L000 궤적·lbpool·모델·stab 초기 상태)는 로컬에서 rsync 로 /home/dkta/work 에 넣는다.
set -uo pipefail
W=/home/dkta/work
mkdir -p $W/wt
pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg opencv-python-headless 2>&1 | tail -2
cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
  curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2
[ -d $W/anchorflow/.git ] || git clone -q https://github.com/wgsong0110/anchorflow $W/anchorflow
python -c "import torch, warp, taichi, h5py, diff_gaussian_rasterization; print('[setup] torch', torch.__version__, torch.cuda.is_available(), 'warp', warp.__version__)"
nvidia-smi --query-gpu=index,name,power.limit,memory.total --format=csv
echo SETUP_DONE > /root/setup_done
