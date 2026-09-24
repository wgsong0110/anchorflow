#!/usr/bin/env bash
# vast 인스턴스 1 회 셋업: 의존성 + 자산 + rclone.
# 이미지는 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel 을 전제한다 (3DGS 휠이
# 그 토치에 맞춰 미리 빌드돼 있다 -- 인스턴스에서 CUDA 빌드를 하지 않는다).
set -uo pipefail
W=/workspace
mkdir -p $W/assets $W/out $W/pgwork
cd $W

echo "[셋업] 시스템 패키지"
apt-get update -qq >/dev/null 2>&1
# taichi 가 headless 컨테이너에서 libX11 을 찾는다
apt-get install -y -qq git curl unzip tmux \
  libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libgl1 libglu1-mesa >/dev/null 2>&1

echo "[셋업] rclone"
if ! command -v rclone >/dev/null; then
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip
  unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone
fi
mkdir -p ~/.config/rclone
[ -f ~/.config/rclone/rclone.conf ] || echo "[경고] rclone.conf 가 없다 -- 별도로 넣어야 한다"

echo "[셋업] 파이썬 의존성"
pip install -q warp-lang taichi h5py plyfile tqdm scipy imageio imageio-ffmpeg \
    opencv-python-headless 2>&1 | tail -2

echo "[셋업] 3DGS CUDA 휠 (미리 빌드된 것)"
cd /tmp && rm -f *.whl
gh_base=https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251
for w in diff_gaussian_rasterization simple_knn; do
  curl -sL -O "$gh_base/$(curl -sL https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-3dgs-torch251 | grep -o "\"name\": \"${w}[^\"]*\.whl\"" | head -1 | cut -d'"' -f4)" 2>/dev/null
done
ls /tmp/*.whl 2>/dev/null && pip install -q /tmp/*.whl 2>&1 | tail -2

echo "[셋업] 자산 내려받기"
cd $W
R=r2:storage/result/anchorflow/vastpkg
for f in pg_code.tar.gz af_code.tar.gz aux.tar.gz pgmodel.tar.gz; do
  rclone copy "$R/$f" /tmp/ --transfers 4 || echo "[실패] $f"
done
tar xzf /tmp/pg_code.tar.gz -C $W                      # -> $W/PG_pgtraj
mkdir -p $W/anchorflow && tar xzf /tmp/af_code.tar.gz -C $W/anchorflow   # exe, lib
tar xzf /tmp/aux.tar.gz -C $W/assets                   # wmats, *fill_*.npy
mkdir -p $W/pgmodel && tar xzf /tmp/pgmodel.tar.gz -C $W/pgmodel
rm -f /tmp/*.tar.gz

echo "[확인] import"
python - <<'PY'
import torch, warp, taichi, h5py
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import sys; sys.path.insert(0, "/workspace/PG_pgtraj")
sys.path.insert(0, "/workspace/PG_pgtraj/gaussian-splatting")
import diff_gaussian_rasterization, simple_knn
print("3DGS 확장 OK")
PY
echo SETUP_DONE
