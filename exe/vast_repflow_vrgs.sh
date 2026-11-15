#!/bin/bash
# repflow A 에 VR-GS (사면체 케이지) 를 GS-Verse 와 따로 넣는 묶음의 vast.ai 판 (exe/vast_hex1pt.sh 규약 그대로). onstart 로 받아 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/repflow_vrgs/code/repflow_vrgs_<커밋>.tar), XS_STG (기본 s0,tune,main)
#   끝 표식 R2 lbours/repflow_vrgs/vast.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}; STG=${XS_STG:-s0,tune,main}
R=r2:storage/result/anchorflow; RX=$R/lbours/repflow_vrgs
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 단계 $STG $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then
  echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $RX/POWERCAP; say "[전력 제한] $PL W < 기본 $PD W -- 일 안 함 (로컬이 destroy·blocklist)"; exit 1
fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/sv/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips torchmetrics torch-fidelity cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, lpips, torchmetrics, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/sv/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
for m in wolf_whitebg-trained bread-trained; do rclone copy $MI/pgmodel/$m $W/pgmodel/$m --transfers 16 -q --exclude "chkpnt*" & done
rclone copy $RX/pgmodel/ship_whitebg-trained $W/pgmodel/ship_whitebg-trained --transfers 16 -q 2>/dev/null &
D=/root/sv/code/$H; mkdir -p $D; rclone cat $RX/code/repflow_vrgs_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
say "[자료] 3DGS $(ls -d $W/pgmodel/* | tr '\n' ' ') 렌더러 $(ls -d $W/i-physgaussian/gaussian-splatting 2>&1) 코드 $(ls $D/exe/go_repflow_vrgs.sh 2>&1) pip $(ls /root/sv/pip.done 2>&1)"
[ -f /root/sv/pip.done ] && [ -f $D/exe/go_repflow_vrgs.sh ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -d $W/pgmodel/wolf_whitebg-trained/point_cloud ] && [ -d $W/pgmodel/bread-trained/point_cloud ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/setup_fail; exit 1; }
say "[준비 끝] GPU 일 시작 (ship 3DGS 모델 $(ls -d $W/pgmodel/ship_whitebg-trained/point_cloud 2>&1 | tail -1))"
bash exe/go_repflow_vrgs.sh $H $STG < /dev/null >> $LOG 2>&1
rclone copyto $LOG $RX/vast.log -q
echo "끝 $(date)" | rclone rcat $RX/vast.END
