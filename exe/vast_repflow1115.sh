#!/bin/bash
# repflow A 에 PG 장면 ficus·vasedeck·plane 추가 (사용자 11-15) 의 vast.ai 판 (장면 하나 = 인스턴스 하나; exe/vast_repflow_vrgs.sh 규약) onstart 로 받아 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/repflow1115/code/repflow1115_<커밋>.tar), XS_STG (기본 prep,main), XS_SCENE
#   끝 표식 R2 lbours/repflow1115/vast_<장면>.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}; STG=${XS_STG:-prep,main}; SC=${XS_SCENE:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/repflow1115
H=$(rclone cat $RX/code/commit.txt 2>/dev/null || echo $H)   # 코드 커밋 = R2 commit.txt (있으면)
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast_$SC.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast_$SC.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 단계 $STG $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
CAP=$(nvidia-smi --query-gpu=index,power.limit,power.default_limit --format=csv,noheader,nounits | awk -F', ' '$2+0 < $3-1 {print $1":"$2"<"$3}' | tr '\n' ' ')   # 카드마다 (한 인스턴스에 기종이 섞여도 오탐 없게)
if [ -n "$CAP" ]; then PL=$CAP; PD=x; else PL=1; PD=1; fi
if [ -n "$CAP" ]; then
  echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $RX/POWERCAP_$SC; say "[전력 제한] $PL W < 기본 $PD W -- 일 안 함 (로컬이 destroy·blocklist)"; exit 1
fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/sv/pip.done ]; then
  pip install -q "numpy<2" h5py scikit-image lpips torchmetrics torch-fidelity PyMCubes cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  pip install -q kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html 2>&1 | tail -2 | tee -a $LOG
  pip install -q warp-lang==1.6.2 2>&1 | tail -1 | tee -a $LOG   # PG 유틸이 warp.torch 를 쓰고 (1.6 까지 있음) kaolin 0.17 도 이 버전에서 동작
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, warp.torch, cupy, lpips, torchmetrics, kaolin, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/sv/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
D=/root/sv/code/$H; mkdir -p $D; rclone cat $RX/code/repflow1115_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
say "[자료] 렌더러 $(ls -d $W/i-physgaussian/gaussian-splatting 2>&1) 코드 $(ls $D/exe/go_repflow1115.sh 2>&1) pip $(ls /root/sv/pip.done 2>&1)"
[ -f /root/sv/pip.done ] && [ -f $D/exe/go_repflow1115.sh ] && [ -d $W/i-physgaussian/gaussian-splatting ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/setup_fail_$SC; exit 1; }
say "[준비 끝] GPU 일 시작"
export JOBS="${XS_JOBS:-}"; [ -n "$JOBS" ] || unset JOBS; export GPUMAP="${XS_GPUMAP:-}"
bash exe/go_repflow1115.sh $H $SC $STG < /dev/null >> $LOG 2>&1
rclone copyto $LOG $RX/vast_$SC.log -q
echo "끝 $(date)" | rclone rcat $RX/vast_$SC.END
