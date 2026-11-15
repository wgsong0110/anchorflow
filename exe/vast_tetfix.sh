#!/bin/bash
# 절 CM178 묶음 tetfix (사면체 TC 레고 발산 추적·수정) vast.ai 인스턴스 세팅 (exe/vast_astick.sh + vast_cm177.sh 규약). onstart 로 받아 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT. 하는 일: rclone·pip·레고 자료 (L000.pt, lbdata, pgmodel lego, i-physgaussian, stab/lego init) 받기 -> 표식 R2 lbours/tetfix/vast.READY.
#   코드는 따로 `bash /root/upd.sh <태그>` (R2 lbours/tetfix/code/tetfix_<태그>.tar -> /root/tf/code), GPU 일은 exe/go_tetfix.sh (tmux).
set -u
R=r2:storage/result/anchorflow; RX=$R/lbours/tetfix
mkdir -p /workspace/shm /workspace/tmpd /root/tf
LOG=/root/tf/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/tf/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/tf/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel/lego_whitebg-trained $W/pgmodel/lego_whitebg-trained --transfers 16 -q &
rclone copy $MI/lbdata/config.json $W/lbdata -q &
rclone copy $MI/lbdata/test_low/L000 $W/lbdata/test_low/L000 --transfers 16 -q &
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q &
wait
cat > /root/upd.sh <<'EOF'
#!/bin/bash
# 사용: bash /root/upd.sh <태그> -> /root/tf/code/<태그> 에 풀고 /root/tf/cur 링크
T=$1; D=/root/tf/code/$T; rm -rf $D; mkdir -p $D
rclone cat r2:storage/result/anchorflow/lbours/tetfix/code/tetfix_$T.tar | tar -x -C $D && ln -sfn $D /root/tf/cur && echo "코드 $T -> $D ($(ls $D/exe | wc -l) exe)"
EOF
say "[자료] L000.pt $(stat -c %s $SD/L000.pt 2>&1) lbdata $(ls $W/lbdata/test_low/L000 2>/dev/null | wc -l) pgmodel $(du -sh $W/pgmodel 2>&1 | cut -f1) pip $(ls /root/tf/pip.done 2>&1)"
[ -f /root/tf/pip.done ] && [ -s $SD/L000.pt ] && [ -f $W/lbdata/config.json ] && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -d $W/pgmodel/lego_whitebg-trained ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] \
  || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast.FAIL; exit 1; }
echo "준비 $(date)" | rclone rcat $RX/vast.READY
say "[준비 끝]"
