#!/bin/bash
# 절 CM194 반복 (r2·r3 = 대기열 rep, 시간 2·3 회 = 대기열 time23) 용 vast 부팅: 전력 확인 -> 환경 (exe/vast_ffx.sh 와 같은 pip·3DGS 휠·경로) ->
#   exe/gqj_boot.sh (자료 받기 + 일꾼, r1 과 같은 exe/gqj_cmd.py 명령) -> 내 대기열 판이 모두 R2 에 DONE 이면 스스로 destroy.
#   env: GQ_H (코드 tar R2 gq_justify/code/gq_<커밋>.tar) GQ_Q (rep|time23) GQ_NW GQ_TIME GQ_TONLY REP_TAG CONTAINER_ID CONTAINER_API_KEY
set -u
export HOME=/root
H=${GQ_H:?}; Q=${GQ_Q:?}; T=${REP_TAG:-gqrep}
R=r2:storage/result/anchorflow; RG=$R/lbours/gq_justify; MK=$RG/marks
mkdir -p /workspace/shm /workspace/tmpd /root/gq
LOG=/root/gqv.log
say() { echo "[gqv $T $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK/$T.log -q 2>/dev/null; }
say "[시작] 커밋 $H 대기열 $Q $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < $PD $(date)" | rclone rcat $MK/$T.POWERCAP; say "[전력 제한] $PL < $PD -- 일 안 함"; exit 1; fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
if [ ! -f /root/gq/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib pymcubes vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, lpips, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/gq/pip.done
fi
python -c "import torch;print(torch.cuda.is_available())" | grep -q True || { say "[CUDA 없음]"; exit 1; }
D=/root/gq/code/$H; mkdir -p $D; [ -f $D/exe/gqj_worker.sh ] || rclone cat $RG/code/gq_$H.tar | tar -x -C $D
say "[환경 끝] 일꾼 $Q x ${GQ_NW:-1} (시간 ${GQ_TIME:-0}, 시간만 ${GQ_TONLY:-0})"
( while sleep 300; do rclone copyto $LOG $MK/$T.log -q; for f in /root/gq/w*.log; do [ -f $f ] && rclone copyto $f $MK/$T.$(basename $f) -q; done; done ) & SY=$!
cd $D && GQ_IDLE=${GQ_IDLE:-120} bash exe/gqj_boot.sh >> $LOG 2>&1
kill $SY 2>/dev/null
for f in /root/gq/w*.log; do [ -f $f ] && rclone copyto $f $MK/$T.$(basename $f) -q; done
miss=0
while IFS=: read -r s v c m r; do [ -z "${r:-}" ] && continue
  [ "$m" = time ] && [ "${GQ_TIME:-0}" != 1 ] && continue; [ "$m" != time ] && [ "${GQ_TONLY:-0}" = 1 ] && continue
  rclone lsf $RG/runs/${s}_${v}_${c}_${m}_r$r/DONE 2>/dev/null | grep -q . || miss=$((miss+1)); done < <(rclone cat $RG/queue/$Q.txt)
if [ $miss = 0 ]; then say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK/$T.ALL_DONE
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tail -1 | tee -a $LOG
else say "[R2 확인: DONE 없는 판 $miss] destroy 안 함"; echo "$(date) miss $miss" | rclone rcat $MK/$T.CHECK_FAIL; fi
