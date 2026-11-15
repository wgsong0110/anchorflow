#!/bin/bash
# [stab1115] 서브스텝 안정성 PG 논문 장면 재측정 vast 부팅. env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT S1115_COMMIT (코드 tar R2 lbours/stab1115/code/s1115_<커밋>.tar)
#   S1115_JOBS = "씬:단계[,단계];씬:단계" (예: ficus:main,vis;plane:main) 순서대로 직렬, S1115_TAG = 인스턴스 이름 (표식 marks/<TAG>), S1115_PAR = 동시 판 수 (기본 4), S1115_EXTRA = 드라이버 추가 인자
#   끝나면 R2 대조 차이 0 이면 스스로 destroy (CONTAINER_ID·CONTAINER_API_KEY).
set -u
H=${S1115_COMMIT:?}; JOBS=${S1115_JOBS:?}; JOBS=${JOBS//+/;}; TAG=${S1115_TAG:?}; PAR=${S1115_PAR:-4}
R=r2:storage/result/anchorflow; RX=$R/lbours/stab1115; MK=$RX/marks/$TAG
mkdir -p /workspace/shm /workspace/tmpd /root/s1
LOG=/root/s1/vast.log
say() { echo "[s1115 $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 작업 $JOBS $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then
  echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $MK/POWERCAP; say "[전력 제한] $PL W < 기본 $PD W -- 일 안 함 (로컬이 destroy·blocklist)"; exit 1
fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
if [ ! -f /root/s1/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai pymcubes einops omegaconf 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, warp, cupy, h5py, lpips, taichi, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/s1/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror; PRX=$R/lbours/paper_remeasure2; HS=$W/h5src; mkdir -p $HS
SCS=$(echo "$JOBS" | tr ';' '\n' | cut -d: -f1 | sort -u | tr '\n' ' ')
[ -d $W/OmniPhysGS ] || (git clone -q https://github.com/wgsxm/OmniPhysGS $W/OmniPhysGS && git -C $W/OmniPhysGS checkout -q 5ab915b014467b911581b0c6df351713d3114a08) &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
for s in $SCS; do
  case $s in ficus) mdl=ficus_whitebg-trained;; lego) mdl=lego_whitebg-trained;; vasedeck) mdl=vasedeck_whitebg-trained;; plane) mdl=plane-trained;; pillow2sofa) mdl=pillow2sofa_whitebg-trained;; esac
  if [ $s = ficus ] || [ $s = lego ]; then rclone copy $MI/pgmodel/$mdl $W/pgmodel/$mdl --transfers 16 -q &
  else (rclone copyto $PRX/assets/$mdl.zip /root/s1/$mdl.zip -q && unzip -q -o /root/s1/$mdl.zip -d $W/pgmodel) & fi
  mkdir -p $HS/$s/init
  if [ $s = ficus ] || [ $s = lego ]; then rclone copyto $MI/stab/$s/init/sim_0000000000.h5 $HS/$s/init/sim_0000000000.h5 -q &
  else rclone copyto $PRX/scenes/$s/h5/simulation_ply/sim_0000000000.h5 $HS/$s/init/sim_0000000000.h5 -q & fi
done
mkdir -p $W/assets_cfg; rclone copy $PRX/assets $W/assets_cfg -q --include "*_config.json" &
rclone copyto $MI/stab/ficus/config.json $W/assets_cfg/ficus_drop_config.json -q &
D=/root/s1/code/$H; mkdir -p $D; rclone cat $RX/code/s1115_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_s1 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext TMPDIR=/dev/shm/dkta/tmp
mkdir -p $CUPY_CACHE_DIR $TMPDIR
sed -i 's/MODEL = {"lego"/MODEL = {"vasedeck": "vasedeck_whitebg-trained", "lego"/' exe/rep_ip.py
python exe/stab1115_patch_models.py
python exe/stab1115_prep.py $W/assets_cfg/ficus_drop_config.json $W/assets_cfg $HS $W/stab $SCS 2>&1 | tee -a $LOG
OK=1; for s in $SCS; do [ -s $W/stab/$s/init/sim_0000000000.h5 ] && [ -s $W/stab/$s/config.json ] || OK=0; done
[ -f /root/s1/pip.done ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ $OK = 1 ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $MK/setup_fail; exit 1; }
echo "[ok] $(date)" | rclone rcat $MK/setup.done
say "[준비 끝] 입자 $(for s in $SCS; do echo -n "$s:"; python -c "import h5py;print(h5py.File('$W/stab/$s/init/sim_0000000000.h5')['x'].shape)"; done | tr '\n' ' ')"
OUT=$W/stab
# [stab4] 이어 돌리기: R2 에 있는 결과를 먼저 받음 (S1115_FETCH_EXCL = 받지 않을 패턴, 공백 구분; 예 plane 의 실패한 gf_*)
FX=""; for x in ${S1115_FETCH_EXCL:-}; do FX="$FX --exclude $x"; done
for s in $SCS; do rclone copy $RX/res/$s $OUT/$s -q --transfers 16 --exclude "drv.log" --exclude "DRV_DONE_*" --exclude "config.json" --exclude "init/**" $FX; done
say "[이어 받기 끝] $(for s in $SCS; do echo -n "$s:$(find $OUT/$s -name '*.fin' -o -name FIN | wc -l) "; done)"
( while true; do sleep 240; rclone copy $OUT $RX/res -q --transfers 8 --exclude "**/vis/dump_*/**" --exclude "**/vis/hexdump_*/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "**/vis/png_*/**" 2>/dev/null
  rclone copy /root/s1 $MK/logs -q --include "*.log" 2>/dev/null; done ) & SY=$!
for job in $(echo "$JOBS" | tr ';' ' '); do
  s=${job%%:*}; stg=${job#*:}
  say "[작업] $s $stg"
  EXTRA=$(rclone cat $RX/code/extra_$TAG.txt 2>/dev/null | head -1)        # [stab4] 드라이버 추가 인자 (따옴표 포함 가능) -- env 로는 공백을 못 넘겨 R2 파일로
  say "[인자] $EXTRA"
  eval python -u exe/stab1115_drv.py --scene $s --par $PAR --stages $stg ${S1115_EXTRA:-} $EXTRA >> /root/s1/drv_$s.log 2>&1
  say "[작업 끝] $s $stg rc=$? $(tail -1 /root/s1/drv_$s.log | cut -c1-200)"
done
kill $SY 2>/dev/null
rclone copy $OUT $RX/res -q --transfers 8 --exclude "**/vis/dump_*/**" --exclude "**/vis/hexdump_*/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "**/vis/png_*/**"
rclone copy /root/s1 $MK/logs -q --include "*.log"
if rclone check $OUT $RX/res --one-way --size-only -q --exclude "**/vis/dump_*/**" --exclude "**/vis/hexdump_*/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "**/vis/png_*/**" 2>/root/s1/check.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK/ALL_DONE
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함"; echo "$(date)" | rclone rcat $MK/ALL_DONE_CHECK_FAIL
fi
