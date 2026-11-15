#!/bin/bash
# [stabfinal] 안정성 실험 마무리 (레고·pillow2sofa·ship_plastic_lego·mic_metal_lego) vast 부팅. stab1115_boot.sh 를 이어 받음.
#   env: SF_COMMIT (코드 tar R2 lbours/stabfinal/code/sf_<커밋>.tar), SF_JOBS = "씬:단계[,단계]+씬:단계" 직렬 (단계 main | fin = 다른 인스턴스 계획 기다림 -> 장면 결과 전부 받기 -> vis -> 분석 -> 비교 영상 | speed = 우리 판 계획 기다림 -> 단독 시간, A6000 에서만),
#        SF_TAG = 인스턴스 이름 (표식 marks/<TAG>, 추가 인자 code/extra_<TAG>.txt, 받을 패턴 code/fetch_<TAG>.txt), SF_PAR (기본 4), R2_* (없으면 기존 rclone 설정)
#   끝나면 R2 대조 차이 0 이면 스스로 destroy (CONTAINER_ID·CONTAINER_API_KEY 가 있을 때).
set -u
H=${SF_COMMIT:?}; JOBS=${SF_JOBS:?}; JOBS=${JOBS//+/;}; TAG=${SF_TAG:?}; PAR=${SF_PAR:-4}
R=r2:storage/result/anchorflow; RX=$R/lbours/stabfinal; MK=$RX/marks/$TAG; DS=r2:storage/datasets/anchorflow/nshape
mkdir -p /workspace/shm /workspace/tmpd /root/sf
LOG=/root/sf/vast.log
say() { echo "[sf $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
if [ -n "${R2_ACCESS_KEY:-}" ]; then
  printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
fi
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
if ! python -c "import torch, warp, cupy, h5py, lpips, taichi, diff_gaussian_rasterization, simple_knn, mcubes, skimage, imageio_ffmpeg" 2>/dev/null; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai pymcubes einops omegaconf 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null || pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
fi
python -c "import torch, warp, cupy, h5py, lpips, taichi, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG
W=/home/dkta/work; MI=$R/_vastmirror; PRX=$R/lbours/paper_remeasure2; HS=$W/h5src_sf; mkdir -p $HS
SCS=$(echo "$JOBS" | tr ';' '\n' | cut -d: -f1 | sort -u | tr '\n' ' ')
[ -d $W/OmniPhysGS/src ] || (rm -rf $W/OmniPhysGS; git clone -q https://github.com/wgsxm/OmniPhysGS $W/OmniPhysGS && git -C $W/OmniPhysGS checkout -q 5ab915b014467b911581b0c6df351713d3114a08) &
[ -f $W/i-physgaussian/gs_simulation.py ] || rclone copy $MI/i-physgaussian $W/i-physgaussian --exclude ".git/**" --transfers 16 -q &
[ -f $W/font.otf ] || rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q &
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $W/BA_1.txt &
for s in $SCS; do
  mkdir -p $HS/$s/init
  case $s in
    lego) rclone copy $MI/pgmodel/lego_whitebg-trained $W/pgmodel/lego_whitebg-trained --transfers 16 -q &
          rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $HS/lego/init/sim_0000000000.h5 -q & ;;
    pillow2sofa) (rclone copyto $PRX/assets/pillow2sofa_whitebg-trained.zip /root/sf/p2s.zip -q && unzip -q -o /root/sf/p2s.zip -d $W/pgmodel) &
          rclone copyto $PRX/scenes/pillow2sofa/h5/simulation_ply/sim_0000000000.h5 $HS/pillow2sofa/init/sim_0000000000.h5 -q & ;;
    ship_plastic_lego|mic_metal_lego) obj=${s%%_*}
          [ -s $W/pgmodel/$obj/point_cloud/iteration_30000/point_cloud.ply ] || rclone copy $DS/model/$obj $W/pgmodel/$obj --transfers 16 -q &
          rclone copy $DS/gt_pg/$s $HS/$s/gt --transfers 32 -q & ;;
  esac
done
mkdir -p $W/assets_cfg; rclone copy $PRX/assets $W/assets_cfg -q --include "*_config.json" &
rclone copyto $MI/stab/ficus/config.json $W/assets_cfg/ficus_drop_config.json -q &
D=/root/sf/code/$H; mkdir -p $D; rclone cat $RX/code/sf_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_sf XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext TMPDIR=/dev/shm/dkta/tmp S1115_TAG=$TAG
mkdir -p $CUPY_CACHE_DIR $TMPDIR
sed -i 's/MODEL = {"lego"/MODEL = {"vasedeck": "vasedeck_whitebg-trained", "lego"/' exe/rep_ip.py
python exe/stab1115_patch_models.py
python exe/stab1115_prep.py $W/assets_cfg/ficus_drop_config.json $W/assets_cfg $HS $W/stab $SCS 2>&1 | tee -a $LOG
for s in $SCS; do
  case $s in ship_plastic_lego|mic_metal_lego)
    [ -s $W/stab/$s/hex_in.pt ] || python -u exe/pg_h5_to_pt.py --h5_dir $HS/$s/gt/simulation_ply --config $HS/$s/gt/config.json --out_pt $W/stab/$s/hex_in.pt --out_dir $W/stab/$s/gt > /root/sf/cv_$s.log 2>&1
    say "[변환] $s $(tail -1 /root/sf/cv_$s.log | cut -c1-200)";;
  esac
done
OK=1; for s in $SCS; do [ -s $W/stab/$s/init/sim_0000000000.h5 ] && [ -s $W/stab/$s/config.json ] || OK=0
  case $s in ship_plastic_lego|mic_metal_lego) [ -s $W/stab/$s/hex_in.pt ] && [ -s $W/BA_1.txt ] || OK=0;; esac; done
[ -d $W/i-physgaussian/gaussian-splatting ] && [ -d $W/OmniPhysGS/src ] && [ $OK = 1 ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $MK/setup_fail; exit 1; }
echo "[ok] $(date)" | rclone rcat $MK/setup.done
say "[준비 끝] 입자 $(for s in $SCS; do echo -n "$s:"; python -c "import h5py;print(h5py.File('$W/stab/$s/init/sim_0000000000.h5')['x'].shape)"; done | tr '\n' ' ')"
OUT=$W/stab
EXC=(--exclude "**/vis/dump_*/**" --exclude "**/vis/hexdump_*/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "**/vis/png_*/**" --exclude "*/gt/**" --exclude "*/traj0/**")
FX=""; FI=$(rclone cat $RX/code/fetch_$TAG.txt 2>/dev/null | head -1)          # 받을 것만 (공백 구분 패턴; 없으면 전부)
for x in $FI; do FX="$FX --include $x"; done
for s in $SCS; do rclone copy $RX/res/$s $OUT/$s -q --transfers 16 --exclude "drv*.log" --exclude "DRV_DONE_*" --exclude "config.json" --exclude "init/**" --exclude "gt/**" $FX; done
say "[이어 받기 끝] $(for s in $SCS; do echo -n "$s:$(find $OUT/$s -name '*.fin' -o -name FIN | wc -l) "; done)"
( while true; do sleep 240; rclone copy $OUT $RX/res -q --transfers 8 "${EXC[@]}" 2>/dev/null
  rclone copy /root/sf $MK/logs -q --include "*.log" 2>/dev/null; done ) & SY=$!
EXTRA=$(rclone cat $RX/code/extra_$TAG.txt 2>/dev/null | head -1)        # 드라이버 추가 인자 (따옴표 포함 가능)
say "[인자] $EXTRA"
for job in $(echo "$JOBS" | tr ';' ' '); do
  s=${job%%:*}; stg=${job#*:}
  say "[작업] $s $stg"
  if [ "$stg" = fin ]; then
    # 다른 인스턴스의 계획 (plan_<방법>.json) 이 R2 에 다 올라올 때까지 기다림 (5 분 간격), 그 뒤 장면 결과 전부 받기
    NEED="plan_pg.json plan_ipg.json plan_omni.json plan_ours_f5.json"
    for w in $(seq 1 400); do
      miss=""; for p in $NEED; do rclone lsf $RX/res/$s/$p >/dev/null 2>&1 && [ -n "$(rclone lsf $RX/res/$s/$p 2>/dev/null)" ] || miss="$miss $p"; done
      [ -z "$miss" ] && break; [ $((w % 6)) = 1 ] && say "[기다림] $s 없음:$miss"; sleep 300
    done
    rclone copy $OUT $RX/res -q --transfers 8 "${EXC[@]}"
    rclone copy $RX/res/$s $OUT/$s -q --transfers 16 --exclude "drv*.log" --exclude "DRV_DONE_*" --exclude "config.json" --exclude "init/**" --exclude "gt/**"
    EXF=$(echo "$EXTRA" | sed -E "s/--methods [^ ]+//"); eval python -u exe/stab1115_drv.py --scene $s --par $PAR --stages vis $EXF --methods pg,ipg,omni,ours >> /root/sf/drv_$s.log 2>&1
    python -u exe/stab1115_an.py --scene $s --out $OUT/$s/an.json --hextag _f5 > /root/sf/an_$s.log 2>&1
    say "[분석] $s $(tail -n 6 /root/sf/an_$s.log | tr '\n' ' ' | cut -c1-400)"
    python -u exe/stabfinal_vid.py --root $OUT --scene $s --font $W/font.otf --out $OUT/$s/vis/${s}_stab_cmp.mp4 > /root/sf/vid_$s.log 2>&1
    say "[영상] $s $(tail -n 2 /root/sf/vid_$s.log | tr '\n' ' ' | cut -c1-300)"
  elif [ "$stg" = speed ]; then
    # 단독 시간 (A6000 인스턴스에서만): 우리 판 계획 (확정 N·사다리 발산) 이 R2 에 올라올 때까지 기다린 뒤, 이 카드에 다른 계산 프로세스가 없을 때 하나씩
    for w in $(seq 1 400); do [ -n "$(rclone lsf $RX/res/$s/plan_ours_f5.json 2>/dev/null)" ] && break; [ $((w % 6)) = 1 ] && say "[기다림] $s plan_ours_f5.json"; sleep 300; done
    rclone copy $RX/res/$s $OUT/$s -q --include "plan_*.json" --include "speed/**"
    EXF=$(echo "$EXTRA" | sed -E "s/--methods [^ ]+//"); eval python -u exe/stab1115_drv.py --scene $s --par 1 --stages speed $EXF --methods ours >> /root/sf/drv_$s.log 2>&1
    say "[단독 시간] $s $(grep -h '프레임당' $OUT/$s/speed/t_N*_f5/log.txt 2>/dev/null | grep -o '프레임당 [0-9.]* ms' | tr '\n' ' ')"
  else
    eval python -u exe/stab1115_drv.py --scene $s --par $PAR --stages $stg $EXTRA >> /root/sf/drv_$s.log 2>&1
  fi
  say "[작업 끝] $s $stg $(tail -1 /root/sf/drv_$s.log | cut -c1-200)"
done
kill $SY 2>/dev/null
rclone copy $OUT $RX/res -q --transfers 8 "${EXC[@]}"
rclone copy /root/sf $MK/logs -q --include "*.log"
if rclone check $OUT $RX/res --one-way --size-only -q "${EXC[@]}" --exclude "drv*.log" 2>/root/sf/check.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK/ALL_DONE
  [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] && vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함"; echo "$(date)" | rclone rcat $MK/ALL_DONE_CHECK_FAIL
fi
