#!/bin/bash
# [final_pa_bl] 파레토 베이스라인 대표 점 (레고·pillow2sofa) -- exe/vast_pr2_boot.sh 의 축소판 (레고·pillow2sofa 자료만, 결과 R2 lbours/final_pa/bl/<태그>).
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT FB_COMMIT (코드 tar R2 lbours/final_pa/bl/code/fpabl_<커밋>.tar), FB_TAG (결과 폴더 이름), FB_JOBS (exe/ 안 작업 JSON),
#        FB_MODE (prep = 준비 판만 [시간 안 재는 카드] | grid = 준비 판 받고 단독 시간 3 회), FB_PREP_TMO_H (준비 판 상한 시간, 기본 40), CONTAINER_ID·CONTAINER_API_KEY (vast)
#   준비 판 재사용: 레고 PhysTwin = paper_remeasure2/stab_A/p/phystwin_L000_prep, 레고 Spring-Gaus = stab_B/p/springgaus_L000_prep,
#                 pillow2sofa 준비 판 = 이 일의 다른 태그 (FB_PREP_FROM, 쉼표) 의 p/*_prep.
#   끝나면 R2 대조 차이 0 일 때 스스로 destroy.
set -u
H=${FB_COMMIT:?}; T=${FB_TAG:?}; MODE=${FB_MODE:-grid}
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; OX=$R/lbours/final_pa/bl; MK=$OX/marks_$T
mkdir -p /workspace/shm /workspace/tmpd /root/pr
LOG=/root/pr/vast.log
say() { echo "[fb $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 태그 $T 모드 $MODE 작업 ${FB_JOBS:-} $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then
  echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $MK/POWERCAP; say "[전력 제한] $PL W < 기본 $PD W -- 일 안 함 (로컬이 destroy)"; exit 1
fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
if [ ! -f /root/pr/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, warp, cupy, h5py, lpips, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/pr/pip.done
fi
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; MI=$R/_vastmirror; mkdir -p $SD
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q --include "lego_whitebg-trained/**" &
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q &
rclone copy $MI/stab/lego/init $W/stab/lego/init -q &
D=/root/pr/code/$H; mkdir -p $D; rclone cat $OX/code/fpabl_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
mkdir -p $CUPY_CACHE_DIR
mkdir -p $W/one/L000; cp $SD/L000.pt $W/one/L000/L000.pt 2>/dev/null
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; mkdir -p $O/p
ln -sfn $O /dev/shm/dkta/lbours/stab                                          # 옛 준비 판 json 의 절대 경로 (/dev/shm/dkta/lbours/stab/p/...) 대응
rclone copy $RX/stab_A/p/phystwin_L000_prep $O/p/phystwin_L000_prep -q --transfers 8 &
rclone copy $RX/stab_B/p/springgaus_L000_prep $O/p/springgaus_L000_prep -q --transfers 8 &
FPF=${FB_PREP_FROM:-}; for f in ${FPF//,/ }; do rclone copy $OX/$f $O -q --transfers 8 --include "p/*_pillow2sofa_prep/**" --include "g/vrgs_*_fit/**" & done
wait
(apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1 >/dev/null 2>&1); pip install -q cloudpickle dataclasses_json plotly narwhals nbformat ipywidgets dash pyquaternion 2>&1 | tail -1
rclone copyto $R/vastpkg/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl /root/pr/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl -q
export ST_P3D_WHL=/root/pr/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl ST_NO_BUILD=1 ST_SIMP_PY=/home/dkta/.conda/envs/af/bin/python
bash exe/stab_bl_setup.sh /root/pr/bl_env.sh > /root/pr/bl_setup.log 2>&1; say "[bl_setup] $(tail -1 /root/pr/bl_setup.log)"; rclone copyto /root/pr/bl_setup.log $MK/bl_setup.log -q
[ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
PR_SC=pillow2sofa; source exe/pr2_scene_fetch.sh
say "[자료] L000 $(stat -c %s $SD/L000.pt 2>&1) lbdata $(ls $W/lbdata | tr '\n' ' ') 준비 판 $(ls $O/p | tr '\n' ' ') pip $(ls /root/pr/pip.done 2>&1)"
[ -f /root/pr/pip.done ] && [ -s $SD/L000.pt ] && [ -s $W/lbdata/config.json ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -s $W/newsc/pillow2sofa.pt ] \
  || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $MK/setup_fail; exit 1; }
echo "[ok] $(date)" | rclone rcat $MK/boot.done
if [ -n "${FB_HOLD:-}" ]; then say "[대기] /root/pr/go 를 기다림 (ssh 로 확인 뒤)"; while [ ! -f /root/pr/go ]; do sleep 20; done; fi
rclone copy $OX/$T $O -q --transfers 8 --exclude "*/h5/**"                    # 이어 돌리기 (같은 태그의 이전 인스턴스)
( while true; do sleep 300; rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"; done ) & SY=$!
EX=""; [ "$MODE" = prep ] && EX="--prep_only"
PR2_PREP_TMO_H=${FB_PREP_TMO_H:-40} python -u exe/pr2_grid.py --out $O --jobs exe/$FB_JOBS --reps 3 --scenes L000 $EX \
  --until $(( $(date +%s) + 60 * 3600 )) --hard $(( $(date +%s) + 61 * 3600 )) >> $O/grid_main.log 2>&1
say "[끝] $(tail -1 $O/grid_main.log)"
kill $SY 2>/dev/null
rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"
if rclone check $O $OX/$T --one-way --size-only -q --exclude "*/h5/**" 2>/root/pr/check.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK/ALL_DONE
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함"; echo "$(date)" | rclone rcat $MK/ALL_DONE_CHECK_FAIL
fi
