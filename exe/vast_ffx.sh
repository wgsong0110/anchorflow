#!/bin/bash
# 절 CM191 (final_fix) vast.ai onstart 진입점 (exe/vast_fpa.sh 복사 + 레고 세팅 점소성 장면 준비, 일꾼 exe/ffx_worker.sh, 결과 lbours/final_fix): 장면 자료 준비 (vast_abl.sh + vast_pr2_boot.sh 의 레고·pillow2sofa 부분) -> 카드마다 일꾼 (exe/fpa_worker.sh, 같은 대기열).
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT, FPA_COMMIT (코드 tar R2 lbours/final_pa/code/fpa_<커밋>.tar), FPA_TAG (인스턴스 이름), FPA_JOBS (쉼표: 장면:판:반복),
#        FPA_SC (준비할 장면 쉼표: L000,pillow2sofa,bread,wolf), CONTAINER_ID
#   표식·로그 = R2 lbours/final_pa/marks/<FPA_TAG>.* ; 결과 = R2 lbours/final_pa/runs/<장면>_<판>_r<반복>/
set -u
H=${FPA_COMMIT:?}; T=${FPA_TAG:?}; export FFX_TAG=$T
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; RA=$R/lbours/final_fix; MK=$RA/marks
mkdir -p /workspace/shm /workspace/tmpd /root/pr /root/ffx
LOG=/root/pr/vast.log
say() { echo "[ffx $T $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/$T.log -q 2>/dev/null; }
say "[시작] 커밋 $H $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  PL=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1); PD=$(nvidia-smi -i $i --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
  if [ "$PL" -lt "$PD" ]; then echo "카드 $i 전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $MK/$T.POWERCAP; say "[전력 제한] 카드 $i $PL < $PD -- 일 안 함"; exit 1; fi
done
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
if [ ! -f /root/pr/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib pymcubes 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, lpips, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/pr/pip.done
fi
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; SD=/dev/shm/dkta/anerr_data; MI=$R/_vastmirror; mkdir -p $AG $SD
SC=${FPA_SC:-L000,pillow2sofa,bread,wolf}
has() { case ",$SC," in *",$1,"*) return 0;; esac; return 1; }
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q &
if has L000; then rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q & rclone copy $MI/lbdata $W/lbdata --transfers 16 -q & rclone copy $MI/stab/lego/init $W/stab/lego/init -q & fi
if has bread; then rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q & rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
  rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q & fi
if has wolf; then rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q & rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
  rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q & fi
D=/root/pr/code/$H; mkdir -p $D; rclone cat $RA/code/fpa_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
mkdir -p $CUPY_CACHE_DIR
has bread && { [ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/pr/gt_bread.log 2>&1; }
has wolf && { [ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/pr/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/pr/gt_wolf.log 2>&1; }
if has pillow2sofa; then export PR_SC=pillow2sofa RX; source exe/pr2_scene_fetch.sh; fi
# 절 CM191: 레고 세팅 점소성 장면 (정답 R2 nshape/gt_pg/<장면>, 3DGS nshape/model/<물체>, BA_1 명령) -- vast_plastfix_lego.sh 의 받기·변환 부분
DS=r2:storage/datasets/anchorflow/nshape
for k in chair_paste_lego drums_honey_lego ship_plastic_lego bread_foam_drop_lego; do has $k || continue
  case $k in chair*) ob=chair;; drums*) ob=drums;; ship*) ob=ship;; bread*) ob=bread-trained;; esac
  [ -d $W/pgmodel/$ob/point_cloud ] || { if [ $ob = bread-trained ]; then rclone copy $MI/pgmodel/$ob $W/pgmodel/$ob --transfers 16 -q; else rclone copy $DS/model/$ob $W/pgmodel/$ob --transfers 16 -q; fi; }
  rclone copy $DS/gt_pg/$k $W/gt/$k --transfers 16 -q
  mkdir -p /dev/shm/dkta/ag_$k
  [ -s /dev/shm/dkta/ag_$k/gt_$k/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/gt/$k/simulation_ply --config $W/gt/$k/config.json --out_pt /dev/shm/dkta/ag_$k/$k.pt --out_dir /dev/shm/dkta/ag_$k/gt_$k > /root/pr/cv_$k.log 2>&1
  say "[장면] $k 3DGS $(ls $W/pgmodel/$ob/point_cloud 2>&1 | tr '\n' ' ') pt $(stat -c %s /dev/shm/dkta/ag_$k/$k.pt 2>&1) 정답 $(ls /dev/shm/dkta/ag_$k/gt_$k 2>/dev/null | wc -l) $(tail -1 /root/pr/cv_$k.log | cut -c1-120)"
  [ -s $W/BA_1.txt ] || rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $W/BA_1.txt
done
mkdir -p $W/gtnew; for k in pillow2sofa; do [ -d $W/newsc/${k}_h5 ] && [ ! -e $W/gtnew/$k ] && ln -s $W/newsc/${k}_h5 $W/gtnew/$k; [ -d $W/newsc/gt_$k ] && [ ! -e $W/gtnew/gt_$k ] && ln -s $W/newsc/gt_$k $W/gtnew/gt_$k; done
say "[자료] L000 $(stat -c %s $SD/L000.pt 2>&1) bread $(stat -c %s $AG/bread.pt 2>&1) wolf $(stat -c %s $AG/wolf.pt 2>&1) newsc $(ls $W/newsc 2>&1 | tr '\n' ' ') lbdata $(ls $W/lbdata 2>&1 | tr '\n' ' ')"
python -c "import torch;print(torch.cuda.is_available())" | grep -q True || { say "[CUDA 없음]"; exit 1; }
echo "[ok] $(date)" | rclone rcat $MK/$T.setup.done
[ -s /root/ffx/queue.txt ] || { rclone cat $MK/$T.queue 2>/dev/null | grep . > /root/ffx/queue.txt; [ -s /root/ffx/queue.txt ] || echo "${FPA_JOBS:-}" | tr ',' '\n' | grep . > /root/ffx/queue.txt; }
say "[대기열] $(wc -l < /root/ffx/queue.txt) 개"
WP=""; for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do FPA_GPU=$i bash exe/ffx_worker.sh "$H" >> $LOG 2>&1 & WP="$WP $!"; sleep 2; done
( while sleep 300; do rclone copyto $LOG $MK/$T.log -q; done ) & SY=$!
wait $WP
kill $SY 2>/dev/null
say "[일꾼 끝]"
echo "$(date)" | rclone rcat $MK/$T.END
