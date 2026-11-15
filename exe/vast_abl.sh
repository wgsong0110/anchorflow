#!/bin/bash
# 절 CM183 ablation/강인성/민감도 -- vast.ai onstart 진입점 (exe/vast_pr2_boot.sh 에서 장면 준비 부분 + 작업 일꾼). 카드 1 장 = 일꾼 하나 (단독 시간).
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT, AB_COMMIT (코드 tar R2 lbours/ablation/code/abl_<커밋>.tar), AB_TAG (인스턴스 이름), AB_JOBS (쉼표: 장면:판:반복),
#        AB_SC (준비할 장면 쉼표: ficus,bread,wolf,vasedeck,plane,pillow2sofa,toast,sand_castle), CONTAINER_ID
#   표식·로그 = R2 lbours/ablation/marks/<AB_TAG>.* ; 결과 = R2 lbours/ablation/runs/<장면>_<판>_r<반복>/
set -u
H=${AB_COMMIT:?}; T=${AB_TAG:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; RA=$R/lbours/ablation; MK=$RA/marks
mkdir -p /workspace/shm /workspace/tmpd /root/pr
LOG=/root/pr/vast.log
say() { echo "[abl $T $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/$T.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $MK/$T.POWERCAP; say "[전력 제한] 일 안 함"; exit 1; fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
if [ ! -f /root/pr/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib pymcubes 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, lpips, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/pr/pip.done
fi
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; SD=/dev/shm/dkta/anerr_data; MI=$R/_vastmirror; mkdir -p $AG $SD
SC=${AB_SC:-ficus,bread}
has() { case ",$SC," in *",$1,"*) return 0;; esac; return 1; }
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q &
if has ficus; then rclone copyto $R/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $SD/F_pg_v0.pt -q & rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt --transfers 16 -q &
  rclone copy $MI/topo/pg_ficus_elastic_v0 $W/topo/pg_ficus_elastic_v0 --transfers 16 -q & fi
if has bread; then rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q & rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
  rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q & fi
if has wolf; then rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q & rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
  rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q & fi
D=/root/pr/code/$H; mkdir -p $D; rclone cat $RA/code/abl_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
mkdir -p $CUPY_CACHE_DIR
has bread && { [ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/pr/gt_bread.log 2>&1; }
has wolf && { [ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/pr/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/pr/gt_wolf.log 2>&1; }
PRSC=$(for k in vasedeck plane pillow2sofa; do has $k && echo -n "$k,"; done | sed 's/,$//')
if [ -n "$PRSC" ]; then export PR_SC=$PRSC RX; source exe/pr2_scene_fetch.sh; fi
if has toast || has sand_castle; then source exe/abl_gf_fetch.sh; fi
say "[자료] ficus $(stat -c %s $SD/F_pg_v0.pt 2>&1) bread $(stat -c %s $AG/bread.pt 2>&1) wolf $(stat -c %s $AG/wolf.pt 2>&1) newsc $(ls $W/newsc 2>&1 | tr '\n' ' ') gf $(ls /root/gfs/ag 2>&1 | tr '\n' ' ')"
echo "[ok] $(date)" | rclone rcat $MK/$T.setup.done
python -c "import torch;print(torch.cuda.is_available())" | grep -q True || { say "[CUDA 없음]"; exit 1; }
bash exe/abl_worker.sh "$H" "${AB_JOBS:?}" >> $LOG 2>&1
say "[일꾼 끝]"
echo "$(date)" | rclone rcat $MK/$T.END
