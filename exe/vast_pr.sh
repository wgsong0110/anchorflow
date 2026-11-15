#!/bin/bash
# [paper_remeasure 11-15] 논문용 파레토·지표 비교표의 "우리 (현재 기본 판)" 다시 재기 -- vast.ai RTX A6000 1 장 (옛 베이스라인 시간과 같은 카드 종류), 모든 판 직렬 = 카드 단독.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT PR_COMMIT (코드 tar R2 lbours/paper_remeasure/code/pr_<커밋>.tar), PR_STG (기본 pareto,stab), CONTAINER_ID·CONTAINER_API_KEY (vast)
#   순서: 파레토 (L000·ficus x K x PCG) -> stab (L000·ficus·wolf·bread: 사다리·단조 확인·덤프·렌더·시간). 결과 R2 lbours/paper_remeasure/{pareto,stab}, 표식 marks/.
#   정답 MPM·VR-GS·PhysTwin·Simplicits·Spring-Gaus 는 우리 판과 무관 -> 안 다시 돌림 (R2 lbours/pareto · lbours/stab 의 A6000 단독 값을 표에 그대로).
#   끝나면 R2 대조 차이 0 일 때 스스로 destroy.
set -u
H=${PR_COMMIT:?}; STG=${PR_STG:-pareto,stab}
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure; MK=$RX/marks
mkdir -p /workspace/shm /workspace/tmpd /root/pr
LOG=/root/pr/vast.log
say() { echo "[pr $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/vast.log -q 2>/dev/null; }
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
  echo "전력 제한 $PL < 기본 $PD $(date)" | rclone rcat $MK/POWERCAP; say "[전력 제한] $PL W < 기본 $PD W -- 일 안 함 (로컬이 destroy·blocklist)"; exit 1
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
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; SD=/dev/shm/dkta/anerr_data; MI=$R/_vastmirror; mkdir -p $AG $SD
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copyto $R/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $SD/F_pg_v0.pt -q &
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q &
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q &
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q &
rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt --transfers 16 -q &
rclone copy $MI/stab/lego/init $W/stab/lego/init -q &
rclone copy $MI/topo/pg_ficus_elastic_v0 $W/topo/pg_ficus_elastic_v0 --transfers 16 -q &
rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
D=/root/pr/code/$H; mkdir -p $D; rclone cat $RX/code/pr_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
mkdir -p $CUPY_CACHE_DIR
[ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/pr/gt_bread.log 2>&1
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/pr/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/pr/gt_wolf.log 2>&1
mkdir -p $W/one/L000; cp $SD/L000.pt $W/one/L000/L000.pt 2>/dev/null
say "[자료] L000 $(stat -c %s $SD/L000.pt 2>&1) ficus $(stat -c %s $SD/F_pg_v0.pt 2>&1) bread $(stat -c %s $AG/bread.pt 2>&1) wolf $(stat -c %s $AG/wolf.pt 2>&1) gt_bread $(ls $AG/gt_bread | wc -l) gt_wolf $(ls $AG/gt_wolf | wc -l) lbdata $(ls $W/lbdata | tr '\n' ' ') 코드 $(ls $D/exe/pr_stab.py 2>&1) pip $(ls /root/pr/pip.done 2>&1)"
[ -f /root/pr/pip.done ] && [ -s $SD/L000.pt ] && [ -s $SD/F_pg_v0.pt ] && [ -s $AG/bread.pt ] && [ -s $AG/wolf.pt ] && [ -f $D/exe/pr_stab.py ] && [ -s $AG/gt_bread/config.json ] && [ -s $AG/gt_wolf/config.json ] \
  && [ -s $W/lbdata/config.json ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -d $W/pgmodel/bread-trained ] && [ -d $W/pgmodel/wolf_whitebg-trained ] && [ -d $W/pgmodel/lego_whitebg-trained ] && [ -d $W/pgmodel/ficus_whitebg-trained ] \
  || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $MK/setup_fail; exit 1; }
echo "[ok] $(date)" | rclone rcat $MK/setup.done
say "[준비 끝] GPU 일 시작: $STG"
O=/dev/shm/dkta/lbours/paper_remeasure; mkdir -p $O/pareto $O/stab
( while true; do sleep 480; rclone copy $O $RX -q --transfers 8 --exclude "*/h5/**" --exclude "code/**"; done ) & SY=$!
if [[ $STG == *pareto* ]]; then
  rclone copy $R/lbours/pareto/points.json $O/pareto/old_points.json -q
  rclone copy $RX/pareto $O/pareto -q --include "t_ours_*/**"                                   # 이어 돌리기
  say "[파레토 시작]"
  python -u exe/pr_pareto.py --out $O/pareto --old $O/pareto/old_points.json >> $O/pareto/pareto_main.log 2>&1
  say "[파레토 끝] $(tail -1 $O/pareto/pareto_main.log)"; rclone copy $O/pareto $RX/pareto -q --transfers 8; echo "$(date)" | rclone rcat $MK/pareto.done
fi
if [[ $STG == *stab* ]]; then
  rclone copy $R/lbours/stab $O/stab -q --include "gt_*/done" --include "gt_*/png/**"           # 정답 렌더 (옛 stab 과 같은 것)
  rclone copy $RX/stab $O/stab -q --include "p/*/res.json" --include "p/*/vis.json" --include "p/*/pred.pt" --include "p/*/summary.json"   # 이어 돌리기
  say "[stab 시작]"
  python -u exe/pr_stab.py --out $O/stab --until $(( $(date +%s) + 7 * 3600 )) --hard $(( $(date +%s) + 8 * 3600 )) --scenes L000,ficus,wolf,bread >> $O/stab/stab_main.log 2>&1
  say "[stab 끝] $(cat $O/stab/stab.END 2>/dev/null)"; echo "$(date)" | rclone rcat $MK/stab.done
fi
kill $SY 2>/dev/null
rclone copy $O $RX -q --transfers 8 --exclude "*/h5/**" --exclude "code/**"
if rclone check $O $RX --one-way --size-only -q --exclude "*/h5/**" 2>/root/pr/check.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK/ALL_DONE
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/pr/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MK/ALL_DONE_CHECK_FAIL
fi
