#!/bin/bash
# 인스턴스에서 돌던 실행을 클러스터에서 이어 돌린다 (R2 체크포인트에서).
# 사용: pool_resume2.sh <TAG> <GPU> [추가인자...]
W=/home/dkta/work
TAG=$1; GPU=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/tb
# 체크포인트를 R2 에서 받아 온다 (없을 때만)
if [ ! -f $W/abl_$TAG/${TAG}_last.pt ]; then
  for src in inst/t3090 inst/t4090; do
    ~/bin/rclone copy r2:storage/result/anchorflow/$src/abl_$TAG $W/abl_$TAG \
      --transfers 4 >/dev/null 2>&1
    [ -f $W/abl_$TAG/${TAG}_last.pt ] && break
  done
fi
ls -la $W/abl_$TAG/${TAG}_last.pt || { echo "체크포인트가 없다: $TAG"; exit 1; }
COMBOS=hotdog_clayC,hotdog_elD,hotdog_viscoplastic,lego_clayC,lego_elD,lego_viscoplastic,mic_clayC,mic_elD,mic_viscoplastic,wolf_clayC,wolf_elD,wolf_viscoplastic
# 검증 집합은 **조합마다 한 개씩 12 개**로 고정한다. traj_hold12 를 그대로 주고
# --val_n 12 로 두면 파일 순서상 앞 12 개(hotdog 3 조합)만 보게 되어, 옮기기
# 전후로 재는 대상이 달라지고 곡선이 튄다 (실제로 재개 지점에서 0.87 -> 0.97).
if [ ! -d $W/hold1 ]; then
  mkdir -p $W/hold1
  for c in $(echo $COMBOS | tr ',' ' '); do
    f=$(ls $W/traj_hold12/${c}_t_*.pt 2>/dev/null | head -1)
    [ -n "$f" ] && ln -sf "$f" $W/hold1/
  done
fi
[ -f $W/hold12.txt ] || (cd $W/hold1 && ls *.pt | sed 's/\.pt$//' | paste -sd, > $W/hold12.txt)
echo "[검증] $(ls $W/hold1 | wc -l) 궤적: $(cat $W/hold12.txt)"
python -u $W/anchorflow/exe/train_deform.py --data $W/hold1 --out $W/abl_$TAG --tag $TAG \
  --mat_film --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr 3e-4 \
  --batch ${BS:-16} --n_pts ${NP:-8000} --pool --pool_combos "$COMBOS" \
  --pool_size ${PS:-128} --pool_fresh 0.25 --pool_thresh ${TH:-0.50} \
  --pool_keep ${KP:-0.5} --pool_window ${WIN:-30} \
  --iters ${ITX:-12000} --hold_traj "$(cat $W/hold12.txt)" --eval_t0 5 --eval_len 3 \
  --save_every 500 --val_every 500 --val_n 12 --val_len 10 --tb $W/tb \
  --resume $W/abl_$TAG/${TAG}_last.pt "$@" \
  >> $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
