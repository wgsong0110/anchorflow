#!/bin/bash
# 인스턴스용 상태 풀 학습 러너. GPU 한 장당 하나씩 돈다 (공유 없음).
W=/root/work
TAG=$1; GPU=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
export AF_WORK=$W
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/tb
HOLD="mic_clayC_t_s200076,mic_clayC_t_s200077,mic_clayC_t_s200078,mic_clayC_t_s200079"
python -u $W/anchorflow/exe/train_deform.py --data $W/evaltraj --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr 3e-4 \
  --batch ${BS:-16} --n_pts ${NP:-8000} --pool --pool_combos mic_clayC \
  --pool_size ${PS:-128} --pool_fresh 0.25 --pool_thresh ${TH:-0.02} \
  --iters ${ITX:-12000} --hold_traj "$HOLD" --eval_t0 5 --eval_len 3 \
  --save_every 500 --val_every 500 --val_n 4 --val_len 10 --tb $W/tb "$@" \
  > $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
rclone copy $W/abl_$TAG r2:storage/result/anchorflow/pool/$TAG 2>/dev/null
rclone copy $W/abl_$TAG.log r2:storage/result/anchorflow/pool/ 2>/dev/null
