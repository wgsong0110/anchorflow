#!/bin/bash
# 인스턴스용 상태 풀 학습. GPU 한 장당 하나씩 돈다 (공유 없음).
# 기본이 **전체 12 조합**이고, 문턱을 넘은 상태는 확률적으로 그대로 남는다.
W=/root/work
TAG=$1; GPU=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
export AF_WORK=$W
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/tb
HOLD=$(cat $W/hold12.txt)
python -u $W/anchorflow/exe/train_deform.py --data $W/evaltraj --out $W/abl_$TAG --tag $TAG \
  ${MAT:---mat_film} --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr 3e-4 \
  --batch ${BS:-16} --n_pts ${NP:-8000} --pool --pool_combos "${COMBOS:-all}" \
  --pool_size ${PS:-128} --pool_fresh 0.25 --pool_thresh ${TH:-0.10} \
  --pool_keep ${KP:-0.5} --pool_window ${WIN:-30} \
  --iters ${ITX:-12000} --hold_traj "$HOLD" --eval_t0 5 --eval_len 3 \
  --save_every 500 --val_every 500 --val_n ${VN:-12} --val_len 10 --tb $W/tb "$@" \
  > $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
rclone copy $W/abl_$TAG r2:storage/result/anchorflow/pool/$TAG >/dev/null 2>&1
rclone copy $W/abl_$TAG.log r2:storage/result/anchorflow/pool/ >/dev/null 2>&1
