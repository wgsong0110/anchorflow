#!/bin/bash
# 클러스터용 상태 풀 학습 (컨테이너 안에서 GPU 한 장씩).
W=/home/dkta/work
TAG=$1; GPU=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/tb
if [ "${COMBOS:-all}" = all ]; then
  MAT=${MAT:---mat_film}; HOLD="$(cat $W/holdF.txt)"; DATA=$W/traj_hold12; VN=${VN:-12}
else
  MAT=${MAT:---no_mat}; DATA=$W/traj_micF; VN=${VN:-4}
  HOLD="mic_clayC_t_s200076,mic_clayC_t_s200077,mic_clayC_t_s200078,mic_clayC_t_s200079"
fi
python -u $W/anchorflow/exe/train_deform.py --data $DATA --out $W/abl_$TAG --tag $TAG \
  $MAT --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr 3e-4 \
  --batch ${BS:-16} --n_pts ${NP:-8000} --pool --pool_combos "${COMBOS:-all}" \
  --pool_size ${PS:-128} --pool_fresh 0.25 --pool_thresh ${TH:-0.10} \
  --pool_keep ${KP:-0.5} --pool_window ${WIN:-30} \
  --iters ${ITX:-12000} --hold_traj "$HOLD" --eval_t0 5 --eval_len 3 \
  --save_every 500 --val_every 500 --val_n $VN --val_len 10 --tb $W/tb "$@" \
  > $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
