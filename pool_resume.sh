#!/bin/bash
# 클러스터에서 잡 만료로 끊긴 상태 풀 학습을 체크포인트에서 이어 돌린다.
# 사용: pool_resume.sh <TAG> <one|all> <GPU> [추가인자...]
W=/home/dkta/work
TAG=$1; SET=$2; GPU=$3; shift 3
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/met_abl_$TAG $W/tb
if [ "$SET" = one ]; then
  COMBOS=mic_clayC; MAT="--no_mat"
  HOLD="mic_clayC_t_s200076,mic_clayC_t_s200077,mic_clayC_t_s200078,mic_clayC_t_s200079"
  DATA=$W/traj_micF
else
  COMBOS=hotdog_clayC,hotdog_elD,hotdog_viscoplastic,lego_clayC,lego_elD,lego_viscoplastic,mic_clayC,mic_elD,mic_viscoplastic,wolf_clayC,wolf_elD,wolf_viscoplastic
  MAT="--mat_film"; HOLD="$(cat $W/holdF.txt)"; DATA=$W/traj_hold12
fi
AR="--arch conv --transfer skin --skin_corners --vox_res 32 --k 16"
python -u $W/anchorflow/exe/train_deform.py --data $DATA --out $W/abl_$TAG --tag $TAG \
  $MAT --control --n_ctrl 2 $AR --hidden 128 --depth 4 --lr 3e-4 \
  --batch ${BS:-16} --n_pts ${NP:-8000} --pool --pool_combos "$COMBOS" \
  --pool_size ${PS:-128} --pool_fresh 0.25 --pool_thresh ${TH:-0.01} \
  --iters ${ITX:-12000} --hold_traj "$HOLD" --eval_t0 5 --eval_len 3 \
  --save_every 500 --val_every 500 --val_n 4 --val_len 10 --tb $W/tb \
  --resume $W/abl_$TAG/${TAG}_last.pt "$@" \
  >> $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
