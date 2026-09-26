#!/bin/bash
# 격자 물리손실(i-PG 격자 목적함수)**만** 으로 학습한다. 바닥 접촉항 포함.
# 위치 감독은 0 이고 (--phys_sup 0), phase2 경로는 loss = phys_w * wE 만 쓴다.
# 사용: phys_only.sh <TAG> <GPU> <one|traj>
#   one  : 한 조합(mic_clayC) 의 여러 궤적
#   traj : 한 궤적만
W=/home/dkta/work
TAG=$1; GPU=$2; MODE=${3:-one}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/abl_$TAG $W/tb
if [ "$MODE" = traj ]; then
  D=$W/one_traj_h2
  mkdir -p $D && ln -sf $W/traj_h2/mic_clayC_t_s400700.pt $D/ 2>/dev/null
  HOLD=""
else
  D=$W/traj_h2_mic
  mkdir -p $D
  for f in $W/traj_h2/mic_clayC_t_*.pt; do ln -sf "$f" $D/ 2>/dev/null; done
  # 한 조합 안에서 시드 두 개를 홀드아웃으로 뺀다
  HOLD="mic_clayC_t_s400706,mic_clayC_t_s400707"
fi
echo "[$TAG] 모드 $MODE, 궤적 $(ls $D | wc -l) 개, 홀드아웃 [$HOLD]"
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr ${LR:-3e-4} \
  --batch ${BS:-8} --n_pts ${NP:-8000} \
  --phase2 --phys_w 1.0 --phys_sup 0 --phys_K 1 --lambda_J 0 --lambda_dmg 0 \
  --iters ${ITX:-12000} ${HOLD:+--hold_traj "$HOLD"} --eval_t0 3 --eval_len 40 \
  --save_every 500 --val_every 500 --val_n ${VN:-2} --val_len 10 --tb $W/tb \
  >> $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
