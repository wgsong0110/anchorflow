#!/bin/bash
# 손잡이 2개 학생의 롤아웃 영상. 여러 실행을 **같은 궤적**으로 비교한다.
# 사용: vid_h2.sh <TAG> <체크포인트경로> [궤적파일] [GPU]
# MAT 로 물성 입력 방식을 맞춘다 -- 전 조합 학생은 --mat_film, 한 조합은 --no_mat.
# 틀리면 입력 채널이 달라 state_dict 가 안 맞는다.
W=/home/dkta/work
TAG=$1; CK=$2; TRJ=${3:-$W/traj_h2/mic_clayC_t_s400700.pt}; GPU=${4:-6}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8 MPLCONFIGDIR=/home/dkta/.mplcache
SEED=$(basename $TRJ .pt | sed 's/.*_t_//')
D=$W/traj_h2_one_$SEED
mkdir -p $D $W/vidh2_out && ln -sf $TRJ $D/ 2>/dev/null
: > $W/vidh2_$TAG.log
export AF_ROLL_DUMP=$W/rollh2_${TAG}.pt AF_ROLL_TAG=$SEED AF_ROLL_T0=3
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/vidh2_out \
  --tag vh2$TAG ${MAT:---mat_film} --control --n_ctrl 2 --arch conv --transfer skin \
  --skin_corners --vox_res 32 --k 16 --hidden 128 --depth 4 \
  --iters 0 --resume $CK --eval_t0 3 --eval_len 40 \
  --n_pts 20000 --gpu_data 0 --save_every 100000 >> $W/vidh2_$TAG.log 2>&1
python -u $W/anchorflow/exe/render_rollout_cmp.py --dump $W/rollh2_${TAG}.pt \
  --out $W/rollh2_${TAG}.mp4 >> $W/vidh2_$TAG.log 2>&1
echo "VIDH2_DONE $TAG" >> $W/vidh2_$TAG.log
