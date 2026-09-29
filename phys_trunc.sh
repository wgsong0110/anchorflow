#!/bin/bash
# 궤적의 **앞 N 프레임만** 물리손실로 학습한 뒤, 40 프레임 전체를 굴려 영상을 낸다.
# phys_only.sh / vid_h2.sh 와 같은 설정을 쓰고 데이터 길이만 다르다.
# 사용: phys_trunc.sh <N> <GPU>   (TRJ, ITX 로 궤적·반복 수 지정)
W=/home/dkta/work
N=$1; GPU=$2
TRJ=${TRJ:-mic_clayC_t_s400706}
TAG=PT$N
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache
D=$W/trunc_$N
mkdir -p $D $W/abl_$TAG $W/tb && rm -f $D/*.pt
python -u $W/anchorflow/exe/trunc_traj.py --src $W/traj_h2/$TRJ.pt \
  --dst $D/$TRJ.pt --frames $((N + 1))
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv --transfer rqs \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr ${LR:-3e-4} \
  --batch ${BS:-8} --n_pts ${NP:-8000} --hold_last 0 \
  --phase2 --phys_w 1.0 --phys_sup 0 --phys_K 1 --lambda_J 0 --lambda_dmg 0 \
  --iters ${ITX:-3000} --eval_t0 1 --eval_len $((N > 2 ? N - 1 : 1)) \
  --save_every 500 --val_every 100000 --tb $W/tb \
  > $W/abl_$TAG.log 2>&1
echo "TRAIN_DONE" >> $W/abl_$TAG.log
# 학습이 본 적 없는 뒷부분까지 포함해 40 프레임 전체를 굴린다
MAT="--no_mat" T0=3 LEN=40 bash $W/anchorflow/vid_h2.sh $TAG \
  $W/abl_$TAG/${TAG}_last.pt $W/traj_h2/$TRJ.pt $GPU
grep -a "요약" $W/vidh2_$TAG.log | tail -1
echo "PT_DONE $N" >> $W/abl_$TAG.log
