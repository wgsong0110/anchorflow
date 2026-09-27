#!/bin/bash
# phys_only.sh 와 **완전히 같은 설정**에 --out_var 만 더한 것.
# 학습 루프·랜덤 프레임 샘플링·배치·손실이 전부 같고, 갱신 대상만
# 망 파라미터에서 그 프레임의 출력(변위·반경·두께)으로 바뀐다.
# 사용: out_var.sh <TAG> <GPU>   (TRJ, ITX 로 궤적·반복 수)
W=/home/dkta/work
TAG=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache
mkdir -p $W/abl_$TAG $W/tb
D=$W/one_traj_h2
mkdir -p $D && ln -sf $W/traj_h2/${TRJ:-mic_clayC_t_s400706}.pt $D/ 2>/dev/null
echo "[$TAG] 출력변수 학습, 궤적: $(ls $D)"
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr ${LR:-3e-4} \
  --batch ${BS:-8} --n_pts ${NP:-8000} \
  --phase2 --phys_w 1.0 --phys_sup 0 --phys_K 1 --lambda_J 0 --lambda_dmg 0 \
  --iters ${ITX:-6000} --eval_t0 3 --eval_len 40 \
  --save_every 500 --val_every 100000 --tb $W/tb \
  --out_var --out_var_lr ${OVLR:-3e-3} \
  > $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
grep -aE "롤아웃\]|요약" $W/abl_$TAG.log | tail -3
