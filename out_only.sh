#!/bin/bash
# phys_only.sh 를 그대로 복사하고 **학습을 출력만 최적화로** 바꾼 것.
# 망을 쓰지 않고 격자점 변위를 자유 변수로 두어 매 프레임 물리손실을 최소화한다.
# 나머지(데이터·손잡이·전달·평가·렌더)는 phys_only.sh / vid_h2.sh 와 같다.
# 사용: out_only.sh <TAG> <GPU>   (TRJ 로 궤적 지정)
W=/home/dkta/work
TAG=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache
mkdir -p $W/abl_$TAG $W/tb
D=$W/one_traj_h2
mkdir -p $D && ln -sf $W/traj_h2/${TRJ:-mic_clayC_t_s400706}.pt $D/ 2>/dev/null
echo "[$TAG] 출력만 최적화, 궤적 $(ls $D | wc -l) 개: $(ls $D)"
# --- 여기만 다르다: --iters 0 + --oracle_roll (망 출력 자리에 자유 변수) ---
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv --transfer skin --skin_corners \
  --vox_res 32 --k 16 --hidden 128 --depth 4 --lr ${LR:-3e-4} \
  --batch ${BS:-8} --n_pts ${NP:-20000} \
  --phase2 --phys_w 1.0 --phys_sup 0 --phys_K 1 --lambda_J 0 --lambda_dmg 0 \
  --iters 0 --eval_t0 3 --eval_len 40 --gpu_data 0 --save_every 100000 \
  ${CK:+--resume $CK} \
  --oracle_roll --oracle_steps ${OS:-300} --oracle_lr ${OLR:-1e-3} \
  --oracle_lr_shape ${OLRS:-1e-2} \
  --oracle_out $W/rollh2_${TAG}.pt \
  > $W/abl_$TAG.log 2>&1
grep -a "오라클\]" $W/abl_$TAG.log | tail -2
# --- 렌더는 vid_h2.sh 와 같은 경로 ---
python -u $W/anchorflow/exe/render_rollout_cmp.py --dump $W/rollh2_${TAG}.pt \
  --out $W/rollh2_${TAG}.mp4 >> $W/abl_$TAG.log 2>&1
echo "OUTONLY_DONE $TAG" >> $W/abl_$TAG.log
ls -la $W/rollh2_${TAG}.mp4
