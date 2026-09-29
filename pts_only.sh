#!/bin/bash
# 증분 포텐셜을 **입자에서 바로** 재는 물리손실만으로 학습한다 (MPM 격자 미사용).
#   - 전달    :  trilinear 고정 가중치. 로컬(CW=none|rqs)과 글로벌(NW=tri|bound)
#                 은 독립이라 따로 고른다 (기본 rqs+bound)
#   - 목적함수:  E = Σ m/(2h²)‖Δu − h v − h² g‖² + Σ V Ψ(∇Φ·F) + 접촉항
#   - 속도    :  v = dΦ_t(x)/dt   (차분 아님, 순방향 AD)
#   - 변형구배:  F ← ∇Φ · F       (격자 B-스플라인 공간미분 아님)
#   - dt      :  조건 변수 (DtFiLM), 변위는 dt 에 비례
# 사용: pts_only.sh <TAG> <GPU> [one|traj]
W=/home/dkta/work
TAG=$1; GPU=$2; MODE=${3:-traj}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
export AF_SITREG=$W/SITReg/src MPLCONFIGDIR=/home/dkta/.mplcache
mkdir -p $W/abl_$TAG $W/tb
if [ "$MODE" = traj ]; then
  D=$W/one_traj_h2
  mkdir -p $D && ln -sf $W/traj_h2/${TRJ:-mic_clayC_t_s400706}.pt $D/ 2>/dev/null
  HOLD=""
else
  D=$W/traj_h2_mic
  mkdir -p $D
  for f in $W/traj_h2/mic_clayC_t_*.pt; do ln -sf "$f" $D/ 2>/dev/null; done
  HOLD="mic_clayC_t_s400706,mic_clayC_t_s400707"
fi
echo "[$TAG] 모드 $MODE, 궤적 $(ls $D | wc -l) 개, 홀드아웃 [$HOLD], 로컬 ${CW:-rqs}/글로벌 ${NW:-bound}"
python -u $W/anchorflow/exe/train_deform.py --data $D --out $W/abl_$TAG --tag $TAG \
  --no_mat --control --n_ctrl 2 --arch conv \
  --cell_warp ${CW:-rqs} --node_warp ${NW:-bound} \
  --vox_res 32 --k 16 --hidden 128 --depth 4 \
  --obj pts --dt_cond --dt_scale --v_from_dt ${DTS:+--dt_sub_set $DTS} \
  --lr ${LR:-3e-4} --batch ${BS:-8} --n_pts ${NP:-8000} \
  --phase2 --phys_w 1.0 --phys_sup 0 --phys_K ${PK:-1} --lambda_J 0 --lambda_dmg 0 \
  --iters ${ITX:-12000} ${HOLD:+--hold_traj "$HOLD"} --eval_t0 3 --eval_len 40 \
  --save_every 500 --val_every 500 --val_n ${VN:-2} --val_len 10 --tb $W/tb \
  ${RES:+--resume $RES} \
  >> $W/abl_$TAG.log 2>&1
echo TRAIN_DONE >> $W/abl_$TAG.log
