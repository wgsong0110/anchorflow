#!/bin/bash
# Spring-Gaus 학습을 다른 카드로 옮겨 R2 저장점에서 이어 돌리기 (조정자 11-17 07:00: 원래 카드가 호스트 자원 없음으로 안 켜짐).
#   bash sg_train_mv.sh bread : final_blx_sgb 의 q4 (sg_resume 정확 저장점, R2 blx/SGB/.../sg_resume/cur) 를 이어 감 -- 같은 코드 blx_991b88a9, 같은 환경 변수
#   bash sg_train_mv.sh p2s   : pillow2sofa (나) = 200 반복 옛 체크포인트 근사 재개 (R2 bl/P2/.../exp/stab_pillow2sofa_2026_1116_0645_20), 그 뒤 5 반복마다 정확 저장점 (R2 bl/P2)
#   재시작하면 같은 명령이 가장 늦은 저장점부터. 끝나면 준비 판 (sg_train.json·마지막 exp) 을 R2 로, 표식 <R2>/MV_TRAIN_DONE.
set -u
S=${1:?bread|p2s}
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
RA=r2:storage/result/anchorflow; RB=$RA/lbours/final_pa/bl; RX=$RA/lbours/final_pa/blx
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; C=/root/sgmv_code_$S
L=/root/sgmv_$S.log
mkdir -p /workspace/shm /home/dkta/.conda/envs/af/bin /home/dkta/bin /home/dkta/work/newsc
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -x /home/dkta/.conda/envs/af/bin/python ] || ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
[ -x /home/dkta/bin/rclone ] || ln -sf $(command -v rclone) /home/dkta/bin/rclone
if [ $S = bread ]; then
  F=$O/p/springgaus_bread_prep; OX=$RX/SGB; FT=$RX/code/sg_fit_bread.tar; CT="$RX/code/blx_991b88a9.tar"; OV=""
  DATA=/dev/shm/dkta/ag_data/bread.pt; DSRC=$RA/lbours/fem_ag_data/bread.pt; TAG=bread; LG=log_4.txt
  ENVX="SG_RESUME_LEGACY=0 SG_NO_EMPTY_CACHE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
else
  F=$O/p/springgaus_pillow2sofa_prep; OX=$RB/P2; FT=$RB/code/sg_fit_p2s.tar; CT="$RB/code/fpabl_cf55da5.tar"; OV=$RB/code/sgresume_overlay_cf55da5.tar
  DATA=/home/dkta/work/newsc/pillow2sofa.pt; DSRC=$RA/lbours/paper_remeasure2/scenes/pillow2sofa/pillow2sofa.pt; TAG=pillow2sofa; LG=log_2.txt
  ENVX="SG_RESUME_LEGACY=1 SG_NO_EMPTY_CACHE=1"
fi
MK=$OX/marks_mv; say() { echo "[sgmv $S $(date -u '+%F %T')] $*" >> $L; rclone copyto $L $MK/sgmv_$S.log -q; }
say "시작 $(hostname)"
rclone lsf $OX/MV_TRAIN_DONE 2>/dev/null | grep -q . && { say "이미 끝"; exit 0; }
[ -d /dev/shm/dkta/pylib/sg ] && [ -d /dev/shm/dkta/ext/Spring-Gaus ] || (cd /dev/shm/dkta && rclone cat $RB/code/sg_env_cf55da5.tar | tar -x)
[ -s $F/$TAG.yaml ] || (mkdir -p $O/p && cd $O/p && rclone cat $FT | tar -x)
mkdir -p $(dirname $DATA); [ -s $DATA ] || rclone copyto $DSRC $DATA -q
[ -s $F/sg_resume/cur/checkpoint/state.pt ] || rclone copy $OX/p/$(basename $F)/sg_resume/cur $F/sg_resume/cur -q 2>/dev/null
if [ $S = p2s ]; then E=stab_pillow2sofa_2026_1116_0645_20; [ -s $F/exp/$E/dump_cfg.yaml ] || rclone copy $OX/p/$(basename $F)/exp/$E $F/exp/$E -q --exclude "evaluations/**" --exclude "dynamic_refine/**"; fi
rm -rf $C; mkdir -p $C; rclone cat $CT | tar -x -C $C; [ -n "$OV" ] && rclone cat $OV | tar -x -C $C
python -c "import open3d" 2>/dev/null || { (apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1 || (apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1)) >/dev/null 2>&1; python -c "import open3d" 2>/dev/null || pip install -q open3d >/dev/null 2>&1; }
say "준비 끝: 저장점 $(python -c "import torch;print(torch.load('$F/sg_resume/cur/checkpoint/state.pt',map_location='cpu',weights_only=False)['iter'])" 2>/dev/null || echo 없음), 카드 $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
if [ ! -s $F/sg_train.json ]; then
  cd $C
  env CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled PYTHONPATH=/dev/shm/dkta/pylib/sg SG_RESUME_EVERY=5 SG_RESUME_R2_MAP="$O=$OX" $ENVX \
    python -u exe/stab_sg.py --mode train --sg /dev/shm/dkta/ext/Spring-Gaus --fit $F --out $F \
    --data $DATA --tag $TAG --frames 300 --floor none --cfg_bc --pg_vdamp >> $F/$LG 2>&1
  rc=$?; say "학습 rc=$rc $(cat $F/sg_train.json 2>/dev/null | tr -d '\n ' | head -c 300)"
  [ -s $F/sg_train.json ] || exit 1
fi
EL=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['exp'])")
rclone copy $F $OX/p/$(basename $F) -q --max-depth 1
rclone copy $EL $OX/p/$(basename $F)/exp/$(basename $EL) -q --transfers 8
echo "$(date -u) $(hostname)" | rclone rcat $OX/MV_TRAIN_DONE -q; say "R2 올림 끝 ($EL)"
