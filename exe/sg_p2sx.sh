#!/bin/bash
# (가) Spring-Gaus pillow2sofa 공식 동역학 학습 정확 재학습 (조정자 11-17: final_fix_bba 에서, final_fix_sg 의 (나) 200 반복 근사 재개와 동시).
#   sg_resume (5 반복마다 저장 + R2 lbours/final_pa/bl/P2X 바로 동기화, 옛 체크포인트 근사 재개 끔) + SG_NO_EMPTY_CACHE=1. 재시작하면 같은 스크립트가 저장점부터.
#   준비: Spring-Gaus 환경 (R2 code/sg_env_cf55da5.tar), 준비 판 (code/sg_fit_p2s.tar = final_fix_sg 의 data·정적 3DGS·init_velocity·BC, 같은 절대 경로), 코드 cf55da5 + 덧판 (repo exe/sg_resume.py).
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/final_pa/bl; OX=$R/P2X; RC=$R/code
L=/root/p2sx.log; say() { echo "[p2sx $(date -u '+%F %T')] $*" >> $L; rclone copyto $L $OX/p2sx.log -q; }
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; F=$O/p/springgaus_pillow2sofa_prep; C=/root/p2sx_code
mkdir -p /workspace/shm /home/dkta/.conda/envs/af/bin /home/dkta/bin /home/dkta/work/newsc
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -x /home/dkta/.conda/envs/af/bin/python ] || ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
[ -x /home/dkta/bin/rclone ] || ln -sf $(command -v rclone) /home/dkta/bin/rclone
say "시작"
[ -d /dev/shm/dkta/pylib/sg ] && [ -d /dev/shm/dkta/ext/Spring-Gaus ] || (cd /dev/shm/dkta && rclone cat $RC/sg_env_cf55da5.tar | tar -x)
[ -s $F/pillow2sofa.yaml ] || (mkdir -p $O/p && cd $O/p && rclone cat $RC/sg_fit_p2s.tar | tar -x)
[ -d $F/sg_resume/cur ] || rclone copy $OX/p/springgaus_pillow2sofa_prep/sg_resume/cur $F/sg_resume/cur -q 2>/dev/null
[ -s /home/dkta/work/newsc/pillow2sofa.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/paper_remeasure2/scenes/pillow2sofa/pillow2sofa.pt /home/dkta/work/newsc/pillow2sofa.pt -q
rm -rf $C; mkdir -p $C; rclone cat $R/code/fpabl_cf55da5.tar | tar -x -C $C; rclone cat $RC/sgresume_overlay_cf55da5_exact.tar | tar -x -C $C
python -c "import open3d" 2>/dev/null || { (apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1 || (apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1)) >/dev/null 2>&1; python -c "import open3d" 2>/dev/null || pip install -q open3d >/dev/null 2>&1; }
say "준비 끝: 저장점 $(python -c "import torch;print(torch.load('$F/sg_resume/cur/checkpoint/state.pt',map_location='cpu',weights_only=False)['iter'])" 2>/dev/null || echo 없음), GPU $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) 프로세스"
if [ ! -s $F/sg_train.json ]; then
  cd $C
  CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled PYTHONPATH=/dev/shm/dkta/pylib/sg \
    SG_RESUME_EVERY=5 SG_RESUME_LEGACY=0 SG_NO_EMPTY_CACHE=1 SG_RESUME_R2_MAP="$O=$OX" \
    python -u exe/stab_sg.py --mode train --sg /dev/shm/dkta/ext/Spring-Gaus --fit $F --out $F \
    --data /home/dkta/work/newsc/pillow2sofa.pt --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp >> $F/log_x.txt 2>&1
  rc=$?; say "학습 rc=$rc $(cat $F/sg_train.json 2>/dev/null | tr -d '\n ' | head -c 300)"
  [ -s $F/sg_train.json ] || exit 1
fi
rclone copy $F $OX/p/springgaus_pillow2sofa_prep -q --transfers 8 --exclude "data/**" --exclude "mv/**"
say "R2 올림 끝 -> DONE"; echo "$(date -u)" | rclone rcat $OX/DONE -q
