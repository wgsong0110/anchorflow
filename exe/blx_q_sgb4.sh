#!/bin/bash
# [blx] final_blx_sgb 대기열 4 판 (조정자 11-16 사용자 지시: 끊겨도 손실 없게): Spring-Gaus bread 공식 동역학 학습을 exe/sg_resume.py 경로로 처음부터
#   (5 반복마다 <FIT>/sg_resume/cur 저장 + R2 바로 동기화, 재시작하면 그 지점부터 같은 궤적; 옛 체크포인트 근사 재개는 끔),
#   가속 = SG_NO_EMPTY_CACHE=1 (프레임마다 empty_cache 만 끔, exe/sgr_test 로 결과 불변 확인) -- 그 밖에는 q_sgb3 와 같음
#   (끝나면 res.json -> 같은 카드 단독 시간 1 회 + 덤프·물리·시각 지표, pr2_grid --reps 1 BLX_DUMP=1). onstart 가 재시작 때 이 스크립트를 다시 띄움.
source /root/fbenv.sh; source /root/pr/shenv.sh; [ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; F=$O/p/springgaus_bread_prep; OX=r2:storage/result/anchorflow/lbours/final_pa/blx; T=SGB
log() { echo "[q4 $(date -u '+%F %T')] $*" | tee -a /root/pr/q.log; rclone copyto /root/pr/q.log $OX/marks_$T/q.log -q; }
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$"; }
QC=$(cat /root/pr/q4commit)
log "대기: 카드 0 다른 계산 프로세스 0 ($(others | tr '\n' ' ')), 코드 $QC"
while [ -n "$(others)" ]; do sleep 20; done
cd $D; rclone cat $OX/code/blx_$QC.tar | tar -x -C $D
( while true; do sleep 300; rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**" --exclude "*/sg_resume/**"; done ) & SY=$!
if [ ! -s $F/res.json ]; then
  log "학습 (sg_resume: $(ls $F/sg_resume/cur/checkpoint/state.pt 2>/dev/null && echo 저장점 있음 || echo 처음부터))"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${ST_SG_PYPATH:-} \
    SG_RESUME_EVERY=5 SG_RESUME_LEGACY=0 SG_NO_EMPTY_CACHE=1 SG_RESUME_R2_MAP="$O=$OX/$T" \
    python -u exe/stab_sg.py --mode train --sg ${ST_SG_REPO:-/dev/shm/dkta/ext/Spring-Gaus} --fit $F --out $F \
    --data /dev/shm/dkta/ag_data/bread.pt --tag bread --frames 300 --floor none --cfg_bc --pg_vdamp >> $F/log_4.txt 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s $F/sg_train.json ]; then log "학습 실패 rc=$rc"; kill $SY; rclone copy $O $OX/$T -q --exclude "*/h5/**"; exit 1; fi
  python -c "import json,time;n=json.load(open('$F/sg_train.json'))['n_step'];json.dump(dict(rc=0,hi=int(n),sec=time.time()-$t0,card='0',resume='sg_resume'),open('$F/res.json','w'),indent=1)"
  log "학습 끝: $(cat $F/res.json | tr -d '\n ')"
  rclone copy $F $OX/$T/p/springgaus_bread_prep -q --max-depth 1
fi
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|opencv-python-headless|scikit-image|scipy)==") > /root/pr/pyiqa.log 2>&1 &
PI=$!
BLX_DUMP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u exe/pr2_grid.py --out $O --jobs exe/blx_jobs_sgb.json --reps 1 --scenes L000 \
  --until $(( $(date +%s) + 20 * 3600 )) --hard $(( $(date +%s) + 21 * 3600 )) >> $O/grid_main.log 2>&1
wait $PI
log "[측정 끝] $(tail -1 $O/grid_main.log)"
kill $SY 2>/dev/null
rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"
if rclone check $O $OX/$T --one-way --size-only -q --exclude "*/h5/**"; then log "[R2 확인 차이 0] destroy 안 함"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_DONE
else log "[R2 확인 차이 있음]"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_CHECK_FAIL; fi
