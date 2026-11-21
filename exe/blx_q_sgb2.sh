#!/bin/bash
# [blx] final_blx_sgb 대기열 (사용자 11-16: 새 인스턴스 금지·destroy 금지): Spring-Gaus bread 준비 판 끝 -> 같은 카드에서 단독 시간 1 회 (조정자 11-16: 이 카드에서 바로) + 덤프·물리·시각 지표 (2·3 회는 다른 GPU 일이 끝난 뒤) -> R2 동기화·대조. destroy 안 함.
source /root/fbenv.sh; source /root/pr/shenv.sh; [ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; OX=r2:storage/result/anchorflow/lbours/final_pa/blx; T=SGB
log() { echo "[q $(date -u '+%F %T')] $*" | tee -a /root/pr/q.log; rclone copyto /root/pr/q.log $OX/marks_$T/q.log -q; }
log "대기: 준비 판 res.json"
until [ -s $O/p/springgaus_bread_prep/res.json ] && grep -q "준비만 끝" $O/grid_main.log 2>/dev/null; do
  if grep -q "준비 실패" $O/stab.log 2>/dev/null; then log "준비 실패 -- 멈춤 (destroy 안 함)"; exit 1; fi
  sleep 60; done
log "준비 끝: $(cat $O/p/springgaus_bread_prep/res.json | tr -d '\n ')"
cd $D; rclone cat $OX/code/blx_$(cat /root/pr/qcommit).tar | tar -x -C $D
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|opencv-python-headless|scikit-image|scipy)==") > /root/pr/pyiqa.log 2>&1
( while true; do sleep 300; rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"; done ) & SY=$!
BLX_DUMP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u exe/pr2_grid.py --out $O --jobs exe/blx_jobs_sgb.json --reps 1 --scenes L000 \
  --until $(( $(date +%s) + 20 * 3600 )) --hard $(( $(date +%s) + 21 * 3600 )) >> $O/grid_main.log 2>&1
log "[측정 끝] $(tail -1 $O/grid_main.log)"
kill $SY 2>/dev/null
rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"
if rclone check $O $OX/$T --one-way --size-only -q --exclude "*/h5/**"; then log "[R2 확인 차이 0] destroy 안 함 (사용자 지시)"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_DONE
else log "[R2 확인 차이 있음]"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_CHECK_FAIL; fi
