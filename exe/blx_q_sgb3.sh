#!/bin/bash
# [blx] final_blx_sgb 대기열 3 판: Spring-Gaus bread 학습이 다른 에이전트 프로세스 (ablEig, 카드 37 GB) 때문에 메모리 부족으로 죽음 (11-16 15:17 UTC) ->
#   카드 0 에 다른 계산 프로세스가 0 이 되면 공식 동역학 학습 다시 (정적 3DGS·속도 단계·다시점 영상은 디스크에 있어 건너뜀; 다시점 렌더·prep 생략 = 같은 파일),
#   끝나면 res.json (pr2_grid prep 과 같은 꼴) -> 같은 카드 단독 시간 1 회 + 덤프·물리·시각 지표 (pr2_grid --reps 1 BLX_DUMP=1) -> R2 대조. destroy 안 함.
source /root/fbenv.sh; source /root/pr/shenv.sh; [ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; F=$O/p/springgaus_bread_prep; OX=r2:storage/result/anchorflow/lbours/final_pa/blx; T=SGB
log() { echo "[q3 $(date -u '+%F %T')] $*" | tee -a /root/pr/q.log; rclone copyto /root/pr/q.log $OX/marks_$T/q.log -q; }
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$"; }
log "대기: 카드 0 다른 계산 프로세스 0 ($(others | tr '\n' ' '))"
while [ -n "$(others)" ]; do sleep 20; done
cd $D; rclone cat $OX/code/blx_$(cat /root/pr/qcommit).tar | tar -x -C $D
( while true; do sleep 300; rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**"; done ) & SY=$!
log "학습 다시 시작"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${ST_SG_PYPATH:-} \
  python -u exe/stab_sg.py --mode train --sg ${ST_SG_REPO:-/dev/shm/dkta/ext/Spring-Gaus} --fit $F --out $F \
  --data /dev/shm/dkta/ag_data/bread.pt --tag bread --frames 300 --floor none --cfg_bc --pg_vdamp >> $F/log_2.txt 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s $F/sg_train.json ]; then log "학습 실패 rc=$rc"; kill $SY; rclone copy $O $OX/$T -q --exclude "*/h5/**"; exit 1; fi
python -c "import json,time;n=json.load(open('$F/sg_train.json'))['n_step'];json.dump(dict(rc=0,hi=int(n),sec=time.time()-$t0,card='0'),open('$F/res.json','w'),indent=1)"
log "학습 끝: $(cat $F/res.json | tr -d '\n ')"
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
