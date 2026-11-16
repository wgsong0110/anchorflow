#!/bin/bash
# [blx] Spring-Gaus bread 학습 뒤 측정 (조정자 11-17 07:00 KST: 학습은 ipg_L000 tmux sgmv_bread (final_fix_sg 담당, exe/sg_train_mv.sh), 측정은 blx).
#   학습 끝 (<FIT>/sg_train.json) -> res.json (q4 와 같은 꼴) -> (1) 덤프·물리·시각 지표 (BLX_NOTIME=1; 카드 공유 무관) -> (2) 단독 시간 1 회 (pr2_grid one_run:
#   카드 0 의 다른 계산 프로세스가 0 이 될 때까지 기다림) -> R2 blx/SGB 동기화·대조 -> marks_SGB/MEAS_DONE. destroy·다른 프로세스 손대지 않음.
#   사용 (인스턴스): [TIME_FIRST=1 [SKIP_DUMP=1]] bash blx_meas_bread.sh <커밋>  (TIME_FIRST: 시간 먼저 -> R2 grid.json, 덤프는 다른 카드에서 병렬 가능)
set -u
H=${1:?커밋}; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1
RA=r2:storage/result/anchorflow; OX=$RA/lbours/final_pa/blx; T=SGB
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; F=$O/p/springgaus_bread_prep; L=/root/blxmeas.log
log() { echo "[blxmeas $(date -u '+%F %T')] $*" | tee -a $L; rclone copyto $L $OX/marks_$T/blxmeas.log -q; }
D=/root/blx_code_$H; mkdir -p $D; rclone cat $OX/code/blx_$H.tar | tar -x -C $D
mkdir -p /tmp/dkta/ag_data /home/dkta/work; [ -e /tmp/dkta/ag_data/gt_bread ] || ln -s /dev/shm/dkta/ag_data/gt_bread_full /tmp/dkta/ag_data/gt_bread   # 렌더용 정답 궤적 (전 프레임)
export ST_SG_PYPATH=/dev/shm/dkta/pylib/sg ST_SG_REPO=/dev/shm/dkta/ext/Spring-Gaus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "대기: 학습 끝 ($F/sg_train.json)"
until [ -s $F/sg_train.json ]; do sleep 60; done
python -c "import json;n=json.load(open('$F/sg_train.json'))['n_step'];json.dump(dict(rc=0,hi=int(n),card='0',resume='sg_resume',host='ipg_L000'),open('$F/res.json','w'),indent=1)"
log "학습 끝: $(cat $F/res.json | tr -d '\n ')"
cd $D
( while true; do sleep 300; rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**" --exclude "p/springgaus_bread_prep/**" --exclude "p/springgaus_pillow2sofa_prep/**"; done ) & SY=$!
GA="--out $O --jobs exe/blx_jobs_sgb.json --reps 1 --scenes L000 --until $(( $(date +%s) + 20 * 3600 )) --hard $(( $(date +%s) + 21 * 3600 ))"
dump() { BLX_NOTIME=1 BLX_DUMP=1 python -u exe/pr2_grid.py $GA >> $O/grid_main.log 2>&1; log "[덤프·지표 끝] $(grep -h '덤프' $O/stab.log | tail -1 | cut -c1-300)"; rm -f $O/grid.END; }
tim() { log "단독 시간 1 회: 카드 0 다른 계산 프로세스 0 기다림 ($(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '))"
  BLX_DUMP=0 python -u exe/pr2_grid.py $GA >> $O/grid_main.log 2>&1; log "[시간 끝] $(grep -h '격자\]' $O/stab.log | tail -1)"; rm -f $O/grid.END
  rclone copy $O/g $OX/$T/g -q --include "springgaus_bread_n*_r*/**"; rclone copyto $O/grid.json $OX/$T/grid.json -q; }
if [ "${TIME_FIRST:-0}" = 1 ]; then tim; [ "${SKIP_DUMP:-0}" = 1 ] || dump; else dump; tim; fi
kill $SY 2>/dev/null
rclone copy $O $OX/$T -q --transfers 8 --exclude "*/h5/**" --exclude "p/springgaus_pillow2sofa_prep/**"
if rclone check $O $OX/$T --one-way --size-only -q --exclude "*/h5/**" --exclude "p/springgaus_pillow2sofa_prep/**"; then log "[R2 확인 차이 0]"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_DONE
else log "[R2 확인 차이 있음]"; echo "$(date)" | rclone rcat $OX/marks_$T/MEAS_CHECK_FAIL; fi
