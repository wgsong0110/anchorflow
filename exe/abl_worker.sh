#!/bin/bash
# 절 CM183 일꾼: 작업 (장면:판:반복) 을 차례로 -- 카드 0 단독. 결과 R2 lbours/ablation/runs/<장면>_<판>_r<반복>.
export PATH=/home/dkta/bin:$PATH
H=$1; JOBS=$2
RA=r2:storage/result/anchorflow/lbours/ablation; O=/dev/shm/dkta/abl; mkdir -p $O /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
say() { echo "[w $(date '+%F %T')] $*"; }
for j in ${JOBS//,/ }; do IFS=: read s v r <<< "$j"
  n=${s}_${v}_r$r; D=$O/$n
  if rclone lsf $RA/runs/$n/metrics.json >/dev/null 2>&1 && [ -n "$(rclone lsf $RA/runs/$n/metrics.json 2>/dev/null)" ]; then say "$n 이미 있음"; continue; fi
  rm -rf $D; mkdir -p $D
  $PY exe/abl_gen.py $s $v $r $D > $D/cmd.txt || { say "$n 명령 생성 실패"; continue; }
  say "$n 시작 (다른 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 5400 bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/log.txt 2>&1; rc=$?
  say "$n rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $D/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $D/log.txt | tail -1 | cut -c1-200)"
  $PY exe/abl_an.py $D $s $v $r > $D/an.log 2>&1; say "$n 분석 rc=$? $(tail -1 $D/an.log | cut -c1-200)"
  [ -f $D/metrics.json ] || echo "{\"rc\": $rc, \"no_summary\": true}" > $D/metrics.json
  rclone copy $D $RA/runs/$n -q --transfers 8 --exclude "q27src/**" --exclude "pred.pt" $([ "$r" = 1 ] && echo "" || echo "")
  [ "$r" = 1 ] && [ -f $D/pred.pt ] && rclone copyto $D/pred.pt $RA/runs/$n/pred.pt -q
  rm -rf $D
done
say "일꾼 끝"
