#!/bin/bash
# 절 CM 잡 안 일꾼 (job_run.sh 로 한 번 띄움; 이후 SSH 없이): R2 lbours/fem_cm/w/tasks/<이름>.sh 를 1 분마다 보고 새 것을 띄움 (동시 여럿).
# 과제 출력 /dev/shm/dkta/cm<칸>/<이름>.log -> 1 분마다·끝에 R2 w/out/<이름>.log, 끝 표식 w/out/<이름>.END. R2 w/STOP 이 있거나 마감 (잡 만료 10 분 전) 이면 끝.
export PATH=$HOME/bin:$PATH
CW=${CMW:-w}; RR=r2:storage/result/anchorflow/lbours/fem_cm/$CW; W=/dev/shm/dkta/cm$CW;   # CMW = 일꾼 칸 (cm2 는 w2)
 mkdir -p $W/tasks
UNTIL=$(date -d '2026-11-05 03:24 UTC' +%s)
{ echo "[일꾼 시작] $(date '+%F %T') $(hostname) 마감 $(date -d @$UNTIL '+%F %T')"; df -h /dev/shm /tmp | tail -2; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; } > $W/worker.log 2>&1
rclone copyto $W/worker.log $RR/out/worker.log -q
run_task() { local n=$1 t=$2
  bash $W/tasks/$t > $W/$n.log 2>&1 < /dev/null & local TP=$!
  while kill -0 $TP 2>/dev/null; do sleep 60; rclone copyto $W/$n.log $RR/out/$n.log -q; done
  wait $TP; local rc=$?
  rclone copyto $W/$n.log $RR/out/$n.log -q
  echo "rc=$rc $(date '+%F %T')" > $W/$n.END; rclone copyto $W/$n.END $RR/out/$n.END -q
  echo "[과제 끝] $n rc=$rc $(date '+%F %T')" >> $W/worker.log; }
while [ "$(date +%s)" -lt "$UNTIL" ]; do
  rclone lsf $RR/ -q 2>/dev/null | grep -qx STOP && break
  for t in $(rclone lsf $RR/tasks/ -q 2>/dev/null | grep '\.sh$' | sort); do
    n=${t%.sh}; [ -f $W/$n.taken ] && continue
    touch $W/$n.taken; rclone copyto $RR/tasks/$t $W/tasks/$t -q || continue
    echo "[과제] $n $(date '+%F %T')" >> $W/worker.log
    run_task $n $t &
  done
  { echo "-- $(date '+%T')"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo; } >> $W/worker.log 2>&1
  rclone copyto $W/worker.log $RR/out/worker.log -q
  sleep 60
done
echo "[일꾼 끝] $(date '+%F %T')" >> $W/worker.log; rclone copyto $W/worker.log $RR/out/worker.log -q
