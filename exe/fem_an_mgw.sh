#!/bin/bash
# 절 AN.20 잡 안 일꾼 (exe/fem_an_pkw.sh 와 같은 식, 이름만 mgw; job_run.sh 로 한 번 띄움; 이후 SSH 없이): R2 lbours/fem_an/mgw/tasks/<이름>.sh 를 1 분마다 보고, 새 것을 차례로 돈다.
# 과제 출력은 노드 /dev/shm/dkta/mgw/<이름>.log -> 1 분마다·끝에 R2 mgw/out/<이름>.log, 끝 표식 mgw/out/<이름>.END. R2 mgw/STOP 이 있거나 마감 (AF_PKW_UNTIL, 초) 이면 끝.
export PATH=$HOME/bin:$PATH
RR=r2:storage/result/anchorflow/lbours/fem_an/mgw; W=/dev/shm/dkta/mgw; mkdir -p $W/tasks
UNTIL=${AF_PKW_UNTIL:-1793849100}
echo "[일꾼 시작] $(date '+%F %T') $(hostname) 마감 $(date -d @$UNTIL '+%F %T')" > $W/worker.log
rclone copyto $W/worker.log $RR/out/worker.log -q
while [ "$(date +%s)" -lt "$UNTIL" ]; do
  rclone lsf $RR/ -q 2>/dev/null | grep -qx STOP && break
  for t in $(rclone lsf $RR/tasks/ -q 2>/dev/null | grep '\.sh$' | sort); do
    n=${t%.sh}; [ -f $W/$n.taken ] && continue
    touch $W/$n.taken; rclone copyto $RR/tasks/$t $W/tasks/$t -q || continue
    echo "[과제] $n $(date '+%F %T')" >> $W/worker.log; rclone copyto $W/worker.log $RR/out/worker.log -q
    bash $W/tasks/$t > $W/$n.log 2>&1 < /dev/null & TP=$!
    while kill -0 $TP 2>/dev/null; do sleep 60; rclone copyto $W/$n.log $RR/out/$n.log -q; done
    wait $TP; rc=$?
    rclone copyto $W/$n.log $RR/out/$n.log -q
    echo "rc=$rc $(date '+%F %T')" > $W/$n.END; rclone copyto $W/$n.END $RR/out/$n.END -q
    echo "[과제 끝] $n rc=$rc $(date '+%F %T')" >> $W/worker.log
  done
  rclone copyto $W/worker.log $RR/out/worker.log -q
  sleep 60
done
echo "[일꾼 끝] $(date '+%F %T')" >> $W/worker.log; rclone copyto $W/worker.log $RR/out/worker.log -q
