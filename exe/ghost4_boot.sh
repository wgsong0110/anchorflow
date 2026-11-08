#!/bin/bash
# 절 CM53 묶음 ghost4 잡 안 상주 시작점 (exe/job_run.sh AF_DETACH=1 로 한 번만 띄움; 인자 없음; SSH 를 단계마다 쓰지 않으려고).
# 1 분마다 R2 lbours/ghost4/code/launch.txt ('<커밋> <단계[,단계]>' 또는 'stop') 를 읽어 새 줄이면 그 커밋 bundle 로 exe/go_ghost4.sh 를 (끝날 때까지) 돌림.
# 살아 있음 표시 = R2 lbours/ghost4/boot_alive.txt (5 분마다), 상주 기록 = R2 lbours/ghost4/boot.log. 'stop' 이나 30 시간 지나면 끝.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/ghost4
CB=/tmp/dkta/ghost4_code; mkdir -p $CB /dev/shm/dkta
BL=/dev/shm/dkta/ghost4_boot.log
blog() { echo "[boot $(date '+%F %T')] $*" >> $BL; rclone copyto $BL $R/boot.log -q; }
LAST=$(cat /dev/shm/dkta/ghost4_boot.last 2>/dev/null)
T0=$(date +%s); TA=0
blog "상주 시작 $(hostname) pid $$ 마지막 처리 '$LAST'"
while [ $(( $(date +%s) - T0 )) -lt 108000 ]; do
  LN=$(rclone cat $R/code/launch.txt 2>/dev/null | head -1)
  if [ "$LN" = "stop" ]; then blog "stop 받음"; break; fi
  if [ -n "$LN" ] && [ "$LN" != "$LAST" ]; then
    LAST="$LN"; echo "$LN" > /dev/shm/dkta/ghost4_boot.last
    read -r H STG <<< "$LN"
    blog "새 실행 '$LN'"
    [ -s $CB/ghost4_$H.bundle ] || rclone copyto $R/code/ghost4_$H.bundle $CB/ghost4_$H.bundle -q
    [ -d $CB/$H/.git ] || git clone -q -b k1new $CB/ghost4_$H.bundle $CB/$H
    if (cd $CB/$H && git checkout -q $H); then
      (cd $CB/$H && bash exe/go_ghost4.sh $H $STG < /dev/null >> $BL 2>&1)
      blog "끝 '$LN'"
    else
      blog "코드 실패 '$LN'"
    fi
  fi
  if [ $(( $(date +%s) - TA )) -ge 300 ]; then TA=$(date +%s); echo "$(date '+%F %T') pid $$ last '$LAST' 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" | rclone rcat $R/boot_alive.txt; fi
  sleep 60
done
blog "상주 끝"
