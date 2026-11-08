#!/bin/bash
# 절 CM53 묶음 ghost4 잡 안 상주 시작점 2 (노드 / 가 꽉 차 (11-08 07:16 UTC, 0 B 남음) /tmp 에 git clone 을 못 해서): 코드 = R2 lbours/ghost4/code/ghost4_<커밋>.tar (git archive exe lib) 를 /dev/shm 에 풂, TMPDIR 도 /dev/shm.
# 1 분마다 R2 lbours/ghost4/code/launch2.txt ('<커밋> <단계[,단계]>' 또는 'stop') 를 읽어 새 줄이면 그 코드로 exe/go_ghost4.sh 를 (끝날 때까지) 돌림. 'stop' 이나 30 시간 지나면 끝.
export PATH=/home/dkta/bin:$PATH
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
R=r2:storage/result/anchorflow/lbours/ghost4
CB=/dev/shm/dkta/ghost4_code; mkdir -p $CB
BL=/dev/shm/dkta/ghost4b_boot.log
blog() { echo "[boot $(date '+%F %T')] $*" >> $BL; rclone copyto $BL $R/boot2.log -q; }
LAST=$(cat /dev/shm/dkta/ghost4b_boot.last 2>/dev/null)
T0=$(date +%s); TA=0
blog "상주 시작 $(hostname) pid $$ 마지막 처리 '$LAST'; / $(df -h / | tail -1 | awk '{print $4}') · /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음"
while [ $(( $(date +%s) - T0 )) -lt 108000 ]; do
  LN=$(rclone cat $R/code/launch2.txt 2>/dev/null | head -1)
  if [ "$LN" = "stop" ]; then blog "stop 받음"; break; fi
  if [ -n "$LN" ] && [ "$LN" != "$LAST" ]; then
    LAST="$LN"; echo "$LN" > /dev/shm/dkta/ghost4b_boot.last
    read -r H STG <<< "$LN"
    blog "새 실행 '$LN'"
    if [ -f $CB/$H/exe/go_ghost4.sh ] || { mkdir -p $CB/$H && rclone cat $R/code/ghost4_$H.tar | tar -x -C $CB/$H && [ -f $CB/$H/exe/go_ghost4.sh ]; }; then
      (cd $CB/$H && bash exe/go_ghost4.sh $H $STG < /dev/null >> $BL 2>&1)
      blog "끝 '$LN'"
    else
      blog "코드 실패 '$LN'"
    fi
  fi
  if [ $(( $(date +%s) - TA )) -ge 300 ]; then TA=$(date +%s); echo "$(date '+%F %T') pid $$ last '$LAST' 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .) / $(df -h / | tail -1 | awk '{print $4}') shm $(df -h /dev/shm | tail -1 | awk '{print $4}')" | rclone rcat $R/boot2_alive.txt; fi
  sleep 60
done
blog "상주 끝"
