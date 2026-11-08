#!/bin/bash
# 절 CM52 묶음 wolfh1 잡 안 상주 시작점 둘째 (코드 받기를 /dev/shm/dkta 로 -- 노드 /tmp 98 %, 조정자 11-08: /tmp 쓰지 말 것). R2 code/launch2.txt 를 읽음.
# 1 분마다 R2 lbours/wolfh1/code/launch2.txt ('<커밋> <단계[,단계]>' 또는 'stop') 를 읽어 새 줄이면 그 커밋 bundle 로 exe/go_wolfh1.sh 를 (끝날 때까지) 돌림.
# 살아 있음 표시 = R2 lbours/wolfh1/boot_alive.txt (5 분마다), 상주 기록 = R2 lbours/wolfh1/boot.log. 'stop' 이나 30 시간 지나면 끝.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/wolfh1
CB=/dev/shm/dkta/wolfh1_code; mkdir -p $CB /dev/shm/dkta
BL=/dev/shm/dkta/wolfh1b_boot.log
blog() { echo "[boot $(date '+%F %T')] $*" >> $BL; rclone copyto $BL $R/boot2.log -q; }
LAST=$(cat /dev/shm/dkta/wolfh1b_boot.last 2>/dev/null)
T0=$(date +%s); TA=0
blog "상주 시작 $(hostname) pid $$ 마지막 처리 '$LAST'"
while [ $(( $(date +%s) - T0 )) -lt 108000 ]; do
  LN=$(rclone cat $R/code/launch2.txt 2>/dev/null | head -1)
  if [ "$LN" = "stop" ]; then blog "stop 받음"; break; fi
  if [ -n "$LN" ] && [ "$LN" != "$LAST" ]; then
    LAST="$LN"; echo "$LN" > /dev/shm/dkta/wolfh1b_boot.last
    read -r H STG <<< "$LN"
    blog "새 실행 '$LN'"
    [ -s $CB/wolfh1_$H.bundle ] || rclone copyto $R/code/wolfh1_$H.bundle $CB/wolfh1_$H.bundle -q
    [ -d $CB/$H/.git ] || git clone -q -b k1new $CB/wolfh1_$H.bundle $CB/$H
    if (cd $CB/$H && git checkout -q $H); then
      (cd $CB/$H && bash exe/go_wolfh1.sh $H $STG < /dev/null >> $BL 2>&1)
      blog "끝 '$LN'"
    else
      blog "코드 실패 '$LN'"
    fi
  fi
  if [ $(( $(date +%s) - TA )) -ge 300 ]; then TA=$(date +%s); echo "$(date '+%F %T') pid $$ last '$LAST' 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" | rclone rcat $R/boot2_alive.txt; fi
  sleep 60
done
blog "상주 끝"
