#!/bin/bash
# 절 CM58 묶음 hhtc 잡 안 상주 시작점 (ghost6_boot.sh 복사) (노드 / 가 꽉 차 (11-08 07:16 UTC, 0 B 남음) /tmp 에 git clone 을 못 해서): 코드 = R2 lbours/hhtc/code/hhtc_<커밋>.tar (git archive exe lib) 를 /dev/shm 에 풂, TMPDIR 도 /dev/shm.
# 1 분마다 R2 lbours/hhtc/code/launch.txt ('<커밋> <단계[,단계]>' 또는 'stop') 를 읽어 새 줄이면 그 코드로 exe/go_hhtc.sh 를 (끝날 때까지) 돌림. 'stop' 이나 30 시간 지나면 끝.
export PATH=/home/dkta/bin:$PATH
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
R=r2:storage/result/anchorflow/lbours/hhtc
CB=/dev/shm/dkta/hhtc_code; mkdir -p $CB
BL=/dev/shm/dkta/hhtc_boot.log
blog() { echo "[boot $(date '+%F %T')] $*" >> $BL; rclone copyto $BL $R/boot.log -q; }
LAST=$(cat /dev/shm/dkta/hhtc_boot.last 2>/dev/null)
T0=$(date +%s); TA=0
blog "상주 시작 $(hostname) pid $$ 마지막 처리 '$LAST'; / $(df -h / | tail -1 | awk '{print $4}') · /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음"
while [ $(( $(date +%s) - T0 )) -lt 108000 ]; do
  LN=$(rclone cat $R/code/launch.txt 2>/dev/null | head -1)
  if [ "$LN" = "stop" ]; then blog "stop 받음"; break; fi
  if [ -n "$LN" ] && [ "$LN" != "$LAST" ]; then
    LAST="$LN"; echo "$LN" > /dev/shm/dkta/hhtc_boot.last
    read -r H STG <<< "$LN"
    blog "새 실행 '$LN'"
    if [ -f $CB/$H/exe/go_hhtc.sh ] || { mkdir -p $CB/$H && rclone cat $R/code/hhtc_$H.tar | tar -x -C $CB/$H && [ -f $CB/$H/exe/go_hhtc.sh ]; }; then
      (cd $CB/$H && bash exe/go_hhtc.sh $H $STG < /dev/null >> $BL 2>&1)
      blog "끝 '$LN'"
    else
      blog "코드 실패 '$LN'"
    fi
  fi
  if [ $(( $(date +%s) - TA )) -ge 300 ]; then TA=$(date +%s); { echo "$(date "+%F %T") pid $$ last '$LAST' / $(df -h / | tail -1 | awk '{print $4, $5}') shm $(df -h /dev/shm | tail -1 | awk '{print $4, $5}')"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } 2>&1 | rclone rcat $R/boot_alive.txt; fi
  sleep 60
done
blog "상주 끝"
