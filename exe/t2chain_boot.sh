#!/bin/bash
# 절 CM56 묶음 t2chain 잡 안 상주 시작점 (exe/ghost4b_boot.sh 꼴; exe/job_run.sh AF_DETACH=1 로 한 번만, 'rclone cat ... | bash'). 노드 / 와 /tmp 는 안 씀:
# 코드 = R2 lbours/t2chain/code/t2chain_<커밋>.tar (git archive exe lib) 를 /dev/shm/dkta/t2chain_code/<커밋> 에 풂, TMPDIR = /dev/shm/dkta/tmp.
# 1 분마다 R2 lbours/t2chain/code/launch.txt ('<커밋> <단계[,단계]>' 또는 'stop') 를 읽어 새 줄이면 그 코드로 exe/go_t2chain.sh 를 (끝날 때까지) 돌림. 'stop' 이나 30 시간 지나면 끝.
export PATH=/home/dkta/bin:$PATH
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
R=r2:storage/result/anchorflow/lbours/t2chain
CB=/dev/shm/dkta/t2chain_code; mkdir -p $CB
BL=/dev/shm/dkta/t2chain_boot.log
blog() { echo "[boot $(date '+%F %T')] $*" >> $BL; rclone copyto $BL $R/boot.log -q; }
LAST=$(cat /dev/shm/dkta/t2chain_boot.last 2>/dev/null)
T0=$(date +%s); TA=0
blog "상주 시작 $(hostname) pid $$ 마지막 처리 '$LAST'; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4" 남음, 사용률 "$5}')"
while [ $(( $(date +%s) - T0 )) -lt 108000 ]; do
  LN=$(rclone cat $R/code/launch.txt 2>/dev/null | head -1)
  if [ "$LN" = "stop" ]; then blog "stop 받음"; break; fi
  if [ -n "$LN" ] && [ "$LN" != "$LAST" ]; then
    LAST="$LN"; echo "$LN" > /dev/shm/dkta/t2chain_boot.last
    read -r H STG <<< "$LN"
    blog "새 실행 '$LN'"
    if [ -f $CB/$H/exe/go_t2chain.sh ] || { mkdir -p $CB/$H && rclone cat $R/code/t2chain_$H.tar | tar -x -C $CB/$H && [ -f $CB/$H/exe/go_t2chain.sh ]; }; then
      (cd $CB/$H && bash exe/go_t2chain.sh $H $STG < /dev/null >> $BL 2>&1)
      blog "끝 '$LN'"
    else
      blog "코드 실패 '$LN'"
    fi
  fi
  if [ $(( $(date +%s) - TA )) -ge 300 ]; then TA=$(date +%s); echo "$(date '+%F %T') pid $$ last '$LAST' 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .) / shm $(df -h /dev/shm | tail -1 | awk '{print $4" "$5}') / 부하 $(cut -d' ' -f1 /proc/loadavg)" | rclone rcat $R/boot_alive.txt; fi
  sleep 60
done
blog "상주 끝"
