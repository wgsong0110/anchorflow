#!/bin/bash
# dkta 의 ssu_whole_a6gpu 잡이 실행(R)에 들어가는 순간 알린다. 10 분 간격 (SSH 반복 접속 금지 규칙).
# qstat 실패는 "잡 없음" 으로 보지 않는다 (그 틱은 건너뛴다).
H=dkta@147.46.121.39; F=/tmp/pbs_watch_run.state
while true; do
  OUT=$(timeout 90 ssh -o ConnectTimeout=25 -o BatchMode=yes $H '/opt/pbs/bin/qstat -answ1 -u dkta 2>&1; echo QSTAT_RC=$?' 2>&1)
  if ! echo "$OUT" | grep -q "QSTAT_RC=0"; then echo "[$(date '+%m-%d %H:%M')] qstat 실패 -- 건너뜀"; sleep 600; continue; fi
  R=$(echo "$OUT" | awk '$10=="R" && /ssu_whole_a6gpu/{split($12,a,"/"); print $1, a[1]}' | head -1)
  if [ -n "$R" ]; then
    if [ "$(cat $F 2>/dev/null)" != "$R" ]; then
      echo "$R" > $F; echo "[$(date '+%m-%d %H:%M')] 잡 실행 시작: $R"
      notify "클러스터 잡 잡힘" "dkta $R -- 7장 할당" --priority high --tags rocket --session anchorflow
    else echo "[$(date '+%m-%d %H:%M')] 실행 중 $R"; fi
  else rm -f $F; echo "[$(date '+%m-%d %H:%M')] 대기 중 ($(echo "$OUT" | grep -c ' Q ') 개)"; fi
  sleep 600
done
