#!/bin/bash
# A5 로컬 감시 (tmux 세션 stab_a5): 12 분마다 한 번 클러스터에 접속 (ControlMaster 재사용, ~/.ssh/config).
#  - 잡 상태는 qstat 으로만 본다. qstat 실패와 "잡 없음" 을 구분하고, PBS 서버 장애(errno 15010)면 그냥 기다린다.
#  - 실행 중인 잡이 있는데 드라이버 심박(stab/a5/heartbeat)이 5 분 넘게 멈췄거나 다른 잡의 것이면 job_run.sh 로 다시 띄운다.
#  - 잡이 없으면 새 잡이 잡힐 때까지 기다린다 (잡 제출·취소는 하지 않는다 -- keep5 가 넣는다).
#  - 끝(stab/a5/STAB_A5_DONE)이면 표·그림을 가져와 writing/anchorflow 에 넣고 컴파일, 로그에 STAB_A5_DONE, notify.
H=dkta@147.46.121.39; LOG=/tmp/stab_a5_local.log; A=/home/dkta/work/stab/a5
WR=/home/wgsong/workspace/writing/anchorflow
RELAUNCH="AF_DETACH=1 bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/stab_a5_go.sh $A/driver.out"
say() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
note_once() { k=$1; shift; [ -f /tmp/stab_a5_note_$k ] && return; touch /tmp/stab_a5_note_$k; notify "$@" --session anchorflow; }
say "감시 시작"
while true; do
  OUT=$(timeout 120 ssh -o ConnectTimeout=25 -o BatchMode=yes $H "/opt/pbs/bin/qstat -answ1 -u dkta 2>&1; echo QSTAT_RC=\$?; echo NOW=\$(date +%s); echo HB=\$(cat $A/heartbeat 2>/dev/null); ls $A/STAB_A5_DONE $A/REPORT_FAIL $A/selftest/*.fail 2>/dev/null | sed 's/^/MARK=/'; echo ===; cat $A/status.txt 2>/dev/null" 2>&1)
  if ! echo "$OUT" | grep -q "^NOW="; then say "접속 실패 -- 이번 틱 건너뜀: $(echo "$OUT" | tail -1 | cut -c1-120)"; sleep 720; continue; fi
  if echo "$OUT" | grep -q "MARK=$A/STAB_A5_DONE"; then
    mkdir -p /tmp/stab_a5_res
    scp -q "$H:$A/{a5.json,stab_a5_section.tex,stab_a5_metrics.png,nconf.json}" /tmp/stab_a5_res/ 2>>$LOG
    if [ -f /tmp/stab_a5_res/stab_a5_section.tex ]; then
      cp /tmp/stab_a5_res/stab_a5_section.tex /tmp/stab_a5_res/stab_a5_metrics.png $WR/
      (cd $WR && lualatex -interaction=nonstopmode stab_results.tex > /tmp/stab_a5_latex.log 2>&1); say "LaTeX 반영 (rc $?)"
    fi
    say "STAB_A5_DONE  $(echo "$OUT" | sed -n '/^===/,$p' | head -3 | tr '\n' ' ')"
    notify "A5 안정성 끝" "plane·pillow2sofa 네 방법 최소 N·물리·시각 지표 -> stab_results.tex" --tags white_check_mark --session anchorflow
    break
  fi
  for f in $(echo "$OUT" | grep "^MARK=" | grep -v STAB_A5_DONE | sed 's/^MARK=//'); do
    note_once "$(basename $f)" "A5 문제" "$(basename $f) -- 확인 필요" --priority high --tags x
  done
  if ! echo "$OUT" | grep -q "QSTAT_RC=0"; then
    if echo "$OUT" | grep -q "15010"; then say "PBS 서버 장애 (15010) -- 기다림"; else say "qstat 실패 -- 건너뜀: $(echo "$OUT" | head -1 | cut -c1-120)"; fi
    sleep 720; continue
  fi
  JOB=$(echo "$OUT" | awk '$10=="R" && /ssu_whole_a6gpu/{print $1; exit}')
  NOW=$(echo "$OUT" | sed -n 's/^NOW=//p'); HB=$(echo "$OUT" | sed -n 's/^HB=//p')
  HBT=$(echo $HB | awk '{print $1}'); HBJ=$(echo $HB | awk '{print $2}')
  ST=$(echo "$OUT" | sed -n '/^===/,$p' | sed 1d | head -6 | cut -c1-300 | tr '\n' '|')
  if [ -z "$JOB" ]; then say "실행 중 잡 없음 -- 새 잡 기다림 (대기 $(echo "$OUT" | grep -c ' Q ') 개)"; sleep 720; continue; fi
  AGE=$(( NOW - ${HBT:-0} ))
  if [ $AGE -gt 300 ] || { [ -n "$HBJ" ] && [ "$HBJ" != "?" ] && [ "$HBJ" != "$JOB" ]; }; then
    say "잡 $JOB 실행 중, 드라이버 심박 ${AGE}s 전 ($HBJ) -> 다시 띄움"
    R=$(timeout 120 ssh -o ConnectTimeout=25 -o BatchMode=yes $H "$RELAUNCH" 2>&1 | tail -3 | tr '\n' ' ')
    say "  $R"
  else
    say "진행: $ST"
  fi
  sleep 720
done
