#!/bin/bash
# 로컬 (tmux s6rs) 에서: 워치독이 끈 (exited) stab 인스턴스를 데이터 준비 뒤에 다시 켜고 새 커밋으로 vast_s6.sh (S6_PART=stab) 를 다시 띄운다 (doc/pareto.md 절 S.7).
#   bash exe/vast_s6_restart.sh <인스턴스 id> <커밋>
# 1) wolf·bread 미러 (로컬 ~/s6x/pull.sh·pull2.sh) 가 R2 에 다 올라갈 때까지 기다림 (GPU 를 놀려 워치독이 끄지 않게 켜기 전에) -> R2 _vastmirror/wb.ok
# 2) vastai start -> running 까지 (30 분 넘게 자원 없음이면 destroy 하지 않고 보고만 -- 조정자 지시)
# 3) onstart 가 예전 AF_COMMIT 으로 띄운 s6 (예전 vast_s6.sh) 를 끄고, 같은 env 에 AF_COMMIT=<커밋>·S6_MWAIT=1 로 github 의 그 커밋 vast_s6.sh 를 tmux s6 에.
# 기록 R2 lbours/vast_s6/marks/restart_stab.log
set -u
ID=$1; C=$2
MI=r2:storage/result/anchorflow/_vastmirror; MK=r2:storage/result/anchorflow/lbours/vast_s6/marks
L=~/s6x/restart_stab.log
say() { echo "[restart $(date -u '+%F %T')] $*" | tee -a $L; rclone copyto $L $MK/restart_stab.log -q 2>/dev/null; }
say "시작 인스턴스 $ID 커밋 $C"
# ---- 1) wolf·bread 미러
SRC=""
for i in $(seq 360); do                                   # 30 초 x 360 = 3 h
  for f in ~/s6x/pull.log ~/s6x/pull2.log; do grep -q "topo/pg_wolf_sand rc=0" $f && grep -q "topo/pg_bread_plastic rc=0" $f && SRC=$f; done
  [ -n "$SRC" ] && break
  if grep -q "topo/pg_bread_plastic rc=" ~/s6x/pull.log && grep -q "topo/pg_bread_plastic rc=" ~/s6x/pull2.log; then   # 둘 다 오류로 끝났으면 한 번 대조
    S=":sftp,ssh='ssh -o ControlMaster=no -o ControlPath=none dkta@147.46.121.39':/home/dkta/work"
    ok=1; for p in topo/pg_wolf_sand topo/pg_bread_plastic; do rclone check "$S/$p" $MI/$p --one-way --size-only -q --sftp-known-hosts-file="" 2>>$L || ok=0; done
    [ $ok = 1 ] && SRC="rclone check" ; break
  fi
  sleep 30
done
if [ -n "$SRC" ]; then
  echo "wb ok $(date -u '+%F %T') ($SRC)" | rclone rcat $MI/wb.ok; say "wolf·bread 미러 끝 ($SRC) -> wb.ok"
else say "wolf·bread 미러 안 끝남 -> wb.ok 없이 (L000·ficus 만)"; fi
# ---- 2) start
vastai start instance $ID 2>&1 | tee -a $L
st() { vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status'), d.get('cur_state'), d.get('public_ipaddr'), (d.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort'), d.get('ssh_host'), d.get('ssh_port'), '|', d.get('status_msg'))"; }
for i in $(seq 180); do S1=$(st); case "$S1" in running*) break;; esac; sleep 10; done
say "상태: $S1"
case "$S1" in running*) ;; *) say "[자원 없음/안 켜짐] 30 분 동안 running 아님 -- destroy 안 함, 보고만"; notify "s6 stab 인스턴스 안 켜짐" "$ID: $S1" 2>/dev/null; exit 1;; esac
read -r _ _ IP PT SH SP _ <<< "$S1"
[ "$PT" = None ] && { IP=$SH; PT=$SP; }                    # 직접 포트가 없으면 프록시
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PT root@$IP"
for i in $(seq 60); do $SSH true 2>/dev/null && break; sleep 10; done
# ---- 3) onstart 의 예전 s6 를 끄고 새 커밋으로
for i in $(seq 18); do $SSH "tmux has-session -t s6" 2>/dev/null && break; sleep 10; done
$SSH 'cat /root/s6env.sh | grep -o "AF_COMMIT=[^ ]*"; ps -o pid,ppid,args -C bash | grep "/root/vast_s6.sh" | grep -v grep' 2>&1 | tee -a $L
$SSH 'pkill -f "bash /root/vast_s6.sh"; tmux kill-session -t s6 2>/dev/null; sleep 3; ps -o pid,args -C bash | grep "/root/vast_s6.sh" | grep -v grep; echo 남은것_위';
$SSH "set -e; sed -i -e 's/^export AF_COMMIT=.*/export AF_COMMIT=$C/' -e '/^export S6_MWAIT=/d' /root/s6env.sh; echo 'export S6_MWAIT=1' >> /root/s6env.sh
  grep -q '^export CONTAINER_ID=' /root/s6env.sh || echo 'export CONTAINER_ID=$ID' >> /root/s6env.sh
  curl -sfL https://raw.githubusercontent.com/wgsong0110/anchorflow/$C/exe/vast_s6.sh -o /root/vast_s6.sh
  grep -q S6_MWAIT /root/vast_s6.sh
  tmux new-session -d -s s6 'source /root/s6env.sh; bash /root/vast_s6.sh > /root/vast_s6.out 2>&1'
  sleep 5; tmux ls; grep -oE 'AF_COMMIT=[^ ]*|S6_PART=[^ ]*|S6_MWAIT=[^ ]*|CONTAINER_ID=[^ ]*' /root/s6env.sh; grep -c CONTAINER_API_KEY /root/s6env.sh" 2>&1 | tee -a $L
say "다시 띄움 끝 (커밋 $C) -- 이후 R2 marks/s6_stab.log"
