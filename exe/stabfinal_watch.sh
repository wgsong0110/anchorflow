#!/bin/bash
# [stabfinal CM191] 로컬 감시 (tmux sfwatch): 5 분마다
#  1) mc_e (mic i-PG 12 시간 재실행) 가 main 을 끝내고 옛 판 (_f5) fin 으로 넘어가면 -> sf tmux 끄고 R2 동기화 뒤 destroy
#     (mic fin 은 f6m 인스턴스가 새 판 _f6 으로 함)
#  2) 크레딧·billing 상태, 각 인스턴스 표식 (ALL_DONE*, POWERCAP, setup_fail) 을 /tmp/sfwatch.log 에
#  3) 사건이 생기면 /tmp/sfwatch.event 에 한 줄 (에이전트가 깨어나서 봄)
RX=r2:storage/result/anchorflow/lbours/stabfinal
LOG=/tmp/sfwatch.log; EV=/tmp/sfwatch.event
K=$HOME/.ssh/id_ed25519_vast
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
ev() { say "[사건] $*"; echo "[$(date '+%F %T')] $*" >> $EV; }
declare -A SEEN
host() { vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); L=d if isinstance(d,list) else d.get('instances',d)
for i in L:
    if str(i['id'])=='$1':
        pp=(i.get('ports') or {}).get('22/tcp'); print(i['public_ipaddr'], pp[0]['HostPort'] if pp else i['ssh_port'], i.get('actual_status'), i.get('status_msg','')[:80].replace(' ','_'))"; }
while true; do
  # 1) mc_e
  if [ -z "${SEEN[mce]:-}" ] && rclone cat $RX/marks/mc_e/vast.log 2>/dev/null | grep -q "\[작업\] mic_metal_lego fin"; then
    read IP PT STT MSG < <(host 60863023)
    if [ -n "${IP:-}" ]; then
      ssh -n -i $K -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $PT root@$IP \
        'tmux kill-session -t sf; sleep 2; pkill -f stab1115_drv; sleep 2; rclone copy /home/dkta/work/stab/mic_metal_lego r2:storage/result/anchorflow/lbours/stabfinal/res/mic_metal_lego -q --update --transfers 8 --exclude "**/vis/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "gt/**" --exclude "traj0/**" --exclude "an.json" --exclude "extra.json" && rclone copy /root/sf r2:storage/result/anchorflow/lbours/stabfinal/marks/mc_e/logs -q --include "*.log" && echo SYNCOK' > /tmp/sfwatch_mce.out 2>&1
      if grep -q SYNCOK /tmp/sfwatch_mce.out; then vastai destroy instance 60863023 -y >> $LOG 2>&1; SEEN[mce]=1; ev "mc_e: mic i-PG main 끝 -> 동기화 뒤 destroy"; else say "mc_e 동기화 실패 $(tail -2 /tmp/sfwatch_mce.out | tr '\n' ' ')"; fi
    fi
  fi
  # 2) 표식
  for t in f6s f6m pw_b mc_e; do
    for m in ALL_DONE ALL_DONE_CHECK_FAIL POWERCAP setup_fail; do
      if [ -z "${SEEN[$t$m]:-}" ] && rclone lsf $RX/marks/$t/$m 2>/dev/null | grep -q .; then
        T=$(rclone lsl $RX/marks/$t/$m | awk '{print $2" "$3}' | cut -c1-19)
        if [[ "$T" > "2026-11-16 01:45" ]]; then SEEN[$t$m]=1; ev "$t $m ($T)"; fi
      fi
    done
  done
  vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); L=d if isinstance(d,list) else d.get('instances',d)
for i in L:
    if 'final_stab' in (i.get('label') or ''): print(i['id'], i['label'], i.get('actual_status'), round(i.get('gpu_util') or 0), (i.get('status_msg') or '')[:60].replace(chr(10),' '))" > /tmp/sfwatch_inst.txt
  say "인스턴스 $(tr '\n' ';' < /tmp/sfwatch_inst.txt) 크레딧 $(vastai show user --raw 2>/dev/null | python3 -c 'import sys,json;print(round(json.load(sys.stdin).get("credit",0),2))')"
  grep -qi "billing" /tmp/sfwatch_inst.txt && [ -z "${SEEN[bill]:-}" ] && { SEEN[bill]=1; ev "billing 문제: $(grep -i billing /tmp/sfwatch_inst.txt | head -1)"; }
  sleep 300
done
