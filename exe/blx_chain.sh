#!/bin/bash
# [blx] 로컬 대기열 (tmux): Spring-Gaus 준비 판 태그 <PREP> 가 R2 에 끝나면 (marks_<PREP>/ALL_DONE + p/springgaus_<장면>_prep/res.json)
#   RTX A6000 (dph_total < 1.0, 전력 확인은 부팅이) 을 하나 띄워 단독 시간 3 회 + 덤프·물리·시각 지표 (grid 모드, 끝나면 R2 대조 뒤 스스로 destroy).
#   사용: bash exe/blx_chain.sh <PREP 태그> <장면> <작업 json> <새 태그> <커밋>
P=$1; T=$2; J=$3; NT=$4; H=$5
R=r2:storage/result/anchorflow/lbours/final_pa/blx
log() { echo "[chain $(date -u '+%F %T')] $*"; }
until rclone lsf $R/marks_$P/ 2>/dev/null | grep -q "^ALL_DONE"; do sleep 120; done
if ! rclone lsf $R/$P/p/springgaus_${T}_prep/res.json >/dev/null 2>&1 || [ -z "$(rclone lsf $R/$P/p/springgaus_${T}_prep/res.json 2>/dev/null)" ]; then
  log "준비 판 실패 ($P) -- 띄우지 않음"; exit 1; fi
log "준비 판 끝 ($P): $(rclone cat $R/$P/p/springgaus_${T}_prep/res.json | tr -d '\n ')"
B=$(python3 -c "import json;b=json.load(open('/home/wgsong/.wexec/blocklist.json'));print(' '.join(set(b.get('hosts',{}))|set(b.get('machines',{}))))")
while true; do
  OF=$(vastai search offers 'gpu_name=RTX_A6000 num_gpus=1 rentable=true dph_total<1.0 inet_down>200 disk_space>100 reliability>0.95' -o 'dph_total' --raw 2>/dev/null | python3 -c "
import json,sys
bl=set('$B'.split())
for o in json.load(sys.stdin):
    if str(o['machine_id']) in bl or str(o.get('host_id')) in bl: continue
    print(o['id']); break")
  if [ -n "$OF" ]; then
    out=$(DISK=100 bash exe/blx_launch.sh $OF final_blx_$(echo $NT | tr 'A-Z' 'a-z') $H $NT $J grid $P 2>&1)
    echo "$out" | grep -q '"success": true' && { log "띄움 offer $OF 태그 $NT: $(echo "$out" | grep new_contract)"; break; }
    log "띄우기 실패 offer $OF: $(echo "$out" | tail -2 | tr '\n' ' ')"
  else log "A6000 (<\$1) 오퍼 없음 -- 5 분 뒤 다시"; fi
  sleep 300
done
