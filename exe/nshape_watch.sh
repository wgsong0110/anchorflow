#!/bin/bash
# 절 CM186 로컬 감시 (tmux nswatch): nshape_* 라벨 인스턴스만. R2 lbours/nshape/<라벨>.END 또는 .POWERCAP 이면 destroy (POWERCAP 은 blocklist 에 적음),
# loading 이 15 분 넘으면 destroy. 상태 줄은 /tmp/nshape/watch.log. 다른 라벨 인스턴스는 건드리지 않는다.
RX=r2:storage/result/anchorflow/lbours/nshape; L=/tmp/nshape/watch.log; declare -A FIRST
while true; do
  python3 - <<'PY' > /tmp/nshape/inst.txt 2>/dev/null
import json,subprocess,time
o=json.loads(subprocess.run(['vastai','show','instances-v1','--raw'],capture_output=True,text=True).stdout)
for i in o['instances']:
    lb=i.get('label') or ''
    if lb.startswith('nshape_'): print(i['id'], lb, i.get('actual_status'), i['machine_id'], i['host_id'], int(time.time()-i.get('start_date',time.time())))
PY
  END=$(rclone lsf $RX --include "*.END" --include "*.POWERCAP" 2>/dev/null)
  while read id lb st mid hid age; do
    [ -z "$id" ] && continue
    if echo "$END" | grep -q "^$lb.POWERCAP$"; then
      echo "$(date '+%F %T') $id $lb 전력 제한 -> destroy, blocklist m $mid h $hid" >> $L; vastai destroy instance $id -y >> $L 2>&1
      python3 -c "
import json;f='/home/wgsong/.vast_blocklist.json';b=json.load(open(f))
for k,v in (('machines',$mid),('hosts',$hid)):
    if v not in b[k]: b[k].append(v)
b['notes']['2026-11-15 nshape']='machine $mid host $hid power limit below default (CM186)'; json.dump(b,open(f,'w'),indent=1)"
      rclone moveto $RX/$lb.POWERCAP $RX/old/$lb.POWERCAP.$id -q
    elif echo "$END" | grep -q "^$lb.END$"; then
      echo "$(date '+%F %T') $id $lb END -> destroy" >> $L; vastai destroy instance $id -y >> $L 2>&1
    elif [ "$st" != running ] && [ "$age" -gt 900 ]; then
      echo "$(date '+%F %T') $id $lb 상태 $st ${age}s -> destroy" >> $L; vastai destroy instance $id -y >> $L 2>&1
    fi
  done < /tmp/nshape/inst.txt
  echo "$(date '+%F %T') $(cat /tmp/nshape/inst.txt | tr '\n' '|')" >> $L
  sleep 120
done
