#!/bin/bash
# 로컬: 인스턴스가 running 이 되면 각 카드에서 i-PG 판 재개
H=f20bc64d
declare -A JOBS=( [60883110]="wolf_fr:25" [60863023]="wolf_fr:50" [60868387]="wolf_t:25" [60887262]="lsrej:L000:v" [60887253]="lsrej:pillow2sofa:v" [60866856]="lsrej:bread:r" )
vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import json,sys
for i in json.load(sys.stdin)['instances']:
  p=(i.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort')
  print(i['id'],i.get('actual_status'),i.get('public_ipaddr'),p)" > /tmp/ipg_inst_now.txt
while read id st ip port; do
  j=${JOBS[$id]:-}; [ -z "$j" ] && continue
  [ "$st" = running ] && [ "$port" != None ] || { echo "$id 아직 $st"; continue; }
  [ -e /tmp/ipg_resumed_$id ] && continue
  echo "== $id $ip:$port $j"
  ssh -n -i ~/.ssh/id_ed25519_vast -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o LogLevel=ERROR -p $port root@$ip "case $id in 60866856|60868387|60883110|60887253|60887262) tmux kill-session -t mb 2>/dev/null; pkill -f 'mpmbl_breadre[p]'; pkill -f 'mb_boo[t].sh';; esac; cd /root/mb/code && rclone cat r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/code/rs_$H.tar | tar x && /home/dkta/.conda/envs/af/bin/python exe/patch_ipg_noisefloor.py --ipg /home/dkta/work/i-physgaussian >/dev/null && tmux new-session -d -s restore \"bash exe/mpmbl_restore.sh $j > /root/mb/restore.out 2>&1\" && echo launched" && touch /tmp/ipg_resumed_$id
done < /tmp/ipg_inst_now.txt
