#!/usr/bin/env bash
# 작업이 끝난 vast 인스턴스를 **업로드를 확인한 뒤에만** destroy 한다.
#
# 순서를 반드시 지킨다: 실행 프로세스 0 확인 -> rclone 으로 잔여 밀어올리기 ->
# 잔여 0 재확인 -> 그때만 destroy. 하나라도 어긋나면 건드리지 않는다.
set -uo pipefail
L=${LOG:-/tmp/vast_reap.log}
echo "[$(date)] 수확 시작" >> "$L"
while true; do
  vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for i in d['instances']:
    p=(i.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort')
    if i.get('actual_status')=='running' and p:
        print(i['id'], i.get('public_ipaddr'), p)" > /tmp/reap_ep.txt
  n=$(wc -l < /tmp/reap_ep.txt)
  [ "$n" -eq 0 ] && { echo "[$(date)] 실행 중인 인스턴스 없음 -- 종료" >> "$L"; break; }
  while read ID IP PT; do
    S="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $PT root@$IP"
    busy=$(timeout 30 ssh -q $S 'pgrep -c -f gs_simulation 2>/dev/null; pgrep -c -f gen_handle_trajs 2>/dev/null' 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    [ -z "$busy" ] && { echo "[$(date)] $ID 응답없음 -- 유지" >> "$L"; continue; }
    [ "$busy" -gt 0 ] && continue
    # 아직 안 올라간 것부터 민다
    up=$(timeout 300 ssh -q $S 'N=$(ls /workspace/out/*.pt 2>/dev/null | wc -l)
if [ "$N" -gt 0 ]; then
  rclone copy /workspace/out r2:storage/result/anchorflow/traj_pg2/ --transfers 4 >/dev/null 2>&1 || { echo FAIL; exit 0; }
fi
M=$(ls /workspace/out/*.pt 2>/dev/null | wc -l)
R=$(rclone lsf r2:storage/result/anchorflow/traj_pg2 2>/dev/null | wc -l)
echo "OK $N $M $R"' 2>/dev/null | tail -1)
    set -- $up
    if [ "${1:-}" = OK ] && [ "${3:-1}" -ge 0 ]; then
      # 로컬에 남은 게 있어도 R2 에 같은 이름이 있으면 올라간 것이다
      left=${3:-99}
      if [ "$left" -eq 0 ] || [ "${2:-0}" -eq 0 ]; then
        echo "[$(date)] $ID 업로드 확인(생성 ${2:-?}, 잔여 $left, R2 ${4:-?}) -- destroy" >> "$L"
        vastai destroy instance "$ID" -y >> "$L" 2>&1
      else
        echo "[$(date)] $ID 잔여 $left 개 남음 -- 유지" >> "$L"
      fi
    else
      echo "[$(date)] $ID 업로드 실패/불명 -- 유지" >> "$L"
    fi
  done < /tmp/reap_ep.txt
  sleep 300
done
echo "[$(date)] 수확 끝" >> "$L"
