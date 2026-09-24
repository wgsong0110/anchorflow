#!/usr/bin/env bash
# 궤적이 저장되는 **즉시** R2 로 올린다.
#
# 묶음이 끝날 때 한 번 올리면 그 사이(20 분 남짓)에 인스턴스가 멈출 때 미업로드가
# 남는다. 여기서는 디렉토리를 계속 보다가 새 .pt 가 완성되면 바로 올리고 지운다.
# 쓰는 중인 파일을 올리지 않도록 크기가 두 번 연속 같을 때만 올린다.
set -uo pipefail
W=${W:-/workspace}
R2=${R2:-r2:storage/result/anchorflow/traj_pg2}
L=$W/uploader.log
declare -A SZ
echo "[$(date)] 업로더 시작" >> "$L"
while true; do
  for f in "$W"/out/*.pt; do
    [ -e "$f" ] || continue
    s=$(stat -c %s "$f" 2>/dev/null) || continue
    if [ "${SZ[$f]:-}" = "$s" ] && [ "$s" -gt 0 ]; then
      if rclone copy "$f" "$R2/" --transfers 2 >/dev/null 2>&1; then
        rm -f "$f"; unset 'SZ[$f]'
        echo "[$(date)] 올림 $(basename "$f")" >> "$L"
      else
        echo "[$(date)] 실패 $(basename "$f") -- 유지" >> "$L"
      fi
    else
      SZ[$f]=$s
    fi
  done
  sleep 20
done
