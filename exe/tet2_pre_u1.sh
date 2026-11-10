#!/bin/bash
# 절 CM103 u1: 동시 올리기 (20 개 넘게) 가 끝나지 않아 끊은 덤프를 하나씩 올림 -- R2 크기 확인 뒤 로컬 사본만 지움 (R2 그대로).
O=$1; RO=$2
for d in $O/*/; do n=$(basename $d); [ -f $d/pred.pt ] && [ -f $d/summary.json ] && [ ! -f $d/.upl ] || continue
  rclone copyto $d/pred.pt $RO/$n/pred.pt -q --transfers 4
  if [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $d/pred.pt)" ]; then rm -f $d/pred.pt; echo "$(date '+%F %T') $n 올림·로컬 사본 지움"; else echo "$(date '+%F %T') $n 올리기 실패"; fi
  rclone copy $d $RO/$n -q --exclude pred.pt
done
