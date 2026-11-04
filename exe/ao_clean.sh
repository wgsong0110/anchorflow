#!/bin/bash
# 절 AO: 노드 /tmp 의 이 가지 덤프 (pred.pt·pred_part.pt) 중 R2 에 같은 크기로 있는 것만 지운다 (지금 돌고 있는 묶음 AO_SKIP 제외). 지운 목록·전후 df 기록.
export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_ao; RR=r2:storage/result/anchorflow/lbours/fem_ao
df -h /tmp | tail -1
for d in $R/s*/*/; do
  b=$(basename $(dirname $d)); v=$(basename $d)
  case " ${AO_SKIP:-} " in *" $b "*) continue;; esac
  for f in pred.pt pred_part.pt; do
    [ -f $d$f ] || continue
    ls_=$(stat -c %s $d$f); rs_=$(rclone size $RR/$b/$v/$f --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["bytes"])' 2>/dev/null)
    if [ "$ls_" = "$rs_" ]; then rm -f $d$f && echo "지움 $b/$v/$f $ls_"; else echo "남김 $b/$v/$f 로컬 $ls_ R2 ${rs_:-없음}"; fi
  done
done
df -h /tmp | tail -1
