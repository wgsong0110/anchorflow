#!/bin/bash
# 절 CM52: /dev/shm 넘침 (s4 vN_1 pred.pt 쓰기 실패·z 판 rc 135) -- 이 묶음 판 덤프 (pred.pt, h5) 를 /dev/shm 에서 노드 /tmp/dkta/wolfh1_keep 로 옮김 (지우지 않음; R2 확인 차이 0 인 사본도 있음). 옮기기 전 R2 대조. 인자: <출력> <R2>
O=$1; RO=$2; K=/tmp/dkta/wolfh1_keep; mkdir -p $K
echo "앞 /dev/shm $(df -h /dev/shm | tail -1)"
for f in $(cd $O && ls -d */pred.pt vid/*/h5 2>/dev/null); do
  case $f in vN_1/*|vid/vN_1/*) continue;; esac
  if [ -f $O/$f ]; then rclone check $O/$(dirname $f) $RO/$(dirname $f) --one-way --size-only --include pred.pt -q 2>/dev/null || { echo "R2 대조 실패 $f -- 안 옮김"; continue; }; fi
  mkdir -p $K/$(dirname $f) && cp -a $O/$f $K/$(dirname $f)/ && rm -rf $O/$f && echo "옮김 $f"
done
rm -rf $O/vN_1
echo "뒤 /dev/shm $(df -h /dev/shm | tail -1)"; echo "/tmp $(df -h /tmp | tail -1)"
