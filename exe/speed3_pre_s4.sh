#!/bin/bash
# 절 CM39 s4 앞: s1 wolf 이탈 진단 판 (uW1~6) 다시 분석 (speed3_wolfdiag.py JSON 키 고침 뒤; pred.pt 는 노드 /dev/shm 에만)
O=$1; RO=$2; PY=/home/dkta/.conda/envs/af/bin/python
for r in uW1 uW2 uW3 uW4 uW5 uW6; do
  [ -f $O/$r/pred.pt ] || { echo "$r pred.pt 없음"; continue; }
  CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$r/log.txt | awk '{print $2}')
  $PY exe/speed3_wolfdiag.py $CELLW $O/$r > $O/$r/wolfdiag.json 2>&1
  rclone copyto $O/$r/wolfdiag.json $RO/$r/wolfdiag.json -q
  echo "$r 끝 $(head -c 300 $O/$r/wolfdiag.json)"
done
