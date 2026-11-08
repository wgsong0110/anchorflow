#!/bin/bash
# 절 CM73 s3 앞: s2 wolf 판 (G'' 바탕) 의 방출 뒤 낙하 몫 (exe/wolfh4_relfall.py, 읽기만) -- pred.pt 는 R2 에만 있어 /dev/shm 에 받아 계산 뒤 로컬 사본만 지움. 인자: <출력> <R2>
O=$1; RO=$2; PY=/home/dkta/.conda/envs/af/bin/python; export PYTHONUTF8=1 OMP_NUM_THREADS=8
AG=/dev/shm/dkta/ag_data
for d in $O/w*_[0-9]*; do n=$(basename $d); [ -s $d/relfall.md ] && continue
  rclone copyto $RO/$n/pred.pt $d/pred.pt -q 2>/dev/null || { echo "$n pred.pt 없음"; continue; }
  CUDA_VISIBLE_DEVICES="" $PY exe/wolfh4_relfall.py $AG/wolf.pt $n=$d/pred.pt > $d/relfall.md 2> $d/relfall.err; echo "$n relfall rc $?"
  rm -f $d/pred.pt; rclone copy $d $RO/$n -q --include "relfall.*"
done
