#!/bin/bash
# 절 CM102 s3: 수박 지표 (exe/tet1_wm.py, 읽기만) 를 노드에서 -- 판 덤프 (pred.pt 6.6 GB) 를 R2 에서 /dev/shm 으로 무리 하나씩 받아 계산 뒤 그 사본만 지움 (R2 그대로).
O=$1; RO=$2
PY=/home/dkta/.conda/envs/af/bin/python
D=/dev/shm/dkta/tet1_an; mkdir -p $D
WM=/dev/shm/dkta/ag_data/wm.pt
[ -s $WM ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WM.$$ -q && mv $WM.$$ $WM; }
get() { mkdir -p $D/$1; rclone copy $RO/$1 $D/$1 -q --include summary.json --include log.txt --include pred.pt --include lj.pt; }
: > $O/tet1_wm_table.md
for g in "H9:vH9M" "X9:xM" "T9:aM" "T9S:bM" "T9_이전:tM" "T9S_이전:sM"; do nm=${g%%:*}; p=${g##*:}; L=""
  if [ $p = vH9M ]; then get vH9M; L=",$D/vH9M"; else for i in 1 2 3 4 5 6 7 8; do get ${p}_$i; L="$L,$D/${p}_$i"; done; fi
  TET1_WM_ROWS=$O/tet1_wm_rows_$nm.json $PY exe/tet1_wm.py $WM $nm=${L#,} >> $O/tet1_wm_table.md 2>> $O/tet1_wm.err
  rclone copy $O $RO -q --include "tet1_wm*"
  rm -rf $D/*
done
rm -rf $D
