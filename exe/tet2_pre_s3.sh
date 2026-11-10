#!/bin/bash
# 절 CM103 s3: 수박 지표 (exe/tet1_wm.py, 읽기만) 를 노드에서 -- 판 덤프를 R2 에서 /dev/shm 으로 무리 하나씩 받아 계산 뒤 그 사본만 지움 (R2 그대로).
# 무리: 새 판 nM_1~8 · 대조 기록 cM (T9S) · hM (H9) · 절 CM102 T9S bM_1~8 · H9 vH9M (lbours/tet1)
O=$1; RO=$2
PY=/home/dkta/.conda/envs/af/bin/python
R1=r2:storage/result/anchorflow/lbours/tet1
D=/dev/shm/dkta/tet2_an; mkdir -p $D
WM=/dev/shm/dkta/ag_data/wm.pt
[ -s $WM ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WM.$$ -q && mv $WM.$$ $WM; }
get() { mkdir -p $D/$2; rclone copy $1/$2 $D/$2 -q --include summary.json --include log.txt --include pred.pt --include lj.pt; }
: > $O/tet2_wm_table.md
for g in "새판|$RO|nM_1,nM_2,nM_3,nM_4,nM_5,nM_6,nM_7,nM_8" "T9S_기록|$RO|cM" "H9_기록|$RO|hM" "T9S_CM102|$R1|bM_1,bM_2,bM_3,bM_4,bM_5,bM_6,bM_7,bM_8" "H9_CM102|$R1|vH9M"; do
  IFS="|" read -r nm src runs <<< "$g"; L=""
  for p in ${runs//,/ }; do get $src $p; L="$L,$D/$p"; done
  TET1_WM_ROWS=$O/tet2_wm_rows_$nm.json $PY exe/tet1_wm.py $WM $nm=${L#,} >> $O/tet2_wm_table.md 2>> $O/tet2_wm.err
  rclone copy $O $RO -q --include "tet2_wm*"
  rm -rf $D/*
done
rm -rf $D
