#!/bin/bash
# 절 AG.18 bread 목 자리 진단 (계산 노드, 비측정). R2 의 우리 덤프 + 정답 데이터
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
D=/tmp/dkta/neck; mkdir -p $D
R=ag_bread_n160_tq_r1
[ -f $D/$R.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_bread/$R/roll_fem.pt $D/$R.pt -q
python -I exe/fem_ag_neck.py $D/$R.pt /tmp/dkta/ag_data/bread.pt $D/neck_$R.json 30 > $D/neck.log 2>&1
rclone copy $D/neck_$R.json r2:storage/result/anchorflow/lbours/fem_ag_bread/$R/ -q; rclone copy $D/neck.log r2:storage/result/anchorflow/lbours/fem_ag_bread/$R/ -q
rm -f $D/$R.pt
echo NECK_DONE
