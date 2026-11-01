#!/bin/bash
# 절 AG.19 수박 판별 진단 (계산 노드, 비측정)
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
D=/tmp/dkta/wmp1; mkdir -p $D
CUDA_VISIBLE_DEVICES=${G:-5} python -I exe/fem_ag19_wmp1.py /tmp/dkta/ag_data/wm.pt $D/wmp1.json 0.0094281 1,2,3,4,5 > $D/wmp1.log 2>&1
rclone copy $D r2:storage/result/anchorflow/lbours/fem_ag_wm/wmp1 -q
echo WMP1_DONE
