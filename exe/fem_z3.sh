#!/bin/bash
# 절 Z 3 차 (비측정 병렬): 장벽 강성·N 보충 (exe/fem_z_jobs2.py) -- 카드 Z_GPUS (기본 "4 5 6") -> 표·R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GS=(${Z_GPUS:-4 5 6}); K=${#GS[@]}
for k in $(seq 0 $((K - 1))); do bash exe/fem_z_gpu.sh ${GS[$k]} $k $K exe/fem_z_jobs2.py < /dev/null > $W/fem_z3_g${GS[$k]}.out 2>&1 & done
wait
python exe/fem_z_table.py $W/lbours/fem_z_lego/z_l_* > $W/lbours/fem_z_lego/table3.md
rclone copy $W/lbours/fem_z_lego r2:storage/result/anchorflow/lbours/fem_z_lego -q
echo Z3_DONE $(date +%T)
