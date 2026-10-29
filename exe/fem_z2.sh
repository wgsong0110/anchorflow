#!/bin/bash
# 절 Z 2 차 스윕 (비측정 병렬): exe/fem_z_jobs.py 를 카드 Z_GPUS (기본 "4 5 6" -- 같은 잡의 절 Y 진단이 0~3 사용) 에 나눠 -> 표·ficus 주기·R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T) c = ${Z_C:-1.5 2 3 5}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GS=(${Z_GPUS:-4 5 6}); K=${#GS[@]}
for k in $(seq 0 $((K - 1))); do bash exe/fem_z_gpu.sh ${GS[$k]} $k $K < /dev/null > $W/fem_z2_g${GS[$k]}.out 2>&1 & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
python exe/fem_p_period.py --data $FD --pred $(ls -d $W/lbours/fem_z_ficus/z_f_*/pred.pt) > $W/lbours/fem_z_ficus/period2.txt 2>&1
python exe/fem_z_table.py $W/lbours/fem_z_lego/z_l_* > $W/lbours/fem_z_lego/table2.md
python exe/fem_z_table.py $W/lbours/fem_z_ficus/z_f_* > $W/lbours/fem_z_ficus/table2.md
rclone copy $W/lbours/fem_z_lego r2:storage/result/anchorflow/lbours/fem_z_lego -q
rclone copy $W/lbours/fem_z_ficus r2:storage/result/anchorflow/lbours/fem_z_ficus -q
echo Z2_DONE $(date +%T)
