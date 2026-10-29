#!/bin/bash
# 절 Q 2 차: 큰 K 쪽 촘촘한 스윕 (fem_q_jobs2.py) -- GPU 0~6 에 나눠 동시에 (비측정) -> 주기·표.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
for G in 0 1 2 3 4 5 6; do bash exe/fem_q_gpu.sh $G exe/fem_q_jobs2.py < /dev/null > $W/fem_q2_g$G.out 2>&1 & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_q_ficus/q_f_*/pred.pt > $W/lbours/fem_q_ficus/period2.txt 2>&1
python exe/fem_q_table.py $W/lbours/fem_q_lego/q_l_* > $W/lbours/fem_q_lego/table2.md
python exe/fem_q_table.py $W/lbours/fem_q_ficus/q_f_* > $W/lbours/fem_q_ficus/table2.md
rclone copy $W/lbours/fem_q_lego r2:storage/result/anchorflow/lbours/fem_q_lego -q
rclone copy $W/lbours/fem_q_ficus r2:storage/result/anchorflow/lbours/fem_q_ficus -q
echo Q2_DONE
