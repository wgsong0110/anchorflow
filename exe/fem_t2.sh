#!/bin/bash
# 절 T 2 차: 1 차 (fem_t1.sh) 카드 7 개가 다 끝난 뒤 큰 N 스윕 (exe/fem_t_jobs2.py, GPU 0~6 동시, 비측정) -> 표·주기·보존 (1·2 차 합쳐서)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
while [ $(ls $W/lbours/fem_s_lego/GPU?_DONE_fem_t_jobs 2>/dev/null | wc -l) -lt 7 ]; do sleep 60; done
echo "[1 차 끝 확인] $(date +%T)"; rm -f $W/lbours/fem_s_lego/GPU?_DONE_fem_t_jobs2
for G in 0 1 2 3 4 5 6; do bash exe/fem_s_gpu.sh $G exe/fem_t_jobs2.py < /dev/null > $W/fem_t2_g$G.out 2>&1 & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
sleep 30
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_t_ficus/t_f_*/pred.pt > $W/lbours/fem_t_ficus/period2.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_t_lego/t_l_* > $W/lbours/fem_t_lego/table2.md
python exe/fem_r_table.py $W/lbours/fem_t_ficus/t_f_* > $W/lbours/fem_t_ficus/table2.md
python exe/fem_s_cons.py $W/lbours/fem_t_lego/t_* $W/lbours/fem_t_ficus/t_* > $W/lbours/fem_t_lego/cons2.txt 2>&1
rclone copy $W/lbours/fem_t_lego r2:storage/result/anchorflow/lbours/fem_t_lego -q
rclone copy $W/lbours/fem_t_ficus r2:storage/result/anchorflow/lbours/fem_t_ficus -q
echo T2_DONE $(date +%T)
