#!/bin/bash
# 절 AA 2 차: 스윕 (exe/fem_aa_jobs.py, 비측정) 카드 1~6 동시 (GPU 0 은 같은 잡 절 Z 단독 측정 몫) -> 표
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_aa_lego; OF=$W/lbours/fem_aa_ficus; mkdir -p $OL $OF
for k in 0 1 2 3 4 5; do bash exe/fem_aa_gpu.sh $((k + 1)) $k 6 > $OL/gpu$((k + 1)).log 2>&1 < /dev/null & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
P=$(ls $OF/aa_f_*/pred.pt 2>/dev/null); [ -n "$P" ] && python exe/fem_p_period.py --data $FD --pred $P > $OF/period2.txt 2>&1
python exe/fem_aa_table.py $OL/aa_l_* > $OL/table2.md
python exe/fem_aa_table.py $OF/aa_f_* > $OF/table2.md
cat $OL/table2.md $OF/table2.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_aa_lego -q
rclone copy $OF r2:storage/result/anchorflow/lbours/fem_aa_ficus -q
echo AA2_DONE $(date +%T)
