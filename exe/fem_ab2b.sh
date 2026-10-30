#!/bin/bash
# 절 AB 2b 차: navg β1 + 클램프 c 3·2 (fem_ab_jobs2b.py, 카드 1·2 x 일꾼 2, 비측정) -> 표 (2 차 표도 다시)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OD=/tmp/dkta/lbours; mkdir -p $OD/fem_ab_lego $OD/fem_ab_ficus
for k in 0 1 2 3; do bash exe/fem_ab_gpu.sh $((1 + k % 2)) $k 4 exe/fem_ab_jobs2b.py > $OD/fem_ab_lego/w2b_$k.log 2>&1 < /dev/null & done
wait
python exe/fem_ab_table.py $OD/fem_ab_lego/ab2_l_* > $OD/fem_ab_lego/table2.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/ab2_f_* > $OD/fem_ab_ficus/table2.md
python exe/fem_ab_table.py $OD/fem_ab_lego/ab_l_* > $OD/fem_ab_lego/table1.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/ab_f_* > $OD/fem_ab_ficus/table1.md
rclone copy $OD/fem_ab_lego r2:storage/result/anchorflow/lbours/fem_ab_lego -q
rclone copy $OD/fem_ab_ficus r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo AB2B_DONE $(date +%T)
