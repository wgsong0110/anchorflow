#!/bin/bash
# 절 AE 1c 차: navg β 0.25·0.5 x 탄성 부피 상한 θ 0.5·0.75·1 x 클램프 (AE_SET=3, lego 10 회·ficus 2 회), 카드 1~6 x 일꾼 2.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OD=/tmp/dkta/lbours; OL=$OD/fem_ae_lego; OF=$OD/fem_ae_ficus; mkdir -p $OL $OF
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
for G in 1 2 3 4 5 6; do for k in 0 1; do
  AE_SET=3 AE_R=10 bash exe/fem_ae_gpu.sh $G $(( (G - 1) * 2 + k )) 12 exe/fem_ae_jobs.py < /dev/null > $OL/w3_${G}_$k.log 2>&1 &
done; done
wait
python exe/fem_ab_table.py $OL/ae_l_* > $OL/table1.md; python exe/fem_ab_table.py $OF/ae_f_* > $OF/table1.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ae_lego -q; rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ae_ficus -q
rclone check $OF r2:storage/result/anchorflow/lbours/fem_ae_ficus --one-way --size-only -q --include "*/pred.pt" && rm -f $OF/ae_f_*/pred.pt
echo AE1C_DONE $(date +%T)
