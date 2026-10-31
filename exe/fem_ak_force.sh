#!/bin/bash
# 절 AK.21: 프레임 시작 노드 힘 진단 (exe/fem_ak_force.py) -- pqf3 덤프 7 궤적, 카드 2~6 나눠.
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
i=0
for tf in 0001:26 0003:20 0016:20 0017:22 0018:23 0005:21 0013:17; do
  t=${tf%%:*}; f=${tf##*:}; O=/tmp/dkta/lbours/fem_ak_lego/${t}_pqf3_z_F1; G=$((2 + i % 5)); i=$((i + 1))
  ( [ -f $O/roll_fem.pt ] || rclone copy r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1/roll_fem.pt $O -q
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ak_force.py $O/roll_fem.pt /tmp/dkta/af_data/$t.pt $f > $O/force2.log 2>&1
    rclone copy $O/force2.log r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1 -q; rclone copy $O/force.json r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1 -q ) &
done
wait; echo "[끝] $(date +%T)"
