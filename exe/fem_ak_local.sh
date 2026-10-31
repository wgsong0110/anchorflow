#!/bin/bash
# 절 AK.15: 미달 5 + 대조 2 의 pqf3 덤프로 남는 탄성의 자리·모양 (exe/fem_ak_local.py). 카드 1~6 나눠, 출력 덤프 폴더 local.json·local.log + R2.
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
i=0
for tc in 0001:16 0003:12 0016:7 0017:17 0018:19 0005:15 0013:12; do
  t=${tc%%:*}; c=${tc##*:}; O=/tmp/dkta/lbours/fem_ak_lego/${t}_pqf3_z_F1; G=$((1 + i % 6)); i=$((i + 1))
  ( [ -f $O/roll_fem.pt ] || rclone copy r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1/roll_fem.pt $O -q
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ak_local.py $O/roll_fem.pt /tmp/dkta/af_data/$t.pt $c > $O/local.log 2>&1
    rclone copy $O/local.json r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1 -q; rclone copy $O/local.log r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1 -q ) &
done
wait
touch /tmp/dkta/lbours/fem_ak_logs/LOCAL_DONE; echo "[끝] $(date +%T)"
