#!/bin/bash
# 절 AK.10: 미달 5 + 대조 2 궤적의 pqf3 덤프 (비측정, f32) -> 충격 창 비교 (exe/fem_ak_press.py). 출력 덤프 폴더의 press.json, press.log + R2.
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
AK_KIND=F AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:pqf3:z:1:: 0001,0003,0016,0017,0018,0005,0013
i=0
for tc in 0001:16 0003:12 0016:7 0017:17 0018:19 0005:15 0013:12; do
  t=${tc%%:*}; c=${tc##*:}; O=/tmp/dkta/lbours/fem_ak_lego/${t}_pqf3_z_F1
  G=$((1 + i % 6)); i=$((i + 1))
  [ -f $O/roll_fem.pt ] || rclone copy r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1/roll_fem.pt $O -q   # 렌더 판이 R2 확인 뒤 /tmp 덤프를 지운 궤적
  [ -f $O/press.json ] && continue
  ( CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ak_press.py $O/roll_fem.pt /tmp/dkta/af_data/$t.pt $c > $O/press.log 2>&1
    rclone copy $O r2:storage/result/anchorflow/lbours/fem_ak_lego/${t}_pqf3_z_F1 -q ) &
done
wait
touch /tmp/dkta/lbours/fem_ak_logs/PRESS_DONE
echo "[끝] $(date +%T)"
