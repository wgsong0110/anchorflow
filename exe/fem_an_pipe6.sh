#!/bin/bash
# 절 AN 파이프 6 (비측정): 실시간 판 MG -- (1) L000 10 프레임 대조 (셀 1·1.5·2 x MG 2x5·2x10, torch MG 판 오차와 비교), (2) 단위 시험 (블록·낙하·ficus) MG 2x5·2x10.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M
GL=(${AN_GPUS:-4 5 6}); j=0
for c in 1 1.5 2; do for b in "2 5" "2 10"; do set -- $b
  V=c$(echo $c | tr . p)_m$1c$2; O=$R/rtmg1/$V; mkdir -p $O
  ( CUDA_VISIBLE_DEVICES=${GL[$((j % ${#GL[@]}))]} python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --newton $1 --cg $2 --mg_levels 5 --prof 8 \
      < /dev/null > $O/log.txt 2>&1 ) & j=$((j + 1)); sleep 2
done; done; wait
for d in $R/rtmg1/*/; do echo "$(basename $d) $(grep -a '롤아웃\|rror' $d/log.txt | tail -1 | cut -c1-170)"; done > $M/rtmg1.out
rclone copy $R/rtmg1 r2:storage/result/anchorflow/lbours/fem_an/rtmg1 -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
AN_GPUS="${GL[*]}" AN_VS="m2c5:--newton_2_--cg_5_--mg_levels_5 m2c10:--newton_2_--cg_10_--mg_levels_5" bash exe/fem_an_rt_unit.sh ru2
