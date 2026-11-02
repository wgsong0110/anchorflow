#!/bin/bash
# 절 AN.err3 1 단계 (다시; generalized-α·HHT 는 --y0 uk): 단위 시험 ficus 주기, 블록 신축·전단 40 프레임, 낙하 반발 x 판 (카드 3·5, R2 lbours/fem_anerr/ga2)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta
S="--pc mg --newton 6 --cg 30 --ls_diff --en"
L=/dev/shm/dkta/anerr_ga2.txt; : > $L
for t in stretch shear drop1 ficus; do
  for k in 1 2; do
    for r in 0.5 0.7 0.9; do echo "ga${r#0.}_K$k $t --K $k --integ ga --rho $r --y0 uk $S" >> $L; done
    for r in 0.7 0.9; do echo "hht${r#0.}_K$k $t --K $k --integ hht --rho $r --y0 uk $S" >> $L; done
  done
  echo "be_K2 $t --K 2 --integ be $S" >> $L
  echo "mid_K2 $t --K 2 --integ mid $S" >> $L
done
sed -i 's/^\(.* \(stretch\|shear\) .*\)$/\1 --frames 40/' $L
wc -l $L
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga2 $L
