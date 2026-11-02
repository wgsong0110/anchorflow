#!/bin/bash
# 절 AN.err2 회차 차이 (조정자 11-02: 이 판은 원자 합 순서로 같은 명령도 회차마다 갈린다 -> 핵심 판 두 번째 회차): 카드 3·5 만, R2 lbours/fem_anerr/r2b
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_rep.txt; : > $L
S="--pc mg --newton 6 --cg 30 --ls_diff --en"; M="--pc mg --newton 4 --cg 20 --ls_diff --en"; FL="--mlow 0.1 --mfl_n 267"
for t in drop1 0001 0003 0016 0018 0005 0013 L000; do
  echo "t55_K1 $t --K 1 --integ theta --theta 0.55 $FL $S" >> $L
  echo "t55ad8_K1 $t --K 1 --kad 8 --integ theta --theta 0.55 $FL $M" >> $L
  echo "t55ad4_K1 $t --K 1 --kad 4 --integ theta --theta 0.55 $FL $S" >> $L
  echo "t55av_K1 $t --K 1 --fl_avg 8 --integ theta --theta 0.55 $FL $S" >> $L
  echo "g_be_K1 $t --K 1 --integ be $S" >> $L
done
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh r2b $L
