#!/bin/bash
# 절 AN.err3 2 단계 (lego 9 궤적): 1 단계 (ga2) 가 끝나면 시작. generalized-α ρ 0.5·0.7·0.9 x K 1·2, HHT ρ 0.7 x K 1·2, ga ρ 0.7 + 관성 질량 하한 x K 1·2 (모두 --y0 uk), 기준 후퇴 K 1·2 의 0015. 카드 3·5, R2 lbours/fem_anerr/ga3
cd "$(dirname "$0")/.." || exit 1
while [ ! -f /dev/shm/dkta/lbours/fem_anerr/ga2/END ]; do sleep 60; done
mkdir -p /dev/shm/dkta
S="--pc mg --newton 6 --cg 30 --ls_diff --en --y0 uk"; FL="--mlow 0.1 --mfl_n 267"
L=/dev/shm/dkta/anerr_ga3.txt; : > $L
echo "be_K1 0015 --K 1 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en" >> $L
echo "be_K2 0015 --K 2 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en" >> $L
for k in 1 2; do for t in 0001 0003 0015 0016 0017 0018 0005 0013 L000; do
  for r in 0.5 0.7 0.9; do echo "ga${r#0.}_K$k $t --K $k --integ ga --rho $r $S" >> $L; done
  echo "hht7_K$k $t --K $k --integ hht --rho 0.7 $S" >> $L
  echo "ga7m_K$k $t --K $k --integ ga --rho 0.7 $FL $S" >> $L
done; done
wc -l $L
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga3 $L
