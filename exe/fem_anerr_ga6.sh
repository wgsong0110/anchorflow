#!/bin/bash
# 절 AN.err3 2 단계 다시 (lego 9 궤적, 접촉 처치 --ga_areset 포함; ga3 은 처치 없이 ρ 0.7·0.9 K 1 이 0001 에서 발산해 중단). generalized-α ρ 0.5·0.7·0.9 x K 1·2, HHT ρ 0.7 x K 1·2, ga ρ 0.7 + 관성 질량 하한 x K 1·2 (모두 --y0 uk), 기준 후퇴 K 1·2 의 0015. 카드 3·5, R2 lbours/fem_anerr/ga6
cd "$(dirname "$0")/.." || exit 1

mkdir -p /dev/shm/dkta
S="--pc mg --newton 6 --cg 30 --ls_diff --en --y0 uk --ga_areset"; FL="--mlow 0.1 --mfl_n 267"
L=/dev/shm/dkta/anerr_ga6.txt; : > $L
for k in 2 1; do for t in 0001 0003 0015 0016 0017 0018 0005 0013 L000; do
  for r in 0.5 0.7 0.9; do echo "ga${r#0.}_K$k $t --K $k --integ ga --rho $r $S" >> $L; done
  echo "hht7_K$k $t --K $k --integ hht --rho 0.7 $S" >> $L
  echo "ga7m_K$k $t --K $k --integ ga --rho 0.7 $FL $S" >> $L
done; done
wc -l $L
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga6 $L
