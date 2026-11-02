#!/bin/bash
# 절 AN.err3 2 회차 (ga6 이 끝나면): 후퇴 오일러 K 2 와 HHT ρ 0.7 K 2 (+ 접촉 처치) x lego 9 (카드 3·5, R2 lbours/fem_anerr/ga6b)
cd "$(dirname "$0")/.." || exit 1
while [ ! -f /dev/shm/dkta/lbours/fem_anerr/ga6/END ]; do sleep 60; done
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ga6b.txt; : > $L
for t in 0001 0003 0015 0016 0017 0018 0005 0013 L000; do
  echo "hht7_K2 $t --K 2 --integ hht --rho 0.7 --pc mg --newton 6 --cg 30 --ls_diff --en --y0 uk --ga_areset" >> $L
  echo "be_K2 $t --K 2 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en" >> $L
done
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga6b $L
