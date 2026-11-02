#!/bin/bash
# 절 AN.err3 접촉 처치 시험: --ga_areset (고착 노드 가속도 상태 0) -- 낙하 80 프레임 (카드 3·5, R2 lbours/fem_anerr/ga5)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ga5.txt
S="--pc mg --newton 6 --cg 30 --ls_diff --en --y0 uk --ga_areset"
cat > $L <<EOT
ar7_K1 drop1 --K 1 --integ ga --rho 0.7 $S
ar7_K2 drop1 --K 2 --integ ga --rho 0.7 $S
ar9_K1 drop1 --K 1 --integ ga --rho 0.9 $S
ar9_K2 drop1 --K 2 --integ ga --rho 0.9 $S
arh7_K2 drop1 --K 2 --integ hht --rho 0.7 $S
EOT
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga5 $L
