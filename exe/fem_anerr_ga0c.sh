#!/bin/bash
# 절 AN.err3 첫 시험 3: 뉴턴 시작점 걸음 시작 위치 (--y0 uk) -- 블록·낙하 (짧게, 카드 3·5)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ga0c.txt
S="--pc mg --newton 6 --cg 30 --ls_diff --en --y0 uk"
cat > $L <<EOT
u_ga7 stretch --K 1 --integ ga --rho 0.7 $S --frames 4
u_ga7 drop1 --K 1 --integ ga --rho 0.7 $S --frames 6
u_hht7 stretch --K 1 --integ hht --rho 0.7 $S --frames 4
u_ga5 drop1 --K 1 --integ ga --rho 0.5 $S --frames 6
EOT
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga0c $L
