#!/bin/bash
# 절 AN.err3 첫 시험 2: 관성 질량 하한 (--mlow 0.1 --mfl_n 267) 으로 블록·낙하의 걸음 0·튐이 풀리는지 (짧게, 카드 3·5)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ga0b.txt
S="--pc mg --newton 6 --cg 30 --ls_diff --en"; FL="--mlow 0.1 --mfl_n 267"
cat > $L <<EOT
f_hht7 stretch --K 1 --integ hht --rho 0.7 $FL $S --frames 3
f_ga7 drop1 --K 1 --integ ga --rho 0.7 $FL $S --frames 5
f_ga7 stretch --K 1 --integ ga --rho 0.7 $FL $S --frames 3
f_be stretch --K 1 --integ be $S --frames 3
EOT
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga0b $L
