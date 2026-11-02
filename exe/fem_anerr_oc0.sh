#!/bin/bash
# 절 AN.err4 1 차 시험: 칸 가우스 점 가중 = 팔분 칸 실제 입자 부피 (--gq_oct κ) -- κ 고르기 (카드 3·5, R2 lbours/fem_anerr/oc0)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_oc0.txt
H="--K 2 --integ hht --rho 0.7 --y0 uk --ga_areset --pc mg --newton 6 --cg 30 --ls_diff --en"; E="--K 2 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en"
cat > $L <<EOT
hk0 ficus $H --gq_oct 0
hk10 ficus $H --gq_oct 0.1
hk25 ficus $H --gq_oct 0.25
bk0 ficus $E --gq_oct 0
hk0 stretch $H --gq_oct 0 --frames 40
hk10 stretch $H --gq_oct 0.1 --frames 40
hk0 0001 $H --gq_oct 0
EOT
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh oc0 $L
