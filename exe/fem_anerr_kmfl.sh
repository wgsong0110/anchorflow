#!/bin/bash
# 절 AN.err 중점 + 관성 질량 하한 (명시적 판 N 267 과 같은 하한: β = (K/267)², 비측정 80 프레임): 낙하 K 8·32·267, lego 0001·0017 K 267
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_kmfl.txt
M="--pc mg --newton 4 --cg 20 --ls_diff --en"; J="--pc jacobi --newton 2 --cg 10 --ls_diff --en"
cat > $L <<EOT
gm_mid_K8 drop1 --K 8 --integ mid --mfl 0.000897758 $M
gm_mid_K32 drop1 --K 32 --integ mid --mfl 0.0143640 $M
gm_mid_K267 drop1 --K 267 --integ mid --mfl 1 $J
gm_mid_K267 0001 --K 267 --integ mid --mfl 1 $J
gm_mid_K267 0017 --K 267 --integ mid --mfl 1 $J
EOT
ERR_PAR=6 bash exe/fem_anerr_pool.sh kmfl $L
