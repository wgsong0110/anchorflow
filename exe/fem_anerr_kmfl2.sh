#!/bin/bash
# 절 AN.err 실시간 예산 판 K 1·2 중점 + 관성 질량 하한 (명시적 N 267 과 같은 하한, β = (K/267)²), 비측정 80 프레임: 낙하 + lego 8
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_kmfl2.txt; : > $L
S="--pc mg --newton 6 --cg 30 --ls_diff --en"
for t in drop1 0001 0003 0016 0017 0018 0005 0013 L000; do
  echo "gm_mid_K1 $t --K 1 --integ mid --mfl 1.40274e-05 $S" >> $L
  echo "gm_mid_K2 $t --K 2 --integ mid --mfl 5.61097e-05 $S" >> $L
done
ERR_PAR=6 bash exe/fem_anerr_pool.sh kmfl $L
