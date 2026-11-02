#!/bin/bash
# 절 AN.err3 공통 적분기 후보 (조정자 11-02: wolf 실시간 판 θ 0.6·K 3·관성 질량 하한 1e-3 x 중앙값, implicit-wolf 절 AO.16) -- 같은 표: 블록 신축·전단 40 프레임, ficus, 낙하, lego 9 (카드 3·5, R2 lbours/fem_anerr/ga4)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ga4.txt; : > $L
S="--K 3 --integ theta --theta 0.6 --mlow 0.001 --pc mg --newton 6 --cg 30 --ls_diff --en"
for t in stretch shear; do echo "w60_K3 $t $S --frames 40" >> $L; done
for t in ficus drop1 0001 0003 0015 0016 0017 0018 0005 0013 L000; do echo "w60_K3 $t $S" >> $L; done
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga4 $L
