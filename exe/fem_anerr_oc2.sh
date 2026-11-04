#!/bin/bash
# 절 AN.err4 κ 스윕: HHT ρ 0.7 K 2 x --gq_oct 0.25·0.5 x {ficus, 블록 신축·전단 40 프레임, lego 9} (κ 0.1 은 ficus 를 고치나 lego 를 나쁘게 함; 원본 = κ 1). 카드 3·5, R2 lbours/fem_anerr/oc2
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_oc2.txt; : > $L
H="--K 2 --integ hht --rho 0.7 --y0 uk --ga_areset --pc mg --newton 6 --cg 30 --ls_diff --en"
for k in 0.5 0.25; do n=hk${k#0.}
  echo "${n}_K2 ficus $H --gq_oct $k" >> $L
  for t in 0001 0003 0015 0016 0017 0018 0005 0013 L000; do echo "${n}_K2 $t $H --gq_oct $k" >> $L; done
  echo "${n}_K2 stretch $H --gq_oct $k --frames 40" >> $L; echo "${n}_K2 shear $H --gq_oct $k --frames 40" >> $L
done
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh oc2 $L
