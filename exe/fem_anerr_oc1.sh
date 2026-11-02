#!/bin/bash
# 절 AN.err4 본 묶음: --gq_oct 0.1 (칸 가우스 점 가중 = 0.9 x 팔분 칸 입자 부피 + 0.1 x 칸 합/8) x {HHT ρ 0.7 K 2, 후퇴 오일러 K 2} x {ficus, 블록 신축·전단 40 프레임, lego 9}; lego·ficus 는 2 회차 (oc1b). 카드 3·5, R2 lbours/fem_anerr/oc1·oc1b
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta
H="--K 2 --integ hht --rho 0.7 --y0 uk --ga_areset --pc mg --newton 6 --cg 30 --ls_diff --en --gq_oct 0.1"; E="--K 2 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en --gq_oct 0.1"
L=/dev/shm/dkta/anerr_oc1.txt; : > $L; L2=/dev/shm/dkta/anerr_oc1b.txt; : > $L2
for v in "hht7:$H" "be:$E"; do n=${v%%:*}; A=${v#*:}
  echo "${n}_K2 ficus $A" >> $L; echo "${n}_K2 ficus $A" >> $L2
  echo "${n}_K2 stretch $A --frames 40" >> $L; echo "${n}_K2 shear $A --frames 40" >> $L
  for t in 0001 0003 0015 0016 0017 0018 0005 0013 L000; do echo "${n}_K2 $t $A" >> $L; echo "${n}_K2 $t $A" >> $L2; done
done
wc -l $L $L2
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh oc1 $L
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh oc1b $L2
