#!/bin/bash
# 절 AN.err 수렴 확인 (비측정): 걸음 끝 |g|/|g0| 를 예산별로 -- lego 0001 앞 4 프레임, 낙하 블록 앞 8 프레임 (접촉 포함)
cd "$(dirname "$0")/.." || exit 1
L=/tmp/dkta/anerr_conv.txt; : > $L
for v in "c_m3x20:--pc mg --newton 3 --cg 20" "c_m8x40:--pc mg --newton 8 --cg 40" "c_m8x40n:--pc mg --newton 8 --cg 40 --fast_ls 0" "c_j8x200:--pc jacobi --newton 8 --cg 200" "c_m15x80:--pc mg --newton 15 --cg 80"; do
  echo "${v%%:*}_K1 0001 --frames 4 --K 1 --integ mid ${v#*:}" | tr '_' ' ' | sed 's/ K1/_K1/; s/^c m/c_m/; s/^c j/c_j/' >> $L; done
echo "c_m8x40_K8 0001 --frames 4 --K 8 --integ mid --pc mg --newton 8 --cg 40" >> $L
echo "c_m8x40_K1 drop1 --frames 8 --K 1 --integ mid --pc mg --newton 8 --cg 40" >> $L
echo "c_m15x80_K1 drop1 --frames 8 --K 1 --integ mid --pc mg --newton 15 --cg 80" >> $L
echo "c_j3x20_K267 drop1 --frames 8 --K 267 --integ mid --pc jacobi --newton 3 --cg 20" >> $L
cat $L
ERR_PAR=3 bash exe/fem_anerr_pool.sh conv $L
