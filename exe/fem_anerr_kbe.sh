#!/bin/bash
# 절 AN.err K 축 묶음 (비측정, 80 프레임): 중점법은 접촉에서 K 8 이상 발산 (진단) -> K 축은 후퇴 오일러로. 낙하 블록 먼저, 그다음 lego 8 궤적.
# 예산: K 1·2·4 MG 6x30, K 8·32 MG 4x20, K 267 야코비 2x10 (CFL 수 0.06), 모두 점별 선탐색 + --en
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta
L=/dev/shm/dkta/anerr_kbe.txt; : > $L
S="--pc mg --newton 6 --cg 30 --ls_diff --en"; M="--pc mg --newton 4 --cg 20 --ls_diff --en"; J="--pc jacobi --newton 2 --cg 10 --ls_diff --en"
cat >> $L <<EOT
g_be_K2 drop1 --K 2 --integ be $S
g_be_K4 drop1 --K 4 --integ be $S
g_be_K8 drop1 --K 8 --integ be $M
g_be_K32 drop1 --K 32 --integ be $M
g_be_K267 drop1 --K 267 --integ be $J
g_mid_K2 drop1 --K 2 --integ mid $S
g_mid_K4 drop1 --K 4 --integ mid $S
p_be_K8 drop1 --K 8 --integ be --quad part $M
p_be_K267 drop1 --K 267 --integ be --quad part $J
px_be_K267 drop1 --K 267 --integ be --quad part --fl_src part --fl_npd 0.005 $J
gx_be_K267 drop1 --K 267 --integ be --fl_src part --fl_npd 0.005 $J
EOT
for t in 0001 0003 0016 0017 0018 0005 0013 L000; do
  echo "g_be_K2 $t --K 2 --integ be $S" >> $L
  echo "g_be_K8 $t --K 8 --integ be $M" >> $L
  echo "g_be_K267 $t --K 267 --integ be $J" >> $L
  echo "g_be_K4 $t --K 4 --integ be $S" >> $L
done
for t in 0001 0003 0016 0017 0018 0005 0013 L000; do
  echo "g_mid_K2 $t --K 2 --integ mid $S" >> $L
  echo "p_be_K8 $t --K 8 --integ be --quad part $M" >> $L
  echo "g_mid_K4 $t --K 4 --integ mid $S" >> $L
done
wc -l $L
ERR_PAR=6 bash exe/fem_anerr_pool.sh kbe $L
