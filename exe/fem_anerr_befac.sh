#!/bin/bash
# 절 AN.err 후퇴 오일러 K 1 위 바닥·적분점 요인 (비측정 80 프레임; 중점 계열은 국소 폭주로 판정 불가): 낙하 + lego 8
# 판: b_flx (입자 판정 바닥 + 비관통 0.005 = 명시적 판 정의), b_flit (고착 상태 뉴턴마다), b_nvo (이탈 해제 끔), pb (입자 적분점)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_befac.txt; : > $L
S="--K 1 --integ be --pc mg --newton 6 --cg 30 --ls_diff --en"
for t in drop1 0001 0003 0016 0017 0018 0005 0013 L000; do
  echo "b_flx_K1 $t --fl_src part --fl_npd 0.005 $S" >> $L
  echo "pb_be_K1 $t --quad part $S" >> $L
  echo "b_flit_K1 $t --fl_iter $S" >> $L
done
echo "b_nvo_K1 drop1 --no_vfl_out $S" >> $L
ERR_PAR=6 bash exe/fem_anerr_pool.sh befac $L
