#!/bin/bash
# 절 AN.err K 1 요인 묶음 (비측정, 80 프레임, 수렴 예산 MG 뉴턴 6 x PCG 30 + 점별 선탐색): 낙하 블록 먼저, 그다음 lego 8 궤적 (미달 5 + 만족 대조 2 + L000)
# 판: g_mid (기준: 가우스·중점), g_be, g_bdf2 (적분기), p_mid (입자 적분점), g_flit (고착 상태 뉴턴마다), g_flx (입자 판정 바닥 + 비관통 0.005 = 명시적 판 정의), g_nvo (이탈 속도 해제 끔)
cd "$(dirname "$0")/.." || exit 1
L=/tmp/dkta/anerr_k1fac.txt; : > $L
B="--K 1 --pc mg --newton 6 --cg 30 --ls_diff --en"
for t in drop1 0001 0003 0016 0017 0018 0005 0013 L000; do
  for v in "g_mid:--integ mid" "g_be:--integ be" "g_bdf2:--integ bdf2" "p_mid:--integ mid --quad part" "g_flit:--integ mid --fl_iter" "g_flx:--integ mid --fl_src part --fl_npd 0.005" "g_nvo:--integ mid --no_vfl_out"; do
    echo "${v%%:*}_K1 $t ${v#*:} $B" >> $L; done; done
ERR_PAR=6 bash exe/fem_anerr_pool.sh k1fac $L
