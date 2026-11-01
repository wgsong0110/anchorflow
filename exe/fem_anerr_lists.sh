#!/bin/bash
# 절 AN.err 할 일 목록 만들기: bash exe/fem_anerr_lists.sh <출력 디렉토리> "<작은 K 예산 인자>" "<큰 K 예산 인자>"
# drop.txt (낙하 블록, 요인마다), kgrid.txt (lego 8 x K 1·2·4·8·267, 가우스·중점), fac.txt (lego 8 x 요인), 모두 --en
O=$1; BS=$2; BB=$3; mkdir -p $O
T8="0001 0003 0016 0017 0018 0005 0013 L000"
declare -a VS=(
 "g_mid_K1:--K 1 --integ mid $BS" "g_mid_K2:--K 2 --integ mid $BS" "g_mid_K4:--K 4 --integ mid $BS" "g_mid_K8:--K 8 --integ mid $BS" "g_mid_K267:--K 267 --integ mid $BB"
 "g_be_K2:--K 2 --integ be $BS" "g_bdf2_K2:--K 2 --integ bdf2 $BS" "g_be_K8:--K 8 --integ be $BS" "g_bdf2_K8:--K 8 --integ bdf2 $BS"
 "p_mid_K2:--K 2 --integ mid --quad part $BS" "p_mid_K267:--K 267 --integ mid --quad part $BB"
 "g_flit_K2:--K 2 --integ mid --fl_iter $BS" "g_flx_K2:--K 2 --integ mid --fl_src part --fl_npd 0.005 $BS" "g_nvo_K2:--K 2 --integ mid --no_vfl_out $BS"
 "p_flx_K267:--K 267 --integ mid --quad part --fl_src part --fl_npd 0.005 $BB" )
: > $O/drop.txt; : > $O/kgrid.txt; : > $O/fac.txt
for e in "${VS[@]}"; do echo "${e%%:*} drop1 ${e#*:} --en" >> $O/drop.txt; done
for t in $T8; do for e in "${VS[@]:0:5}"; do echo "${e%%:*} $t ${e#*:} --en" >> $O/kgrid.txt; done; done
for t in $T8; do for e in "${VS[@]:5}"; do echo "${e%%:*} $t ${e#*:} --en" >> $O/fac.txt; done; done
wc -l $O/*.txt
