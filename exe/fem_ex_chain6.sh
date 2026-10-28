#!/bin/bash
# 절 L 6 차: 입자 정렬 + 워프 구간 합 P2G (sub 융합 판) 확인·반복 (5 차 연쇄 끝난 뒤)
cd "$(dirname "$0")/.." || exit 1
while ! grep -q "CHAIN5_DONE" /home/dkta/work/lbours/fem_ex_chain5.out 2>/dev/null; do sleep 30; done
R="bash exe/fem_ex_run.sh"; TM="--mass tet --opt --near32"
$R s6_prof_sort_n128 0 --mode sub --N 128 $TM --no_graph --tprof 3 --frames 5 --no_dump
$R s6_sort_n128_p1 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s6_sort_n128_p2 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s6_sort_n96_p1 0 --mode sub --N 96 $TM --no_dump --dump_pred
$R s6_sort_n96_p2 0 --mode sub --N 96 $TM --no_dump --dump_pred
$R s6_fu_n96_p2 0 --mode sub --N 96 $TM --no_sort --no_dump --dump_pred
echo CHAIN6_DONE
