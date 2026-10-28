#!/bin/bash
# 절 L 4 차: 스윕 3 끝난 뒤 sub 융합 판 (lego, 사면체 등분 질량) 계측·반복·경계 -> ficus 스윕 (같은 커밋)
cd "$(dirname "$0")/.." || exit 1
while ! grep -q "_DONE" /home/dkta/work/lbours/fem_ex_sweep3.out 2>/dev/null; do sleep 30; done
R="bash exe/fem_ex_run.sh"; TM="--mass tet --opt --near32"
$R s4_prof_fu_n128 0 --mode sub --N 128 $TM --no_graph --tprof 3 --frames 5 --no_dump
$R s4_fu_n128_p1 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s4_fu_n128_p2 0 --mode sub --N 128 $TM --no_dump --dump_pred
for NN in 96 112; do $R s4_fu_n$NN 0 --mode sub --N $NN $TM --no_dump; done
echo CHAIN4_LEGO_DONE
bash exe/fem_ex_ficus_sweep.sh
echo CHAIN4_DONE
