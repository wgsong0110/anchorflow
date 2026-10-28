#!/bin/bash
# 절 L 스윕 3 (사면체 등분 질량, lego L000): sub 경계, 바닥 없음 경계, lag 기준 판 대조, 결과 불변 (반복·기준 판·그래프 끔·바닥 후보 f64), 구간 계측
cd "$(dirname "$0")/.." || exit 1
WAIT_FOR=${WAIT_FOR:-/home/dkta/work/lbours/fem_ex_sweep2.out}
while ! grep -q "_DONE" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
R="bash exe/fem_ex_run.sh"; TM="--mass tet --opt --near32"
$R s3_prof_sub_n128 0 --mode sub --N 128 $TM --no_graph --tprof 3 --frames 5 --no_dump
for NN in 80 96 112; do $R s3_sub_tm_n$NN 0 --mode sub --N $NN $TM --no_dump; done
for NN in 48 64 80 96 128; do $R s3_sub_tm_nofl_n$NN 0 --mode sub --N $NN $TM --floor none --frames 20 --no_dump; done
$R s3_sub_tm_n128_p1 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s3_sub_tm_n128_p2 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s3_sub_tm_n128_nog 0 --mode sub --N 128 $TM --no_dump --dump_pred --no_graph
$R s3_sub_tm_n128_f64n 0 --mode sub --N 128 --mass tet --opt --no_dump --dump_pred
$R s3_sub_tm_n128_base 0 --mode sub --N 128 --mass tet --no_dump --dump_pred
$R s3_lag_tm_n128_base 0 --mode lag --N 128 --mass tet --no_dump
echo SWEEP3_DONE
