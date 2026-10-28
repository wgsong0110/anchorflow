#!/bin/bash
# 절 L 5 차 (발산 판정 추가 커밋): 스윕 3 남은 것 + 융합 판 계측·반복·경계 + ficus 스윕
cd "$(dirname "$0")/.." || exit 1
R="bash exe/fem_ex_run.sh"; TM="--mass tet --opt --near32"
for NN in 80 96 128; do $R s5_sub_tm_nofl_n$NN 0 --mode sub --N $NN $TM --floor none --frames 20 --no_dump; done
$R s5_sub_tm_n80 0 --mode sub --N 80 $TM --no_dump
$R s4_prof_fu_n128 0 --mode sub --N 128 $TM --no_graph --tprof 3 --frames 5 --no_dump
$R s5_fu_n128_p1 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s5_fu_n128_p2 0 --mode sub --N 128 $TM --no_dump --dump_pred
$R s5_fu_n96_p1 0 --mode sub --N 96 $TM --no_dump --dump_pred
$R s5_nofu_n128_p 0 --mode sub --N 128 $TM --no_fuse --no_dump --dump_pred
$R s5_nofu_nog_n128_p 0 --mode sub --N 128 $TM --no_fuse --no_graph --no_dump --dump_pred
$R s5_nofu_f64n_n128_p 0 --mode sub --N 128 --mass tet --opt --no_fuse --no_dump --dump_pred
$R s5_base_n128_p 0 --mode sub --N 128 --mass tet --no_dump --dump_pred
$R s5_lag_base_n128 0 --mode lag --N 128 --mass tet --no_dump
echo CHAIN5_LEGO_DONE
bash exe/fem_ex_ficus_sweep.sh
echo CHAIN5_DONE
