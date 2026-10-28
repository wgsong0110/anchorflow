#!/bin/bash
# 절 N 5 차 (비측정, 병렬): APIC/아핀 + 바닥 후보 C=0 (--near_c0)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
while pgrep -f "exe/fem_bs_sweep|exe/fem_ex_run.sh|exe/fem_bs_final|exe/gt_sens.py" > /dev/null; do echo "[대기] 다른 실행 $(date +%T)"; sleep 60; done
R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt --no_dump"
LG="--mode lag --opt --near32 --mass tet --no_dump --affine --near_c0"
export EX_NOWAIT=1 EX_ROOT=fem_n_lego
( $R b_apic_c0_n96 0 $B --N 96 --apic --near_c0 --dump_pred ) &
( $R b_apic_c0_n128 1 $B --N 128 --apic --near_c0 ) &
( $R b_apic_c0_d011_n96 2 $B --N 96 --apic --near_c0 --near_diag 0,1,1 ) &
( $R lg_aff_c0_n96 3 $LG --N 96 ) &
( $R lg_aff_c0_n128 4 $LG --N 128 ) &
( $R lg_d011_n128_80 5 --mode lag --opt --near32 --mass tet --no_dump --near_diag 0,1,1 --N 128 ) &
( $R lg_d011_n192 6 --mode lag --opt --near32 --mass tet --no_dump --near_diag 0,1,1 --N 192 ) &
wait
O=$W/lbours/fem_n/n5_summary.txt; : > $O
for d in $W/lbours/fem_n_lego/*c0* $W/lbours/fem_n_lego/lg_d011_n128_80 $W/lbours/fem_n_lego/lg_d011_n192; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; done
cat $O; rclone copy $W/lbours/fem_n r2:storage/result/anchorflow/lbours/fem_n -q
echo N5_DONE
