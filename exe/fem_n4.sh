#!/bin/bash
# 절 N 4 차 (비측정, 카드별 병렬): lag 판 아핀 전달 (--affine) -- 원래 바닥 식 / 관성 보정 항 뺀 판 (정의 변경 후보, 진단), ficus 같은 설정
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
sleep 30
while pgrep -f "exe/fem_bs_sweep|exe/fem_ex_run.sh|exe/fem_bs_final|exe/gt_sens.py" > /dev/null; do echo "[대기] 다른 실행 $(date +%T)"; sleep 60; done
R="bash exe/fem_ex_run.sh"
LG="--mode lag --opt --near32 --mass tet --no_dump --affine"
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_NOWAIT=1 EX_ROOT=fem_n_lego
( $R lg_aff_n128 0 $LG --N 128 ) &
( $R lg_aff_n96 1 $LG --N 96 --lagdiag ) &
( $R lg_aff_d011_n96 2 $LG --N 96 --near_diag 0,1,1 --dump_pred ) &
( $R lg_aff_d011_n128 3 $LG --N 128 --near_diag 0,1,1 ) &
( $R lg_aff_d011_n64 4 $LG --N 64 --near_diag 0,1,1 ) &
( EX_ROOT=fem_n_ficus $R f_aff_n768 5 $LG --N 768 $PHYS ) &
( EX_ROOT=fem_n_ficus $R f_aff_n640 6 $LG --N 640 $PHYS ) &
wait
O=$W/lbours/fem_n/n4_summary.txt; : > $O
for d in $W/lbours/fem_n_lego/lg_aff* $W/lbours/fem_n_ficus/f_aff*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; done
cat $O; rclone copy $W/lbours/fem_n r2:storage/result/anchorflow/lbours/fem_n -q
echo N4_DONE
