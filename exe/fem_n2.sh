#!/bin/bash
# 절 N 2 차 (비측정, 카드별 병렬): (A) 정답 MPM 민감도 다시 (EXT 고침) (B) lag 발산 원인 -- 바닥 후보 힘 항별 배율 진단 (C) APIC 바닥 없음 (구현 확인)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
while pgrep -f "exe/fem_bs_sweep|exe/fem_ex_run.sh|exe/fem_bs_final" > /dev/null; do echo "[대기] 다른 측정 연쇄 $(date +%T)"; sleep 60; done
GO=$W/lbours/fem_n/gt; mkdir -p $GO
gt(){ local tg=$1 g=$2; shift 2; local o=$GO/$tg
  [ -f $o/err.json ] && { echo "[건너뜀] $tg"; return; }
  mkdir -p $o; echo "커밋 $C $*" > $o/cmd.txt
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$g python -u $AF/exe/gt_sens.py --out $o "$@" < /dev/null > $o/log.txt 2>&1)
  tail -1 $o/log.txt; rclone copy $o r2:storage/result/anchorflow/lbours/fem_n/gt/$tg -q; }
R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt"
LG="--mode lag --N 128 --opt --near32 --mass tet --no_dump --lagdiag --frames 30"
export EX_NOWAIT=1 EX_ROOT=fem_n_lego
( gt g_base_r1 1; gt g_nsub96 1 --nsub 96; gt g_fz049 1 --floor_z 0.49 ) &
( gt g_base_r2 2; gt g_pic 2 --rpic -1; gt g_fz047 2 --floor_z 0.47 ) &
( gt g_pic_nsub96 3 --rpic -1 --nsub 96; gt g_ng141 3 --n_grid 141 ) &
( gt g_ng71 4 --n_grid 71; gt g_slip 4 --surface slip; gt g_nsub48 4 --nsub 48 ) &
( $R lg_d011_n128 5 $LG --near_diag 0,1,1; $R lg_d100_n128 5 $LG --near_diag 1,0,0 ) &
( $R lg_d000_n128 6 $LG --near_diag 0,0,0; $R b_apic_nofl_n96 6 $B --N 96 --apic --no_dump --floor none --frames 20 ) &
( $R b_pic_nofl_n96 0 $B --N 96 --no_dump --floor none --frames 20; $R lg_d110_n128 0 $LG --near_diag 1,1,0 ) &
wait
O=$W/lbours/fem_n/n2_summary.txt; : > $O
for d in $GO/g_*; do echo "== gt $(basename $d) $(tail -1 $d/log.txt)" >> $O; done
for d in $W/lbours/fem_n_lego/lg_d* $W/lbours/fem_n_lego/b_*nofl*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; done
cat $O; rclone copy $W/lbours/fem_n r2:storage/result/anchorflow/lbours/fem_n -q
echo N2_DONE
