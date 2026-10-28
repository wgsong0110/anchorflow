#!/bin/bash
# 절 N 3 차 (비측정, 카드별 병렬): (A) 정답 민감도 남은 셋 (warp 캐시 경합으로 실패 -- 한 카드 순차) (B) 관성 보정 항 뺀 판 (--near_diag 0,1,1; 정의 변경 후보 진단) 80 프레임
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
LG="--mode lag --opt --near32 --mass tet --no_dump --near_diag 0,1,1"
export EX_NOWAIT=1 EX_ROOT=fem_n_lego
( gt g_base_r2 1; gt g_ng71 1 --n_grid 71; gt g_pic_nsub96 1 --rpic -1 --nsub 96; gt g_nsub96_fz049 1 --nsub 96 --floor_z 0.49 ) &
( $R lg_d011_n128 2 $LG --N 128 ) &
( $R lg_d011_n96 3 $LG --N 96 --dump_pred ) &
( $R lg_d011_n64 4 $LG --N 64 ) &
( $R b_apic_d011_n96 5 $B --N 96 --apic --no_dump --near_diag 0,1,1 ) &
( $R b_pic_d011_n96 6 $B --N 96 --no_dump --near_diag 0,1,1 ) &
( $R s_tet_d011_n96 0 --mode sub --N 96 --mass tet --opt --near32 --no_sort --no_dump --near_diag 0,1,1 ) &
wait
O=$W/lbours/fem_n/n3_summary.txt; : > $O
for d in $GO/g_*; do echo "== gt $(basename $d) $(tail -1 $d/log.txt)" >> $O; done
for d in $W/lbours/fem_n_lego/*d011*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; done
cat $O; rclone copy $W/lbours/fem_n r2:storage/result/anchorflow/lbours/fem_n -q
echo N3_DONE
