#!/bin/bash
# 절 N.1~N.2 (사용자 10-28 목표: lego·ficus 60 fps + 오차 < 10 %, 명시적 통일) 1 차 진단 -- 모두 오차·진단만 (비측정, 카드별 병렬):
#  (A) 정답 MPM 민감도 (exe/gt_sens.py, PG 공식 솔버 그대로 조건 하나씩): 재현 2 회, 서브스텝 96, PIC, PIC+96, 격자 141·71, 바닥 높이 ±반 칸, 미끄럼 바닥
#  (B) 우리 B-스플라인 판 APIC (--apic) N 96·128·192, 격자 절반 (node_h/2) PIC·APIC
#  (C) lag 판 진단 (--lagdiag): 바닥 있음 N128 (발산 25), 바닥 없음 N 96·128·256 80 프레임
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
# 다른 측정 연쇄 (fem_bs_*/fem_ex_* 스크립트) 가 돌고 있으면 끝날 때까지 기다린다 (그쪽 측정을 흐리지 않게)
while pgrep -f "exe/fem_bs_sweep|exe/fem_ex_run.sh|exe/fem_bs_final" > /dev/null; do echo "[대기] 다른 측정 연쇄 $(date +%T)"; sleep 60; done
GO=$W/lbours/fem_n/gt; mkdir -p $GO
gt(){ # 태그 GPU 인자...
  local tg=$1 g=$2; shift 2; local o=$GO/$tg
  [ -f $o/err.json ] && { echo "[건너뜀] $tg"; return; }
  mkdir -p $o; echo "커밋 $C $*" > $o/cmd.txt
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$g python -u $AF/exe/gt_sens.py --out $o "$@" < /dev/null > $o/log.txt 2>&1)
  tail -1 $o/log.txt; rclone copy $o r2:storage/result/anchorflow/lbours/fem_n/gt/$tg -q; }
R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt"
export EX_NOWAIT=1 EX_ROOT=fem_n_lego
( gt g_base_r1 1; gt g_nsub96 1 --nsub 96; gt g_fz049 1 --floor_z 0.49 ) &
( gt g_base_r2 2; gt g_pic 2 --rpic -1; gt g_fz047 2 --floor_z 0.47 ) &
( gt g_pic_nsub96 3 --rpic -1 --nsub 96; gt g_ng141 3 --n_grid 141 ) &
( gt g_ng71 4 --n_grid 71; gt g_slip 4 --surface slip; gt g_nsub48 4 --nsub 48 ) &
( $R b_apic_n96 5 $B --N 96 --apic --no_dump --dump_pred; $R b_apic_n128 5 $B --N 128 --apic --no_dump; $R b_apic_n192 5 $B --N 192 --apic --no_dump ) &
( $R b_half_n192 6 $B --N 192 --node_h 0.014142 --no_dump; $R b_half_apic_n192 6 $B --N 192 --node_h 0.014142 --apic --no_dump ) &
( $R lg_diag_n128 0 --mode lag --N 128 --opt --near32 --mass tet --no_dump --lagdiag; $R lg_nofl_n96 0 --mode lag --N 96 --opt --near32 --mass tet --no_dump --floor none;
  $R lg_nofl_n128 0 --mode lag --N 128 --opt --near32 --mass tet --no_dump --floor none --lagdiag; $R lg_nofl_n256 0 --mode lag --N 256 --opt --near32 --mass tet --no_dump --floor none ) &
wait
O=$W/lbours/fem_n/n1_summary.txt; : > $O
for d in $GO/g_*; do echo "== gt $(basename $d) $(tail -1 $d/log.txt)" >> $O; done
for d in $W/lbours/fem_n_lego/b_* $W/lbours/fem_n_lego/lg_*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; done
cat $O; rclone copy $W/lbours/fem_n r2:storage/result/anchorflow/lbours/fem_n -q
echo N1_DONE
