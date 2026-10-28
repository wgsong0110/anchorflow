#!/bin/bash
# 절 O 2 차 (비측정 병렬, 진단): 속도 바닥 lag 판 발산 원인 -- --lagdiag (서브스텝별 최대 |v_n|, 프레임 끝 σ(F̄)·σ(A F̄)·입자 F σ, 첫 비유한 서브스텝)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
while pgrep -f "fem_o1.sh" > /dev/null; do echo "[대기] 1 차 $(date +%T)"; sleep 30; done
R="bash exe/fem_ex_run.sh"
LG="--mode lag --opt --mass tet --no_dump --lagdiag"
export EX_NOWAIT=1 EX_ROOT=fem_o_lego
( $R o_dg_fr_n96 1 $LG --floor vel --N 96 --frames 18 ) &
( $R o_dg_fr_n192 2 $LG --floor vel --N 192 --frames 28 ) &
( $R o_dg_fu_n96 3 $LG --floor vel --vfl_mode full --N 96 --frames 18 ) &
( $R o_dg_nofl_n96 4 $LG --floor none --N 96 --frames 30 ) &
wait
O=$W/lbours/fem_o_lego/o2_summary.txt; : > $O
for d in $W/lbours/fem_o_lego/o_dg_*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-200) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-90)" >> $O; grep -a "lag 진단" $d/v_fem.log | tail -6 | cut -c1-700 >> $O; done
cat $O; rclone copy $W/lbours/fem_o_lego r2:storage/result/anchorflow/lbours/fem_o_lego -q
echo O2_DONE
