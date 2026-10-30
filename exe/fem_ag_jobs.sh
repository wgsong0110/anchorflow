#!/bin/bash
# 절 AG N 스윕 작업 목록 (비측정): 줄 = <EX_ROOT> <태그> <fem_ex_sim 인자...>.  AG_SWEEP="hotdog:128,256 wolf:..." AG_R (반복, 기본 3)
cd "$(dirname "$0")/.." || exit 1
source exe/fem_ag_scene.sh
for SN in ${AG_SWEEP:?}; do
  S=${SN%%:*}; NL=${SN#*:}; ag_scene $S || exit 1
  for r in $(seq 1 ${AG_R:-3}); do for N in ${NL//,/ }; do
    echo "fem_ag_$S ag_${S}_n${N}_r$r $SARGS $REC --no_dump --N $N"
  done; done
done
