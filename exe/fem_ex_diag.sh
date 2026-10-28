#!/bin/bash
# 절 L 진단: 첫 프레임 노드 국소 안정 걸음 분포 + 서브스텝 |v| (기준 lag), 바닥 있음/없음, N 여러 개 (2 프레임)
cd "$(dirname "$0")/.." || exit 1
for NN in 64 1024; do
  bash exe/fem_ex_run.sh diag_lag_n$NN 0 --mode lag --N $NN --frames 2 --no_dump --warm 0 --diag
  bash exe/fem_ex_run.sh diag_lag_nofl_n$NN 0 --mode lag --N $NN --frames 2 --no_dump --warm 0 --diag --floor none
done
echo DIAG_DONE
