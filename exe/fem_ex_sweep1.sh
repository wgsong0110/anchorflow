#!/bin/bash
# 절 L 서브스텝 N 스윕 1 (lego L000 80 프레임, 단독 GPU0 순차): lag/sub 최적화 판 (바닥 후보 float32), 바닥 없음 안정 경계 (20 프레임)
cd "$(dirname "$0")/.." || exit 1
for NN in 2048 3072 4096 6144 8192; do
  bash exe/fem_ex_run.sh s1_lag_n$NN 0 --mode lag --N $NN --opt --near32 --no_dump
done
for NN in 1024 2048 4096; do
  bash exe/fem_ex_run.sh s1_lag_nofl_n$NN 0 --mode lag --N $NN --opt --floor none --frames 20 --no_dump
done
for NN in 2048 4096 8192; do
  bash exe/fem_ex_run.sh s1_sub_n$NN 0 --mode sub --N $NN --opt --near32 --no_dump
done
echo SWEEP1_DONE
