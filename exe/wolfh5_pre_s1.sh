#!/bin/bash
# 절 CM70 할 일 1: 정답 입자 대조 (PG h5 원본 f32, f1~f30; CPU, 판과 겹쳐 뒤에서). 인자: <출력> <R2>
PY=/home/dkta/.conda/envs/af/bin/python; export PYTHONUTF8=1 OMP_NUM_THREADS=8
( CUDA_VISIBLE_DEVICES="" $PY exe/wolfh5_gt.py /dev/shm/dkta/ag_data/wolf.pt /home/dkta/work/topo/pg_wolf_sand $1/w5gt.json > $1/w5gt.log 2>&1; echo "rc $?" >> $1/w5gt.log; rclone copy $1 $2 -q --include "w5gt.*" ) < /dev/null > /dev/null 2>&1 &
echo "정답 대조 뒤에서 pid $!"
