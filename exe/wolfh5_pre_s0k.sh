#!/bin/bash
# 절 CM70 연기: 정답 입자 대조 스크립트를 2 프레임만 (CPU). 인자: <출력> <R2>
PY=/home/dkta/.conda/envs/af/bin/python; export PYTHONUTF8=1 OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES="" W5_FMAX=2 $PY exe/wolfh5_gt.py /dev/shm/dkta/ag_data/wolf.pt /home/dkta/work/topo/pg_wolf_sand $1/w5gt_k.json > $1/w5gt_k.log 2>&1
echo "rc $?"; tail -3 $1/w5gt_k.log; ls -la $1/w5gt_k.json
rclone copy $1 $2 -q --include "w5gt_k.*"
