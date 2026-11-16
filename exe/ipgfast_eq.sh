#!/bin/bash
# [final_fix_mpmbl] i-PG wolf N100 동일성: 원 구현 두 번 (비결정성 기준 A·B) + 빠른 구현 (AF_IPG_FAST=1), 같은 카드 동시, F 프레임 덤프 -> exe/ipgfast_cmp.py
#   사용: bash exe/ipgfast_eq.sh [F=1]
set -u
F=${1:-1}; N=100
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=4 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1 AF_FD_EPS=1e-5
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/ipg_wolf/ipgfast; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipgfast/eq$F; mkdir -p $O
ARGS="--method ipg --nsub $N --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames $F --pg_cfg --time --dump"
AF_IPG_FAST=0 python -u exe/pareto_mpmbl.py $ARGS --out $O/origA > $O/origA.log 2>&1 &
AF_IPG_FAST=0 python -u exe/pareto_mpmbl.py $ARGS --out $O/origB > $O/origB.log 2>&1 &
AF_IPG_FAST=1 python -u exe/pareto_mpmbl.py $ARGS --out $O/fast > $O/fast.log 2>&1 &
wait
python exe/ipgfast_cmp.py --ref $O/origA --rep $O/origB --new $O/fast > $O/cmp.txt 2>&1; cat $O/cmp.txt
rclone copy $O $RB/eq$F -q --exclude "*/pred.pt"
echo "[끝] $(date)"
