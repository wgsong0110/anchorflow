#!/bin/bash
# [final_fix_mpmbl] i-PG wolf 바닥 접촉 진단: N 판의 프레임 F 체크포인트 (R2) 에서 이어 프레임 F+1 만 AF_IPG_DIAG=1 로 (Newton 반복마다 잔차 위치·줄 탐색).
N=${1:-50}; F=${2:-8}
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1 AF_FD_EPS=1e-5 AF_IPG_FAST=1 AF_IPG_DIAG=1 ${DIAG_ENV:-}
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/ipg_wolf; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipgdiag/N${N}_f${F}${DIAG_TAG:-}; mkdir -p $O
[ -s $O/ckpt.pt ] || rclone copyto $RB/N${N}e5_fa/ckpt.pt $O/ckpt.pt -q
python -c "import torch;print('ckpt 프레임', torch.load('$O/ckpt.pt',weights_only=False,map_location='cpu')['frame'])"
python -u exe/pareto_mpmbl.py --method ipg --nsub $N --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames $((F+1)) --pg_cfg --out $O --resume > $O/diag.log 2>&1
rclone copyto $O/diag.log $RB/diag/floor_N${N}_f${F}${DIAG_TAG:-}.log -q
