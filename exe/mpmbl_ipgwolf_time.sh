#!/bin/bash
# [final_fix_mpmbl] i-PG wolf: k=20·10·5 (N 100·200·400) 은 첫 프레임 Newton 발산 (잔차 0.24 -> 302, GMRES 최소제곱 NaN), k=2 (N 1000) 만 진행 (서브스텝 5.8 s, 50 프레임 ~80 h).
#   -> 카드 단독으로 N 1000 처음 3 프레임만 (시간 = 첫 2 프레임 뺀 평균 규약이라 3 프레임째) + 그 3 프레임 오차.
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipg_wolf; mkdir -p $O
python -u exe/pareto_mpmbl.py --method ipg --nsub 1000 --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 3 --pg_cfg --time --out $O/N1000_t3 > $O/N1000_t3.log 2>&1
rclone copy $O $RB/ipg_wolf -q
echo "[i-PG wolf 3 프레임] $(tail -1 $O/N1000_t3.log | cut -c1-200)"
