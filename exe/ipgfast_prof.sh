#!/bin/bash
# [final_fix_mpmbl] i-PG wolf N100 프로파일: 원 구현 (AF_IPG_GPU=1) vs 빠른 구현 (AF_IPG_FAST=1), 단독 카드에서 직렬.
#   사용: bash exe/ipgfast_prof.sh [steps=8]   결과 /workspace/mb/ipgfast/ -> R2 lbours/final_fix/bl_mpm/ipg_wolf/ipgfast/
set -u
K=${1:-8}; N=100
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1 AF_FD_EPS=1e-5
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/ipg_wolf/ipgfast; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipgfast; mkdir -p $O
python exe/patch_ipg_noisefloor.py --ipg $W/i-physgaussian
ARGS="--method ipg --nsub $N --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 1 --pg_cfg --time"
for M in orig fast; do
  [ $M = fast ] && export AF_IPG_FAST=1 || export AF_IPG_FAST=0
  python -u exe/prof_ipg.py --prof_steps $K --prof_skip 2 --prof_torch 4:5 --prof_out $O/prof_$M.json -- $ARGS --out $O/run_$M > $O/prof_$M.log 2>&1
  echo "[$M] $(grep '프로파일 요약' $O/prof_$M.log | cut -c1-400)"
  if command -v py-spy >/dev/null; then
    py-spy record -r 50 -f raw -o $O/pyspy_$M.txt -- python -u exe/prof_ipg.py --prof_steps $((K-2)) --prof_skip 2 --prof_out $O/prof2_$M.json -- $ARGS --out $O/run2_$M > $O/pyspy_$M.log 2>&1
  fi
  rclone copy $O $RB -q --exclude "run*/**"
done
echo "[끝] $(date)"
