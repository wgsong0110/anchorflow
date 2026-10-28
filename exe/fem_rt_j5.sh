#!/bin/bash
# doc/fem_realtime.md 절 J 5 차 묶음: PCG 이전 프레임 시작 해 (--warm_pcg) 벤치·80 프레임, E·기울기 float32 대수 (--eg32) 벤치·80 프레임. 한 GPU 순차.
# 사용: bash exe/fem_rt_j5.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j5_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j5_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_start.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_warm_bench.py --t 0 10 40 70 < /dev/null > $L/warm.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_eg_bench.py --t 0 40 70 < /dev/null > $L/eg.log 2>&1
rclone copy $L $R -q
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
B10=${BASE/--pcg2 6/--pcg2 10}
S="bash exe/fem_rt_sim.sh"
$S j1_warm $G $B10 --rawptr --t2_kern pcg_t2_r1 --warm_pcg
$S j1_warm_r2 $G $B10 --rawptr --t2_kern pcg_t2_r1 --warm_pcg
$S j1_t2r1 $G $B10 --rawptr --t2_kern pcg_t2_r1
$S j4_eg32tet $G $BASE --rawptr --eg32 tet
$S j4_eg32near $G $BASE --rawptr --eg32 near
$S j4_eg32both $G $BASE --rawptr --eg32 both
touch $L/J5_DONE; rclone copy $L $R -q
