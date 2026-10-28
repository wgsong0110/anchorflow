#!/bin/bash
# doc/fem_realtime.md 절 J 4 차 묶음: 그래프 재생 혼합 PCG (pcg_g*, --pcg2 12) 벤치·80 프레임. 한 GPU 순차.   사용: bash exe/fem_rt_j4.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j4_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j4_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_start.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_t2_bench.py --t 0 40 70 --kerns graph10,graph20,graph40 < /dev/null > $L/graph.log 2>&1
rclone copy $L $R -q
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
B12=${BASE/--pcg2 6/--pcg2 12}
S="bash exe/fem_rt_sim.sh"
$S j2_g20 $G $B12 --rawptr --gk 20
$S j2_g20_r2 $G $B12 --rawptr --gk 20
touch $L/J4_DONE; rclone copy $L $R -q
