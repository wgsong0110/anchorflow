#!/bin/bash
# doc/fem_realtime.md 절 J 6 차 묶음: 헤시안 양정치 문 (--hgate) 벤치·80 프레임, pcg_t2 실제 잔차 기록 고친 판 재측정. 한 GPU 순차.   사용: bash exe/fem_rt_j6.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j6_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j6_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_start.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_hess_bench.py --t 0 40 70 < /dev/null > $L/hess.log 2>&1
rclone copy $L $R -q
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
S="bash exe/fem_rt_sim.sh"
$S j4_hgate $G $BASE --rawptr --hgate
$S j4_hgate_prof $G $BASE --rawptr --hgate --prof
$S j2_t2r1_fix $G ${BASE/--pcg2 6/--pcg2 10} --rawptr --t2_kern pcg_t2_r1
touch $L/J6_DONE; rclone copy $L $R -q
