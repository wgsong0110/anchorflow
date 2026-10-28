#!/bin/bash
# doc/fem_realtime.md 절 J 3 차 묶음: 파이프라인 PCG (pcg_p1) 벤치·80 프레임, 헤시안 단계 벤치, 블록 지역 SGS 반복 수. 한 GPU 순차.
# 사용: bash exe/fem_rt_j3.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j3_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j3_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_start.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_t2_bench.py --t 0 40 70 --kerns pcg_t2_r2,pcg_p1_r1,pcg_p1_r2,pcg_p1_r4,pcg_p1_r2b2 < /dev/null > $L/p1.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_hess_bench.py --t 0 40 70 < /dev/null > $L/hess.log 2>&1
rclone copy $L $R -q
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
B11=${BASE/--pcg2 6/--pcg2 11}
S="bash exe/fem_rt_sim.sh"
$S j2_p1r2 $G $B11 --rawptr --t2_kern pcg_p1_r2
$S j2_p1r4 $G $B11 --rawptr --t2_kern pcg_p1_r4
until gpu_free; do sleep 20; done
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_prec_bench.py --t 0 40 70 --eig 0 --variants bj,ssor,lsgs64,lsgs256,lsgs1024,lsgs4096 --out $L < /dev/null > $L/lsgs.log 2>&1
touch $L/J3_DONE; rclone copy $L $R -q
