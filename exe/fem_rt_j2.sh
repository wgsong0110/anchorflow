#!/bin/bash
# doc/fem_realtime.md 절 J 2 차 묶음: (a) 지속 PCG 반복 비용 벤치 (fem_rt_t2_bench.py), (b) 항목 4 (호스트 동기화·인자 변환·격자·P2G) 80 프레임,
# (c) 항목 2 pcg_t2 (곱 메모리 병렬도) 80 프레임. 한 GPU 순차 (속도 측정 단독).   사용: bash exe/fem_rt_j2.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j2_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j2_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_t2.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_t2_bench.py --t 0 40 70 < /dev/null > $L/t2.log 2>&1
rclone copy $L $R -q
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
S="bash exe/fem_rt_sim.sh"
$S j4_raw $G $BASE --rawptr
$S j4_raw_ls $G $BASE --rawptr --lowsync
$S j4_raw_ls_g2 $G $BASE --rawptr --lowsync --gridk2
$S j4_raw_ls_g2_p2g $G $BASE --rawptr --lowsync --gridk2 --p2gk
$S j4_all_prof $G $BASE --rawptr --lowsync --gridk2 --p2gk --prof
B10=${BASE/--pcg2 6/--pcg2 10}
$S j2_t2r2 $G $B10 --rawptr --t2_kern pcg_t2_r2
$S j2_t2r4 $G $B10 --rawptr --t2_kern pcg_t2_r4
$S j2_t2r2b2 $G $B10 --rawptr --t2_kern pcg_t2_r2b2
touch $L/J2_DONE; rclone copy $L $R -q
