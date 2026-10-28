#!/bin/bash
# doc/fem_realtime.md 절 J 1 차 묶음 (사용자 10-28 "할 수 있는 거 다 해보고 오차랑 영상이랑 시간 측정"): 한 GPU 에서 순차 (속도 측정 단독).
#  (a) 기준 i_tmix_asm_r3 설정 재측정 2 번 + 구간 동기화 계측 1 번, (b) 항목 3 대조: --near32 뺀 판,
#  (c) 전처리·행렬 정밀도 탐색 (fem_rt_prec_bench.py), (d) 16 비트 행렬·재정렬 마이크로벤치 (fem_rt_th_bench.py).
# 사용: bash exe/fem_rt_j1.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j1_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j1_bench_$C; mkdir -p $L
( while true; do sleep 120; rclone copy $L $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred --asm_k --pcg2 6 --mix_delta 0.1"
S="bash exe/fem_rt_sim.sh"
$S j_base_r1 $G $BASE
$S j_base_r2 $G $BASE
$S j_base_prof $G $BASE --prof
$S j_nonear32 $G ${BASE/--near32 /}
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_th.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_th_bench.py --t 0 40 < /dev/null > $L/th.log 2>&1
rclone copy $L $R -q
until gpu_free; do sleep 20; done
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_prec_bench.py --t 0 40 70 10 --out $L < /dev/null > $L/prec.log 2>&1
rclone copy $L $R -q
touch $L/J1_DONE; rclone copy $L $R -q
