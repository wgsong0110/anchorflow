#!/bin/bash
# doc/fem_realtime.md 절 K 7 차: 지속 PCG 반복 비용 분해 벤치. 사용: bash exe/fem_rt_j7.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/j7_bench_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/j7_bench_$C; mkdir -p $L
gpu_free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -le 500 ]; }
until gpu_free; do sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/gpu_start.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_iter_bench.py --t 0 40 < /dev/null > $L/iter.log 2>&1
touch $L/J7_DONE; rclone copy $L $R -q
