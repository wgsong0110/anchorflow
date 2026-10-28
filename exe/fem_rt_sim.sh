#!/bin/bash
# FEM 실시간 판 단독 시뮬레이터 실행 (doc/fem_realtime.md 절 H, exe/fem_rt_sim.py). 결과는 lbours/fem_rt_lego/<태그> 에 쓰고 R2 로 동기화.
# 사용: bash exe/fem_rt_sim.sh <태그> <GPU> [fem_rt_sim.py 인자...]      표식 RT_DONE / RT_FAIL
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; shift 2
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
O=$W/lbours/fem_rt_lego/$TG; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/$TG; mkdir -p $O
( while true; do sleep 120; rclone copy $O $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do echo "[대기] GPU$G $(date +%T)"; sleep 20; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
echo "[명령] 커밋 $C CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_sim.py --out $O $*" | tee $O/cmd.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_sim.py --out $O "$@" < /dev/null > $O/v_fem.log 2>&1
grep -a "롤아웃\]\|구간 중앙" $O/v_fem.log | tail -2 > $O/err_fem.txt
if [ -f $O/summary.json ]; then touch $O/RT_DONE; else grep -a "Traceback\|Error" $O/v_fem.log | tail -3 > $O/RT_FAIL; fi
cat $O/err_fem.txt
rclone copy $O $R -q
