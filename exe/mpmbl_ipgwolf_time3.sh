#!/bin/bash
# [final_fix_mpmbl] i-PG wolf k=2 (N 1000) 3 프레임 단독 시간 반복 (판 = $1). k=20·10·5 는 첫 프레임 발산 (float32 응력 잡음 바닥, exe/diag_ipg_wolf.py),
# k=2 는 50 프레임 ~73 h 라 오차 칸 불가 -> 시간만 (첫 2 프레임 뺀 평균 규약 = 3 프레임째). 앞선 bread 시간 판 (mpmbl_breadrep) 이 끝난 뒤 카드 단독.
set -u
REP=$1; WAITPID=$2
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipg_wolf; mkdir -p $O
while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "[대기] 카드에 다른 프로세스 $(date)"; sleep 60; done
echo "[시작] $(date -u) 카드 $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader)" | tee $O/N1000_t3_$REP.start
( while sleep 600; do rclone copyto $O/N1000_t3_$REP.log $RB/ipg_wolf/N1000_t3_$REP.log -q; done ) & SY=$!
python -u exe/pareto_mpmbl.py --method ipg --nsub 1000 --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 3 --pg_cfg --time --out $O/N1000_t3_$REP > $O/N1000_t3_$REP.log 2>&1
kill $SY; rclone copy $O/N1000_t3_$REP $RB/ipg_wolf/N1000_t3_$REP -q; rclone copyto $O/N1000_t3_$REP.log $RB/ipg_wolf/N1000_t3_$REP.log -q
echo "[i-PG wolf 시간 판 $REP] $(tail -1 $O/N1000_t3_$REP.log | cut -c1-200)"
