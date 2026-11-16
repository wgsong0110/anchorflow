#!/bin/bash
# [final_fix_mpmbl] i-PG bread N 10 (k=10; k=20 N 5 는 프레임 166 에서 Newton/GMRES NaN 발산) 의 시간 판 하나 (판 이름 = MB_TAG 끝 글자 b|c), 카드 단독. 판 a (덤프) 는 bread 인스턴스.
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1
REP=${MB_TAG: -1}; RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipg_bread; mkdir -p $O
( while sleep 300; do rclone copyto $O/N10_$REP.log $RB/ipg_bread/N10_$REP.log -q; done ) & SY=$!
python -u exe/pareto_mpmbl.py --method ipg --nsub 10 --gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $AG/bread.pt --frames 300 --pg_cfg --time --out $O/N10_$REP > $O/N10_$REP.log 2>&1
kill $SY; rclone copy $O/N10_$REP $RB/ipg_bread/N10_$REP -q; rclone copyto $O/N10_$REP.log $RB/ipg_bread/N10_$REP.log -q
echo "[i-PG bread 판 $REP] $(tail -1 $O/N10_$REP.log | cut -c1-200)"
