#!/bin/bash
# [final_fix_mpmbl] i-PG bread N10 판 a (덤프 + 시간, 카드 단독): 레고 boot 가 끝난 뒤 이 카드에서. (원래 bread 카드는 재시작 뒤 프레임당 540~840 s 로 느려져 옮김)
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1
while pgrep -f mb_boot.sh >/dev/null; do sleep 60; done
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipg_bread; mkdir -p $O
( while sleep 300; do rclone copyto $O/N10_a.log $RB/ipg_bread/N10_a.log -q; done ) & SY=$!
python -u exe/pareto_mpmbl.py --method ipg --nsub 10 --gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $AG/bread.pt --frames 300 --pg_cfg --time --dump --out $O/N10_a > $O/N10_a.log 2>&1
kill $SY; rclone copy $O/N10_a $RB/ipg_bread/N10_a -q; rclone copyto $O/N10_a.log $RB/ipg_bread/N10_a.log -q
echo "[i-PG bread 판 a] $(tail -1 $O/N10_a.log | cut -c1-200)"
