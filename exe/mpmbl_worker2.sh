#!/bin/bash
# [final_fix_mpmbl] wolf i-PG 판을 접은 인스턴스의 후속 작업 (사람 손 없이): mpmbl_fix.sh (재질 값 전 구간 확인 + 필요하면 OmniPhysGS bread·wolf 다시)
#   -> i-PG bread 판 c (시간, 카드 단독; 원래 bread 인스턴스의 판 b 와 나란히) -> R2 대조 -> 스스로 destroy.
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0
R=r2:storage/result/anchorflow; RB=$R/lbours/final_fix/bl_mpm; X=/workspace/mb; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data
LOG=/root/mb/w2.log; say() { echo "[w2 $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RB/marks/w2_${MB_TAG:-x}.log -q; }
say "[시작] fix -> ipg_bread 판 c"
bash exe/mpmbl_fix.sh > /root/mb/fix.out 2>&1; say "[fix] $(tail -1 /root/mb/fix.log)"
O=$X/ipg_bread; mkdir -p $O
if [ -z "$(rclone lsf $RB/ipg_bread/N5_c/err.json 2>/dev/null)" ]; then
  AF_IPG_FUSED=0 AF_IPG_GPU=1 python -u exe/pareto_mpmbl.py --method ipg --nsub 5 --gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json \
    --pt $AG/bread.pt --frames 300 --pg_cfg --time --out $O/N5_c > $O/N5_c.log 2>&1
  rclone copy $O/N5_c $RB/ipg_bread/N5_c -q; rclone copyto $O/N5_c.log $RB/ipg_bread/N5_c.log -q
  say "[ipg_bread 판 c] $(tail -1 $O/N5_c.log | cut -c1-200)"
fi
rclone copy $X $RB -q --transfers 8 --exclude "*/h5/**"
if rclone check $X $RB --one-way --size-only -q --exclude "*/h5/**" 2>/root/mb/check2.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $RB/marks/${MB_TAG:-x}.ALL_DONE
  [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] && vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함 $(head -3 /root/mb/check2.txt)"
fi
