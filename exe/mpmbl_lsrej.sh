#!/bin/bash
# [final_fix_mpmbl] i-PG 줄 탐색 실패 걸음 버림 (AF_LS_REJECT=1) 판 1 회: Lego·pillow2sofa·bread (공정성 -- wolf 와 같은 규칙). 덤프 + 시간 + [LS 실패] 집계, 체크포인트 이어 돌리기.
#  사용: bash exe/mpmbl_lsrej.sh <L000|pillow2sofa|bread> [판=r]
set -u
S=$1; REP=${2:-r}
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_LS_REJECT=1
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm
case $S in
  L000) N=42; export AF_IPG_FUSED=1; SIM="--gt $W/lbdata/test_low/L000 --config $W/lbdata/config.json --pt $SD/L000.pt --frames 80";;
  pillow2sofa) N=10; export AF_IPG_FUSED=1; SIM="--gt $NW/pillow2sofa_h5/simulation_ply --config $NW/pillow2sofa_config.json --pt $NW/pillow2sofa.pt --frames 300 --pg_cfg";;
  bread) N=10; export AF_IPG_FUSED=0 AF_IPG_GPU=1 AF_IPG_FAST=1; SIM="--gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $AG/bread.pt --frames 300 --pg_cfg";;
esac
python exe/patch_ipg_noisefloor.py --ipg $W/i-physgaussian
grep -q "LS 실패" $W/i-physgaussian/implicit_mpm_solver.py || { echo "패치 없음"; exit 1; }
O=/workspace/mb/ipg_${S}; mkdir -p $O; T=N${N}_ls$REP
( while sleep 600; do rclone copyto $O/$T.log $RB/ipg_${S}/$T.log -q; done ) & SY=$!
python -u exe/pareto_mpmbl.py --method ipg --nsub $N $SIM --time --dump --out $O/$T --ckpt_every 10 --ckpt_r2 $RB/ipg_${S}/$T/ckpt.pt --resume >> $O/$T.log 2>&1
kill $SY; rclone copy $O/$T $RB/ipg_${S}/$T -q; rclone copyto $O/$T.log $RB/ipg_${S}/$T.log -q
echo "[$S $T] $(tail -1 $O/$T.log | cut -c1-200) / LS 실패 $(grep -c 'LS 실패' $O/$T.log) 회"
grep -q "^\[결과\]" <(tail -1 $O/$T.log)
