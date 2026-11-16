#!/bin/bash
# [final_fix_mpmbl] i-PG wolf 전 50 프레임 판 (덤프): 전방차분 섭동 AF_FD_EPS=1e-5 (원 1e-4 는 첫 프레임 폭주; exe/patch_ipg_noisefloor.py)
#   -> 렌더·물리·시각 -> R2. 프레임마다 체크포인트 (R2), 다시 부르면 이어 돌림.  사용: bash exe/mpmbl_ipgwolf_full.sh [판=a] [N=100]
set -u
REP=${1:-a}; N=${2:-100}; EPS=1e-5
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1 AF_FD_EPS=$EPS
RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/ipg_wolf; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/ipg_wolf; mkdir -p $O
T=N${N}e5_$REP
[ -e $O/post_$T/DONE ] && { echo "[이미 끝]"; exit 0; }
python exe/patch_ipg_noisefloor.py --ipg /home/dkta/work/i-physgaussian
grep -q AF_FD_EPS /home/dkta/work/i-physgaussian/implicit_mpm_solver.py || { echo "패치 없음"; exit 1; }
( while sleep 600; do rclone copyto $O/$T.log $RB/$T.log -q; done ) & SY=$!
python -u exe/pareto_mpmbl.py --method ipg --nsub $N --gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 50 --pg_cfg --time --dump --out $O/$T --ckpt_every 1 --ckpt_r2 $RB/$T/ckpt.pt --resume >> $O/$T.log 2>&1
grep -q "^\[결과\]" <(tail -1 $O/$T.log) || { echo "[결과 없음 -- 다시 부르면 이어 돌림]"; kill $SY; exit 1; }
kill $SY; rclone copy $O/$T $RB/$T -q; rclone copyto $O/$T.log $RB/$T.log -q
echo "[판 $REP] $(tail -1 $O/$T.log | cut -c1-200)"
case $REP in a|fa) ;; *) exit 0;; esac
GTD=$AG/gt_wolf; P=$O/post_$T; mkdir -p $P
python -u exe/lb_roll_to_h5.py --dump $O/$T/pred.pt --traj $GTD --out $P/h5 > $P/roll_h5.log 2>&1
python -u exe/lb_render.py --sim $P/h5 --config $GTD/config.json --shape wolf --out $P/ipg_wolf.mp4 --png $P/png --frames 50 < /dev/null > $P/render.log 2>&1; rm -rf $P/h5
python -u exe/ffx_phys.py --method ipg --scene wolf --N $N --data $AG/wolf.pt --pred $O/$T/pred.pt --floor vel --frames 50 --out $P/phys.json --note "대표 설정 덤프 판 (ipg N $N k=20, AF_FD_EPS $EPS)" > $P/phys.log 2>&1
python -u exe/ffx_vis.py --png $P/png --ref /root/mb/ref/wolf --out $P/vis.json > $P/vis.log 2>&1
rclone copy $P $RB/post_$T -q --transfers 8
[ -s $P/phys.json ] && [ -s $P/vis.json ] && touch $P/DONE && rclone copyto $P/DONE $RB/post_$T/DONE -q
echo "[후처리] $(tail -1 $P/vis.log | cut -c1-120)"
