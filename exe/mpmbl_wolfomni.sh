#!/bin/bash
# [final_fix_mpmbl] OmniPhysGS wolf: 공식 dt 3e-4 (N 134) 는 E 5e7 모래에서 첫 프레임에 바닥을 뚫고 폭주 (명시 MPM CFL: c = sqrt(E/rho), dt < dx/c ~ 6e-5).
#   -> 공식 dt 부터 서브스텝을 2 배씩 (PG 공식 dt 2e-5 = N 1999 까지) 올려 처음 발산 없는 N (판 a 덤프) -> 판 b·c (시간, 단독) -> 렌더·물리·시각.  마찰각 = 30 (재질 전 구간 확인: 정답 경로가 config 값).
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUDA_VISIBLE_DEVICES=0
unset AF_IPG_FUSED AF_IPG_GPU
R=r2:storage/result/anchorflow; RB=$R/lbours/final_fix/bl_mpm; X=/workspace/mb; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data
LOG=/root/mb/wo.log; say() { echo "[wo $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RB/marks/wolfomni.log -q; }
SIM="--gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 50 --pg_cfg --omni_fric 30.0"
J=omni_wolf_cfl; O=$X/$J; mkdir -p $O; N=""; TR=""
for n in 134 268 536 1072 1999; do
  [ -s $O/N${n}_a/err.json ] || python -u exe/pareto_mpmbl.py --method omni --nsub $n $SIM --time --dump --out $O/N${n}_a > $O/N${n}_a.log 2>&1
  ok=$(python -c "import json;d=json.load(open('$O/N${n}_a/err.json'));print(1 if d['div'] is None and d['frames_done']==d['frames_total'] else 0)" 2>/dev/null || echo 0)
  say "[판 a] $J N $n ok=$ok $(grep '발산' $O/N${n}_a.log | head -1 | cut -c1-120) $(tail -1 $O/N${n}_a.log | cut -c1-160)"; TR="$TR $n:$ok"
  [ $ok = 1 ] && { N=$n; break; }
  rm -f $O/N${n}_a/pred.pt
done
echo "$TR" > $O/tried.txt; echo "${N:-none} 134 1 --omni_fric 30.0" > $O/chosen.txt
[ -z "$N" ] && { say "[전부 발산] $TR"; rclone copy $O $RB/$J -q; exit 0; }
rclone copyto $O/N${N}_a/pred.pt $RB/$J/N${N}_a/pred.pt -q
for rep in b c; do
  [ -s $O/N${N}_$rep/err.json ] || python -u exe/pareto_mpmbl.py --method omni --nsub $N $SIM --time --out $O/N${N}_$rep > $O/N${N}_$rep.log 2>&1
  say "[판 $rep] $(tail -1 $O/N${N}_$rep.log | cut -c1-160)"
done
GTD=$AG/gt_wolf
python -u exe/lb_roll_to_h5.py --dump $O/N${N}_a/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
python -u exe/lb_render.py --sim $O/h5 --config $GTD/config.json --shape wolf --out $O/$J.mp4 --png $O/png --frames 50 < /dev/null > $O/render.log 2>&1; rm -rf $O/h5
python -u exe/ffx_phys.py --method omni --scene wolf --N $N --data $AG/wolf.pt --pred $O/N${N}_a/pred.pt --floor vel --frames 50 --out $O/phys.json --note "대표 설정 덤프 판 (omni N $N, 공식 dt 3e-4 발산 -> CFL)" > $O/phys.log 2>&1
python -u exe/ffx_vis.py --png $O/png --ref /root/mb/ref/wolf --out $O/vis.json > $O/vis.log 2>&1
rclone copy $O $RB/$J -q --transfers 8 --exclude "h5/**"
[ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RB/$J/DONE -q
say "[끝] $J N $N"
