# 10-31 토폴로지 추적 본 실행: FEM 과 같은 표현 (fem) · 사면체 메쉬 · 그리드 × wolf bread watermelon ficus hotdog
# 공통: riem (로그 장벽 계량, ε=0.01λmax, CG 50, Armijo 30, 400/200 반복) + 입자 누적 det F 로그 장벽 (--riem_partF) + 가능성 선탐색 (--feas)
# 이어 돌리기: 다시 실행하면 끝난 것은 건너뛰고 체크포인트(5 프레임마다)에서 이어간다. 결과는 15 분마다 R2 로 통째로
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=bee476b; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/main; mkdir -p $O; cd $WT; git log -1 --oneline
R2=r2:storage/result/anchorflow/topo/fem31/main
dxof() { case $1 in wolf) echo 0.01;; bread) echo 0.013333333333333334;; watermelon) echo 0.006666666666666667;; ficus) echo 0.04;; hotdog) echo 0.02;; esac; }
fl() { case $1 in ficus|hotdog) echo $1;; *) echo ${1}_v2;; esac; }
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas --ckpt_every 5"
run() { S=$1; M=$2; G=$3; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx $(dxof $S)"
  [ $S$M = breadtetmesh ] && X="$X --riem_damp fix --riem_eps0 0.02603914"
  [ -f $O/${S}_$M.npz ] && { echo "$S $M: 이미 있음"; return; }
  RES=""; [ -f $O/${S}_$M.npz.ckpt.pt ] && RES=--resume
  echo "$(date +%H:%M) [명령 GPU $G] exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X $RES --out $O/${S}_$M.npz"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X $RES --out $O/${S}_$M.npz >> $O/${S}_$M.log 2>&1
  echo "$(date +%H:%M) $S $M: $(grep -a '요약\|rror' $O/${S}_$M.log | tail -1 | cut -c1-230)"
  ~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1; }
( while true; do sleep 900; ~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1; done ) & SYNC=$!
P=""
run watermelon fem 1 & P="$P $!"
run watermelon tetmesh 2 & P="$P $!"
run watermelon grid 3 & P="$P $!"
( run wolf fem 4 & run wolf tetmesh 4 & run wolf grid 4 & run hotdog fem 4 & wait ) & P="$P $!"
( run bread fem 5 & run bread tetmesh 5 & run bread grid 5 & run hotdog tetmesh 5 & wait ) & P="$P $!"
( run ficus fem 6 & run ficus tetmesh 6 & run ficus grid 6 & run hotdog grid 6 & wait ) & P="$P $!"
wait $P
kill $SYNC 2>/dev/null
~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1
echo FEM31_MAIN_DONE
