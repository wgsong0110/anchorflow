# 11-02 수박 30 프레임 비교 (조정자 결정): 강체 시작점 (--warm_rigid) + 계량 H_L2 + 정규화 G + 1e-3 λmax(H_L2), 정지 = |g|/|g0| ≤ 1e-3 또는 10 반복 상대 L2 감소 ≤ 1e-6, 세 표현 같은 최대 30 반복
# (최근 10 반복 상대 L2 감소 ≤ 1e-2, 최소 20·최대 2000). 나머지 (입자 det 장벽·가능성 검사·씬 node_h·목표 파일) 는 main 과 같다. bread 사면체 메쉬도 같은 규칙 (감쇠 고정판 아님)
# SCENES="wolf bread ficus hotdog" GPUS="4 5 6" METHODS="fem tetmesh grid" 로 고른다
# 공통: riem (로그 장벽 계량, ε=0.01λmax, CG 50, Armijo 30, 400/200 반복) + 입자 누적 det F 로그 장벽 (--riem_partF) + 가능성 선탐색 (--feas)
# 이어 돌리기: 다시 실행하면 끝난 것은 건너뛰고 체크포인트(5 프레임마다)에서 이어간다. 결과는 15 분마다 R2 로 통째로
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=8ebc6dd; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/wm30; mkdir -p $O; cd $WT; git log -1 --oneline
R2=r2:storage/result/anchorflow/topo/fem31/wm30
dxof() { case $1 in wolf) echo 0.01;; bread) echo 0.013333333333333334;; watermelon) echo 0.006666666666666667;; ficus) echo 0.04;; hotdog) echo 0.02;; esac; }
fl() { case $1 in ficus|hotdog) echo $1;; *) echo ${1}_v2;; esac; }
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas --riem_eps_ref l2 --riem_eps 1e-3 --riem_addl2 --riem_norm --warm_rigid --riem_gtol 1e-3 --riem_tol 1e-6 --riem_tmin 20 --riem_maxit 30 --riem_cgtol 1e-2 --riem_cgmax 500 --riem_pcg --riem_prog 10 --frames 30 --ckpt_every 5"
run() { S=$1; M=$2; G=$3; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx $(dxof $S)"
  [ -f $O/${S}_$M.npz ] && { echo "$S $M: 이미 있음"; return; }
  RES=""; [ -f $O/${S}_$M.npz.ckpt.pt ] && RES=--resume
  echo "$(date +%H:%M) [명령 GPU $G] exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X $RES --out $O/${S}_$M.npz"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X $RES --out $O/${S}_$M.npz >> $O/${S}_$M.log 2>&1
  echo "$(date +%H:%M) $S $M: $(grep -a '요약\|rror' $O/${S}_$M.log | tail -1 | cut -c1-230)"
  ~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1; }
( while true; do sleep 900; ~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1; done ) & SYNC=$!
SCENES=${SCENES:-"wolf bread ficus hotdog"}; GA=(${GPUS:-4 5 6}); P=""; n=0
for S in $SCENES; do for M in ${METHODS:-fem tetmesh grid}; do
  run $S $M ${GA[$((n % ${#GA[@]}))]} & P="$P $!"; n=$((n+1))
done; done
wait $P
kill $SYNC 2>/dev/null
~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1
echo FEM31_WM30_DONE
