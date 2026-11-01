# 공정 판 시험 (11-01 조정자 지시): 감쇠 ε = 0.01 λmax(H_L2) + 프레임별 수렴 기준, 입자 det 장벽·가능성 검사·씬 node_h 유지. wolf·bread 우리 표현 30 프레임
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=e81a1cc; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/fix_test; mkdir -p $O; cd $WT; git log -1 --oneline
R2=r2:storage/result/anchorflow/topo/fem31/fix_test
dxof() { case $1 in wolf) echo 0.01;; bread) echo 0.013333333333333334;; esac; }
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas --riem_eps_ref l2 --frames 30 --ckpt_every 5"
run() { S=$1; TOL=$2; G=$3; X="--method fem --mpm_dx $(dxof $S) --riem_tol $TOL"
  echo "$(date +%H:%M) [명령 GPU $G] exe/rep_track2.py --flow $T/flow_${S}_v2.npz --aux $T/aux_${S}_v2.npz $COM $X --out $O/${S}_tol$TOL.npz"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_${S}_v2.npz --aux $T/aux_${S}_v2.npz $COM $X --out $O/${S}_tol$TOL.npz > $O/${S}_tol$TOL.log 2>&1
  echo "$(date +%H:%M) $S tol $TOL: $(grep -a '요약\|rror' $O/${S}_tol$TOL.log | tail -1 | cut -c1-230)"; }
run wolf 1e-3 4 & run wolf 1e-4 5 & run bread 1e-3 6 & run bread 1e-4 6 &
wait
~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1
echo FIXT_DONE
