# 강체 시작점 + 계량 H_L2 + 정규화 G 시험 (11-02 조정자): 수박 프레임 1 에서 3, 세 표현
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=83c8cef; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/w_test; mkdir -p $O; cd $WT; git log -1 --oneline
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas --riem_eps_ref l2 --riem_eps 1e-3 --riem_addl2 --riem_norm --warm_rigid --riem_gtol 1e-3 --riem_tol 1e-6 --riem_tmin 20 --riem_maxit 3000 --riem_cgtol 1e-2 --riem_cgmax 500 --riem_pcg --frames 3 --ckpt_every 0 --riem_prog 20"
run() { M=$1; G=$2; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx 0.006666666666666667"
  echo "$(date +%H:%M) [명령 GPU $G] exe/rep_track2.py $COM $X"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_watermelon_v2.npz --aux $T/aux_watermelon_v2.npz $COM $X --out $O/watermelon_$M.npz > $O/watermelon_$M.log 2>&1
  echo "$(date +%H:%M) $M: $(grep -a 't=\|요약\|rror' $O/watermelon_$M.log | cut -c1-200 | tr '\n' ' ')"; }
run fem 2 & run tetmesh 6 & run grid 4 &
wait; ~/bin/rclone copy $O r2:storage/result/anchorflow/topo/fem31/w_test 2>&1 | tail -1; echo WT_DONE
