# 공정 판 진단: 프레임 안 L2 곡선 (수렴 기준 정하기). 감쇠 L2 기준 ε = k·λmax(H_L2), k = 0.01 / 0.1 / 1, 최대 1500 반복, 멈춤 없음
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=e81a1cc; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; T=$W/topo; O=$T/fem31/fix_diag; mkdir -p $O; cd $WT
COM="--opt riem --kappa 0 --tb none --riem_energy logbarrier --riem_partF --feas --riem_eps_ref l2 --riem_tol 1e-12 --riem_maxit 1500 --ckpt_every 0"
r() { S=$1; K=$2; F=$3; G=$4; X="--method fem --mpm_dx $( [ $S = wolf ] && echo 0.01 || echo 0.013333333333333334 )"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_${S}_v2.npz --aux $T/aux_${S}_v2.npz $COM $X --riem_eps $K --frames $F --out $O/${S}_k$K.npz > $O/${S}_k$K.log 2>&1
  echo "$(date +%H:%M) $S k$K: $(grep -a '요약\|rror' $O/${S}_k$K.log | tail -1 | cut -c1-200)"; }
r bread 0.01 1 4 & r bread 0.1 1 4 & r bread 1 1 5 & r wolf 0.01 2 5 & r wolf 0.1 2 6 & r wolf 1 2 6 &
wait; ~/bin/rclone copy $O r2:storage/result/anchorflow/topo/fem31/fix_diag 2>&1 | tail -1; echo FIXD_DONE
