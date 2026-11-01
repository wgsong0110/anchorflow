# 공정성 확인 (11-01 조정자): 세 표현 모두 프레임 1 L2 곡선으로 감쇠 배수 k 고르기 (ε = k·λmax(H_L2)), bread·ficus, 600 반복 멈춤 없음
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=e81a1cc; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; T=$W/topo; O=$T/fem31/kdiag; mkdir -p $O; cd $WT
COM="--opt riem --kappa 0 --tb none --riem_energy logbarrier --riem_partF --feas --riem_eps_ref l2 --riem_tol 1e-12 --riem_maxit 600 --ckpt_every 0 --frames 1"
fl() { case $1 in ficus) echo ficus;; *) echo ${1}_v2;; esac; }
r() { S=$1; M=$2; K=$3; G=$4; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx $( [ $S = ficus ] && echo 0.04 || echo 0.013333333333333334 )"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X --riem_eps $K --out $O/${S}_${M}_k$K.npz > $O/${S}_${M}_k$K.log 2>&1
  echo "$(date +%H:%M) $S $M k$K: $(grep -a '요약\|rror' $O/${S}_${M}_k$K.log | tail -1 | cut -c1-120)"; }
g=4
for S in bread ficus; do for M in fem tetmesh grid; do
  ( for K in 0.01 0.03 0.1 0.3 1; do r $S $M $K $g; done ) &
  g=$((g+1)); [ $g -gt 6 ] && g=4
done; done
wait; ~/bin/rclone copy $O r2:storage/result/anchorflow/topo/fem31/kdiag 2>&1 | tail -1; echo KDIAG_DONE
