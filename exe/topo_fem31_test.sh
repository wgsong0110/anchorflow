# 10-31 토폴로지 추적 FEM 표현 판 시험: 격자 대조 (5 씬) + bread 3 표현 3 프레임 + 이어 돌리기 확인 + 수박 fem 2 프레임 (메모리·시간)
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=f5b2369; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/test; mkdir -p $O; cd $WT; git log -1 --oneline
G1=${G1:-1}; G2=${G2:-2}; G3=${G3:-3}; G4=${G4:-4}
dxof() { case $1 in wolf) echo 0.01;; bread) echo 0.013333333333333334;; watermelon) echo 0.006666666666666667;; ficus) echo 0.04;; hotdog) echo 0.02;; esac; }
fl() { case $1 in ficus|hotdog) echo $1;; *) echo ${1}_v2;; esac; }
for S in wolf bread watermelon ficus hotdog; do CUDA_VISIBLE_DEVICES=$G4 python -u exe/topo_fem_equiv.py --aux $T/aux_$(fl $S).npz --mpm_dx $(dxof $S) 2>&1 | grep -a "대조\|rror" | sed "s/^/$S /"; done
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas"
run() { S=$1; M=$2; G=$3; shift 3; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx $(dxof $S)"
  [ $S$M = breadtetmesh ] && X="$X --riem_damp fix --riem_eps0 0.02603914"
  echo "[명령] python -u exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X $*"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_$(fl $S).npz --aux $T/aux_$(fl $S).npz $COM $X "$@" > $O/${S}_$M$TAG.log 2>&1
  echo "$(date +%H:%M) $S $M$TAG: $(grep -a '요약\|rror' $O/${S}_$M$TAG.log | tail -1 | cut -c1-220)"; }
TAG= run bread fem $G1 --frames 3 --out $O/bread_fem.npz &
TAG= run bread tetmesh $G2 --frames 3 --out $O/bread_tetmesh.npz &
TAG= run bread grid $G3 --frames 3 --out $O/bread_grid.npz &
TAG= run watermelon fem $G4 --frames 2 --out $O/watermelon_fem.npz &
wait
# 이어 돌리기: A = 4 프레임 한 번에 (2 프레임에 체크포인트), B = A 의 체크포인트에서 이어서 3·4 프레임
TAG=_A run bread fem $G1 --frames 4 --ckpt_every 2 --out $O/bread_fem_A.npz
cp $O/bread_fem_A.npz.ckpt.pt $O/bread_fem_B.npz.ckpt.pt
TAG=_B run bread fem $G1 --frames 4 --ckpt_every 2 --resume --out $O/bread_fem_B.npz
grep -a "이어 돌리기" $O/bread_fem_B.log
python -c "
import numpy as np; A=np.load('$O/bread_fem_A.npz'); B=np.load('$O/bread_fem_B.npz')
print('[이어 돌리기 대조] A RMSE', A['metrics'][:,1], ' B', B['metrics'][:,1]); print('[이어 돌리기 대조] 궤적 최대 차', float(np.abs(A['traj']-B['traj']).max()))"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo FEM31_TEST_DONE
