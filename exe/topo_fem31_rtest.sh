# 이어 돌리기 시험 다시 (bee476b): bread fem·grid, A = 4 프레임 한 번에, B = A 의 2 프레임 체크포인트에서 이어서
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=bee476b; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/test/resume; mkdir -p $O; cd $WT; git log -1 --oneline
B="--flow $T/flow_bread_v2.npz --aux $T/aux_bread_v2.npz --opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --riem_partF --feas --frames 4 --ckpt_every 2"
one() { M=$1; g=$2; X="--method $M"; [ $M = fem ] && X="$X --mpm_dx 0.013333333333333334"
  CUDA_VISIBLE_DEVICES=$g python -u exe/rep_track2.py $B $X --out $O/${M}_A.npz > $O/${M}_A.log 2>&1
  cp $O/${M}_A.npz.ckpt.pt $O/${M}_B.npz.ckpt.pt
  CUDA_VISIBLE_DEVICES=$g python -u exe/rep_track2.py $B $X --resume --out $O/${M}_B.npz > $O/${M}_B.log 2>&1
  grep -a "이어 돌리기\]\|rror" $O/${M}_B.log | head -3
  python -c "
import numpy as np; A=np.load('$O/${M}_A.npz'); B=np.load('$O/${M}_B.npz')
print('[이어 돌리기 대조 $M] A RMSE', A['metrics'][:,1], ' B', B['metrics'][:,1], ' 궤적 최대 차', float(np.abs(A['traj']-B['traj']).max()))"; }
one fem 5 & one grid 6 & wait
echo RTEST_DONE
