#!/bin/bash
# exe/fem_fly.sh 결과의 lagdump.pt -> lagdump.npz (CPU), R2 동기화.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
CUDA_VISIBLE_DEVICES= python exe/fem_fly_npz.py $W/lbours/fem_fly_lego/fly_*/lagdump.pt < /dev/null
rclone copy $W/lbours/fem_fly_lego r2:storage/result/anchorflow/lbours/fem_fly_lego -q
touch $W/lbours/fem_fly_lego/NPZ_DONE
echo NPZ_DONE $(date +%T)
