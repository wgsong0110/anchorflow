#!/bin/bash
# 절 AK.33 진단: 가우스 판 0019 발산 (프레임 21) 직전 F_q·F_p det·속도 (gqd), 끝나면 R2 표식.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
bash exe/fem_bsel_cuda_run.sh gqd 0019 --theta 0 --vfl_out --gq
echo "[끝] $(date +%T)" > /tmp/dkta/lbours/fem_ak_bselc/0019_gqd/DONE_MARK; rclone copy /tmp/dkta/lbours/fem_ak_bselc/0019_gqd r2:storage/result/anchorflow/lbours/fem_ak_bselc/0019_gqd -q
