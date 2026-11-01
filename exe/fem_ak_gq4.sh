#!/bin/bash
# 절 AK.33 넷째: 회전·신장 평균의 비유한 대비 뒤 발산 3 궤적 다시 (gqx).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_ak_bselc
bash exe/fem_bsel_cuda_run.sh gqx 0006,0013,0019 --theta 0 --vfl_out --gq
for t in 0006 0013 0019; do echo "$t gqx $(grep -a '롤아웃' $R/${t}_gqx/log.txt | tail -1 | grep -o '[0-9]*/80 프레임[^,]*, 평균 [0-9.]*%') $(grep -a '가우스 점' $R/${t}_gqx/log.txt | grep -o '뒤집힌 입자 든 점 [0-9]*' | sort | uniq -c | tail -3 | tr '\n' ' ')"; done
echo "[끝] $(date +%T)"
