#!/bin/bash
# 절 AK.33 셋째: 칸 8 가우스 점 판 (gqw) lego 21 비측정.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_ak_bselc; SC=L000,$(seq -f %04g 0 19 | paste -sd,)
df -h /tmp | tail -1
bash exe/fem_bsel_cuda_run.sh gqw $SC --theta 0 --vfl_out --gq
for t in L000 $(seq -f %04g 0 19); do echo "$t gqw $(grep -a '롤아웃' $R/${t}_gqw/log.txt | tail -1 | cut -c1-170) $(grep -a '가우스 점' $R/${t}_gqw/log.txt | tail -1 | grep -o '가우스 점.*')"; done
echo "[끝] $(date +%T)"
