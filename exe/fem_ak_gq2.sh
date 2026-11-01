#!/bin/bash
# 절 AK.33 둘째: 칸마다 8 가우스 점 같은 가중 (gqw) -- 블록·ficus 주기·떨어지는 블록 (+ ficus 입자 적분점 판 덤프).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_ak_bselc
rm -f $R/ficus_bnv/summary.json
bash exe/fem_bsel_cuda_run.sh gqw stretch,shear,ficus,drop1 --theta 0 --vfl_out --gq
bash exe/fem_bsel_cuda_run.sh bnv ficus --theta 0 --vfl_out
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt bn:$R/${k}_bnv/pred.pt gq:$R/${k}_gqw/pred.pt 2>&1 | grep -v "^ \|Traceback\|Warning"; done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $R/ficus_bnv/pred.pt $R/ficus_gqw/pred.pt 2>&1 | grep -v Warning
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 bn:$R/drop1_bnv/pred.pt:0.0166666667 gq:$R/drop1_gqw/pred.pt:0.0166666667 2>&1 | grep -v "Warning\|Traceback\|File\|^ "
for t in stretch shear ficus drop1; do echo "${t}_gqw $(grep -a '롤아웃' $R/${t}_gqw/log.txt | tail -1 | cut -c1-170) $(grep -a '가우스 점' $R/${t}_gqw/log.txt | tail -1 | grep -o '가우스 점.*')"; done
echo "[시험 끝] $(date +%T)"
