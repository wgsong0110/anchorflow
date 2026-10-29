#!/bin/bash
# 절 AA 1b: 힘 검산 몫별 (ψ / X 경유 q / S 경유 q) -- lego 40 프레임 제한 0.3, 프레임 20·40 (비측정, GPU 1)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1 EX_NOWAIT=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_aa_lego; mkdir -p $OL
EX_ROOT=fem_aa_lego bash exe/fem_ex_run.sh aa_chk2_l40 1 --mode lag --opt --mass tet --no_dump --floor vel --rstat --cq --frames 40 --N 192 --lim 0.3 --lim_mode box --cqchk 3,20,40 < /dev/null > /dev/null 2>&1
grep -a "AA 검산\|롤아웃\]\|Error\|Traceback" $OL/aa_chk2_l40/v_fem.log | cut -c1-3000
rclone copy $OL/aa_chk2_l40 r2:storage/result/anchorflow/lbours/fem_aa_lego/aa_chk2_l40 -q
echo AA1B_DONE $(date +%T)
