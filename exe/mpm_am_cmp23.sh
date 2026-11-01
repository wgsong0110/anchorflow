#!/bin/bash
# 절 AM: 2 차 묶음 (am_nt2) 을 3 차 묶음의 원래 솔버 판 (am_nt3/pg, 같은 GPU0 조건) 과 비교. CPU 만 (torch 는 카드 없이도 됨) -- GPU 는 쓰지 않는다.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/lbours/mpm_speed
until [ -f $B/am_nt3/AM_DONE ]; do sleep 60; done
CUDA_VISIBLE_DEVICES= python -u exe/mpm_fast_cmp.py $B/am_nt2 pg $B/am_nt3/pg > $B/am_nt2/cmp_vs_nt3pg.txt 2>&1
CUDA_VISIBLE_DEVICES= python -u exe/mpm_fast_cmp.py $B/am_nt pg $B/am_nt3/pg > $B/am_nt/cmp_vs_nt3pg.txt 2>&1
rclone copy $B/am_nt2/cmp_vs_nt3pg.txt r2:storage/result/anchorflow/lbours/mpm_speed/am_nt2 -q
rclone copy $B/am_nt/cmp_vs_nt3pg.txt r2:storage/result/anchorflow/lbours/mpm_speed/am_nt -q
touch $B/am_nt2/CMP23_DONE; rclone copy $B/am_nt2/CMP23_DONE r2:storage/result/anchorflow/lbours/mpm_speed/am_nt2 -q
