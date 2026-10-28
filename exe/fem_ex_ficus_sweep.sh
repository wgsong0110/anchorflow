#!/bin/bash
# 절 L ficus 명시적 판 N 스윕 (125 프레임, 단독 GPU0 순차). 물리 조건 = 암시적 실시간 판 rt_tmix_fix 와 같은 선택지
# (--floor none --cfg_bc --region_density --pg_vdamp, node_h 0.056569). 사용 (잡 안에서만): AF_DETACH=1 bash exe/job_run.sh <작업트리>/exe/fem_ex_ficus_sweep.sh <로그>
cd "$(dirname "$0")/.." || exit 1
WAIT_FOR=${WAIT_FOR:-/home/dkta/work/lbours/fem_ex_sweep3.out}
while ! grep -q "_DONE" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
W=/home/dkta/work
export EX_ROOT=fem_ex_ficus
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
R="bash exe/fem_ex_run.sh"
for NN in 256 384 512 768 1024 2048; do $R f_lag_tm_n$NN 0 --mode lag --N $NN --opt --mass tet --no_dump $PHYS; done
for NN in 384 512 1024; do $R f_sub_tm_n$NN 0 --mode sub --N $NN --opt --mass tet --no_dump $PHYS; done
for NN in 1024 4096 16384; do $R f_lag_n$NN 0 --mode lag --N $NN --opt --no_dump $PHYS; done
for NN in 1024 4096; do $R f_sub_n$NN 0 --mode sub --N $NN --opt --no_dump $PHYS; done
echo FICUS_SWEEP_DONE
