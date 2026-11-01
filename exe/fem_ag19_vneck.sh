#!/bin/bash
# 절 AG.19: bread rpic 등가 점성 -- 공통 설정 판 (덤프 + 목 진단) (비측정)
cd "$(dirname "$0")/.." || exit 1
Q="--lim_sr,1e30,--sp_vcap,0.5,--ak_pq,--ak_capmu,--akpl_el,--ak_sfcr,--ah_mcfl,1,--ak_pq_plast,inc,--visc_rpic"
AHAG_ID=vn1 AHAG_G="${AHAG_G:-2}" AHAG_DUMP=1 AHAG_RUNS="bread:160:_uv:$Q" bash exe/fem_ah_ag.sh
for k in $(seq 1 20); do rclone lsf r2:storage/result/anchorflow/lbours/fem_ag_bread/ag_bread_n160_uv_r1/roll_fem.pt -q >/dev/null 2>&1 && break; sleep 60; done
NECK_R=ag_bread_n160_uv_r1 bash exe/fem_ag_neck.sh
rclone lsl r2:storage/result/anchorflow/lbours/fem_ag_bread/ag_bread_n160_uv_r1/roll_fem.pt && rm -f /tmp/dkta/lbours/fem_ag_bread/ag_bread_n160_uv_r1/roll_fem.pt
echo VNECK_DONE
