#!/bin/bash
# 절 AG.18: 빈 사면체 묶음 bread (geA) 를 덤프와 함께 돌린 뒤 목 자리 진단
cd "$(dirname "$0")/.." || exit 1
AHAG_ID=gn1 AHAG_G="${AHAG_G:-2}" AHAG_DUMP=1 AHAG_RUNS="bread:160:_geAd:--lim_sr,1e30,--sp_vcap,1e9,--ak_pq,--akg_empty,--akg_hg,1,--akg_k,100000,--ak_pq_plast,frame,--tet_strain,10_30_60" bash exe/fem_ah_ag.sh
D=/tmp/dkta/lbours/fem_ag_bread/ag_bread_n160_geAd_r1
for k in 1 2 3 4 5 6 7 8 9 10; do rclone lsf r2:storage/result/anchorflow/lbours/fem_ag_bread/ag_bread_n160_geAd_r1/roll_fem.pt -q >/dev/null 2>&1 && break; sleep 60; done
NECK_R=ag_bread_n160_geAd_r1 bash exe/fem_ag_neck.sh
