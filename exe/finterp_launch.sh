#!/bin/bash
# 절 CM92 묶음 finterp 띄우기 (잡 133077 안에서 pbs_attach 로): R2 의 finterp_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/finterp/code/finterp_boot.sh | bash
