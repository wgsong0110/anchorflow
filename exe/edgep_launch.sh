#!/bin/bash
# 절 CM105 묶음 edgep 띄우기 (잡 133078 안에서 pbs_attach 로): R2 의 edgep_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/edgep/code/edgep_boot.sh | bash
