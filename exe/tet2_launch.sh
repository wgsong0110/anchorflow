#!/bin/bash
# 절 CM103 묶음 tet2 띄우기 (잡 133078 안에서 pbs_attach 로): R2 의 tet2_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/tet2/code/tet2_boot.sh | bash
