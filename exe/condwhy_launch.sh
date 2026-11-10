#!/bin/bash
# 절 CM98 묶음 condwhy (exe/convwhy_launch.sh 복사) 띄우기 (잡 133078 안에서 pbs_attach 로): R2 의 condwhy_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/condwhy/code/condwhy_boot.sh | bash
