#!/bin/bash
# 절 CM97 묶음 wmwhy 띄우기 (잡 133078 안에서 pbs_attach 로): R2 의 wmwhy_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/wmwhy/code/wmwhy_boot.sh | bash
