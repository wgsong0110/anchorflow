#!/bin/bash
# 묶음 wolffields (wolf 칸 모은 영상) 띄우기 (살아 있는 잡 안에서 pbs_attach 로): R2 의 wolffields_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/wolffields/code/wolffields_boot.sh | bash
