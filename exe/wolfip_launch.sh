#!/bin/bash
# 묶음 wolfip (wolf 증분 포텐셜 항별) 띄우기 (살아 있는 잡 안에서 pbs_attach 로): R2 의 wolfip_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/wolfip/code/wolfip_boot.sh | bash
