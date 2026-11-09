#!/bin/bash
# 절 CM93 묶음 wolfh12 띄우기 (잡 133077 안에서 pbs_attach 로): R2 의 wolfh12_boot.sh 를 받아 상주 시작점으로 돌림
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/wolfh12/code/wolfh12_boot.sh | bash
